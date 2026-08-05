"""Compact CAD-dimension-conditioned pose head for SAM3 detection tokens."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from torch import Tensor

from .geometry import normalize_intrinsics, rotation_6d_to_matrix
from .types import CADPosePredictions


class _PredictionBranch(nn.Sequential):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))


class CADPromptCrossAttention(nn.Module):
    """Pose-only residual attention from detection queries to CAD prompt tokens."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int = 8,
        initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if initial_gate < 0:
            raise ValueError("initial_gate must be nonnegative")
        self.query_norm = nn.LayerNorm(token_dim)
        self.prompt_norm = nn.LayerNorm(token_dim)
        self.cross_attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(
        self,
        detection_tokens_bnc: Tensor,
        cad_prompt_tokens_bkc: Tensor | None,
        cad_prompt_padding_mask_bk: Tensor | None = None,
    ) -> Tensor:
        """Return pose-adapted queries without mutating the detector tokens."""

        if cad_prompt_tokens_bkc is None or cad_prompt_tokens_bkc.shape[1] == 0:
            return detection_tokens_bnc
        if cad_prompt_tokens_bkc.ndim != 3:
            raise ValueError("cad_prompt_tokens_bkc must have shape BxKxC")
        batch, _, token_dim = detection_tokens_bnc.shape
        if cad_prompt_tokens_bkc.shape[0] != batch:
            raise ValueError("CAD prompt and detection tokens must share the batch dimension")
        if cad_prompt_tokens_bkc.shape[2] != token_dim:
            raise ValueError("CAD prompt and detection tokens must share the token dimension")
        if cad_prompt_padding_mask_bk is not None:
            expected_mask_shape = cad_prompt_tokens_bkc.shape[:2]
            if cad_prompt_padding_mask_bk.shape != expected_mask_shape:
                raise ValueError(
                    "cad_prompt_padding_mask_bk must match the CAD prompt token dimensions"
                )
            cad_prompt_padding_mask_bk = cad_prompt_padding_mask_bk.to(
                device=detection_tokens_bnc.device,
                dtype=torch.bool,
            )
            if torch.any(cad_prompt_padding_mask_bk.all(dim=1)):
                raise ValueError("Every batch item must contain at least one unpadded CAD prompt token")

        cad_prompt_tokens_bkc = cad_prompt_tokens_bkc.to(detection_tokens_bnc)
        prompt = self.prompt_norm(cad_prompt_tokens_bkc)
        residual, _ = self.cross_attention(
            self.query_norm(detection_tokens_bnc),
            prompt,
            prompt,
            key_padding_mask=cad_prompt_padding_mask_bk,
            need_weights=False,
        )
        return detection_tokens_bnc + self.gate.to(residual) * residual


class SAMV3CADPoseHead(nn.Module):
    """Predict surface-centroid projection/depth, 6-D rotation, and pose quality."""

    architecture_version = 4
    legacy_promptless_architecture_version = 3
    dimension_log_mean: Tensor
    dimension_log_std: Tensor
    pose_score_temperature: Tensor
    cad_prompt_enabled: Tensor

    def __init__(
        self,
        token_dim: int = 256,
        hidden_dim: int = 256,
        dimension_log_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dimension_log_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        pose_score_temperature: float = 1.0,
        cad_prompt_num_heads: int = 8,
        cad_prompt_initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if any(std <= 0 for std in dimension_log_std):
            raise ValueError("Dimension log standard deviations must be positive")
        if pose_score_temperature <= 0:
            raise ValueError("Pose-score calibration temperature must be positive")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.cad_prompt_num_heads = int(cad_prompt_num_heads)
        self.cad_prompt_initial_gate = float(cad_prompt_initial_gate)
        self.cad_prompt_adapter = CADPromptCrossAttention(
            token_dim,
            num_heads=cad_prompt_num_heads,
            initial_gate=cad_prompt_initial_gate,
        )
        # Prompting is explicit so loading an older pose checkpoint cannot
        # silently activate a freshly initialized adapter at inference.
        self.register_buffer("cad_prompt_enabled", torch.tensor(False, dtype=torch.bool))
        # The shared pose representation is scale invariant: it sees the
        # detection token, normalized box center/extent, and CAD aspect ratios.
        # Absolute metric size and camera calibration are reserved for depth.
        input_dim = token_dim + 4 + 3
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Depth additionally sees normalized log dimensions and camera features:
        # log focal lengths, principal point, center ray, and angular box extent.
        depth_input_dim = hidden_dim + 3 + 8
        self.depth_fusion = nn.Sequential(
            nn.Linear(depth_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Keep task-specific output layers separate after feature fusion.
        self.center_branch = _PredictionBranch(hidden_dim, 2)
        self.depth_branch = _PredictionBranch(hidden_dim, 1)
        self.rotation_branch = _PredictionBranch(hidden_dim, 6)
        self.quality_branch = _PredictionBranch(hidden_dim, 1)
        # Persist normalization and calibration values with model checkpoints.
        self.register_buffer("dimension_log_mean", torch.tensor(dimension_log_mean, dtype=torch.float32))
        self.register_buffer("dimension_log_std", torch.tensor(dimension_log_std, dtype=torch.float32))
        self.register_buffer("pose_score_temperature", torch.tensor(float(pose_score_temperature), dtype=torch.float32))

    def architecture_config(self) -> dict[str, int | float]:
        return {
            "token_dim": self.token_dim,
            "hidden_dim": self.hidden_dim,
            "cad_prompt_num_heads": self.cad_prompt_num_heads,
            "cad_prompt_initial_gate": self.cad_prompt_initial_gate,
        }

    def set_cad_prompt_enabled(self, enabled: bool) -> None:
        self.cad_prompt_enabled.fill_(bool(enabled))

    def load_checkpoint_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        architecture_version: int,
    ) -> bool:
        """Load a native checkpoint or migrate the promptless v3 pose head.

        Returns ``True`` when a v3 checkpoint was migrated. Only the newly
        introduced CAD-prompt state may be absent during that migration.
        """

        if architecture_version == self.architecture_version:
            self.load_state_dict(state_dict)
            return False
        if architecture_version != self.legacy_promptless_architecture_version:
            raise ValueError(
                "CAD pose-head checkpoint architecture version "
                f"{architecture_version} is incompatible with model version "
                f"{self.architecture_version}"
            )

        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing = {"cad_prompt_enabled"}
        allowed_missing.update(
            f"cad_prompt_adapter.{key}"
            for key in self.cad_prompt_adapter.state_dict()
        )
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise ValueError(
                "Pose-head v3 migration found incompatible state keys: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        self.set_cad_prompt_enabled(False)
        return True

    def set_dimension_statistics(self, mean: Tensor, std: Tensor) -> None:
        if mean.numel() != 3 or std.numel() != 3 or torch.any(std <= 0):
            raise ValueError("Dimension log statistics must contain three values with positive std")
        self.dimension_log_mean.copy_(mean.reshape(3).to(self.dimension_log_mean))
        self.dimension_log_std.copy_(std.reshape(3).to(self.dimension_log_std))

    def set_pose_score_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("Pose-score calibration temperature must be positive")
        self.pose_score_temperature.fill_(temperature)

    def forward(
        self,
        detection_tokens_bnc: Tensor,
        boxes_xy1xy2_bn22: Tensor,
        cad_dimensions_m_b3: Tensor,
        cad_effective_surface_centroid_m_b3: Tensor,
        camera_intrinsics_b33: Tensor,
        image_size_wh: tuple[int, int],
        cad_geometry_tokens_bkc: Tensor | None = None,
        cad_geometry_padding_mask_bk: Tensor | None = None,
    ) -> CADPosePredictions:
        """Predict a pose for each detection candidate in a batch.

        Detection tokens and boxes supply candidate-level evidence. Scale-free
        CAD aspect ratios condition the shared center/rotation representation;
        effective metric dimensions and normalized camera intrinsics condition
        depth only.
        """
        # Validate the leading batch/candidate dimensions before combining inputs.
        if detection_tokens_bnc.ndim != 3:
            raise ValueError("detection_tokens_bnc must have shape BxNxC")
        batch, candidates, _ = detection_tokens_bnc.shape
        if boxes_xy1xy2_bn22.shape != (batch, candidates, 2, 2):
            raise ValueError("boxes_xy1xy2_bn22 must match detection tokens and end in 2x2")
        if cad_dimensions_m_b3.shape != (batch, 3):
            raise ValueError(f"cad_dimensions_m_b3 must have shape ({batch}, 3)")
        if not torch.isfinite(cad_dimensions_m_b3).all() or torch.any(cad_dimensions_m_b3 <= 0):
            raise ValueError("Effective CAD dimensions must be finite and strictly positive")
        if cad_effective_surface_centroid_m_b3.shape != (batch, 3):
            raise ValueError(
                f"cad_effective_surface_centroid_m_b3 must have shape ({batch}, 3)"
            )
        if not torch.isfinite(cad_effective_surface_centroid_m_b3).all():
            raise ValueError("Effective CAD surface centroids must be finite")
        if camera_intrinsics_b33.shape != (batch, 3, 3):
            raise ValueError(f"camera_intrinsics_b33 must have shape ({batch}, 3, 3)")
        if not torch.isfinite(camera_intrinsics_b33).all():
            raise ValueError("Camera intrinsics must be finite")
        boxes_xy1xy2_bn22 = boxes_xy1xy2_bn22.to(detection_tokens_bnc)
        cad_dimensions_m_b3 = cad_dimensions_m_b3.to(detection_tokens_bnc)
        cad_effective_surface_centroid_m_b3 = cad_effective_surface_centroid_m_b3.to(
            detection_tokens_bnc
        )
        camera_intrinsics_b33 = camera_intrinsics_b33.to(detection_tokens_bnc)

        pose_tokens_bnc = detection_tokens_bnc
        if bool(self.cad_prompt_enabled):
            pose_tokens_bnc = self.cad_prompt_adapter(
                detection_tokens_bnc,
                cad_geometry_tokens_bkc,
                cad_geometry_padding_mask_bk,
            )

        # Flatten the two box corners, then express the box as center and extent.
        boxes = boxes_xy1xy2_bn22.reshape(batch, candidates, 4)
        xy1, xy2 = boxes[..., :2], boxes[..., 2:]
        box_extent = xy2 - xy1
        box_features = torch.cat(((xy1 + xy2) * 0.5, box_extent), dim=-1)
        box_center = box_features[..., :2]
        # Separate scale-free CAD shape from absolute metric dimensions. Uniform
        # changes in physical size therefore cannot move the predicted 2-D center
        # or rotation, but can change metric depth.
        log_dimensions = cad_dimensions_m_b3.log()
        shape_features = log_dimensions - log_dimensions.mean(dim=-1, keepdim=True)
        dimension_features = (
            log_dimensions - self.dimension_log_mean.to(cad_dimensions_m_b3)
        ) / self.dimension_log_std.to(cad_dimensions_m_b3).clamp_min(1e-6)
        shape_features = shape_features.unsqueeze(1).expand(-1, candidates, -1)
        dimension_features = dimension_features.unsqueeze(1).expand(-1, candidates, -1)
        shared = self.shared(torch.cat((pose_tokens_bnc, box_features, shape_features), dim=-1))

        # Predict center correction in box-relative units. This keeps center
        # refinement equivariant to apparent object size and image resolution.
        center_residual = self.center_branch(shared)
        center_uv = box_center + center_residual * box_extent

        # Normalize the already resize-adjusted camera matrix, then describe each
        # candidate by its center ray and angular box extent. Matrix inversion
        # runs in float32 for half-precision training compatibility.
        normalized_intrinsics = normalize_intrinsics(camera_intrinsics_b33, image_size_wh)
        focal_xy = torch.diagonal(normalized_intrinsics, dim1=-2, dim2=-1)[..., :2]
        if torch.any(focal_xy <= 0):
            raise ValueError("Camera focal lengths must be positive")
        principal_xy = normalized_intrinsics[..., :2, 2]
        solve_dtype = (
            torch.float32
            if normalized_intrinsics.dtype in (torch.float16, torch.bfloat16)
            else normalized_intrinsics.dtype
        )
        if torch.any(torch.linalg.det(normalized_intrinsics.to(solve_dtype)).abs() <= 1e-8):
            raise ValueError("Camera intrinsics must be nonsingular")
        image_points = torch.stack((box_center, xy1, xy2), dim=2)
        homogeneous = torch.cat((image_points, torch.ones_like(image_points[..., :1])), dim=-1)
        inverse_intrinsics = torch.linalg.inv(normalized_intrinsics.to(solve_dtype))
        rays = torch.matmul(
            homogeneous.to(solve_dtype),
            inverse_intrinsics.transpose(-1, -2).unsqueeze(1),
        )
        rays = (rays[..., :2] / rays[..., 2:].clamp_min(1e-6)).to(boxes)
        center_rays, corner1_rays, corner2_rays = rays.unbind(dim=2)
        angular_extent = (corner2_rays - corner1_rays).abs()
        camera_global = torch.cat((focal_xy.log(), principal_xy), dim=-1)
        camera_global = camera_global.unsqueeze(1).expand(-1, candidates, -1)
        camera_features = torch.cat((camera_global, center_rays, angular_extent), dim=-1)
        depth_features = self.depth_fusion(
            torch.cat((shared, dimension_features, camera_features), dim=-1)
        )
        log_depth = self.depth_branch(depth_features).squeeze(-1)
        rotation_6d = self.rotation_branch(shared)
        # Convert the continuous 6-D representation into an orthonormal matrix.
        rotation_matrix = rotation_6d_to_matrix(rotation_6d)
        pose_score_logits = self.quality_branch(shared).squeeze(-1)
        # Temperature calibration adjusts confidence sharpness without changing
        # candidate ordering, then sigmoid maps logits into the [0, 1] range.
        temperature = self.pose_score_temperature.to(pose_score_logits).clamp_min(1e-6)
        pose_score = torch.sigmoid(pose_score_logits / temperature)
        return CADPosePredictions(
            center_residual,
            center_uv,
            log_depth,
            rotation_6d,
            rotation_matrix,
            pose_score_logits,
            pose_score,
            cad_dimensions_m_b3=cad_dimensions_m_b3,
            cad_effective_surface_centroid_m_b3=cad_effective_surface_centroid_m_b3,
        )
