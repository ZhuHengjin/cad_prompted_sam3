"""Compact CAD-dimension-conditioned pose head for SAM3 detection tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .geometry import normalize_intrinsics, rotation_6d_to_matrix
from .types import CADPosePredictions


class _PredictionBranch(nn.Sequential):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))


class SAMV3CADPoseHead(nn.Module):
    """Predict box-relative center residual, log-depth, 6-D rotation, and pose quality."""

    architecture_version = 2
    dimension_log_mean: Tensor
    dimension_log_std: Tensor
    pose_score_temperature: Tensor

    def __init__(
        self,
        token_dim: int = 256,
        hidden_dim: int = 256,
        dimension_log_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dimension_log_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        pose_score_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if any(std <= 0 for std in dimension_log_std):
            raise ValueError("Dimension log standard deviations must be positive")
        if pose_score_temperature <= 0:
            raise ValueError("Pose-score calibration temperature must be positive")
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
        camera_intrinsics_b33: Tensor,
        image_size_wh: tuple[int, int],
        cad_geometry_tokens_bkc: Tensor | None = None,
    ) -> CADPosePredictions:
        """Predict a pose for each detection candidate in a batch.

        Detection tokens and boxes supply candidate-level evidence. Scale-free
        CAD aspect ratios condition the shared center/rotation representation;
        effective metric dimensions and normalized camera intrinsics condition
        depth only.
        """
        del cad_geometry_tokens_bkc  # Reserved by the baseline interface.
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
        if camera_intrinsics_b33.shape != (batch, 3, 3):
            raise ValueError(f"camera_intrinsics_b33 must have shape ({batch}, 3, 3)")
        if not torch.isfinite(camera_intrinsics_b33).all():
            raise ValueError("Camera intrinsics must be finite")
        boxes_xy1xy2_bn22 = boxes_xy1xy2_bn22.to(detection_tokens_bnc)
        cad_dimensions_m_b3 = cad_dimensions_m_b3.to(detection_tokens_bnc)
        camera_intrinsics_b33 = camera_intrinsics_b33.to(detection_tokens_bnc)

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
        shared = self.shared(torch.cat((detection_tokens_bnc, box_features, shape_features), dim=-1))

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
        )
