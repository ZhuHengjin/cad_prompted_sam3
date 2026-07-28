"""Compact CAD-dimension-conditioned pose head for SAM3 detection tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .geometry import rotation_6d_to_matrix
from .types import CADPosePredictions


class _PredictionBranch(nn.Sequential):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))


class SAMV3CADPoseHead(nn.Module):
    """Predict center residual, log-depth, 6-D rotation, and pose quality."""

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
        input_dim = token_dim + 4 + 3
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.center_branch = _PredictionBranch(hidden_dim, 2)
        self.depth_branch = _PredictionBranch(hidden_dim, 1)
        self.rotation_branch = _PredictionBranch(hidden_dim, 6)
        self.quality_branch = _PredictionBranch(hidden_dim, 1)
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
        cad_geometry_tokens_bkc: Tensor | None = None,
    ) -> CADPosePredictions:
        del cad_geometry_tokens_bkc  # Reserved by the baseline interface.
        if detection_tokens_bnc.ndim != 3:
            raise ValueError("detection_tokens_bnc must have shape BxNxC")
        batch, candidates, _ = detection_tokens_bnc.shape
        if boxes_xy1xy2_bn22.shape != (batch, candidates, 2, 2):
            raise ValueError("boxes_xy1xy2_bn22 must match detection tokens and end in 2x2")
        if cad_dimensions_m_b3.shape != (batch, 3):
            raise ValueError(f"cad_dimensions_m_b3 must have shape ({batch}, 3)")
        if not torch.isfinite(cad_dimensions_m_b3).all() or torch.any(cad_dimensions_m_b3 <= 0):
            raise ValueError("Effective CAD dimensions must be finite and strictly positive")

        boxes = boxes_xy1xy2_bn22.reshape(batch, candidates, 4)
        xy1, xy2 = boxes[..., :2], boxes[..., 2:]
        box_features = torch.cat(((xy1 + xy2) * 0.5, xy2 - xy1), dim=-1)
        box_center = box_features[..., :2]
        dimension_features = (
            cad_dimensions_m_b3.log() - self.dimension_log_mean.to(cad_dimensions_m_b3)
        ) / self.dimension_log_std.to(cad_dimensions_m_b3).clamp_min(1e-6)
        dimension_features = dimension_features.unsqueeze(1).expand(-1, candidates, -1)
        shared = self.shared(torch.cat((detection_tokens_bnc, box_features, dimension_features), dim=-1))

        center_residual = self.center_branch(shared)
        center_uv = box_center + center_residual
        log_depth = self.depth_branch(shared).squeeze(-1)
        rotation_6d = self.rotation_branch(shared)
        rotation_matrix = rotation_6d_to_matrix(rotation_6d)
        pose_score_logits = self.quality_branch(shared).squeeze(-1)
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
