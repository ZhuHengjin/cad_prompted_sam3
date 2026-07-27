"""Pose targets and the first-milestone CAD pose objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .symmetry import symmetry_aware_rotation_error
from .types import CADPosePredictions, CADPoseTarget


@dataclass(frozen=True)
class CADPoseLossConfig:
    center_weight: float = 1.0
    depth_weight: float = 1.0
    rotation_weight: float = 1.0
    quality_weight: float = 1.0
    pose_weight: float = 1.0
    log_depth_mean: float = 0.0
    log_depth_std: float = 1.0
    rotation_tolerance_rad: float = 0.08726646259971647
    translation_tolerance: float = 0.1
    rotation_soft_width_rad: float = 0.017453292519943295
    translation_soft_width: float = 0.02
    normalize_translation_error: bool = True

    def __post_init__(self) -> None:
        if self.log_depth_std <= 0:
            raise ValueError("log_depth_std must be positive")
        if self.rotation_soft_width_rad <= 0 or self.translation_soft_width <= 0:
            raise ValueError("Pose-quality soft-boundary widths must be positive")


@dataclass(frozen=True)
class CADPoseLosses:
    total: Tensor
    center: Tensor
    depth: Tensor
    rotation: Tensor
    quality: Tensor
    mean_rotation_error_rad: Tensor
    mean_translation_error_m: Tensor


def compute_cad_pose_losses(
    predictions: CADPosePredictions,
    targets: Sequence[CADPoseTarget],
    matches: Sequence[tuple[int, int]],
    config: CADPoseLossConfig,
    *,
    batch_index: int = 0,
) -> CADPoseLosses | None:
    """Compute losses for one image's one-to-one pose matches."""

    if not matches:
        return None
    if predictions.translation_m_bn3 is None:
        raise ValueError("Pose losses require translation reconstructed with adjusted intrinsics")
    if batch_index < 0 or batch_index >= predictions.center_uv_norm_bn2.shape[0]:
        raise IndexError(f"batch_index {batch_index} is outside the pose prediction batch")

    center_losses: list[Tensor] = []
    depth_losses: list[Tensor] = []
    rotation_losses: list[Tensor] = []
    quality_losses: list[Tensor] = []
    rotation_errors: list[Tensor] = []
    translation_errors: list[Tensor] = []
    for gt_index, prediction_index in matches:
        target = targets[gt_index]
        center_pred = predictions.center_uv_norm_bn2[batch_index, prediction_index]
        depth_pred = predictions.log_depth_bn[batch_index, prediction_index]
        rotation_pred = predictions.rotation_matrix_bn33[batch_index, prediction_index]
        translation_pred = predictions.translation_m_bn3[batch_index, prediction_index]

        center_target = target.center_uv_norm.to(center_pred)
        depth_target = target.log_depth.to(depth_pred)
        center_losses.append(F.smooth_l1_loss(center_pred, center_target))
        depth_losses.append(
            F.smooth_l1_loss(
                (depth_pred - config.log_depth_mean) / config.log_depth_std,
                (depth_target - config.log_depth_mean) / config.log_depth_std,
            )
        )

        if target.rotation_eligible:
            rotation_error = symmetry_aware_rotation_error(
                rotation_pred,
                target.rotation_matrix.to(rotation_pred),
                symmetry_type=target.symmetry_type,
                symmetry_transforms=(
                    target.symmetry_transforms.to(rotation_pred) if target.symmetry_transforms is not None else None
                ),
                axis_cad=target.axis_cad.to(rotation_pred) if target.axis_cad is not None else None,
            )
            rotation_losses.append(rotation_error)
        else:
            rotation_error = rotation_pred.new_tensor(float("nan"))

        translation_error = torch.linalg.vector_norm(translation_pred - target.translation_m.to(translation_pred))
        translation_errors.append(translation_error)
        if target.rotation_eligible:
            rotation_errors.append(rotation_error)
            if config.normalize_translation_error:
                diagonal = torch.linalg.vector_norm(target.dimensions_m.to(translation_error)).clamp_min(1e-8)
                quality_translation_error = translation_error / diagonal
            else:
                quality_translation_error = translation_error
            quality_target = torch.sigmoid(
                (config.rotation_tolerance_rad - rotation_error) / config.rotation_soft_width_rad
            ) * torch.sigmoid(
                (config.translation_tolerance - quality_translation_error) / config.translation_soft_width
            )
            quality_losses.append(
                F.binary_cross_entropy_with_logits(
                    predictions.pose_score_logits_bn[batch_index, prediction_index], quality_target.detach()
                )
            )

    zero = predictions.log_depth_bn.sum() * 0.0
    center = torch.stack(center_losses).mean() if center_losses else zero
    depth = torch.stack(depth_losses).mean() if depth_losses else zero
    rotation = torch.stack(rotation_losses).mean() if rotation_losses else zero
    quality = torch.stack(quality_losses).mean() if quality_losses else zero
    total = config.pose_weight * (
        config.center_weight * center
        + config.depth_weight * depth
        + config.rotation_weight * rotation
        + config.quality_weight * quality
    )
    mean_rotation = torch.stack(rotation_errors).mean() if rotation_errors else zero
    mean_translation = torch.stack(translation_errors).mean() if translation_errors else zero
    return CADPoseLosses(total, center, depth, rotation, quality, mean_rotation, mean_translation)
