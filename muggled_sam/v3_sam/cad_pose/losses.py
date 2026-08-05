"""Pose targets and the first-milestone CAD pose objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .geometry import rotate_points
from .symmetry import symmetry_aware_rotation_error
from .types import CADPosePredictions, CADPoseTarget


@dataclass(frozen=True)
class CADPoseLossConfig:
    center_weight: float = 1.0
    depth_weight: float = 1.0
    rotation_weight: float = 1.0
    full_pose_weight: float = 0.0
    quality_weight: float = 1.0
    pose_weight: float = 1.0
    log_depth_mean: float = 0.0
    log_depth_std: float = 1.0
    centroid_tolerance: float = 0.1
    point_set_tolerance: float = 0.1
    centroid_soft_width: float = 0.02
    point_set_soft_width: float = 0.02
    point_loss_beta: float = 0.01
    point_distance_chunk_size: int = 512
    # Legacy v1 quality configuration. These fields remain loadable so old
    # checkpoints and symmetry-catalog datasets fail neither parsing nor resume.
    rotation_tolerance_rad: float = 0.08726646259971647
    translation_tolerance: float = 0.1
    rotation_soft_width_rad: float = 0.017453292519943295
    translation_soft_width: float = 0.02
    normalize_translation_error: bool = True

    def __post_init__(self) -> None:
        if self.log_depth_std <= 0:
            raise ValueError("log_depth_std must be positive")
        if self.centroid_soft_width <= 0 or self.point_set_soft_width <= 0:
            raise ValueError("Point-set pose-quality soft-boundary widths must be positive")
        if self.point_loss_beta < 0:
            raise ValueError("point_loss_beta must be nonnegative")
        if self.point_distance_chunk_size <= 0:
            raise ValueError("point_distance_chunk_size must be positive")
        if self.rotation_soft_width_rad <= 0 or self.translation_soft_width <= 0:
            raise ValueError("Pose-quality soft-boundary widths must be positive")


@dataclass(frozen=True)
class CADPoseLosses:
    total: Tensor
    center: Tensor
    depth: Tensor
    rotation: Tensor
    full_pose: Tensor
    quality: Tensor
    mean_rotation_error_rad: Tensor
    mean_translation_error_m: Tensor
    mean_point_set_error_norm: Tensor
    mean_centroid_error_m: Tensor


def nearest_neighbor_distances(query_points: Tensor, target_points: Tensor, *, chunk_size: int = 512) -> Tensor:
    """Return one-sided nearest-neighbor distances without a giant pair tensor."""

    if query_points.ndim != 2 or query_points.shape[-1] != 3 or len(query_points) == 0:
        raise ValueError(f"query_points must have nonempty shape Nx3, got {tuple(query_points.shape)}")
    if target_points.ndim != 2 or target_points.shape[-1] != 3 or len(target_points) == 0:
        raise ValueError(f"target_points must have nonempty shape Mx3, got {tuple(target_points.shape)}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    target_points = target_points.to(query_points)
    calculation_dtype = (
        torch.float32
        if query_points.dtype in (torch.float16, torch.bfloat16)
        else query_points.dtype
    )
    query_points = query_points.to(calculation_dtype)
    target_points = target_points.to(calculation_dtype)
    chunks = []
    for start in range(0, len(query_points), chunk_size):
        pairwise = torch.cdist(query_points[start : start + chunk_size], target_points)
        chunks.append(pairwise.amin(dim=-1))
    return torch.cat(chunks, dim=0)


def point_set_pose_errors(
    rotation_pred: Tensor,
    centroid_pred: Tensor,
    target: CADPoseTarget,
    *,
    chunk_size: int = 512,
    full_pose_grad: bool = False,
    compute_full_pose: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return normalized rotation/full-set errors and centroid error.

    The point arrays are already uniformly scaled and centered at the canonical
    surface centroid. The final two values are the per-query normalized
    rotation and full-placement distances.
    """

    if (
        not target.point_set_eligible
        or target.point_query_m is None
        or target.point_target_m is None
        or target.centroid_m is None
    ):
        raise ValueError("Point-set pose errors require an eligible point-set target")
    calculation_dtype = (
        torch.float32
        if rotation_pred.dtype in (torch.float16, torch.bfloat16)
        else rotation_pred.dtype
    )
    rotation_pred = rotation_pred.to(calculation_dtype)
    centroid_pred = centroid_pred.to(calculation_dtype)
    query = target.point_query_m.to(device=rotation_pred.device, dtype=calculation_dtype)
    dense = target.point_target_m.to(device=rotation_pred.device, dtype=calculation_dtype)
    rotation_target = target.rotation_matrix.to(device=rotation_pred.device, dtype=calculation_dtype)
    centroid_target = target.centroid_m.to(device=centroid_pred.device, dtype=calculation_dtype)
    diagonal = torch.linalg.vector_norm(
        target.dimensions_m.to(device=rotation_pred.device, dtype=calculation_dtype)
    ).clamp_min(1e-8)

    rotated_query = rotate_points(query, rotation_pred)
    rotated_dense = rotate_points(dense, rotation_target)
    rotation_distances = nearest_neighbor_distances(rotated_query, rotated_dense, chunk_size=chunk_size) / diagonal
    if compute_full_pose:
        full_query = rotated_query + centroid_pred
        full_target = rotated_dense + centroid_target
        if not full_pose_grad:
            full_query = full_query.detach()
            full_target = full_target.detach()
        full_distances = (
            nearest_neighbor_distances(
                full_query,
                full_target,
                chunk_size=chunk_size,
            )
            / diagonal
        )
        full_set_error = full_distances.mean()
    else:
        full_distances = rotation_distances.new_empty((0,))
        full_set_error = rotation_distances.new_tensor(float("nan"))
    centroid_error_m = torch.linalg.vector_norm(centroid_pred - centroid_target)
    centroid_error_norm = centroid_error_m / diagonal
    return (
        rotation_distances.mean(),
        full_set_error,
        centroid_error_norm,
        centroid_error_m,
        rotation_distances,
        full_distances,
    )


def compute_cad_pose_losses(
    predictions: CADPosePredictions,
    targets: Sequence[CADPoseTarget],
    matches: Sequence[tuple[int, int]],
    config: CADPoseLossConfig,
    *,
    batch_index: int = 0,
    compute_expensive_metrics: bool = True,
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
    full_pose_losses: list[Tensor] = []
    quality_losses: list[Tensor] = []
    rotation_errors: list[Tensor] = []
    translation_errors: list[Tensor] = []
    point_set_errors: list[Tensor] = []
    centroid_errors: list[Tensor] = []
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

        if target.point_set_eligible:
            if predictions.centroid_m_bn3 is None:
                raise ValueError("Point-set losses require back-projected surface centroids")
            centroid_pred = predictions.centroid_m_bn3[batch_index, prediction_index]
            compute_full_pose = (
                compute_expensive_metrics
                or config.full_pose_weight != 0
                or config.quality_weight != 0
            )
            (
                rotation_set_error,
                full_set_error,
                centroid_error_norm,
                centroid_error_m,
                rotation_distances,
                full_distances,
            ) = point_set_pose_errors(
                rotation_pred,
                centroid_pred,
                target,
                chunk_size=config.point_distance_chunk_size,
                full_pose_grad=config.full_pose_weight != 0,
                compute_full_pose=compute_full_pose,
            )
            # Smooth-L1 retains a stable gradient near exact matches while the
            # raw normalized distances remain available for quality/metrics.
            rotation_losses.append(
                F.smooth_l1_loss(
                    rotation_distances,
                    torch.zeros_like(rotation_distances),
                    beta=config.point_loss_beta,
                )
            )
            if compute_full_pose:
                full_pose_losses.append(
                    F.smooth_l1_loss(
                        full_distances,
                        torch.zeros_like(full_distances),
                        beta=config.point_loss_beta,
                    )
                )
                point_set_errors.append(full_set_error)
            centroid_errors.append(centroid_error_m)
            if compute_expensive_metrics or config.quality_weight != 0:
                quality_target = torch.sigmoid(
                    (config.centroid_tolerance - centroid_error_norm) / config.centroid_soft_width
                ) * torch.sigmoid(
                    (config.point_set_tolerance - full_set_error) / config.point_set_soft_width
                )
                quality_losses.append(
                    F.binary_cross_entropy_with_logits(
                        predictions.pose_score_logits_bn[batch_index, prediction_index],
                        quality_target.detach(),
                    )
                )
            rotation_error = rotation_pred.new_tensor(float("nan"))
        elif target.rotation_eligible:
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
        if (
            target.rotation_eligible
            and not target.point_set_eligible
            and (compute_expensive_metrics or config.quality_weight != 0)
        ):
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
    full_pose = torch.stack(full_pose_losses).mean() if full_pose_losses else zero
    quality = torch.stack(quality_losses).mean() if quality_losses else zero
    total = config.pose_weight * (
        config.center_weight * center
        + config.depth_weight * depth
        + config.rotation_weight * rotation
        + config.full_pose_weight * full_pose
        + config.quality_weight * quality
    )
    mean_rotation = (
        torch.stack(rotation_errors).mean()
        if rotation_errors
        else predictions.log_depth_bn.new_tensor(float("nan"))
    )
    mean_translation = torch.stack(translation_errors).mean() if translation_errors else zero
    mean_point_set = (
        torch.stack(point_set_errors).mean()
        if point_set_errors
        else predictions.log_depth_bn.new_tensor(float("nan"))
    )
    mean_centroid = (
        torch.stack(centroid_errors).mean()
        if centroid_errors
        else predictions.log_depth_bn.new_tensor(float("nan"))
    )
    return CADPoseLosses(
        total=total,
        center=center,
        depth=depth,
        rotation=rotation,
        full_pose=full_pose,
        quality=quality,
        mean_rotation_error_rad=mean_rotation,
        mean_translation_error_m=mean_translation,
        mean_point_set_error_norm=mean_point_set,
        mean_centroid_error_m=mean_centroid,
    )
