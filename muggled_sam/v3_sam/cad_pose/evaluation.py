"""Label-free point-set pose metrics and held-out score calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .losses import point_set_pose_errors
from .symmetry import symmetry_aware_rotation_error
from .types import CADPosePredictions, CADPoseTarget


@dataclass(frozen=True)
class PoseEvaluation:
    count: int
    mean_surface_distance_norm: float
    p95_surface_distance_norm: float
    centroid_error_cm: float
    translation_error_cm: float
    center_error_norm: float
    depth_error_m: float
    pose_success_rate: float
    brier_score: float
    expected_calibration_error: float
    # Legacy v1 diagnostic metrics. Point-set datasets report NaN here.
    rotation_error_deg: float
    accuracy_5deg_5cm: float
    accuracy_10deg_10cm: float
    # Retained only so callers can compute a true dataset-level percentile.
    # It is excluded from repr/comparison because it is an aggregation payload,
    # not another reported scalar metric.
    surface_distances_norm: Tensor | None = field(default=None, repr=False, compare=False)


def evaluate_pose_matches(
    predictions: CADPosePredictions,
    targets: Sequence[CADPoseTarget],
    matches: Sequence[tuple[int, int]],
    *,
    batch_index: int = 0,
    calibration_bins: int = 10,
    centroid_tolerance: float = 0.1,
    point_set_tolerance: float = 0.1,
    point_distance_chunk_size: int = 512,
    rotation_tolerance_rad: float = 0.08726646259971647,
    translation_tolerance: float = 0.05,
    normalize_translation_error: bool = False,
) -> PoseEvaluation:
    if predictions.translation_m_bn3 is None:
        raise ValueError("Evaluation requires reconstructed metric translations")
    rotation_errors, translation_errors, legacy_translation_errors = [], [], []
    center_errors, depth_errors, scores = [], [], []
    surface_errors, surface_distance_values, centroid_errors, success_targets = [], [], [], []
    for gt_index, prediction_index in matches:
        target = targets[gt_index]
        if not target.rotation_eligible and not target.point_set_eligible:
            continue
        predicted_rotation = predictions.rotation_matrix_bn33[batch_index, prediction_index]
        if target.point_set_eligible:
            if predictions.centroid_m_bn3 is None:
                raise ValueError("Point-set evaluation requires back-projected centroids")
            centroid_pred = predictions.centroid_m_bn3[batch_index, prediction_index]
            _, full_error, centroid_error_norm, centroid_error_m, _, full_distances = point_set_pose_errors(
                predicted_rotation,
                centroid_pred,
                target,
                chunk_size=point_distance_chunk_size,
            )
            surface_errors.append(full_error)
            surface_distance_values.append(full_distances)
            centroid_errors.append(centroid_error_m)
            success_targets.append(
                ((centroid_error_norm <= centroid_tolerance) & (full_error <= point_set_tolerance)).float()
            )
        else:
            rotation_error = symmetry_aware_rotation_error(
                predicted_rotation,
                target.rotation_matrix.to(predicted_rotation),
                symmetry_type=target.symmetry_type,
                symmetry_transforms=(
                    target.symmetry_transforms.to(predicted_rotation)
                    if target.symmetry_transforms is not None
                    else None
                ),
                axis_cad=target.axis_cad.to(predicted_rotation) if target.axis_cad is not None else None,
            )
            rotation_errors.append(rotation_error)
        translation = predictions.translation_m_bn3[batch_index, prediction_index]
        translation_error = torch.linalg.vector_norm(translation - target.translation_m.to(translation))
        translation_errors.append(translation_error)
        if not target.point_set_eligible:
            legacy_translation_errors.append(translation_error)
            success_translation_error = translation_error
            if normalize_translation_error:
                success_translation_error = translation_error / torch.linalg.vector_norm(
                    target.dimensions_m.to(translation_error)
                ).clamp_min(1e-8)
            success_targets.append(
                (
                    (rotation_error <= rotation_tolerance_rad)
                    & (success_translation_error <= translation_tolerance)
                ).float()
            )
        center_errors.append(
            torch.linalg.vector_norm(
                predictions.center_uv_norm_bn2[batch_index, prediction_index]
                - target.center_uv_norm.to(predictions.center_uv_norm_bn2)
            )
        )
        predicted_depth = predictions.log_depth_bn[batch_index, prediction_index].exp()
        depth_errors.append((predicted_depth - target.log_depth.to(predicted_depth).exp()).abs())
        scores.append(predictions.pose_score_bn[batch_index, prediction_index])
    if not scores:
        nan = float("nan")
        return PoseEvaluation(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    translation = torch.stack(translation_errors)
    score = torch.stack(scores)
    success = torch.stack(success_targets).to(score)
    if rotation_errors:
        rotation = torch.stack(rotation_errors)
        legacy_translation = torch.stack(legacy_translation_errors)
        success_5 = (
            (rotation <= torch.deg2rad(rotation.new_tensor(5.0))) & (legacy_translation <= 0.05)
        ).float()
        success_10 = (
            (rotation <= torch.deg2rad(rotation.new_tensor(10.0))) & (legacy_translation <= 0.10)
        ).float()
        rotation_error_deg = float(torch.rad2deg(rotation).mean())
        accuracy_5 = float(success_5.mean())
        accuracy_10 = float(success_10.mean())
    else:
        rotation_error_deg = accuracy_5 = accuracy_10 = float("nan")
    if surface_errors:
        surface_mean = float(torch.stack(surface_errors).mean())
        surface_p95 = float(torch.quantile(torch.cat(surface_distance_values), 0.95))
        centroid_cm = float(torch.stack(centroid_errors).mean() * 100.0)
    else:
        surface_mean = surface_p95 = centroid_cm = float("nan")
    return PoseEvaluation(
        count=len(scores),
        mean_surface_distance_norm=surface_mean,
        p95_surface_distance_norm=surface_p95,
        centroid_error_cm=centroid_cm,
        translation_error_cm=float(translation.mean() * 100.0),
        center_error_norm=float(torch.stack(center_errors).mean()),
        depth_error_m=float(torch.stack(depth_errors).mean()),
        pose_success_rate=float(success.mean()),
        brier_score=float(((score - success) ** 2).mean()),
        expected_calibration_error=float(expected_calibration_error(score, success, calibration_bins)),
        rotation_error_deg=rotation_error_deg,
        accuracy_5deg_5cm=accuracy_5,
        accuracy_10deg_10cm=accuracy_10,
        surface_distances_norm=(
            torch.cat(surface_distance_values).detach().float().cpu()
            if surface_distance_values
            else None
        ),
    )


def expected_calibration_error(probabilities: Tensor, targets: Tensor, bins: int = 10) -> Tensor:
    if bins <= 0:
        raise ValueError("bins must be positive")
    probabilities, targets = probabilities.flatten(), targets.flatten().to(probabilities)
    if probabilities.numel() != targets.numel():
        raise ValueError("probabilities and targets must have the same size")
    error = probabilities.new_zeros(())
    edges = torch.linspace(0, 1, bins + 1, device=probabilities.device, dtype=probabilities.dtype)
    for index in range(bins):
        selected = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        )
        if selected.any():
            fraction = selected.float().mean()
            error = error + fraction * (probabilities[selected].mean() - targets[selected].mean()).abs()
    return error


def fit_pose_score_temperature(logits: Tensor, success_targets: Tensor, *, max_iterations: int = 50) -> float:
    """Fit one positive temperature on validation logits only."""

    logits = logits.detach().float().flatten()
    success_targets = success_targets.detach().float().flatten()
    if logits.numel() == 0 or logits.numel() != success_targets.numel():
        raise ValueError("Validation logits and targets must be non-empty and equally sized")
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iterations)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp_min(1e-6)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, success_targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp_min(1e-6))
