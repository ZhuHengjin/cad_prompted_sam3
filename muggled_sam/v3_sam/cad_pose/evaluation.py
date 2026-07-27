"""Symmetry-aware pose metrics and held-out pose-score calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .symmetry import symmetry_aware_rotation_error
from .types import CADPosePredictions, CADPoseTarget


@dataclass(frozen=True)
class PoseEvaluation:
    count: int
    rotation_error_deg: float
    translation_error_cm: float
    center_error_norm: float
    depth_error_m: float
    accuracy_5deg_5cm: float
    accuracy_10deg_10cm: float
    brier_score: float
    expected_calibration_error: float


def evaluate_pose_matches(
    predictions: CADPosePredictions,
    targets: Sequence[CADPoseTarget],
    matches: Sequence[tuple[int, int]],
    *,
    batch_index: int = 0,
    calibration_bins: int = 10,
) -> PoseEvaluation:
    if predictions.translation_m_bn3 is None:
        raise ValueError("Evaluation requires reconstructed metric translations")
    rotation_errors, translation_errors, center_errors, depth_errors, scores = [], [], [], [], []
    for gt_index, prediction_index in matches:
        target = targets[gt_index]
        if not target.rotation_eligible:
            continue
        predicted_rotation = predictions.rotation_matrix_bn33[batch_index, prediction_index]
        rotation_errors.append(
            symmetry_aware_rotation_error(
                predicted_rotation,
                target.rotation_matrix.to(predicted_rotation),
                symmetry_type=target.symmetry_type,
                symmetry_transforms=(
                    target.symmetry_transforms.to(predicted_rotation) if target.symmetry_transforms is not None else None
                ),
                axis_cad=target.axis_cad.to(predicted_rotation) if target.axis_cad is not None else None,
            )
        )
        translation = predictions.translation_m_bn3[batch_index, prediction_index]
        translation_errors.append(torch.linalg.vector_norm(translation - target.translation_m.to(translation)))
        center_errors.append(
            torch.linalg.vector_norm(
                predictions.center_uv_norm_bn2[batch_index, prediction_index]
                - target.center_uv_norm.to(predictions.center_uv_norm_bn2)
            )
        )
        predicted_depth = predictions.log_depth_bn[batch_index, prediction_index].exp()
        depth_errors.append((predicted_depth - target.log_depth.to(predicted_depth).exp()).abs())
        scores.append(predictions.pose_score_bn[batch_index, prediction_index])
    if not rotation_errors:
        nan = float("nan")
        return PoseEvaluation(0, nan, nan, nan, nan, nan, nan, nan, nan)
    rotation = torch.stack(rotation_errors)
    translation = torch.stack(translation_errors)
    score = torch.stack(scores)
    success_5 = ((rotation <= torch.deg2rad(rotation.new_tensor(5.0))) & (translation <= 0.05)).float()
    success_10 = ((rotation <= torch.deg2rad(rotation.new_tensor(10.0))) & (translation <= 0.10)).float()
    return PoseEvaluation(
        len(rotation_errors),
        float(torch.rad2deg(rotation).mean()),
        float(translation.mean() * 100.0),
        float(torch.stack(center_errors).mean()),
        float(torch.stack(depth_errors).mean()),
        float(success_5.mean()),
        float(success_10.mean()),
        float(((score - success_5) ** 2).mean()),
        float(expected_calibration_error(score, success_5, calibration_bins)),
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
