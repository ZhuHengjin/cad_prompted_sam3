"""One-to-one mask-based assignment for pose supervision."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def mask_iou_matrix(prediction_logits_nhw: Tensor, targets_hw: Sequence[Tensor]) -> Tensor:
    """Return a ``GT x prediction`` binary-mask IoU matrix."""

    if prediction_logits_nhw.ndim != 3:
        raise ValueError("prediction_logits_nhw must have shape NxHxW")
    if not targets_hw:
        return prediction_logits_nhw.new_zeros((0, prediction_logits_nhw.shape[0]))
    target_stack = torch.stack([target.to(prediction_logits_nhw.device) > 0.5 for target in targets_hw])
    predicted = prediction_logits_nhw > 0
    intersection = (target_stack.unsqueeze(1) & predicted.unsqueeze(0)).sum(dim=(-2, -1))
    union = (target_stack.unsqueeze(1) | predicted.unsqueeze(0)).sum(dim=(-2, -1)).clamp_min(1)
    return intersection.float() / union.float()


def match_pose_predictions_one_to_one(
    prediction_logits_nhw: Tensor,
    targets_hw: Sequence[Tensor],
    *,
    eligible_gt_indices: Sequence[int] | None = None,
) -> tuple[list[tuple[int, int]], Tensor]:
    """Greedily select the highest-IoU unique prediction for each eligible GT.

    Returns ``(gt_index, prediction_index)`` pairs. Pose predictions never enter
    the assignment cost, which keeps early training stable.
    """

    iou = mask_iou_matrix(prediction_logits_nhw, targets_hw)
    eligible = list(range(len(targets_hw))) if eligible_gt_indices is None else list(eligible_gt_indices)
    if not eligible or iou.shape[1] == 0:
        return [], iou
    bad = [idx for idx in eligible if idx < 0 or idx >= len(targets_hw)]
    if bad:
        raise IndexError(f"Eligible GT indices are out of range: {bad}")

    candidates = []
    for gt_index in eligible:
        for prediction_index in range(iou.shape[1]):
            candidates.append((float(iou[gt_index, prediction_index].detach()), gt_index, prediction_index))
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gt_index, prediction_index in candidates:
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        matches.append((gt_index, prediction_index))
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        if len(used_gt) == min(len(eligible), iou.shape[1]):
            break
    matches.sort()
    return matches, iou
