"""Helpers for preserving candidate identity and formatting CAD pose results."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .types import CADPosePredictions


def mask_nms_indices(mask_logits_nhw: Tensor, detection_scores_n: Tensor, iou_threshold: float) -> Tensor:
    """Return score-ordered retained indices for mask-IoU/containment NMS."""

    order = torch.argsort(detection_scores_n, descending=True)
    if iou_threshold <= 0 or order.numel() == 0:
        return order
    binary = mask_logits_nhw > 0
    keep: list[int] = []
    for index in order.tolist():
        candidate = binary[index]
        candidate_area = candidate.sum().clamp_min(1)
        suppress = False
        for kept_index in keep:
            kept = binary[kept_index]
            intersection = (candidate & kept).sum()
            union = (candidate | kept).sum().clamp_min(1)
            if (
                float(intersection.float() / union.float()) > iou_threshold
                or float(intersection.float() / candidate_area.float()) >= 0.95
                or float(intersection.float() / kept.sum().clamp_min(1).float()) >= 0.95
            ):
                suppress = True
                break
        if not suppress:
            keep.append(int(index))
    return torch.as_tensor(keep, device=detection_scores_n.device, dtype=torch.long)


def select_detection_candidates(
    masks_nhw: Tensor,
    boxes_n22: Tensor,
    detection_scores_n: Tensor,
    poses: CADPosePredictions,
    retained_indices: Tensor,
) -> tuple[Tensor, Tensor, Tensor, CADPosePredictions]:
    """Apply threshold/NMS indices to every candidate-aligned output."""

    if poses.center_uv_norm_bn2.shape[0] != 1:
        raise ValueError("Select candidates one image at a time")
    retained_indices = retained_indices.to(device=detection_scores_n.device, dtype=torch.long)
    return (
        masks_nhw[retained_indices],
        boxes_n22[retained_indices],
        detection_scores_n[retained_indices],
        poses.index_candidates(retained_indices),
    )


def format_cad_pose_results(
    masks_nhw: Tensor,
    boxes_n22: Tensor,
    detection_scores_n: Tensor,
    poses: CADPosePredictions,
    *,
    cad_id: str,
) -> list[dict[str, Any]]:
    """Create the plan's one-dictionary-per-retained-instance output contract."""

    if poses.translation_m_bn3 is None or poses.cad_dimensions_m_b3 is None:
        raise ValueError("Formatted pose results require translation and echoed CAD dimensions")
    if poses.center_uv_norm_bn2.shape[0] != 1:
        raise ValueError("Format one image batch at a time")
    if not (len(masks_nhw) == len(boxes_n22) == len(detection_scores_n) == poses.center_uv_norm_bn2.shape[1]):
        raise ValueError("Candidate-aligned result tensors have different lengths")
    dimensions = poses.cad_dimensions_m_b3[0]
    return [
        {
            "mask_logits": masks_nhw[index],
            "box_xyxy": boxes_n22[index],
            "detection_score": detection_scores_n[index],
            "rotation_matrix": poses.rotation_matrix_bn33[0, index],
            "translation_m": poses.translation_m_bn3[0, index],
            "cad_dimensions_m": dimensions,
            "pose_score": poses.pose_score_bn[0, index],
            "cad_id": cad_id,
        }
        for index in range(len(detection_scores_n))
    ]
