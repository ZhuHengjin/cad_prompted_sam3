from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


def _bbox_from_mask_xy1xy2(mask_hw: torch.Tensor) -> Optional[torch.Tensor]:
    if mask_hw.numel() == 0:
        return None
    mask_bin = mask_hw > 0.5
    if not mask_bin.any():
        return None
    ys, xs = torch.where(mask_bin)
    if ys.numel() == 0 or xs.numel() == 0:
        return None
    y1 = ys.min().float()
    y2 = ys.max().float()
    x1 = xs.min().float()
    x2 = xs.max().float()
    h, w = mask_hw.shape[-2:]
    den_w = float(max(int(w - 1), 1))
    den_h = float(max(int(h - 1), 1))
    return torch.stack((x1 / den_w, y1 / den_h, x2 / den_w, y2 / den_h))


def compute_bbox_l1_loss_from_matches(
    box_preds_n22: torch.Tensor,
    gt_targets_hw: Sequence[torch.Tensor],
    matches: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    if not matches or box_preds_n22.numel() == 0:
        return torch.zeros((), device=box_preds_n22.device)
    if box_preds_n22.ndim == 3 and box_preds_n22.shape[-2:] == (2, 2):
        box_preds_n4 = box_preds_n22.reshape(-1, 4)
    elif box_preds_n22.ndim == 2 and box_preds_n22.shape[1] == 4:
        box_preds_n4 = box_preds_n22
    elif box_preds_n22.ndim == 1 and box_preds_n22.numel() == 4:
        box_preds_n4 = box_preds_n22.unsqueeze(0)
    else:
        return torch.zeros((), device=box_preds_n22.device)

    pred_boxes: List[torch.Tensor] = []
    gt_boxes: List[torch.Tensor] = []
    for gt_idx, pred_idx in matches:
        if pred_idx < 0 or pred_idx >= box_preds_n4.shape[0]:
            continue
        if gt_idx < 0 or gt_idx >= len(gt_targets_hw):
            continue
        gt_box = _bbox_from_mask_xy1xy2(gt_targets_hw[gt_idx])
        if gt_box is None:
            continue
        pred_boxes.append(box_preds_n4[pred_idx])
        gt_boxes.append(gt_box)

    if not pred_boxes:
        return torch.zeros((), device=box_preds_n4.device)
    pred_stack = torch.stack(pred_boxes)
    gt_stack = torch.stack(gt_boxes).to(device=box_preds_n4.device, dtype=box_preds_n4.dtype)
    return F.l1_loss(pred_stack, gt_stack, reduction="mean")


def select_best_mask(mask_preds_nhw: torch.Tensor, gt_mask_hw: torch.Tensor) -> Tuple[int, torch.Tensor]:
    pred_bin = mask_preds_nhw > 0
    gt_bin = gt_mask_hw > 0.5
    intersection = (pred_bin & gt_bin).sum(dim=(1, 2))
    union = (pred_bin | gt_bin).sum(dim=(1, 2)).clamp_min(1)
    iou = intersection.float() / union.float()
    best_idx = int(torch.argmax(iou).item())
    return best_idx, iou[best_idx]


def dice_loss_from_logits(logits_hw: torch.Tensor, target_hw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits_hw)
    num = 2 * (probs * target_hw).sum() + eps
    den = probs.sum() + target_hw.sum() + eps
    return 1.0 - num / den


def compute_mask_loss_best(
    logits_mhw: torch.Tensor,
    gt_target_hw: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
) -> Tuple[torch.Tensor, int]:
    gt_broadcast = gt_target_hw.unsqueeze(0).expand_as(logits_mhw)
    loss_bce_per = F.binary_cross_entropy_with_logits(
        logits_mhw,
        gt_broadcast,
        reduction="none",
    ).mean(dim=(1, 2))
    probs_mhw = torch.sigmoid(logits_mhw)
    eps = 1e-6
    dice_num = 2 * (probs_mhw * gt_broadcast).sum(dim=(1, 2)) + eps
    dice_den = probs_mhw.sum(dim=(1, 2)) + gt_broadcast.sum(dim=(1, 2)) + eps
    loss_dice_per = 1.0 - dice_num / dice_den
    loss_per_mask = bce_weight * loss_bce_per + dice_weight * loss_dice_per
    best_idx, _ = select_best_mask(logits_mhw, gt_target_hw)
    loss_mask = loss_per_mask[best_idx]
    return loss_mask, best_idx


def compute_mask_loss_per_pred(
    logits_mhw: torch.Tensor,
    gt_target_hw: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
) -> torch.Tensor:
    gt_broadcast = gt_target_hw.unsqueeze(0).expand_as(logits_mhw)
    loss_bce_per = F.binary_cross_entropy_with_logits(
        logits_mhw,
        gt_broadcast,
        reduction="none",
    ).mean(dim=(1, 2))
    probs_mhw = torch.sigmoid(logits_mhw)
    eps = 1e-6
    dice_num = 2 * (probs_mhw * gt_broadcast).sum(dim=(1, 2)) + eps
    dice_den = probs_mhw.sum(dim=(1, 2)) + gt_broadcast.sum(dim=(1, 2)) + eps
    loss_dice_per = 1.0 - dice_num / dice_den
    return bce_weight * loss_bce_per + dice_weight * loss_dice_per


def compute_matched_mask_losses(
    logits_mhw: torch.Tensor,
    gt_targets_hw: Sequence[torch.Tensor],
    matches: Sequence[Tuple[int, int]],
    bce_weight: float,
    dice_weight: float,
) -> List[torch.Tensor]:
    losses: List[torch.Tensor] = []
    for gt_idx, pred_idx in matches:
        loss_per_mask = compute_mask_loss_per_pred(
            logits_mhw,
            gt_targets_hw[gt_idx],
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )
        losses.append(loss_per_mask[pred_idx])
    return losses


def compute_score_loss(scores_logits: torch.Tensor, best_idx: int) -> torch.Tensor:
    scores_logits = scores_logits.float().unsqueeze(0)
    target_idx = torch.tensor([best_idx], device=scores_logits.device)
    return F.cross_entropy(scores_logits, target_idx)


def compute_noobj_loss(scores_logits: torch.Tensor, best_idx: int) -> torch.Tensor:
    scores_logits = scores_logits.float()
    if scores_logits.numel() <= 1:
        return torch.zeros((), device=scores_logits.device)
    noobj_mask = torch.ones_like(scores_logits, dtype=torch.bool)
    noobj_mask[best_idx] = False
    return F.binary_cross_entropy_with_logits(
        scores_logits[noobj_mask],
        torch.zeros_like(scores_logits[noobj_mask]),
    )


def compute_presence_loss(
    scores_logits: torch.Tensor,
    positive_indices: Union[int, Sequence[int]],
    pos_weight: float,
    neg_weight: float,
    use_focal: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    focal_weight: float = 1.0,
) -> torch.Tensor:
    scores_logits = scores_logits.float()
    # scores_logits are already probabilities in [0, 1]
    prob = scores_logits.clamp(1e-4, 1 - 1e-4)
    target = torch.zeros_like(scores_logits)
    if scores_logits.numel() > 0:
        if isinstance(positive_indices, int):
            pos_indices = [positive_indices]
        else:
            pos_indices = list(positive_indices)
        for idx in pos_indices:
            if 0 <= idx < target.numel():
                target[idx] = 1.0
    loss_per = F.binary_cross_entropy(prob, target, reduction="none")
    pos_mask = target > 0.5
    neg_mask = ~pos_mask
    if pos_mask.any():
        loss_pos = loss_per[pos_mask].mean()
    else:
        loss_pos = torch.zeros((), device=scores_logits.device)
    if neg_mask.any():
        loss_neg = loss_per[neg_mask].mean()
    else:
        loss_neg = torch.zeros((), device=scores_logits.device)
    loss = pos_weight * loss_pos + neg_weight * loss_neg
    if use_focal and scores_logits.numel() > 0:
        p_t = prob * target + (1 - prob) * (1 - target)
        focal = -((1 - p_t) ** focal_gamma) * torch.log(p_t)
        if focal_alpha >= 0:
            alpha_t = focal_alpha * target + (1 - focal_alpha) * (1 - target)
            focal = alpha_t * focal
        loss = loss + focal_weight * focal.mean()
    return loss


def compute_presence_loss_logits(
    scores_logits: torch.Tensor,
    matches: Sequence[Tuple[int, int]],
    iou: Optional[torch.Tensor],
    pos_weight: float,
    neg_weight: float,
    alpha: float = 0.2,
    use_focal: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    focal_weight: float = 1.0,
) -> torch.Tensor:
    scores_logits = scores_logits.float()
    target = torch.zeros_like(scores_logits)
    if scores_logits.numel() > 0 and matches:
        for gt_idx, pred_idx in matches:
            if 0 <= pred_idx < target.numel():
                soft_target = torch.tensor(1.0, device=target.device, dtype=target.dtype)
                if iou is not None and 0 <= gt_idx < iou.shape[0] and 0 <= pred_idx < iou.shape[1]:
                    prob = torch.sigmoid(scores_logits[pred_idx])
                    iou_val = iou[gt_idx, pred_idx].clamp(0.0, 1.0)
                    soft = (prob**alpha) * (iou_val ** (1.0 - alpha))
                    soft_target = soft.clamp(min=0.11).detach()
                target[pred_idx] = torch.maximum(target[pred_idx], soft_target)
    loss_per = F.binary_cross_entropy_with_logits(scores_logits, target, reduction="none")
    pos_mask = target > 0.1
    neg_mask = ~pos_mask
    if pos_mask.any():
        loss_pos = loss_per[pos_mask].mean()
    else:
        loss_pos = torch.zeros((), device=scores_logits.device)
    if neg_mask.any():
        loss_neg = loss_per[neg_mask].mean()
    else:
        loss_neg = torch.zeros((), device=scores_logits.device)
    loss = pos_weight * loss_pos + neg_weight * loss_neg
    if use_focal and scores_logits.numel() > 0:
        prob = torch.sigmoid(scores_logits)
        p_t = prob * target + (1 - prob) * (1 - target)
        focal = -((1 - p_t) ** focal_gamma) * torch.log(p_t.clamp_min(1e-6))
        if focal_alpha >= 0:
            alpha_t = focal_alpha * target + (1 - focal_alpha) * (1 - target)
            focal = alpha_t * focal
        loss = loss + focal_weight * focal.mean()
    return loss


def compute_score_weighted_mask_loss(
    logits_mhw: torch.Tensor,
    gt_target_hw: torch.Tensor,
    scores_logits: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
    power: float,
) -> torch.Tensor:
    gt_broadcast = gt_target_hw.unsqueeze(0).expand_as(logits_mhw)
    loss_bce_per = F.binary_cross_entropy_with_logits(
        logits_mhw,
        gt_broadcast,
        reduction="none",
    ).mean(dim=(1, 2))
    probs_mhw = torch.sigmoid(logits_mhw)
    eps = 1e-6
    dice_num = 2 * (probs_mhw * gt_broadcast).sum(dim=(1, 2)) + eps
    dice_den = probs_mhw.sum(dim=(1, 2)) + gt_broadcast.sum(dim=(1, 2)) + eps
    loss_dice_per = 1.0 - dice_num / dice_den
    loss_per_mask = bce_weight * loss_bce_per + dice_weight * loss_dice_per

    # scores = torch.sigmoid(scores_logits.float())
    # weights = torch.softmax(scores**power, dim=0)
    # compute_score_weighted_mask_loss / _from_matches: remove sigmoid
    scores = scores_probs.float()  # no torch.sigmoid(...)
    weights = torch.softmax(scores**power, dim=0)

    return (weights * loss_per_mask).sum()


def compute_score_weighted_mask_loss_from_matches(
    scores_logits: torch.Tensor,
    matches: Sequence[Tuple[int, int]],
    matched_losses: Sequence[torch.Tensor],
    power: float,
) -> torch.Tensor:
    if not matches or not matched_losses:
        return torch.zeros((), device=scores_logits.device)
    scores = torch.sigmoid(scores_logits.float())
    weights = torch.softmax(scores**power, dim=0)
    pred_indices = [pred_idx for _, pred_idx in matches]
    sel_weights = weights[pred_indices]
    sel_weights = sel_weights / sel_weights.sum().clamp_min(1e-6)
    weighted = [weight * loss for weight, loss in zip(sel_weights, matched_losses)]
    return torch.stack(weighted).sum()
