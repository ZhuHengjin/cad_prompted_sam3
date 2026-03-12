#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from muggled_sam.make_sam import make_sam_from_state_dict
from finetune_image_exemplar import save_debug_collage


IGNORED_LABELS = {"BACKGROUND", "UNLABELLED"}


def parse_color_key(key: str) -> Tuple[int, ...]:
    stripped = key.strip().strip("()")
    parts = [part.strip() for part in stripped.split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def load_color_mapping(json_path: Path) -> Dict[str, List[Tuple[int, ...]]]:
    with open(json_path, "r") as handle:
        raw = json.load(handle)
    object_colors: Dict[str, List[Tuple[int, ...]]] = defaultdict(list)
    for color_key, label in raw.items():
        clean_label = label.strip()
        if not clean_label or clean_label.upper() in IGNORED_LABELS:
            continue
        object_colors[clean_label].append(parse_color_key(color_key))
    return dict(object_colors)


def collect_multi_object_samples(
    dataset_root: str,
    sub_sample: int = 5,
) -> Tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
    dataset_path = Path(dataset_root).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_root)
    pattern = re.compile(r"instance_segmentation_(\d{4})\.png$")
    object_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    all_entries: List[Dict[str, str]] = []
    if sub_sample < 1:
        raise ValueError("sub_sample must be >= 1")
    for idx, inst_path in enumerate(sorted(dataset_path.glob("instance_segmentation_*.png"))):
        if sub_sample > 1 and (idx % sub_sample) != 0:
            continue
        match = pattern.match(inst_path.name)
        if not match:
            continue
        frame_id = match.group(1)
        rgb_path = dataset_path / f"rgb_{frame_id}.png"
        if not rgb_path.is_file():
            jpg_fallback = dataset_path / f"rgb_{frame_id}.jpg"
            if jpg_fallback.is_file():
                rgb_path = jpg_fallback
        mapping_path = dataset_path / f"instance_segmentation_mapping_{frame_id}.json"
        if not (rgb_path.is_file() and mapping_path.is_file()):
            continue
        color_map = load_color_mapping(mapping_path)
        if not color_map:
            continue
        for object_id, colors in color_map.items():
            for color in colors:
                entry = {
                    "object_id": object_id,
                    "frame_id": frame_id,
                    "rgb_path": str(rgb_path),
                    "inst_path": str(inst_path),
                    "color": color,
                }
                object_map[object_id].append(entry)
                all_entries.append(entry)
    if not all_entries:
        raise RuntimeError(f"No multi-object samples found under {dataset_root}")
    return object_map, all_entries


def collect_itodd_object_samples(
    dataset_root: str,
    sub_sample: int = 1,
) -> Tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
    dataset_path = Path(dataset_root).expanduser().resolve()
    targets_path = dataset_path / "itodd" / "test_targets_bop19.json"
    if not targets_path.is_file():
        raise FileNotFoundError(targets_path)
    if sub_sample < 1:
        raise ValueError("sub_sample must be >= 1")

    with open(targets_path, "r") as handle:
        targets = json.load(handle)

    object_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    all_entries: List[Dict[str, str]] = []
    for idx, target in enumerate(targets):
        if sub_sample > 1 and (idx % sub_sample) != 0:
            continue
        scene_id = int(target["scene_id"])
        im_id = int(target["im_id"])
        obj_id = int(target["obj_id"])
        inst_count = int(target.get("inst_count", 0))
        scene_dir = dataset_path / "test" / f"{scene_id:06d}"
        gray_dir = scene_dir / "gray"
        mask_dir = scene_dir / "mask_visib"
        rgb_path = gray_dir / f"{im_id:06d}.tif"
        if not rgb_path.is_file():
            png_fallback = gray_dir / f"{im_id:06d}.png"
            if png_fallback.is_file():
                rgb_path = png_fallback
        if not (rgb_path.is_file() and mask_dir.is_dir()):
            continue

        object_id = f"obj_{obj_id:06d}"
        frame_id = f"{scene_id:06d}_{im_id:06d}"
        entry = {
            "object_id": object_id,
            "frame_id": frame_id,
            "rgb_path": str(rgb_path),
            "inst_path": str(mask_dir),
            "scene_id": scene_id,
            "im_id": im_id,
            "inst_count": inst_count,
        }
        object_map[object_id].append(entry)
        all_entries.append(entry)
    if not all_entries:
        raise RuntimeError(f"No ITODD samples found under {dataset_root}")
    return dict(object_map), all_entries


def load_instance_mask(inst_path: str, color: Tuple[int, ...]) -> np.ndarray:
    seg = load_instance_segmentation(inst_path)
    return mask_for_color(seg, color)


def load_instance_segmentation(inst_path: str) -> np.ndarray:
    seg = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        raise FileNotFoundError(inst_path)
    if seg.ndim == 2:
        seg = seg[..., None]
    elif seg.shape[2] == 4:
        seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2RGBA)
    elif seg.shape[2] == 3:
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    return seg.astype(np.uint8)


def mask_for_color(seg: np.ndarray, color: Tuple[int, ...]) -> np.ndarray:
    target = np.array(color, dtype=np.uint8)
    channels = seg.shape[2]
    if target.shape[0] > channels:
        target = target[:channels]
    elif target.shape[0] < channels:
        pad = np.zeros(channels - target.shape[0], dtype=np.uint8)
        target = np.concatenate([target, pad], axis=0)
    target = target.reshape(1, 1, -1)
    mask = np.all(seg == target, axis=-1)
    return mask.astype(np.float32)


def load_instance_masks_for_object(
    inst_path: str,
    mapping_path: str,
    object_id: str,
    seg_cache: Optional[Dict[str, np.ndarray]] = None,
    mapping_cache: Optional[Dict[str, Dict[str, List[Tuple[int, ...]]]]] = None,
) -> List[np.ndarray]:
    mapping_key = str(mapping_path)
    if mapping_cache is not None and mapping_key in mapping_cache:
        color_map = mapping_cache[mapping_key]
    else:
        color_map = load_color_mapping(Path(mapping_path))
        if mapping_cache is not None:
            mapping_cache[mapping_key] = color_map
    colors = color_map.get(object_id, [])
    if not colors:
        return []

    inst_key = str(inst_path)
    if seg_cache is not None and inst_key in seg_cache:
        seg = seg_cache[inst_key]
    else:
        seg = load_instance_segmentation(inst_path)
        if seg_cache is not None:
            seg_cache[inst_key] = seg

    masks: List[np.ndarray] = []
    for color in colors:
        mask = mask_for_color(seg, color)
        if mask.sum() > 0:
            masks.append(mask)
    return masks


def load_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_mask_gray(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def load_itodd_masks(
    mask_dir: str,
    im_id: int,
    inst_count: int = 0,
) -> List[np.ndarray]:
    mask_dir_path = Path(mask_dir)
    mask_paths = sorted(mask_dir_path.glob(f"{im_id:06d}_*.png"))
    if inst_count > 0 and len(mask_paths) > inst_count:
        mask_paths = mask_paths[:inst_count]
    masks: List[np.ndarray] = []
    for mask_path in mask_paths:
        try:
            mask = load_mask_gray(str(mask_path))
        except FileNotFoundError:
            continue
        if mask.sum() > 0:
            masks.append(mask)
    return masks


def sample_points_from_mask(mask_image: np.ndarray, num_points_approx: int = 25) -> List[Tuple[float, float]]:
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    num_fib_pts = golden_ratio * num_points_approx
    pt_idx = np.arange(0, num_fib_pts, dtype=np.float32)
    r = (1) * np.sqrt(pt_idx / num_fib_pts) / np.sqrt(2, dtype=np.float32)
    theta = 2.0 * np.pi * (pt_idx / golden_ratio)

    fib_sample_x_norm = 0.5 + r * np.cos(theta)
    fib_sample_y_norm = 0.5 + r * np.sin(theta)

    ok_x_pts = np.bitwise_and(fib_sample_x_norm > 0.0, fib_sample_x_norm < 1.0)
    ok_y_pts = np.bitwise_and(fib_sample_y_norm > 0.0, fib_sample_y_norm < 1.0)
    ok_pts = np.bitwise_and(ok_x_pts, ok_y_pts)
    fib_sample_x_norm, fib_sample_y_norm = fib_sample_x_norm[ok_pts], fib_sample_y_norm[ok_pts]

    ref_h, ref_w = mask_image.shape[0:2]
    if mask_image.ndim > 2:
        if mask_image.shape[2] == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        else:
            mask_image = mask_image[:, :, 0]

    mask_bin = mask_image > 0
    contours_px_list, _ = cv2.findContours(np.uint8(mask_bin), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_sample_xy_px_list = []
    for contour_pts_list in contours_px_list:
        if len(contour_pts_list) < 3:
            continue
        x1_px, y1_px, w_px, h_px = cv2.boundingRect(contour_pts_list)
        if w_px < 1 or h_px < 1:
            continue

        sample_x_px = np.round(x1_px + fib_sample_x_norm * (w_px - 1)).astype(np.int32)
        sample_y_px = np.round(y1_px + fib_sample_y_norm * (h_px - 1)).astype(np.int32)
        is_in_mask = mask_bin[sample_y_px, sample_x_px]
        final_sample_xy_px = np.column_stack((sample_x_px[is_in_mask], sample_y_px[is_in_mask]))
        final_sample_xy_px_list.append(final_sample_xy_px)

    if not final_sample_xy_px_list:
        return []

    out_xy_norm = np.concatenate(final_sample_xy_px_list) / np.float32((ref_w - 1, ref_h - 1))
    return out_xy_norm.tolist()


def resize_mask(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)


def parse_ref_view_ids(value: str) -> List[str]:
    if not value:
        return []
    parts = [part.strip() for part in value.split(",") if part.strip()]
    ids: List[str] = []
    for part in parts:
        try:
            as_int = int(part)
            ids.append(f"{as_int:02d}")
        except ValueError:
            ids.append(part)
    return ids


def generate_detections_train(
    detmodel,
    encoded_image_features_list: List[torch.Tensor],
    encoded_exemplars_bnc: torch.Tensor,
    detection_filter_threshold: float = 0.0,
    exemplar_padding_mask_bn: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lowres_imgenc_bchw, hiresx2_imgenc_bchw, hiresx4_imgenc_bchw = encoded_image_features_list
    no_exemplars = encoded_exemplars_bnc.shape[1] == 0
    if no_exemplars:
        blk_tok, blk_box, blk_score, blk_score_logits, blk_pres = detmodel.exemplar_detector.create_blank_output(
            lowres_imgenc_bchw
        )
        blk_masks, _ = detmodel.exemplar_segmentation.create_blank_output(blk_tok, lowres_imgenc_bchw)
        return blk_masks, blk_box, blk_score, blk_pres

    fused_imgexm_tokens_bchw = detmodel.image_exemplar_fusion(
        lowres_imgenc_bchw,
        encoded_exemplars_bnc,
        exemplar_padding_mask_bn,
    )
    enc_det_tokens_bnc, boxes_xy1xy2_bn22, det_scores_bn, det_scores_logits_bn, pres_scores = detmodel.exemplar_detector(
        fused_imgexm_tokens_bchw, encoded_exemplars_bnc, exemplar_padding_mask_bn
    )

    if detection_filter_threshold > 1e-3:
        if det_scores_bn.shape[0] != 1:
            raise ValueError("Cannot pre-filter detections when using batched inputs!")
        ok_filter = det_scores_bn[0] > detection_filter_threshold
        enc_det_tokens_bnc = enc_det_tokens_bnc[:, ok_filter]
        boxes_xy1xy2_bn22 = boxes_xy1xy2_bn22[:, ok_filter]
        det_scores_bn = det_scores_bn[:, ok_filter]

    mask_preds_bnhw, _ = detmodel.exemplar_segmentation(
        enc_det_tokens_bnc,
        fused_imgexm_tokens_bchw,
        hiresx2_imgenc_bchw,
        hiresx4_imgenc_bchw,
        encoded_exemplars_bnc,
        exemplar_padding_mask_bn,
    )
    return mask_preds_bnhw, boxes_xy1xy2_bn22, det_scores_bn, pres_scores


def encode_detection_image_no_infer(
    detmodel,
    image_bgr: np.ndarray,
    max_side_length: int,
    use_square_sizing: bool,
) -> Tuple[List[torch.Tensor], Tuple[int, int], Tuple[int, int]]:
    image_rgb_normalized_bchw = detmodel.image_encoder.prepare_image(
        image_bgr, max_side_length=max_side_length, use_square_sizing=use_square_sizing
    )
    with torch.no_grad():
        encoded_img = detmodel.image_encoder(image_rgb_normalized_bchw)
        encoded_image_features_list = detmodel.image_projection.v3_projection(encoded_img)
    patch_grid_hw = encoded_image_features_list[0].shape[2:]
    image_preenc_hw = image_rgb_normalized_bchw.shape[2:]
    return encoded_image_features_list, patch_grid_hw, image_preenc_hw


def encode_exemplars_no_infer(
    detmodel,
    encoded_image_features_list: List[torch.Tensor],
    text: Optional[str],
    point_xy_norm_list: Optional[List[Tuple[float, float]]],
    include_coordinate_encodings: bool,
) -> torch.Tensor:
    lowres_imgenc_bchw = encoded_image_features_list[0]
    img_b, img_c, _, _ = lowres_imgenc_bchw.shape
    device, dtype = lowres_imgenc_bchw.device, lowres_imgenc_bchw.dtype
    missing_input_tensor_bnc = torch.empty((img_b, 0, img_c), device=device, dtype=dtype)
    encoded_text_bnc = missing_input_tensor_bnc
    encoded_sampling_bnc = missing_input_tensor_bnc
    with torch.no_grad():
        if isinstance(text, str) and len(text) > 0:
            encoded_text_bnc = detmodel.text_encoder(text)
        if point_xy_norm_list is not None:
            encoded_sampling_bnc = detmodel.sampling_encoder(
                lowres_imgenc_bchw,
                boxes_bn22=detmodel.sampling_encoder.prepare_box_input(None),
                points_bn2=detmodel.sampling_encoder.prepare_point_input(point_xy_norm_list),
                negative_boxes_bn22=detmodel.sampling_encoder.prepare_box_input(None),
                negative_points_bn2=detmodel.sampling_encoder.prepare_point_input(None),
                include_coordinate_encodings=include_coordinate_encodings,
            )
    return torch.cat((encoded_sampling_bnc, encoded_text_bnc), dim=1)


def select_best_mask(mask_preds_nhw: torch.Tensor, gt_mask_hw: torch.Tensor) -> Tuple[int, torch.Tensor]:
    pred_bin = mask_preds_nhw > 0
    gt_bin = gt_mask_hw > 0.5
    intersection = (pred_bin & gt_bin).sum(dim=(1, 2))
    union = (pred_bin | gt_bin).sum(dim=(1, 2)).clamp_min(1)
    iou = intersection.float() / union.float()
    best_idx = int(torch.argmax(iou).item())
    return best_idx, iou[best_idx]


def compute_mask_iou(pred_mask_hw: torch.Tensor, gt_mask_hw: torch.Tensor) -> torch.Tensor:
    pred_bin = pred_mask_hw > 0
    gt_bin = gt_mask_hw > 0.5
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum().clamp_min(1)
    return intersection.float() / union.float()


def compute_pq_stats(
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    pred_scores: Optional[torch.Tensor] = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.2,
) -> Tuple[float, int, int, int]:
    if pred_scores is not None and pred_masks:
        filtered_masks: List[torch.Tensor] = []
        for idx, mask in enumerate(pred_masks):
            if float(pred_scores[idx].item()) >= score_threshold:
                filtered_masks.append(mask)
        pred_masks = filtered_masks

    if not pred_masks and not gt_masks:
        return 0.0, 0, 0, 0
    if not pred_masks:
        return 0.0, 0, 0, len(gt_masks)
    if not gt_masks:
        return 0.0, 0, len(pred_masks), 0

    pairs: List[Tuple[float, int, int]] = []
    for p_idx, pred in enumerate(pred_masks):
        for g_idx, gt in enumerate(gt_masks):
            iou = compute_mask_iou(pred, gt)
            iou_val = float(iou.item())
            if iou_val >= iou_threshold:
                pairs.append((iou_val, p_idx, g_idx))
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = [False] * len(pred_masks)
    matched_gt = [False] * len(gt_masks)
    sum_iou = 0.0
    tp = 0
    for iou_val, p_idx, g_idx in pairs:
        if matched_pred[p_idx] or matched_gt[g_idx]:
            continue
        matched_pred[p_idx] = True
        matched_gt[g_idx] = True
        tp += 1
        sum_iou += iou_val

    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    return sum_iou, tp, fp, fn


def apply_mask_nms(
    box_preds_n22: torch.Tensor,
    mask_preds_nhw: torch.Tensor,
    det_scores_n: torch.Tensor,
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if det_scores_n.numel() == 0:
        return box_preds_n22, mask_preds_nhw, det_scores_n

    score_order = torch.argsort(det_scores_n, descending=True)
    if iou_threshold <= 0:
        return box_preds_n22[score_order], mask_preds_nhw[score_order], det_scores_n[score_order]

    masks_bin = mask_preds_nhw > 0
    keep: List[int] = []
    for idx in score_order.tolist():
        if not keep:
            keep.append(int(idx))
            continue
        suppress = False
        cand = masks_bin[idx]
        cand_area = cand.sum().clamp_min(1)
        for kept_idx in keep:
            kept = masks_bin[kept_idx]
            inter = (cand & kept).sum()
            union = (cand | kept).sum().clamp_min(1)
            kept_area = kept.sum().clamp_min(1)
            iou = inter.float() / union.float()
            overlap_cand = inter.float() / cand_area.float()
            overlap_kept = inter.float() / kept_area.float()
            if (
                float(iou.item()) > iou_threshold
                or float(overlap_cand.item()) >= 0.95
                or float(overlap_kept.item()) >= 0.95
            ):
                suppress = True
                break
        if not suppress:
            keep.append(int(idx))

    if not keep:
        empty = det_scores_n[:0]
        return box_preds_n22[:0], mask_preds_nhw[:0], empty
    keep_tensor = torch.as_tensor(keep, device=det_scores_n.device, dtype=torch.long)
    return box_preds_n22[keep_tensor], mask_preds_nhw[keep_tensor], det_scores_n[keep_tensor]


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def update_map_stats(
    map_stats: Dict[str, Dict[str, object]],
    object_id: str,
    pred_masks: List[torch.Tensor],
    pred_scores: Optional[torch.Tensor],
    gt_masks: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> None:
    stats = map_stats.setdefault(
        object_id,
        {"scores": [], "tps": [], "fps": [], "num_gt": 0},
    )
    stats["num_gt"] += len(gt_masks)

    if pred_scores is None or not pred_masks:
        return

    scores_cpu = pred_scores.detach().float().cpu()
    order = torch.argsort(scores_cpu, descending=True)
    matched_gt = [False] * len(gt_masks)

    for idx in order.tolist():
        score_val = float(scores_cpu[idx].item())
        if not gt_masks:
            stats["scores"].append(score_val)
            stats["tps"].append(0)
            stats["fps"].append(1)
            continue

        pred = pred_masks[idx]
        best_iou = 0.0
        best_gt = -1
        for g_idx, gt in enumerate(gt_masks):
            if matched_gt[g_idx]:
                continue
            iou = compute_mask_iou(pred, gt)
            iou_val = float(iou.item())
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt = g_idx

        if best_iou >= iou_threshold and best_gt >= 0:
            matched_gt[best_gt] = True
            stats["scores"].append(score_val)
            stats["tps"].append(1)
            stats["fps"].append(0)
        else:
            stats["scores"].append(score_val)
            stats["tps"].append(0)
            stats["fps"].append(1)


def pad_exemplar_batch(
    exemplars_list: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not exemplars_list:
        raise ValueError("No exemplars provided for batching.")
    max_n = max(t.shape[1] for t in exemplars_list)
    feat_dim = exemplars_list[0].shape[2]
    batch = []
    padding_masks = []
    for ex in exemplars_list:
        ex = ex.to(device)
        n = ex.shape[1]
        if n < max_n:
            pad = torch.zeros((1, max_n - n, feat_dim), device=device, dtype=ex.dtype)
            ex = torch.cat([ex, pad], dim=1)
            mask = torch.zeros((max_n,), device=device, dtype=torch.bool)
            mask[n:] = True
        else:
            mask = torch.zeros((max_n,), device=device, dtype=torch.bool)
        batch.append(ex)
        padding_masks.append(mask)
    batch_bnc = torch.cat(batch, dim=0)
    padding_mask_bn = torch.stack(padding_masks, dim=0)
    return batch_bnc, padding_mask_bn


def save_debug_triptych(
    image_bgr: np.ndarray,
    box_preds_n22: torch.Tensor,
    detection_scores_n: torch.Tensor,
    mask_preds_nhw: torch.Tensor,
    object_id: str,
    output_path: str,
) -> None:
    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()
    mask_canvas = image_bgr.copy()

    def score_to_bgr(score: float) -> Tuple[int, int, int]:
        if score <= 0.2:
            return (255, 0, 0)
        if score >= 0.8:
            return (0, 0, 255)
        t = (score - 0.2) / 0.6
        b = int(round(255 * (1.0 - t)))
        r = int(round(255 * t))
        return (b, 0, r)

    if box_preds_n22.numel() > 0:
        scores_cpu = detection_scores_n.detach().float().cpu()
        best_idx = int(torch.argmax(scores_cpu).item())
        box = box_preds_n22.detach().float().cpu().numpy()[best_idx]
        score = float(scores_cpu.numpy()[best_idx])

        (x1n, y1n), (x2n, y2n) = box
        x1 = int(round(x1n * (w - 1)))
        y1 = int(round(y1n * (h - 1)))
        x2 = int(round(x2n * (w - 1)))
        y2 = int(round(y2n * (h - 1)))
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))

        if x2 > x1 and y2 > y1:
            box_color = score_to_bgr(score)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                overlay,
                f"{score:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
                cv2.LINE_AA,
            )

        topk = min(3, scores_cpu.numel())
        if topk > 0:
            top_idx = torch.topk(scores_cpu, k=topk).indices
            rank_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
            for rank, idx in sorted(enumerate(top_idx), key=lambda pair: pair[0], reverse=True):
                mask = mask_preds_nhw[int(idx)]
                mask_bin = (mask > 0).detach().float().cpu().numpy()
                if mask_bin.max() <= 0:
                    continue
                mask_u8 = (mask_bin * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
                color = rank_colors[rank]
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(mask_canvas, contours, -1, color, 2)
            for rank, idx in enumerate(top_idx):
                color = rank_colors[rank]
                score_val = float(scores_cpu[int(idx)].item())
                cv2.putText(
                    mask_canvas,
                    f"{score_val:.2f}",
                    (20, 30 + 26 * rank),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

    text_panel = np.zeros_like(image_bgr, dtype=np.uint8)
    cv2.putText(
        text_panel,
        f"Object: {object_id}",
        (20, max(30, h // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    triptych = np.concatenate([text_panel, overlay, mask_canvas], axis=1)
    cv2.imwrite(output_path, triptych)


def build_exemplar_tokens_for_object(
    detmodel,
    object_id: str,
    reference_dir: Path,
    ref_view_ids: List[str],
    max_side_length: int,
    use_square_sizing: bool,
    num_points_approx: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    lookup_id = object_id
    feats: List[torch.Tensor] = []
    for ref_id in ref_view_ids:
        ref_stub = f"{lookup_id}_stl_base_{ref_id}"
        ref_img_path = reference_dir / f"{ref_stub}.png"
        ref_mask_path = reference_dir / f"{ref_stub}_mask.png"
        if not (ref_img_path.is_file() and ref_mask_path.is_file()):
            continue
        try:
            ref_image = load_bgr(str(ref_img_path))
            ref_mask = load_mask_gray(str(ref_mask_path))
        except FileNotFoundError:
            continue
        if ref_mask.shape[:2] != ref_image.shape[:2]:
            ref_mask = resize_mask(ref_mask, ref_image.shape[:2])
        pts = sample_points_from_mask(ref_mask, num_points_approx=num_points_approx)
        if not pts:
            continue
        encimg_ref, _, _ = encode_detection_image_no_infer(
            detmodel, ref_image, max_side_length=max_side_length, use_square_sizing=use_square_sizing
        )
        exemplar_tokens = encode_exemplars_no_infer(
            detmodel,
            encimg_ref,
            text="visual",
            point_xy_norm_list=pts,
            include_coordinate_encodings=False,
        )
        feats.append(exemplar_tokens.detach().cpu())
    if not feats:
        return None
    exemplar_ref = torch.cat(feats, dim=1).to(device)
    return exemplar_ref


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAMv3 exemplar detection modules.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/kevin/sam3.pt",
        help="Path to SAMv3 checkpoint (.pt).",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/sata1/data/kevin/realworld_datasets/persam_v2",
        help="Comma-separated dataset roots.",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="/sata1/data/kevin/realworld_datasets/persam_real_coco/stl_renders_blender_2442_0120",
        help="Path to reference renders.",
    )
    # parser.add_argument(
    #     "--dataset_root",
    #     type=str,
    #     default="/sata1/data/kevin/lego_datasets/named_lego_structure_converted",
    #     help="Comma-separated dataset roots.",
    # )
    # parser.add_argument(
    #     "--reference_dir",
    #     type=str,
    #     default="/sata1/data/kevin/lego_datasets/lego_structure_refs",
    #     help="Path to reference renders.",
    # )
    # parser.add_argument(
    #     "--dataset_root",
    #     type=str,
    #     nargs="+",
    #     default=["/sata1/data/kevin/realworld_datasets/primesense_converted/000006", "/sata1/data/kevin/realworld_datasets/primesense_converted/000001", "/sata1/data/kevin/realworld_datasets/primesense_converted/000003"],
    #     help="Dataset roots (space-separated, and/or comma-separated).",
    # )
    # parser.add_argument(
    #     "--reference_dir",
    #     type=str,
    #     default="/sata1/data/kevin/realworld_datasets/primesense_converted/cad_renders",
    #     help="Path to reference renders.",
    # )
    #"0,3,6,9"
    #"0,1,2,3,4,5,6,7,8,9,10,11"
    parser.add_argument("--ref_view_ids", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11", help="Reference view ids to use.")
    parser.add_argument("--max_side_length", type=int, default=1008)
    parser.add_argument("--no_square", action="store_true", help="Disable square resizing in encoder.")
    parser.add_argument("--num_points_approx", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument(
        "--sub_sample",
        type=int,
        default=5 ,
        help="Evaluate every Nth image in the dataset (1 = use all images).",
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.5,
        help="IoU threshold for mask NMS (<=0 disables NMS).",
    )
    parser.add_argument("--det_filter", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="outputs_eval_exemplar")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument(
        "--multi_gt_only",
        default = False,
        help="Only evaluate samples with multiple GT instances for the target object.",
    )
    parser.add_argument("--itodd", default = True, help="Use ITODD BOP dataset layout.")
    parser.add_argument("--finetune_ckpt", type=str, default="/home/kevin/muggled_sam/finetune_exemplar/multi_object_best/finetune_epoch_017.pth", help="Optional finetuned detector checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.itodd:
        dataset_roots = ["/sata1/data/kevin/bop_datasets/ITODD"]
        reference_dir = Path("/sata1/data/kevin/bop_datasets/ITODD/models_rendered_blender")
    else:
        dataset_roots: List[str] = []
        for item in args.dataset_root:
            dataset_roots.extend([part.strip() for part in item.split(",") if part.strip()])
        if not dataset_roots:
            raise ValueError("No dataset roots provided.")
        reference_dir = Path(args.reference_dir).expanduser().resolve()

    ref_view_ids = parse_ref_view_ids(args.ref_view_ids)
    if not ref_view_ids:
        raise ValueError("No reference view ids resolved.")

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.dtype:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    _, base_model = make_sam_from_state_dict(args.model_path)
    base_model.to(device=device, dtype=dtype)
    detmodel = base_model.make_detector_model()
    detmodel.to(device=device, dtype=dtype)
    detmodel.eval()

    if args.finetune_ckpt:
        ckpt = torch.load(args.finetune_ckpt, map_location="cpu")
        detmodel.image_exemplar_fusion.load_state_dict(ckpt["image_exemplar_fusion"])
        detmodel.exemplar_detector.load_state_dict(ckpt["exemplar_detector"])
        detmodel.exemplar_segmentation.load_state_dict(ckpt["exemplar_segmentation"])
        print("Loaded finetuned detector weights from", args.finetune_ckpt)

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "step_outputs")
    os.makedirs(vis_dir, exist_ok=True)

    object_samples: Dict[str, List[Dict[str, str]]] = {}
    all_entries: List[Dict[str, str]] = []
    for root in dataset_roots:
        if args.itodd:
            cur_samples, cur_entries = collect_itodd_object_samples(root, sub_sample=args.sub_sample)
        else:
            cur_samples, cur_entries = collect_multi_object_samples(root, sub_sample=args.sub_sample)
        for obj_id, entries in cur_samples.items():
            object_samples.setdefault(obj_id, []).extend(entries)
        all_entries.extend(cur_entries)
    if not all_entries:
        raise RuntimeError("No dataset entries found.")

    unique_entries: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for entry in all_entries:
        key = (entry["frame_id"], entry["object_id"], entry["rgb_path"], entry["inst_path"])
        if key not in unique_entries:
            unique_entries[key] = entry
    all_entries = list(unique_entries.values())

    frame_entries: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for entry in all_entries:
        frame_key = (entry["frame_id"], entry["rgb_path"], entry["inst_path"])
        if frame_key not in frame_entries:
            frame_entries[frame_key] = {
                "frame_id": entry["frame_id"],
                "rgb_path": entry["rgb_path"],
                "inst_path": entry["inst_path"],
            }
            if args.itodd:
                frame_entries[frame_key]["targets"] = []
        if args.itodd:
            frame_entries[frame_key]["targets"].append(entry)
    frames = list(frame_entries.values())

    if args.shuffle:
        random.shuffle(frames)

    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)

    ref_cache: Dict[str, torch.Tensor] = {}
    ref_image_cache: Dict[str, List[np.ndarray]] = {}
    seg_cache: Dict[str, np.ndarray] = {}
    mapping_cache: Dict[str, Dict[str, List[Tuple[int, ...]]]] = {}
    batch_step = 0

    total_iou_sum = 0.0
    total_iou_count = 0
    total_correct_count = 0
    object_iou_sum: Dict[str, float] = defaultdict(float)
    object_iou_count: Dict[str, int] = defaultdict(int)
    pq_sum_iou = 0.0
    pq_tp = 0
    pq_fp = 0
    pq_fn = 0
    map_iou_threshold = 0.5
    map_stats: Dict[str, Dict[str, object]] = {}

    if object_samples:
        all_object_ids = sorted(object_samples.keys())
        with torch.no_grad():
            for obj_id in all_object_ids:
                exemplar_ref = build_exemplar_tokens_for_object(
                    detmodel=detmodel,
                    object_id=obj_id,
                    reference_dir=reference_dir,
                    ref_view_ids=ref_view_ids,
                    max_side_length=args.max_side_length,
                    use_square_sizing=not args.no_square,
                    num_points_approx=args.num_points_approx,
                    device=device,
                )
                if exemplar_ref is None:
                    continue
                ref_cache[obj_id] = exemplar_ref.detach().cpu()
    if not ref_cache:
        raise RuntimeError("No reference exemplars found for any object.")

    with torch.no_grad():
        for frame in frames:
            try:
                image_bgr = load_bgr(frame["rgb_path"])
            except FileNotFoundError:
                continue
            if not args.itodd:
                mapping_path = Path(frame["inst_path"]).with_name(
                    f"instance_segmentation_mapping_{frame['frame_id']}.json"
                )
                if not mapping_path.is_file():
                    continue
            img_t = detmodel.image_encoder.prepare_image(
                image_bgr,
                max_side_length=args.max_side_length,
                use_square_sizing=not args.no_square,
            )
            preencode_hw = img_t.shape[2:]

            prepared_all: List[Dict[str, object]] = []
            if args.itodd:
                targets = frame.get("targets", [])
                target_map = {target["object_id"]: target for target in targets}
                for obj_id, exemplar_ref in ref_cache.items():
                    gt_masks: List[np.ndarray] = []
                    target = target_map.get(obj_id)
                    if target is not None:
                        gt_masks = load_itodd_masks(
                            frame["inst_path"],
                            int(target["im_id"]),
                            int(target.get("inst_count", 0)),
                        )
                    if args.multi_gt_only and len(gt_masks) < 2:
                        continue
                    prepared_all.append(
                        {
                            "object_id": obj_id,
                            "frame_id": frame["frame_id"],
                            "image_bgr": image_bgr,
                            "gt_masks": gt_masks,
                            "has_gt": bool(gt_masks),
                            "gt_count": len(gt_masks),
                            "exemplar_ref": exemplar_ref,
                            "img_tensor": img_t,
                            "preencode_hw": preencode_hw,
                        }
                    )
            else:
                for obj_id, exemplar_ref in ref_cache.items():
                    try:
                        gt_masks = load_instance_masks_for_object(
                            frame["inst_path"],
                            str(mapping_path),
                            obj_id,
                            seg_cache=seg_cache,
                            mapping_cache=mapping_cache,
                        )
                    except FileNotFoundError:
                        gt_masks = []
                    if args.multi_gt_only and len(gt_masks) < 2:
                        continue
                    prepared_all.append(
                        {
                            "object_id": obj_id,
                            "frame_id": frame["frame_id"],
                            "image_bgr": image_bgr,
                            "gt_masks": gt_masks,
                            "has_gt": bool(gt_masks),
                            "gt_count": len(gt_masks),
                            "exemplar_ref": exemplar_ref,
                            "img_tensor": img_t,
                            "preencode_hw": preencode_hw,
                        }
                    )

            if not prepared_all:
                continue

            for start in range(0, len(prepared_all), args.batch_size):
                prepared = prepared_all[start : start + args.batch_size]
                if not prepared:
                    continue
                vis_target_idx = random.randrange(len(prepared))

                group_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
                for idx, entry in enumerate(prepared):
                    shape_key = entry["preencode_hw"]
                    group_map[shape_key].append(idx)

                batch_ious: List[float] = []
                batch_correct = 0
                for _, idxs in group_map.items():
                    img_batch = torch.cat([prepared[i]["img_tensor"] for i in idxs], dim=0)
                    encoded_img = detmodel.image_encoder(img_batch)
                    encoded_image_features_list = detmodel.image_projection.v3_projection(encoded_img)

                    exemplars_list = [prepared[i]["exemplar_ref"] for i in idxs]
                    exemplar_batch, padding_mask = pad_exemplar_batch(exemplars_list, device=device)

                    mask_preds, box_preds, det_scores, _ = generate_detections_train(
                        detmodel,
                        encoded_image_features_list,
                        exemplar_batch,
                        detection_filter_threshold=args.det_filter,
                        exemplar_padding_mask_bn=padding_mask,
                    )
                    if mask_preds.shape[1] == 0:
                        for data_idx in idxs:
                            obj_id = prepared[data_idx]["object_id"]
                            preencode_hw = prepared[data_idx]["preencode_hw"]
                            gt_down_list: List[torch.Tensor] = []
                            if prepared[data_idx]["has_gt"]:
                                for gt_mask in prepared[data_idx]["gt_masks"]:
                                    gt_preenc = resize_mask(gt_mask, preencode_hw)
                                    gt_tensor = torch.from_numpy(gt_preenc).to(device).unsqueeze(0).unsqueeze(0)
                                    gt_down = F.interpolate(
                                        gt_tensor, size=mask_preds.shape[-2:], mode="nearest"
                                    ).squeeze(0).squeeze(0)
                                    gt_down_list.append(gt_down > 0.5)
                            update_map_stats(
                                map_stats,
                                obj_id,
                                [],
                                None,
                                gt_down_list,
                                iou_threshold=map_iou_threshold,
                            )
                            sum_iou, tp, fp, fn = compute_pq_stats([], gt_down_list)
                            pq_sum_iou += sum_iou
                            pq_tp += tp
                            pq_fp += fp
                            pq_fn += fn
                        continue

                    for local_idx, data_idx in enumerate(idxs):
                        preencode_hw = prepared[data_idx]["preencode_hw"]
                        has_gt = prepared[data_idx]["has_gt"]

                        scores = det_scores[local_idx]
                        if scores.numel() == 0:
                            obj_id = prepared[data_idx]["object_id"]
                            gt_down_list: List[torch.Tensor] = []
                            if has_gt:
                                for gt_mask in prepared[data_idx]["gt_masks"]:
                                    gt_preenc = resize_mask(gt_mask, preencode_hw)
                                    gt_tensor = torch.from_numpy(gt_preenc).to(device).unsqueeze(0).unsqueeze(0)
                                    gt_down = F.interpolate(
                                        gt_tensor, size=mask_preds.shape[-2:], mode="nearest"
                                    ).squeeze(0).squeeze(0)
                                    gt_down_list.append(gt_down > 0.5)
                            update_map_stats(
                                map_stats,
                                obj_id,
                                [],
                                None,
                                gt_down_list,
                                iou_threshold=map_iou_threshold,
                            )
                            sum_iou, tp, fp, fn = compute_pq_stats([], gt_down_list)
                            pq_sum_iou += sum_iou
                            pq_tp += tp
                            pq_fp += fp
                            pq_fn += fn
                            continue

                        boxes_nms, masks_nms, scores_nms = apply_mask_nms(
                            box_preds[local_idx],
                            mask_preds[local_idx],
                            scores,
                            iou_threshold=args.nms_iou,
                        )
                        if scores_nms.numel() == 0:
                            obj_id = prepared[data_idx]["object_id"]
                            gt_down_list = []
                            if has_gt:
                                for gt_mask in prepared[data_idx]["gt_masks"]:
                                    gt_preenc = resize_mask(gt_mask, preencode_hw)
                                    gt_tensor = torch.from_numpy(gt_preenc).to(device).unsqueeze(0).unsqueeze(0)
                                    gt_down = F.interpolate(
                                        gt_tensor, size=mask_preds.shape[-2:], mode="nearest"
                                    ).squeeze(0).squeeze(0)
                                    gt_down_list.append(gt_down > 0.5)
                            update_map_stats(
                                map_stats,
                                obj_id,
                                [],
                                None,
                                gt_down_list,
                                iou_threshold=map_iou_threshold,
                            )
                            sum_iou, tp, fp, fn = compute_pq_stats([], gt_down_list)
                            pq_sum_iou += sum_iou
                            pq_tp += tp
                            pq_fp += fp
                            pq_fn += fn
                            continue

                        gt_down_list: List[torch.Tensor] = []
                        if has_gt:
                            for gt_mask in prepared[data_idx]["gt_masks"]:
                                gt_preenc = resize_mask(gt_mask, preencode_hw)
                                gt_tensor = torch.from_numpy(gt_preenc).to(device).unsqueeze(0).unsqueeze(0)
                                gt_down = F.interpolate(
                                    gt_tensor, size=mask_preds.shape[-2:], mode="nearest"
                                ).squeeze(0).squeeze(0)
                                gt_down_list.append(gt_down > 0.5)

                        pred_masks_list = [(masks_nms[k] > 0) for k in range(masks_nms.shape[0])]
                        update_map_stats(
                            map_stats,
                            prepared[data_idx]["object_id"],
                            pred_masks_list,
                            scores_nms,
                            gt_down_list,
                            iou_threshold=map_iou_threshold,
                        )
                        sum_iou, tp, fp, fn = compute_pq_stats(
                            pred_masks_list,
                            gt_down_list,
                            pred_scores=scores_nms,
                        )
                        pq_sum_iou += sum_iou
                        pq_tp += tp
                        pq_fp += fp
                        pq_fn += fn

                        best_iou = 0.0
                        if has_gt:
                            for gt_down in gt_down_list:
                                iou = compute_mask_iou(masks_nms[0], gt_down)
                                best_iou = max(best_iou, float(iou.item()))
                            batch_ious.append(best_iou)
                            total_iou_sum += best_iou
                            total_iou_count += 1
                            if best_iou > 0.5:
                                batch_correct += 1
                                total_correct_count += 1
                            obj_id = prepared[data_idx]["object_id"]
                            object_iou_sum[obj_id] += best_iou
                            object_iou_count[obj_id] += 1

                        if data_idx == vis_target_idx:
                            out_path = os.path.join(vis_dir, f"step_{batch_step:06d}.png")
                            save_debug_collage(
                                prepared[data_idx]["image_bgr"],
                                boxes_nms,
                                scores_nms,
                                masks_nms,
                                out_path,
                                object_id=prepared[data_idx]["object_id"],
                                reference_dir=reference_dir,
                                ref_view_ids=ref_view_ids,
                                gt_masks=prepared[data_idx]["gt_masks"],
                                ref_image_cache=ref_image_cache,
                            )

                if batch_ious:
                    avg_iou = sum(batch_ious) / max(1, len(batch_ious))
                    correct_rate = batch_correct / max(1, len(batch_ious))
                    print(
                        f"step={batch_step} avg_iou={avg_iou:.4f} "
                        f"correct_rate={correct_rate:.3f} samples={len(batch_ious)}"
                    )
                batch_step += 1

                if args.max_batches > 0 and batch_step >= args.max_batches:
                    break
            if args.max_batches > 0 and batch_step >= args.max_batches:
                break

    if total_iou_count > 0:
        overall_avg = total_iou_sum / total_iou_count
        overall_correct = total_correct_count / total_iou_count
        print(
            f"overall_avg_iou={overall_avg:.4f} "
            f"correct_rate={overall_correct:.3f} samples={total_iou_count}"
        )
    if object_iou_count:
        print("per_object_iou:")
        for obj_id in sorted(object_iou_count.keys()):
            count = object_iou_count[obj_id]
            avg_iou = object_iou_sum[obj_id] / max(1, count)
            print(f"  {obj_id}: avg_iou={avg_iou:.4f} samples={count}")
    denom = pq_tp + 0.5 * pq_fp + 0.5 * pq_fn
    if denom > 0:
        pq = pq_sum_iou / denom
    else:
        pq = 0.0
    print(f"PQ={pq:.4f} tp={pq_tp} fp={pq_fp} fn={pq_fn}")

    ap_values: List[float] = []
    for obj_id in sorted(map_stats.keys()):
        stats = map_stats[obj_id]
        num_gt = int(stats["num_gt"])
        scores = np.asarray(stats["scores"], dtype=np.float32)
        tps = np.asarray(stats["tps"], dtype=np.float32)
        fps = np.asarray(stats["fps"], dtype=np.float32)
        if num_gt == 0:
            ap = 0.0
        elif scores.size == 0:
            ap = 0.0
        else:
            order = np.argsort(-scores)
            tps = tps[order]
            fps = fps[order]
            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(fps)
            recalls = cum_tp / max(1, num_gt)
            precisions = cum_tp / np.maximum(1.0, cum_tp + cum_fp)
            ap = compute_ap(recalls, precisions)
        if num_gt > 0:
            ap_values.append(ap)

    map_value = float(np.mean(ap_values)) if ap_values else 0.0
    print(f"mAP@{map_iou_threshold:.2f}={map_value:.4f} objects={len(ap_values)}")


if __name__ == "__main__":
    main()
