#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from muggled_sam.make_sam import make_sam_from_state_dict


IGNORED_LABELS = {"BACKGROUND", "UNLABELLED"}


@dataclass(frozen=True)
class EvalConfig:
    dataset_root: object
    reference_dir: str
    ref_view_ids: str
    max_side_length: int = 1008
    use_square_sizing: bool = True
    num_points_approx: int = 25
    batch_size: int = 8
    det_filter: float = 0.0
    shuffle: bool = False
    max_batches: int = 0


EVAL_CONFIG = EvalConfig(
    dataset_root="/sata1/data/kevin/realworld_datasets/persam_v2",
    reference_dir="/sata1/data/kevin/realworld_datasets/persam_real_coco/stl_renders_blender_2442_0120",
    ref_view_ids="0,3,6,9",
    max_side_length=1008,
    use_square_sizing=True,
    num_points_approx=25,
    batch_size=8,
    det_filter=0.0,
    shuffle=False,
    max_batches=0,
)


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
) -> Tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
    dataset_path = Path(dataset_root).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_root)
    pattern = re.compile(r"instance_segmentation_(\d+)\.png$")
    object_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    all_entries: List[Dict[str, str]] = []
    for inst_path in sorted(dataset_path.glob("instance_segmentation_*.png")):
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


def _coerce_multiplier(value: object) -> float:
    try:
        multiplier = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid dataset multiplier: {value}") from exc
    if multiplier < 0:
        raise ValueError(f"Dataset multiplier must be >= 0 (got {multiplier}).")
    return multiplier


def normalize_dataset_roots(dataset_root: object) -> List[Tuple[str, float, bool]]:
    if dataset_root is None:
        return []
    items: List[object]
    if isinstance(dataset_root, (list, tuple)):
        items = list(dataset_root)
    else:
        items = [dataset_root]
    roots: List[Tuple[str, float, bool]] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) in (2, 3):
            path = str(item[0]).strip()
            if path:
                multiplier = _coerce_multiplier(item[1])
                use_filter = bool(item[2]) if len(item) == 3 else False
                roots.append((path, multiplier, use_filter))
            continue
        for part in str(item).split(","):
            part = part.strip()
            if part:
                roots.append((part, 1.0, False))
    return roots


def load_dataset_filter_set(csv_path: Path) -> Tuple[Optional[set], Optional[set]]:
    if not csv_path.is_file():
        return None, None
    try:
        with open(csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return None, None
            frame_ids: set = set()
            tuple_keys: set = set()
            for row in reader:
                frame_id = (row.get("frame_id") or "").strip()
                rgb_path = (row.get("rgb_path") or "").strip()
                inst_path = (row.get("inst_path") or "").strip()
                if frame_id:
                    frame_ids.add(frame_id)
                if frame_id and rgb_path and inst_path:
                    tuple_keys.add((frame_id, rgb_path, inst_path))
            return (tuple_keys if tuple_keys else None), (frame_ids if frame_ids else None)
    except FileNotFoundError:
        return None, None


def _slugify(path: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", path.strip().strip("/"))
    return cleaned or "dataset"


def resolve_dataset_filter_path(root: str) -> Optional[Path]:
    root_path = Path(root).expanduser().resolve()
    slug = _slugify(str(root_path))
    candidate = Path.cwd() / f"dataset_filter_{slug}.csv"
    if candidate.is_file():
        return candidate
    return None


def apply_dataset_filter_entries(
    entries: Sequence[Dict[str, str]],
    filter_path: Optional[Path],
) -> List[Dict[str, str]]:
    if not filter_path:
        return list(entries)
    tuple_keys, frame_ids = load_dataset_filter_set(filter_path)
    if tuple_keys is None and frame_ids is None:
        return list(entries)
    filtered: List[Dict[str, str]] = []
    for entry in entries:
        key = (entry["frame_id"], entry["rgb_path"], entry["inst_path"])
        if tuple_keys is not None:
            if key in tuple_keys:
                filtered.append(entry)
            continue
        if frame_ids is not None and entry["frame_id"] in frame_ids:
            filtered.append(entry)
    return filtered


def unique_object_entries(entries: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    unique_entries: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for entry in entries:
        key = (entry["frame_id"], entry["object_id"], entry["rgb_path"], entry["inst_path"])
        if key not in unique_entries:
            unique_entries[key] = entry
    return list(unique_entries.values())


def apply_dataset_multiplier(entries: Sequence[Dict[str, str]], multiplier: float) -> List[Dict[str, str]]:
    if multiplier <= 0 or not entries:
        return []
    target = int(round(len(entries) * multiplier))
    if target <= 0:
        return []
    if target == len(entries):
        return list(entries)
    if target < len(entries):
        return random.sample(list(entries), target)
    base, extra = divmod(target, len(entries))
    scaled = list(entries) * base
    if extra:
        scaled.extend(random.sample(list(entries), extra))
    return scaled


def apply_random_color_distortion(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = random.uniform(-90.0, 90.0)
    sat_scale = random.uniform(0.8, 1.2)
    val_scale = random.uniform(0.8, 1.2)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def data_augmentation(
    image_bgr: np.ndarray,
    gt_mask: np.ndarray,
    min_crop_scale: float = 0.6,
    max_crop_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    orig_h, orig_w = image_bgr.shape[:2]
    mask_bin = gt_mask > 0.5
    if mask_bin.any():
        ys, xs = np.where(mask_bin)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        bbox_h = y2 - y1 + 1
        bbox_w = x2 - x1 + 1
        for _ in range(50):
            scale = random.uniform(min_crop_scale, max_crop_scale)
            crop_h = max(1, int(round(orig_h * scale)))
            crop_w = max(1, int(round(orig_w * scale)))
            if crop_h < bbox_h or crop_w < bbox_w:
                continue
            max_y0 = orig_h - crop_h
            max_x0 = orig_w - crop_w
            y0_min = max(0, y2 - crop_h + 1)
            y0_max = min(y1, max_y0)
            x0_min = max(0, x2 - crop_w + 1)
            x0_max = min(x1, max_x0)
            if y0_min > y0_max or x0_min > x0_max:
                continue
            y0 = random.randint(y0_min, y0_max)
            x0 = random.randint(x0_min, x0_max)
            image_bgr = image_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
            gt_mask = gt_mask[y0 : y0 + crop_h, x0 : x0 + crop_w]
            image_bgr = cv2.resize(image_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            gt_mask = cv2.resize(gt_mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            break
    image_bgr = apply_random_color_distortion(image_bgr)
    return image_bgr, gt_mask


def data_augmentation_multi(
    image_bgr: np.ndarray,
    gt_masks: Sequence[np.ndarray],
    min_crop_scale: float = 0.6,
    max_crop_scale: float = 1.0,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    if not gt_masks:
        aug_image = apply_random_color_distortion(image_bgr)
        return aug_image, []

    orig_h, orig_w = image_bgr.shape[:2]
    union_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
    for gt_mask in gt_masks:
        union_mask = np.maximum(union_mask, gt_mask.astype(np.float32))

    mask_bin = union_mask > 0.5
    out_masks = [gt_mask.astype(np.float32) for gt_mask in gt_masks]
    if mask_bin.any():
        ys, xs = np.where(mask_bin)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        bbox_h = y2 - y1 + 1
        bbox_w = x2 - x1 + 1
        for _ in range(50):
            scale = random.uniform(min_crop_scale, max_crop_scale)
            crop_h = max(1, int(round(orig_h * scale)))
            crop_w = max(1, int(round(orig_w * scale)))
            if crop_h < bbox_h or crop_w < bbox_w:
                continue
            max_y0 = orig_h - crop_h
            max_x0 = orig_w - crop_w
            y0_min = max(0, y2 - crop_h + 1)
            y0_max = min(y1, max_y0)
            x0_min = max(0, x2 - crop_w + 1)
            x0_max = min(x1, max_x0)
            if y0_min > y0_max or x0_min > x0_max:
                continue
            y0 = random.randint(y0_min, y0_max)
            x0 = random.randint(x0_min, x0_max)
            image_bgr = image_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
            out_masks = [gt_mask[y0 : y0 + crop_h, x0 : x0 + crop_w] for gt_mask in out_masks]
            image_bgr = cv2.resize(image_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            out_masks = [
                cv2.resize(gt_mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                for gt_mask in out_masks
            ]
            break

    image_bgr = apply_random_color_distortion(image_bgr)
    return image_bgr, out_masks


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


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    module.train()
    for param in module.parameters():
        param.requires_grad = True


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
    best_idx: int,
    pos_weight: float,
    neg_weight: float,
) -> torch.Tensor:
    scores_logits = scores_logits.float()
    target = torch.zeros_like(scores_logits)
    if scores_logits.numel() > 0:
        target[best_idx] = 1.0
    loss_per = F.binary_cross_entropy_with_logits(scores_logits, target, reduction="none")
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
    return pos_weight * loss_pos + neg_weight * loss_neg


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

    scores = torch.sigmoid(scores_logits.float())
    weights = torch.softmax(scores**power, dim=0)
    return (weights * loss_per_mask).sum()


def compute_mask_ious_per_gt(
    logits_mhw: torch.Tensor,
    gt_targets_hw: Sequence[torch.Tensor],
) -> torch.Tensor:
    pred_bin = logits_mhw > 0
    iou_rows: List[torch.Tensor] = []
    for gt_target_hw in gt_targets_hw:
        gt_bin = gt_target_hw > 0.5
        intersection = (pred_bin & gt_bin.unsqueeze(0)).sum(dim=(1, 2))
        union = (pred_bin | gt_bin.unsqueeze(0)).sum(dim=(1, 2)).clamp_min(1)
        iou_rows.append(intersection.float() / union.float())
    return torch.stack(iou_rows, dim=0)


def compute_score_weighted_mask_loss_best_gt_per_pred(
    logits_mhw: torch.Tensor,
    gt_targets_hw: Sequence[torch.Tensor],
    scores_logits: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
    power: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not gt_targets_hw:
        zero = torch.zeros((), device=logits_mhw.device)
        return zero, torch.empty((0, logits_mhw.shape[0]), device=logits_mhw.device)

    iou_gm = compute_mask_ious_per_gt(logits_mhw, gt_targets_hw)
    best_gt_idx_per_pred = torch.argmax(iou_gm, dim=0)

    matched_loss_per_pred: List[torch.Tensor] = []
    for pred_idx, gt_idx in enumerate(best_gt_idx_per_pred.tolist()):
        gt_target_hw = gt_targets_hw[gt_idx]
        loss_per_mask = compute_mask_loss_best(
            logits_mhw[pred_idx : pred_idx + 1],
            gt_target_hw,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )[0]
        matched_loss_per_pred.append(loss_per_mask)

    loss_per_pred = torch.stack(matched_loss_per_pred)
    scores = torch.sigmoid(scores_logits.float())
    weights = torch.softmax(scores**power, dim=0)
    return (weights * loss_per_pred).sum(), iou_gm


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


def score_to_bgr(score: float) -> Tuple[int, int, int]:
    if score <= 0.2:
        return (255, 0, 0)
    if score >= 0.8:
        return (0, 0, 255)
    t = (score - 0.2) / 0.6
    b = int(round(255 * (1.0 - t)))
    r = int(round(255 * t))
    return (b, 0, r)


def draw_debug_boxes(
    image_bgr: np.ndarray,
    box_preds_n22: torch.Tensor,
    detection_scores_n: torch.Tensor,
) -> np.ndarray:
    out = image_bgr.copy()
    if box_preds_n22.numel() == 0:
        return out
    if box_preds_n22.ndim == 2 and box_preds_n22.shape == (2, 2):
        box_preds_n22 = box_preds_n22.unsqueeze(0)
    elif box_preds_n22.ndim == 2 and box_preds_n22.shape[1] == 4:
        box_preds_n22 = box_preds_n22.reshape(-1, 2, 2)
    elif box_preds_n22.ndim == 1 and box_preds_n22.numel() == 4:
        box_preds_n22 = box_preds_n22.reshape(1, 2, 2)
    if box_preds_n22.ndim != 3 or box_preds_n22.shape[-2:] != (2, 2):
        return out
    num_boxes = box_preds_n22.shape[0]
    if num_boxes == 0:
        return out
    scores_cpu = detection_scores_n.detach().float().cpu()
    topk = min(5, num_boxes)
    top_idx = torch.topk(scores_cpu, k=topk).indices
    top_idx_list = [int(top_idx.item())] if topk == 1 else [int(idx) for idx in top_idx.tolist()]
    boxes_cpu = box_preds_n22.detach().float().cpu().numpy()[top_idx_list]
    scores_cpu = scores_cpu.numpy()[top_idx_list]
    h, w = image_bgr.shape[:2]
    if boxes_cpu.ndim == 2:
        boxes_cpu = boxes_cpu.reshape(1, 2, 2)

    for idx in range(min(topk, len(scores_cpu))):
        (x1n, y1n), (x2n, y2n) = boxes_cpu[idx]
        x1 = int(round(x1n * (w - 1)))
        y1 = int(round(y1n * (h - 1)))
        x2 = int(round(x2n * (w - 1)))
        y2 = int(round(y2n * (h - 1)))

        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        score = float(scores_cpu[idx])
        color = score_to_bgr(score)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{score:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def draw_gt_contours(
    image_bgr: np.ndarray,
    gt_masks: Optional[Union[np.ndarray, Sequence[np.ndarray]]],
    color: Tuple[int, int, int] = (0, 255,0),
    thickness: int = 2,
) -> np.ndarray:
    if gt_masks is None:
        return image_bgr
    if isinstance(gt_masks, np.ndarray):
        masks = [gt_masks]
    else:
        masks = [mask for mask in gt_masks if isinstance(mask, np.ndarray)]
    if not masks:
        return image_bgr
    h, w = image_bgr.shape[:2]
    for mask in masks:
        if mask is None:
            continue
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        if mask_u8.shape[:2] != (h, w):
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(image_bgr, contours, -1, color, thickness)
    return image_bgr


def load_reference_images(
    object_id: str,
    reference_dir: Path,
    ref_view_ids: List[str],
    max_views: int = 4,
) -> List[np.ndarray]:
    ref_images: List[np.ndarray] = []
    lookup_ids = [object_id, object_id.upper(), object_id.lower()]
    for ref_id in ref_view_ids:
        if len(ref_images) >= max_views:
            break
        ref_img_path = None
        for lookup_id in lookup_ids:
            ref_stub = f"{lookup_id}_stl_base_{ref_id}"
            candidate = reference_dir / f"{ref_stub}.png"
            if candidate.is_file():
                ref_img_path = candidate
                break
        if ref_img_path is None:
            continue
        try:
            ref_images.append(load_bgr(str(ref_img_path)))
        except FileNotFoundError:
            continue
    return ref_images


def build_reference_grid(ref_images: List[np.ndarray], target_height: int) -> np.ndarray:
    tile_h = max(1, target_height // 2)
    tile_w = tile_h
    tiles: List[np.ndarray] = []
    for idx in range(4):
        if idx < len(ref_images):
            tile = cv2.resize(ref_images[idx], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        else:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tiles.append(tile)

    row_top = np.concatenate(tiles[:2], axis=1)
    row_bottom = np.concatenate(tiles[2:], axis=1)
    grid = np.concatenate([row_top, row_bottom], axis=0)

    if grid.shape[0] < target_height:
        pad_h = target_height - grid.shape[0]
        pad = np.zeros((pad_h, grid.shape[1], 3), dtype=np.uint8)
        grid = np.concatenate([grid, pad], axis=0)
    elif grid.shape[0] > target_height:
        grid = grid[:target_height, :, :]
    return grid


def build_topk_mask_overlay(
    image_bgr: np.ndarray,
    mask_preds_nhw: torch.Tensor,
    detection_scores_n: torch.Tensor,
    topk: int = 3,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask_canvas = image_bgr.copy()
    if detection_scores_n.numel() == 0 or mask_preds_nhw.numel() == 0:
        return mask_canvas

    scores_cpu = detection_scores_n.detach().float().cpu()
    topk = min(topk, scores_cpu.numel())
    if topk <= 0:
        return mask_canvas

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
    return mask_canvas


def save_debug_collage(
    image_bgr: np.ndarray,
    box_preds_n22: torch.Tensor,
    detection_scores_n: torch.Tensor,
    mask_preds_nhw: torch.Tensor,
    output_path: str,
    object_id: str,
    reference_dir: Path,
    ref_view_ids: List[str],
    gt_masks: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    ref_image_cache: Optional[Dict[str, List[np.ndarray]]] = None,
) -> None:
    target_overlay = draw_debug_boxes(image_bgr, box_preds_n22, detection_scores_n)
    target_overlay = draw_gt_contours(target_overlay, gt_masks, color=(0, 255, 0), thickness=2)
    if ref_image_cache is not None and object_id in ref_image_cache:
        ref_images = ref_image_cache[object_id]
    else:
        ref_images = load_reference_images(object_id, reference_dir, ref_view_ids)
        if ref_image_cache is not None:
            ref_image_cache[object_id] = ref_images

    ref_grid = build_reference_grid(ref_images, target_overlay.shape[0])
    if ref_grid.shape[0] != target_overlay.shape[0]:
        target_h = target_overlay.shape[0]
        if ref_grid.shape[0] < target_h:
            pad_h = target_h - ref_grid.shape[0]
            pad = np.zeros((pad_h, ref_grid.shape[1], 3), dtype=np.uint8)
            ref_grid = np.concatenate([ref_grid, pad], axis=0)
        else:
            ref_grid = ref_grid[:target_h, :, :]

    mask_overlay = build_topk_mask_overlay(image_bgr, mask_preds_nhw, detection_scores_n, topk=3)
    combined = np.concatenate([ref_grid, target_overlay, mask_overlay], axis=1)
    cv2.imwrite(output_path, combined)


def build_exemplar_tokens_for_object(
    detmodel,
    object_id: str,
    reference_dir: Path,
    ref_view_ids: List[str],
    max_side_length: int,
    use_square_sizing: bool,
    num_points_approx: int,
    device: torch.device,
    upper_object_id: bool = False,
) -> Optional[torch.Tensor]:
    lookup_id = object_id.upper() if upper_object_id else object_id
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
    parser = argparse.ArgumentParser(description="Fine-tune SAMv3 exemplar detection modules.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/zhenrant/rendering_prompted_muggled_sam/sam3.pt",
        help="Path to SAMv3 checkpoint (.pt).",
    )
    # parser.add_argument(
    #     "--dataset_root",
    #     type=str,
    #     nargs="+",
    #     default=["/home/kevin/rendering/perseve/output/multi_object_3_per_frame_1125_table_texture", "/home/kevin/rendering/perseve/output/multi_object_3_per_frame_1125_table_texture_b", "/home/kevin/rendering/perseve/output/multi_object_3_per_frame_1125_table_texture_c"],
    #     help="Dataset roots (space or comma separated).",
    # )
    # parser.add_argument(
    #     "--reference_dir",
    #     type=str,
    #     default="/sata1/data/kevin/multi_object_1125/stl_renders_blender_2442_0120",
    #     help="Path to reference renders.",
    # )
    # parser.add_argument(
    #     "--dataset_root",
    #     type=str,
    #     default="/sata1/data/kevin/realworld_datasets/persam_v2",
    #     help="Comma-separated dataset roots.",
    # )
    # parser.add_argument(
    #     "--reference_dir",
    #     type=str,
    #     default="/sata1/data/kevin/realworld_datasets/persam_real_coco/stl_renders_blender_2442_0120",
    #     help="Path to reference renders.",
    # )
    # parser.add_argument(
    #     "--dataset_root",
    #     type=str,
    #     default=["/sata1/data/kevin/realworld_datasets/primesense_converted/000003", "/sata1/data/kevin/realworld_datasets/primesense_converted/000005", "/sata1/data/kevin/realworld_datasets/primesense_converted/000007", "/sata1/data/kevin/realworld_datasets/primesense_converted/000009"],
    #     help="Comma-separated dataset roots.",
    # )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=[
            ("/sata1/data/kevin/v2_dataset/v2_multi_gt/v2_sdg_output", 1, True),
            ("/sata1/data/kevin/v2_imgs/train_1", 1, True),
            ("/sata1/data/kevin/v2_imgs/train_2", 1, True),
            ("/sata1/data/kevin/v2_dataset/v2_multi_gt_merged_345", 1,True)
        ],
        help="Comma-separated dataset roots. Can also set a list of (path, multiplier, use_filter) in code defaults.",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="/sata1/data/kevin/v2_3d/v2_usds/v2_stl_render_0212",
        help="Path to reference renders.",
    )
    parser.add_argument("--ref_view_ids", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12", help="Reference view ids to use.")
    parser.add_argument("--max_side_length", type=int, default=1008)
    parser.add_argument("--no_square", action="store_true", help="Disable square resizing in encoder.")
    parser.add_argument("--num_points_approx", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bce_weight", type=float, default=2.0)
    parser.add_argument("--dice_weight", type=float, default=2.0)
    parser.add_argument("--score_weight", type=float, default=0.3)
    parser.add_argument("--no_object_weight", type=float, default=0.3)
    parser.add_argument("--det_filter", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--save_debug_every",
        type=int,
        default=50,
        help="Save debug collage every N batches (0 disables).",
    )
    parser.add_argument("--output_dir", type=str, default="finetune_exemplar")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default="")
    parser.add_argument(
        "--resume_path",
        type=str,
        default="",
        help="Path to a finetune checkpoint (.pth) to resume from.",
    )
    parser.add_argument(
        "--no_resume_optimizer",
        action="store_true",
        help="Do not load optimizer state when resuming.",
    )
    parser.add_argument(
        "--resume_in_place",
        default= False,
        help="When resuming, save checkpoints into the original run directory.",
    )
    return parser.parse_args()


def create_run_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{stamp}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    for idx in range(1, 1000):
        candidate = f"{run_dir}_{idx:03d}"
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
    raise RuntimeError(f"Unable to create run directory under {base_dir}")


def load_finetune_checkpoint(
    checkpoint_path: Path,
    detmodel,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    load_optimizer: bool = True,
) -> Dict[str, object]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    required = {
        "image_exemplar_fusion": detmodel.image_exemplar_fusion,
        "exemplar_detector": detmodel.exemplar_detector,
        "exemplar_segmentation": detmodel.exemplar_segmentation,
    }
    for key, module in required.items():
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing '{key}' state.")
        module.load_state_dict(checkpoint[key])
    if optimizer is not None and load_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def _capture_training_state(module: torch.nn.Module) -> List[Tuple[torch.nn.Module, bool]]:
    return [(submodule, submodule.training) for submodule in module.modules()]


def _restore_training_state(state: List[Tuple[torch.nn.Module, bool]]) -> None:
    for submodule, was_training in state:
        submodule.train(was_training)


def run_exemplar_eval(
    detmodel,
    config: EvalConfig,
    device: torch.device,
) -> Tuple[float, int, float]:
    dataset_roots = normalize_dataset_roots(config.dataset_root)
    if not dataset_roots:
        raise ValueError("No eval dataset roots provided.")

    ref_view_ids = parse_ref_view_ids(config.ref_view_ids)
    if not ref_view_ids:
        raise ValueError("No eval reference view ids resolved.")

    reference_dir = Path(config.reference_dir).expanduser().resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)

    all_entries: List[Dict[str, str]] = []
    for root, multiplier, use_filter in dataset_roots:
        _, cur_entries = collect_multi_object_samples(root)
        if use_filter:
            filter_path = resolve_dataset_filter_path(root)
            cur_entries = apply_dataset_filter_entries(cur_entries, filter_path)
        cur_entries = unique_object_entries(cur_entries)
        cur_entries = apply_dataset_multiplier(cur_entries, multiplier)
        all_entries.extend(cur_entries)
    if not all_entries:
        raise RuntimeError("No eval dataset entries found.")

    if config.shuffle:
        random.shuffle(all_entries)

    ref_cache: Dict[str, torch.Tensor] = {}
    seg_cache: Dict[str, np.ndarray] = {}
    mapping_cache: Dict[str, Dict[str, List[Tuple[int, ...]]]] = {}
    total_iou_sum = 0.0
    total_iou_count = 0
    total_correct = 0
    batch_step = 0

    training_state = _capture_training_state(detmodel)
    detmodel.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(all_entries), config.batch_size):
                subset = all_entries[start : start + config.batch_size]
                prepared: List[Dict[str, object]] = []
                for entry in subset:
                    obj_id = entry["object_id"]
                    try:
                        image_bgr = load_bgr(entry["rgb_path"])
                    except FileNotFoundError:
                        continue
                    try:
                        mapping_path = Path(entry["inst_path"]).with_name(
                            f"instance_segmentation_mapping_{entry['frame_id']}.json"
                        )
                        gt_masks = load_instance_masks_for_object(
                            entry["inst_path"],
                            str(mapping_path),
                            obj_id,
                            seg_cache=seg_cache,
                            mapping_cache=mapping_cache,
                        )
                    except FileNotFoundError:
                        continue
                    if not gt_masks:
                        continue

                    if obj_id not in ref_cache:
                        exemplar_ref = build_exemplar_tokens_for_object(
                            detmodel=detmodel,
                            object_id=obj_id,
                            reference_dir=reference_dir,
                            ref_view_ids=ref_view_ids,
                            max_side_length=config.max_side_length,
                            use_square_sizing=config.use_square_sizing,
                            num_points_approx=config.num_points_approx,
                            device=device,
                            upper_object_id=False,
                        )
                        if exemplar_ref is None:
                            continue
                        ref_cache[obj_id] = exemplar_ref.detach().cpu()

                    exemplar_ref = ref_cache[obj_id]
                    prepared.append(
                        {
                            "image_bgr": image_bgr,
                            "gt_masks": gt_masks,
                            "exemplar_ref": exemplar_ref,
                        }
                    )

                if not prepared:
                    continue

                group_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
                for idx, entry in enumerate(prepared):
                    img_t = detmodel.image_encoder.prepare_image(
                        entry["image_bgr"],
                        max_side_length=config.max_side_length,
                        use_square_sizing=config.use_square_sizing,
                    )
                    entry["img_tensor"] = img_t
                    entry["preencode_hw"] = img_t.shape[2:]
                    shape_key = (img_t.shape[2], img_t.shape[3])
                    group_map[shape_key].append(idx)

                batch_ious: List[float] = []
                batch_correct = 0
                for _, idxs in group_map.items():
                    img_batch = torch.cat([prepared[i]["img_tensor"] for i in idxs], dim=0)
                    encoded_img = detmodel.image_encoder(img_batch)
                    encoded_image_features_list = detmodel.image_projection.v3_projection(encoded_img)

                    exemplars_list = [prepared[i]["exemplar_ref"] for i in idxs]
                    exemplar_batch, padding_mask = pad_exemplar_batch(exemplars_list, device=device)

                    mask_preds, _, det_scores, _ = generate_detections_train(
                        detmodel,
                        encoded_image_features_list,
                        exemplar_batch,
                        detection_filter_threshold=config.det_filter,
                        exemplar_padding_mask_bn=padding_mask,
                    )
                    if mask_preds.shape[1] == 0:
                        continue

                    for local_idx, data_idx in enumerate(idxs):
                        preencode_hw = prepared[data_idx]["preencode_hw"]
                        scores = det_scores[local_idx]
                        if scores.numel() == 0:
                            continue
                        top_idx = int(torch.argmax(scores).item())
                        pred_bin = mask_preds[local_idx, top_idx] > 0
                        best_iou = 0.0
                        for gt_mask in prepared[data_idx]["gt_masks"]:
                            gt_preenc = resize_mask(gt_mask, preencode_hw)
                            gt_tensor = torch.from_numpy(gt_preenc).to(device).unsqueeze(0).unsqueeze(0)
                            gt_down = F.interpolate(
                                gt_tensor, size=mask_preds.shape[-2:], mode="nearest"
                            ).squeeze(0)
                            gt_bin = gt_down[0] > 0.5
                            intersection = (pred_bin & gt_bin).sum()
                            union = (pred_bin | gt_bin).sum().clamp_min(1)
                            iou = intersection.float() / union.float()
                            best_iou = max(best_iou, float(iou.item()))
                        batch_ious.append(best_iou)
                        total_iou_sum += best_iou
                        total_iou_count += 1
                        if best_iou > 0.7:
                            batch_correct += 1
                            total_correct += 1

                if batch_ious:
                    avg_iou = sum(batch_ious) / max(1, len(batch_ious))
                    correct_rate = batch_correct / max(1, len(batch_ious))
                    print(
                        f"[eval] step={batch_step} avg_iou={avg_iou:.4f} "
                        f"correct_rate={correct_rate:.3f} samples={len(batch_ious)}"
                    )

                batch_step += 1
                if config.max_batches > 0 and batch_step >= config.max_batches:
                    break
    finally:
        _restore_training_state(training_state)

    if total_iou_count > 0:
        overall_avg = total_iou_sum / total_iou_count
        overall_correct = total_correct / total_iou_count
        return overall_avg, total_iou_count, overall_correct
    return 0.0, 0, 0.0


def main() -> None:
    print("Starting fine-tuning with SAMv3 exemplar detection modules.")
    args = parse_args()
    dataset_roots = normalize_dataset_roots(args.dataset_root)
    if not dataset_roots:
        raise ValueError("No dataset roots provided.")

    ref_view_ids = parse_ref_view_ids(args.ref_view_ids)
    if not ref_view_ids:
        raise ValueError("No reference view ids resolved.")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dtype:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    _, base_model = make_sam_from_state_dict(args.model_path)
    base_model.to(device=device, dtype=dtype)
    detmodel = base_model.make_detector_model()
    detmodel.to(device=device, dtype=dtype)

    unfreeze_module(detmodel.image_encoder)
    unfreeze_module(detmodel.image_projection)
    unfreeze_module(detmodel.text_encoder)
    unfreeze_module(detmodel.sampling_encoder)
    unfreeze_module(detmodel.image_exemplar_fusion)
    unfreeze_module(detmodel.exemplar_detector)
    unfreeze_module(detmodel.exemplar_segmentation)

    trainable_params: List[torch.nn.Parameter] = []
    for module in (detmodel.image_exemplar_fusion, detmodel.exemplar_detector, detmodel.exemplar_segmentation):
        trainable_params.extend([p for p in module.parameters() if p.requires_grad])
    total_params = sum(p.numel() for p in detmodel.parameters())
    trainable_params_count = sum(p.numel() for p in detmodel.parameters() if p.requires_grad)
    print(f"Parameter counts: total={total_params:,} trainable={trainable_params_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    global_step = 0
    batch_step = 0

    if args.resume_in_place and not args.resume_path:
        raise ValueError("--resume_in_place requires --resume_path.")

    if args.resume_path:
        resume_path = Path(args.resume_path).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        checkpoint = load_finetune_checkpoint(
            resume_path,
            detmodel,
            optimizer,
            device=device,
            load_optimizer=not args.no_resume_optimizer,
        )
        ckpt_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = max(1, ckpt_epoch + 1)
        global_step = int(checkpoint.get("global_step", 0))
        batch_step = int(checkpoint.get("batch_step", 0))
        print(
            f"Resuming from {resume_path} (epoch={ckpt_epoch}, global_step={global_step}, batch_step={batch_step}). "
            f"Starting at epoch {start_epoch}."
        )
    if args.resume_in_place and args.resume_path:
        run_dir = str(Path(args.resume_path).expanduser().resolve().parent)
    else:
        run_dir = create_run_dir(args.output_dir)
    args.output_dir = run_dir
    debug_dir = os.path.join(run_dir, "debug_boxes")
    os.makedirs(debug_dir, exist_ok=True)

    object_samples: Dict[str, List[Dict[str, str]]] = {}
    all_entries: List[Dict[str, str]] = []
    for root, multiplier, use_filter in dataset_roots:
        cur_samples, cur_entries = collect_multi_object_samples(root)
        if use_filter:
            filter_path = resolve_dataset_filter_path(root)
            cur_entries = apply_dataset_filter_entries(cur_entries, filter_path)
        for obj_id, entries in cur_samples.items():
            object_samples.setdefault(obj_id, []).extend(entries)
        cur_entries = apply_dataset_multiplier(cur_entries, multiplier)
        all_entries.extend(cur_entries)
    if not all_entries:
        raise RuntimeError("No dataset entries found.")

    reference_dir = Path(args.reference_dir).expanduser().resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)

    ref_cache: Dict[str, torch.Tensor] = {}
    ref_image_cache: Dict[str, List[np.ndarray]] = {}
    running_losses: deque[float] = deque(maxlen=5)

    if start_epoch > args.epochs:
        print(f"Resume epoch {start_epoch - 1} exceeds requested --epochs={args.epochs}. Nothing to do.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch != start_epoch:
            eval_avg, eval_count, eval_correct = run_exemplar_eval(detmodel, EVAL_CONFIG, device=device)
            if eval_count > 0:
                print(
                    f"[eval] epoch={epoch} avg_iou={eval_avg:.4f} "
                    f"correct_rate={eval_correct:.3f} samples={eval_count}"
                )
            else:
                print(f"[eval] epoch={epoch} no valid samples")

        random.shuffle(all_entries)
        epoch_loss = 0.0
        epoch_count = 0

        for start in range(0, len(all_entries), args.batch_size):
            subset = all_entries[start : start + args.batch_size]
            prepared: List[Dict[str, object]] = []
            for entry in subset:
                obj_id = entry["object_id"]
                mapping_path = Path(entry["inst_path"]).with_name(f"instance_segmentation_mapping_{entry['frame_id']}.json")
                try:
                    image_bgr = load_bgr(entry["rgb_path"])
                except FileNotFoundError:
                    continue
                try:
                    gt_masks = load_instance_masks_for_object(
                        entry["inst_path"],
                        str(mapping_path),
                        obj_id,
                    )
                except FileNotFoundError:
                    continue
                if not gt_masks:
                    continue
                gt_pixels = max(int(gt_mask.sum()) for gt_mask in gt_masks)
                if gt_pixels < 100:
                    print(
                        f"Skipping object {obj_id} (frame={entry['frame_id']}) "
                        f"due to small gt_mask size ({gt_pixels} pixels)."
                    )
                    continue

                image_bgr, gt_masks = data_augmentation_multi(image_bgr, gt_masks)

                if obj_id not in ref_cache:
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

                exemplar_ref = ref_cache[obj_id]
                prepared.append(
                    {
                        "object_id": obj_id,
                        "image_bgr": image_bgr,
                        "gt_masks": gt_masks,
                        "exemplar_ref": exemplar_ref,
                    }
                )

            if not prepared:
                continue

            debug_target_idx = 0
            debug_saved = False
            should_save_debug = args.save_debug_every > 0 and (batch_step + 1) % args.save_debug_every == 0

            group_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
            for idx, entry in enumerate(prepared):
                img_t = detmodel.image_encoder.prepare_image(
                    entry["image_bgr"],
                    max_side_length=args.max_side_length,
                    use_square_sizing=not args.no_square,
                )
                entry["img_tensor"] = img_t
                entry["preencode_hw"] = img_t.shape[2:]
                shape_key = (img_t.shape[2], img_t.shape[3])
                group_map[shape_key].append(idx)

            batch_losses: List[torch.Tensor] = []
            batch_top_ious: List[float] = []
            for _, idxs in group_map.items():
                img_batch = torch.cat([prepared[i]["img_tensor"] for i in idxs], dim=0)
                with torch.no_grad():
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
                    continue

                for local_idx, data_idx in enumerate(idxs):
                    preencode_hw = prepared[data_idx]["preencode_hw"]
                    gt_targets: List[torch.Tensor] = []
                    for gt_mask in prepared[data_idx]["gt_masks"]:
                        gt_preenc = resize_mask(gt_mask, preencode_hw)
                        gt_tensor = torch.from_numpy(gt_preenc).to(device).unsqueeze(0).unsqueeze(0)
                        gt_down = F.interpolate(gt_tensor, size=mask_preds.shape[-2:], mode="nearest").squeeze(0)
                        gt_targets.append(gt_down[0].float())
                    if not gt_targets:
                        continue

                    logits_mhw = mask_preds[local_idx].float()
                    iou_gm = compute_mask_ious_per_gt(logits_mhw, gt_targets)
                    flat_best_idx = int(torch.argmax(iou_gm).item())
                    best_gt_idx = flat_best_idx // max(1, logits_mhw.shape[0])
                    best_idx = flat_best_idx % max(1, logits_mhw.shape[0])
                    loss_mask, _ = compute_mask_loss_best(
                        logits_mhw,
                        gt_targets[best_gt_idx],
                        bce_weight=args.bce_weight,
                        dice_weight=args.dice_weight,
                    )
                    loss_presence = compute_presence_loss(
                        det_scores[local_idx],
                        best_idx,
                        pos_weight=args.score_weight,
                        neg_weight=args.no_object_weight,
                    )
                    loss_weighted_mask, _ = compute_score_weighted_mask_loss_best_gt_per_pred(
                        logits_mhw,
                        gt_targets,
                        det_scores[local_idx],
                        bce_weight=args.bce_weight,
                        dice_weight=args.dice_weight,
                        power=2.0,
                    )
                    loss = loss_weighted_mask + loss_mask * 0.6
                    batch_losses.append(loss)

                    top_idx = int(torch.argmax(det_scores[local_idx]).item())
                    top_pred = mask_preds[local_idx, top_idx]
                    pred_mask = torch.sigmoid(top_pred) > 0.5
                    top_gt_idx = int(torch.argmax(iou_gm[:, top_idx]).item())
                    gt_mask = gt_targets[top_gt_idx] > 0.5
                    intersection = torch.logical_and(pred_mask, gt_mask).sum().float()
                    union = torch.logical_or(pred_mask, gt_mask).sum().float()
                    if union.item() > 0:
                        batch_top_ious.append(float((intersection / union).item()))
                    else:
                        batch_top_ious.append(0.0)

                    if should_save_debug and not debug_saved and data_idx == debug_target_idx:
                        debug_path = os.path.join(debug_dir, f"boxes_{epoch:03d}_{batch_step:06d}.png")
                        save_debug_collage(
                            prepared[data_idx]["image_bgr"],
                            box_preds[local_idx],
                            det_scores[local_idx],
                            mask_preds[local_idx],
                            debug_path,
                            object_id=prepared[data_idx]["object_id"],
                            reference_dir=reference_dir,
                            ref_view_ids=ref_view_ids,
                            gt_masks=prepared[data_idx]["gt_masks"],
                            ref_image_cache=ref_image_cache,
                        )
                        debug_saved = True

            if not batch_losses:
                continue

            batch_loss = torch.stack(batch_losses).mean()
            running_losses.append(float(batch_loss.item()))
            batch_top_iou = float(np.mean(batch_top_ious)) if batch_top_ious else 0.0
            batch_loss.backward()
            batch_step += 1
            global_step += 1
            if global_step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(batch_loss.item())
            epoch_count += 1

            if args.log_every > 0 and global_step % args.log_every == 0:
                avg_loss = epoch_loss / max(1, epoch_count)
                running_avg = sum(running_losses) / max(1, len(running_losses))
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={batch_loss.item():.4f} avg_loss={avg_loss:.4f} "
                    f"run5_loss={running_avg:.4f} top_iou={batch_top_iou:.4f}"
                )
                save_path = os.path.join(args.output_dir, f"finetune.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch_step": batch_step,
                        "image_exemplar_fusion": detmodel.image_exemplar_fusion.state_dict(),
                        "exemplar_detector": detmodel.exemplar_detector.state_dict(),
                        "exemplar_segmentation": detmodel.exemplar_segmentation.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    save_path,
                )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"finetune_epoch_{epoch:03d}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "batch_step": batch_step,
                    "image_exemplar_fusion": detmodel.image_exemplar_fusion.state_dict(),
                    "exemplar_detector": detmodel.exemplar_detector.state_dict(),
                    "exemplar_segmentation": detmodel.exemplar_segmentation.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                save_path,
            )
            print(f"Saved checkpoint to {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
