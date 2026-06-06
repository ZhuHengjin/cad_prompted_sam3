#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


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
) -> Tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
    dataset_path = Path(dataset_root).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_root)
    pattern = re.compile(r"instance_segmentation_(\d{4})\.png$")
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
    return mask.astype(np.uint8)


def normalize_dataset_roots(dataset_root: object) -> List[Tuple[str, float]]:
    if dataset_root is None:
        return []
    items: List[object]
    if isinstance(dataset_root, (list, tuple)):
        items = list(dataset_root)
    else:
        items = [dataset_root]
    roots: List[Tuple[str, float]] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            path = str(item[0]).strip()
            if path:
                roots.append((path, float(item[1])))
            continue
        for part in str(item).split(","):
            part = part.strip()
            if part:
                roots.append((part, 1.0))
    return roots


def unique_frame_entries(entries: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    unique_entries: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for entry in entries:
        key = (entry["frame_id"], entry["rgb_path"], entry["inst_path"])
        if key not in unique_entries:
            unique_entries[key] = entry
    return list(unique_entries.values())


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


def load_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_reference_images(
    object_id: str,
    reference_dir: Path,
    ref_view_ids: List[str],
    max_views: int = 1,
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


def build_object_masks(
    seg: np.ndarray,
    color_map: Dict[str, List[Tuple[int, ...]]],
    min_area: int,
) -> Dict[str, np.ndarray]:
    object_masks: Dict[str, np.ndarray] = {}
    for object_id, colors in color_map.items():
        combined = np.zeros(seg.shape[:2], dtype=np.uint8)
        for color in colors:
            mask = mask_for_color(seg, color)
            if mask.sum() > 0:
                combined = np.maximum(combined, mask)
        if combined.sum() >= min_area:
            object_masks[object_id] = combined
    return object_masks


def _draw_text_box(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    x = max(2, min(x, image.shape[1] - text_w - 2))
    y = max(text_h + baseline + 2, min(y, image.shape[0] - 2))
    box_tl = (x - 2, y - text_h - baseline - 2)
    box_br = (x + text_w + 2, y + baseline + 2)
    cv2.rectangle(image, box_tl, box_br, (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_contours_with_labels(
    image_bgr: np.ndarray,
    object_masks: Dict[str, np.ndarray],
) -> np.ndarray:
    out = image_bgr.copy()
    palette: List[Tuple[int, int, int]] = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 128, 0),
        (128, 255, 0),
        (0, 128, 255),
        (255, 0, 128),
    ]
    for idx, object_id in enumerate(sorted(object_masks)):
        mask = object_masks[object_id]
        if mask is None or mask.sum() == 0:
            continue
        color = palette[idx % len(palette)]
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(out, contours, -1, color, 2)
        ys, xs = np.where(mask_u8 > 0)
        if xs.size > 0 and ys.size > 0:
            x1, y1 = int(xs.min()), int(ys.min())
            _draw_text_box(out, object_id, (x1, max(0, y1 - 6)), color, 0.5, 1)
    return out


def build_reference_panel(
    object_ids: Sequence[str],
    reference_dir: Path,
    ref_view_ids: List[str],
    target_height: int,
    target_width: int,
    cache: Dict[str, Optional[np.ndarray]],
    max_objects: int = 9,
) -> np.ndarray:
    if target_height <= 0 or target_width <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    object_ids = list(object_ids)[:max_objects]
    tile_h = max(1, target_height // 3)
    tile_w = max(1, target_width // 3)

    grid = np.zeros((tile_h * 3, tile_w * 3, 3), dtype=np.uint8)
    for idx, object_id in enumerate(object_ids):
        row = idx // 3
        col = idx % 3
        if row >= 3:
            break
        if object_id in cache:
            ref_img = cache[object_id]
        else:
            ref_images = load_reference_images(object_id, reference_dir, ref_view_ids, max_views=1)
            ref_img = ref_images[0] if ref_images else None
            cache[object_id] = ref_img
        if ref_img is None:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        else:
            tile = cv2.resize(ref_img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        _draw_text_box(tile, object_id, (4, tile_h - 6), (255, 255, 255), 0.5, 1)
        y0, x0 = row * tile_h, col * tile_w
        grid[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

    panel = grid
    if panel.shape[0] != target_height or panel.shape[1] != target_width:
        panel = cv2.resize(panel, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return panel


def add_title_bar(image_bgr: np.ndarray, title: str) -> np.ndarray:
    bar_h = 36
    bar = np.zeros((bar_h, image_bgr.shape[1], 3), dtype=np.uint8)
    _draw_text_box(bar, title, (8, bar_h - 10), (255, 255, 255), 0.5, 1)
    return np.concatenate([bar, image_bgr], axis=0)


def _slugify(path: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", path.strip().strip("/"))
    return cleaned or "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-frame debug images for multi-GT datasets.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        nargs="+",
        default=[("/sata1/data/kevin/v2_dataset/v2_multi_gt/v2_sdg_output", 2), ("/sata1/data/kevin/v2_imgs/train_1", 1)],
        help="Dataset roots (space or comma separated).",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="/sata1/data/kevin/v2_3d/v2_usds/v2_stl_render_0212",
        help="Path to reference renders.",
    )
    parser.add_argument(
        "--ref_view_ids",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12",
        help="Reference view ids to use (comma-separated).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_debug",
        help="Output directory for debug images.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional cap on number of frames to render (0 = all).",
    )
    parser.add_argument(
        "--min_mask_area",
        type=int,
        default=1,
        help="Minimum mask area (in pixels) to keep an object.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_roots = normalize_dataset_roots(args.dataset_root)
    if not dataset_roots:
        raise ValueError("No dataset roots provided.")

    ref_view_ids = parse_ref_view_ids(args.ref_view_ids)
    if not ref_view_ids:
        raise ValueError("No reference view ids resolved.")

    reference_dir = Path(args.reference_dir).expanduser().resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    total_saved = 0
    ref_cache: Dict[str, Optional[np.ndarray]] = {}

    for root, _multiplier in dataset_roots:
        root_slug = _slugify(root)
        root_dir = Path(args.output_dir) / root_slug
        root_dir.mkdir(parents=True, exist_ok=True)

        _obj_map, entries = collect_multi_object_samples(root)
        entries = unique_frame_entries(entries)
        entries = sorted(entries, key=lambda item: item["rgb_path"])

        for entry in entries:
            if args.max_images > 0 and total_saved >= args.max_images:
                return
            rgb_path = entry["rgb_path"]
            inst_path = entry["inst_path"]
            frame_id = entry["frame_id"]
            mapping_path = Path(inst_path).with_name(f"instance_segmentation_mapping_{frame_id}.json")
            if not mapping_path.is_file():
                continue

            try:
                image_bgr = load_bgr(rgb_path)
                seg = load_instance_segmentation(inst_path)
                color_map = load_color_mapping(mapping_path)
            except FileNotFoundError:
                continue
            if not color_map:
                continue

            object_masks = build_object_masks(seg, color_map, min_area=args.min_mask_area)
            if not object_masks:
                continue

            object_ids = sorted(object_masks.keys())
            model_panel = build_reference_panel(
                object_ids,
                reference_dir,
                ref_view_ids,
                target_height=image_bgr.shape[0],
                target_width=image_bgr.shape[1],
                cache=ref_cache,
                max_objects=9,
            )
            labeled_panel = draw_contours_with_labels(image_bgr, object_masks)
            raw_panel = image_bgr.copy()

            if model_panel.shape[0] != labeled_panel.shape[0]:
                target_h = max(model_panel.shape[0], labeled_panel.shape[0])
                if model_panel.shape[0] < target_h:
                    pad = np.zeros((target_h - model_panel.shape[0], model_panel.shape[1], 3), dtype=np.uint8)
                    model_panel = np.concatenate([model_panel, pad], axis=0)
                if labeled_panel.shape[0] < target_h:
                    pad = np.zeros((target_h - labeled_panel.shape[0], labeled_panel.shape[1], 3), dtype=np.uint8)
                    labeled_panel = np.concatenate([labeled_panel, pad], axis=0)

            if raw_panel.shape[0] != labeled_panel.shape[0]:
                target_h = labeled_panel.shape[0]
                if raw_panel.shape[0] < target_h:
                    pad = np.zeros((target_h - raw_panel.shape[0], raw_panel.shape[1], 3), dtype=np.uint8)
                    raw_panel = np.concatenate([raw_panel, pad], axis=0)
                else:
                    raw_panel = raw_panel[:target_h, :, :]

            combined = np.concatenate([model_panel, labeled_panel, raw_panel], axis=1)
            combined = add_title_bar(combined, rgb_path)

            out_name = f"{Path(rgb_path).stem}_debug.png"
            out_path = root_dir / out_name
            cv2.imwrite(str(out_path), combined)
            total_saved += 1


if __name__ == "__main__":
    main()
