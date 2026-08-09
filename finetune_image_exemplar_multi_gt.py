#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical SAMv3 multi-GT exemplar training and evaluation entry point.

Manifest mode resolves exact dataset/camera/frame rows, trains only on the
``train`` split, and evaluates every target object in the unweighted
``validation`` split between epochs. Training epochs retain the original total
view count while allocating draws equally among dataset IDs; small domains are
sampled with replacement using an epoch-local ``seed + epoch`` RNG. Final test
evaluation is available only through explicit ``--eval_only --eval_split test``
with a fine-tuned checkpoint.

Each manifest-backed run stores a manifest copy, SHA-256 digest, resolved split
counts, sampling policy, data root, and CLI arguments beside its checkpoints.
The older root and frame-CSV path remains temporarily available for migration,
but cannot be mixed with the dataset-qualified manifest interface. Model loss,
matching, augmentation, checkpoint keys, and debug visualization behavior are
otherwise retained from the original multi-GT trainer.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from loss_fns import (
    compute_bbox_l1_loss_from_matches,
    compute_presence_loss_logits,
)
from dataset_manifest import (
    ManifestRow,
    balanced_epoch_entries,
    load_manifest,
    manifest_sha256,
)
from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.v3_sam.cad_pose.geometry import adjust_intrinsics_for_resize_and_pad
from muggled_sam.v3_sam.cad_pose.evaluation import (
    PoseEvaluation,
    evaluate_pose_matches,
    expected_calibration_error,
    fit_pose_score_temperature,
)
from muggled_sam.v3_sam.cad_pose.inference import mask_nms_indices
from muggled_sam.v3_sam.cad_pose.losses import (
    CADPoseLossConfig,
    compute_cad_pose_losses,
    point_set_pose_errors,
)
from muggled_sam.v3_sam.cad_pose.types import CADPosePredictions, CADPoseTarget
from muggled_sam.v3_sam.cad_pose.matching import match_pose_predictions_one_to_one
from muggled_sam.v3_sam.cad_pose.symmetry import symmetry_aware_rotation_error
from muggled_sam.v3_sam.exemplar_view_pose import (
    EXEMPLAR_VIEW_MODES,
    ExemplarViewBundle,
    load_reference_view_rotations,
    pad_exemplar_view_batch,
)
from muggled_sam.v3_sam.cad_pose.dataset import (
    effective_surface_centroid_m,
    instance_mask_rgba,
    load_perseve_pose_sample,
    make_pose_target,
    validate_scale_sharing,
)


IGNORED_LABELS = {"BACKGROUND", "UNLABELLED"}
METRIC_FIELDS = (
    "phase",
    "epoch",
    "global_step",
    "batch_step",
    "loss",
    "avg_loss",
    "avg_iou",
    "correct_rate",
    "mask_loss",
    "bbox_loss",
    "objectness_loss",
    "pose_center_loss",
    "pose_depth_loss",
    "pose_rotation_loss",
    "pose_full_set_loss",
    "pose_quality_loss",
    "pose_aux_loss",
    "mean_surface_distance_norm",
    "p95_surface_distance_norm",
    "centroid_error_cm",
    "pose_success_rate",
    "rotation_error_deg",
    "translation_error_cm",
    "center_error_norm",
    "depth_error_m",
    "accuracy_5deg_5cm",
    "accuracy_10deg_10cm",
    "brier_score",
    "expected_calibration_error",
    "pose_score_temperature",
    "pose_match_iou_threshold",
    "pose_assignment_coverage",
    "pose_match_coverage",
    "pose_match_acceptance_rate",
    "pose_end_to_end_success_rate",
    "eligible_samples",
    "pose_accepted_matches",
    "pose_total_matches",
    "samples",
)


@dataclass(frozen=True)
class AuxiliaryPosePredictions:
    """Pose predictions from opt-in intermediate detector supervision."""

    # Layer numbers are one-based in logs/configuration for consistency with
    # the six-layer detector description used by training documentation.
    layers: tuple[int, ...]
    predictions: tuple[CADPosePredictions, ...]

    def __post_init__(self) -> None:
        if len(self.layers) != len(self.predictions):
            raise ValueError("Auxiliary pose layers and predictions must have equal length")

    def index_candidates(self, index: torch.Tensor) -> "AuxiliaryPosePredictions":
        return AuxiliaryPosePredictions(
            self.layers,
            tuple(prediction.index_candidates(index) for prediction in self.predictions),
        )


@dataclass(frozen=True)
class EvalConfig:
    dataset_root: object
    reference_dir: str
    ref_view_ids: str
    split_csv: str = ""
    dataset_entries: Optional[Sequence[Dict[str, str]]] = None
    max_side_length: int = 1008
    use_square_sizing: bool = True
    num_points_approx: int = 25
    batch_size: int = 8
    det_filter: float = 0.0
    nms_iou: float = 0.5
    matches_per_gt: int = 1
    bce_weight: float = 2.0
    dice_weight: float = 2.0
    bbox_weight: float = 1.0
    score_weight: float = 0.3
    no_object_weight: float = 0.45
    shuffle: bool = False
    max_batches: int = 0
    exemplar_view_mode: str = "none"
    exemplar_view_shuffle_seed: int = 0


@dataclass(frozen=True)
class MultiGTDetectionLosses:
    """Separately weighted mask, box, and objectness supervision."""

    mask: torch.Tensor
    bbox: torch.Tensor
    objectness: torch.Tensor

    def total(
        self,
        *,
        mask_weight: float = 2.0,
        bbox_weight: float = 1.0,
        objectness_weight: float = 1.0,
    ) -> torch.Tensor:
        return (
            mask_weight * self.mask
            + bbox_weight * self.bbox
            + objectness_weight * self.objectness
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


def select_object_with_most_instances(
    mapping_path: str,
    mapping_cache: Optional[Dict[str, Dict[str, List[Tuple[int, ...]]]]] = None,
) -> Optional[str]:
    mapping_key = str(mapping_path)
    if mapping_cache is not None and mapping_key in mapping_cache:
        color_map = mapping_cache[mapping_key]
    else:
        color_map = load_color_mapping(Path(mapping_path))
        if mapping_cache is not None:
            mapping_cache[mapping_key] = color_map
    if not color_map:
        return None
    obj_ids = list(color_map.keys())
    if not obj_ids:
        return None
    return random.choice(obj_ids)


def select_target_object_and_masks(
    inst_path: str,
    mapping_path: str,
    seg_cache: Optional[Dict[str, np.ndarray]] = None,
    mapping_cache: Optional[Dict[str, Dict[str, List[Tuple[int, ...]]]]] = None,
) -> Tuple[Optional[str], List[np.ndarray]]:
    obj_id = select_object_with_most_instances(mapping_path, mapping_cache=mapping_cache)
    if obj_id is None:
        return None, []
    masks = load_instance_masks_for_object(
        inst_path,
        mapping_path,
        obj_id,
        seg_cache=seg_cache,
        mapping_cache=mapping_cache,
    )
    return obj_id, masks


def select_pose_target_and_masks(
    entry: Dict[str, str],
    pose_cache: Dict[Tuple[str, str], object],
    *,
    require_pose_eligible: bool,
):
    """Select one CAD prompt and retain the exact sidecar-instance/mask join order."""

    camera_root = Path(entry["inst_path"]).parent
    cache_key = (str(camera_root), entry["frame_id"])
    sample = pose_cache.get(cache_key)
    if sample is None:
        sample = load_perseve_pose_sample(camera_root, entry["frame_id"], validate_pixels=True)
        pose_cache[cache_key] = sample
    grouped: Dict[str, List[object]] = defaultdict(list)
    for instance in sample.frame.instances:
        if instance.annotation_state != "visible" or instance.mask_rgba is None or instance.dimensions_m is None:
            continue
        if require_pose_eligible and not instance.pose_training_eligible:
            continue
        grouped[instance.cad_id].append(instance)
    if not grouped:
        return None, [], [], sample, []
    cad_id = random.choice(sorted(grouped))
    masks, instances = [], []
    for instance in grouped[cad_id]:
        mask = instance_mask_rgba(Path(entry["inst_path"]), instance).astype(np.float32)
        if int(mask.sum()) >= 100:
            masks.append(mask)
            instances.append(instance)
    if not masks:
        return None, [], [], sample, []
    dimensions = [instance.dimensions_m for instance in instances]
    if any(value is None or not np.allclose(value, dimensions[0], atol=1e-6, rtol=1e-5) for value in dimensions):
        raise ValueError(f"Instances for CAD {cad_id!r} do not share one effective dimension prompt")
    eligible_gt_indices = [index for index, instance in enumerate(instances) if instance.pose_training_eligible]
    return cad_id, masks, instances, sample, eligible_gt_indices


def pose_prompt_surface_centroid_m(
    instances: Sequence[object],
    eligible_gt_indices: Sequence[int],
    catalog_object: object,
) -> np.ndarray:
    """Return valid centroid metadata for a mixed detection/pose training item.

    Joint stages retain visible, pose-ineligible instances as detector anchors.
    If none of the selected instances is pose eligible, the pose prediction is
    discarded and a zero centroid keeps that detection-only forward pass valid.
    """

    if not eligible_gt_indices:
        return np.zeros(3, dtype=np.float64)
    prompt_index = int(eligible_gt_indices[0])
    if prompt_index < 0 or prompt_index >= len(instances):
        raise IndexError(f"Pose-eligible GT index {prompt_index} is outside the selected instances")
    return effective_surface_centroid_m(instances[prompt_index], catalog_object)


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


def parse_split_ratios(value: str) -> Tuple[float, float, float]:
    """Parse and normalize train/validation/test ratios from CLI text.

    The ratios do not need to sum to 1.0; for example, ``8,1,1`` and
    ``0.8,0.1,0.1`` are equivalent. The returned tuple always sums to 1.0.
    """
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated split ratios, got: {value}")
    try:
        train_ratio, val_ratio, test_ratio = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Invalid split ratios: {value}") from exc
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError(f"Split ratios must be non-negative, got: {value}")
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError(f"At least one split ratio must be positive, got: {value}")
    return train_ratio / total, val_ratio / total, test_ratio / total


def load_split_frame_ids(csv_path: Path) -> set:
    """Load frame IDs from a split CSV for logging and validation.

    Split CSVs are compatible with ``load_dataset_filter_set`` and can contain
    either a simple ``frame_id`` column or the fuller dataset-filter format with
    ``frame_id``, ``rgb_path``, and ``inst_path``. For the canonical train/val/
    test split, only ``frame_id`` is written and required.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    _, frame_ids = load_dataset_filter_set(csv_path)
    return frame_ids if frame_ids is not None else set()


def write_frame_split_csv(csv_path: Path, frame_ids: Sequence[str]) -> None:
    """Write a simple frame-level split CSV.

    The file contains only ``frame_id`` values so the same frame ID can match
    entries from every camera directory. This keeps camera views of the same
    scene together in train, validation, or test.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id"])
        writer.writeheader()
        for frame_id in frame_ids:
            writer.writerow({"frame_id": frame_id})


def create_frame_split_csvs(
    split_dir: Path,
    frame_ids: Sequence[str],
    ratios: Tuple[float, float, float],
    seed: int,
    overwrite: bool = False,
) -> Tuple[Path, Path, Path]:
    """Create or reuse train/validation/test CSV files under ``split_dir``.

    Behavior is intentionally conservative:
    - If all three files exist and ``overwrite`` is false, reuse them.
    - If only some files exist and ``overwrite`` is false, fail loudly.
    - If ``overwrite`` is true, regenerate all three files using ``seed``.

    The split is generated from unique frame IDs after any dataset-level filter
    has been applied, before dataset multipliers are applied. Multipliers can
    duplicate training examples, but they do not change split membership.
    """
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"
    split_paths = (train_path, val_path, test_path)
    if not overwrite:
        existing = [path.is_file() for path in split_paths]
        if all(existing):
            return split_paths
        if any(existing):
            raise FileExistsError(
                f"Partial split files exist under {split_dir}. "
                "Provide all of train.csv/val.csv/test.csv or pass --recreate_splits."
            )
    if not frame_ids:
        raise ValueError("Cannot create data splits with no frame ids.")

    shuffled = sorted(set(frame_ids))
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    train_ratio, val_ratio, _ = ratios
    total = len(shuffled)
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    train_count = min(max(train_count, 0), total)
    val_count = min(max(val_count, 0), total - train_count)

    train_ids = sorted(shuffled[:train_count])
    val_ids = sorted(shuffled[train_count : train_count + val_count])
    test_ids = sorted(shuffled[train_count + val_count :])

    write_frame_split_csv(train_path, train_ids)
    write_frame_split_csv(val_path, val_ids)
    write_frame_split_csv(test_path, test_ids)
    return split_paths


def unique_frame_entries(entries: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    unique_entries: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for entry in entries:
        key = (entry["frame_id"], entry["rgb_path"], entry["inst_path"])
        if key not in unique_entries:
            unique_entries[key] = entry
    return list(unique_entries.values())


def unique_object_entries(entries: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    unique_entries: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for entry in entries:
        key = (entry["frame_id"], entry["object_id"], entry["rgb_path"], entry["inst_path"])
        if key not in unique_entries:
            unique_entries[key] = entry
    return list(unique_entries.values())


def _partition_unresolved_pose_frames(
    camera_root: Path, frame_ids: Sequence[str]
) -> Tuple[List[str], List[str]]:
    """Separate valid no-supervision pose frames from genuinely unresolved rows.

    A manifest describes every captured frame, including valid captures with no
    visible instances. The legacy segmentation parser intentionally omits those
    frames because they cannot produce a supervised sample. Keep that behavior,
    but do not confuse a valid empty pose sidecar with a missing or corrupt
    label join. Sidecars with any visible instance remain hard failures.
    """

    empty_frames: List[str] = []
    unresolved_frames: List[str] = []
    for frame_id in frame_ids:
        sidecar_path = camera_root / f"pose_annotations_{frame_id}.json"
        if not sidecar_path.is_file():
            unresolved_frames.append(frame_id)
            continue
        sample = load_perseve_pose_sample(camera_root, frame_id, validate_pixels=True)
        if any(instance.annotation_state == "visible" for instance in sample.frame.instances):
            unresolved_frames.append(frame_id)
        else:
            empty_frames.append(frame_id)
    return empty_frames, unresolved_frames


def entries_from_manifest_rows(
    rows: Sequence[ManifestRow], data_root: Path, *, object_level: bool
) -> List[Dict[str, str]]:
    """Resolve exact manifest samples through the existing dataset parser."""
    requested: Dict[Tuple[str, str], Dict[str, ManifestRow]] = defaultdict(dict)
    for row in rows:
        camera_root = str((data_root / row.dataset_path / row.camera_dir).resolve())
        requested[(row.dataset_id, camera_root)][row.frame_id] = row

    resolved: List[Dict[str, str]] = []
    for (dataset_id, camera_root), frame_rows in sorted(requested.items()):
        _, camera_entries = collect_multi_object_samples(camera_root)
        camera_entries = (
            unique_object_entries(camera_entries) if object_level else unique_frame_entries(camera_entries)
        )
        found_frames = set()
        for entry in camera_entries:
            row = frame_rows.get(entry["frame_id"])
            if row is None:
                continue
            enriched = dict(entry)
            enriched["dataset_id"] = dataset_id
            enriched["group_id"] = row.group_id
            enriched["split"] = row.split
            resolved.append(enriched)
            found_frames.add(entry["frame_id"])
        missing = sorted(set(frame_rows) - found_frames)
        empty_pose_frames, unresolved = _partition_unresolved_pose_frames(
            Path(camera_root), missing
        )
        if empty_pose_frames:
            print(
                f"Skipping {len(empty_pose_frames)} valid pose frames with no visible supervision "
                f"under {camera_root}: {empty_pose_frames[:10]}"
            )
        if unresolved:
            raise RuntimeError(
                f"Manifest rows under {camera_root} yielded no usable labeled entries for frames: "
                f"{unresolved[:10]}"
            )
    return resolved


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
    hue_shift = random.uniform(-9.0, 9.0)
    sat_scale = random.uniform(0.8, 1.2)
    val_scale = random.uniform(0.8, 1.2)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _ensure_mask_list(gt_mask: Union[np.ndarray, Sequence[np.ndarray]]) -> List[np.ndarray]:
    if isinstance(gt_mask, np.ndarray):
        return [gt_mask]
    return [mask for mask in gt_mask if isinstance(mask, np.ndarray)]


def select_top_gt_masks(gt_masks: Sequence[np.ndarray], max_instances: Optional[int] = None) -> List[np.ndarray]:
    if not gt_masks:
        return []
    if max_instances is None:
        return list(gt_masks)
    sorted_masks = sorted(gt_masks, key=lambda mask: float(mask.sum()), reverse=True)
    return sorted_masks[: max(1, max_instances)]


def data_augmentation(
    image_bgr: np.ndarray,
    gt_mask: Union[np.ndarray, Sequence[np.ndarray]],
    min_crop_scale: float = 0.6,
    max_crop_scale: float = 1.0,
) -> Tuple[np.ndarray, Union[np.ndarray, List[np.ndarray]]]:
    orig_h, orig_w = image_bgr.shape[:2]
    mask_list = _ensure_mask_list(gt_mask)
    if mask_list:
        combined = np.zeros((orig_h, orig_w), dtype=np.float32)
        aligned_masks: List[np.ndarray] = []
        for mask in mask_list:
            if mask.shape[:2] != (orig_h, orig_w):
                mask = cv2.resize(mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            aligned_masks.append(mask)
            combined = np.maximum(combined, mask.astype(np.float32))
        mask_list = aligned_masks
        mask_bin = combined > 0.5
    else:
        mask_bin = np.zeros((orig_h, orig_w), dtype=bool)

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
            if mask_list:
                mask_list = [mask[y0 : y0 + crop_h, x0 : x0 + crop_w] for mask in mask_list]
            image_bgr = cv2.resize(image_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            if mask_list:
                mask_list = [
                    cv2.resize(mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    for mask in mask_list
                ]
            break
    image_bgr = apply_random_color_distortion(image_bgr)
    if isinstance(gt_mask, np.ndarray):
        return image_bgr, (mask_list[0] if mask_list else gt_mask)
    return image_bgr, mask_list


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


def build_gt_down_list(
    gt_masks: Sequence[np.ndarray],
    preencode_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    device: torch.device,
) -> List[torch.Tensor]:
    gt_down_list: List[torch.Tensor] = []
    for gt_mask in gt_masks:
        gt_preenc = resize_mask(gt_mask, preencode_hw)
        gt_tensor = torch.from_numpy(gt_preenc).to(device).unsqueeze(0).unsqueeze(0)
        gt_down = F.interpolate(gt_tensor, size=target_hw, mode="nearest").squeeze(0).squeeze(0)
        gt_down_list.append(gt_down > 0.5)
    return gt_down_list


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


def parse_pose_aux_layer_indices(value: str, num_layers: int) -> tuple[int, ...]:
    """Parse one-based auxiliary layers and return zero-based detector indices."""

    try:
        layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--pose_aux_layers must be a comma-separated integer list") from exc
    if not layers:
        raise ValueError("--pose_aux_layers must contain at least one layer")
    if len(set(layers)) != len(layers):
        raise ValueError("--pose_aux_layers must not contain duplicates")
    # The final detector layer already receives the primary loss. Auxiliary
    # supervision is intentionally restricted to earlier query states.
    if any(layer < 1 or layer >= num_layers for layer in layers):
        raise ValueError(
            f"--pose_aux_layers must be between 1 and {num_layers - 1}; "
            f"layer {num_layers} is the primary output"
        )
    return tuple(layer - 1 for layer in layers)


def reference_view_id_candidates(ref_id: str) -> List[str]:
    """Return compatible padded and unpadded spellings for a numeric view ID."""

    candidates = [str(ref_id)]
    try:
        numeric_id = int(ref_id)
    except ValueError:
        return candidates
    for candidate in (str(numeric_id), f"{numeric_id:02d}"):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def resolve_reference_pair(
    reference_dir: Path, object_id: str, ref_id: str
) -> Optional[Tuple[Path, Path]]:
    """Resolve one render/mask pair across legacy and unpadded view names."""

    for candidate_ref_id in reference_view_id_candidates(ref_id):
        ref_stub = f"{object_id}_stl_base_{candidate_ref_id}"
        image_path = reference_dir / f"{ref_stub}.png"
        mask_path = reference_dir / f"{ref_stub}_mask.png"
        if image_path.is_file() and mask_path.is_file():
            return image_path, mask_path
    return None


def validate_pose_reference_metadata(
    reference_dir: Path,
    cad_ids: Sequence[str],
    catalog: Mapping[str, Any],
    ref_view_ids: Sequence[str],
) -> Dict[str, int]:
    """Require pose exemplars to preserve the active catalog's CAD frame."""

    errors: List[str] = []
    validated_views = 0
    for cad_id in sorted(set(cad_ids)):
        metadata_path = reference_dir / f"{cad_id}_render_transform.json"
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
        except FileNotFoundError:
            errors.append(f"{cad_id}: missing {metadata_path.name}")
            continue
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{cad_id}: unreadable {metadata_path.name}: {error}")
            continue

        if not isinstance(metadata, dict):
            errors.append(f"{cad_id}: transform metadata must be a JSON object")
            continue
        geometry = metadata.get("geometry")
        if not isinstance(geometry, dict):
            errors.append(f"{cad_id}: transform metadata has no geometry object")
            continue
        if metadata.get("object_id") != cad_id:
            errors.append(
                f"{cad_id}: metadata object_id is {metadata.get('object_id')!r}"
            )
        if geometry.get("orientation_mode") != "canonical":
            errors.append(
                f"{cad_id}: orientation mode is {geometry.get('orientation_mode')!r}, "
                "expected 'canonical'"
            )
        if geometry.get("catalog_driven") is not True:
            errors.append(f"{cad_id}: canonical render is not catalog-driven")

        catalog_object = catalog.get(cad_id)
        if catalog_object is None:
            errors.append(f"{cad_id}: absent from the active pose catalog")
            continue
        try:
            source_to_meters = float(geometry["source_to_meters"])
            canonical_transform = np.asarray(
                geometry["T_cad_from_source_meters"], dtype=np.float64
            )
            presentation_transform = np.asarray(
                geometry["T_presentation_from_cad"], dtype=np.float64
            )
            canonical_dimensions = np.asarray(
                geometry["canonical_dimensions_m"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{cad_id}: invalid geometry transform fields: {error}")
            continue
        if not np.isclose(
            source_to_meters,
            float(catalog_object.source_to_meters),
            atol=1e-12,
            rtol=0.0,
        ):
            errors.append(f"{cad_id}: source-to-metre scale differs from pose catalog")
        if canonical_transform.shape != (4, 4) or not np.allclose(
            canonical_transform,
            catalog_object.T_cad_from_source_meters,
            atol=1e-10,
            rtol=0.0,
        ):
            errors.append(f"{cad_id}: source-to-CAD transform differs from pose catalog")
        if presentation_transform.shape != (4, 4) or not np.allclose(
            presentation_transform, np.eye(4), atol=1e-10, rtol=0.0
        ):
            errors.append(f"{cad_id}: canonical render contains a presentation rotation")
        if canonical_dimensions.shape != (3,) or not np.allclose(
            canonical_dimensions,
            catalog_object.base_dimensions_m,
            atol=1e-6,
            rtol=1e-5,
        ):
            errors.append(f"{cad_id}: rendered canonical dimensions differ from pose catalog")

        views = metadata.get("views")
        if not isinstance(views, list):
            errors.append(f"{cad_id}: transform metadata has no reference views")
            continue
        views_by_id = {
            str(view.get("view_id")): view
            for view in views
            if isinstance(view, dict) and view.get("view_id") is not None
        }
        for ref_id in ref_view_ids:
            reference_pair = resolve_reference_pair(reference_dir, cad_id, ref_id)
            if reference_pair is None:
                errors.append(f"{cad_id}/{ref_id}: missing render/mask pair")
                continue
            view = next(
                (
                    views_by_id[candidate]
                    for candidate in reference_view_id_candidates(ref_id)
                    if candidate in views_by_id
                ),
                None,
            )
            if view is None:
                errors.append(f"{cad_id}/{ref_id}: missing view transform metadata")
                continue
            image_path, mask_path = reference_pair
            if view.get("image") != image_path.name or view.get("mask") != mask_path.name:
                errors.append(f"{cad_id}/{ref_id}: view metadata names different files")
                continue
            try:
                rotation = np.asarray(
                    view.get("R_refcam_cv_from_cad"), dtype=np.float64
                )
            except (TypeError, ValueError):
                errors.append(f"{cad_id}/{ref_id}: invalid reference-camera rotation")
                continue
            if (
                rotation.shape != (3, 3)
                or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0)
            ):
                errors.append(f"{cad_id}/{ref_id}: invalid reference-camera rotation")
                continue
            validated_views += 1

    if errors:
        raise ValueError(
            "Pose reference transform preflight failed; regenerate canonical catalog-driven "
            f"renders. First issues: {errors[:10]}"
        )
    return {"cad_id_count": len(set(cad_ids)), "view_count": validated_views}


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
    cad_dimensions_m_b3: Optional[torch.Tensor] = None,
    cad_effective_surface_centroid_m_b3: Optional[torch.Tensor] = None,
    adjusted_intrinsics_b33: Optional[torch.Tensor] = None,
    model_image_size_wh: Optional[Tuple[int, int]] = None,
    return_pose: bool = False,
    pose_aux_layer_indices: tuple[int, ...] = (),
):
    if pose_aux_layer_indices and not return_pose:
        raise ValueError("Intermediate pose outputs require return_pose=True")
    lowres_imgenc_bchw, hiresx2_imgenc_bchw, hiresx4_imgenc_bchw = encoded_image_features_list
    no_exemplars = encoded_exemplars_bnc.shape[1] == 0
    if no_exemplars:
        blk_tok, blk_box, blk_score, blk_score_logits, blk_pres = detmodel.exemplar_detector.create_blank_output(
            lowres_imgenc_bchw
        )
        blk_masks, _ = detmodel.exemplar_segmentation.create_blank_output(blk_tok, lowres_imgenc_bchw)
        if not return_pose:
            return blk_masks, blk_box, blk_score, blk_score_logits, blk_pres
        # Handle pose prediction when there are no exemplars.
        if cad_dimensions_m_b3 is None:
            raise ValueError("cad_dimensions_m_b3 is required for pose training")
        if cad_effective_surface_centroid_m_b3 is None:
            raise ValueError("cad_effective_surface_centroid_m_b3 is required for pose training")
        if adjusted_intrinsics_b33 is None or model_image_size_wh is None:
            raise ValueError("Adjusted intrinsics and model_image_size_wh are required for pose training")
        pose_predictions = detmodel.cad_pose_head(
            blk_tok,
            blk_box,
            cad_dimensions_m_b3,
            cad_effective_surface_centroid_m_b3,
            adjusted_intrinsics_b33,
            model_image_size_wh,
            encoded_exemplars_bnc,
            exemplar_padding_mask_bn,
        )
        pose_predictions = pose_predictions.with_translation(adjusted_intrinsics_b33, model_image_size_wh)
        outputs = (blk_masks, blk_box, blk_score, blk_score_logits, blk_pres, pose_predictions)
        if pose_aux_layer_indices:
            return (*outputs, AuxiliaryPosePredictions((), ()))
        return outputs

    fused_imgexm_tokens_bchw = detmodel.image_exemplar_fusion(
        lowres_imgenc_bchw,
        encoded_exemplars_bnc,
        exemplar_padding_mask_bn,
    )
    detector_outputs = detmodel.exemplar_detector(
        fused_imgexm_tokens_bchw,
        encoded_exemplars_bnc,
        exemplar_padding_mask_bn,
        intermediate_layer_indices=pose_aux_layer_indices or None,
    )
    if pose_aux_layer_indices:
        (
            enc_det_tokens_bnc,
            boxes_xy1xy2_bn22,
            det_scores_bn,
            det_scores_logits_bn,
            pres_scores,
            intermediate_detector_outputs,
        ) = detector_outputs
    else:
        (
            enc_det_tokens_bnc,
            boxes_xy1xy2_bn22,
            det_scores_bn,
            det_scores_logits_bn,
            pres_scores,
        ) = detector_outputs
        intermediate_detector_outputs = ()

    pose_predictions = None
    # Handle pose prediction for detected candidates.
    if return_pose:
        if cad_dimensions_m_b3 is None:
            raise ValueError("cad_dimensions_m_b3 is required for pose training")
        if cad_effective_surface_centroid_m_b3 is None:
            raise ValueError("cad_effective_surface_centroid_m_b3 is required for pose training")
        if adjusted_intrinsics_b33 is None or model_image_size_wh is None:
            raise ValueError("Adjusted intrinsics and model_image_size_wh are required for pose training")
        pose_predictions = detmodel.cad_pose_head(
            enc_det_tokens_bnc,
            boxes_xy1xy2_bn22,
            cad_dimensions_m_b3,
            cad_effective_surface_centroid_m_b3,
            adjusted_intrinsics_b33,
            model_image_size_wh,
            encoded_exemplars_bnc,
            exemplar_padding_mask_bn,
        )
        pose_predictions = pose_predictions.with_translation(adjusted_intrinsics_b33, model_image_size_wh)

    auxiliary_pose_predictions = AuxiliaryPosePredictions(
        layers=tuple(output.layer_index + 1 for output in intermediate_detector_outputs),
        predictions=tuple(
            detmodel.cad_pose_head(
                output.detection_tokens_bnc,
                output.boxes_xy1xy2_bn22,
                cad_dimensions_m_b3,
                cad_effective_surface_centroid_m_b3,
                adjusted_intrinsics_b33,
                model_image_size_wh,
                encoded_exemplars_bnc,
                exemplar_padding_mask_bn,
            ).with_translation(adjusted_intrinsics_b33, model_image_size_wh)
            for output in intermediate_detector_outputs
        ),
    )

    if detection_filter_threshold > 1e-3:
        if det_scores_bn.shape[0] != 1:
            raise ValueError("Cannot pre-filter detections when using batched inputs!")
        ok_filter = det_scores_bn[0] > detection_filter_threshold
        enc_det_tokens_bnc = enc_det_tokens_bnc[:, ok_filter]
        boxes_xy1xy2_bn22 = boxes_xy1xy2_bn22[:, ok_filter]
        det_scores_bn = det_scores_bn[:, ok_filter]
        det_scores_logits_bn = det_scores_logits_bn[:, ok_filter]
        if pose_predictions is not None:
            pose_predictions = pose_predictions.index_candidates(ok_filter)
        if auxiliary_pose_predictions.predictions:
            auxiliary_pose_predictions = auxiliary_pose_predictions.index_candidates(ok_filter)

    mask_preds_bnhw, _ = detmodel.exemplar_segmentation(
        enc_det_tokens_bnc,
        fused_imgexm_tokens_bchw,
        hiresx2_imgenc_bchw,
        hiresx4_imgenc_bchw,
        encoded_exemplars_bnc,
        exemplar_padding_mask_bn,
    )
    outputs = (mask_preds_bnhw, boxes_xy1xy2_bn22, det_scores_bn, det_scores_logits_bn, pres_scores)
    if not return_pose:
        return outputs
    pose_outputs = (*outputs, pose_predictions)
    if pose_aux_layer_indices:
        return (*pose_outputs, auxiliary_pose_predictions)
    return pose_outputs


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


def compute_mask_iou(pred_mask_hw: torch.Tensor, gt_mask_hw: torch.Tensor) -> torch.Tensor:
    pred_bin = pred_mask_hw > 0
    gt_bin = gt_mask_hw > 0.5
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum().clamp_min(1)
    return intersection.float() / union.float()


def match_masks_to_gts(
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    if not pred_masks or not gt_masks:
        return []
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
    matches: List[Tuple[int, int, float]] = []
    for iou_val, p_idx, g_idx in pairs:
        if matched_pred[p_idx] or matched_gt[g_idx]:
            continue
        matched_pred[p_idx] = True
        matched_gt[g_idx] = True
        matches.append((p_idx, g_idx, iou_val))
    return matches


def compute_pq_stats(
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    pred_scores: Optional[torch.Tensor] = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
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

    matches = match_masks_to_gts(pred_masks, gt_masks, iou_threshold=iou_threshold)
    sum_iou = sum(match[2] for match in matches)
    tp = len(matches)

    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    return sum_iou, tp, fp, fn


def update_pq_accumulators(
    pq_stats: Dict[float, Dict[str, float]],
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    pred_scores: Optional[torch.Tensor],
    iou_threshold: float,
) -> None:
    for score_threshold, stats in pq_stats.items():
        sum_iou, tp, fp, fn = compute_pq_stats(
            pred_masks,
            gt_masks,
            pred_scores=pred_scores,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        stats["sum_iou"] += sum_iou
        stats["tp"] += tp
        stats["fp"] += fp
        stats["fn"] += fn


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


def match_predictions_to_gts_hungarian(
    logits_mhw: torch.Tensor,
    gt_targets_hw: Sequence[torch.Tensor],
    max_matches: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], Optional[torch.Tensor]]:
    if not gt_targets_hw or logits_mhw.numel() == 0:
        return [], None
    num_gt = len(gt_targets_hw)
    if max_matches is not None:
        num_gt = min(num_gt, max_matches)
    gt_stack = torch.stack([gt_targets_hw[idx] > 0.5 for idx in range(num_gt)], dim=0)
    pred_bin = logits_mhw > 0
    intersection = (pred_bin.unsqueeze(0) & gt_stack.unsqueeze(1)).sum(dim=(2, 3))
    union = (pred_bin.unsqueeze(0) | gt_stack.unsqueeze(1)).sum(dim=(2, 3)).clamp_min(1)
    iou = intersection.float() / union.float()
    num_pred = int(iou.shape[1])
    if num_pred == 0:
        return [], iou

    cost = (1.0 - iou).detach().float().cpu().numpy()
    matches: List[Tuple[int, int]] = hungarian_assign(cost)
    return matches, iou


def match_predictions_to_gts_greedy_k(
    logits_mhw: torch.Tensor,
    gt_targets_hw: Sequence[torch.Tensor],
    max_matches: Optional[int] = None,
    max_per_gt: int = 1,
) -> Tuple[List[Tuple[int, int]], Optional[torch.Tensor]]:
    if not gt_targets_hw or logits_mhw.numel() == 0:
        return [], None
    num_gt = len(gt_targets_hw)
    if max_matches is not None:
        num_gt = min(num_gt, max_matches)
    gt_stack = torch.stack([gt_targets_hw[idx] > 0.5 for idx in range(num_gt)], dim=0)
    pred_bin = logits_mhw > 0
    intersection = (pred_bin.unsqueeze(0) & gt_stack.unsqueeze(1)).sum(dim=(2, 3))
    union = (pred_bin.unsqueeze(0) | gt_stack.unsqueeze(1)).sum(dim=(2, 3)).clamp_min(1)
    iou = intersection.float() / union.float()
    num_pred = int(iou.shape[1])
    if num_pred == 0:
        return [], iou

    cost = (1.0 - iou).detach().float().cpu().numpy()
    matches: List[Tuple[int, int]] = greedy_unique_k_assign(cost, max_per_gt=max_per_gt)
    return matches, iou


def hungarian_assign(cost: np.ndarray) -> List[Tuple[int, int]]:
    if cost.size == 0:
        return []
    n_rows, n_cols = cost.shape
    transposed = False
    if n_rows > n_cols:
        cost = cost.T
        n_rows, n_cols = cost.shape
        transposed = True

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    matches: List[Tuple[int, int]] = []
    for j in range(1, n_cols + 1):
        if p[j] == 0:
            continue
        row = int(p[j] - 1)
        col = int(j - 1)
        if transposed:
            matches.append((col, row))
        else:
            matches.append((row, col))
    return matches


def greedy_unique_k_assign(cost: np.ndarray, max_per_gt: int = 1) -> List[Tuple[int, int]]:
    if cost.size == 0:
        return []
    if max_per_gt <= 0:
        return []
    num_gt, num_pred = cost.shape
    if num_gt == 0 or num_pred == 0:
        return []

    flat = cost.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    gt_counts = np.zeros(num_gt, dtype=np.int64)
    pred_used = np.zeros(num_pred, dtype=bool)
    matches: List[Tuple[int, int]] = []
    total_needed = min(num_pred, num_gt * max_per_gt)

    for idx in order:
        gt_idx = int(idx // num_pred)
        pred_idx = int(idx % num_pred)
        if pred_used[pred_idx]:
            continue
        if gt_counts[gt_idx] >= max_per_gt:
            continue
        matches.append((gt_idx, pred_idx))
        pred_used[pred_idx] = True
        gt_counts[gt_idx] += 1
        if len(matches) >= total_needed:
            break
        if np.all(gt_counts >= max_per_gt):
            break
        if pred_used.all():
            break
    return matches


def compute_multi_gt_detection_losses(
    logits_mhw: torch.Tensor,
    box_preds_n22: torch.Tensor,
    det_scores_logits_n: torch.Tensor,
    gt_targets_hw: Sequence[torch.Tensor],
    bce_weight: float,
    dice_weight: float,
    score_weight: float,
    no_object_weight: float,
    max_per_gt: int = 12,
) -> Optional[MultiGTDetectionLosses]:
    """Return separable multi-GT mask, box, and objectness losses.

    Validation uses this helper under ``torch.no_grad()`` so it can report loss
    without changing model state. The one-to-many matching setting mirrors the
    historical training objective. Callers decide how strongly to anchor each
    component.
    """
    if logits_mhw.numel() == 0 or not gt_targets_hw:
        return None
    gt_targets = [target.float() for target in gt_targets_hw]
    logits_mhw = logits_mhw.float()
    o2m_matches, o2m_iou = match_predictions_to_gts_greedy_k(
        logits_mhw,
        gt_targets,
        max_matches=None,
        max_per_gt=max_per_gt,
    )
    if not o2m_matches:
        return None

    matched_losses: List[torch.Tensor] = []
    eps = 1e-6
    for gt_idx, pred_idx in o2m_matches:
        if gt_idx < 0 or gt_idx >= len(gt_targets):
            continue
        if pred_idx < 0 or pred_idx >= logits_mhw.shape[0]:
            continue
        logits_hw = logits_mhw[pred_idx]
        target_hw = gt_targets[gt_idx].to(device=logits_hw.device, dtype=logits_hw.dtype)
        loss_bce = F.binary_cross_entropy_with_logits(logits_hw, target_hw, reduction="mean")
        probs_hw = torch.sigmoid(logits_hw)
        dice_num = 2 * (probs_hw * target_hw).sum() + eps
        dice_den = probs_hw.sum() + target_hw.sum() + eps
        loss_dice = 1.0 - dice_num / dice_den
        matched_losses.append(bce_weight * loss_bce + dice_weight * loss_dice)
    if not matched_losses:
        return None

    loss_mask = torch.stack(matched_losses).mean()
    loss_bbox = compute_bbox_l1_loss_from_matches(
        box_preds_n22,
        gt_targets,
        o2m_matches,
    )
    loss_objectness = compute_presence_loss_logits(
        det_scores_logits_n,
        o2m_matches,
        o2m_iou,
        pos_weight=score_weight,
        neg_weight=no_object_weight,
        alpha=0.5,
        use_focal=False,
        focal_alpha=0.25,
        focal_gamma=4.0,
        focal_weight=300.0,
    )
    return MultiGTDetectionLosses(
        mask=loss_mask,
        bbox=loss_bbox,
        objectness=loss_objectness,
    )


def compute_multi_gt_detection_loss(
    logits_mhw: torch.Tensor,
    box_preds_n22: torch.Tensor,
    det_scores_logits_n: torch.Tensor,
    gt_targets_hw: Sequence[torch.Tensor],
    bce_weight: float,
    dice_weight: float,
    bbox_weight: float,
    score_weight: float,
    no_object_weight: float,
    max_per_gt: int = 12,
) -> Optional[torch.Tensor]:
    """Compute the historical combined objective with backward-compatible weights."""

    losses = compute_multi_gt_detection_losses(
        logits_mhw,
        box_preds_n22,
        det_scores_logits_n,
        gt_targets_hw,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        score_weight=score_weight,
        no_object_weight=no_object_weight,
        max_per_gt=max_per_gt,
    )
    if losses is None:
        return None
    return losses.total(mask_weight=2.0, bbox_weight=bbox_weight)


def filter_pose_matches_by_iou(
    matches: Sequence[Tuple[int, int]],
    iou_gt_prediction: torch.Tensor,
    min_iou: float,
) -> List[Tuple[int, int]]:
    """Keep one-to-one pose matches whose detached mask IoU meets ``min_iou``."""

    if not 0.0 <= min_iou <= 1.0:
        raise ValueError(f"min_iou must be in [0, 1], got {min_iou}")
    filtered: List[Tuple[int, int]] = []
    for gt_index, prediction_index in matches:
        if gt_index < 0 or gt_index >= iou_gt_prediction.shape[0]:
            raise IndexError(f"GT index {gt_index} is outside the IoU matrix")
        if prediction_index < 0 or prediction_index >= iou_gt_prediction.shape[1]:
            raise IndexError(f"Prediction index {prediction_index} is outside the IoU matrix")
        if float(iou_gt_prediction[gt_index, prediction_index].detach()) >= min_iou:
            filtered.append((gt_index, prediction_index))
    return filtered


def compute_auxiliary_pose_loss(
    predictions: AuxiliaryPosePredictions,
    pose_targets: Sequence[CADPoseTarget],
    pose_matches: Sequence[Tuple[int, int]],
    pose_config: CADPoseLossConfig,
    *,
    batch_index: int,
) -> Optional[torch.Tensor]:
    """Average geometric pose loss across intermediate layers.

    Candidate assignments come from the final masks and are reused unchanged.
    Pose-quality supervision and its expensive full-placement target are omitted
    from auxiliary layers; the caller applies the configured total aux weight.
    """

    auxiliary_config = replace(pose_config, quality_weight=0.0)
    layer_totals: List[torch.Tensor] = []
    for layer, layer_predictions in zip(predictions.layers, predictions.predictions):
        layer_losses = compute_cad_pose_losses(
            layer_predictions,
            pose_targets,
            pose_matches,
            auxiliary_config,
            batch_index=batch_index,
            compute_expensive_metrics=False,
        )
        if layer_losses is None:
            continue
        if not torch.isfinite(layer_losses.total):
            raise FloatingPointError(
                f"Non-finite auxiliary CAD pose loss at detector layer {layer}"
            )
        layer_totals.append(layer_losses.total)
    return torch.stack(layer_totals).mean() if layer_totals else None


def pad_exemplar_batch(
    exemplars_list: Sequence[Union[torch.Tensor, ExemplarViewBundle]],
    device: torch.device,
    *,
    pose_encoder=None,
    mode: str = "none",
    shuffle_seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return pad_exemplar_view_batch(
        exemplars_list,
        device=device,
        pose_encoder=pose_encoder,
        mode=mode,
        shuffle_seed=shuffle_seed,
    )


def cache_exemplar(
    exemplar: Union[torch.Tensor, ExemplarViewBundle],
) -> Union[torch.Tensor, ExemplarViewBundle]:
    """Detach exemplar features and move them to the shared CPU cache."""

    if isinstance(exemplar, ExemplarViewBundle):
        return exemplar.detach_cpu()
    return exemplar.detach().cpu()


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
    topk = min(8, num_boxes)
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
            for candidate_ref_id in reference_view_id_candidates(ref_id):
                ref_stub = f"{lookup_id}_stl_base_{candidate_ref_id}"
                candidate = reference_dir / f"{ref_stub}.png"
                if candidate.is_file():
                    ref_img_path = candidate
                    break
            if ref_img_path is not None:
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
    min_score: float = 0.15,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask_canvas = image_bgr.copy()
    if detection_scores_n.numel() == 0 or mask_preds_nhw.numel() == 0:
        return mask_canvas

    scores_cpu = detection_scores_n.detach().float().cpu()
    valid_mask = scores_cpu >= min_score
    if valid_mask.sum().item() == 0:
        return mask_canvas
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    valid_scores = scores_cpu[valid_mask]
    topk = min(topk, valid_scores.numel())
    if topk <= 0:
        return mask_canvas

    top_idx = valid_indices[torch.topk(valid_scores, k=topk).indices]
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
    target_overlay = image_bgr.copy()
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
    include_view_metadata: bool = False,
) -> Optional[Union[torch.Tensor, ExemplarViewBundle]]:
    lookup_id = object_id.upper() if upper_object_id else object_id
    feats: List[torch.Tensor] = []
    token_view_indices: List[torch.Tensor] = []
    used_view_ids: List[str] = []
    for ref_id in ref_view_ids:
        reference_pair = resolve_reference_pair(reference_dir, lookup_id, ref_id)
        if reference_pair is None:
            continue
        ref_img_path, ref_mask_path = reference_pair
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
        token_view_indices.append(
            torch.full((exemplar_tokens.shape[1],), len(used_view_ids), dtype=torch.long)
        )
        used_view_ids.append(ref_id)
    if not feats:
        return None
    exemplar_ref = torch.cat(feats, dim=1).to(device)
    if include_view_metadata:
        rotations = load_reference_view_rotations(
            reference_dir / f"{lookup_id}_render_transform.json",
            used_view_ids,
        )
        return ExemplarViewBundle(
            tokens_bnc=exemplar_ref,
            token_view_indices_n=torch.cat(token_view_indices, dim=0),
            view_rotations_v33=rotations,
            view_ids=tuple(used_view_ids),
            object_id=object_id,
        )
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
        default="",
        help="Deprecated unsplit fallback: comma-separated camera dataset roots.",
    )
    parser.add_argument(
        "--dataset_manifest",
        type=str,
        default="",
        help="Canonical versioned CSV manifest containing train/validation/test samples.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="Parent directory used to resolve dataset_path values in --dataset_manifest.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Load --resume_path and evaluate one manifest split without training.",
    )
    parser.add_argument(
        "--eval_split",
        choices=["validation", "test"],
        default="validation",
        help="Manifest split evaluated by --eval_only.",
    )
    parser.add_argument(
        "--validate_before_training",
        action="store_true",
        help="Record validation and conditional pose baselines before the first resumed training epoch.",
    )
    parser.add_argument(
        "--validate_on_train",
        action="store_true",
        help=(
            "Evaluate the unaugmented training rows after each epoch instead of the validation split. "
            "Intended for explicit memorization/overfit diagnostics."
        ),
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="",
        help=(
            "Directory containing frame-level train.csv/val.csv/test.csv, or where they should be created. "
            "Existing complete split files are reused."
        ),
    )
    parser.add_argument(
        "--train_split_csv",
        type=str,
        default="",
        help="CSV of frame_id values to use for training. Overrides --split_dir/train.csv when provided.",
    )
    parser.add_argument(
        "--val_split_csv",
        type=str,
        default="",
        help="CSV of frame_id values to use for validation during training. Overrides --split_dir/val.csv.",
    )
    parser.add_argument(
        "--test_split_csv",
        type=str,
        default="",
        help=(
            "CSV of frame_id values reserved for final test evaluation. "
            "It is checked/logged but not used by the training loop."
        ),
    )
    parser.add_argument(
        "--split_ratios",
        type=str,
        default="0.8,0.1,0.1",
        help="Train,val,test ratios used when --split_dir needs to create split CSVs. Values are normalized.",
    )
    parser.add_argument(
        "--recreate_splits",
        action="store_true",
        help="Overwrite train.csv/val.csv/test.csv under --split_dir.",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="",
        help="Path to reference renders.",
    )
    #"0,3,6,9"
    parser.add_argument("--ref_view_ids", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11", help="Reference view ids to use.")
    parser.add_argument("--max_side_length", type=int, default=1008)
    parser.add_argument("--no_square", action="store_true", help="Disable square resizing in encoder.")
    parser.add_argument("--num_points_approx", type=int, default=25)
    parser.add_argument(
        "--exemplar_view_mode",
        choices=EXEMPLAR_VIEW_MODES,
        default="none",
        help=(
            "Reference-view experiment: exact baseline, camera pose, deterministic camera "
            "shuffle, zero-camera control, or learned view-ID control."
        ),
    )
    parser.add_argument(
        "--exemplar_view_shuffle_seed",
        type=int,
        default=None,
        help="Seed for shuffled-camera assignments (default: --seed).",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=13)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bce_weight", type=float, default=2.0)
    parser.add_argument("--dice_weight", type=float, default=2.0)
    parser.add_argument("--bbox_weight", type=float, default=1.0)
    parser.add_argument("--score_weight", type=float, default=0.3)
    parser.add_argument("--no_object_weight", type=float, default=0.45)
    parser.add_argument(
        "--enable_pose",
        action="store_true",
        help="Train from validated Perseve point-set pose-v2 or legacy pose-v1 sidecars.",
    )
    parser.add_argument(
        "--pose_stage",
        choices=["head", "joint", "joint_lite"],
        default="head",
        help=(
            "Train only the pose head, fully adapt fusion/detector/mask modules, or use pose-first "
            "joint-lite adaptation with a frozen mask decoder and lower shared-module LR."
        ),
    )
    parser.add_argument(
        "--pose_train_min_match_iou",
        type=float,
        default=0.0,
        help="Minimum mask IoU for a matched candidate to receive pose loss (default: 0, historical behavior).",
    )
    parser.add_argument(
        "--pose_eval_min_match_iou",
        type=float,
        default=0.0,
        help="Also report conditional pose metrics above this match IoU (default: 0 disables the extra row).",
    )
    parser.add_argument(
        "--joint_shared_lr_scale",
        type=float,
        default=0.1,
        help="Fusion/detector LR multiplier relative to --lr in pose_stage=joint_lite.",
    )
    parser.add_argument(
        "--enable_cad_prompt",
        action="store_true",
        help=(
            "Apply pose-only cross-attention from detector queries to the padded CAD "
            "exemplar tokens; segmentation continues to consume the original queries."
        ),
    )
    parser.add_argument(
        "--pose_prompt_lr_scale",
        type=float,
        default=1.0,
        help="CAD prompt-adapter LR multiplier relative to --lr.",
    )
    parser.add_argument(
        "--pose_deep_supervision",
        action="store_true",
        help=(
            "Apply the shared pose head to selected intermediate detector layers using "
            "the final mask assignment. Inference remains final-layer only."
        ),
    )
    parser.add_argument(
        "--pose_aux_layers",
        type=str,
        default="3,4,5",
        help="One-based detector layers receiving auxiliary pose supervision.",
    )
    parser.add_argument(
        "--pose_aux_loss_weight",
        type=float,
        default=0.5,
        help="Total weight on the mean auxiliary pose loss.",
    )
    parser.add_argument(
        "--joint_bbox_weight",
        type=float,
        default=0.25,
        help="Bounding-box anchor weight in pose_stage=joint_lite.",
    )
    parser.add_argument(
        "--joint_objectness_weight",
        type=float,
        default=0.25,
        help="Objectness anchor weight in pose_stage=joint_lite.",
    )
    parser.add_argument(
        "--joint_mask_weight",
        type=float,
        default=0.10,
        help="Mask anchor weight in pose_stage=joint_lite; applies after BCE/Dice weighting.",
    )
    parser.add_argument("--pose_weight", type=float, default=1.0)
    parser.add_argument("--pose_center_weight", type=float, default=1.0)
    parser.add_argument("--pose_depth_weight", type=float, default=1.0)
    parser.add_argument("--pose_rotation_weight", type=float, default=1.0)
    parser.add_argument("--pose_full_set_weight", type=float, default=0.0)
    parser.add_argument("--pose_quality_weight", type=float, default=1.0)
    parser.add_argument("--log_depth_mean", type=float, default=0.0)
    parser.add_argument("--log_depth_std", type=float, default=1.0)
    parser.add_argument("--centroid_tolerance", type=float, default=0.1)
    parser.add_argument("--point_set_tolerance", type=float, default=0.1)
    parser.add_argument("--centroid_soft_width", type=float, default=0.02)
    parser.add_argument("--point_set_soft_width", type=float, default=0.02)
    parser.add_argument("--point_loss_beta", type=float, default=0.01)
    parser.add_argument("--point_distance_chunk_size", type=int, default=512)
    # Legacy v1 quality flags remain accepted for checkpoint/config compatibility.
    parser.add_argument("--rotation_tolerance_deg", type=float, default=5.0)
    parser.add_argument("--translation_tolerance", type=float, default=0.1)
    parser.add_argument("--rotation_soft_width_deg", type=float, default=1.0)
    parser.add_argument("--translation_soft_width", type=float, default=0.02)
    parser.add_argument(
        "--absolute_translation_tolerance",
        action="store_true",
        help="Interpret translation tolerance/width in metres instead of object-diagonal-normalized units.",
    )
    parser.add_argument(
        "--matches_per_gt",
        type=int,
        default=1,
        help="Max number of predictions to assign per GT during greedy matching.",
    )
    parser.add_argument("--det_filter", type=float, default=0.0)
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.5,
        help="IoU threshold for mask NMS used in metrics/debug (<=0 disables NMS).",
    )
    parser.add_argument("--grad_accum", type=int, default=12)
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=0.0,
        help="Clip the global gradient norm before optimizer steps; <=0 disables clipping.",
    )
    parser.add_argument("--log_every", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--save_debug_every",
        type=int,
        default=20,
        help="Save debug collage every N batches (0 disables).",
    )
    parser.add_argument("--output_dir", type=str, default="runs/finetune_exemplar")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default="")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy, and PyTorch.",
    )
    parser.add_argument(
        "--init_path",
        type=str,
        default="",
        help=(
            "Segmentation-only finetune checkpoint used to initialize detector modules. "
            "Unlike --resume_path, starts at epoch 1 with fresh optimizer, pose head, and view adapter."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="",
        help="Path to a trusted local finetune checkpoint (.pth) to resume from.",
    )
    parser.add_argument(
        "--transfer_path",
        type=str,
        default="",
        help=(
            "Pose finetune checkpoint used to initialize all trained model modules for a new run. "
            "Unlike --resume_path, permits a new manifest and resets optimizer, epoch, and step state."
        ),
    )
    parser.add_argument(
        "--no_resume_optimizer",
        action="store_true",
        help="Do not load optimizer state when resuming.",
    )
    parser.add_argument(
        "--resume_in_place",
        action="store_true",
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


def resolve_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    """Return the owning run directory for old and new checkpoint layouts."""

    parent = checkpoint_path.expanduser().resolve().parent
    return parent.parent if parent.name == "checkpoints" else parent


def initialize_metrics_log(metrics_path: Path) -> None:
    """Create a metrics log or migrate an older compatible header by name."""

    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        with metrics_path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writeheader()
        return

    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        existing_fields = tuple(reader.fieldnames or ())
        if existing_fields == METRIC_FIELDS:
            return
        if not existing_fields or len(existing_fields) != len(set(existing_fields)):
            raise ValueError(f"Metrics CSV has an invalid header: {metrics_path}")
        unknown_fields = sorted(set(existing_fields) - set(METRIC_FIELDS))
        if unknown_fields:
            raise ValueError(
                f"Metrics CSV has unsupported columns {unknown_fields}: {metrics_path}"
            )
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(
            f"Metrics CSV contains rows wider than its header and cannot be migrated safely: {metrics_path}"
        )

    temporary = metrics_path.with_name(f".{metrics_path.name}.schema-migration.tmp")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
        os.replace(temporary, metrics_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_metric(metrics_path: Path, **values: object) -> None:
    row = {field: values.get(field, "") for field in METRIC_FIELDS}
    with metrics_path.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writerow(row)


def finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def surface_distance_percentile(
    chunks: Sequence[torch.Tensor], quantile: float = 0.95
) -> float:
    """Compute a true percentile over every retained point distance."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    finite_chunks = [chunk.detach().float().cpu().flatten() for chunk in chunks if chunk.numel()]
    if not finite_chunks:
        return float("nan")
    values = torch.cat(finite_chunks)
    values = values[torch.isfinite(values)]
    return float(torch.quantile(values, quantile)) if values.numel() else float("nan")


def pose_iou_phase(base_phase: str, min_iou: float) -> str:
    """Build a stable CSV phase name for conditional pose evaluation."""

    return f"{base_phase}_iou_{int(round(min_iou * 100)):03d}"


def append_pose_metric_rows(
    metrics_path: Path,
    *,
    phase: str,
    epoch: int,
    global_step: int,
    batch_step: int,
    metrics: Dict[str, float],
    conditional_metrics: Optional[Dict[str, float]],
) -> None:
    """Write unconditional and optional IoU-conditioned pose metric rows."""

    append_metric(
        metrics_path,
        phase=phase,
        epoch=epoch,
        global_step=global_step,
        batch_step=batch_step,
        samples=int(metrics.get("count", 0)),
        **{key: value for key, value in metrics.items() if key != "count"},
    )
    if conditional_metrics is None:
        return
    threshold = float(conditional_metrics["pose_match_iou_threshold"])
    append_metric(
        metrics_path,
        phase=pose_iou_phase(phase, threshold),
        epoch=epoch,
        global_step=global_step,
        batch_step=batch_step,
        samples=int(conditional_metrics.get("count", 0)),
        **{key: value for key, value in conditional_metrics.items() if key != "count"},
    )


def restore_finetune_checkpoint(
    checkpoint: Dict[str, object],
    detmodel,
    optimizer: Optional[torch.optim.Optimizer],
    load_optimizer: bool = True,
    exemplar_view_mode: str = "none",
) -> Dict[str, object]:
    """Restore a trusted, already-loaded fine-tuning checkpoint."""

    required = {
        "image_exemplar_fusion": detmodel.image_exemplar_fusion,
        "exemplar_detector": detmodel.exemplar_detector,
        "exemplar_segmentation": detmodel.exemplar_segmentation,
    }
    for key, module in required.items():
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing '{key}' state.")
        module.load_state_dict(checkpoint[key])
    checkpoint_args = checkpoint.get("args") or {}
    checkpoint_view_mode = str(checkpoint_args.get("exemplar_view_mode", "none"))
    checkpoint_pose_enabled = bool(checkpoint_args.get("enable_pose", False))
    if checkpoint_view_mode != exemplar_view_mode and (
        checkpoint_view_mode != "none" or checkpoint_pose_enabled
    ):
        raise ValueError(
            "Checkpoint exemplar-view mode "
            f"{checkpoint_view_mode!r} is incompatible with requested mode {exemplar_view_mode!r}"
        )
    if "exemplar_view_pose_encoder" in checkpoint:
        checkpoint_version = int(checkpoint.get("exemplar_view_pose_architecture_version", 1))
        if checkpoint_version != detmodel.exemplar_view_pose_encoder.architecture_version:
            raise ValueError(
                "Exemplar-view adapter checkpoint architecture version "
                f"{checkpoint_version} is incompatible with model version "
                f"{detmodel.exemplar_view_pose_encoder.architecture_version}"
            )
        expected_config = detmodel.exemplar_view_pose_encoder.architecture_config()
        checkpoint_config = checkpoint.get("exemplar_view_pose_architecture_config")
        if checkpoint_config is not None and checkpoint_config != expected_config:
            raise ValueError("Checkpoint exemplar-view adapter architecture config is incompatible")
        detmodel.exemplar_view_pose_encoder.load_state_dict(
            checkpoint["exemplar_view_pose_encoder"]
        )
    elif checkpoint_view_mode != "none":
        raise KeyError("Checkpoint is missing its trained exemplar-view adapter state")
    if "cad_pose_head" in checkpoint:
        checkpoint_version = int(checkpoint.get("cad_pose_head_architecture_version", 1))
        supported_versions = {
            detmodel.cad_pose_head.architecture_version,
            detmodel.cad_pose_head.legacy_promptless_architecture_version,
        }
        if checkpoint_version in supported_versions:
            checkpoint_config = checkpoint.get("cad_pose_head_architecture_config")
            if (
                checkpoint_version == detmodel.cad_pose_head.architecture_version
                and checkpoint_config is not None
                and checkpoint_config != detmodel.cad_pose_head.architecture_config()
            ):
                raise ValueError("Checkpoint CAD pose-head architecture config is incompatible")
            migrated = detmodel.cad_pose_head.load_checkpoint_state_dict(
                checkpoint["cad_pose_head"],
                checkpoint_version,
            )
            if migrated:
                warnings.warn(
                    "Migrated promptless CAD pose head v3 to v4; the new pose-only "
                    "CAD prompt adapter starts from fresh initialization."
                )
        elif checkpoint_pose_enabled or checkpoint.get("pose_config") is not None:
            raise ValueError(
                "CAD pose-head checkpoint architecture version "
                f"{checkpoint_version} is incompatible with model version "
                f"{detmodel.cad_pose_head.architecture_version}; retrain the pose head"
            )
        else:
            warnings.warn("Ignoring an incompatible untrained pose head from a segmentation checkpoint")
    if optimizer is not None and load_optimizer and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            warnings.warn(f"Optimizer state is incompatible with the selected training stage; starting fresh: {exc}")
    return checkpoint


def load_finetune_checkpoint(
    checkpoint_path: Path,
    detmodel,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    load_optimizer: bool = True,
    exemplar_view_mode: str = "none",
) -> Dict[str, object]:
    """Load and restore a trusted fine-tuning checkpoint from disk."""

    # Fine-tuning checkpoints are full, trusted training-state files produced
    # by this script (modules, optimizer, arguments, provenance, and NumPy
    # scalar configuration values), not tensor-only weight bundles. PyTorch
    # 2.6+ defaults to weights_only=True, which rejects that metadata.
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    return restore_finetune_checkpoint(
        checkpoint,
        detmodel,
        optimizer,
        load_optimizer=load_optimizer,
        exemplar_view_mode=exemplar_view_mode,
    )


def load_detector_initialization_checkpoint(
    checkpoint_path: Path,
    detmodel,
    device: torch.device,
) -> Dict[str, object]:
    """Load segmentation modules while deliberately resetting pose experiment state."""

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args") or {}
    if bool(checkpoint_args.get("enable_pose", False)) or checkpoint.get("pose_config") is not None:
        raise ValueError(
            "--init_path must be a segmentation-only checkpoint; use --resume_path to "
            "continue a pose run"
        )
    checkpoint_mode = str(checkpoint_args.get("exemplar_view_mode", "none"))
    if checkpoint_mode != "none":
        raise ValueError("--init_path checkpoint contains a trained exemplar-view adapter")
    required = {
        "image_exemplar_fusion": detmodel.image_exemplar_fusion,
        "exemplar_detector": detmodel.exemplar_detector,
        "exemplar_segmentation": detmodel.exemplar_segmentation,
    }
    for key, module in required.items():
        if key not in checkpoint:
            raise KeyError(f"Initialization checkpoint missing '{key}' state")
        module.load_state_dict(checkpoint[key])
    return checkpoint


def build_finetune_checkpoint(
    detmodel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    epoch: int,
    global_step: int,
    batch_step: int,
    pose_config: Optional[CADPoseLossConfig],
    manifest_checksum: str,
    catalog_checksums: set,
    dataset_meta_checksums: set,
    annotation_checksums: set,
    symmetry_pipeline_versions: set,
    point_set_checksums: set,
    sampling_pipeline_versions: set,
    sampling_parameter_checksums: set,
    schema_checksums: set,
    schema_versions: set,
) -> Dict[str, object]:
    annotation_digest = checksum_set_digest(annotation_checksums)
    return {
        "epoch": epoch,
        "global_step": global_step,
        "batch_step": batch_step,
        "image_exemplar_fusion": detmodel.image_exemplar_fusion.state_dict(),
        "exemplar_detector": detmodel.exemplar_detector.state_dict(),
        "exemplar_segmentation": detmodel.exemplar_segmentation.state_dict(),
        "cad_pose_head": detmodel.cad_pose_head.state_dict(),
        "cad_pose_head_architecture_version": detmodel.cad_pose_head.architecture_version,
        "cad_pose_head_architecture_config": detmodel.cad_pose_head.architecture_config(),
        "exemplar_view_pose_encoder": detmodel.exemplar_view_pose_encoder.state_dict(),
        "exemplar_view_pose_architecture_version": (
            detmodel.exemplar_view_pose_encoder.architecture_version
        ),
        "exemplar_view_pose_architecture_config": (
            detmodel.exemplar_view_pose_encoder.architecture_config()
        ),
        "optimizer": optimizer.state_dict(),
        "pose_config": None if pose_config is None else pose_config.__dict__,
        "manifest_sha256": manifest_checksum,
        "catalog_checksums": sorted(catalog_checksums),
        "dataset_meta_checksums": sorted(dataset_meta_checksums),
        "annotation_checksums": sorted(annotation_checksums),
        "annotation_checksum_sha256": annotation_digest,
        "symmetry_pipeline_versions": sorted(symmetry_pipeline_versions),
        "point_set_checksums": sorted(point_set_checksums),
        "sampling_pipeline_versions": sorted(sampling_pipeline_versions),
        "sampling_parameter_checksums": sorted(sampling_parameter_checksums),
        "schema_checksums": sorted(schema_checksums),
        "schema_versions": sorted(schema_versions),
        "args": vars(args),
    }


def checksum_set_digest(values: Iterable[str]) -> str:
    """Return a stable digest for an unordered collection of checksums."""

    return hashlib.sha256("\n".join(sorted(set(values))).encode("utf-8")).hexdigest()


def validate_resume_manifest_checksum(
    checkpoint: Dict[str, object], current_manifest_checksum: str
) -> None:
    """Reject resumes that change the persisted manifest or split assignment."""

    if "manifest_sha256" not in checkpoint:
        return
    expected = str(checkpoint.get("manifest_sha256") or "")
    if expected != current_manifest_checksum:
        raise ValueError("Resume checkpoint manifest checksum differs from the current manifest")


def validate_pose_resume_provenance(
    checkpoint: Dict[str, object],
    *,
    catalog_checksums: set,
    dataset_meta_checksums: set,
    annotation_checksums: set,
    symmetry_pipeline_versions: set,
    point_set_checksums: set,
    sampling_pipeline_versions: set,
    sampling_parameter_checksums: set,
    schema_checksums: set,
    schema_versions: set,
) -> None:
    """Reject a pose resume when any persisted data contract has changed."""

    comparisons = (
        ("catalog_checksums", catalog_checksums, "object catalog"),
        ("dataset_meta_checksums", dataset_meta_checksums, "dataset metadata"),
        ("symmetry_pipeline_versions", symmetry_pipeline_versions, "symmetry pipeline"),
        ("point_set_checksums", point_set_checksums, "point-set artifacts"),
        ("sampling_pipeline_versions", sampling_pipeline_versions, "point-set sampling version"),
        (
            "sampling_parameter_checksums",
            sampling_parameter_checksums,
            "point-set sampling parameters",
        ),
        ("schema_checksums", schema_checksums, "pose schema files"),
        ("schema_versions", schema_versions, "pose schema version"),
    )
    for field, current_values, label in comparisons:
        expected_values = set(checkpoint.get(field, []) or [])
        if expected_values and expected_values != set(current_values):
            raise ValueError(f"Resume checkpoint {label} differs from the current dataset")

    current_annotation_digest = checksum_set_digest(annotation_checksums)
    expected_annotation_digest = str(checkpoint.get("annotation_checksum_sha256") or "")
    if not expected_annotation_digest:
        expected_annotations = set(checkpoint.get("annotation_checksums", []) or [])
        if expected_annotations:
            expected_annotation_digest = checksum_set_digest(expected_annotations)
    if expected_annotation_digest and expected_annotation_digest != current_annotation_digest:
        raise ValueError("Resume checkpoint pose annotations differ from the current dataset")


def _capture_training_state(module: torch.nn.Module) -> List[Tuple[torch.nn.Module, bool]]:
    return [(submodule, submodule.training) for submodule in module.modules()]


def _restore_training_state(state: List[Tuple[torch.nn.Module, bool]]) -> None:
    for submodule, was_training in state:
        submodule.train(was_training)


def run_exemplar_eval(
    detmodel,
    config: EvalConfig,
    device: torch.device,
) -> Tuple[float, int, float, float]:
    ref_view_ids = parse_ref_view_ids(config.ref_view_ids)
    if not ref_view_ids:
        raise ValueError("No eval reference view ids resolved.")

    reference_dir = Path(config.reference_dir).expanduser().resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)
    split_path = Path(config.split_csv).expanduser().resolve() if config.split_csv else None
    if split_path is not None and not split_path.is_file():
        raise FileNotFoundError(split_path)

    all_entries: List[Dict[str, str]] = []
    if config.dataset_entries is not None:
        all_entries = list(config.dataset_entries)
    else:
        dataset_roots = normalize_dataset_roots(config.dataset_root)
        if not dataset_roots:
            raise ValueError("No eval dataset roots or manifest entries provided.")
        for root, multiplier, use_filter in dataset_roots:
            _, cur_entries = collect_multi_object_samples(root)
            cur_entries = unique_object_entries(cur_entries)
            if use_filter:
                filter_path = resolve_dataset_filter_path(root)
                if filter_path is not None:
                    cur_entries = apply_dataset_filter_entries(cur_entries, filter_path)
            if split_path is not None:
                cur_entries = apply_dataset_filter_entries(cur_entries, split_path)
            cur_entries = apply_dataset_multiplier(cur_entries, multiplier)
            all_entries.extend(cur_entries)
    if not all_entries:
        raise RuntimeError("No eval dataset entries found.")

    if config.shuffle:
        random.shuffle(all_entries)

    ref_cache: Dict[str, Union[torch.Tensor, ExemplarViewBundle]] = {}
    seg_cache: Dict[str, np.ndarray] = {}
    mapping_cache: Dict[str, Dict[str, List[Tuple[int, ...]]]] = {}
    total_iou_sum = 0.0
    total_iou_count = 0
    total_correct = 0
    total_loss_sum = 0.0
    total_loss_count = 0
    batch_step = 0
    pq_iou_threshold = 0.5
    pq_score_thresholds = [round(0.10 + 0.01 * idx, 2) for idx in range(30)]
    pq_stats: Dict[float, Dict[str, float]] = {
        thresh: {"sum_iou": 0.0, "tp": 0, "fp": 0, "fn": 0} for thresh in pq_score_thresholds
    }

    training_state = _capture_training_state(detmodel)
    detmodel.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(all_entries), config.batch_size):
                subset = all_entries[start : start + config.batch_size]
                prepared: List[Dict[str, object]] = []
                for entry in subset:
                    try:
                        image_bgr = load_bgr(entry["rgb_path"])
                    except FileNotFoundError:
                        continue
                    try:
                        mapping_path = Path(entry["inst_path"]).with_name(
                            f"instance_segmentation_mapping_{entry['frame_id']}.json"
                        )
                        obj_id = entry["object_id"]
                        gt_masks = load_instance_masks_for_object(
                            entry["inst_path"],
                            str(mapping_path),
                            obj_id,
                            seg_cache=seg_cache,
                            mapping_cache=mapping_cache,
                        )
                    except FileNotFoundError:
                        continue
                    if obj_id is None or not gt_masks:
                        continue
                    gt_masks = [mask for mask in gt_masks if int(mask.sum()) >= 400]
                    if not gt_masks:
                        continue
                    gt_masks = select_top_gt_masks(gt_masks, max_instances=None)

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
                            include_view_metadata=config.exemplar_view_mode != "none",
                        )
                        if exemplar_ref is None:
                            continue
                        ref_cache[obj_id] = cache_exemplar(exemplar_ref)

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
                batch_losses: List[float] = []
                batch_correct = 0
                for _, idxs in group_map.items():
                    img_batch = torch.cat([prepared[i]["img_tensor"] for i in idxs], dim=0)
                    encoded_img = detmodel.image_encoder(img_batch)
                    encoded_image_features_list = detmodel.image_projection.v3_projection(encoded_img)

                    exemplars_list = [prepared[i]["exemplar_ref"] for i in idxs]
                    exemplar_batch, padding_mask = pad_exemplar_batch(
                        exemplars_list,
                        device=device,
                        pose_encoder=detmodel.exemplar_view_pose_encoder,
                        mode=config.exemplar_view_mode,
                        shuffle_seed=config.exemplar_view_shuffle_seed,
                    )

                    mask_preds, box_preds, det_scores, det_scores_logits, _ = generate_detections_train(
                        detmodel,
                        encoded_image_features_list,
                        exemplar_batch,
                        detection_filter_threshold=config.det_filter,
                        exemplar_padding_mask_bn=padding_mask,
                    )
                    if mask_preds.shape[1] == 0:
                        for data_idx in idxs:
                            preencode_hw = prepared[data_idx]["preencode_hw"]
                            gt_down_list = build_gt_down_list(
                                prepared[data_idx]["gt_masks"],
                                preencode_hw,
                                mask_preds.shape[-2:],
                                device,
                            )
                            update_pq_accumulators(pq_stats, [], gt_down_list, None, pq_iou_threshold)
                        continue

                    for local_idx, data_idx in enumerate(idxs):
                        preencode_hw = prepared[data_idx]["preencode_hw"]
                        scores = det_scores[local_idx]
                        if scores.numel() == 0:
                            gt_down_list = build_gt_down_list(
                                prepared[data_idx]["gt_masks"],
                                preencode_hw,
                                mask_preds.shape[-2:],
                                device,
                            )
                            update_pq_accumulators(pq_stats, [], gt_down_list, None, pq_iou_threshold)
                            continue

                        gt_down_list = build_gt_down_list(
                            prepared[data_idx]["gt_masks"],
                            preencode_hw,
                            mask_preds.shape[-2:],
                            device,
                        )
                        if not gt_down_list:
                            continue

                        val_loss = compute_multi_gt_detection_loss(
                            mask_preds[local_idx].float(),
                            box_preds[local_idx],
                            det_scores_logits[local_idx],
                            gt_down_list,
                            bce_weight=config.bce_weight,
                            dice_weight=config.dice_weight,
                            bbox_weight=config.bbox_weight,
                            score_weight=config.score_weight,
                            no_object_weight=config.no_object_weight,
                        )
                        if val_loss is not None:
                            val_loss_float = float(val_loss.item())
                            batch_losses.append(val_loss_float)
                            total_loss_sum += val_loss_float
                            total_loss_count += 1

                        boxes_nms, masks_nms, scores_nms = apply_mask_nms(
                            box_preds[local_idx],
                            mask_preds[local_idx],
                            scores,
                            iou_threshold=config.nms_iou,
                        )
                        if scores_nms.numel() == 0:
                            update_pq_accumulators(pq_stats, [], gt_down_list, None, pq_iou_threshold)
                        else:
                            pred_masks_list = [(masks_nms[k] > 0) for k in range(masks_nms.shape[0])]
                            update_pq_accumulators(
                                pq_stats,
                                pred_masks_list,
                                gt_down_list,
                                pred_scores=scores_nms,
                                iou_threshold=pq_iou_threshold,
                            )

                        topk = min(len(gt_down_list) * max(1, config.matches_per_gt), scores.numel())
                        top_idx = torch.topk(scores, k=topk).indices
                        logits_sel = mask_preds[local_idx, top_idx].float()
                        matches, iou = match_predictions_to_gts_greedy_k(
                            logits_sel,
                            gt_down_list,
                            max_matches=None,
                            max_per_gt=config.matches_per_gt,
                        )
                        if not matches or iou is None:
                            continue
                        matched_ious = [float(iou[g, p].item()) for g, p in matches]
                        avg_iou = sum(matched_ious) / len(matched_ious)
                        batch_ious.append(avg_iou)
                        total_iou_sum += avg_iou
                        total_iou_count += 1
                        if avg_iou > 0.7:
                            batch_correct += 1
                            total_correct += 1

                if batch_ious or batch_losses:
                    avg_iou = sum(batch_ious) / max(1, len(batch_ious)) if batch_ious else 0.0
                    correct_rate = batch_correct / max(1, len(batch_ious))
                    avg_loss = sum(batch_losses) / max(1, len(batch_losses)) if batch_losses else float("nan")
                    loss_part = f" val_loss={avg_loss:.4f}" if np.isfinite(avg_loss) else ""
                    print(
                        f"[eval] step={batch_step}{loss_part} avg_iou={avg_iou:.4f} "
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
        overall_loss = total_loss_sum / total_loss_count if total_loss_count > 0 else float("nan")
        for score_threshold in sorted(pq_stats.keys()):
            stats = pq_stats[score_threshold]
            denom = stats["tp"] + 0.5 * stats["fp"] + 0.5 * stats["fn"]
            if denom > 0:
                pq = stats["sum_iou"] / denom
            else:
                pq = 0.0
            print(
                f"[eval] PQ@score>={score_threshold:.2f}={pq:.4f} "
                f"tp={int(stats['tp'])} fp={int(stats['fp'])} fn={int(stats['fn'])}"
            )
        return overall_avg, total_iou_count, overall_correct, overall_loss
    for score_threshold in sorted(pq_stats.keys()):
        stats = pq_stats[score_threshold]
        denom = stats["tp"] + 0.5 * stats["fp"] + 0.5 * stats["fn"]
        if denom > 0:
            pq = stats["sum_iou"] / denom
        else:
            pq = 0.0
        print(
            f"[eval] PQ@score>={score_threshold:.2f}={pq:.4f} "
            f"tp={int(stats['tp'])} fp={int(stats['fp'])} fn={int(stats['fn'])}"
        )
    overall_loss = total_loss_sum / total_loss_count if total_loss_count > 0 else float("nan")
    return 0.0, 0, 0.0, overall_loss


def run_cad_pose_eval(
    detmodel,
    entries: Sequence[Dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
    pose_config: CADPoseLossConfig,
    *,
    calibrate_temperature: bool,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """Evaluate all pose matches and an optional IoU-conditioned subset in one pass."""

    reference_dir = Path(args.reference_dir).expanduser().resolve()
    ref_view_ids = parse_ref_view_ids(args.ref_view_ids)
    frame_entries = unique_frame_entries(entries)
    pose_cache: Dict[Tuple[str, str], object] = {}
    ref_cache: Dict[str, Union[torch.Tensor, ExemplarViewBundle]] = {}
    metric_sums: Dict[str, float] = defaultdict(float)
    metric_counts: Dict[str, int] = defaultdict(int)
    conditional_metric_sums: Dict[str, float] = defaultdict(float)
    conditional_metric_counts: Dict[str, int] = defaultdict(int)
    surface_distance_chunks: List[torch.Tensor] = []
    conditional_surface_distance_chunks: List[torch.Tensor] = []
    evaluated_fields = (
        "mean_surface_distance_norm",
        "centroid_error_cm",
        "rotation_error_deg",
        "translation_error_cm",
        "center_error_norm",
        "depth_error_m",
        "pose_success_rate",
        "accuracy_5deg_5cm",
        "accuracy_10deg_10cm",
    )
    calibration_logits: List[torch.Tensor] = []
    calibration_targets: List[torch.Tensor] = []
    conditional_calibration_logits: List[torch.Tensor] = []
    conditional_calibration_targets: List[torch.Tensor] = []
    total_count = 0
    conditional_count = 0
    eligible_count = 0
    conditional_threshold = float(args.pose_eval_min_match_iou)

    def accumulate_evaluation(
        evaluation: PoseEvaluation,
        sums: Dict[str, float],
        counts: Dict[str, int],
        distance_chunks: List[torch.Tensor],
    ) -> None:
        for field in evaluated_fields:
            value = float(getattr(evaluation, field))
            if np.isfinite(value):
                sums[field] += value * evaluation.count
                counts[field] += evaluation.count
        if evaluation.surface_distances_norm is not None:
            distance_chunks.append(evaluation.surface_distances_norm)

    training_state = _capture_training_state(detmodel)
    detmodel.eval()
    try:
        with torch.no_grad():
            for entry in frame_entries:
                image_bgr = load_bgr(entry["rgb_path"])
                camera_root = Path(entry["inst_path"]).parent
                cache_key = (str(camera_root), entry["frame_id"])
                sample = pose_cache.get(cache_key)
                if sample is None:
                    sample = load_perseve_pose_sample(camera_root, entry["frame_id"], validate_pixels=True)
                    pose_cache[cache_key] = sample
                grouped: Dict[str, List[object]] = defaultdict(list)
                for instance in sample.frame.eligible_instances():
                    grouped[instance.cad_id].append(instance)
                for cad_id, instances in sorted(grouped.items()):
                    eligible_count += len(instances)
                    masks = [instance_mask_rgba(Path(entry["inst_path"]), instance).astype(np.float32) for instance in instances]
                    if not masks:
                        continue
                    if cad_id not in ref_cache:
                        exemplar = build_exemplar_tokens_for_object(
                            detmodel,
                            cad_id,
                            reference_dir,
                            ref_view_ids,
                            args.max_side_length,
                            not args.no_square,
                            args.num_points_approx,
                            device,
                            include_view_metadata=args.exemplar_view_mode != "none",
                        )
                        if exemplar is None:
                            continue
                        ref_cache[cad_id] = cache_exemplar(exemplar)
                    image_tensor = detmodel.image_encoder.prepare_image(
                        image_bgr,
                        max_side_length=args.max_side_length,
                        use_square_sizing=not args.no_square,
                    )
                    encoded = detmodel.image_encoder(image_tensor)
                    features = detmodel.image_projection.v3_projection(encoded)
                    pre_h, pre_w = image_tensor.shape[-2:]
                    model_dtype = image_tensor.dtype
                    raw_k = torch.as_tensor(sample.frame.intrinsics, device=device, dtype=model_dtype)
                    adjusted_k = adjust_intrinsics_for_resize_and_pad(
                        raw_k, sample.frame.image_size_wh, (pre_w, pre_h)
                    ).unsqueeze(0)
                    dimensions = torch.as_tensor(
                        instances[0].dimensions_m, device=device, dtype=model_dtype
                    ).unsqueeze(0)
                    effective_centroid = torch.as_tensor(
                        effective_surface_centroid_m(instances[0], sample.catalog[cad_id]),
                        device=device,
                        dtype=model_dtype,
                    ).unsqueeze(0)
                    exemplar_batch, exemplar_padding_mask = pad_exemplar_batch(
                        [ref_cache[cad_id]],
                        device=device,
                        pose_encoder=detmodel.exemplar_view_pose_encoder,
                        mode=args.exemplar_view_mode,
                        shuffle_seed=args.exemplar_view_shuffle_seed,
                    )
                    outputs = generate_detections_train(
                        detmodel,
                        features,
                        exemplar_batch,
                        detection_filter_threshold=args.det_filter,
                        exemplar_padding_mask_bn=exemplar_padding_mask,
                        cad_dimensions_m_b3=dimensions,
                        cad_effective_surface_centroid_m_b3=effective_centroid,
                        adjusted_intrinsics_b33=adjusted_k,
                        model_image_size_wh=(pre_w, pre_h),
                        return_pose=True,
                    )
                    mask_predictions, _, detection_scores, _, _, pose_predictions = outputs
                    logits_nhw = mask_predictions[0].float()
                    scores_n = detection_scores[0]
                    retained = mask_nms_indices(logits_nhw, scores_n, args.nms_iou)
                    logits_nhw = logits_nhw[retained]
                    pose_predictions = pose_predictions.index_candidates(retained)
                    gt_targets = build_gt_down_list(
                        masks,
                        (pre_h, pre_w),
                        logits_nhw.shape[-2:],
                        device,
                    )
                    matches, match_iou = match_pose_predictions_one_to_one(logits_nhw, gt_targets)
                    conditional_matches = (
                        filter_pose_matches_by_iou(matches, match_iou, conditional_threshold)
                        if conditional_threshold > 0.0
                        else []
                    )
                    pose_targets = [
                        make_pose_target(
                            instance,
                            sample.catalog[cad_id],
                            raw_k,
                            sample.frame.image_size_wh,
                            (pre_w, pre_h),
                        )
                        for instance in instances
                    ]
                    evaluation = evaluate_pose_matches(
                        pose_predictions,
                        pose_targets,
                        matches,
                        centroid_tolerance=pose_config.centroid_tolerance,
                        point_set_tolerance=pose_config.point_set_tolerance,
                        point_distance_chunk_size=pose_config.point_distance_chunk_size,
                        rotation_tolerance_rad=pose_config.rotation_tolerance_rad,
                        translation_tolerance=pose_config.translation_tolerance,
                        normalize_translation_error=pose_config.normalize_translation_error,
                    )
                    if evaluation.count == 0:
                        continue
                    total_count += evaluation.count
                    accumulate_evaluation(
                        evaluation,
                        metric_sums,
                        metric_counts,
                        surface_distance_chunks,
                    )
                    if conditional_matches:
                        conditional_evaluation = evaluate_pose_matches(
                            pose_predictions,
                            pose_targets,
                            conditional_matches,
                            centroid_tolerance=pose_config.centroid_tolerance,
                            point_set_tolerance=pose_config.point_set_tolerance,
                            point_distance_chunk_size=pose_config.point_distance_chunk_size,
                            rotation_tolerance_rad=pose_config.rotation_tolerance_rad,
                            translation_tolerance=pose_config.translation_tolerance,
                            normalize_translation_error=pose_config.normalize_translation_error,
                        )
                        conditional_count += conditional_evaluation.count
                        accumulate_evaluation(
                            conditional_evaluation,
                            conditional_metric_sums,
                            conditional_metric_counts,
                            conditional_surface_distance_chunks,
                        )
                    conditional_match_set = set(conditional_matches)
                    for gt_index, prediction_index in matches:
                        target = pose_targets[gt_index]
                        if not target.rotation_eligible and not target.point_set_eligible:
                            continue
                        predicted_rotation = pose_predictions.rotation_matrix_bn33[0, prediction_index]
                        if target.point_set_eligible:
                            if pose_predictions.centroid_m_bn3 is None:
                                raise ValueError("Point-set calibration requires predicted centroids")
                            _, full_error, centroid_error, _, _, _ = point_set_pose_errors(
                                predicted_rotation,
                                pose_predictions.centroid_m_bn3[0, prediction_index],
                                target,
                                chunk_size=pose_config.point_distance_chunk_size,
                            )
                            success = (
                                (centroid_error <= pose_config.centroid_tolerance)
                                & (full_error <= pose_config.point_set_tolerance)
                            ).float()
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
                                axis_cad=(
                                    target.axis_cad.to(predicted_rotation)
                                    if target.axis_cad is not None
                                    else None
                                ),
                            )
                            translation = pose_predictions.translation_m_bn3[0, prediction_index]
                            translation_error = torch.linalg.vector_norm(
                                translation - target.translation_m.to(translation)
                            )
                            if pose_config.normalize_translation_error:
                                translation_error = translation_error / torch.linalg.vector_norm(
                                    target.dimensions_m.to(translation_error)
                                ).clamp_min(1e-8)
                            success = (
                                (rotation_error <= pose_config.rotation_tolerance_rad)
                                & (translation_error <= pose_config.translation_tolerance)
                            ).float()
                        calibration_logits.append(pose_predictions.pose_score_logits_bn[0, prediction_index].float())
                        calibration_targets.append(success.float())
                        if (gt_index, prediction_index) in conditional_match_set:
                            conditional_calibration_logits.append(
                                pose_predictions.pose_score_logits_bn[0, prediction_index].float()
                            )
                            conditional_calibration_targets.append(success.float())
    finally:
        _restore_training_state(training_state)

    if total_count == 0:
        empty = {
            "count": 0.0,
            "eligible_samples": float(eligible_count),
            "pose_assignment_coverage": 0.0,
            "pose_match_iou_threshold": 0.0,
        }
        conditional_empty = None
        if conditional_threshold > 0.0:
            conditional_empty = {
                **empty,
                "pose_match_iou_threshold": conditional_threshold,
                "pose_match_coverage": 0.0,
                "pose_end_to_end_success_rate": 0.0,
            }
        return empty, conditional_empty
    metrics = {
        name: (
            metric_sums[name] / metric_counts[name]
            if metric_counts[name]
            else float("nan")
        )
        for name in evaluated_fields
    }
    metrics["p95_surface_distance_norm"] = surface_distance_percentile(
        surface_distance_chunks
    )
    metrics["count"] = float(total_count)
    metrics["eligible_samples"] = float(eligible_count)
    metrics["pose_match_iou_threshold"] = 0.0
    metrics["pose_assignment_coverage"] = total_count / max(1, eligible_count)
    conditional_metrics: Optional[Dict[str, float]] = None
    if conditional_threshold > 0.0:
        conditional_metrics = {
            name: (
                conditional_metric_sums[name] / conditional_metric_counts[name]
                if conditional_metric_counts[name]
                else float("nan")
            )
            for name in evaluated_fields
        }
        conditional_metrics["p95_surface_distance_norm"] = surface_distance_percentile(
            conditional_surface_distance_chunks
        )
        conditional_metrics["count"] = float(conditional_count)
        conditional_metrics["eligible_samples"] = float(eligible_count)
        conditional_metrics["pose_match_iou_threshold"] = conditional_threshold
        conditional_metrics["pose_assignment_coverage"] = total_count / max(
            1, eligible_count
        )
        conditional_metrics["pose_match_coverage"] = conditional_count / max(1, eligible_count)
        conditional_success = conditional_metrics.get("pose_success_rate", float("nan"))
        conditional_metrics["pose_end_to_end_success_rate"] = (
            float(conditional_success) * conditional_count / max(1, eligible_count)
            if np.isfinite(conditional_success)
            else float("nan")
        )
    if calibration_logits:
        logits = torch.stack(calibration_logits)
        targets = torch.stack(calibration_targets)
        if calibrate_temperature:
            temperature = fit_pose_score_temperature(logits, targets)
            detmodel.cad_pose_head.set_pose_score_temperature(temperature)
        else:
            temperature = float(detmodel.cad_pose_head.pose_score_temperature)
        probabilities = torch.sigmoid(logits / max(temperature, 1e-6))
        metrics["pose_score_temperature"] = temperature
        metrics["brier_score"] = float(((probabilities - targets) ** 2).mean())
        metrics["expected_calibration_error"] = float(expected_calibration_error(probabilities, targets))
        if conditional_metrics is not None and conditional_calibration_logits:
            conditional_logits = torch.stack(conditional_calibration_logits)
            conditional_targets = torch.stack(conditional_calibration_targets)
            conditional_probabilities = torch.sigmoid(
                conditional_logits / max(temperature, 1e-6)
            )
            conditional_metrics["pose_score_temperature"] = temperature
            conditional_metrics["brier_score"] = float(
                ((conditional_probabilities - conditional_targets) ** 2).mean()
            )
            conditional_metrics["expected_calibration_error"] = float(
                expected_calibration_error(conditional_probabilities, conditional_targets)
            )
    return metrics, conditional_metrics


def main() -> None:
    print("Starting fine-tuning with SAMv3 exemplar detection modules.")
    args = parse_args()
    if args.exemplar_view_shuffle_seed is None:
        args.exemplar_view_shuffle_seed = args.seed
    if args.exemplar_view_mode != "none" and not args.enable_pose:
        raise ValueError("Reference-view conditioning experiments require --enable_pose")
    checkpoint_sources = [args.init_path, args.resume_path, args.transfer_path]
    if sum(bool(path) for path in checkpoint_sources) > 1:
        raise ValueError("--init_path, --resume_path, and --transfer_path are mutually exclusive")
    if args.transfer_path and not args.enable_pose:
        raise ValueError("--transfer_path requires --enable_pose")
    args.init_checkpoint_sha256 = ""
    args.transfer_checkpoint_sha256 = ""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(
        f"Using random seed {args.seed}; exemplar_view_mode={args.exemplar_view_mode} "
        f"shuffle_seed={args.exemplar_view_shuffle_seed}"
    )

    for name in ("pose_train_min_match_iou", "pose_eval_min_match_iou"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}")
    for name in (
        "joint_shared_lr_scale",
        "pose_prompt_lr_scale",
        "pose_aux_loss_weight",
        "grad_clip_norm",
        "joint_bbox_weight",
        "joint_objectness_weight",
        "joint_mask_weight",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name} must be nonnegative, got {value}")
    if args.pose_stage == "joint_lite" and not args.enable_pose:
        raise ValueError("--pose_stage joint_lite requires --enable_pose")
    if args.enable_cad_prompt and not args.enable_pose:
        raise ValueError("--enable_cad_prompt requires --enable_pose")
    if args.pose_deep_supervision and not args.enable_pose:
        raise ValueError("--pose_deep_supervision requires --enable_pose")

    manifest_path = Path(args.dataset_manifest).expanduser().resolve() if args.dataset_manifest else None
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    legacy_split_requested = any(
        (args.split_dir, args.train_split_csv, args.val_split_csv, args.test_split_csv, args.recreate_splits)
    )
    if manifest_path is not None:
        if args.dataset_root:
            raise ValueError("--dataset_manifest and --dataset_root are mutually exclusive.")
        if legacy_split_requested:
            raise ValueError("Manifest mode cannot be combined with legacy split CSV arguments.")
        if data_root is None:
            raise ValueError("--data_root is required with --dataset_manifest.")
        manifest_rows, manifest_summary = load_manifest(manifest_path, data_root, validate_files=True)
        manifest_checksum = manifest_sha256(manifest_path)
        dataset_roots: List[Tuple[str, float, bool]] = []
        print(f"Loaded manifest {manifest_path} (sha256={manifest_checksum})")
        print(json.dumps(manifest_summary, indent=2, sort_keys=True))
    else:
        manifest_checksum = ""
        manifest_rows = []
        manifest_summary = {}
        dataset_roots = normalize_dataset_roots(args.dataset_root)
        if not dataset_roots:
            raise ValueError("Provide --dataset_manifest with --data_root, or deprecated --dataset_root.")
        warnings.warn(
            "--dataset_root is deprecated; create a versioned manifest for reproducible multi-dataset training.",
            FutureWarning,
            stacklevel=2,
        )
        if args.eval_only:
            raise ValueError("--eval_only requires --dataset_manifest.")
    if args.enable_pose and not manifest_rows:
        raise ValueError("--enable_pose requires the versioned --dataset_manifest interface.")

    if not args.reference_dir:
        raise ValueError("--reference_dir is required.")
    if args.eval_only and not args.resume_path:
        raise ValueError("--eval_only requires an explicit --resume_path checkpoint.")

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
    pose_aux_layer_indices = (
        parse_pose_aux_layer_indices(
            args.pose_aux_layers,
            len(detmodel.exemplar_detector.fusion_layers),
        )
        if args.pose_deep_supervision
        else ()
    )

    freeze_module(detmodel.text_encoder)
    if args.enable_pose:
        freeze_module(detmodel.image_encoder)
        freeze_module(detmodel.image_projection)
        freeze_module(detmodel.sampling_encoder)
    else:
        unfreeze_module(detmodel.image_encoder)
        unfreeze_module(detmodel.image_projection)
        unfreeze_module(detmodel.sampling_encoder)
    if args.enable_pose and args.pose_stage == "head":
        freeze_module(detmodel.image_exemplar_fusion)
        freeze_module(detmodel.exemplar_detector)
        freeze_module(detmodel.exemplar_segmentation)
    elif args.enable_pose and args.pose_stage == "joint_lite":
        unfreeze_module(detmodel.image_exemplar_fusion)
        unfreeze_module(detmodel.exemplar_detector)
        # Keep the fixed decoder in the differentiable forward path. Mask loss
        # still anchors its input tokens while decoder parameters remain fixed.
        freeze_module(detmodel.exemplar_segmentation)
    else:
        unfreeze_module(detmodel.image_exemplar_fusion)
        unfreeze_module(detmodel.exemplar_detector)
        unfreeze_module(detmodel.exemplar_segmentation)
    if args.enable_pose:
        unfreeze_module(detmodel.cad_pose_head)
        if not args.enable_cad_prompt:
            freeze_module(detmodel.cad_pose_head.cad_prompt_adapter)
    else:
        freeze_module(detmodel.cad_pose_head)
    if args.enable_pose and args.exemplar_view_mode != "none":
        unfreeze_module(detmodel.exemplar_view_pose_encoder)
    else:
        freeze_module(detmodel.exemplar_view_pose_encoder)

    trainable_params: List[torch.nn.Parameter] = []
    trainable_modules = [detmodel.image_exemplar_fusion, detmodel.exemplar_detector, detmodel.exemplar_segmentation]
    if args.enable_pose:
        trainable_modules.append(detmodel.cad_pose_head)
    if args.exemplar_view_mode != "none":
        trainable_modules.append(detmodel.exemplar_view_pose_encoder)
    for module in trainable_modules:
        trainable_params.extend([p for p in module.parameters() if p.requires_grad])
    total_params = sum(p.numel() for p in detmodel.parameters())
    trainable_params_count = sum(p.numel() for p in detmodel.parameters() if p.requires_grad)
    print(f"Parameter counts: total={total_params:,} trainable={trainable_params_count:,}")

    if args.enable_pose:
        shared_params = [
            p
            for module in (
                detmodel.image_exemplar_fusion,
                detmodel.exemplar_detector,
                detmodel.exemplar_segmentation,
            )
            for p in module.parameters()
            if p.requires_grad
        ]
        prompt_params = [
            p for p in detmodel.cad_pose_head.cad_prompt_adapter.parameters()
            if p.requires_grad
        ]
        prompt_param_ids = {id(parameter) for parameter in prompt_params}
        pose_params = [
            p for p in detmodel.cad_pose_head.parameters()
            if p.requires_grad and id(p) not in prompt_param_ids
        ]
        pose_params.extend(
            p for p in detmodel.exemplar_view_pose_encoder.parameters() if p.requires_grad
        )
        shared_lr = (
            args.lr * args.joint_shared_lr_scale
            if args.pose_stage == "joint_lite"
            else args.lr
        )
        optimizer_groups = []
        if shared_params:
            optimizer_groups.append(
                {"params": shared_params, "lr": shared_lr, "name": "shared_detection"}
            )
        if pose_params:
            optimizer_groups.append(
                {"params": pose_params, "lr": args.lr, "name": "pose_and_view"}
            )
        if prompt_params:
            optimizer_groups.append(
                {
                    "params": prompt_params,
                    "lr": args.lr * args.pose_prompt_lr_scale,
                    "name": "cad_prompt_adapter",
                }
            )
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print(
            "Pose optimizer: "
            + ", ".join(
                f"{group['name']}_lr={float(group['lr']):.3g}"
                for group in optimizer_groups
            )
            + f"; segmentation_decoder={'trainable' if any(p.requires_grad for p in detmodel.exemplar_segmentation.parameters()) else 'frozen'}"
        )
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    global_step = 0
    batch_step = 0
    resume_checkpoint = None

    if args.resume_in_place and not args.resume_path:
        raise ValueError("--resume_in_place requires --resume_path.")

    if args.init_path:
        init_path = Path(args.init_path).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(init_path)
        load_detector_initialization_checkpoint(init_path, detmodel, device)
        args.init_path = str(init_path)
        args.init_checkpoint_sha256 = manifest_sha256(init_path)
        print(
            f"Initialized detector modules from {init_path} "
            f"(sha256={args.init_checkpoint_sha256}); pose head, view adapter, optimizer, "
            "and epoch counters remain fresh."
        )

    if args.transfer_path:
        transfer_path = Path(args.transfer_path).expanduser().resolve()
        if not transfer_path.is_file():
            raise FileNotFoundError(transfer_path)
        checkpoint = load_finetune_checkpoint(
            transfer_path,
            detmodel,
            optimizer=None,
            device=device,
            load_optimizer=False,
            exemplar_view_mode=args.exemplar_view_mode,
        )
        checkpoint_args = checkpoint.get("args") or {}
        if not bool(checkpoint_args.get("enable_pose", False)):
            raise ValueError("--transfer_path must contain a pose-trained checkpoint")
        args.transfer_path = str(transfer_path)
        args.transfer_checkpoint_sha256 = manifest_sha256(transfer_path)
        args.init_path = str(checkpoint_args.get("init_path", ""))
        args.init_checkpoint_sha256 = str(
            checkpoint_args.get("init_checkpoint_sha256", "")
        )
        print(
            f"Transferred trained modules from {transfer_path} "
            f"(source_epoch={int(checkpoint.get('epoch', 0))}, "
            f"sha256={args.transfer_checkpoint_sha256}); optimizer, epoch, step, "
            "manifest, and pose statistics start fresh."
        )

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
            exemplar_view_mode=args.exemplar_view_mode,
        )
        resume_checkpoint = checkpoint
        checkpoint_args = checkpoint.get("args") or {}
        args.init_path = str(checkpoint_args.get("init_path", ""))
        args.init_checkpoint_sha256 = str(
            checkpoint_args.get("init_checkpoint_sha256", "")
        )
        validate_resume_manifest_checksum(checkpoint, manifest_checksum)
        ckpt_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = max(1, ckpt_epoch + 1)
        global_step = int(checkpoint.get("global_step", 0))
        batch_step = int(checkpoint.get("batch_step", 0))
        print(
            f"Resuming from {resume_path} (epoch={ckpt_epoch}, global_step={global_step}, batch_step={batch_step}). "
            f"Starting at epoch {start_epoch}."
        )
    detmodel.cad_pose_head.set_cad_prompt_enabled(args.enable_cad_prompt)
    if args.enable_pose:
        print(
            f"Pose features: cad_prompt={args.enable_cad_prompt}, "
            f"deep_supervision_layers={tuple(index + 1 for index in pose_aux_layer_indices) or 'none'}, "
            f"aux_weight={args.pose_aux_loss_weight if pose_aux_layer_indices else 0.0:g}"
        )
    if args.resume_in_place and args.resume_path:
        run_dir = str(resolve_run_dir_from_checkpoint(Path(args.resume_path)))
    else:
        run_dir = create_run_dir(args.output_dir)
    args.output_dir = run_dir
    if manifest_path is not None and args.resume_in_place:
        manifest_copy = Path(run_dir) / "dataset_manifest.csv"
        if os.path.lexists(manifest_copy):
            if not manifest_copy.is_file():
                raise ValueError(
                    f"Existing run manifest snapshot is not a regular file: {manifest_copy}"
                )
            if manifest_sha256(manifest_copy) != manifest_checksum:
                raise ValueError(
                    "In-place resume manifest differs from the run's existing dataset_manifest.csv"
                )

    # Split handling is frame-level. Build or load split CSVs before training
    # entries are multiplied so duplicate sampling cannot move data across
    # train/validation/test boundaries.
    split_dir = Path(args.split_dir).expanduser().resolve() if args.split_dir else None
    train_split_csv = Path(args.train_split_csv).expanduser().resolve() if args.train_split_csv else None
    val_split_csv = Path(args.val_split_csv).expanduser().resolve() if args.val_split_csv else None
    test_split_csv = Path(args.test_split_csv).expanduser().resolve() if args.test_split_csv else None
    if split_dir is not None:
        if train_split_csv is None:
            train_split_csv = split_dir / "train.csv"
        if val_split_csv is None:
            val_split_csv = split_dir / "val.csv"
        if test_split_csv is None:
            test_split_csv = split_dir / "test.csv"

    dataset_parts: List[Tuple[str, float, List[Dict[str, str]]]] = []
    all_frame_ids: set = set()
    for root, multiplier, use_filter in dataset_roots:
        _, cur_entries = collect_multi_object_samples(root)
        cur_entries = unique_frame_entries(cur_entries)
        if use_filter:
            filter_path = resolve_dataset_filter_path(root)
            if filter_path is not None:
                cur_entries = apply_dataset_filter_entries(cur_entries, filter_path)
        all_frame_ids.update(entry["frame_id"] for entry in cur_entries)
        dataset_parts.append((root, multiplier, cur_entries))

    if split_dir is not None:
        # First run: create train/val/test CSVs. Later runs: reuse them so
        # checkpoints remain comparable unless --recreate_splits is explicit.
        ratios = parse_split_ratios(args.split_ratios)
        train_path, val_path, test_path = create_frame_split_csvs(
            split_dir,
            sorted(all_frame_ids),
            ratios,
            seed=args.seed,
            overwrite=args.recreate_splits,
        )
        print(f"Using split CSVs: train={train_path} val={val_path} test={test_path}")

    for split_name, split_path in (
        ("train", train_split_csv),
        ("val", val_split_csv),
        ("test", test_split_csv),
    ):
        if split_path is not None and not split_path.is_file():
            raise FileNotFoundError(f"Missing {split_name} split CSV: {split_path}")

    if train_split_csv is not None:
        train_frame_ids = load_split_frame_ids(train_split_csv)
        print(f"Train split: {len(train_frame_ids)} frame ids from {train_split_csv}")
    if val_split_csv is not None:
        val_frame_ids = load_split_frame_ids(val_split_csv)
        print(f"Validation split: {len(val_frame_ids)} frame ids from {val_split_csv}")
    if test_split_csv is not None:
        test_frame_ids = load_split_frame_ids(test_split_csv)
        print(f"Test split reserved: {len(test_frame_ids)} frame ids from {test_split_csv}")

    all_entries: List[Dict[str, str]] = []
    for _, multiplier, cur_entries in dataset_parts:
        if train_split_csv is not None:
            cur_entries = apply_dataset_filter_entries(cur_entries, train_split_csv)
        cur_entries = apply_dataset_multiplier(cur_entries, multiplier)
        all_entries.extend(cur_entries)
    entries_by_dataset: Dict[str, List[Dict[str, str]]] = {}
    if manifest_rows:
        train_rows = [row for row in manifest_rows if row.split == "train"]
        all_entries = entries_from_manifest_rows(train_rows, data_root, object_level=False)
        for entry in all_entries:
            entries_by_dataset.setdefault(entry["dataset_id"], []).append(entry)
    if not all_entries:
        raise RuntimeError("No dataset entries found.")
    train_frames = {(entry.get("dataset_id", "legacy"), entry["frame_id"]) for entry in all_entries}
    print(f"Training entries: {len(all_entries)} frames/views from {len(train_frames)} dataset-qualified frames.")
    if entries_by_dataset:
        print(f"Equal-domain training pools: { {key: len(value) for key, value in sorted(entries_by_dataset.items())} }")

    reference_dir = Path(args.reference_dir).expanduser().resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)

    validation_config: Optional[EvalConfig] = None
    validation_pose_entries: List[Dict[str, str]] = []
    if manifest_rows:
        validation_split = "train" if args.validate_on_train else "validation"
        validation_rows = [row for row in manifest_rows if row.split == validation_split]
        if args.validate_on_train:
            print(
                f"Overfit diagnostic: evaluating {len(validation_rows)} unaugmented "
                "training captures after each epoch."
            )
        validation_entries = entries_from_manifest_rows(validation_rows, data_root, object_level=True)
        validation_pose_entries = entries_from_manifest_rows(validation_rows, data_root, object_level=False)
        validation_config = EvalConfig(
            dataset_root=None,
            dataset_entries=validation_entries,
            reference_dir=args.reference_dir,
            ref_view_ids=args.ref_view_ids,
            max_side_length=args.max_side_length,
            use_square_sizing=not args.no_square,
            num_points_approx=args.num_points_approx,
            batch_size=args.batch_size,
            det_filter=args.det_filter,
            nms_iou=args.nms_iou,
            matches_per_gt=args.matches_per_gt,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            bbox_weight=args.bbox_weight,
            score_weight=args.score_weight,
            no_object_weight=args.no_object_weight,
            shuffle=False,
            max_batches=0,
            exemplar_view_mode=args.exemplar_view_mode,
            exemplar_view_shuffle_seed=args.exemplar_view_shuffle_seed,
        )
    elif val_split_csv is not None:
        # Validation uses the same dataset roots and exemplar renders as
        # training, but filters entries through val.csv and disables training
        # augmentation inside run_exemplar_eval.
        validation_config = EvalConfig(
            dataset_root=args.dataset_root,
            reference_dir=args.reference_dir,
            ref_view_ids=args.ref_view_ids,
            split_csv=str(val_split_csv),
            max_side_length=args.max_side_length,
            use_square_sizing=not args.no_square,
            num_points_approx=args.num_points_approx,
            batch_size=args.batch_size,
            det_filter=args.det_filter,
            nms_iou=args.nms_iou,
            matches_per_gt=args.matches_per_gt,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            bbox_weight=args.bbox_weight,
            score_weight=args.score_weight,
            no_object_weight=args.no_object_weight,
            shuffle=False,
            max_batches=0,
            exemplar_view_mode=args.exemplar_view_mode,
            exemplar_view_shuffle_seed=args.exemplar_view_shuffle_seed,
        )

    ref_cache: Dict[str, Union[torch.Tensor, ExemplarViewBundle]] = {}
    ref_image_cache: Dict[str, List[np.ndarray]] = {}
    seg_cache: Dict[str, np.ndarray] = {}
    mapping_cache: Dict[str, Dict[str, List[Tuple[int, ...]]]] = {}
    pose_cache: Dict[Tuple[str, str], object] = {}
    catalog_checksums: set = set()
    dataset_meta_checksums: set = set()
    annotation_checksums: set = set()
    symmetry_pipeline_versions: set = set()
    point_set_checksums: set = set()
    sampling_pipeline_versions: set = set()
    sampling_parameter_checksums: set = set()
    schema_checksums: set = set()
    schema_versions: set = set()
    pose_config = None
    if args.enable_pose:
        pose_config = CADPoseLossConfig(
            center_weight=args.pose_center_weight,
            depth_weight=args.pose_depth_weight,
            rotation_weight=args.pose_rotation_weight,
            full_pose_weight=args.pose_full_set_weight,
            quality_weight=args.pose_quality_weight,
            pose_weight=args.pose_weight,
            log_depth_mean=args.log_depth_mean,
            log_depth_std=args.log_depth_std,
            centroid_tolerance=args.centroid_tolerance,
            point_set_tolerance=args.point_set_tolerance,
            centroid_soft_width=args.centroid_soft_width,
            point_set_soft_width=args.point_set_soft_width,
            point_loss_beta=args.point_loss_beta,
            point_distance_chunk_size=args.point_distance_chunk_size,
            rotation_tolerance_rad=np.deg2rad(args.rotation_tolerance_deg),
            translation_tolerance=args.translation_tolerance,
            rotation_soft_width_rad=np.deg2rad(args.rotation_soft_width_deg),
            translation_soft_width=args.translation_soft_width,
            normalize_translation_error=not args.absolute_translation_tolerance,
        )
        if resume_checkpoint is not None and resume_checkpoint.get("pose_config") is not None:
            pose_config = CADPoseLossConfig(**resume_checkpoint["pose_config"])
        preflight_entries = list(all_entries)
        if manifest_rows:
            preflight_rows = [row for row in manifest_rows if row.split in ("train", "validation")]
            preflight_entries = entries_from_manifest_rows(preflight_rows, data_root, object_level=False)
        preflight_samples = []
        training_log_dimensions = []
        training_log_depths = []
        pose_reference_cad_ids: set[str] = set()
        pose_catalog = None
        for entry in preflight_entries:
            camera_root = Path(entry["inst_path"]).parent
            key = (str(camera_root), entry["frame_id"])
            sample = pose_cache.get(key)
            if sample is None:
                sample = load_perseve_pose_sample(camera_root, entry["frame_id"], validate_pixels=True)
                pose_cache[key] = sample
            if pose_catalog is None:
                pose_catalog = sample.catalog
            preflight_samples.append(sample)
            catalog_checksums.add(sample.catalog_checksum)
            dataset_meta_checksums.add(sample.dataset_meta_checksum)
            annotation_checksums.add(sample.annotation_checksum)
            symmetry_pipeline_versions.update(sample.symmetry_pipeline_versions)
            point_set_checksums.update(sample.point_set_checksums)
            sampling_pipeline_versions.update(sample.sampling_pipeline_versions)
            sampling_parameter_checksums.update(sample.sampling_parameter_checksums)
            schema_checksums.update(sample.schema_checksums.values())
            schema_versions.add(sample.frame.schema_version)
            pose_reference_cad_ids.update(
                instance.cad_id for instance in sample.frame.eligible_instances()
            )
            if entry.get("split", "train") == "train":
                for instance in sample.frame.eligible_instances():
                    training_log_dimensions.append(np.log(instance.dimensions_m))
                    catalog_object = sample.catalog[instance.cad_id]
                    effective_centroid = effective_surface_centroid_m(instance, catalog_object)
                    centroid_camera = (
                        instance.rotation_matrix @ effective_centroid + instance.translation_m
                    )
                    training_log_depths.append(np.log(centroid_camera[2]))
        validate_scale_sharing(preflight_samples)
        if len(catalog_checksums) != 1:
            raise ValueError(f"Pose training requires one object-catalog checksum, got {sorted(catalog_checksums)}")
        if point_set_checksums and symmetry_pipeline_versions:
            raise ValueError("Do not mix point-set pose-v2 and legacy symmetry pose-v1 data in one run")
        if sampling_pipeline_versions and len(sampling_pipeline_versions) != 1:
            raise ValueError(
                "Pose training requires one point-set sampling-pipeline version, got "
                f"{sorted(sampling_pipeline_versions)}"
            )
        if not sampling_pipeline_versions and len(symmetry_pipeline_versions) != 1:
            raise ValueError(
                "Legacy pose-v1 training requires one symmetry-pipeline version, got "
                f"{sorted(symmetry_pipeline_versions)}"
            )
        if resume_checkpoint is not None:
            validate_pose_resume_provenance(
                resume_checkpoint,
                catalog_checksums=catalog_checksums,
                dataset_meta_checksums=dataset_meta_checksums,
                annotation_checksums=annotation_checksums,
                symmetry_pipeline_versions=symmetry_pipeline_versions,
                point_set_checksums=point_set_checksums,
                sampling_pipeline_versions=sampling_pipeline_versions,
                sampling_parameter_checksums=sampling_parameter_checksums,
                schema_checksums=schema_checksums,
                schema_versions=schema_versions,
            )
        if not training_log_dimensions or not training_log_depths:
            raise ValueError("Pose training split contains no eligible pose instances")
        reference_summary = validate_pose_reference_metadata(
            reference_dir,
            sorted(pose_reference_cad_ids),
            pose_catalog,
            ref_view_ids,
        )
        print(
            "Reference preflight passed: "
            f"cad_ids={reference_summary['cad_id_count']} "
            f"validated_views={reference_summary['view_count']}"
        )
        dimension_values = np.stack(training_log_dimensions)
        dimension_mean = torch.as_tensor(dimension_values.mean(axis=0), device=device, dtype=dtype)
        dimension_std = torch.as_tensor(dimension_values.std(axis=0).clip(min=1e-6), device=device, dtype=dtype)
        detmodel.cad_pose_head.set_dimension_statistics(dimension_mean, dimension_std)
        depth_mean = float(np.mean(training_log_depths))
        depth_std = float(max(np.std(training_log_depths), 1e-6))
        pose_config = replace(pose_config, log_depth_mean=depth_mean, log_depth_std=depth_std)
        annotation_digest = checksum_set_digest(annotation_checksums)
        pose_provenance = {
            "manifest_sha256": manifest_checksum,
            "schema_versions": sorted(schema_versions),
            "schema_checksums": sorted(schema_checksums),
            "catalog_checksums": sorted(catalog_checksums),
            "dataset_meta_checksums": sorted(dataset_meta_checksums),
            "annotation_checksums": sorted(annotation_checksums),
            "annotation_checksum_sha256": annotation_digest,
            "symmetry_pipeline_versions": sorted(symmetry_pipeline_versions),
            "point_set_checksums": sorted(point_set_checksums),
            "sampling_pipeline_versions": sorted(sampling_pipeline_versions),
            "sampling_parameter_checksums": sorted(sampling_parameter_checksums),
            "pose_config": pose_config.__dict__,
            "dimension_log_mean": detmodel.cad_pose_head.dimension_log_mean.detach().float().cpu().tolist(),
            "dimension_log_std": detmodel.cad_pose_head.dimension_log_std.detach().float().cpu().tolist(),
            "training_eligible_instances": len(training_log_depths),
        }
        with (Path(run_dir) / "pose_provenance.json").open("w") as handle:
            json.dump(pose_provenance, handle, indent=2, sort_keys=True, default=str)
        print(
            "Pose preflight passed: "
            f"frames={len(preflight_samples)} eligible_instances={len(training_log_depths)} "
            f"log_depth_mean={depth_mean:.6f} log_depth_std={depth_std:.6f}"
        )

    debug_dir = os.path.join(run_dir, "debug_boxes")
    os.makedirs(debug_dir, exist_ok=True)
    checkpoints_dir = Path(run_dir) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(run_dir) / "metrics.csv"
    initialize_metrics_log(metrics_path)
    if manifest_path is not None:
        manifest_copy = Path(run_dir) / "dataset_manifest.csv"
        if os.path.lexists(manifest_copy):
            if not manifest_copy.is_file() or manifest_sha256(manifest_copy) != manifest_checksum:
                raise ValueError(
                    f"Refusing to replace a different run manifest snapshot: {manifest_copy}"
                )
        else:
            shutil.copy2(manifest_path, manifest_copy)
        provenance = {
            "manifest_source": str(manifest_path),
            "manifest_copy": str(manifest_copy),
            "manifest_sha256": manifest_checksum,
            "data_root": str(data_root),
            "sampling_policy": "equal_domain_with_replacement",
            "manifest_summary": manifest_summary,
            "args": vars(args),
        }
        with (Path(run_dir) / "run_config.json").open("w") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True, default=str)

    running_losses: deque[float] = deque(maxlen=5)
    running_top_ious: deque[float] = deque(maxlen=5)

    if args.eval_only:
        eval_rows = [row for row in manifest_rows if row.split == args.eval_split]
        eval_entries = entries_from_manifest_rows(eval_rows, data_root, object_level=True)
        eval_config = EvalConfig(
            dataset_root=None,
            dataset_entries=eval_entries,
            reference_dir=args.reference_dir,
            ref_view_ids=args.ref_view_ids,
            max_side_length=args.max_side_length,
            use_square_sizing=not args.no_square,
            num_points_approx=args.num_points_approx,
            batch_size=args.batch_size,
            det_filter=args.det_filter,
            nms_iou=args.nms_iou,
            matches_per_gt=args.matches_per_gt,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            bbox_weight=args.bbox_weight,
            score_weight=args.score_weight,
            no_object_weight=args.no_object_weight,
            shuffle=False,
            max_batches=0,
            exemplar_view_mode=args.exemplar_view_mode,
            exemplar_view_shuffle_seed=args.exemplar_view_shuffle_seed,
        )
        eval_avg, eval_count, eval_correct, eval_loss = run_exemplar_eval(detmodel, eval_config, device=device)
        append_metric(
            metrics_path,
            phase=f"eval_{args.eval_split}",
            epoch=start_epoch - 1,
            global_step=global_step,
            batch_step=batch_step,
            loss=eval_loss,
            avg_iou=eval_avg,
            correct_rate=eval_correct,
            samples=eval_count,
        )
        print(
            f"[eval:{args.eval_split}] loss={eval_loss:.4f} avg_iou={eval_avg:.4f} "
            f"correct_rate={eval_correct:.3f} samples={eval_count}"
        )
        if args.enable_pose:
            pose_eval_entries = entries_from_manifest_rows(eval_rows, data_root, object_level=False)
            pose_metrics, conditional_pose_metrics = run_cad_pose_eval(
                detmodel,
                pose_eval_entries,
                args,
                device,
                pose_config,
                calibrate_temperature=args.eval_split == "validation",
            )
            append_pose_metric_rows(
                metrics_path,
                phase=f"eval_{args.eval_split}_pose",
                epoch=start_epoch - 1,
                global_step=global_step,
                batch_step=batch_step,
                metrics=pose_metrics,
                conditional_metrics=conditional_pose_metrics,
            )
            print(f"[eval:{args.eval_split}:pose] {json.dumps(pose_metrics, sort_keys=True)}")
            if conditional_pose_metrics is not None:
                print(
                    f"[eval:{args.eval_split}:pose:iou] "
                    f"{json.dumps(conditional_pose_metrics, sort_keys=True)}"
                )
        return

    if start_epoch > args.epochs:
        print(f"Resume epoch {start_epoch - 1} exceeds requested --epochs={args.epochs}. Nothing to do.")
        return

    if args.validate_before_training and validation_config is not None:
        baseline_epoch = start_epoch - 1
        eval_avg, eval_count, eval_correct, eval_loss = run_exemplar_eval(
            detmodel, validation_config, device=device
        )
        append_metric(
            metrics_path,
            phase="validation",
            epoch=baseline_epoch,
            global_step=global_step,
            batch_step=batch_step,
            loss=eval_loss,
            avg_iou=eval_avg,
            correct_rate=eval_correct,
            samples=eval_count,
        )
        print(
            f"[eval:baseline] epoch={baseline_epoch} loss={eval_loss:.4f} "
            f"avg_iou={eval_avg:.4f} correct_rate={eval_correct:.3f} samples={eval_count}"
        )
        if args.enable_pose and validation_pose_entries:
            pose_metrics, conditional_pose_metrics = run_cad_pose_eval(
                detmodel,
                validation_pose_entries,
                args,
                device,
                pose_config,
                calibrate_temperature=False,
            )
            append_pose_metric_rows(
                metrics_path,
                phase="validation_pose",
                epoch=baseline_epoch,
                global_step=global_step,
                batch_step=batch_step,
                metrics=pose_metrics,
                conditional_metrics=conditional_pose_metrics,
            )
            print(
                f"[eval:pose:baseline] epoch={baseline_epoch} "
                f"{json.dumps(pose_metrics, sort_keys=True)}"
            )
            if conditional_pose_metrics is not None:
                print(
                    f"[eval:pose:baseline:iou] epoch={baseline_epoch} "
                    f"{json.dumps(conditional_pose_metrics, sort_keys=True)}"
                )

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch != start_epoch and validation_config is not None:
            eval_avg, eval_count, eval_correct, eval_loss = run_exemplar_eval(
                detmodel, validation_config, device=device
            )
            append_metric(
                metrics_path,
                phase="validation",
                epoch=epoch - 1,
                global_step=global_step,
                batch_step=batch_step,
                loss=eval_loss,
                avg_iou=eval_avg,
                correct_rate=eval_correct,
                samples=eval_count,
            )
            if eval_count > 0:
                loss_part = f" val_loss={eval_loss:.4f}" if np.isfinite(eval_loss) else ""
                print(
                    f"[eval] epoch={epoch}{loss_part} avg_iou={eval_avg:.4f} "
                    f"correct_rate={eval_correct:.3f} samples={eval_count}"
                )
            else:
                loss_part = f" val_loss={eval_loss:.4f}" if np.isfinite(eval_loss) else ""
                print(f"[eval] epoch={epoch}{loss_part} no valid samples")
            if args.enable_pose and validation_pose_entries:
                pose_metrics, conditional_pose_metrics = run_cad_pose_eval(
                    detmodel,
                    validation_pose_entries,
                    args,
                    device,
                    pose_config,
                    calibrate_temperature=False,
                )
                append_pose_metric_rows(
                    metrics_path,
                    phase="validation_pose",
                    epoch=epoch - 1,
                    global_step=global_step,
                    batch_step=batch_step,
                    metrics=pose_metrics,
                    conditional_metrics=conditional_pose_metrics,
                )
                print(f"[eval:pose] epoch={epoch - 1} {json.dumps(pose_metrics, sort_keys=True)}")
                if conditional_pose_metrics is not None:
                    print(
                        f"[eval:pose:iou] epoch={epoch - 1} "
                        f"{json.dumps(conditional_pose_metrics, sort_keys=True)}"
                    )

        if entries_by_dataset:
            epoch_entries = balanced_epoch_entries(
                entries_by_dataset, epoch_size=len(all_entries), seed=args.seed, epoch=epoch
            )
        else:
            epoch_entries = list(all_entries)
            random.shuffle(epoch_entries)
        epoch_loss = 0.0
        epoch_count = 0
        epoch_iou_sum = 0.0
        epoch_iou_count = 0

        for start in range(0, len(epoch_entries), args.batch_size):
            subset = epoch_entries[start : start + args.batch_size]
            prepared: List[Dict[str, object]] = []
            for entry in subset:
                try:
                    image_bgr = load_bgr(entry["rgb_path"])
                except FileNotFoundError:
                    continue
                try:
                    if args.enable_pose:
                        obj_id, gt_masks, pose_instances, pose_sample, pose_gt_indices = select_pose_target_and_masks(
                            entry,
                            pose_cache,
                            require_pose_eligible=args.pose_stage == "head",
                        )
                        catalog_checksums.add(pose_sample.catalog_checksum)
                        dataset_meta_checksums.add(pose_sample.dataset_meta_checksum)
                        annotation_checksums.add(pose_sample.annotation_checksum)
                        symmetry_pipeline_versions.update(pose_sample.symmetry_pipeline_versions)
                        point_set_checksums.update(pose_sample.point_set_checksums)
                        sampling_pipeline_versions.update(pose_sample.sampling_pipeline_versions)
                        sampling_parameter_checksums.update(
                            pose_sample.sampling_parameter_checksums
                        )
                    else:
                        mapping_path = Path(entry["inst_path"]).with_name(
                            f"instance_segmentation_mapping_{entry['frame_id']}.json"
                        )
                        obj_id, gt_masks = select_target_object_and_masks(
                            entry["inst_path"],
                            str(mapping_path),
                            seg_cache=seg_cache,
                            mapping_cache=mapping_cache,
                        )
                        pose_instances, pose_sample, pose_gt_indices = [], None, []
                except (FileNotFoundError, ValueError):
                    if args.enable_pose:
                        raise
                    continue
                if obj_id is None or not gt_masks:
                    continue
                gt_masks = [mask for mask in gt_masks if int(mask.sum()) >= 100]
                if not gt_masks:
                    # print(
                    #     f"Skipping object {obj_id} (frame={entry['frame_id']}) "
                    #     "due to small gt_mask size (<100 pixels)."
                    # )
                    continue
                gt_masks = select_top_gt_masks(gt_masks, max_instances=None)

                if args.enable_pose:
                    image_bgr = apply_random_color_distortion(image_bgr)
                else:
                    image_bgr, gt_masks = data_augmentation(image_bgr, gt_masks)

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
                        include_view_metadata=args.exemplar_view_mode != "none",
                    )
                    if exemplar_ref is None:
                        continue
                    ref_cache[obj_id] = cache_exemplar(exemplar_ref)

                exemplar_ref = ref_cache[obj_id]
                prepared.append(
                    {
                        "object_id": obj_id,
                        "image_bgr": image_bgr,
                        "gt_masks": gt_masks,
                        "exemplar_ref": exemplar_ref,
                        "pose_instances": pose_instances,
                        "pose_sample": pose_sample,
                        "pose_gt_indices": pose_gt_indices,
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
            batch_pose_components: List[object] = []
            batch_pose_aux_losses: List[torch.Tensor] = []
            batch_detection_components: List[MultiGTDetectionLosses] = []
            batch_pose_match_ious: List[float] = []
            batch_pose_eligible_targets = 0
            batch_pose_total_matches = 0
            batch_pose_accepted_matches = 0
            for _, idxs in group_map.items():
                img_batch = torch.cat([prepared[i]["img_tensor"] for i in idxs], dim=0)
                with torch.no_grad():
                    encoded_img = detmodel.image_encoder(img_batch)
                    encoded_image_features_list = detmodel.image_projection.v3_projection(encoded_img)

                exemplars_list = [prepared[i]["exemplar_ref"] for i in idxs]
                exemplar_batch, padding_mask = pad_exemplar_batch(
                    exemplars_list,
                    device=device,
                    pose_encoder=detmodel.exemplar_view_pose_encoder,
                    mode=args.exemplar_view_mode,
                    shuffle_seed=args.exemplar_view_shuffle_seed,
                )
                pose_kwargs = {}
                if args.enable_pose:
                    dimensions_batch = torch.stack(
                        [
                            torch.as_tensor(prepared[i]["pose_instances"][0].dimensions_m, device=device, dtype=img_batch.dtype)
                            for i in idxs
                        ]
                    )
                    effective_centroid_batch = torch.stack(
                        [
                            torch.as_tensor(
                                pose_prompt_surface_centroid_m(
                                    prepared[i]["pose_instances"],
                                    prepared[i]["pose_gt_indices"],
                                    prepared[i]["pose_sample"].catalog[
                                        prepared[i]["pose_instances"][0].cad_id
                                    ],
                                ),
                                device=device,
                                dtype=img_batch.dtype,
                            )
                            for i in idxs
                        ]
                    )
                    adjusted_intrinsics = []
                    for i in idxs:
                        sample = prepared[i]["pose_sample"]
                        pre_h, pre_w = prepared[i]["preencode_hw"]
                        raw_k = torch.as_tensor(sample.frame.intrinsics, device=device, dtype=img_batch.dtype)
                        adjusted_intrinsics.append(
                            adjust_intrinsics_for_resize_and_pad(
                                raw_k, sample.frame.image_size_wh, (pre_w, pre_h)
                            )
                        )
                    pose_kwargs = {
                        "cad_dimensions_m_b3": dimensions_batch,
                        "cad_effective_surface_centroid_m_b3": effective_centroid_batch,
                        "adjusted_intrinsics_b33": torch.stack(adjusted_intrinsics),
                        "model_image_size_wh": (img_batch.shape[-1], img_batch.shape[-2]),
                        "return_pose": True,
                    }

                detection_outputs = generate_detections_train(
                    detmodel,
                    encoded_image_features_list,
                    exemplar_batch,
                    detection_filter_threshold=args.det_filter,
                    exemplar_padding_mask_bn=padding_mask,
                    pose_aux_layer_indices=pose_aux_layer_indices,
                    **pose_kwargs,
                )
                if args.enable_pose:
                    if pose_aux_layer_indices:
                        (
                            mask_preds,
                            box_preds,
                            det_scores,
                            det_scores_logits,
                            _,
                            pose_predictions,
                            auxiliary_pose_predictions,
                        ) = detection_outputs
                    else:
                        mask_preds, box_preds, det_scores, det_scores_logits, _, pose_predictions = detection_outputs
                        auxiliary_pose_predictions = AuxiliaryPosePredictions((), ())
                else:
                    mask_preds, box_preds, det_scores, det_scores_logits, _ = detection_outputs
                    pose_predictions = None
                    auxiliary_pose_predictions = AuxiliaryPosePredictions((), ())
                if mask_preds.shape[1] == 0:
                    continue

                for local_idx, data_idx in enumerate(idxs):
                    preencode_hw = prepared[data_idx]["preencode_hw"]
                    gt_targets: List[torch.Tensor] = []
                    for gt_mask in prepared[data_idx]["gt_masks"]:
                        gt_preenc = resize_mask(gt_mask, preencode_hw)
                        gt_tensor = torch.from_numpy(gt_preenc).to(device).unsqueeze(0).unsqueeze(0)
                        gt_down = F.interpolate(
                            gt_tensor, size=mask_preds.shape[-2:], mode="nearest"
                        ).squeeze(0)
                        gt_targets.append(gt_down[0].float())
                    if not gt_targets:
                        continue

                    logits_mhw = mask_preds[local_idx].float()
                    detection_losses = compute_multi_gt_detection_losses(
                        logits_mhw,
                        box_preds[local_idx],
                        det_scores_logits[local_idx],
                        gt_targets,
                        bce_weight=args.bce_weight,
                        dice_weight=args.dice_weight,
                        score_weight=args.score_weight,
                        no_object_weight=args.no_object_weight,
                    )
                    if detection_losses is None:
                        continue
                    batch_detection_components.append(detection_losses)
                    if args.enable_pose and args.pose_stage == "joint_lite":
                        loss = detection_losses.total(
                            mask_weight=args.joint_mask_weight,
                            bbox_weight=args.joint_bbox_weight,
                            objectness_weight=args.joint_objectness_weight,
                        )
                    else:
                        loss = detection_losses.total(
                            mask_weight=2.0,
                            bbox_weight=args.bbox_weight,
                        )
                    if args.enable_pose:
                        sample = prepared[data_idx]["pose_sample"]
                        pre_h, pre_w = preencode_hw
                        raw_k = torch.as_tensor(sample.frame.intrinsics, device=device, dtype=logits_mhw.dtype)
                        pose_targets = []
                        for gt_index in prepared[data_idx]["pose_gt_indices"]:
                            instance = prepared[data_idx]["pose_instances"][gt_index]
                            pose_targets.append(
                                make_pose_target(
                                    instance,
                                    sample.catalog[instance.cad_id],
                                    raw_k,
                                    sample.frame.image_size_wh,
                                    (pre_w, pre_h),
                                )
                            )
                        batch_pose_eligible_targets += len(pose_targets)
                        full_pose_matches, pose_match_iou = match_pose_predictions_one_to_one(
                            logits_mhw,
                            gt_targets,
                            eligible_gt_indices=prepared[data_idx]["pose_gt_indices"],
                        )
                        batch_pose_total_matches += len(full_pose_matches)
                        full_pose_matches = filter_pose_matches_by_iou(
                            full_pose_matches,
                            pose_match_iou,
                            args.pose_train_min_match_iou,
                        )
                        batch_pose_accepted_matches += len(full_pose_matches)
                        batch_pose_match_ious.extend(
                            float(pose_match_iou[gt_index, prediction_index].detach())
                            for gt_index, prediction_index in full_pose_matches
                        )
                        target_index = {
                            gt_index: local_index
                            for local_index, gt_index in enumerate(prepared[data_idx]["pose_gt_indices"])
                        }
                        pose_matches = [
                            (target_index[gt_index], prediction_index)
                            for gt_index, prediction_index in full_pose_matches
                        ]
                        pose_losses = compute_cad_pose_losses(
                            pose_predictions,
                            pose_targets,
                            pose_matches,
                            pose_config,
                            batch_index=local_idx,
                        )
                        if pose_losses is not None:
                            if not torch.isfinite(pose_losses.total):
                                raise FloatingPointError("Non-finite CAD pose loss")
                            loss = loss + pose_losses.total
                            batch_pose_components.append(pose_losses)
                            if auxiliary_pose_predictions.predictions:
                                auxiliary_mean = compute_auxiliary_pose_loss(
                                    auxiliary_pose_predictions,
                                    pose_targets,
                                    pose_matches,
                                    pose_config,
                                    batch_index=local_idx,
                                )
                                if auxiliary_mean is not None:
                                    loss = loss + args.pose_aux_loss_weight * auxiliary_mean
                                    batch_pose_aux_losses.append(auxiliary_mean)
                    # Head-only training can have no accepted pose match after
                    # IoU gating. Its frozen detection anchor then has no
                    # trainable graph, so omit that image from backward.
                    if not loss.requires_grad:
                        continue
                    batch_losses.append(loss)

                    scores = det_scores[local_idx]
                    if scores.numel() > 0:
                        _, nms_logits, nms_scores = apply_mask_nms(
                            box_preds[local_idx],
                            logits_mhw,
                            scores,
                            iou_threshold=args.nms_iou,
                        )
                        if nms_scores.numel() > 0:
                            topk = min(len(gt_targets) * max(1, args.matches_per_gt), nms_scores.numel())
                            top_idx = torch.topk(nms_scores, k=topk).indices
                            logits_sel = nms_logits[top_idx]
                            top_matches, top_iou = match_predictions_to_gts_greedy_k(
                                logits_sel,
                                gt_targets,
                                max_matches=None,
                                max_per_gt=args.matches_per_gt,
                            )
                            if top_matches and top_iou is not None:
                                matched_ious = [float(top_iou[g, p].item()) for g, p in top_matches]
                                batch_top_ious.append(sum(matched_ious) / len(matched_ious))
                            else:
                                batch_top_ious.append(0.0)
                        else:
                            batch_top_ious.append(0.0)
                    else:
                        batch_top_ious.append(0.0)

                    if should_save_debug and not debug_saved and data_idx == debug_target_idx:
                        debug_path = os.path.join(debug_dir, f"boxes_{epoch:03d}_{batch_step:06d}.png")
                        boxes_vis, masks_vis, scores_vis = apply_mask_nms(
                            box_preds[local_idx],
                            mask_preds[local_idx],
                            det_scores[local_idx],
                            iou_threshold=args.nms_iou,
                        )
                        save_debug_collage(
                            prepared[data_idx]["image_bgr"],
                            boxes_vis,
                            scores_vis,
                            masks_vis,
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
            running_top_ious.append(batch_top_iou)
            epoch_iou_sum += batch_top_iou
            epoch_iou_count += 1
            batch_loss.backward()
            batch_step += 1
            global_step += 1
            if global_step % args.grad_accum == 0:
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for group in optimizer.param_groups
                            for parameter in group["params"]
                            if parameter.grad is not None
                        ],
                        args.grad_clip_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(batch_loss.item())
            epoch_count += 1
            avg_loss = epoch_loss / epoch_count
            avg_epoch_iou = epoch_iou_sum / max(1, epoch_iou_count)
            append_metric(
                metrics_path,
                phase="train_batch",
                epoch=epoch,
                global_step=global_step,
                batch_step=batch_step,
                loss=float(batch_loss.item()),
                avg_loss=avg_loss,
                avg_iou=avg_epoch_iou,
                mask_loss=(
                    float(np.mean([value.mask.detach().float().item() for value in batch_detection_components]))
                    if batch_detection_components else ""
                ),
                bbox_loss=(
                    float(np.mean([value.bbox.detach().float().item() for value in batch_detection_components]))
                    if batch_detection_components else ""
                ),
                objectness_loss=(
                    float(np.mean([value.objectness.detach().float().item() for value in batch_detection_components]))
                    if batch_detection_components else ""
                ),
                pose_center_loss=(
                    float(np.mean([value.center.detach().float().item() for value in batch_pose_components]))
                    if batch_pose_components else ""
                ),
                pose_depth_loss=(
                    float(np.mean([value.depth.detach().float().item() for value in batch_pose_components]))
                    if batch_pose_components else ""
                ),
                pose_rotation_loss=(
                    float(np.mean([value.rotation.detach().float().item() for value in batch_pose_components]))
                    if batch_pose_components else ""
                ),
                pose_full_set_loss=(
                    float(np.mean([value.full_pose.detach().float().item() for value in batch_pose_components]))
                    if batch_pose_components else ""
                ),
                pose_quality_loss=(
                    float(np.mean([value.quality.detach().float().item() for value in batch_pose_components]))
                    if batch_pose_components else ""
                ),
                pose_aux_loss=(
                    float(np.mean([value.detach().float().item() for value in batch_pose_aux_losses]))
                    if batch_pose_aux_losses else ""
                ),
                rotation_error_deg=(
                    float(
                        np.rad2deg(
                            finite_mean(
                                [
                                    value.mean_rotation_error_rad.detach().float().item()
                                    for value in batch_pose_components
                                ]
                            )
                        )
                    )
                    if batch_pose_components else ""
                ),
                mean_surface_distance_norm=(
                    finite_mean(
                        [
                            value.mean_point_set_error_norm.detach().float().item()
                            for value in batch_pose_components
                        ]
                    )
                    if batch_pose_components else ""
                ),
                centroid_error_cm=(
                    100.0
                    * finite_mean(
                        [
                            value.mean_centroid_error_m.detach().float().item()
                            for value in batch_pose_components
                        ]
                    )
                    if batch_pose_components else ""
                ),
                translation_error_cm=(
                    float(100.0 * np.mean([value.mean_translation_error_m.detach().float().item() for value in batch_pose_components]))
                    if batch_pose_components else ""
                ),
                pose_match_iou_threshold=(args.pose_train_min_match_iou if args.enable_pose else ""),
                pose_assignment_coverage=(
                    batch_pose_total_matches / batch_pose_eligible_targets
                    if batch_pose_eligible_targets else ""
                ),
                pose_match_coverage=(
                    batch_pose_accepted_matches / batch_pose_eligible_targets
                    if batch_pose_eligible_targets else ""
                ),
                pose_match_acceptance_rate=(
                    batch_pose_accepted_matches / batch_pose_total_matches
                    if batch_pose_total_matches else ""
                ),
                pose_accepted_matches=(batch_pose_accepted_matches if args.enable_pose else ""),
                pose_total_matches=(batch_pose_total_matches if args.enable_pose else ""),
                eligible_samples=(batch_pose_eligible_targets if args.enable_pose else ""),
                samples=len(batch_losses),
            )

            if args.log_every > 0 and global_step % args.log_every == 0:
                running_avg = sum(running_losses) / max(1, len(running_losses))
                running_iou = sum(running_top_ious) / max(1, len(running_top_ious))
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={batch_loss.item():.4f} avg_loss={avg_loss:.4f} "
                    f"avg_iou={avg_epoch_iou:.4f} "
                    f"run5_loss={running_avg:.4f} run5_iou={running_iou:.4f}"
                    + (
                        f" pose_matches={batch_pose_accepted_matches}/{batch_pose_total_matches} "
                        f"accepted_iou={float(np.mean(batch_pose_match_ious)):.3f}"
                        if batch_pose_match_ious else ""
                    )
                    + (
                        f" pose_aux={float(np.mean([value.detach().float().item() for value in batch_pose_aux_losses])):.4f}"
                        if batch_pose_aux_losses else ""
                    )
                )
                save_path = str(checkpoints_dir / "finetune.pth")
                torch.save(
                    build_finetune_checkpoint(
                        detmodel,
                        optimizer,
                        args,
                        epoch=epoch,
                        global_step=global_step,
                        batch_step=batch_step,
                        pose_config=pose_config,
                        manifest_checksum=manifest_checksum,
                        catalog_checksums=catalog_checksums,
                        dataset_meta_checksums=dataset_meta_checksums,
                        annotation_checksums=annotation_checksums,
                        symmetry_pipeline_versions=symmetry_pipeline_versions,
                        point_set_checksums=point_set_checksums,
                        sampling_pipeline_versions=sampling_pipeline_versions,
                        sampling_parameter_checksums=sampling_parameter_checksums,
                        schema_checksums=schema_checksums,
                        schema_versions=schema_versions,
                    ),
                    save_path,
                )
        if epoch_count == 0:
            raise RuntimeError(
                f"Epoch {epoch} produced zero optimization steps; refusing to write an untrained "
                "checkpoint. Check exemplar render/mask resolution and supervised masks."
            )
        append_metric(
            metrics_path,
            phase="train_epoch",
            epoch=epoch,
            global_step=global_step,
            batch_step=batch_step,
            avg_loss=epoch_loss / epoch_count,
            avg_iou=epoch_iou_sum / max(1, epoch_iou_count),
            samples=epoch_count,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_path = str(checkpoints_dir / f"finetune_epoch_{epoch:03d}.pth")
            torch.save(
                build_finetune_checkpoint(
                    detmodel,
                    optimizer,
                    args,
                    epoch=epoch,
                    global_step=global_step,
                    batch_step=batch_step,
                    pose_config=pose_config,
                    manifest_checksum=manifest_checksum,
                    catalog_checksums=catalog_checksums,
                    dataset_meta_checksums=dataset_meta_checksums,
                    annotation_checksums=annotation_checksums,
                    symmetry_pipeline_versions=symmetry_pipeline_versions,
                    point_set_checksums=point_set_checksums,
                    sampling_pipeline_versions=sampling_pipeline_versions,
                    sampling_parameter_checksums=sampling_parameter_checksums,
                    schema_checksums=schema_checksums,
                    schema_versions=schema_versions,
                ),
                save_path,
            )
            print(f"Saved checkpoint to {save_path}")

        # Earlier epochs are validated at the start of the following epoch.
        # The final epoch has no following iteration, so evaluate it here too.
        if epoch == args.epochs and validation_config is not None:
            eval_avg, eval_count, eval_correct, eval_loss = run_exemplar_eval(
                detmodel, validation_config, device=device
            )
            append_metric(
                metrics_path,
                phase="validation",
                epoch=epoch,
                global_step=global_step,
                batch_step=batch_step,
                loss=eval_loss,
                avg_iou=eval_avg,
                correct_rate=eval_correct,
                samples=eval_count,
            )
            loss_part = f" val_loss={eval_loss:.4f}" if np.isfinite(eval_loss) else ""
            if eval_count > 0:
                print(
                    f"[eval] epoch={epoch}{loss_part} avg_iou={eval_avg:.4f} "
                    f"correct_rate={eval_correct:.3f} samples={eval_count}"
                )
            else:
                print(f"[eval] epoch={epoch}{loss_part} no valid samples")
            if args.enable_pose and validation_pose_entries:
                pose_metrics, conditional_pose_metrics = run_cad_pose_eval(
                    detmodel,
                    validation_pose_entries,
                    args,
                    device,
                    pose_config,
                    calibrate_temperature=True,
                )
                append_pose_metric_rows(
                    metrics_path,
                    phase="validation_pose_calibrated",
                    epoch=epoch,
                    global_step=global_step,
                    batch_step=batch_step,
                    metrics=pose_metrics,
                    conditional_metrics=conditional_pose_metrics,
                )
                calibrated_path = str(checkpoints_dir / "finetune_calibrated.pth")
                torch.save(
                    build_finetune_checkpoint(
                        detmodel,
                        optimizer,
                        args,
                        epoch=epoch,
                        global_step=global_step,
                        batch_step=batch_step,
                        pose_config=pose_config,
                        manifest_checksum=manifest_checksum,
                        catalog_checksums=catalog_checksums,
                        dataset_meta_checksums=dataset_meta_checksums,
                        annotation_checksums=annotation_checksums,
                        symmetry_pipeline_versions=symmetry_pipeline_versions,
                        point_set_checksums=point_set_checksums,
                        sampling_pipeline_versions=sampling_pipeline_versions,
                        sampling_parameter_checksums=sampling_parameter_checksums,
                        schema_checksums=schema_checksums,
                        schema_versions=schema_versions,
                    ),
                    calibrated_path,
                )
                print(f"[eval:pose:calibrated] {json.dumps(pose_metrics, sort_keys=True)}")
                if conditional_pose_metrics is not None:
                    print(
                        "[eval:pose:calibrated:iou] "
                        f"{json.dumps(conditional_pose_metrics, sort_keys=True)}"
                    )
                print(f"Saved calibrated checkpoint to {calibrated_path}")

    print("Done.")


if __name__ == "__main__":
    main()
