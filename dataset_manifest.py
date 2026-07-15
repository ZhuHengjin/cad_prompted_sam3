#!/usr/bin/env python3
"""Manifest primitives for reproducible multi-dataset exemplar fine-tuning.

The module defines the CSV schema shared by manifest generation and training,
validates dataset-qualified sample and group identities, resolves every row to
its RGB/instance/mapping triplet, and calculates reproducibility summaries and
checksums. Split membership is enforced at ``(dataset_id, group_id)`` scope so
related views cannot leak while coincident frame numbers in unrelated datasets
remain independent. It also provides deterministic per-dataset group splitting
and equal-domain epoch sampling with replacement for imbalanced datasets.

This module deliberately has no OpenCV or PyTorch dependency, allowing dataset
manifests to be generated and audited on lightweight preprocessing machines.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


VALID_SPLITS = ("train", "validation", "test")
MANIFEST_FIELDS = ("dataset_id", "dataset_path", "camera_dir", "frame_id", "group_id", "split")


@dataclass(frozen=True)
class ManifestRow:
    dataset_id: str
    dataset_path: str
    camera_dir: str
    frame_id: str
    group_id: str
    split: str

    @property
    def sample_key(self) -> Tuple[str, str, str]:
        return self.dataset_id, self.camera_dir, self.frame_id

    @property
    def group_key(self) -> Tuple[str, str]:
        return self.dataset_id, self.group_id


def _clean_relative_path(value: str, field: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a non-empty path relative to --data_root: {value!r}")
    return path.as_posix()


def sample_paths(row: ManifestRow, data_root: Path) -> Tuple[Path, Path, Path]:
    camera_root = data_root / row.dataset_path / row.camera_dir
    rgb_path = camera_root / f"rgb_{row.frame_id}.png"
    if not rgb_path.is_file():
        jpg_path = camera_root / f"rgb_{row.frame_id}.jpg"
        if jpg_path.is_file():
            rgb_path = jpg_path
    inst_path = camera_root / f"instance_segmentation_{row.frame_id}.png"
    mapping_path = camera_root / f"instance_segmentation_mapping_{row.frame_id}.json"
    return rgb_path, inst_path, mapping_path


def validate_manifest_rows(
    rows: Sequence[ManifestRow], data_root: Path, *, validate_files: bool = True
) -> Dict[str, object]:
    if not rows:
        raise ValueError("Manifest contains no rows.")
    data_root = data_root.expanduser().resolve()
    dataset_paths: Dict[str, str] = {}
    sample_keys = set()
    group_splits: Dict[Tuple[str, str], str] = {}
    counts = Counter()
    group_counts: Dict[str, set] = defaultdict(set)

    for row in rows:
        if not row.dataset_id or not row.frame_id or not row.group_id or not row.camera_dir:
            raise ValueError(f"Manifest row has an empty required value: {row}")
        if row.split not in VALID_SPLITS:
            raise ValueError(f"Invalid split {row.split!r}; expected one of {VALID_SPLITS}.")
        _clean_relative_path(row.dataset_path, "dataset_path")
        _clean_relative_path(row.camera_dir, "camera_dir")
        prior_path = dataset_paths.setdefault(row.dataset_id, row.dataset_path)
        if prior_path != row.dataset_path:
            raise ValueError(
                f"Dataset {row.dataset_id!r} has inconsistent paths: {prior_path!r} and {row.dataset_path!r}."
            )
        if row.sample_key in sample_keys:
            raise ValueError(f"Duplicate manifest sample: {row.sample_key}")
        sample_keys.add(row.sample_key)
        previous_split = group_splits.setdefault(row.group_key, row.split)
        if previous_split != row.split:
            raise ValueError(
                f"Group {row.group_key} crosses splits: {previous_split!r} and {row.split!r}."
            )
        if validate_files:
            missing = [str(path) for path in sample_paths(row, data_root) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Manifest sample {row.sample_key} is missing files: {missing}")
        counts[(row.dataset_id, row.split)] += 1
        group_counts[f"{row.dataset_id}:{row.split}"].add(row.group_id)

    return {
        "rows": len(rows),
        "datasets": sorted(dataset_paths),
        "sample_counts": {f"{dataset}:{split}": count for (dataset, split), count in sorted(counts.items())},
        "group_counts": {key: len(value) for key, value in sorted(group_counts.items())},
    }


def load_manifest(
    manifest_path: Path, data_root: Path, *, validate_files: bool = True
) -> Tuple[List[ManifestRow], Dict[str, object]]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows: List[ManifestRow] = []
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = [field for field in MANIFEST_FIELDS if field not in (reader.fieldnames or [])]
        if missing_fields:
            raise ValueError(f"Manifest is missing columns: {missing_fields}")
        for line_number, raw in enumerate(reader, start=2):
            values = {field: (raw.get(field) or "").strip() for field in MANIFEST_FIELDS}
            try:
                rows.append(ManifestRow(**values))
            except TypeError as exc:
                raise ValueError(f"Invalid manifest row at line {line_number}: {raw}") from exc
    summary = validate_manifest_rows(rows, data_root, validate_files=validate_files)
    return rows, summary


def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_group_splits(
    dataset_id: str,
    group_ids: Iterable[str],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Mapping[str, str]:
    """Assign groups deterministically without coupling independent datasets."""
    import random

    groups = sorted(set(group_ids))
    if not groups:
        raise ValueError(f"Dataset {dataset_id!r} has no groups.")
    if len(ratios) != 3 or any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError(f"Invalid split ratios: {ratios}")
    normalized = tuple(value / sum(ratios) for value in ratios)
    stable_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{dataset_id}".encode("utf-8")).digest()[:8], "big"
    )
    random.Random(stable_seed).shuffle(groups)
    total = len(groups)
    train_count = int(round(total * normalized[0]))
    validation_count = int(round(total * normalized[1]))
    train_count = min(max(train_count, 0), total)
    validation_count = min(max(validation_count, 0), total - train_count)
    assignments: Dict[str, str] = {}
    for group_id in groups[:train_count]:
        assignments[group_id] = "train"
    for group_id in groups[train_count : train_count + validation_count]:
        assignments[group_id] = "validation"
    for group_id in groups[train_count + validation_count :]:
        assignments[group_id] = "test"
    return assignments


def balanced_epoch_entries(
    entries_by_dataset: Mapping[str, Sequence[dict]], epoch_size: int, seed: int, epoch: int
) -> List[dict]:
    """Draw an equal-domain epoch deterministically, sampling with replacement."""
    import random

    dataset_ids = sorted(entries_by_dataset)
    if not dataset_ids:
        return []
    if any(not entries_by_dataset[dataset_id] for dataset_id in dataset_ids):
        empty = [dataset_id for dataset_id in dataset_ids if not entries_by_dataset[dataset_id]]
        raise ValueError(f"Cannot balance datasets with empty training pools: {empty}")
    rng = random.Random(seed + epoch)
    base, remainder = divmod(epoch_size, len(dataset_ids))
    sampled: List[dict] = []
    for index, dataset_id in enumerate(dataset_ids):
        count = base + (1 if index < remainder else 0)
        pool = entries_by_dataset[dataset_id]
        sampled.extend(pool[rng.randrange(len(pool))] for _ in range(count))
    rng.shuffle(sampled)
    return sampled
