#!/usr/bin/env python3
"""Build and validate a versioned manifest for SAM exemplar datasets.

The builder supports the original wrist/side brick datasets and flat Perseve
pose datasets supplied as ``--pose-dataset DATASET_ID=RELATIVE_PATH``. Flat
pose rows use the sidecar ``scene_id`` as their provenance group, so related
frames cannot cross train/validation/test splits. Each dataset is shuffled
independently with a stable hash-derived seed before applying the requested
ratios.

The output contains paths relative to ``--data-root`` and is validated again
after writing. Existing versioned manifests are immutable by default: callers
must validate them, choose a new output version, or explicitly pass
``--overwrite``. The utility writes no image data and never rearranges the
source datasets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from dataset_manifest import ManifestRow, assign_group_splits, load_manifest, write_manifest


FRAME_PATTERN = re.compile(r"instance_segmentation_(\d+)\.png$")
POSE_FRAME_PATTERN = re.compile(r"pose_annotations_(.+)\.json$")


def parse_ratios(value: str) -> Tuple[float, float, float]:
    try:
        ratios = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ratios: {value}") from exc
    if len(ratios) != 3 or any(item < 0 for item in ratios) or sum(ratios) <= 0:
        raise argparse.ArgumentTypeError("Ratios must contain three non-negative values with a positive sum.")
    return ratios  # type: ignore[return-value]


def discover_camera_frames(dataset_root: Path, camera_dirs: Iterable[Path]) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    for camera_dir in sorted(camera_dirs):
        if not camera_dir.is_dir() or camera_dir.parent != dataset_root:
            raise ValueError(f"Camera directory must be directly under {dataset_root}: {camera_dir}")
        for inst_path in sorted(camera_dir.glob("instance_segmentation_*.png")):
            match = FRAME_PATTERN.fullmatch(inst_path.name)
            if match:
                samples.append((camera_dir.name, match.group(1)))
    if not samples:
        raise ValueError(f"No processed samples found under {dataset_root}")
    return samples


def load_wrist_groups(metadata_path: Path) -> Dict[str, str]:
    with metadata_path.open("r") as handle:
        metadata = json.load(handle)
    frame_groups: Dict[str, str] = {}
    for scene in metadata.get("scenes", []):
        scene_id = str(scene.get("scene_id", "")).strip()
        if not scene_id:
            raise ValueError("Wrist metadata contains a scene without scene_id.")
        for capture in scene.get("captures", []):
            if not capture.get("accepted", False):
                continue
            raw_frame_id = capture.get("frame_id")
            if raw_frame_id is None:
                raise ValueError(f"Accepted wrist capture in {scene_id} has no frame_id.")
            frame_id = f"{int(raw_frame_id):04d}"
            prior = frame_groups.setdefault(frame_id, scene_id)
            if prior != scene_id:
                raise ValueError(f"Wrist frame {frame_id} maps to both {prior} and {scene_id}.")
    if not frame_groups:
        raise ValueError(f"No accepted captures found in {metadata_path}")
    return frame_groups


def build_rows(
    data_root: Path,
    wrist_path: str,
    side_path: str,
    ratios: Tuple[float, float, float],
    seed: int,
) -> List[ManifestRow]:
    rows: List[ManifestRow] = []

    wrist_root = data_root / wrist_path
    wrist_samples = discover_camera_frames(wrist_root, [wrist_root / "Wrist_Camera"])
    wrist_groups = load_wrist_groups(wrist_root / "metadata.json")
    discovered_wrist_ids = {frame_id for _, frame_id in wrist_samples}
    metadata_wrist_ids = set(wrist_groups)
    if discovered_wrist_ids != metadata_wrist_ids:
        missing_metadata = sorted(discovered_wrist_ids - metadata_wrist_ids)[:10]
        missing_processed = sorted(metadata_wrist_ids - discovered_wrist_ids)[:10]
        raise ValueError(
            "Wrist processed/metadata coverage differs: "
            f"missing_metadata={missing_metadata}, missing_processed={missing_processed}"
        )
    wrist_splits = assign_group_splits("wrist_type2", wrist_groups.values(), ratios, seed)
    for camera_dir, frame_id in wrist_samples:
        group_id = wrist_groups[frame_id]
        rows.append(
            ManifestRow("wrist_type2", wrist_path, camera_dir, frame_id, group_id, wrist_splits[group_id])
        )

    side_root = data_root / side_path
    side_samples = discover_camera_frames(side_root, side_root.glob("Side_Camera_*"))
    side_group_ids = {frame_id for _, frame_id in side_samples}
    side_splits = assign_group_splits("yaw20_side", side_group_ids, ratios, seed)
    for camera_dir, frame_id in side_samples:
        rows.append(ManifestRow("yaw20_side", side_path, camera_dir, frame_id, frame_id, side_splits[frame_id]))

    return sorted(rows, key=lambda row: (row.dataset_id, row.group_id, row.camera_dir, row.frame_id))


def parse_pose_dataset_spec(value: str) -> Tuple[str, str]:
    """Parse ``DATASET_ID=RELATIVE_PATH`` without permitting path escapes."""

    dataset_id, separator, dataset_path = value.partition("=")
    dataset_id, dataset_path = dataset_id.strip(), dataset_path.strip()
    path = Path(dataset_path)
    if not separator or not dataset_id or not dataset_path:
        raise argparse.ArgumentTypeError(
            "Pose dataset must use DATASET_ID=RELATIVE_PATH, for example abc_pose=v2_sdg_output"
        )
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            f"Pose dataset path must be relative to --data-root: {dataset_path!r}"
        )
    return dataset_id, path.as_posix()


def build_pose_rows(
    data_root: Path,
    pose_datasets: Iterable[Tuple[str, str]],
    ratios: Tuple[float, float, float],
    seed: int,
) -> List[ManifestRow]:
    """Discover flat pose frames and split them by their declared scene IDs."""

    rows: List[ManifestRow] = []
    seen_dataset_ids: set[str] = set()
    for dataset_id, dataset_path in pose_datasets:
        if dataset_id in seen_dataset_ids:
            raise ValueError(f"Duplicate pose dataset ID: {dataset_id}")
        seen_dataset_ids.add(dataset_id)
        dataset_root = data_root / dataset_path
        if not dataset_root.is_dir():
            raise FileNotFoundError(dataset_root)

        samples: List[Tuple[str, str]] = []
        seen_frame_ids: set[str] = set()
        for sidecar_path in sorted(dataset_root.glob("pose_annotations_*.json")):
            match = POSE_FRAME_PATTERN.fullmatch(sidecar_path.name)
            if match is None:
                continue
            frame_id = match.group(1)
            if frame_id in seen_frame_ids:
                raise ValueError(f"Duplicate frame ID {frame_id!r} under {dataset_root}")
            with sidecar_path.open(encoding="utf-8") as handle:
                sidecar = json.load(handle)
            if str(sidecar.get("frame_id", "")) != frame_id:
                raise ValueError(
                    f"{sidecar_path.name} declares frame_id={sidecar.get('frame_id')!r}, "
                    f"expected {frame_id!r}"
                )
            scene_id = str(sidecar.get("scene_id", "")).strip()
            if not scene_id:
                raise ValueError(f"{sidecar_path.name} has no scene_id")
            samples.append((frame_id, scene_id))
            seen_frame_ids.add(frame_id)
        if not samples:
            raise ValueError(f"No pose_annotations_*.json files found under {dataset_root}")

        group_splits = assign_group_splits(
            dataset_id,
            (scene_id for _, scene_id in samples),
            ratios,
            seed,
        )
        rows.extend(
            ManifestRow(
                dataset_id=dataset_id,
                dataset_path=dataset_path,
                camera_dir=".",
                frame_id=frame_id,
                group_id=scene_id,
                split=group_splits[scene_id],
            )
            for frame_id, scene_id in samples
        )
    return sorted(rows, key=lambda row: (row.dataset_id, row.group_id, row.frame_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--wrist-dataset", default="random_type2_wrist_1000x4_10_buffer_final_refined"
    )
    parser.add_argument(
        "--side-dataset", default="run_500_scenes_yaw20_not_stud_aligned_repaired"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=parse_ratios, default=(0.8, 0.1, 0.1))
    parser.add_argument(
        "--pose-dataset",
        action="append",
        type=parse_pose_dataset_spec,
        default=[],
        metavar="DATASET_ID=RELATIVE_PATH",
        help=(
            "Build from a flat Perseve pose dataset below --data-root. May be repeated. "
            "When supplied, the legacy wrist/side arguments are unused."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.validate_only:
        _, summary = load_manifest(output, data_root, validate_files=True)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace versioned manifest {output}; use --validate-only, a new version, or --overwrite."
        )
    if args.pose_dataset:
        rows = build_pose_rows(data_root, args.pose_dataset, args.ratios, args.seed)
    else:
        rows = build_rows(data_root, args.wrist_dataset, args.side_dataset, args.ratios, args.seed)
    write_manifest(output, rows)
    _, summary = load_manifest(output, data_root, validate_files=True)
    split_counts = Counter(row.split for row in rows)
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Split rows: {dict(sorted(split_counts.items()))}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
