#!/usr/bin/env python3
"""Validate Perseve pose-v1 sidecars referenced by a SAM3 dataset manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dataset_manifest import load_manifest, manifest_sha256
from muggled_sam.v3_sam.cad_pose.dataset import load_perseve_pose_sample, validate_scale_sharing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument(
        "--splits",
        default="train,validation,test",
        help="Comma-separated manifest splits to validate.",
    )
    parser.add_argument("--skip_pixels", action="store_true", help="Skip PNG/mapping/box joins (schema and geometry only).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, manifest_summary = load_manifest(args.manifest, args.data_root, validate_files=True)
    selected_splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    selected_rows = [row for row in rows if row.split in selected_splits]
    if not selected_rows:
        raise ValueError(f"No manifest rows matched splits {sorted(selected_splits)}")

    samples = []
    states: Counter[str] = Counter()
    eligible = 0
    for row in selected_rows:
        camera_root = args.data_root / row.dataset_path / row.camera_dir
        sample = load_perseve_pose_sample(camera_root, row.frame_id, validate_pixels=not args.skip_pixels)
        samples.append(sample)
        for instance in sample.frame.instances:
            states[instance.annotation_state] += 1
            eligible += int(instance.pose_training_eligible)
    validate_scale_sharing(samples)

    report = {
        "manifest_sha256": manifest_sha256(args.manifest),
        "manifest_summary": manifest_summary,
        "validated_splits": sorted(selected_splits),
        "validated_frames": len(samples),
        "annotation_states": dict(sorted(states.items())),
        "pose_training_eligible_instances": eligible,
        "catalog_checksums": sorted({sample.catalog_checksum for sample in samples}),
        "dataset_meta_checksums": sorted({sample.dataset_meta_checksum for sample in samples}),
        "schema_versions": sorted({sample.frame.schema_version for sample in samples}),
        "schema_checksums": sorted({checksum for sample in samples for checksum in sample.schema_checksums.values()}),
        "symmetry_pipeline_versions": sorted(
            {version for sample in samples for version in sample.symmetry_pipeline_versions}
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
