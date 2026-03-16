#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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


def normalize_dataset_roots(dataset_root: object) -> List[str]:
    if dataset_root is None:
        return []
    items: List[object]
    if isinstance(dataset_root, (list, tuple)):
        items = list(dataset_root)
    else:
        items = [dataset_root]
    roots: List[str] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            path = str(item[0]).strip()
            if path:
                roots.append(path)
            continue
        for part in str(item).split(","):
            part = part.strip()
            if part:
                roots.append(part)
    return roots


def _slugify(path: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", path.strip().strip("/"))
    return cleaned or "dataset"


def iter_frames(dataset_root: str) -> Iterable[Tuple[str, Path, Path, Path]]:
    dataset_path = Path(dataset_root).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_root)
    pattern = re.compile(r"instance_segmentation_(\d{4})\.png$")
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
        yield frame_id, rgb_path, inst_path, mapping_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dataset_filter CSV containing frames whose object ids differ from the previous frame."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        nargs="+",
        default=["/sata1/data/kevin/v2_dataset/v2_multi_gt_merged_345"],
        help="Dataset roots (space or comma separated).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="dataset_filter_{slug}.csv",
        help="Output CSV filename template written in repo root. Use {slug} for dataset slug.",
    )
    parser.add_argument(
        "--include_object_ids",
        action="store_true",
        help="Include a pipe-delimited object_ids column in the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_roots = normalize_dataset_roots(args.dataset_root)
    if not dataset_roots:
        raise ValueError("No dataset roots provided.")

    total_frames = 0
    total_valid = 0
    total_invalid = 0

    header = ["dataset_root", "frame_id", "rgb_path", "inst_path", "mapping_path"]
    if args.include_object_ids:
        header.append("object_ids")

    for root in dataset_roots:
        root_path = Path(root).expanduser().resolve()
        slug = _slugify(str(root_path))
        filename = args.output_csv.format(slug=slug)
        output_path = Path.cwd() / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prev_ids: Optional[Tuple[str, ...]] = None
        root_total = 0
        root_valid = 0
        root_invalid = 0

        with open(output_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)

            for frame_id, rgb_path, inst_path, mapping_path in iter_frames(root):
                root_total += 1
                total_frames += 1

                if not (rgb_path.is_file() and mapping_path.is_file()):
                    root_invalid += 1
                    total_invalid += 1
                    continue

                try:
                    color_map = load_color_mapping(mapping_path)
                except FileNotFoundError:
                    root_invalid += 1
                    total_invalid += 1
                    continue

                object_ids = tuple(sorted(color_map.keys()))
                if not object_ids:
                    root_invalid += 1
                    total_invalid += 1
                    prev_ids = object_ids
                    continue

                is_valid = prev_ids is None or object_ids != prev_ids
                prev_ids = object_ids

                if is_valid:
                    row = [root, frame_id, str(rgb_path), str(inst_path), str(mapping_path)]
                    if args.include_object_ids:
                        row.append("|".join(object_ids))
                    writer.writerow(row)
                    root_valid += 1
                    total_valid += 1
                else:
                    root_invalid += 1
                    total_invalid += 1

        print(f"[{root}] total={root_total} valid={root_valid} invalid={root_invalid} wrote={output_path}")

    print(f"[all] total={total_frames} valid={total_valid} invalid={total_invalid}")


if __name__ == "__main__":
    main()
