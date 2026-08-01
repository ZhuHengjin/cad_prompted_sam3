#!/usr/bin/env python3
"""Flatten a nested ABC STL corpus into resumable Blender render inputs.

The output directory contains symlinks only. Existing entries are accepted only
when they are symlinks that already resolve to the corresponding source mesh.
Optionally, a completed render directory can be checked for all twelve Blender
views and their masks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


VIEW_IDS = tuple(range(12))
OUTPUT_BASE_SUFFIX = "stl_base"
MAX_REPORTED_RENDER_ISSUES = 20


class PreparationError(ValueError):
    """Raised when the source or staging directory violates the safety contract."""


class RenderValidationError(PreparationError):
    """Raised when one or more expected render image/mask pairs are incomplete."""

    def __init__(self, report: dict) -> None:
        self.report = report
        super().__init__(
            f"Render validation found {report['incomplete_pair_count']} incomplete "
            f"image/mask pairs across {report['incomplete_object_count']} objects"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Root of the nested ABC STL corpus; every STL below it is processed.",
    )
    parser.add_argument(
        "staging_dir",
        type=Path,
        help="Flat directory of symlinks to create for blender_renderer.py.",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        help="After staging, validate all 12 image/mask render pairs for every STL.",
    )
    return parser.parse_args(argv)


def discover_stls(source_dir: Path) -> list[Path]:
    """Return every STL below ``source_dir``, with globally unique stems."""

    if not source_dir.is_dir():
        raise PreparationError(f"Source directory does not exist or is not a directory: {source_dir}")

    meshes = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".stl"
    )
    if not meshes:
        raise PreparationError(f"No STL files found recursively in {source_dir}")

    by_stem: dict[str, Path] = {}
    duplicates: list[tuple[Path, Path]] = []
    for mesh_path in meshes:
        stem_key = mesh_path.stem.casefold()
        previous = by_stem.get(stem_key)
        if previous is None:
            by_stem[stem_key] = mesh_path
        else:
            duplicates.append((previous, mesh_path))

    if duplicates:
        examples = "; ".join(f"{first} and {second}" for first, second in duplicates[:5])
        remainder = len(duplicates) - 5
        suffix = f"; and {remainder} more" if remainder > 0 else ""
        raise PreparationError(
            "ABC STL filename stems must be unique (case-insensitive); conflicts: "
            f"{examples}{suffix}"
        )
    return meshes


def prepare_symlink_directory(source_dir: Path, staging_dir: Path) -> tuple[list[Path], dict]:
    """Create or resume a flat, symlink-only view of a nested STL corpus."""

    requested_source_dir = source_dir.expanduser().absolute()
    source_dir = requested_source_dir.resolve(strict=False)
    requested_staging_dir = staging_dir.expanduser().absolute()
    if requested_staging_dir.is_symlink() and not requested_staging_dir.exists():
        raise PreparationError(f"Staging directory is a broken symlink: {requested_staging_dir}")
    staging_dir = requested_staging_dir.resolve(strict=False)
    if (
        requested_staging_dir == requested_source_dir
        or requested_staging_dir.is_relative_to(requested_source_dir)
        or staging_dir == source_dir
        or staging_dir.is_relative_to(source_dir)
    ):
        raise PreparationError(
            f"Staging directory must not be the source directory or live inside it: {staging_dir}"
        )

    meshes = discover_stls(source_dir)
    expected = {mesh_path.name: mesh_path for mesh_path in meshes}

    if staging_dir.exists() and not staging_dir.is_dir():
        raise PreparationError(f"Staging path exists but is not a directory: {staging_dir}")

    existing_count = 0
    conflicts: list[str] = []
    if staging_dir.is_dir():
        for entry in staging_dir.iterdir():
            source_path = expected.get(entry.name)
            if source_path is None:
                conflicts.append(f"unexpected entry {entry}")
                continue
            if not entry.is_symlink():
                conflicts.append(f"expected a symlink but found another entry type at {entry}")
                continue
            try:
                actual_target = entry.resolve(strict=True)
            except (FileNotFoundError, RuntimeError, OSError):
                conflicts.append(f"stale or unreadable symlink {entry} -> {os.readlink(entry)}")
                continue
            if actual_target != source_path.resolve(strict=True):
                conflicts.append(
                    f"conflicting symlink {entry} -> {os.readlink(entry)}; expected {source_path}"
                )
                continue
            existing_count += 1

    if conflicts:
        examples = "; ".join(conflicts[:5])
        remainder = len(conflicts) - 5
        suffix = f"; and {remainder} more" if remainder > 0 else ""
        raise PreparationError(
            "Staging directory contains stale or conflicting entries; nothing was changed: "
            f"{examples}{suffix}"
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    created_count = 0
    for output_name, source_path in sorted(expected.items()):
        link_path = staging_dir / output_name
        if link_path.is_symlink():
            continue
        try:
            link_path.symlink_to(source_path)
        except FileExistsError as exc:
            raise PreparationError(
                f"Refusing to overwrite an entry created during staging: {link_path}"
            ) from exc
        created_count += 1

    summary = {
        "source_dir": str(source_dir),
        "staging_dir": str(staging_dir),
        "stl_count": len(meshes),
        "symlinks": {
            "created": created_count,
            "existing": existing_count,
            "total": len(meshes),
        },
    }
    return meshes, summary


def _render_pair_candidates(render_dir: Path, stem: str, view_id: int) -> list[tuple[Path, Path]]:
    view_labels = (f"{view_id:02d}", str(view_id))
    candidates: list[tuple[Path, Path]] = []
    for view_label in dict.fromkeys(view_labels):
        base = f"{stem}_{OUTPUT_BASE_SUFFIX}_{view_label}"
        candidates.append(
            (render_dir / f"{base}.png", render_dir / f"{base}_mask.png")
        )
    return candidates


def validate_render_directory(meshes: Sequence[Path], render_dir: Path) -> dict:
    """Require one complete padded or unpadded image/mask pair for every view."""

    render_dir = render_dir.expanduser().resolve(strict=False)
    if not render_dir.is_dir():
        raise PreparationError(f"Render directory does not exist or is not a directory: {render_dir}")

    incomplete_pair_count = 0
    incomplete_objects: set[str] = set()
    issue_examples: list[dict] = []
    for mesh_path in meshes:
        stem = mesh_path.stem
        for view_id in VIEW_IDS:
            candidates = _render_pair_candidates(render_dir, stem, view_id)
            if any(image_path.is_file() and mask_path.is_file() for image_path, mask_path in candidates):
                continue

            incomplete_pair_count += 1
            incomplete_objects.add(stem)
            if len(issue_examples) < MAX_REPORTED_RENDER_ISSUES:
                issue_examples.append(
                    {
                        "object_id": stem,
                        "view_id": view_id,
                        "candidates": [
                            {
                                "image": image_path.name,
                                "image_exists": image_path.is_file(),
                                "mask": mask_path.name,
                                "mask_exists": mask_path.is_file(),
                            }
                            for image_path, mask_path in candidates
                        ],
                    }
                )

    report = {
        "render_dir": str(render_dir),
        "object_count": len(meshes),
        "views_per_object": len(VIEW_IDS),
        "expected_pair_count": len(meshes) * len(VIEW_IDS),
        "complete_pair_count": len(meshes) * len(VIEW_IDS) - incomplete_pair_count,
        "incomplete_pair_count": incomplete_pair_count,
        "incomplete_object_count": len(incomplete_objects),
        "issue_examples": issue_examples,
    }
    if incomplete_pair_count:
        raise RenderValidationError(report)
    return report


def run(args: argparse.Namespace) -> dict:
    meshes, summary = prepare_symlink_directory(args.source_dir, args.staging_dir)
    summary["render_validation"] = (
        validate_render_directory(meshes, args.render_dir) if args.render_dir is not None else None
    )
    return {"status": "ok", **summary}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args)
    except RenderValidationError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": str(exc),
                    "render_validation": exc.report,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    except (PreparationError, OSError) as exc:
        print(
            json.dumps({"status": "error", "error": str(exc)}, indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
