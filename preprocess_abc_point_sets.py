#!/usr/bin/env python3
"""Batch-preprocess reusable ABC USD assets into canonical CAD point sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from muggled_sam.v3_sam.cad_pose.point_sets import (
    build_point_set_arrays,
    save_point_set_artifact,
    sha256_file,
)
from preprocess_cad_point_set import load_usd_triangles_with_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--point_count", type=int, default=4096)
    parser.add_argument("--query_count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N assets.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    meshes = sorted(input_dir.glob("*.usd"))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        meshes = meshes[: args.limit]
    if not meshes:
        raise ValueError(f"No .usd assets found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "point_sets.json"
    manifest = _load_manifest(manifest_path)

    completed = 0
    skipped = 0
    for index, mesh_path in enumerate(meshes, start=1):
        cad_id = mesh_path.stem
        output_path = output_dir / f"{cad_id}.npz"
        mesh_relative_path = Path(os.path.relpath(mesh_path, output_dir)).as_posix()
        mesh_sha256 = sha256_file(mesh_path)
        existing = manifest["objects"].get(cad_id)
        if not args.overwrite and output_path.exists():
            if (
                existing is not None
                and _matches_request(
                    existing,
                    args,
                    mesh_relative_path=mesh_relative_path,
                    mesh_sha256=mesh_sha256,
                )
                and str(existing.get("point_set", {}).get("sha256", "")).lower()
                == sha256_file(output_path).lower()
            ):
                skipped += 1
                print(f"[{index}/{len(meshes)}] skip {cad_id}", flush=True)
                continue
            raise ValueError(
                f"Existing artifact is incompatible or untracked: {output_path}; "
                "pass --overwrite to replace it"
            )

        triangles, source_to_meters, source_up_axis, center_m, dimensions_m = (
            load_usd_triangles_with_bounds(mesh_path)
        )
        rotation = source_to_z_up_rotation(source_up_axis)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = -(rotation @ center_m)
        canonical = apply_metric_transform(triangles, source_to_meters, transform)
        point_set = build_point_set_arrays(
            canonical,
            point_count=args.point_count,
            query_count=args.query_count,
            seed=args.seed,
        )
        _save_atomically(output_path, point_set)
        parameters = {
            "sampling_method": "surface_area_deterministic_v1",
            "point_count": args.point_count,
            "query_count": args.query_count,
            "seed": args.seed,
            "numpy_version": np.__version__,
            "source_to_meters": source_to_meters,
            "source_up_axis": source_up_axis,
            "target_up_axis": "Z",
            "T_cad_from_source_meters": transform.tolist(),
        }
        parameters_sha256 = hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        canonical_dimensions = np.abs(rotation) @ dimensions_m
        manifest["objects"][cad_id] = {
            "mesh_path": mesh_relative_path,
            "mesh_sha256": mesh_sha256,
            "source_to_meters": source_to_meters,
            "T_cad_from_source_meters": transform.tolist(),
            "bbox_min_m": (-0.5 * canonical_dimensions).tolist(),
            "bbox_max_m": (0.5 * canonical_dimensions).tolist(),
            "base_dimensions_m": canonical_dimensions.tolist(),
            "canonical_origin": "local_aabb_center",
            "point_set": {
                "path": output_path.name,
                "sha256": sha256_file(output_path),
                "point_count": args.point_count,
                "sampling_method": parameters["sampling_method"],
                "sampling_parameters_sha256": parameters_sha256,
                "surface_centroid_m": point_set.surface_centroid_m.tolist(),
            },
            "sampling_parameters": parameters,
        }
        _write_json_atomically(manifest_path, manifest)
        completed += 1
        print(f"[{index}/{len(meshes)}] wrote {output_path.name}", flush=True)

    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "selected": len(meshes),
                "completed": completed,
                "skipped": skipped,
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def source_to_z_up_rotation(source_up_axis: str) -> np.ndarray:
    """Return the right-handed source-axis to Isaac Z-up rotation."""

    if source_up_axis.upper() == "Z":
        return np.eye(3, dtype=np.float64)
    if source_up_axis.upper() == "Y":
        return np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported USD up axis {source_up_axis!r}; expected Y or Z")


def apply_metric_transform(
    triangles: np.ndarray,
    source_to_meters: float,
    transform: np.ndarray,
) -> np.ndarray:
    points_m = np.asarray(triangles, dtype=np.float64).reshape(-1, 3) * source_to_meters
    homogeneous = np.concatenate((points_m, np.ones((len(points_m), 1))), axis=1)
    return (homogeneous @ transform.T)[:, :3].reshape(-1, 3, 3)


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": "1.0.0", "objects": {}}
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != "1.0.0" or not isinstance(value.get("objects"), dict):
        raise ValueError(f"Unsupported point-set manifest: {path}")
    return value


def _matches_request(
    entry: dict,
    args: argparse.Namespace,
    *,
    mesh_relative_path: str,
    mesh_sha256: str,
) -> bool:
    parameters = entry.get("sampling_parameters", {})
    return (
        entry.get("mesh_path") == mesh_relative_path
        and str(entry.get("mesh_sha256", "")).lower() == mesh_sha256.lower()
        and entry.get("point_set", {}).get("point_count") == args.point_count
        and parameters.get("point_count") == args.point_count
        and parameters.get("query_count") == args.query_count
        and parameters.get("seed") == args.seed
        and parameters.get("target_up_axis") == "Z"
    )


def _save_atomically(path: Path, point_set) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    save_point_set_artifact(temporary, point_set)
    temporary.replace(path)


def _write_json_atomically(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
