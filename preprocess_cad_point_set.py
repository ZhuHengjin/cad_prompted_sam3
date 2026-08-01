#!/usr/bin/env python3
"""Preprocess an STL or USD render mesh into a deterministic pose point set."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from muggled_sam.v3_sam.cad_pose.point_sets import (
    build_point_set_arrays,
    save_point_set_artifact,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--point_count", type=int, default=4096)
    parser.add_argument("--query_count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source_to_meters",
        type=float,
        default=None,
        help="Source-unit multiplier. USD defaults to stage meters-per-unit; STL defaults to 1.",
    )
    parser.add_argument(
        "--T_cad_from_source_meters",
        type=float,
        nargs=16,
        default=np.eye(4, dtype=np.float64).reshape(-1).tolist(),
        metavar=tuple(f"M{row}{column}" for row in range(4) for column in range(4)),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.expanduser().resolve()
    suffix = mesh_path.suffix.lower()
    if suffix == ".stl":
        triangles = load_stl_triangles(mesh_path)
        inferred_scale = 1.0
    elif suffix in {".usd", ".usda", ".usdc"}:
        triangles, inferred_scale = load_usd_triangles(mesh_path)
    else:
        raise ValueError(f"Unsupported mesh extension {suffix!r}; expected STL or USD")
    source_to_meters = inferred_scale if args.source_to_meters is None else args.source_to_meters
    if not np.isfinite(source_to_meters) or source_to_meters <= 0:
        raise ValueError("source_to_meters must be finite and positive")
    transform = np.asarray(args.T_cad_from_source_meters, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(transform).all() or not np.allclose(transform[3], (0, 0, 0, 1)):
        raise ValueError("T_cad_from_source_meters must be a finite homogeneous transform")
    points_m = triangles.reshape(-1, 3) * source_to_meters
    homogeneous = np.concatenate((points_m, np.ones((len(points_m), 1))), axis=1)
    canonical = (homogeneous @ transform.T)[:, :3].reshape(-1, 3, 3)
    point_set = build_point_set_arrays(
        canonical,
        point_count=args.point_count,
        query_count=args.query_count,
        seed=args.seed,
    )
    output = args.output.expanduser().resolve()
    save_point_set_artifact(output, point_set)
    parameters = {
        "sampling_method": "surface_area_deterministic_v1",
        "point_count": args.point_count,
        "query_count": args.query_count,
        "seed": args.seed,
        "numpy_version": np.__version__,
        "source_to_meters": source_to_meters,
        "T_cad_from_source_meters": transform.tolist(),
    }
    parameters_sha256 = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    print(
        json.dumps(
            {
                "mesh_sha256": sha256_file(mesh_path),
                "point_set": {
                    "path": str(args.output),
                    "sha256": sha256_file(output),
                    "point_count": args.point_count,
                    "sampling_method": parameters["sampling_method"],
                    "sampling_parameters_sha256": parameters_sha256,
                    "surface_centroid_m": point_set.surface_centroid_m.tolist(),
                },
                "sampling_parameters": parameters,
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_stl_triangles(path: Path) -> np.ndarray:
    """Read binary or ASCII STL without adding a heavyweight mesh dependency."""

    payload = path.read_bytes()
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if len(payload) == 84 + 50 * triangle_count:
            records = np.frombuffer(
                payload,
                dtype=np.dtype(
                    [
                        ("normal", "<f4", (3,)),
                        ("vertices", "<f4", (3, 3)),
                        ("attribute", "<u2"),
                    ]
                ),
                offset=84,
                count=triangle_count,
            )
            return np.asarray(records["vertices"], dtype=np.float64)
    vertices = []
    for raw_line in payload.decode("utf-8", errors="strict").splitlines():
        fields = raw_line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append(tuple(float(value) for value in fields[1:]))
    if len(vertices) == 0 or len(vertices) % 3:
        raise ValueError(f"Could not parse valid STL triangles from {path}")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def load_usd_triangles(path: Path) -> tuple[np.ndarray, float]:
    """Read composed USD mesh surfaces when Pixar USD is available."""

    stage, UsdGeom = _open_usd_stage(path)
    triangles = _load_usd_stage_triangles(stage, UsdGeom)
    if not triangles:
        raise ValueError(f"USD stage {path} contains no render-purpose mesh triangles")
    return np.asarray(triangles, dtype=np.float64), float(UsdGeom.GetStageMetersPerUnit(stage))


def load_usd_triangles_with_bounds(
    path: Path,
) -> tuple[np.ndarray, float, str, np.ndarray, np.ndarray]:
    """Read USD triangles plus composed local bounds in source-axis metres."""

    from pxr import Usd

    stage, UsdGeom = _open_usd_stage(path)
    triangles = _load_usd_stage_triangles(stage, UsdGeom)
    if not triangles:
        raise ValueError(f"USD stage {path} contains no render-purpose mesh triangles")
    source_to_meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    ).ComputeLocalBound(stage.GetPseudoRoot())
    aligned = bounds.ComputeAlignedRange()
    center_m = np.asarray(aligned.GetMidpoint(), dtype=np.float64) * source_to_meters
    dimensions_m = np.asarray(aligned.GetSize(), dtype=np.float64) * source_to_meters
    return (
        np.asarray(triangles, dtype=np.float64),
        source_to_meters,
        str(UsdGeom.GetStageUpAxis(stage)),
        center_m,
        dimensions_m,
    )


def _open_usd_stage(path: Path):
    try:
        from pxr import Usd, UsdGeom
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "USD preprocessing requires the 'pxr' package from an Isaac Sim/USD environment"
        ) from exc
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Could not open USD stage {path}")
    return stage, UsdGeom


def _load_usd_stage_triangles(stage, UsdGeom) -> list[np.ndarray]:
    from pxr import Gf

    cache = UsdGeom.XformCache()
    triangles = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        purpose = imageable.ComputePurpose()
        if purpose in {UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide}:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or counts is None or indices is None:
            continue
        transform = cache.GetLocalToWorldTransform(prim)
        transformed = np.asarray(
            [
                tuple(transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))))
                for point in points
            ],
            dtype=np.float64,
        )
        offset = 0
        for count in counts:
            face = np.asarray(indices[offset : offset + count], dtype=np.int64)
            offset += count
            if len(face) < 3:
                continue
            for index in range(1, len(face) - 1):
                triangles.append(transformed[[face[0], face[index], face[index + 1]]])
    return triangles


if __name__ == "__main__":
    main()
