"""Deterministic CAD surface point-set preprocessing and artifact loading."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LoadedPointSet:
    """Validated arrays loaded from one immutable point-set artifact."""

    points_m: np.ndarray
    query_indices: np.ndarray
    surface_centroid_m: np.ndarray

    @property
    def query_points_m(self) -> np.ndarray:
        return self.points_m[self.query_indices]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def surface_centroid_from_triangles(triangles: np.ndarray) -> np.ndarray:
    """Return the exact triangle-area-weighted surface centroid."""

    triangles = _validated_triangles(triangles)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > np.finfo(np.float64).eps
    if not np.any(valid):
        raise ValueError("Mesh has no nondegenerate surface triangles")
    centroids = triangles.mean(axis=1)
    return np.average(centroids[valid], axis=0, weights=areas[valid])


def sample_surface_points(
    triangles: np.ndarray,
    point_count: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Sample a triangle mesh uniformly by surface area."""

    triangles = _validated_triangles(triangles)
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > np.finfo(np.float64).eps
    if not np.any(valid):
        raise ValueError("Mesh has no nondegenerate surface triangles")
    triangles = triangles[valid]
    probabilities = areas[valid] / areas[valid].sum()
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), size=point_count, p=probabilities)]
    uv = rng.random((point_count, 2))
    sqrt_u = np.sqrt(uv[:, :1])
    barycentric = np.concatenate((1.0 - sqrt_u, sqrt_u * (1.0 - uv[:, 1:]), sqrt_u * uv[:, 1:]), axis=1)
    return np.einsum("ni,nij->nj", barycentric, selected)


def build_point_set_arrays(
    triangles: np.ndarray,
    *,
    point_count: int = 4096,
    query_count: int = 512,
    seed: int = 0,
) -> LoadedPointSet:
    """Build deterministic dense/query point arrays and an exact centroid."""

    if query_count <= 0 or query_count > point_count:
        raise ValueError("query_count must be in [1, point_count]")
    points = sample_surface_points(triangles, point_count, seed=seed).astype(np.float32)
    # Keeping the query as a subset of the target makes identical rotations
    # attain exactly zero loss despite finite sampling.
    query_indices = np.arange(query_count, dtype=np.int64)
    centroid = surface_centroid_from_triangles(triangles).astype(np.float64)
    return LoadedPointSet(points, query_indices, centroid)


def save_point_set_artifact(path: Path, point_set: LoadedPointSet) -> None:
    """Write the stable NPZ artifact consumed by the pose loader."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            points_m=np.asarray(point_set.points_m, dtype=np.float32),
            query_indices=np.asarray(point_set.query_indices, dtype=np.int64),
            surface_centroid_m=np.asarray(point_set.surface_centroid_m, dtype=np.float64),
        )


def load_point_set_artifact(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_point_count: int | None = None,
    expected_centroid_m: np.ndarray | None = None,
    centroid_atol_m: float = 1e-8,
) -> LoadedPointSet:
    """Load and validate a point-set artifact, with immutable-array caching."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    file_stat = resolved.stat()
    expected_centroid_key = (
        None
        if expected_centroid_m is None
        else tuple(float(value) for value in np.asarray(expected_centroid_m).reshape(3))
    )
    loaded = _load_point_set_artifact_cached(
        str(resolved),
        file_stat.st_mtime_ns,
        file_stat.st_size,
        expected_sha256 or "",
        -1 if expected_point_count is None else int(expected_point_count),
        expected_centroid_key,
        float(centroid_atol_m),
    )
    # Callers must not mutate globally cached catalog geometry.
    return LoadedPointSet(loaded.points_m, loaded.query_indices, loaded.surface_centroid_m)


@lru_cache(maxsize=256)
def _load_point_set_artifact_cached(
    path_string: str,
    _file_mtime_ns: int,
    _file_size: int,
    expected_sha256: str,
    expected_point_count: int,
    expected_centroid_key: tuple[float, float, float] | None,
    centroid_atol_m: float,
) -> LoadedPointSet:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError(f"Point-set checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        required = {"points_m", "query_indices", "surface_centroid_m"}
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"Point-set artifact {path} is missing arrays: {sorted(missing)}")
        points = np.asarray(artifact["points_m"], dtype=np.float32)
        query_indices = np.asarray(artifact["query_indices"], dtype=np.int64)
        centroid = np.asarray(artifact["surface_centroid_m"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"points_m in {path} must be a nonempty finite Nx3 array")
    if expected_point_count >= 0 and len(points) != expected_point_count:
        raise ValueError(f"Point count in {path} is {len(points)}, expected {expected_point_count}")
    if query_indices.ndim != 1 or len(query_indices) == 0:
        raise ValueError(f"query_indices in {path} must be a nonempty vector")
    if np.any(query_indices < 0) or np.any(query_indices >= len(points)) or len(np.unique(query_indices)) != len(
        query_indices
    ):
        raise ValueError(f"query_indices in {path} must be unique valid point indices")
    if centroid.shape != (3,) or not np.isfinite(centroid).all():
        raise ValueError(f"surface_centroid_m in {path} must be a finite vec3")
    if expected_centroid_key is not None and not np.allclose(
        centroid, np.asarray(expected_centroid_key), atol=centroid_atol_m, rtol=0
    ):
        raise ValueError(f"Point-set centroid in {path} differs from the catalog")
    points.setflags(write=False)
    query_indices.setflags(write=False)
    centroid.setflags(write=False)
    return LoadedPointSet(points, query_indices, centroid)


def _validated_triangles(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"Expected triangles with shape Tx3x3, got {triangles.shape}")
    if len(triangles) == 0 or not np.isfinite(triangles).all():
        raise ValueError("Triangles must be nonempty and finite")
    return triangles
