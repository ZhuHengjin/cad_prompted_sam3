"""Pure geometry helpers for pose-aware Blender exemplar rendering."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ORIENTATION_MODES = ("canonical", "largest-face-up")


class RenderGeometryError(ValueError):
    """Raised when a mesh cannot be mapped into a safe render frame."""


@dataclass(frozen=True)
class RenderGeometryTransform:
    """Transforms and dimensions governing one normalized render mesh."""

    source_to_meters: float
    T_cad_from_source_meters: np.ndarray
    T_presentation_from_cad: np.ndarray
    display_scale: float
    T_render_from_cad: np.ndarray
    T_render_from_source: np.ndarray
    canonical_dimensions_m: np.ndarray
    render_dimensions: np.ndarray
    orientation_mode: str
    catalog_driven: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "source_to_meters": self.source_to_meters,
            "T_cad_from_source_meters": self.T_cad_from_source_meters.tolist(),
            "T_presentation_from_cad": self.T_presentation_from_cad.tolist(),
            "display_scale": self.display_scale,
            "T_render_from_cad": self.T_render_from_cad.tolist(),
            "T_render_from_source": self.T_render_from_source.tolist(),
            "canonical_dimensions_m": self.canonical_dimensions_m.tolist(),
            "render_dimensions": self.render_dimensions.tolist(),
            "orientation_mode": self.orientation_mode,
            "catalog_driven": self.catalog_driven,
        }


def load_object_catalog(path: str | Path) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    """Load the object mapping from a Perseve pose catalog."""

    catalog_path = Path(path).expanduser().resolve()
    with catalog_path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    objects = value.get("objects")
    if not isinstance(objects, dict):
        raise RenderGeometryError(f"Object catalog has no object mapping: {catalog_path}")
    return str(value.get("schema_version", "unknown")), objects


def compute_render_geometry_transform(
    source_points: np.ndarray,
    *,
    orientation_mode: str = "canonical",
    catalog_entry: Mapping[str, Any] | None = None,
) -> RenderGeometryTransform:
    """Build a centered, uniformly scaled source-to-render transform.

    Catalog-driven rendering preserves the catalog CAD frame exactly. Without a
    catalog, the source axes define the CAD axes and only AABB centering is
    introduced. The optional presentation rotation is always applied after the
    CAD frame has been established.
    """

    if orientation_mode not in ORIENTATION_MODES:
        raise RenderGeometryError(
            f"Unsupported orientation mode {orientation_mode!r}; expected one of {ORIENTATION_MODES}"
        )
    points = np.asarray(source_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise RenderGeometryError("source_points must be a nonempty Nx3 array")
    if not np.isfinite(points).all():
        raise RenderGeometryError("source_points must contain only finite values")

    catalog_driven = catalog_entry is not None
    if catalog_entry is None:
        source_to_meters = 1.0
        source_center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        T_cad_from_source_meters = np.eye(4, dtype=np.float64)
        T_cad_from_source_meters[:3, 3] = -source_center
    else:
        try:
            source_to_meters = float(catalog_entry["source_to_meters"])
            T_cad_from_source_meters = np.asarray(
                catalog_entry["T_cad_from_source_meters"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RenderGeometryError(
                "Catalog entry must define source_to_meters and T_cad_from_source_meters"
            ) from error
        _validate_rigid_transform(T_cad_from_source_meters)
    if not math.isfinite(source_to_meters) or source_to_meters <= 0:
        raise RenderGeometryError("source_to_meters must be finite and positive")

    source_units_to_meters = np.diag(
        [source_to_meters, source_to_meters, source_to_meters, 1.0]
    )
    T_cad_from_source = T_cad_from_source_meters @ source_units_to_meters
    canonical_points = transform_points(points, T_cad_from_source)
    canonical_min = canonical_points.min(axis=0)
    canonical_max = canonical_points.max(axis=0)
    canonical_center = 0.5 * (canonical_min + canonical_max)
    canonical_dimensions = canonical_max - canonical_min
    max_extent = float(canonical_dimensions.max())
    if not math.isfinite(max_extent) or max_extent <= 0:
        raise RenderGeometryError("Mesh must have a positive finite extent")
    if catalog_driven:
        center_tolerance = 1e-8 + 1e-6 * max_extent
        if not np.allclose(canonical_center, np.zeros(3), atol=center_tolerance, rtol=0.0):
            raise RenderGeometryError(
                "Catalog transform does not place the mesh AABB center at the CAD origin"
            )
        _validate_catalog_geometry(
            catalog_entry,
            canonical_min,
            canonical_max,
            canonical_dimensions,
        )

    T_presentation_from_cad = np.eye(4, dtype=np.float64)
    if orientation_mode == "largest-face-up":
        T_presentation_from_cad[:3, :3] = largest_face_up_rotation(canonical_dimensions)

    display_scale = 1.0 / max_extent
    display_scale_matrix = np.diag([display_scale, display_scale, display_scale, 1.0])
    T_render_from_cad = display_scale_matrix @ T_presentation_from_cad
    T_render_from_source = T_render_from_cad @ T_cad_from_source
    render_points = transform_points(points, T_render_from_source)
    render_dimensions = render_points.max(axis=0) - render_points.min(axis=0)
    return RenderGeometryTransform(
        source_to_meters=source_to_meters,
        T_cad_from_source_meters=T_cad_from_source_meters,
        T_presentation_from_cad=T_presentation_from_cad,
        display_scale=display_scale,
        T_render_from_cad=T_render_from_cad,
        T_render_from_source=T_render_from_source,
        canonical_dimensions_m=canonical_dimensions,
        render_dimensions=render_dimensions,
        orientation_mode=orientation_mode,
        catalog_driven=catalog_driven,
    )


def largest_face_up_rotation(dimensions: np.ndarray) -> np.ndarray:
    """Return the legacy rotation that aligns the thinnest AABB axis with Z."""

    extents = np.asarray(dimensions, dtype=np.float64)
    if extents.shape != (3,) or not np.isfinite(extents).all():
        raise RenderGeometryError("dimensions must be a finite vec3")
    min_axis = int(np.argmin(extents))
    if min_axis == 0:
        return np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    if min_axis == 1:
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
    return np.eye(3, dtype=np.float64)


def camera_cv_from_world(
    camera_position_world: np.ndarray,
    camera_rotation_world_from_blender: np.ndarray,
) -> np.ndarray:
    """Convert a Blender camera world pose to OpenCV camera-from-world."""

    position = np.asarray(camera_position_world, dtype=np.float64)
    rotation = np.asarray(camera_rotation_world_from_blender, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise RenderGeometryError("camera position must be a finite vec3")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise RenderGeometryError("camera rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise RenderGeometryError("camera rotation must be orthonormal")
    world_from_camera_blender = np.eye(4, dtype=np.float64)
    world_from_camera_blender[:3, :3] = rotation
    world_from_camera_blender[:3, 3] = position
    cv_from_blender = np.diag([1.0, -1.0, -1.0, 1.0])
    return cv_from_blender @ np.linalg.inv(world_from_camera_blender)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a column-vector homogeneous transform to row-stored points."""

    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    homogeneous = np.concatenate(
        (values, np.ones((len(values), 1), dtype=np.float64)), axis=1
    )
    return (homogeneous @ matrix.T)[:, :3]


def _validate_rigid_transform(transform: np.ndarray) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RenderGeometryError("T_cad_from_source_meters must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise RenderGeometryError("T_cad_from_source_meters has an invalid bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0):
        raise RenderGeometryError("T_cad_from_source_meters contains scale or shear")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
        raise RenderGeometryError("T_cad_from_source_meters rotation must be in SO(3)")


def _validate_catalog_geometry(
    catalog_entry: Mapping[str, Any],
    canonical_min: np.ndarray,
    canonical_max: np.ndarray,
    canonical_dimensions: np.ndarray,
) -> None:
    """Catch a source mesh that does not match its selected catalog entry."""

    if catalog_entry.get("canonical_origin", "local_aabb_center") != "local_aabb_center":
        raise RenderGeometryError("Catalog must use the local AABB center as the CAD origin")
    expected_fields = (
        ("bbox_min_m", canonical_min),
        ("bbox_max_m", canonical_max),
        ("base_dimensions_m", canonical_dimensions),
    )
    for field, actual in expected_fields:
        if field not in catalog_entry:
            continue
        expected = np.asarray(catalog_entry[field], dtype=np.float64)
        if expected.shape != (3,) or not np.isfinite(expected).all():
            raise RenderGeometryError(f"Catalog {field} must be a finite vec3")
        if not np.allclose(actual, expected, atol=1e-6, rtol=1e-5):
            raise RenderGeometryError(f"Rendered source geometry does not match catalog {field}")
