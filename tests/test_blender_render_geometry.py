"""Tests for pose-safe Blender render-frame construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from itertools import product
from pathlib import Path

import numpy as np

from blender_render_geometry import (
    RenderGeometryError,
    camera_cv_from_world,
    compute_render_geometry_transform,
    load_object_catalog,
    transform_points,
)


def cuboid(center=(0.0, 0.0, 0.0), dimensions=(2.0, 4.0, 6.0)) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    half = 0.5 * np.asarray(dimensions, dtype=np.float64)
    return np.asarray(
        [center + half * np.asarray(signs) for signs in product((-1.0, 1.0), repeat=3)],
        dtype=np.float64,
    )


class BlenderRenderGeometryTests(unittest.TestCase):
    def test_canonical_mode_centers_and_uniformly_scales_without_rotating(self):
        points = cuboid(center=(7.0, -4.0, 2.0), dimensions=(2.0, 4.0, 6.0))

        geometry = compute_render_geometry_transform(points, orientation_mode="canonical")
        rendered = transform_points(points, geometry.T_render_from_source)

        np.testing.assert_allclose(
            0.5 * (rendered.min(0) + rendered.max(0)), np.zeros(3), atol=1e-15
        )
        np.testing.assert_allclose(np.ptp(rendered, axis=0), [1 / 3, 2 / 3, 1.0])
        np.testing.assert_allclose(geometry.T_presentation_from_cad, np.eye(4))
        self.assertFalse(geometry.catalog_driven)

    def test_catalog_transform_defines_the_pose_safe_canonical_frame(self):
        points = cuboid(center=(10.0, 20.0, 30.0), dimensions=(2.0, 4.0, 6.0))
        source_to_meters = 0.001
        rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        center_m = source_to_meters * np.asarray((10.0, 20.0, 30.0))
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -(rotation @ center_m)
        entry = {
            "source_to_meters": source_to_meters,
            "T_cad_from_source_meters": transform.tolist(),
        }

        geometry = compute_render_geometry_transform(
            points,
            orientation_mode="canonical",
            catalog_entry=entry,
        )
        canonical = transform_points(
            points,
            transform @ np.diag([source_to_meters] * 3 + [1.0]),
        )
        rendered = transform_points(points, geometry.T_render_from_source)

        np.testing.assert_allclose(geometry.canonical_dimensions_m, [0.002, 0.006, 0.004])
        np.testing.assert_allclose(rendered, canonical * geometry.display_scale)
        np.testing.assert_allclose(geometry.T_presentation_from_cad, np.eye(4))
        self.assertTrue(geometry.catalog_driven)

    def test_largest_face_up_is_optional_and_applied_after_canonicalization(self):
        points = cuboid(dimensions=(1.0, 2.0, 3.0))
        canonical = compute_render_geometry_transform(points, orientation_mode="canonical")
        presentation = compute_render_geometry_transform(
            points, orientation_mode="largest-face-up"
        )

        np.testing.assert_allclose(canonical.render_dimensions, [1 / 3, 2 / 3, 1.0])
        np.testing.assert_allclose(presentation.render_dimensions, [1.0, 2 / 3, 1 / 3])
        self.assertFalse(np.allclose(presentation.T_presentation_from_cad, np.eye(4)))
        self.assertAlmostEqual(
            np.linalg.det(presentation.T_presentation_from_cad[:3, :3]), 1.0
        )

    def test_catalog_transform_must_center_the_declared_cad_frame(self):
        entry = {
            "source_to_meters": 1.0,
            "T_cad_from_source_meters": np.eye(4).tolist(),
        }
        with self.assertRaisesRegex(RenderGeometryError, "AABB center"):
            compute_render_geometry_transform(
                cuboid(center=(3.0, 0.0, 0.0)), catalog_entry=entry
            )

    def test_catalog_dimensions_must_match_the_rendered_source(self):
        entry = {
            "source_to_meters": 1.0,
            "T_cad_from_source_meters": np.eye(4).tolist(),
            "base_dimensions_m": [2.0, 4.0, 7.0],
        }
        with self.assertRaisesRegex(RenderGeometryError, "base_dimensions_m"):
            compute_render_geometry_transform(cuboid(), catalog_entry=entry)

    def test_catalog_loader_and_camera_axis_conversion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "objects.json"
            path.write_text(
                json.dumps({"schema_version": "2.0.0", "objects": {"cad-a": {}}}),
                encoding="utf-8",
            )
            version, objects = load_object_catalog(path)

        self.assertEqual(version, "2.0.0")
        self.assertEqual(set(objects), {"cad-a"})
        camera = camera_cv_from_world(np.zeros(3), np.eye(3))
        np.testing.assert_allclose(camera, np.diag([1.0, -1.0, -1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
