"""Deterministic CAD surface point-set preprocessing tests."""

import argparse
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from muggled_sam.v3_sam.cad_pose.point_sets import (
    build_point_set_arrays,
    load_point_set_artifact,
    save_point_set_artifact,
    sha256_file,
    surface_centroid_from_triangles,
)
from preprocess_abc_point_sets import (
    _matches_request,
    apply_metric_transform,
    source_to_z_up_rotation,
)


class CADPosePointSetTests(unittest.TestCase):
    def test_abc_y_up_is_rotated_and_centered_in_z_up_cad_frame(self):
        rotation = source_to_z_up_rotation("Y")
        np.testing.assert_array_equal(
            rotation @ np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.0, -3.0, 2.0]),
        )
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -(rotation @ np.asarray([1.0, 2.0, 3.0]))
        triangles = np.asarray([[[1, 2, 3], [2, 2, 3], [1, 3, 3]]], dtype=np.float64)
        canonical = apply_metric_transform(triangles, 1.0, transform)
        np.testing.assert_array_equal(canonical[0, 0], np.zeros(3))

    def test_area_weighted_centroid_is_not_vertex_density_weighted(self):
        triangles = np.asarray(
            [
                [[0, 0, 0], [2, 0, 0], [0, 2, 0]],
                [[10, 0, 0], [11, 0, 0], [10, 1, 0]],
            ],
            dtype=np.float64,
        )
        centroid = surface_centroid_from_triangles(triangles)
        expected = (
            2.0 * np.asarray([2 / 3, 2 / 3, 0])
            + 0.5 * np.asarray([31 / 3, 1 / 3, 0])
        ) / 2.5
        np.testing.assert_allclose(centroid, expected)

    def test_artifact_is_deterministic_and_checksum_validated(self):
        triangles = np.asarray(
            [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]],
            dtype=np.float64,
        )
        first = build_point_set_arrays(triangles, point_count=32, query_count=8, seed=7)
        second = build_point_set_arrays(triangles, point_count=32, query_count=8, seed=7)
        np.testing.assert_array_equal(first.points_m, second.points_m)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "points.npz"
            duplicate_path = Path(temp_dir) / "points-copy.npz"
            save_point_set_artifact(path, first)
            save_point_set_artifact(duplicate_path, second)
            checksum = sha256_file(path)
            self.assertEqual(checksum, sha256_file(duplicate_path))
            loaded = load_point_set_artifact(
                path,
                expected_sha256=checksum,
                expected_point_count=32,
                expected_centroid_m=first.surface_centroid_m,
            )
            np.testing.assert_array_equal(loaded.query_points_m, first.points_m[:8])
            self.assertFalse(loaded.points_m.flags.writeable)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_point_set_artifact(path, expected_sha256="0" * 64)

    def test_replacing_artifact_invalidates_cached_validation(self):
        triangles = np.asarray(
            [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "points.npz"
            first = build_point_set_arrays(triangles, point_count=32, query_count=8, seed=1)
            save_point_set_artifact(path, first)
            checksum = sha256_file(path)
            load_point_set_artifact(path, expected_sha256=checksum, expected_point_count=32)

            previous_mtime_ns = path.stat().st_mtime_ns
            replacement = build_point_set_arrays(triangles, point_count=16, query_count=4, seed=2)
            save_point_set_artifact(path, replacement)
            os.utime(path, ns=(previous_mtime_ns + 1, previous_mtime_ns + 1))

            with self.assertRaisesRegex(ValueError, "checksum"):
                load_point_set_artifact(path, expected_sha256=checksum, expected_point_count=32)

    def test_resume_requires_same_source_path_and_checksum(self):
        args = argparse.Namespace(point_count=32, query_count=8, seed=7)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "points"
            output_dir.mkdir()
            source = root / "source" / "cad-a.usd"
            source.parent.mkdir()
            source.write_bytes(b"first USD payload")
            mesh_relative_path = Path(os.path.relpath(source, output_dir)).as_posix()
            entry = {
                "mesh_path": mesh_relative_path,
                "mesh_sha256": sha256_file(source),
                "point_set": {"point_count": 32},
                "sampling_parameters": {
                    "point_count": 32,
                    "query_count": 8,
                    "seed": 7,
                    "target_up_axis": "Z",
                },
            }
            self.assertTrue(
                _matches_request(
                    entry,
                    args,
                    mesh_relative_path=mesh_relative_path,
                    mesh_sha256=sha256_file(source),
                )
            )

            relocated = root / "other" / source.name
            relocated.parent.mkdir()
            relocated.write_bytes(source.read_bytes())
            self.assertFalse(
                _matches_request(
                    entry,
                    args,
                    mesh_relative_path=Path(os.path.relpath(relocated, output_dir)).as_posix(),
                    mesh_sha256=sha256_file(relocated),
                )
            )

            source.write_bytes(b"replacement USD payload")
            self.assertFalse(
                _matches_request(
                    entry,
                    args,
                    mesh_relative_path=mesh_relative_path,
                    mesh_sha256=sha256_file(source),
                )
            )


if __name__ == "__main__":
    unittest.main()
