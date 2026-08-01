"""Tests for the point-set pose-v2 metadata upgrade."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from muggled_sam.v3_sam.cad_pose.point_sets import (
    LoadedPointSet,
    save_point_set_artifact,
    sha256_file,
)
from upgrade_perseve_pose_dataset_v2 import (
    Y_UP_TO_Z_UP,
    _load_schemas,
    _validate_json,
    apply_upgrade,
    upgrade_annotation,
    upgrade_catalog,
    upgrade_metadata,
)


class PersevePoseV2UpgradeTests(unittest.TestCase):
    def test_apply_refuses_different_existing_point_files_before_backup(self):
        for conflicting_name in ("cad-a.npz", "point_sets.json"):
            with self.subTest(conflicting_name=conflicting_name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                point_source_dir = root / "prepared_points"
                point_source_dir.mkdir()
                point_source = point_source_dir / "cad-a.npz"
                point_source.write_bytes(b"prepared point artifact")
                manifest_source = point_source_dir / "point_sets.json"
                manifest_source.write_bytes(b'{"schema_version":"1.0.0"}')
                install_root = root / "cad_points"
                install_root.mkdir()
                (install_root / conflicting_name).write_bytes(b"different existing content")
                old_metadata = b'{"schema_version":"1.0.0"}'
                self._write_apply_source_metadata(root, old_metadata)

                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    self._apply_minimal_upgrade(root, point_source, manifest_source)

                self.assertEqual((root / "dataset_meta.json").read_bytes(), old_metadata)
                self.assertEqual(
                    (install_root / conflicting_name).read_bytes(),
                    b"different existing content",
                )
                self.assertEqual(list(root.glob("pose_v1_backup_*")), [])

    def test_apply_allows_identical_existing_point_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            point_source_dir = root / "prepared_points"
            point_source_dir.mkdir()
            point_source = point_source_dir / "cad-a.npz"
            point_source.write_bytes(b"prepared point artifact")
            manifest_source = point_source_dir / "point_sets.json"
            manifest_source.write_bytes(b'{"schema_version":"1.0.0"}')
            install_root = root / "cad_points"
            install_root.mkdir()
            installed_point = install_root / point_source.name
            installed_manifest = install_root / manifest_source.name
            installed_point.write_bytes(point_source.read_bytes())
            installed_manifest.write_bytes(manifest_source.read_bytes())
            self._write_apply_source_metadata(root, b'{"schema_version":"1.0.0"}')

            backup_dir = self._apply_minimal_upgrade(root, point_source, manifest_source)

            self.assertTrue(backup_dir.is_dir())
            self.assertEqual(installed_point.read_bytes(), point_source.read_bytes())
            self.assertEqual(installed_manifest.read_bytes(), manifest_source.read_bytes())
            self.assertEqual(json.loads((root / "dataset_meta.json").read_text()), {"upgraded": True})

    def test_upgrade_joins_point_manifest_and_marks_valid_visible_pose_eligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            point_dir = root / "cad_points"
            point_dir.mkdir()
            artifact = point_dir / "cad-a.npz"
            point_set = LoadedPointSet(
                points_m=np.asarray(
                    [[-0.5, 0, 0], [0.5, 0, 0], [0, -0.5, 0], [0, 0.5, 0]],
                    dtype=np.float32,
                ),
                query_indices=np.asarray([0, 1], dtype=np.int64),
                surface_centroid_m=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            )
            save_point_set_artifact(artifact, point_set)
            mesh_sha = "a" * 64
            identity = np.eye(4).tolist()
            point_transform = np.eye(4)
            point_transform[:3, :3] = Y_UP_TO_Z_UP
            source_object = {
                "mesh_path": "cad_usd/cad-a/cad-a.usd",
                "mesh_sha256": mesh_sha,
                "source_length_unit": "m",
                "source_to_meters": 1.0,
                "T_cad_from_source_meters": identity,
                "bbox_min_m": [-0.5, -0.5, -0.5],
                "bbox_max_m": [0.5, 0.5, 0.5],
                "base_dimensions_m": [1.0, 1.0, 1.0],
                "canonical_origin": "local_aabb_center",
                "symmetry": {
                    "type": "none",
                    "transforms": [np.eye(3).tolist()],
                    "label_source": "automatic_geometry",
                    "status": "needs_review",
                    "pipeline_version": "external_label_pending",
                    "parameters_sha256": "b" * 64,
                },
                "extensions": {},
            }
            catalog = {"schema_version": "1.0.0", "objects": {"cad-a": source_object}}
            point_manifest = {
                "schema_version": "1.0.0",
                "objects": {
                    "cad-a": {
                        **{
                            key: source_object[key]
                            for key in (
                                "mesh_sha256",
                                "source_to_meters",
                                "bbox_min_m",
                                "bbox_max_m",
                                "base_dimensions_m",
                                "canonical_origin",
                            )
                        },
                        "T_cad_from_source_meters": point_transform.tolist(),
                        "point_set": {
                            "path": artifact.name,
                            "sha256": sha256_file(artifact),
                            "point_count": 4,
                            "sampling_method": "surface_area_deterministic_v1",
                            "sampling_parameters_sha256": hashlib.sha256(b"parameters").hexdigest(),
                            "surface_centroid_m": [0.0, 0.0, 0.0],
                        },
                        "sampling_parameters": {
                            "seed": 0,
                            "query_count": 2,
                            "source_to_meters": 1.0,
                            "source_up_axis": "Y",
                            "target_up_axis": "Z",
                            "T_cad_from_source_meters": point_transform.tolist(),
                        },
                    }
                },
            }
            manifest_path = point_dir / "point_sets.json"
            manifest_path.write_text(json.dumps(point_manifest))

            upgraded_catalog, sources, migrations = upgrade_catalog(
                catalog,
                point_manifest,
                manifest_path,
                Path("cad_points"),
                atol=1e-6,
                rtol=1e-5,
            )
            self.assertNotIn("symmetry", upgraded_catalog["objects"]["cad-a"])
            self.assertEqual(sources, {"cad-a": artifact})
            self.assertEqual(migrations, {"legacy_y_up_to_z_up": 1})
            np.testing.assert_allclose(
                upgraded_catalog["objects"]["cad-a"]["T_cad_from_source_meters"],
                point_transform,
            )
            self.assertEqual(
                upgraded_catalog["objects"]["cad-a"]["point_set"]["path"],
                "cad_points/cad-a.npz",
            )

            transform = np.eye(4)
            transform[2, 3] = 2.0
            annotation = {
                "schema_version": "1.0.0",
                "frame_id": "0000",
                "scene_id": "scene-a",
                "image": {
                    "rgb_path": "rgb_0000.png",
                    "instance_path": "instance_segmentation_0000.png",
                    "mapping_path": "instance_segmentation_mapping_0000.json",
                    "size_wh": [16, 16],
                },
                "camera": {
                    "model": "pinhole",
                    "distortion_model": "none",
                    "K": [[10, 0, 8], [0, 10, 8], [0, 0, 1]],
                    "T_world_from_camera_cv": identity,
                },
                "instances": [
                    {
                        "instance_id": "object_0",
                        "cad_id": "cad-a",
                        "prim_path": "/World/object_0",
                        "annotation_state": "visible",
                        "pose_training_eligible": False,
                        "mask": {
                            "mapping_key": "(1, 2, 3, 255)",
                            "value": [1, 2, 3, 255],
                            "value_order": "RGBA",
                            "match_alpha": True,
                        },
                        "bbox_xyxy_px": [1, 1, 4, 4],
                        "T_cam_from_cad": transform.tolist(),
                        "render_scale_xyz": [0.5, 0.5, 0.5],
                        "dimensions_m": [0.5, 0.5, 0.5],
                        "visibility": {
                            "visible_pixel_count": 16,
                            "visible_fraction": None,
                            "truncated": False,
                        },
                        "extensions": {},
                    }
                ],
            }
            upgraded_annotation, reasons = upgrade_annotation(
                annotation,
                upgraded_catalog["objects"],
                atol=1e-6,
                rtol=1e-5,
            )
            self.assertTrue(upgraded_annotation["instances"][0]["pose_training_eligible"])
            self.assertEqual(
                upgraded_annotation["instances"][0]["T_cam_from_cad"],
                annotation["instances"][0]["T_cam_from_cad"],
            )
            self.assertEqual(reasons, {})

            schemas = _load_schemas(Path(__file__).resolve().parents[1] / "schemas" / "perseve-pose-v2")
            metadata = {
                "schema": "perseve.pose",
                "schema_version": "1.0.0",
                "length_unit": "m",
                "transform_convention": "p_target = T_target_from_source @ p_source",
                "vector_convention": "column",
                "matrix_storage": "row_major_json",
                "camera_frame": "opencv_x_right_y_down_z_forward",
                "image_origin": "top_left",
                "pixel_convention": "top_left_pixel_center_is_0_0",
                "segmentation_encoding": "rgba_png_v1",
                "bbox_convention": "xyxy_inclusive_integer",
                "asset_catalog": "objects.json",
                "schemas": {},
                "validation_tolerances": {
                    "rotation_atol": 1e-6,
                    "dimension_atol_m": 1e-6,
                    "dimension_rtol": 1e-5,
                },
                "generator": {
                    "name": "perseve",
                    "git_commit": "abc",
                    "isaac_sim_version": "5.1.0",
                    "seed": 1,
                    "config_sha256": "c" * 64,
                },
            }
            upgraded_metadata = upgrade_metadata(metadata, schemas)
            _validate_json(upgraded_metadata, schemas["dataset_meta"], "dataset_meta.json")
            _validate_json(upgraded_catalog, schemas["objects"], "objects.json")
            _validate_json(upgraded_annotation, schemas["pose_annotations"], "pose_annotations_0000.json")

    @staticmethod
    def _write_apply_source_metadata(root: Path, metadata: bytes) -> None:
        (root / "dataset_meta.json").write_bytes(metadata)
        (root / "objects.json").write_text('{"schema_version":"1.0.0"}')
        (root / "pose_annotations_0000.json").write_text('{"schema_version":"1.0.0"}')

    @staticmethod
    def _apply_minimal_upgrade(root: Path, point_source: Path, manifest_source: Path) -> Path:
        return apply_upgrade(
            dataset_root=root,
            metadata={"upgraded": True},
            catalog={"upgraded": True},
            annotations={"pose_annotations_0000.json": {"upgraded": True}},
            schemas={
                "dataset_meta": {"schema": "dataset_meta"},
                "objects": {"schema": "objects"},
                "pose_annotations": {"schema": "pose_annotations"},
            },
            point_sources={"cad-a": point_source},
            point_manifest_path=manifest_source,
            install_point_dir=Path("cad_points"),
        )


if __name__ == "__main__":
    unittest.main()
