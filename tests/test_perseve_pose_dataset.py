"""End-to-end logical-RGBA and pose-sidecar validation test."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import jsonschema  # noqa: F401
    import numpy as np

    from muggled_sam.v3_sam.cad_pose.dataset import load_perseve_pose_sample

    DEPS_AVAILABLE = True
except ModuleNotFoundError:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "pose dataset dependencies are not installed")
class PersevePoseDatasetTests(unittest.TestCase):
    def test_valid_frame_round_trips_exact_rgba_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            schema_source = Path(__file__).resolve().parents[1] / "schemas" / "perseve-pose-v1"
            shutil.copytree(schema_source, root / "schemas")
            meta = {
                "schema": "perseve.pose", "schema_version": "1.0.0", "length_unit": "m",
                "transform_convention": "p_target = T_target_from_source @ p_source", "vector_convention": "column",
                "matrix_storage": "row_major_json", "camera_frame": "opencv_x_right_y_down_z_forward",
                "image_origin": "top_left", "pixel_convention": "top_left_pixel_center_is_0_0",
                "segmentation_encoding": "rgba_png_v1", "bbox_convention": "xyxy_inclusive_integer",
                "asset_catalog": "objects.json",
                "schemas": {"dataset_meta": "schemas/dataset-meta.schema.json", "objects": "schemas/objects.schema.json", "pose_annotations": "schemas/pose-annotations.schema.json"},
                "validation_tolerances": {"rotation_atol": 1e-6, "dimension_atol_m": 1e-6, "dimension_rtol": 1e-5},
                "generator": {"name": "perseve", "git_commit": "abc", "isaac_sim_version": "5.1.0", "seed": 1, "config_sha256": "abc"}
            }
            symmetry = {"type": "none", "transforms": [np.eye(3).tolist()], "label_source": "automatic_geometry", "status": "verified_auto", "pipeline_version": "mesh_symmetry_v1", "parameters_sha256": "abc"}
            catalog = {"schema_version": "1.0.0", "objects": {"cad-a": {"mesh_path": "mesh.usd", "mesh_sha256": "abc", "source_length_unit": "m", "source_to_meters": 1.0, "T_cad_from_source_meters": np.eye(4).tolist(), "bbox_min_m": [-0.05, -0.1, -0.15], "bbox_max_m": [0.05, 0.1, 0.15], "base_dimensions_m": [0.1, 0.2, 0.3], "canonical_origin": "local_aabb_center", "symmetry": symmetry}}}
            rgba = np.zeros((8, 10, 4), dtype=np.uint8)
            rgba[2:6, 3:8] = (11, 22, 33, 255)
            cv2.imwrite(str(root / "instance_segmentation_0000.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            cv2.imwrite(str(root / "rgb_0000.png"), np.zeros((8, 10, 3), dtype=np.uint8))
            mapping_key = "(11, 22, 33, 255)"
            (root / "instance_segmentation_mapping_0000.json").write_text(json.dumps({mapping_key: "cad-a"}))
            transform = np.eye(4)
            transform[2, 3] = 1.0
            annotation = {
                "schema_version": "1.0.0", "frame_id": "0000", "scene_id": "0000",
                "image": {"rgb_path": "rgb_0000.png", "instance_path": "instance_segmentation_0000.png", "mapping_path": "instance_segmentation_mapping_0000.json", "size_wh": [10, 8]},
                "camera": {"model": "pinhole", "distortion_model": "none", "K": [[100, 0, 5], [0, 100, 4], [0, 0, 1]], "T_world_from_camera_cv": np.eye(4).tolist()},
                "instances": [{"instance_id": "object_0", "cad_id": "cad-a", "prim_path": "/World/Object_0", "annotation_state": "visible", "pose_training_eligible": True, "mask": {"mapping_key": mapping_key, "value": [11, 22, 33, 255], "value_order": "RGBA", "match_alpha": True}, "bbox_xyxy_px": [3, 2, 7, 5], "T_cam_from_cad": transform.tolist(), "render_scale_xyz": [2, 2, 2], "dimensions_m": [0.2, 0.4, 0.6], "visibility": {"visible_pixel_count": 20, "visible_fraction": None, "truncated": False}}]
            }
            for name, value in (("dataset_meta.json", meta), ("objects.json", catalog), ("pose_annotations_0000.json", annotation)):
                (root / name).write_text(json.dumps(value))
            sample = load_perseve_pose_sample(root, "0000", validate_pixels=True)
            self.assertEqual(sample.frame.instances[0].bbox_xyxy_px, (3, 2, 7, 5))
            self.assertTrue(sample.frame.instances[0].pose_training_eligible)


if __name__ == "__main__":
    unittest.main()
