"""Regression tests for the model-independent CAD-pose renderer."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np

from scripts.visualize_cad_pose_predictions import (
    _spread_reference_view_ids,
    _write_visualizations,
)
from muggled_sam.v3_sam.cad_pose.visualization import (
    GT_COLOR_BGR,
    PRED_COLOR_BGR,
    PoseOverlay,
    canonical_box_corners,
    draw_mask_error_overlay,
    draw_pose_axes,
    draw_pose_box,
    project_camera_points,
    render_pose_comparison,
    render_pose_overlay,
    resize_binary_mask,
    transform_cad_points,
)


class CADPoseVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.intrinsics = np.asarray(
            [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]
        )

    def test_identity_transform_and_projection_follow_opencv_convention(self):
        points = np.asarray([[0.0, 0.0, 2.0], [0.2, -0.4, 2.0]])
        camera = transform_cad_points(points, np.eye(3), np.asarray([0.1, 0.2, 0.0]))
        pixels, valid = project_camera_points(camera, self.intrinsics)

        np.testing.assert_allclose(camera, [[0.1, 0.2, 2.0], [0.3, -0.2, 2.0]])
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_allclose(pixels, [[85.0, 70.0], [95.0, 50.0]])

    def test_four_display_exemplars_are_spread_across_configured_views(self):
        selected = _spread_reference_view_ids([str(index) for index in range(12)], 4)
        self.assertEqual(selected, ["0", "4", "7", "11"])

    def test_box_is_origin_centered_and_invalid_camera_points_are_skipped(self):
        corners = canonical_box_corners(np.asarray([2.0, 4.0, 6.0]))
        np.testing.assert_allclose(corners.min(axis=0), [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(corners.max(axis=0), [1.0, 2.0, 3.0])

        pixels, valid = project_camera_points(
            np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [np.nan, 0.0, 1.0]]),
            self.intrinsics,
        )
        np.testing.assert_array_equal(valid, [True, False, False])
        np.testing.assert_allclose(pixels[0], [80.0, 60.0])
        self.assertTrue(np.isnan(pixels[1:]).all())

    def test_nearest_resize_preserves_a_binary_mask(self):
        mask = np.asarray([[False, True], [False, False]])
        resized = resize_binary_mask(mask, (4, 4))
        expected = cv2.resize(mask.astype(np.uint8), (4, 4), interpolation=cv2.INTER_NEAREST)
        self.assertEqual(resized.dtype, np.bool_)
        np.testing.assert_array_equal(resized, expected > 0)

    def test_surface_centroid_and_public_translation_point_paths_are_equivalent(self):
        points = np.asarray([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [-0.1, -0.1, 0.0]])
        surface_centroid = points.mean(axis=0)
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.asarray([0.2, -0.1, 1.5])
        centroid_camera = surface_centroid @ rotation.T + translation

        direct = transform_cad_points(points, rotation, translation)
        centered_path = (points - surface_centroid) @ rotation.T + centroid_camera
        np.testing.assert_allclose(direct, centered_path)

    def test_renderer_does_not_mutate_input_and_comparison_has_three_panels(self):
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        mask[45:75, 65:95] = True
        points = canonical_box_corners(np.asarray([0.4, 0.4, 0.4]))
        gt = PoseOverlay(
            label="GT",
            rotation_cam_from_cad=np.eye(3),
            translation_cam_from_cad_m=np.asarray([0.0, 0.0, 2.0]),
            dimensions_m=np.asarray([0.4, 0.4, 0.4]),
            color_bgr=GT_COLOR_BGR,
            mask_hw=mask,
            points_cad_m=points,
        )
        prediction = PoseOverlay(
            label="prediction",
            rotation_cam_from_cad=np.eye(3),
            translation_cam_from_cad_m=np.asarray([0.0, 0.0, 2.0]),
            dimensions_m=np.asarray([0.4, 0.4, 0.4]),
            color_bgr=PRED_COLOR_BGR,
            mask_hw=mask,
            points_cad_m=points,
        )
        source = self.image.copy()

        rendered = render_pose_overlay(self.image, self.intrinsics, [gt])
        comparison = render_pose_comparison(
            self.image,
            self.intrinsics,
            gt,
            prediction,
            metric_lines=["mask IoU 1.000"],
        )

        np.testing.assert_array_equal(self.image, source)
        self.assertGreater(int(rendered.sum()), 0)
        self.assertEqual(comparison.shape, (120, 480, 3))

    def test_box_axes_and_mask_error_can_be_rendered_independently(self):
        overlay = PoseOverlay(
            label="pose",
            rotation_cam_from_cad=np.eye(3),
            translation_cam_from_cad_m=np.asarray([0.0, 0.0, 2.0]),
            dimensions_m=np.asarray([0.4, 0.3, 0.2]),
            color_bgr=GT_COLOR_BGR,
        )
        box_only = draw_pose_box(self.image, overlay, self.intrinsics)
        axes_only = draw_pose_axes(self.image, overlay, self.intrinsics)
        self.assertGreater(int(box_only.sum()), 0)
        self.assertGreater(int(axes_only.sum()), 0)
        self.assertFalse(np.array_equal(box_only, axes_only))

        gt_mask = np.zeros(self.image.shape[:2], dtype=bool)
        pred_mask = np.zeros_like(gt_mask)
        gt_mask[20:40, 20:40] = True
        pred_mask[30:50, 30:50] = True
        error = draw_mask_error_overlay(self.image, gt_mask, pred_mask, alpha=1.0)
        np.testing.assert_array_equal(error[35, 35], GT_COLOR_BGR)
        np.testing.assert_array_equal(error[45, 45], PRED_COLOR_BGR)
        np.testing.assert_array_equal(error[25, 25], [40, 150, 245])

    def test_instance_output_writes_every_component_and_four_exemplars(self):
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        mask[45:75, 65:95] = True
        instance = SimpleNamespace(
            instance_id="instance_1",
            rotation_matrix=np.eye(3),
            translation_m=np.asarray([0.0, 0.0, 2.0]),
            dimensions_m=np.asarray([0.4, 0.3, 0.2]),
            render_scale_xyz=None,
        )
        sample = SimpleNamespace(
            catalog={"cad_1": SimpleNamespace(point_set=None)},
            frame=SimpleNamespace(scene_id="scene_1"),
        )
        exemplars = [np.full((40, 60, 3), value, np.uint8) for value in (20, 40, 60, 80)]
        with TemporaryDirectory() as directory:
            record = _write_visualizations(
                image_bgr=self.image,
                intrinsics=self.intrinsics,
                sample=sample,
                instance=instance,
                gt_mask=mask,
                prediction=(np.eye(3), np.asarray([0.01, 0.0, 2.05])),
                prediction_mask=mask,
                evaluation=None,
                mask_iou=1.0,
                detection_score=0.9,
                pose_score=0.8,
                output_dir=Path(directory),
                entry={"dataset_id": "dataset", "group_id": "group", "frame_id": "0001"},
                scene_index=0,
                cad_id="cad_1",
                status_reason=None,
                max_render_points=512,
                line_thickness=1,
                exemplar_images=exemplars,
            )

            self.assertEqual(len(record["images"]), 15)
            self.assertEqual(record["exemplar_count"], 4)
            for path in record["images"].values():
                self.assertTrue(Path(path).is_file())
            clean_rgb = cv2.imread(record["images"]["rgb"], cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(clean_rgb, self.image)
            overview = cv2.imread(record["images"]["overview"], cv2.IMREAD_COLOR)
            self.assertGreater(overview.shape[0], self.image.shape[0])
            self.assertGreater(overview.shape[1], self.image.shape[1] * 2)
            self.assertTrue((Path(record["directory"]) / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
