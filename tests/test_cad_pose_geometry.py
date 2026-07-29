"""Geometry, symmetry, matching, and pose-head regression tests."""

import math
import unittest

try:
    import torch

    from muggled_sam.v3_sam.cad_pose.geometry import (
        adjust_intrinsics_for_resize_and_pad,
        normalize_intrinsics,
        project_points,
        reconstruct_translation,
        rotation_6d_to_matrix,
    )
    from muggled_sam.v3_sam.cad_pose.head import SAMV3CADPoseHead
    from muggled_sam.v3_sam.cad_pose.matching import match_pose_predictions_one_to_one
    from muggled_sam.v3_sam.cad_pose.symmetry import symmetry_aware_rotation_error

    DEPS_AVAILABLE = True
except ModuleNotFoundError:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "PyTorch is not installed")
class CADPoseGeometryTests(unittest.TestCase):
    def test_rotation_6d_is_proper(self):
        matrix = rotation_6d_to_matrix(torch.tensor([1.0, 0, 0, 0, 1.0, 0]))
        torch.testing.assert_close(matrix, torch.eye(3))
        torch.testing.assert_close(matrix.T @ matrix, torch.eye(3))
        torch.testing.assert_close(torch.linalg.det(matrix), torch.tensor(1.0))
        degenerate = rotation_6d_to_matrix(torch.zeros(6))
        torch.testing.assert_close(degenerate.T @ degenerate, torch.eye(3))
        torch.testing.assert_close(torch.linalg.det(degenerate), torch.tensor(1.0))

    def test_translation_round_trip(self):
        intrinsics = torch.tensor([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        translation = torch.tensor([0.12, -0.06, 1.5])
        center = project_points(translation, intrinsics)
        reconstructed = reconstruct_translation(center, translation[2].log(), intrinsics)
        torch.testing.assert_close(reconstructed, translation)

    def test_projection_is_preserved_by_anisotropic_model_resize(self):
        intrinsics = torch.tensor([[400.0, 0, 320.0], [0, 420.0, 240.0], [0, 0, 1.0]])
        point = torch.tensor([0.1, -0.2, 2.0])
        original = project_points(point, intrinsics)
        adjusted = adjust_intrinsics_for_resize_and_pad(intrinsics, (640, 480), (1008, 1008))
        resized = project_points(point, adjusted)
        torch.testing.assert_close(resized, original * torch.tensor([1008 / 640, 1008 / 480]))

    def test_intrinsics_normalization_matches_normalized_image_coordinates(self):
        intrinsics = torch.tensor([[500.0, 2.0, 320.0], [0, 510.0, 240.0], [0, 0, 1.0]])
        normalized = normalize_intrinsics(intrinsics, (641, 481))
        expected = torch.tensor([[500 / 640, 2 / 640, 0.5], [0, 510 / 480, 0.5], [0, 0, 1.0]])
        torch.testing.assert_close(normalized, expected)

    def test_pose_head_routes_absolute_size_and_intrinsics_to_depth_only(self):
        torch.manual_seed(3)
        head = SAMV3CADPoseHead(token_dim=8, hidden_dim=16)
        tokens = torch.zeros(1, 2, 8)
        boxes = torch.tensor([[[[0.1, 0.2], [0.4, 0.6]], [[0.2, 0.3], [0.5, 0.7]]]])
        intrinsics = torch.tensor([[[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]])
        small = head(tokens, boxes, torch.tensor([[0.01, 0.02, 0.03]]), intrinsics, (640, 480))
        large = head(tokens, boxes, torch.tensor([[0.10, 0.20, 0.30]]), intrinsics, (640, 480))
        self.assertFalse(torch.allclose(small.log_depth_bn, large.log_depth_bn))
        torch.testing.assert_close(small.center_uv_norm_bn2, large.center_uv_norm_bn2)
        torch.testing.assert_close(small.rotation_matrix_bn33, large.rotation_matrix_bn33)

        wider_fov = intrinsics.clone()
        wider_fov[:, 0, 0] = 250.0
        wider_fov[:, 1, 1] = 250.0
        changed_camera = head(tokens, boxes, torch.tensor([[0.01, 0.02, 0.03]]), wider_fov, (640, 480))
        self.assertFalse(torch.allclose(small.log_depth_bn, changed_camera.log_depth_bn))
        torch.testing.assert_close(small.center_uv_norm_bn2, changed_camera.center_uv_norm_bn2)
        self.assertFalse(any("scale" in name or "size" in name for name, _ in head.named_parameters()))

    def test_pose_head_center_residual_is_box_relative(self):
        torch.manual_seed(4)
        head = SAMV3CADPoseHead(token_dim=8, hidden_dim=16)
        tokens = torch.zeros(1, 1, 8)
        boxes = torch.tensor([[[[0.1, 0.2], [0.5, 0.8]]]])
        dimensions = torch.tensor([[0.01, 0.02, 0.03]])
        intrinsics = torch.tensor([[[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]])
        predictions = head(tokens, boxes, dimensions, intrinsics, (640, 480))
        box_center = torch.tensor([[[0.3, 0.5]]])
        box_extent = torch.tensor([[[0.4, 0.6]]])
        torch.testing.assert_close(
            predictions.center_uv_norm_bn2,
            box_center + predictions.center_residual_bn2 * box_extent,
        )

    def test_discrete_and_continuous_symmetries(self):
        identity = torch.eye(3)
        half_turn_z = torch.tensor([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
        discrete_error = symmetry_aware_rotation_error(
            half_turn_z,
            identity,
            symmetry_type="discrete",
            symmetry_transforms=torch.stack((identity, half_turn_z)),
        )
        continuous_error = symmetry_aware_rotation_error(
            half_turn_z,
            identity,
            symmetry_type="continuous_axis",
            axis_cad=torch.tensor([0.0, 0, 1.0]),
        )
        self.assertLess(float(discrete_error), 1e-5)
        self.assertLess(float(continuous_error), 1e-5)

    def test_pose_matching_is_one_to_one(self):
        target_a = torch.zeros(8, 8)
        target_a[:4, :4] = 1
        target_b = torch.zeros(8, 8)
        target_b[4:, 4:] = 1
        predictions = torch.full((3, 8, 8), -10.0)
        predictions[0, :4, :4] = 10
        predictions[1, 4:, 4:] = 10
        predictions[2] = predictions[0]
        matches, _ = match_pose_predictions_one_to_one(predictions, [target_a, target_b])
        self.assertEqual(len(matches), 2)
        self.assertEqual(len({prediction for _, prediction in matches}), 2)
        self.assertEqual({ground_truth for ground_truth, _ in matches}, {0, 1})


if __name__ == "__main__":
    unittest.main()
