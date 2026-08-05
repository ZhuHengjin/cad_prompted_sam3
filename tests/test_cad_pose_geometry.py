"""Geometry, symmetry, matching, and pose-head regression tests."""

import math
import unittest

try:
    import torch

    from muggled_sam.v3_sam.cad_pose.geometry import (
        adjust_intrinsics_for_resize_and_pad,
        aabb_translation_from_surface_centroid,
        normalize_intrinsics,
        project_points,
        reconstruct_translation,
        rotation_6d_to_matrix,
        surface_centroid_camera,
    )
    from muggled_sam.v3_sam.cad_pose.head import SAMV3CADPoseHead
    from muggled_sam.v3_sam.cad_pose.evaluation import evaluate_pose_matches
    from muggled_sam.v3_sam.cad_pose.losses import (
        CADPoseLossConfig,
        compute_cad_pose_losses,
        nearest_neighbor_distances,
        point_set_pose_errors,
    )
    from muggled_sam.v3_sam.cad_pose.matching import match_pose_predictions_one_to_one
    from muggled_sam.v3_sam.cad_pose.symmetry import symmetry_aware_rotation_error
    from muggled_sam.v3_sam.cad_pose.types import CADPosePredictions, CADPoseTarget

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

    def test_surface_centroid_to_aabb_translation_round_trip(self):
        rotation = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
        translation = torch.tensor([0.12, -0.06, 1.5])
        effective_centroid = torch.tensor([0.03, -0.02, 0.01])
        centroid = surface_centroid_camera(rotation, translation, effective_centroid)
        reconstructed = aabb_translation_from_surface_centroid(
            centroid,
            rotation,
            effective_centroid,
        )
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
        centroid = torch.tensor([[0.001, -0.002, 0.003]])
        small = head(tokens, boxes, torch.tensor([[0.01, 0.02, 0.03]]), centroid, intrinsics, (640, 480))
        large = head(tokens, boxes, torch.tensor([[0.10, 0.20, 0.30]]), centroid * 10, intrinsics, (640, 480))
        self.assertFalse(torch.allclose(small.log_depth_bn, large.log_depth_bn))
        torch.testing.assert_close(small.center_uv_norm_bn2, large.center_uv_norm_bn2)
        torch.testing.assert_close(small.rotation_matrix_bn33, large.rotation_matrix_bn33)

        wider_fov = intrinsics.clone()
        wider_fov[:, 0, 0] = 250.0
        wider_fov[:, 1, 1] = 250.0
        changed_camera = head(
            tokens,
            boxes,
            torch.tensor([[0.01, 0.02, 0.03]]),
            centroid,
            wider_fov,
            (640, 480),
        )
        self.assertFalse(torch.allclose(small.log_depth_bn, changed_camera.log_depth_bn))
        torch.testing.assert_close(small.center_uv_norm_bn2, changed_camera.center_uv_norm_bn2)
        self.assertFalse(any("scale" in name or "size" in name for name, _ in head.named_parameters()))

    def test_pose_head_center_residual_is_box_relative(self):
        torch.manual_seed(4)
        head = SAMV3CADPoseHead(token_dim=8, hidden_dim=16)
        tokens = torch.zeros(1, 1, 8)
        boxes = torch.tensor([[[[0.1, 0.2], [0.5, 0.8]]]])
        dimensions = torch.tensor([[0.01, 0.02, 0.03]])
        centroid = torch.tensor([[0.001, -0.002, 0.003]])
        intrinsics = torch.tensor([[[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]])
        predictions = head(tokens, boxes, dimensions, centroid, intrinsics, (640, 480))
        box_center = torch.tensor([[[0.3, 0.5]]])
        box_extent = torch.tensor([[[0.4, 0.6]]])
        torch.testing.assert_close(
            predictions.center_uv_norm_bn2,
            box_center + predictions.center_residual_bn2 * box_extent,
        )
        reconstructed = predictions.with_translation(intrinsics, (640, 480))
        recovered_centroid = surface_centroid_camera(
            reconstructed.rotation_matrix_bn33,
            reconstructed.translation_m_bn3,
            centroid,
        )
        torch.testing.assert_close(recovered_centroid, reconstructed.centroid_m_bn3)

    def test_pose_only_cad_prompt_is_explicit_masked_and_differentiable(self):
        torch.manual_seed(5)
        head = SAMV3CADPoseHead(token_dim=8, hidden_dim=16)
        tokens = torch.randn(1, 2, 8, requires_grad=True)
        original_tokens = tokens.detach().clone()
        boxes = torch.tensor([[[[0.1, 0.2], [0.4, 0.6]], [[0.2, 0.3], [0.5, 0.7]]]])
        dimensions = torch.tensor([[0.01, 0.02, 0.03]])
        centroid = torch.tensor([[0.001, -0.002, 0.003]])
        intrinsics = torch.tensor([[[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]])
        prompt = torch.randn(1, 3, 8)
        padding_mask = torch.tensor([[False, False, True]])

        disabled = head(
            tokens, boxes, dimensions, centroid, intrinsics, (640, 480), prompt, padding_mask
        )
        baseline = head(tokens, boxes, dimensions, centroid, intrinsics, (640, 480))
        torch.testing.assert_close(disabled.log_depth_bn, baseline.log_depth_bn)

        head.set_cad_prompt_enabled(True)
        prompted = head(
            tokens, boxes, dimensions, centroid, intrinsics, (640, 480), prompt, padding_mask
        )
        changed_padding = prompt.clone()
        changed_padding[:, 2] = 1e4
        prompted_with_changed_padding = head(
            tokens,
            boxes,
            dimensions,
            centroid,
            intrinsics,
            (640, 480),
            changed_padding,
            padding_mask,
        )
        torch.testing.assert_close(
            prompted.log_depth_bn,
            prompted_with_changed_padding.log_depth_bn,
        )
        self.assertFalse(torch.allclose(prompted.log_depth_bn, baseline.log_depth_bn))
        self.assertTrue(torch.equal(tokens.detach(), original_tokens))

        prompted.log_depth_bn.sum().backward()
        attention_gradient = head.cad_prompt_adapter.cross_attention.in_proj_weight.grad
        self.assertIsNotNone(attention_gradient)
        self.assertGreater(float(attention_gradient.abs().sum()), 0.0)

    def test_pose_head_migrates_promptless_v3_state(self):
        torch.manual_seed(6)
        source = SAMV3CADPoseHead(token_dim=8, hidden_dim=16)
        legacy_state = {
            key: value.clone()
            for key, value in source.state_dict().items()
            if key != "cad_prompt_enabled" and not key.startswith("cad_prompt_adapter.")
        }
        target = SAMV3CADPoseHead(token_dim=8, hidden_dim=16)
        prompt_gate_before = target.cad_prompt_adapter.gate.detach().clone()

        migrated = target.load_checkpoint_state_dict(legacy_state, 3)

        self.assertTrue(migrated)
        self.assertFalse(bool(target.cad_prompt_enabled))
        torch.testing.assert_close(target.shared[0].weight, source.shared[0].weight)
        torch.testing.assert_close(target.cad_prompt_adapter.gate, prompt_gate_before)

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

    def test_nearest_neighbor_set_distance_accepts_unlabeled_half_turn(self):
        centered = torch.tensor(
            [[1.0, 0, 0], [-1.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0]]
        )
        half_turn_z = torch.tensor([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
        predicted = centered @ half_turn_z.T
        distances = nearest_neighbor_distances(predicted, centered)
        torch.testing.assert_close(distances, torch.zeros_like(distances))

    def test_point_set_pose_error_uses_centroid_anchor(self):
        centered = torch.tensor(
            [[1.0, 0, 0], [-1.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0]]
        )
        identity = torch.eye(3)
        half_turn_z = torch.tensor([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
        centroid = torch.tensor([0.25, -0.4, 1.2])
        target = CADPoseTarget(
            center_uv_norm=torch.zeros(2),
            log_depth=centroid[2].log(),
            rotation_matrix=identity,
            translation_m=torch.tensor([0.0, 0.0, 1.0]),
            dimensions_m=torch.tensor([2.0, 4.0, 1.0]),
            centroid_m=centroid,
            point_query_m=centered,
            point_target_m=centered,
            point_set_eligible=True,
        )
        rotation_error, full_error, centroid_error, _, _, _ = point_set_pose_errors(
            half_turn_z,
            centroid,
            target,
        )
        self.assertLess(float(rotation_error), 1e-6)
        self.assertLess(float(full_error), 1e-6)
        self.assertLess(float(centroid_error), 1e-6)

    def test_point_set_pose_loss_has_finite_rotation_gradient(self):
        angle = torch.tensor(0.2, requires_grad=True)
        zero = torch.zeros_like(angle)
        one = torch.ones_like(angle)
        rotation = torch.stack(
            (
                torch.stack((torch.cos(angle), -torch.sin(angle), zero)),
                torch.stack((torch.sin(angle), torch.cos(angle), zero)),
                torch.stack((zero, zero, one)),
            )
        )
        centroid = torch.tensor([0.25, -0.4, 1.2])
        points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0]])
        target = CADPoseTarget(
            center_uv_norm=torch.zeros(2),
            log_depth=centroid[2].log(),
            rotation_matrix=torch.eye(3),
            translation_m=torch.tensor([0.0, 0.0, 1.0]),
            dimensions_m=torch.tensor([2.0, 4.0, 1.0]),
            centroid_m=centroid,
            point_query_m=points,
            point_target_m=points,
            point_set_eligible=True,
        )
        predictions = CADPosePredictions(
            center_residual_bn2=torch.zeros(1, 1, 2),
            center_uv_norm_bn2=torch.zeros(1, 1, 2),
            log_depth_bn=target.log_depth.reshape(1, 1),
            rotation_6d_bn6=torch.zeros(1, 1, 6),
            rotation_matrix_bn33=rotation.reshape(1, 1, 3, 3),
            pose_score_logits_bn=torch.zeros(1, 1, requires_grad=True),
            pose_score_bn=torch.full((1, 1), 0.5),
            translation_m_bn3=target.translation_m.reshape(1, 1, 3),
            centroid_m_bn3=centroid.reshape(1, 1, 3),
        )
        losses = compute_cad_pose_losses(
            predictions,
            [target],
            [(0, 0)],
            CADPoseLossConfig(),
        )
        self.assertIsNotNone(losses)
        self.assertTrue(torch.isfinite(losses.total))
        losses.total.backward()
        self.assertIsNotNone(angle.grad)
        self.assertTrue(torch.isfinite(angle.grad))
        self.assertGreater(abs(float(angle.grad)), 1e-8)

    def test_full_pose_point_set_error_backpropagates_to_centroid(self):
        target_centroid = torch.tensor([0.25, -0.4, 1.2])
        predicted_centroid = torch.tensor([0.35, -0.4, 1.2], requires_grad=True)
        points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0]])
        target = CADPoseTarget(
            center_uv_norm=torch.zeros(2),
            log_depth=target_centroid[2].log(),
            rotation_matrix=torch.eye(3),
            translation_m=torch.tensor([0.0, 0.0, 1.0]),
            dimensions_m=torch.tensor([2.0, 4.0, 1.0]),
            centroid_m=target_centroid,
            point_query_m=points,
            point_target_m=points,
            point_set_eligible=True,
        )
        _, full_error, _, _, _, _ = point_set_pose_errors(
            torch.eye(3),
            predicted_centroid,
            target,
            full_pose_grad=True,
        )

        full_error.backward()

        self.assertIsNotNone(predicted_centroid.grad)
        self.assertGreater(float(torch.linalg.vector_norm(predicted_centroid.grad)), 1e-8)

    def test_point_set_evaluation_reports_geometric_success(self):
        centroid = torch.tensor([0.25, -0.4, 1.2])
        points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0]])
        target = CADPoseTarget(
            center_uv_norm=torch.zeros(2),
            log_depth=centroid[2].log(),
            rotation_matrix=torch.eye(3),
            translation_m=torch.tensor([0.0, 0.0, 1.0]),
            dimensions_m=torch.tensor([2.0, 4.0, 1.0]),
            centroid_m=centroid,
            point_query_m=points,
            point_target_m=points,
            point_set_eligible=True,
        )
        predictions = CADPosePredictions(
            center_residual_bn2=torch.zeros(1, 1, 2),
            center_uv_norm_bn2=torch.zeros(1, 1, 2),
            log_depth_bn=target.log_depth.reshape(1, 1),
            rotation_6d_bn6=torch.tensor([[[1.0, 0, 0, 0, 1.0, 0]]]),
            rotation_matrix_bn33=torch.eye(3).reshape(1, 1, 3, 3),
            pose_score_logits_bn=torch.full((1, 1), 10.0),
            pose_score_bn=torch.sigmoid(torch.full((1, 1), 10.0)),
            translation_m_bn3=target.translation_m.reshape(1, 1, 3),
            centroid_m_bn3=centroid.reshape(1, 1, 3),
        )
        evaluation = evaluate_pose_matches(predictions, [target], [(0, 0)])
        self.assertEqual(evaluation.count, 1)
        self.assertLess(evaluation.mean_surface_distance_norm, 1e-6)
        self.assertLess(evaluation.centroid_error_cm, 1e-6)
        self.assertEqual(evaluation.pose_success_rate, 1.0)
        self.assertTrue(math.isnan(evaluation.rotation_error_deg))
        distances = evaluation.surface_distances_norm
        self.assertIsNotNone(distances)
        assert distances is not None
        torch.testing.assert_close(
            distances,
            torch.zeros_like(distances),
        )

    def test_legacy_pose_success_uses_configured_translation_contract(self):
        target = CADPoseTarget(
            center_uv_norm=torch.zeros(2),
            log_depth=torch.tensor(0.0),
            rotation_matrix=torch.eye(3),
            translation_m=torch.tensor([0.0, 0.0, 1.0]),
            dimensions_m=torch.ones(3),
            symmetry_type="none",
            symmetry_transforms=torch.eye(3).unsqueeze(0),
            rotation_eligible=True,
        )
        predictions = CADPosePredictions(
            center_residual_bn2=torch.zeros(1, 1, 2),
            center_uv_norm_bn2=torch.zeros(1, 1, 2),
            log_depth_bn=torch.zeros(1, 1),
            rotation_6d_bn6=torch.tensor([[[1.0, 0, 0, 0, 1.0, 0]]]),
            rotation_matrix_bn33=torch.eye(3).reshape(1, 1, 3, 3),
            pose_score_logits_bn=torch.zeros(1, 1),
            pose_score_bn=torch.full((1, 1), 0.5),
            translation_m_bn3=torch.tensor([[[0.08, 0.0, 1.0]]]),
        )

        absolute = evaluate_pose_matches(
            predictions,
            [target],
            [(0, 0)],
            translation_tolerance=0.05,
        )
        normalized = evaluate_pose_matches(
            predictions,
            [target],
            [(0, 0)],
            translation_tolerance=0.05,
            normalize_translation_error=True,
        )

        self.assertEqual(absolute.pose_success_rate, 0.0)
        self.assertEqual(normalized.pose_success_rate, 1.0)

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
