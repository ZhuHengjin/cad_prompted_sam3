"""Numerical regression test for the consolidated multi-GT loss helper.

The test reconstructs the exact pre-consolidation mask, presence, and bounding
box loss expression and compares it with ``compute_multi_gt_detection_loss`` on
fixed random tensors. It is skipped on preprocessing-only systems where the
optional PyTorch/OpenCV training stack is unavailable.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch

    from finetune_image_exemplar_multi_gt import (
        AuxiliaryPosePredictions,
        compute_auxiliary_pose_loss,
        compute_multi_gt_detection_loss,
        compute_multi_gt_detection_losses,
        filter_pose_matches_by_iou,
        match_predictions_to_gts_greedy_k,
        parse_pose_aux_layer_indices,
    )
    from muggled_sam.v3_sam.cad_pose.losses import CADPoseLossConfig
    from loss_fns import compute_bbox_l1_loss_from_matches, compute_matched_mask_losses, compute_presence_loss_logits

    TRAINING_DEPS_AVAILABLE = True
except ModuleNotFoundError:
    TRAINING_DEPS_AVAILABLE = False


@unittest.skipUnless(TRAINING_DEPS_AVAILABLE, "torch/cv2 training dependencies are not installed")
class MultiGtLossEquivalenceTests(unittest.TestCase):
    def test_pose_aux_layers_use_one_based_configuration(self):
        self.assertEqual(parse_pose_aux_layer_indices("3,4,5", 6), (2, 3, 4))
        with self.assertRaisesRegex(ValueError, "primary output"):
            parse_pose_aux_layer_indices("6", 6)

    def test_auxiliary_pose_loss_is_mean_with_quality_disabled(self):
        predictions = AuxiliaryPosePredictions(
            layers=(3, 4, 5),
            predictions=(object(), object(), object()),
        )
        calls = []

        def fake_pose_loss(prediction, targets, matches, config, **kwargs):
            calls.append((prediction, targets, matches, config, kwargs))
            return SimpleNamespace(total=torch.tensor(float(len(calls))))

        matches = [(0, 7)]
        with patch(
            "finetune_image_exemplar_multi_gt.compute_cad_pose_losses",
            side_effect=fake_pose_loss,
        ):
            mean_loss = compute_auxiliary_pose_loss(
                predictions,
                [object()],
                matches,
                CADPoseLossConfig(quality_weight=1.0),
                batch_index=2,
            )

        torch.testing.assert_close(mean_loss, torch.tensor(2.0))
        torch.testing.assert_close(0.5 * mean_loss, torch.tensor(1.0))
        self.assertEqual(len(calls), 3)
        for _, _, actual_matches, config, kwargs in calls:
            self.assertIs(actual_matches, matches)
            self.assertEqual(config.quality_weight, 0.0)
            self.assertFalse(kwargs["compute_expensive_metrics"])
            self.assertEqual(kwargs["batch_index"], 2)

    def test_refactored_training_loss_matches_previous_expression(self):
        torch.manual_seed(7)
        logits = torch.randn(4, 8, 8)
        boxes = torch.rand(4, 2, 2)
        score_logits = torch.randn(4)
        targets = [(torch.rand(8, 8) > 0.65).float(), (torch.rand(8, 8) > 0.7).float()]
        matches, ious = match_predictions_to_gts_greedy_k(logits, targets, max_matches=None, max_per_gt=12)
        matched_losses = compute_matched_mask_losses(logits, targets, matches, bce_weight=2.0, dice_weight=2.0)
        expected = (
            torch.stack(matched_losses).mean() * 2.0
            + compute_presence_loss_logits(
                score_logits,
                matches,
                ious,
                pos_weight=0.3,
                neg_weight=0.45,
                alpha=0.5,
                use_focal=False,
                focal_alpha=0.25,
                focal_gamma=4.0,
                focal_weight=300.0,
            )
            + compute_bbox_l1_loss_from_matches(boxes, targets, matches)
        )
        actual = compute_multi_gt_detection_loss(
            logits,
            boxes,
            score_logits,
            targets,
            bce_weight=2.0,
            dice_weight=2.0,
            bbox_weight=1.0,
            score_weight=0.3,
            no_object_weight=0.45,
        )
        self.assertIsNotNone(actual)
        torch.testing.assert_close(actual, expected)

    def test_joint_lite_components_can_be_reweighted_independently(self):
        torch.manual_seed(11)
        logits = torch.randn(3, 6, 6)
        boxes = torch.rand(3, 2, 2)
        score_logits = torch.randn(3)
        targets = [(torch.rand(6, 6) > 0.6).float()]

        losses = compute_multi_gt_detection_losses(
            logits,
            boxes,
            score_logits,
            targets,
            bce_weight=2.0,
            dice_weight=2.0,
            score_weight=0.3,
            no_object_weight=0.45,
        )

        self.assertIsNotNone(losses)
        actual = losses.total(mask_weight=0.10, bbox_weight=0.25, objectness_weight=0.25)
        expected = 0.10 * losses.mask + 0.25 * losses.bbox + 0.25 * losses.objectness
        torch.testing.assert_close(actual, expected)

    def test_ground_truth_detection_predictions_minimize_joint_loss(self):
        target_a = torch.zeros(8, 10)
        target_a[1:4, 2:5] = 1.0
        target_b = torch.zeros(8, 10)
        target_b[4:7, 6:9] = 1.0
        targets = [target_a, target_b]
        perfect_logits = torch.stack(
            (
                torch.where(target_a > 0.5, 20.0, -20.0),
                torch.where(target_b > 0.5, 20.0, -20.0),
                torch.full_like(target_a, -20.0),
            )
        )
        perfect_boxes = torch.tensor(
            [
                [[2 / 9, 1 / 7], [4 / 9, 3 / 7]],
                [[6 / 9, 4 / 7], [8 / 9, 6 / 7]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
        perfect_scores = torch.tensor([20.0, 20.0, -20.0])

        losses = compute_multi_gt_detection_losses(
            perfect_logits,
            perfect_boxes,
            perfect_scores,
            targets,
            bce_weight=2.0,
            dice_weight=2.0,
            score_weight=0.3,
            no_object_weight=0.45,
            max_per_gt=1,
        )

        self.assertIsNotNone(losses)
        assert losses is not None
        self.assertLess(float(losses.mask), 1e-6)
        torch.testing.assert_close(losses.bbox, torch.zeros_like(losses.bbox))
        self.assertLess(float(losses.objectness), 1e-6)
        self.assertLess(
            float(losses.total(mask_weight=2.0, bbox_weight=1.0, objectness_weight=1.0)),
            3e-6,
        )

        wrong_mask_logits = perfect_logits.clone()
        wrong_mask_logits[0, 2, 3] = -20.0
        wrong_mask_losses = compute_multi_gt_detection_losses(
            wrong_mask_logits,
            perfect_boxes,
            perfect_scores,
            targets,
            bce_weight=2.0,
            dice_weight=2.0,
            score_weight=0.3,
            no_object_weight=0.45,
            max_per_gt=1,
        )
        self.assertIsNotNone(wrong_mask_losses)
        assert wrong_mask_losses is not None
        self.assertGreater(float(wrong_mask_losses.mask), float(losses.mask))
        torch.testing.assert_close(wrong_mask_losses.bbox, losses.bbox)

        wrong_boxes = perfect_boxes.clone()
        wrong_boxes[0, 0, 0] += 0.1
        wrong_box_losses = compute_multi_gt_detection_losses(
            perfect_logits,
            wrong_boxes,
            perfect_scores,
            targets,
            bce_weight=2.0,
            dice_weight=2.0,
            score_weight=0.3,
            no_object_weight=0.45,
            max_per_gt=1,
        )
        self.assertIsNotNone(wrong_box_losses)
        assert wrong_box_losses is not None
        self.assertGreater(float(wrong_box_losses.bbox), float(losses.bbox))
        torch.testing.assert_close(wrong_box_losses.mask, losses.mask)

        wrong_scores = perfect_scores.clone()
        wrong_scores[0] = -20.0
        wrong_score_losses = compute_multi_gt_detection_losses(
            perfect_logits,
            perfect_boxes,
            wrong_scores,
            targets,
            bce_weight=2.0,
            dice_weight=2.0,
            score_weight=0.3,
            no_object_weight=0.45,
            max_per_gt=1,
        )
        self.assertIsNotNone(wrong_score_losses)
        assert wrong_score_losses is not None
        self.assertGreater(float(wrong_score_losses.objectness), float(losses.objectness))
        torch.testing.assert_close(wrong_score_losses.mask, losses.mask)
        torch.testing.assert_close(wrong_score_losses.bbox, losses.bbox)

    def test_pose_match_iou_filter_keeps_only_qualified_assignments(self):
        iou = torch.tensor(
            [
                [0.49, 0.80, 0.10],
                [0.20, 0.30, 0.50],
            ]
        )
        matches = [(0, 1), (1, 2)]

        self.assertEqual(filter_pose_matches_by_iou(matches, iou, 0.5), matches)
        self.assertEqual(filter_pose_matches_by_iou(matches, iou, 0.7), [(0, 1)])
        self.assertEqual(filter_pose_matches_by_iou(matches, iou, 0.9), [])

    def test_pose_match_iou_filter_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "min_iou"):
            filter_pose_matches_by_iou([], torch.zeros(0, 0), 1.1)


if __name__ == "__main__":
    unittest.main()
