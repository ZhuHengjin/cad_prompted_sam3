"""Numerical regression test for the consolidated multi-GT loss helper.

The test reconstructs the exact pre-consolidation mask, presence, and bounding
box loss expression and compares it with ``compute_multi_gt_detection_loss`` on
fixed random tensors. It is skipped on preprocessing-only systems where the
optional PyTorch/OpenCV training stack is unavailable.
"""

import unittest

try:
    import torch

    from finetune_image_exemplar_multi_gt import (
        compute_multi_gt_detection_loss,
        match_predictions_to_gts_greedy_k,
    )
    from loss_fns import compute_bbox_l1_loss_from_matches, compute_matched_mask_losses, compute_presence_loss_logits

    TRAINING_DEPS_AVAILABLE = True
except ModuleNotFoundError:
    TRAINING_DEPS_AVAILABLE = False


@unittest.skipUnless(TRAINING_DEPS_AVAILABLE, "torch/cv2 training dependencies are not installed")
class MultiGtLossEquivalenceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
