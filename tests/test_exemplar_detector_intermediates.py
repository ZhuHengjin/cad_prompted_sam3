"""Regression tests for opt-in detector-layer pose supervision outputs."""

import unittest

try:
    import torch

    from muggled_sam.v3_sam.exemplar_detector_model import SAMV3ExemplarDetector

    DEPS_AVAILABLE = True
except ModuleNotFoundError:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "PyTorch is not installed")
class ExemplarDetectorIntermediateTests(unittest.TestCase):
    def make_detector(self) -> "SAMV3ExemplarDetector":
        torch.manual_seed(11)
        detector = SAMV3ExemplarDetector(
            features_per_token=8,
            num_detections=3,
            num_layers=6,
            num_heads=2,
            mlp_ratio=2.0,
        )
        # Production construction immediately restores these three upstream
        # checkpoint parameters; initialize them explicitly in this isolated
        # unit test because the module intentionally allocates them with empty().
        torch.nn.init.normal_(detector.detection_tokens, std=0.02)
        torch.nn.init.zeros_(detector.anchor_boxes_cxcywh)
        torch.nn.init.normal_(detector.presence_token, std=0.02)
        detector.eval()
        return detector

    def test_intermediate_outputs_are_opt_in_and_preserve_final_results(self) -> None:
        detector = self.make_detector()
        image = torch.randn(1, 8, 2, 2)
        exemplars = torch.randn(1, 4, 8)

        standard = detector(image, exemplars)
        with_intermediates = detector(
            image,
            exemplars,
            intermediate_layer_indices=(2, 3, 4, 5),
        )

        self.assertEqual(len(standard), 5)
        self.assertEqual(len(with_intermediates), 6)
        for expected, actual in zip(standard, with_intermediates[:5]):
            torch.testing.assert_close(actual, expected)
        layer_outputs = with_intermediates[5]
        self.assertEqual(tuple(output.layer_index for output in layer_outputs), (2, 3, 4, 5))
        for output in layer_outputs:
            self.assertEqual(output.detection_tokens_bnc.shape, (1, 3, 8))
            self.assertEqual(output.boxes_xy1xy2_bn22.shape, (1, 3, 2, 2))
        torch.testing.assert_close(layer_outputs[-1].detection_tokens_bnc, standard[0])
        torch.testing.assert_close(layer_outputs[-1].boxes_xy1xy2_bn22, standard[1])

    def test_intermediate_loss_reaches_early_fusion_layers(self) -> None:
        detector = self.make_detector()
        image = torch.randn(1, 8, 2, 2)
        exemplars = torch.randn(1, 4, 8)
        outputs = detector(
            image,
            exemplars,
            intermediate_layer_indices=(2, 3, 4),
        )

        auxiliary_loss = sum(
            output.detection_tokens_bnc.square().mean()
            + output.boxes_xy1xy2_bn22.square().mean()
            for output in outputs[5]
        )
        auxiliary_loss.backward()

        early_gradients = [
            parameter.grad
            for parameter in detector.fusion_layers[0].parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in early_gradients))
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in early_gradients if gradient is not None),
            0.0,
        )

    def test_intermediate_layer_validation_rejects_duplicates_and_final_overflow(self) -> None:
        detector = self.make_detector()
        image = torch.randn(1, 8, 2, 2)
        exemplars = torch.randn(1, 4, 8)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            detector(image, exemplars, intermediate_layer_indices=(2, 2))
        with self.assertRaisesRegex(ValueError, "must be in"):
            detector(image, exemplars, intermediate_layer_indices=(6,))


if __name__ == "__main__":
    unittest.main()
