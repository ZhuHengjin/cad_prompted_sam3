"""Tests for structured reference views and camera-conditioned token residuals."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from finetune_image_exemplar_multi_gt import (
    load_detector_initialization_checkpoint,
    load_finetune_checkpoint,
)
from muggled_sam.v3_sam.exemplar_view_pose import (
    EXEMPLAR_VIEW_MODES,
    ExemplarViewBundle,
    ExemplarViewPoseEncoder,
    load_exemplar_view_adapter_for_inference,
    load_reference_view_rotations,
    pad_exemplar_view_batch,
)


def make_bundle(token_dim: int = 8) -> ExemplarViewBundle:
    rotations = torch.stack(
        (
            torch.eye(3),
            torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
            ),
            torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
            ),
        )
    )
    return ExemplarViewBundle(
        tokens_bnc=torch.arange(6 * token_dim, dtype=torch.float32).reshape(1, 6, token_dim),
        token_view_indices_n=torch.tensor([0, 0, 1, 1, 1, 2]),
        view_rotations_v33=rotations,
        view_ids=("00", "01", "02"),
        object_id="cad-a",
    )


class ExemplarViewPoseTests(unittest.TestCase):
    def test_metadata_loader_preserves_requested_order_and_normalizes_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cad-a_render_transform.json"
            path.write_text(
                json.dumps(
                    {
                        "camera_frame": "opencv_x_right_y_down_z_forward",
                        "views": [
                            {"view_id": "01", "R_refcam_cv_from_cad": torch.eye(3).tolist()},
                            {
                                "view_id": "00",
                                "R_refcam_cv_from_cad": torch.diag(
                                    torch.tensor([-1.0, 1.0, -1.0])
                                ).tolist(),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rotations = load_reference_view_rotations(path, ["0", "1"])
            self.assertEqual(tuple(rotations.shape), (2, 3, 3))
            torch.testing.assert_close(
                rotations[0], torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
            )
            torch.testing.assert_close(rotations[1], torch.eye(3))

    def test_metadata_loader_rejects_improper_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "camera_frame": "opencv_x_right_y_down_z_forward",
                        "views": [
                            {
                                "view_id": "00",
                                "R_refcam_cv_from_cad": torch.diag(
                                    torch.tensor([-1.0, 1.0, 1.0])
                                ).tolist(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Improper camera rotation"):
                load_reference_view_rotations(path, ["00"])

    def test_none_mode_is_exactly_the_legacy_padding_path(self) -> None:
        bundle = make_bundle()
        shorter = bundle.tokens_bnc[:, :3].clone()
        legacy_inputs = [bundle.tokens_bnc, shorter]
        max_tokens = max(tensor.shape[1] for tensor in legacy_inputs)
        expected_parts = []
        expected_masks = []
        for tensor in legacy_inputs:
            num_tokens = tensor.shape[1]
            mask = torch.zeros(max_tokens, dtype=torch.bool)
            if num_tokens < max_tokens:
                tensor = torch.cat(
                    (
                        tensor,
                        torch.zeros(
                            1,
                            max_tokens - num_tokens,
                            tensor.shape[2],
                            dtype=tensor.dtype,
                        ),
                    ),
                    dim=1,
                )
                mask[num_tokens:] = True
            expected_parts.append(tensor)
            expected_masks.append(mask)
        expected = torch.cat(expected_parts, dim=0)
        expected_mask = torch.stack(expected_masks, dim=0)
        actual, actual_mask = pad_exemplar_view_batch(
            [bundle, shorter], device=torch.device("cpu"), mode="none"
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_mask, expected_mask))

    def test_every_learned_mode_is_a_noop_at_initialization(self) -> None:
        bundle = make_bundle()
        encoder = ExemplarViewPoseEncoder(token_dim=bundle.tokens_bnc.shape[-1])
        for mode in EXEMPLAR_VIEW_MODES:
            actual = encoder.forward_bundle(bundle, mode=mode, shuffle_seed=17)
            self.assertTrue(torch.equal(actual, bundle.tokens_bnc), mode)

    def test_camera_adapter_receives_gradients_through_cached_tokens(self) -> None:
        bundle = make_bundle()
        encoder = ExemplarViewPoseEncoder(token_dim=bundle.tokens_bnc.shape[-1])
        result = encoder.forward_bundle(bundle.detach_cpu(), mode="camera")
        result.square().mean().backward()
        output_projection = encoder.camera_mlp[-1]
        self.assertIsNotNone(output_projection.weight.grad)
        self.assertGreater(float(output_projection.weight.grad.abs().sum()), 0.0)

    def test_identity_rotation_uses_opencv_forward_and_up_axes(self) -> None:
        directions = ExemplarViewPoseEncoder._camera_directions_in_cad(
            torch.eye(3).unsqueeze(0)
        )
        torch.testing.assert_close(
            directions,
            torch.tensor([[0.0, 0.0, 1.0, 0.0, -1.0, 0.0]]),
        )

    def test_shuffle_is_stable_and_object_specific(self) -> None:
        first = ExemplarViewPoseEncoder._stable_permutation(12, "cad-a", 42)
        repeated = ExemplarViewPoseEncoder._stable_permutation(12, "cad-a", 42)
        other = ExemplarViewPoseEncoder._stable_permutation(12, "cad-b", 42)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, other))
        self.assertFalse(torch.any(first == torch.arange(12)))

    def test_inference_mode_and_state_are_restored_from_checkpoint(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.exemplar_view_pose_encoder = ExemplarViewPoseEncoder(token_dim=8)

        source = Model()
        checkpoint = {
            "args": {"exemplar_view_mode": "camera"},
            "exemplar_view_pose_encoder": source.exemplar_view_pose_encoder.state_dict(),
            "exemplar_view_pose_architecture_version": 1,
            "exemplar_view_pose_architecture_config": (
                source.exemplar_view_pose_encoder.architecture_config()
            ),
        }
        target = Model()
        self.assertEqual(
            load_exemplar_view_adapter_for_inference(target, checkpoint, "auto"),
            "camera",
        )
        with self.assertRaisesRegex(ValueError, "not 'view_id'"):
            load_exemplar_view_adapter_for_inference(target, checkpoint, "view_id")

    def test_training_resume_rejects_cross_mode_checkpoint(self) -> None:
        class PoseHead(torch.nn.Linear):
            architecture_version = 1

        class Detector(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.image_exemplar_fusion = torch.nn.Linear(2, 2)
                self.exemplar_detector = torch.nn.Linear(2, 2)
                self.exemplar_segmentation = torch.nn.Linear(2, 2)
                self.cad_pose_head = PoseHead(2, 2)
                self.exemplar_view_pose_encoder = ExemplarViewPoseEncoder(token_dim=8)

        source = Detector()
        checkpoint = {
            "args": {"enable_pose": True, "exemplar_view_mode": "camera"},
            "image_exemplar_fusion": source.image_exemplar_fusion.state_dict(),
            "exemplar_detector": source.exemplar_detector.state_dict(),
            "exemplar_segmentation": source.exemplar_segmentation.state_dict(),
            "exemplar_view_pose_encoder": source.exemplar_view_pose_encoder.state_dict(),
            "exemplar_view_pose_architecture_version": 1,
            "exemplar_view_pose_architecture_config": (
                source.exemplar_view_pose_encoder.architecture_config()
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.pth"
            torch.save(checkpoint, path)
            load_finetune_checkpoint(
                path,
                Detector(),
                optimizer=None,
                device=torch.device("cpu"),
                exemplar_view_mode="camera",
            )
            with self.assertRaisesRegex(ValueError, "incompatible with requested mode"):
                load_finetune_checkpoint(
                    path,
                    Detector(),
                    optimizer=None,
                    device=torch.device("cpu"),
                    exemplar_view_mode="view_id",
                )

    def test_segmentation_initializer_does_not_load_pose_or_adapter_state(self) -> None:
        class PoseHead(torch.nn.Linear):
            architecture_version = 1

        class Detector(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.image_exemplar_fusion = torch.nn.Linear(2, 2)
                self.exemplar_detector = torch.nn.Linear(2, 2)
                self.exemplar_segmentation = torch.nn.Linear(2, 2)
                self.cad_pose_head = PoseHead(2, 2)
                self.exemplar_view_pose_encoder = ExemplarViewPoseEncoder(token_dim=8)

        source = Detector()
        checkpoint = {
            "args": {"enable_pose": False},
            "image_exemplar_fusion": source.image_exemplar_fusion.state_dict(),
            "exemplar_detector": source.exemplar_detector.state_dict(),
            "exemplar_segmentation": source.exemplar_segmentation.state_dict(),
            "cad_pose_head": source.cad_pose_head.state_dict(),
            "exemplar_view_pose_encoder": source.exemplar_view_pose_encoder.state_dict(),
        }
        target = Detector()
        pose_before = {key: value.clone() for key, value in target.cad_pose_head.state_dict().items()}
        adapter_before = {
            key: value.clone()
            for key, value in target.exemplar_view_pose_encoder.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "segmentation.pth"
            torch.save(checkpoint, path)
            load_detector_initialization_checkpoint(path, target, torch.device("cpu"))
        for key, value in source.image_exemplar_fusion.state_dict().items():
            self.assertTrue(torch.equal(target.image_exemplar_fusion.state_dict()[key], value))
        for key, value in pose_before.items():
            self.assertTrue(torch.equal(target.cad_pose_head.state_dict()[key], value))
        for key, value in adapter_before.items():
            self.assertTrue(
                torch.equal(target.exemplar_view_pose_encoder.state_dict()[key], value)
            )


if __name__ == "__main__":
    unittest.main()
