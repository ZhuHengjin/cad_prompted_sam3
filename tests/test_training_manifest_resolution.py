"""Regression tests for resolving manifest rows into supervised samples."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from dataset_manifest import ManifestRow
from finetune_image_exemplar_multi_gt import (
    entries_from_manifest_rows,
    checksum_set_digest,
    initialize_metrics_log,
    load_finetune_checkpoint,
    pose_prompt_surface_centroid_m,
    reference_view_id_candidates,
    resolve_run_dir_from_checkpoint,
    resolve_reference_pair,
    surface_distance_percentile,
    validate_pose_reference_metadata,
    validate_resume_manifest_checksum,
    validate_pose_resume_provenance,
)


class TrainingManifestResolutionTests(unittest.TestCase):
    def test_checkpoint_path_resolves_new_and_legacy_run_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            current_run = root / "experiment" / "run_20260801_120000"
            legacy_run = root / "experiment" / "run_20260701_120000"
            self.assertEqual(
                resolve_run_dir_from_checkpoint(
                    current_run / "checkpoints" / "finetune.pth"
                ),
                current_run,
            )
            self.assertEqual(
                resolve_run_dir_from_checkpoint(legacy_run / "finetune.pth"),
                legacy_run,
            )

    def test_metrics_log_migrates_the_previous_header_by_column_name(self):
        previous_fields = (
            "phase",
            "epoch",
            "global_step",
            "batch_step",
            "loss",
            "avg_loss",
            "avg_iou",
            "correct_rate",
            "pose_center_loss",
            "pose_depth_loss",
            "pose_rotation_loss",
            "pose_quality_loss",
            "rotation_error_deg",
            "translation_error_cm",
            "center_error_norm",
            "depth_error_m",
            "accuracy_5deg_5cm",
            "accuracy_10deg_10cm",
            "brier_score",
            "expected_calibration_error",
            "pose_score_temperature",
            "samples",
        )
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.csv"
            with metrics_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=previous_fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "phase": "train_batch",
                        "epoch": 3,
                        "global_step": 12,
                        "pose_center_loss": 2.0,
                        "samples": 9,
                    }
                )

            initialize_metrics_log(metrics_path)

            with metrics_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["pose_center_loss"], "2.0")
            self.assertEqual(rows[0]["samples"], "9")
            self.assertEqual(rows[0]["mask_loss"], "")
            self.assertNotIn(None, rows[0])

    def test_surface_percentile_combines_raw_distance_chunks(self):
        chunks = [torch.tensor([0.0, 1.0]), torch.tensor([10.0, 11.0])]
        expected = float(torch.quantile(torch.cat(chunks), 0.75))
        self.assertEqual(surface_distance_percentile(chunks, 0.75), expected)

    def test_detection_only_pose_prompt_does_not_require_uniform_scale(self):
        with patch(
            "finetune_image_exemplar_multi_gt.effective_surface_centroid_m"
        ) as effective_centroid:
            centroid = pose_prompt_surface_centroid_m([object()], [], object())
        np.testing.assert_array_equal(centroid, np.zeros(3))
        effective_centroid.assert_not_called()

    def test_resume_provenance_checks_annotations_and_schema_contents(self):
        current = {
            "catalog_checksums": {"catalog"},
            "dataset_meta_checksums": {"metadata"},
            "annotation_checksums": {"annotation-a", "annotation-b"},
            "symmetry_pipeline_versions": set(),
            "point_set_checksums": {"points"},
            "sampling_pipeline_versions": {"sampler"},
            "sampling_parameter_checksums": {"parameters"},
            "schema_checksums": {"schema"},
            "schema_versions": {"2.0.0"},
        }
        checkpoint = {
            key: sorted(value)
            for key, value in current.items()
            if key != "annotation_checksums"
        }
        checkpoint["annotation_checksum_sha256"] = checksum_set_digest(
            current["annotation_checksums"]
        )
        validate_pose_resume_provenance(checkpoint, **current)

        with self.assertRaisesRegex(ValueError, "annotations"):
            validate_pose_resume_provenance(
                checkpoint,
                **{**current, "annotation_checksums": {"changed"}},
            )
        with self.assertRaisesRegex(ValueError, "schema files"):
            validate_pose_resume_provenance(
                checkpoint,
                **{**current, "schema_checksums": {"changed"}},
            )

    def test_resume_manifest_checksum_binds_split_assignment(self):
        checkpoint = {"manifest_sha256": "original"}
        validate_resume_manifest_checksum(checkpoint, "original")
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            validate_resume_manifest_checksum(checkpoint, "changed")
        validate_resume_manifest_checksum({}, "legacy-checkpoint-without-this-field")
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            validate_resume_manifest_checksum({"manifest_sha256": ""}, "new-manifest")

    def test_full_training_checkpoint_uses_explicit_trusted_load(self):
        modules = {
            name: SimpleNamespace(load_state_dict=Mock())
            for name in (
                "image_exemplar_fusion",
                "exemplar_detector",
                "exemplar_segmentation",
            )
        }
        detmodel = SimpleNamespace(**modules)
        checkpoint = {name: {} for name in modules}
        with patch(
            "finetune_image_exemplar_multi_gt.torch.load", return_value=checkpoint
        ) as load:
            loaded = load_finetune_checkpoint(
                Path("trusted-local.pth"), detmodel, None, torch.device("cpu")
            )
        self.assertIs(loaded, checkpoint)
        load.assert_called_once_with(
            "trusted-local.pth", map_location=torch.device("cpu"), weights_only=False
        )

    def test_reference_pair_accepts_padded_and_unpadded_view_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cad-a_stl_base_2.png").write_bytes(b"image")
            (root / "cad-a_stl_base_2_mask.png").write_bytes(b"mask")
            self.assertEqual(reference_view_id_candidates("02"), ["02", "2"])
            self.assertEqual(
                resolve_reference_pair(root, "cad-a", "02"),
                (root / "cad-a_stl_base_2.png", root / "cad-a_stl_base_2_mask.png"),
            )

    def test_pose_reference_metadata_is_bound_to_the_active_cad_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "cad-a_stl_base_02.png"
            mask_path = root / "cad-a_stl_base_02_mask.png"
            image_path.write_bytes(b"image")
            mask_path.write_bytes(b"mask")
            metadata_path = root / "cad-a_render_transform.json"
            metadata = {
                "object_id": "cad-a",
                "geometry": {
                    "orientation_mode": "canonical",
                    "catalog_driven": True,
                    "source_to_meters": 0.001,
                    "T_cad_from_source_meters": np.eye(4).tolist(),
                    "T_presentation_from_cad": np.eye(4).tolist(),
                    "canonical_dimensions_m": [0.1, 0.2, 0.3],
                },
                "views": [
                    {
                        "view_id": "02",
                        "image": image_path.name,
                        "mask": mask_path.name,
                        "R_refcam_cv_from_cad": np.eye(3).tolist(),
                    }
                ],
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            catalog = {
                "cad-a": SimpleNamespace(
                    source_to_meters=0.001,
                    T_cad_from_source_meters=np.eye(4),
                    base_dimensions_m=np.asarray([0.1, 0.2, 0.3]),
                )
            }

            report = validate_pose_reference_metadata(root, ["cad-a"], catalog, ["2"])
            self.assertEqual(report, {"cad_id_count": 1, "view_count": 1})

            metadata["geometry"]["orientation_mode"] = "largest-face-up"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 'canonical'"):
                validate_pose_reference_metadata(root, ["cad-a"], catalog, ["02"])

    def _rows_and_entry(self, root: Path):
        dataset_root = root / "pose"
        dataset_root.mkdir()
        (dataset_root / "pose_annotations_0002.json").write_text("{}")
        rows = [
            ManifestRow("pose", "pose", ".", "0001", "scene_1", "train"),
            ManifestRow("pose", "pose", ".", "0002", "scene_2", "train"),
        ]
        entry = {
            "object_id": "cad-a",
            "frame_id": "0001",
            "rgb_path": str(dataset_root / "rgb_0001.png"),
            "inst_path": str(dataset_root / "instance_segmentation_0001.png"),
            "color": (1, 2, 3, 255),
        }
        return rows, entry

    def test_valid_empty_pose_frame_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, entry = self._rows_and_entry(root)
            empty_sample = SimpleNamespace(frame=SimpleNamespace(instances=[]))
            with (
                patch(
                    "finetune_image_exemplar_multi_gt.collect_multi_object_samples",
                    return_value=({}, [entry]),
                ),
                patch(
                    "finetune_image_exemplar_multi_gt.load_perseve_pose_sample",
                    return_value=empty_sample,
                ),
            ):
                resolved = entries_from_manifest_rows(rows, root, object_level=False)
            self.assertEqual([item["frame_id"] for item in resolved], ["0001"])

    def test_unresolved_visible_pose_frame_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, entry = self._rows_and_entry(root)
            visible = SimpleNamespace(annotation_state="visible")
            visible_sample = SimpleNamespace(frame=SimpleNamespace(instances=[visible]))
            with (
                patch(
                    "finetune_image_exemplar_multi_gt.collect_multi_object_samples",
                    return_value=({}, [entry]),
                ),
                patch(
                    "finetune_image_exemplar_multi_gt.load_perseve_pose_sample",
                    return_value=visible_sample,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "0002"):
                    entries_from_manifest_rows(rows, root, object_level=False)


if __name__ == "__main__":
    unittest.main()
