"""Regression tests for manifest generation, validation, and domain sampling.

The suite uses temporary synthetic dataset trees to exercise file resolution,
dataset-qualified group isolation, deterministic ratios, wrist scene
provenance, multi-camera grouping, immutable identities, validation failures,
and reproducible equal-domain sampling without requiring OpenCV or PyTorch.
"""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from build_dataset_manifest import build_rows, load_wrist_groups
from dataset_manifest import (
    ManifestRow,
    assign_group_splits,
    balanced_epoch_entries,
    load_manifest,
    validate_manifest_rows,
    write_manifest,
)


def make_sample(root: Path, dataset: str, camera: str, frame_id: str) -> None:
    camera_root = root / dataset / camera
    camera_root.mkdir(parents=True, exist_ok=True)
    (camera_root / f"rgb_{frame_id}.png").write_bytes(b"rgb")
    (camera_root / f"instance_segmentation_{frame_id}.png").write_bytes(b"seg")
    (camera_root / f"instance_segmentation_mapping_{frame_id}.json").write_text("{}")


class DatasetManifestTests(unittest.TestCase):
    def test_split_assignment_is_deterministic_and_dataset_qualified(self):
        groups = [f"scene_{index:04d}" for index in range(100)]
        first = assign_group_splits("dataset_a", groups, seed=42)
        self.assertEqual(first, assign_group_splits("dataset_a", reversed(groups), seed=42))
        self.assertNotEqual(first, assign_group_splits("dataset_b", groups, seed=42))
        self.assertEqual(Counter(first.values()), {"train": 80, "validation": 10, "test": 10})

    def test_equal_domain_sampling_is_deterministic(self):
        pools = {"large": [{"id": index} for index in range(20)], "small": [{"id": 99}]}
        first = balanced_epoch_entries(pools, epoch_size=10, seed=42, epoch=3)
        second = balanced_epoch_entries(pools, epoch_size=10, seed=42, epoch=3)
        self.assertEqual(first, second)
        self.assertEqual(sum(entry["id"] == 99 for entry in first), 5)
        self.assertNotEqual(first, balanced_epoch_entries(pools, epoch_size=10, seed=42, epoch=4))

    def test_manifest_round_trip_and_path_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_sample(root, "dataset", "Camera", "0001")
            row = ManifestRow("id", "dataset", "Camera", "0001", "scene_1", "train")
            path = root / "manifest.csv"
            write_manifest(path, [row])
            loaded, summary = load_manifest(path, root)
            self.assertEqual(loaded, [row])
            self.assertEqual(summary["rows"], 1)

    def test_validation_rejects_duplicate_leakage_invalid_paths_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_sample(root, "dataset", "Camera", "0001")
            valid = ManifestRow("id", "dataset", "Camera", "0001", "scene", "train")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_manifest_rows([valid, valid], root)
            leaked = ManifestRow("id", "dataset", "Other", "0002", "scene", "test")
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                validate_manifest_rows([valid, leaked], root, validate_files=False)
            invalid_split = ManifestRow("id", "dataset", "Camera", "0001", "scene", "dev")
            with self.assertRaisesRegex(ValueError, "Invalid split"):
                validate_manifest_rows([invalid_split], root, validate_files=False)
            absolute = ManifestRow("id", "/dataset", "Camera", "0001", "scene", "train")
            with self.assertRaisesRegex(ValueError, "relative"):
                validate_manifest_rows([absolute], root, validate_files=False)
            inconsistent = ManifestRow("id", "other", "Camera", "0002", "other", "train")
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                validate_manifest_rows([valid, inconsistent], root, validate_files=False)
            missing = ManifestRow("id", "dataset", "Camera", "9999", "missing", "train")
            with self.assertRaises(FileNotFoundError):
                validate_manifest_rows([missing], root)

    def test_wrist_scene_mapping_and_side_camera_grouping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrist = "wrist"
            side = "side"
            make_sample(root, wrist, "Wrist_Camera", "0000")
            make_sample(root, wrist, "Wrist_Camera", "0001")
            metadata = {
                "scenes": [
                    {
                        "scene_id": "scene_0000",
                        "captures": [
                            {"accepted": True, "frame_id": 0},
                            {"accepted": True, "frame_id": 1},
                        ],
                    }
                ]
            }
            (root / wrist / "metadata.json").write_text(json.dumps(metadata))
            make_sample(root, side, "Side_Camera_0", "0007")
            make_sample(root, side, "Side_Camera_3", "0007")

            groups = load_wrist_groups(root / wrist / "metadata.json")
            self.assertEqual(groups, {"0000": "scene_0000", "0001": "scene_0000"})
            rows = build_rows(root, wrist, side, (0.8, 0.1, 0.1), 42)
            wrist_rows = [row for row in rows if row.dataset_id == "wrist_type2"]
            side_rows = [row for row in rows if row.dataset_id == "yaw20_side"]
            self.assertEqual({row.group_id for row in wrist_rows}, {"scene_0000"})
            self.assertEqual({row.group_id for row in side_rows}, {"0007"})
            self.assertEqual(len({row.split for row in side_rows}), 1)

    def test_wrist_builder_rejects_missing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_sample(root, "wrist", "Wrist_Camera", "0000")
            (root / "wrist" / "metadata.json").write_text(json.dumps({"scenes": []}))
            make_sample(root, "side", "Side_Camera_0", "0000")
            with self.assertRaises(ValueError):
                build_rows(root, "wrist", "side", (0.8, 0.1, 0.1), 42)


if __name__ == "__main__":
    unittest.main()
