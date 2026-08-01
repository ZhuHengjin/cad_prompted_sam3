"""Regression tests for detailed fine-tuning metric plots."""

import csv
import tempfile
import unittest
from pathlib import Path

from plot_finetune_log import (
    OPTIONAL_METRIC_COLUMNS,
    REQUIRED_COLUMNS,
    parse_metrics_csv,
    plot_curves,
)


class FineTunePlotTests(unittest.TestCase):
    def test_pose_metrics_are_parsed_and_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "metrics.csv"
            fieldnames = sorted(REQUIRED_COLUMNS | set(OPTIONAL_METRIC_COLUMNS))
            rows = [
                {
                    "phase": "train_batch",
                    "epoch": 1,
                    "global_step": 1,
                    "batch_step": 1,
                    "loss": 4.5,
                    "avg_loss": 4.5,
                    "avg_iou": 0.1,
                    "samples": 4,
                    "pose_center_loss": 0.001,
                    "pose_depth_loss": 0.4,
                    "pose_rotation_loss": 0.08,
                    "pose_full_set_loss": 0.7,
                    "pose_quality_loss": 0.1,
                    "mean_surface_distance_norm": 0.6,
                    "centroid_error_cm": 12.0,
                    "translation_error_cm": 13.0,
                },
                {
                    "phase": "train_epoch",
                    "epoch": 1,
                    "global_step": 1,
                    "batch_step": 1,
                    "avg_loss": 4.5,
                    "avg_iou": 0.1,
                    "samples": 1,
                },
                {
                    "phase": "validation",
                    "epoch": 1,
                    "global_step": 1,
                    "batch_step": 1,
                    "loss": 4.7,
                    "avg_iou": 0.08,
                    "correct_rate": 0.03,
                    "samples": 10,
                },
                {
                    "phase": "validation_pose_calibrated",
                    "epoch": 1,
                    "global_step": 1,
                    "batch_step": 1,
                    "samples": 10,
                    "mean_surface_distance_norm": 0.7,
                    "p95_surface_distance_norm": 0.9,
                    "centroid_error_cm": 14.0,
                    "translation_error_cm": 14.5,
                    "pose_success_rate": 0.02,
                    "rotation_error_deg": "nan",
                    "expected_calibration_error": 0.01,
                },
                {
                    "phase": "validation_pose_calibrated_iou_070",
                    "epoch": 1,
                    "global_step": 1,
                    "batch_step": 1,
                    "samples": 7,
                    "mean_surface_distance_norm": 0.5,
                    "p95_surface_distance_norm": 0.7,
                    "pose_success_rate": 0.04,
                    "pose_match_coverage": 0.7,
                    "pose_end_to_end_success_rate": 0.028,
                },
            ]
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            batches, epochs, evaluations = parse_metrics_csv(csv_path)
            self.assertEqual(len(batches), 1)
            self.assertEqual(len(epochs), 1)
            self.assertEqual(len(evaluations), 3)
            self.assertEqual(batches[0]["pose_rotation_loss"], 0.08)
            self.assertIsNone(evaluations[1]["rotation_error_deg"])

            output = root / "curves.png"
            plot_curves(batches, epochs, evaluations, output, smooth_window=2)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
