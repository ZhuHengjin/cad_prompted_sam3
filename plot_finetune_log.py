#!/usr/bin/env python3
"""Plot detailed metrics produced by ``finetune_image_exemplar_multi_gt.py``.

The trainer writes append-only ``metrics.csv`` files with one row per batch,
an epoch summary, and validation (or standalone evaluation) results. For
example:

    python plot_finetune_log.py runs/finetune_exemplar/run_20260713_032020/metrics.csv

The dashboard includes total and component pose losses, 3D translation and
point-set errors, detector IoU, and epoch-level validation metrics. The plot is
written next to the input by default.
"""

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cad-prompted-sam3-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


REQUIRED_COLUMNS = {
    "phase", "epoch", "global_step", "batch_step", "loss", "avg_loss",
    "avg_iou", "correct_rate", "samples",
}
EVALUATION_PHASES = {"validation", "eval_validation", "eval_test"}
POSE_EVALUATION_PHASES = {
    "validation_pose",
    "validation_pose_calibrated",
    "eval_validation_pose",
    "eval_test_pose",
}
OPTIONAL_METRIC_COLUMNS = (
    "mask_loss",
    "bbox_loss",
    "objectness_loss",
    "pose_center_loss",
    "pose_depth_loss",
    "pose_rotation_loss",
    "pose_full_set_loss",
    "pose_quality_loss",
    "pose_aux_loss",
    "mean_surface_distance_norm",
    "p95_surface_distance_norm",
    "centroid_error_cm",
    "pose_success_rate",
    "rotation_error_deg",
    "translation_error_cm",
    "center_error_norm",
    "depth_error_m",
    "accuracy_5deg_5cm",
    "accuracy_10deg_10cm",
    "brier_score",
    "expected_calibration_error",
    "pose_score_temperature",
    "pose_match_iou_threshold",
    "pose_assignment_coverage",
    "pose_match_coverage",
    "pose_match_acceptance_rate",
    "pose_end_to_end_success_rate",
)
MetricRow = dict[str, float | int | str | None]


def is_pose_evaluation_phase(phase: str) -> bool:
    return phase in POSE_EVALUATION_PHASES or any(
        phase.startswith(f"{base}_iou_") for base in POSE_EVALUATION_PHASES
    )


def optional_float(value: str | None) -> float | None:
    """Return a float for a populated CSV value, otherwise ``None``."""
    if value in (None, ""):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def parse_metrics_csv(
    csv_path: Path,
) -> tuple[list[MetricRow], list[MetricRow], list[MetricRow]]:
    """Read batch, epoch-summary, and evaluation rows from a trainer CSV."""
    batch_rows: list[MetricRow] = []
    epoch_rows: list[MetricRow] = []
    evaluation_rows: list[MetricRow] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is not a fine-tuning metrics CSV; missing columns: {missing}")

        for line_number, raw_row in enumerate(reader, start=2):
            try:
                row = {
                    "epoch": int(raw_row["epoch"]),
                    "global_step": int(raw_row["global_step"]),
                    "batch_step": int(raw_row["batch_step"]),
                    "loss": optional_float(raw_row["loss"]),
                    "avg_loss": optional_float(raw_row["avg_loss"]),
                    "avg_iou": optional_float(raw_row["avg_iou"]),
                    "correct_rate": optional_float(raw_row["correct_rate"]),
                    "samples": int(raw_row["samples"] or 0),
                }
                row.update(
                    {
                        column: optional_float(raw_row.get(column))
                        for column in OPTIONAL_METRIC_COLUMNS
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid metric row in {csv_path}:{line_number}: {error}") from error

            phase = raw_row["phase"]
            if phase == "train_batch":
                batch_rows.append(row)
            elif phase == "train_epoch":
                epoch_rows.append(row)
            elif phase in EVALUATION_PHASES or is_pose_evaluation_phase(phase):
                row["phase"] = phase
                evaluation_rows.append(row)

    return batch_rows, epoch_rows, evaluation_rows


def parse_metrics_csvs(
    csv_paths: list[Path],
) -> tuple[list[MetricRow], list[MetricRow], list[MetricRow]]:
    """Combine metrics CSVs in the order supplied on the command line."""
    batches: list[MetricRow] = []
    epochs: list[MetricRow] = []
    evaluations: list[MetricRow] = []
    for csv_path in csv_paths:
        batch_rows, epoch_rows, evaluation_rows = parse_metrics_csv(csv_path)
        batches.extend(batch_rows)
        epochs.extend(epoch_rows)
        evaluations.extend(evaluation_rows)
    return batches, epochs, evaluations


def values(
    rows: list[MetricRow], key: str, *, x_key: str = "epoch"
) -> tuple[list[int], list[float]]:
    """Return epochs and non-empty numeric values for a metric."""
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row[key]
        if value is not None:
            xs.append(int(row[x_key]))
            ys.append(float(value))
    return xs, ys


def rolling_average(xs: list[int], ys: list[float], window: int) -> tuple[list[int], list[float]]:
    """Return a trailing moving average without changing the x coordinates."""

    if window <= 1:
        return xs, ys
    smoothed = []
    running_sum = 0.0
    for index, value in enumerate(ys):
        running_sum += value
        if index >= window:
            running_sum -= ys[index - window]
        smoothed.append(running_sum / min(index + 1, window))
    return xs, smoothed


def plot_batch_metric(
    axis,
    rows: list[MetricRow],
    key: str,
    label: str,
    color: str,
    smooth_window: int,
    *,
    show_raw: bool = False,
) -> bool:
    """Plot one finite batch metric and return whether any values existed."""

    xs, ys = values(rows, key, x_key="global_step")
    if not ys:
        return False
    if show_raw:
        axis.plot(xs, ys, color=color, alpha=0.16, linewidth=0.7)
    smooth_xs, smooth_ys = rolling_average(xs, ys, smooth_window)
    axis.plot(smooth_xs, smooth_ys, color=color, linewidth=1.7, label=label)
    return True


def finish_axis(axis, title: str, ylabel: str, xlabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if axis.lines:
        axis.legend(fontsize=8)
    else:
        axis.text(
            0.5,
            0.5,
            "No metrics recorded yet",
            ha="center",
            va="center",
            transform=axis.transAxes,
            color="#6b7280",
        )


def zoom_rate_axis(axis, plotted_values: list[float], *, minimum_span: float = 0.05) -> None:
    """Tightly frame finite rate values while retaining a little visual margin."""

    finite = [value for value in plotted_values if math.isfinite(value)]
    if not finite:
        return
    low, high = min(finite), max(finite)
    span = max(high - low, minimum_span)
    margin = span * 0.12
    midpoint = (low + high) / 2.0
    axis.set_ylim(max(0.0, midpoint - span / 2.0 - margin), min(1.0, midpoint + span / 2.0 + margin))


def plot_curves(
    batch_rows: list[MetricRow],
    epoch_rows: list[MetricRow],
    evaluation_rows: list[MetricRow],
    output_path: Path,
    *,
    smooth_window: int = 25,
) -> None:
    """Write a detailed loss, detection, and point-set pose dashboard."""

    detection_evaluations = [
        row for row in evaluation_rows if str(row.get("phase")) in EVALUATION_PHASES
    ]
    pose_evaluations = [
        row for row in evaluation_rows if str(row.get("phase")) in POSE_EVALUATION_PHASES
    ]
    conditional_pose_evaluations = [
        row
        for row in evaluation_rows
        if is_pose_evaluation_phase(str(row.get("phase")))
        and str(row.get("phase")) not in POSE_EVALUATION_PHASES
    ]
    fig = plt.figure(figsize=(16, 20), constrained_layout=True)
    grid = fig.add_gridspec(4, 2)
    loss_axis = fig.add_subplot(grid[0, 0])
    center_axis = fig.add_subplot(grid[0, 1])
    pose_components_axis = fig.add_subplot(grid[1, 0])
    rotation_loss_axis = fig.add_subplot(grid[1, 1])
    translation_axis = fig.add_subplot(grid[2, 0])
    placement_axis = fig.add_subplot(grid[2, 1])
    detection_axis = fig.add_subplot(grid[3, 0])
    pose_grid = grid[3, 1].subgridspec(2, 1, height_ratios=(1.0, 2.0), hspace=0.06)
    pose_coverage_axis = fig.add_subplot(pose_grid[0])
    pose_quality_axis = fig.add_subplot(pose_grid[1], sharex=pose_coverage_axis)
    fig.suptitle("SAM3 Point-Set Pose Fine-tuning Metrics", fontsize=16)

    # Epoch totals provide the directly comparable train/validation overview.
    train_epochs, train_loss = values(epoch_rows, "avg_loss")
    if train_loss:
        loss_axis.plot(train_epochs, train_loss, "o-", color="#2563eb", linewidth=2, label="training loss")
    eval_epochs, eval_loss = values(detection_evaluations, "loss")
    if eval_loss:
        loss_axis.plot(eval_epochs, eval_loss, "o-", color="#dc2626", linewidth=2, label="validation loss")
    finish_axis(loss_axis, "Total Training vs Validation Loss", "loss", "epoch")

    # Center loss is much smaller than the other pose terms and needs its own scale.
    plot_batch_metric(
        center_axis,
        batch_rows,
        "pose_center_loss",
        "center loss",
        "#0891b2",
        smooth_window,
        show_raw=True,
    )
    finish_axis(
        center_axis,
        f"Pose Center Loss (moving average={smooth_window})",
        "loss",
        "global step",
    )

    # The remaining trainable/logged pose components have broadly comparable scales.
    for key, label, color in (
        ("pose_depth_loss", "depth loss", "#2563eb"),
        ("pose_full_set_loss", "full-set loss (logged)", "#9333ea"),
        ("pose_quality_loss", "quality loss", "#ea580c"),
        ("pose_aux_loss", "auxiliary pose loss", "#16a34a"),
    ):
        plot_batch_metric(pose_components_axis, batch_rows, key, label, color, smooth_window)
    finish_axis(pose_components_axis, "Pose Loss Components (Other)", "loss", "global step")
    pose_components_axis.set_ylim(0.0, 1.0)
    pose_components_axis.yaxis.set_major_locator(MultipleLocator(0.1))
    pose_components_axis.yaxis.set_minor_locator(MultipleLocator(0.05))
    pose_components_axis.grid(which="minor", axis="y", alpha=0.12)

    # Rotation loss gets a dedicated, automatically scaled axis so small trends remain visible.
    plot_batch_metric(
        rotation_loss_axis,
        batch_rows,
        "pose_rotation_loss",
        "point-set rotation loss",
        "#dc2626",
        smooth_window,
        show_raw=False,
    )
    finish_axis(
        rotation_loss_axis,
        f"Point-Set Rotation Loss (moving average={smooth_window})",
        "loss",
        "global step",
    )
    rotation_loss_axis.margins(y=0.12)

    # Translation is reconstructed from center and depth, so it is an error metric rather than a direct loss.
    for key, label, color in (
        ("centroid_error_cm", "surface-centroid error", "#7c3aed"),
        ("translation_error_cm", "CAD-origin translation error", "#dc2626"),
    ):
        plot_batch_metric(translation_axis, batch_rows, key, label, color, smooth_window, show_raw=True)
    finish_axis(translation_axis, "3D Translation Metrics", "error (cm)", "global step")

    plot_batch_metric(
        placement_axis,
        batch_rows,
        "mean_surface_distance_norm",
        "train mean surface distance",
        "#2563eb",
        smooth_window,
        show_raw=True,
    )
    for key, label, color in (
        ("mean_surface_distance_norm", "validation mean", "#dc2626"),
        ("p95_surface_distance_norm", "validation p95", "#16a34a"),
    ):
        xs, ys = values(pose_evaluations, key, x_key="global_step")
        if ys:
            placement_axis.plot(xs, ys, "o-", color=color, linewidth=1.8, label=label)
    for key, label, color in (
        ("mean_surface_distance_norm", "validation mean (IoU-filtered)", "#9333ea"),
        ("p95_surface_distance_norm", "validation p95 (IoU-filtered)", "#ea580c"),
    ):
        xs, ys = values(conditional_pose_evaluations, key, x_key="global_step")
        if ys:
            placement_axis.plot(xs, ys, "o--", color=color, linewidth=1.5, label=label)
    finish_axis(placement_axis, "Point-Set Placement Error", "normalized surface distance", "global step")

    detection_values: list[float] = []
    for rows, key, label, color in (
        (epoch_rows, "avg_iou", "train IoU", "#16a34a"),
        (detection_evaluations, "avg_iou", "validation IoU", "#ea580c"),
        (detection_evaluations, "correct_rate", "validation correct rate", "#7c3aed"),
    ):
        xs, ys = values(rows, key)
        if ys:
            detection_axis.plot(xs, ys, "o-", color=color, linewidth=1.8, label=label)
            detection_values.extend(ys)
    finish_axis(detection_axis, "Epoch Detection Quality", "rate", "epoch")
    zoom_rate_axis(detection_axis, detection_values)

    pose_quality_values: list[float] = []
    for key, label, color in (
        ("pose_success_rate", "pose success", "#16a34a"),
        ("accuracy_5deg_5cm", "5deg / 5cm", "#2563eb"),
        ("accuracy_10deg_10cm", "10deg / 10cm", "#9333ea"),
        ("expected_calibration_error", "calibration error", "#ea580c"),
        ("brier_score", "Brier score", "#dc2626"),
    ):
        xs, ys = values(pose_evaluations, key)
        if ys:
            pose_quality_axis.plot(xs, ys, "o-", color=color, linewidth=1.8, label=label)
            pose_quality_values.extend(ys)
    for key, label, color in (
        ("pose_success_rate", "pose success (IoU-filtered)", "#14532d"),
        ("pose_end_to_end_success_rate", "end-to-end pose success", "#0f766e"),
    ):
        xs, ys = values(conditional_pose_evaluations, key)
        if ys:
            pose_quality_axis.plot(xs, ys, "o--", color=color, linewidth=1.5, label=label)
            pose_quality_values.extend(ys)
    coverage_xs, coverage_ys = values(conditional_pose_evaluations, "pose_match_coverage")
    if coverage_ys:
        pose_coverage_axis.plot(
            coverage_xs,
            coverage_ys,
            "o--",
            color="#475569",
            linewidth=1.7,
            label="pose match coverage",
        )

    pose_coverage_axis.set_title("Validation Pose Success and Calibration")
    pose_coverage_axis.set_ylabel("coverage")
    pose_quality_axis.set_ylabel("rate / score")
    pose_quality_axis.set_xlabel("epoch")
    pose_coverage_axis.grid(alpha=0.25)
    pose_quality_axis.grid(alpha=0.25)
    zoom_rate_axis(pose_coverage_axis, coverage_ys)
    zoom_rate_axis(pose_quality_axis, pose_quality_values)
    pose_coverage_axis.spines["bottom"].set_visible(False)
    pose_quality_axis.spines["top"].set_visible(False)
    pose_coverage_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    break_size = 0.008
    break_kwargs = dict(color="k", clip_on=False, linewidth=1.0)
    pose_coverage_axis.plot((-break_size, +break_size), (-break_size, +break_size), transform=pose_coverage_axis.transAxes, **break_kwargs)
    pose_coverage_axis.plot((1 - break_size, 1 + break_size), (-break_size, +break_size), transform=pose_coverage_axis.transAxes, **break_kwargs)
    pose_quality_axis.plot((-break_size, +break_size), (1 - break_size, 1 + break_size), transform=pose_quality_axis.transAxes, **break_kwargs)
    pose_quality_axis.plot((1 - break_size, 1 + break_size), (1 - break_size, 1 + break_size), transform=pose_quality_axis.transAxes, **break_kwargs)
    handles_upper, labels_upper = pose_coverage_axis.get_legend_handles_labels()
    handles_lower, labels_lower = pose_quality_axis.get_legend_handles_labels()
    if handles_upper or handles_lower:
        pose_quality_axis.legend(handles_lower + handles_upper, labels_lower + labels_upper, fontsize=8, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_paths", type=Path, nargs="+", help="Fine-tuning metrics.csv file(s), in plot order.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: training_validation_curves.png beside the first CSV).")
    parser.add_argument(
        "--smooth",
        type=int,
        default=25,
        help="Trailing window for per-batch curves (default: 25; use 1 for no smoothing).",
    )
    args = parser.parse_args()

    if args.smooth <= 0:
        parser.error("--smooth must be a positive integer")

    metrics_paths = [path.expanduser().resolve() for path in args.metrics_paths]
    output_path = args.out.expanduser().resolve() if args.out else metrics_paths[0].parent / "training_validation_curves.png"
    batch_rows, epoch_rows, evaluation_rows = parse_metrics_csvs(metrics_paths)
    if not (batch_rows or epoch_rows or evaluation_rows):
        paths = ", ".join(str(path) for path in metrics_paths)
        raise SystemExit(f"No recognized metrics rows found in: {paths}")

    plot_curves(
        batch_rows,
        epoch_rows,
        evaluation_rows,
        output_path,
        smooth_window=args.smooth,
    )
    print(
        f"Parsed {len(batch_rows)} batch, {len(epoch_rows)} epoch-summary, and "
        f"{len(evaluation_rows)} evaluation rows from {len(metrics_paths)} metrics CSV file(s)."
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
