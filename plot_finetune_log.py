#!/usr/bin/env python3
"""Plot metrics produced by ``finetune_image_exemplar_multi_gt.py``.

The trainer writes append-only ``metrics.csv`` files with one row per batch,
an epoch summary, and validation (or standalone evaluation) results. For
example:

    python plot_finetune_log.py finetune_exemplar/run_20260713_032020/metrics.csv

The plot is written next to the input by default.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "phase", "epoch", "global_step", "batch_step", "loss", "avg_loss",
    "avg_iou", "correct_rate", "samples",
}
EVALUATION_PHASES = {"validation", "eval_validation", "eval_test"}


def optional_float(value: str | None) -> float | None:
    """Return a float for a populated CSV value, otherwise ``None``."""
    return float(value) if value not in (None, "") else None


def parse_metrics_csv(
    csv_path: Path,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], list[dict[str, float | int]]]:
    """Read batch, epoch-summary, and evaluation rows from a trainer CSV."""
    batch_rows: list[dict[str, float | int]] = []
    epoch_rows: list[dict[str, float | int]] = []
    evaluation_rows: list[dict[str, float | int]] = []

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
                    "samples": int(raw_row["samples"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid metric row in {csv_path}:{line_number}: {error}") from error

            phase = raw_row["phase"]
            if phase == "train_batch":
                batch_rows.append(row)
            elif phase == "train_epoch":
                epoch_rows.append(row)
            elif phase in EVALUATION_PHASES:
                row["phase"] = phase
                evaluation_rows.append(row)

    return batch_rows, epoch_rows, evaluation_rows


def parse_metrics_csvs(
    csv_paths: list[Path],
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], list[dict[str, float | int]]]:
    """Combine metrics CSVs in the order supplied on the command line."""
    batches: list[dict[str, float | int]] = []
    epochs: list[dict[str, float | int]] = []
    evaluations: list[dict[str, float | int]] = []
    for csv_path in csv_paths:
        batch_rows, epoch_rows, evaluation_rows = parse_metrics_csv(csv_path)
        batches.extend(batch_rows)
        epochs.extend(epoch_rows)
        evaluations.extend(evaluation_rows)
    return batches, epochs, evaluations


def values(rows: list[dict[str, float | int]], key: str) -> tuple[list[int], list[float]]:
    """Return epochs and non-empty numeric values for a metric."""
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row[key]
        if value is not None:
            xs.append(int(row["epoch"]))
            ys.append(float(value))
    return xs, ys


def plot_curves(
    batch_rows: list[dict[str, float | int]],
    epoch_rows: list[dict[str, float | int]],
    evaluation_rows: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Write a four-panel overview of training and evaluation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("SAM3 Exemplar Fine-tuning Metrics", fontsize=16)

    train_epochs, train_loss = values(epoch_rows, "avg_loss")
    if train_loss:
        axes[0, 0].plot(train_epochs, train_loss, "o-", color="#2563eb", linewidth=2, label="train avg loss")
    eval_epochs, eval_loss = values(evaluation_rows, "loss")
    if eval_loss:
        axes[0, 0].plot(eval_epochs, eval_loss, "o-", color="#dc2626", linewidth=2, label="evaluation loss")

    train_epochs, train_iou = values(epoch_rows, "avg_iou")
    if train_iou:
        axes[0, 1].plot(train_epochs, train_iou, "o-", color="#16a34a", linewidth=2, label="train avg IoU")
    eval_epochs, eval_iou = values(evaluation_rows, "avg_iou")
    if eval_iou:
        axes[0, 1].plot(eval_epochs, eval_iou, "o-", color="#ea580c", linewidth=2, label="evaluation avg IoU")
    correct_epochs, correct_rates = values(evaluation_rows, "correct_rate")
    if correct_rates:
        axes[0, 1].plot(correct_epochs, correct_rates, "o-", color="#7c3aed", linewidth=2, label="evaluation correct rate")

    if batch_rows:
        steps = [int(row["global_step"]) for row in batch_rows]
        batch_loss = [float(row["loss"]) for row in batch_rows]
        running_loss = [float(row["avg_loss"]) for row in batch_rows]
        batch_iou = [float(row["avg_iou"]) for row in batch_rows]
        axes[1, 0].plot(steps, batch_loss, color="#9ca3af", alpha=0.25, label="batch loss")
        axes[1, 0].plot(steps, running_loss, color="#2563eb", linewidth=1.5, label="running avg loss")
        axes[1, 1].plot(steps, batch_iou, color="#16a34a", linewidth=1.5, label="batch avg IoU")

    for axis, title, ylabel, xlabel in (
        (axes[0, 0], "Training vs Evaluation Loss by Epoch", "loss", "epoch"),
        (axes[0, 1], "Training vs Evaluation Quality by Epoch", "rate", "epoch"),
        (axes[1, 0], "Training Loss by Global Step", "loss", "global step"),
        (axes[1, 1], "Training IoU by Global Step", "IoU", "global step"),
    ):
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()

    axes[0, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylim(0.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_paths", type=Path, nargs="+", help="Fine-tuning metrics.csv file(s), in plot order.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: training_validation_curves.png beside the first CSV).")
    args = parser.parse_args()

    metrics_paths = [path.expanduser().resolve() for path in args.metrics_paths]
    output_path = args.out.expanduser().resolve() if args.out else metrics_paths[0].parent / "training_validation_curves.png"
    batch_rows, epoch_rows, evaluation_rows = parse_metrics_csvs(metrics_paths)
    if not (batch_rows or epoch_rows or evaluation_rows):
        paths = ", ".join(str(path) for path in metrics_paths)
        raise SystemExit(f"No recognized metrics rows found in: {paths}")

    plot_curves(batch_rows, epoch_rows, evaluation_rows, output_path)
    print(
        f"Parsed {len(batch_rows)} batch, {len(epoch_rows)} epoch-summary, and "
        f"{len(evaluation_rows)} evaluation rows from {len(metrics_paths)} metrics CSV file(s)."
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
