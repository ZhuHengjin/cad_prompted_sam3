#!/usr/bin/env python3
"""Plot fine-tuning losses and metrics from captured training logs."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)\s+"
    r"step=(?P<step>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"avg_loss=(?P<avg_loss>[-+0-9.eE]+)\s+"
    r"avg_iou=(?P<avg_iou>[-+0-9.eE]+)\s+"
    r"run5_loss=(?P<run5_loss>[-+0-9.eE]+)\s+"
    r"run5_iou=(?P<run5_iou>[-+0-9.eE]+)"
)

EVAL_RE = re.compile(
    r"^\[eval\]\s+epoch=(?P<epoch>\d+)\s+"
    r"val_loss=(?P<val_loss>[-+0-9.eE]+)\s+"
    r"avg_iou=(?P<avg_iou>[-+0-9.eE]+)\s+"
    r"correct_rate=(?P<correct_rate>[-+0-9.eE]+)\s+"
    r"samples=(?P<samples>\d+)"
)


def parse_log(log_path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    train_rows: list[dict[str, float]] = []
    eval_rows: list[dict[str, float]] = []

    for line in log_path.read_text(errors="replace").splitlines():
        train_match = TRAIN_RE.match(line)
        if train_match:
            row = {
                "epoch": int(train_match.group("epoch")),
                "step": int(train_match.group("step")),
                "loss": float(train_match.group("loss")),
                "avg_loss": float(train_match.group("avg_loss")),
                "avg_iou": float(train_match.group("avg_iou")),
                "run5_loss": float(train_match.group("run5_loss")),
                "run5_iou": float(train_match.group("run5_iou")),
            }
            train_rows.append(row)
            continue

        eval_match = EVAL_RE.match(line)
        if eval_match:
            row = {
                "epoch": int(eval_match.group("epoch")),
                "val_loss": float(eval_match.group("val_loss")),
                "val_avg_iou": float(eval_match.group("avg_iou")),
                "correct_rate": float(eval_match.group("correct_rate")),
                "samples": int(eval_match.group("samples")),
            }
            eval_rows.append(row)

    return train_rows, eval_rows


def parse_logs(log_paths: list[Path]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    train_rows: list[dict[str, float]] = []
    eval_rows: list[dict[str, float]] = []

    for log_path in log_paths:
        log_train_rows, log_eval_rows = parse_log(log_path)
        train_rows.extend(log_train_rows)
        eval_rows.extend(log_eval_rows)

    return train_rows, eval_rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def last_train_row_by_epoch(train_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    rows_by_epoch: dict[int, dict[str, float]] = {}
    for row in train_rows:
        rows_by_epoch[int(row["epoch"])] = row
    return [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]


def plot_curves(train_rows: list[dict[str, float]], eval_rows: list[dict[str, float]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("SAM3 Exemplar Fine-tuning Metrics", fontsize=16)
    train_epoch_rows = last_train_row_by_epoch(train_rows)

    if train_epoch_rows:
        train_epochs = [row["epoch"] for row in train_epoch_rows]
        axes[0, 0].plot(
            train_epochs,
            [row["avg_loss"] for row in train_epoch_rows],
            color="#2563eb",
            marker="o",
            linewidth=2,
            label="train epoch avg loss",
        )
        axes[0, 1].plot(
            train_epochs,
            [row["avg_iou"] for row in train_epoch_rows],
            color="#16a34a",
            marker="o",
            linewidth=2,
            label="train epoch avg IoU",
        )

    if train_rows:
        steps = [row["step"] for row in train_rows]
        axes[1, 0].plot(
            steps,
            [row["loss"] for row in train_rows],
            color="#9ca3af",
            alpha=0.35,
            label="train batch loss",
        )
        axes[1, 0].plot(steps, [row["run5_loss"] for row in train_rows], color="#2563eb", label="train run5 loss")
        axes[1, 1].plot(steps, [row["run5_iou"] for row in train_rows], color="#16a34a", label="train run5 IoU")

    if eval_rows:
        epochs = [row["epoch"] for row in eval_rows]
        axes[0, 0].plot(
            epochs,
            [row["val_loss"] for row in eval_rows],
            color="#dc2626",
            marker="o",
            linewidth=2,
            label="val loss",
        )
        axes[0, 1].plot(
            epochs,
            [row["val_avg_iou"] for row in eval_rows],
            color="#ea580c",
            marker="o",
            linewidth=2,
            label="val avg IoU",
        )
        axes[0, 1].plot(
            epochs,
            [row["correct_rate"] for row in eval_rows],
            color="#7c3aed",
            marker="o",
            linewidth=2,
            label="val correct rate",
        )

    axes[0, 0].set_title("Train vs Validation Loss By Epoch")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].set_title("Train vs Validation Metrics By Epoch")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("rate")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].set_title("Training Loss Detail By Step")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("loss")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].set_title("Training IoU Detail By Step")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("IoU")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_paths", type=Path, nargs="+", help="Captured training log(s) to parse, in plot order.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for plot and CSV outputs.")
    args = parser.parse_args()

    log_paths = [path.expanduser().resolve() for path in args.log_paths]
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else log_paths[0].parent
    train_rows, eval_rows = parse_logs(log_paths)

    if not train_rows and not eval_rows:
        logs = ", ".join(str(path) for path in log_paths)
        raise SystemExit(f"No training or eval metric lines found in: {logs}")

    write_csv(out_dir / "training_metrics.csv", train_rows)
    write_csv(out_dir / "validation_metrics.csv", eval_rows)
    plot_curves(train_rows, eval_rows, out_dir / "training_validation_curves.png")

    print(f"Parsed {len(train_rows)} training rows and {len(eval_rows)} validation rows from {len(log_paths)} log(s).")
    print(f"Wrote {out_dir / 'training_validation_curves.png'}")


if __name__ == "__main__":
    main()
