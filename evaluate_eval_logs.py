#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import List, Optional


DEFAULT_LOG_PATH = (
    "/home/zhenrant/rendering_prompted_muggled_sam/eval_logs/"
    "eval_0321_multi_k12_bbox_ns.log"
)

CHECKPOINT_HEADER_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+)$")
CHECKPOINT_EPOCH_RE = re.compile(r"finetune_epoch_(\d+)\.pth$")
PQ_RE = re.compile(
    r"^PQ@score>=([0-9.]+)=([0-9.]+)\s+tp=(\d+)\s+fp=(\d+)\s+fn=(\d+)$"
)
OVERALL_RE = re.compile(
    r"^overall_avg_iou=([0-9.]+)\s+correct_rate=([0-9.]+)\s+samples=(\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank checkpoints from an eval log by best PQ score."
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default=DEFAULT_LOG_PATH,
        help="Path to an eval log produced by eval_image_exemplar_all_checkpoints.py",
    )
    return parser.parse_args()


def checkpoint_label_from_name(checkpoint_name: str) -> str:
    match = CHECKPOINT_EPOCH_RE.search(checkpoint_name)
    if match:
        return f"epoch{int(match.group(1))}"
    if checkpoint_name == "finetune.pth":
        return "latest"
    return checkpoint_name


def maybe_checkpoint_epoch(checkpoint_name: str) -> Optional[int]:
    match = CHECKPOINT_EPOCH_RE.search(checkpoint_name)
    if match:
        return int(match.group(1))
    return None


def finalize_result(
    results: List[dict],
    checkpoint_name: Optional[str],
    avg_iou: Optional[float],
    correct_rate: Optional[float],
    current_pq: List[tuple],
) -> None:
    if checkpoint_name is None or avg_iou is None or correct_rate is None:
        return

    best_thr = None
    best_pq = None
    if current_pq:
        best_thr, best_pq = max(current_pq, key=lambda item: item[1])

    results.append(
        {
            "checkpoint": checkpoint_name,
            "epoch": maybe_checkpoint_epoch(checkpoint_name),
            "label": checkpoint_label_from_name(checkpoint_name),
            "best_pq": best_pq,
            "threshold": best_thr,
            "avg_iou": avg_iou,
            "correct_rate": correct_rate,
        }
    )


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_path).expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(log_path)

    results: List[dict] = []
    current_checkpoint: Optional[str] = None
    current_pq: List[tuple] = []
    current_avg_iou: Optional[float] = None
    current_correct_rate: Optional[float] = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            header_match = CHECKPOINT_HEADER_RE.match(line)
            if header_match:
                finalize_result(
                    results,
                    current_checkpoint,
                    current_avg_iou,
                    current_correct_rate,
                    current_pq,
                )
                current_checkpoint = header_match.group(3)
                current_pq = []
                current_avg_iou = None
                current_correct_rate = None
                continue

            overall_match = OVERALL_RE.match(line)
            if overall_match:
                current_avg_iou = float(overall_match.group(1))
                current_correct_rate = float(overall_match.group(2))
                continue

            pq_match = PQ_RE.match(line)
            if pq_match:
                threshold = float(pq_match.group(1))
                pq_score = float(pq_match.group(2))
                current_pq.append((threshold, pq_score))
                continue

    finalize_result(
        results,
        current_checkpoint,
        current_avg_iou,
        current_correct_rate,
        current_pq,
    )

    results.sort(
        key=lambda item: item["best_pq"] if item["best_pq"] is not None else -1,
        reverse=True,
    )

    print("evaluating log:", log_path)
    print()
    print("Ranked checkpoints by best PQ score")
    print("--------------------------------------------------------------------")
    print(f"{'rank':>4} {'checkpoint':>12} {'pq':>8} {'thr':>6} {'corr_rate':>10} {'avg_iou':>10}")
    print("--------------------------------------------------------------------")

    for index, result in enumerate(results, 1):
        pq = f"{result['best_pq']:.4f}" if result["best_pq"] is not None else "N/A"
        thr = f"{result['threshold']:.2f}" if result["threshold"] is not None else "N/A"
        print(
            f"{index:>4} {result['label']:>12} {pq:>8} {thr:>6} "
            f"{result['correct_rate']:>10.4f} {result['avg_iou']:>10.4f}"
        )


if __name__ == "__main__":
    main()
