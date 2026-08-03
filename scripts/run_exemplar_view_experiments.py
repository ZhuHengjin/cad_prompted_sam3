#!/usr/bin/env python3
"""Launch a matched reference-view conditioning experiment matrix.

Arguments after ``--`` are passed unchanged to the canonical trainer.  This
launcher owns only ``--exemplar_view_mode`` and ``--output_dir`` so every arm
uses the same manifest, initialization checkpoint, seed, and hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODES = ("none", "camera", "shuffled_camera", "zero_camera", "view_id")
REQUIRED_TRAINER_FLAGS = (
    "--enable_pose",
    "--init_path",
    "--dataset_manifest",
    "--data_root",
    "--reference_dir",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer",
        type=Path,
        default=repo_root / "finetune_image_exemplar_multi_gt.py",
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print and record commands without starting training.",
    )
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Canonical trainer arguments following --.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    if not args.trainer.is_file():
        raise FileNotFoundError(args.trainer)
    for controlled in ("--exemplar_view_mode", "--output_dir"):
        if controlled in trainer_args:
            raise ValueError(f"{controlled} is controlled by this experiment launcher")
    missing = [flag for flag in REQUIRED_TRAINER_FLAGS if flag not in trainer_args]
    if missing:
        raise ValueError(f"Matched pose experiments require trainer flags: {missing}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    commands = []
    for mode in args.modes:
        output_dir = (args.output_root / mode).resolve()
        command = [
            sys.executable,
            str(args.trainer.resolve()),
            *trainer_args,
            "--exemplar_view_mode",
            mode,
            "--output_dir",
            str(output_dir),
        ]
        commands.append({"mode": mode, "output_dir": str(output_dir), "command": command})

    matrix_record = {
        "schema": "cad_prompted_sam3.exemplar_view_experiment_matrix",
        "schema_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "modes": list(args.modes),
        "trainer": str(args.trainer.resolve()),
        "commands": commands,
    }
    record_path = args.output_root / "experiment_matrix.json"
    record_path.write_text(json.dumps(matrix_record, indent=2) + "\n", encoding="utf-8")

    for item in commands:
        print(f"[{item['mode']}] {shlex.join(item['command'])}", flush=True)
        if not args.dry_run:
            subprocess.run(item["command"], check=True)
    print(f"Experiment matrix provenance: {record_path}")


if __name__ == "__main__":
    main()
