#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


CHECKPOINT_PATTERN = re.compile(r"finetune_epoch_(\d+)\.pth$")


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run eval_image_exemplar.py over every checkpoint in a training run directory."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default = "/home/zhenrant/rendering_prompted_muggled_sam/finetune_exemplar/run_20260321_202813",
        help="Training run directory containing finetune_epoch_*.pth checkpoints.",
    )
    parser.add_argument(
        "--eval_script",
        type=str,
        default="eval_image_exemplar.py",
        help="Path to the evaluation script to invoke.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch the evaluation script.",
    )
    parser.add_argument(
        "--checkpoint_glob",
        type=str,
        default="finetune_epoch_*.pth",
        help="Glob used to discover checkpoints inside run_dir.",
    )
    parser.add_argument(
        "--include_latest",
        action="store_true",
        help="Also evaluate finetune.pth if present.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if any checkpoint evaluation fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without launching them.",
    )
    args, eval_args = parser.parse_known_args()
    return args, eval_args


def checkpoint_sort_key(path: Path) -> Tuple[int, str]:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match:
        return int(match.group(1)), path.name
    return sys.maxsize, path.name


def collect_checkpoints(run_dir: Path, checkpoint_glob: str, include_latest: bool) -> List[Path]:
    checkpoints = [path for path in run_dir.glob(checkpoint_glob) if path.is_file()]
    checkpoints = sorted(checkpoints, key=checkpoint_sort_key)

    if include_latest:
        latest_path = run_dir / "finetune.pth"
        if latest_path.is_file():
            checkpoints.append(latest_path)

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints matching {checkpoint_glob!r} found under {run_dir}"
        )
    return checkpoints


def validate_forwarded_args(eval_args: Sequence[str]) -> None:
    blocked_flags = {"--finetune_ckpt"}
    conflicts = [arg for arg in eval_args if arg in blocked_flags]
    if conflicts:
        raise ValueError(
            "Do not pass "
            + ", ".join(sorted(set(conflicts)))
            + " through this wrapper; they are set per checkpoint."
        )


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> None:
    args, eval_args = parse_args()
    validate_forwarded_args(eval_args)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    eval_script = Path(args.eval_script).expanduser().resolve()
    if not eval_script.is_file():
        raise FileNotFoundError(eval_script)

    checkpoints = collect_checkpoints(run_dir, args.checkpoint_glob, args.include_latest)
    print(f"Found {len(checkpoints)} checkpoint(s) under {run_dir}")

    failures: List[Tuple[Path, int]] = []
    launched = 0

    for checkpoint_path in checkpoints:
        command = [
            args.python,
            str(eval_script),
            "--finetune_ckpt",
            str(checkpoint_path),
            *eval_args,
        ]
        print(f"\n[{launched + 1}/{len(checkpoints)}] {checkpoint_path.name}")
        print(format_command(command))

        if args.dry_run:
            launched += 1
            continue

        result = subprocess.run(command, check=False)
        launched += 1
        if result.returncode != 0:
            failures.append((checkpoint_path, result.returncode))
            print(
                f"Checkpoint {checkpoint_path.name} failed with exit code {result.returncode}"
            )
            if args.stop_on_error:
                break

    completed = launched - len(failures)
    print(
        f"\nFinished. attempted={launched} succeeded={completed} failed={len(failures)}"
    )
    if failures:
        for checkpoint_path, returncode in failures:
            print(f"  - {checkpoint_path}: exit_code={returncode}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
