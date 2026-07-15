#!/usr/bin/env python3
"""Deprecated command-line compatibility shim for split-aware fine-tuning.

The former split implementation duplicated the complete training program and
was prone to drifting from ``finetune_image_exemplar_multi_gt.py``. This file
now contains no training or dataset logic: it emits a visible migration warning
and delegates argument parsing and execution to the canonical module. Existing
launchers may therefore keep the old executable name temporarily, while new
runs should use the versioned ``--dataset_manifest`` and ``--data_root``
interface directly.
"""

import warnings

from finetune_image_exemplar_multi_gt import main


if __name__ == "__main__":
    warnings.warn(
        "finetune_image_exemplar_multi_gt_split.py is deprecated; use "
        "finetune_image_exemplar_multi_gt.py with --dataset_manifest and --data_root.",
        FutureWarning,
        stacklevel=1,
    )
    main()
