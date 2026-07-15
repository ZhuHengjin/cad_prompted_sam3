#!/usr/bin/env bash
set -euo pipefail

# Continue the Run B epoch-80 checkpoint through epoch 180 (100 additional epochs).
# The run gets a new timestamped output directory; it does not modify the source checkpoint.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Batch size 13 exhausts a 48 GiB A6000 at 1008px. Start conservatively.
export PYTORCH_ALLOC_CONF="expandable_segments:True"

uv run python finetune_image_exemplar_multi_gt.py \
  --model_path /home/hengjinz/repos/LegoSegmentation/weights/sam3.pt \
  --resume_path /home/hengjinz/repos/cad_prompted_sam3/finetune_exemplar/run_20260713_032020/finetune.pth \
  --dataset_manifest /home/hengjinz/data/brick_sam_sdg/splits/v1/manifest.csv \
  --data_root /home/hengjinz/data/brick_sam_sdg \
  --reference_dir /home/hengjinz/repos/LegoSegmentation/exemplars/renders \
  --device cuda:2 \
  --output_dir finetune_exemplar \
  --epochs 180 \
  --batch_size 2 \
  --grad_accum 12 \
  --seed 42 \
  --log_every 20 \
  --lr 3e-5
