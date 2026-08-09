#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/henryzhu/repos/perseve/src/perseve/synthetic_data_generation/_out_pose_easy10_5k}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/runs/manifests/pose_easy10_overfit50_v1/manifest.csv}"
REFERENCE_DIR="${REFERENCE_DIR:-/home/henryzhu/repos/perseve/src/perseve/synthetic_data_generation/_out_v2_abc_clean_20260727/abc_exemplars}"
MODEL_PATH="${MODEL_PATH:-/home/henryzhu/repos/LegoSegmentation/weights/sam3.pt}"
START_CHECKPOINT="${START_CHECKPOINT:-${REPO_ROOT}/runs/exemplar_view_pose_3mode_long_20260803/view_id/run_20260803_061122/checkpoints/finetune_epoch_060.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/cad_pose_easy10_overfit50_view_id}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-300}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/finetune_image_exemplar_multi_gt.py" \
  --model_path "${MODEL_PATH}" \
  --dataset_manifest "${MANIFEST}" \
  --data_root "${DATA_ROOT}" \
  --reference_dir "${REFERENCE_DIR}" \
  --transfer_path "${START_CHECKPOINT}" \
  --validate_before_training \
  --validate_on_train \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --dtype bf16 \
  --enable_pose \
  --pose_stage joint_lite \
  --exemplar_view_mode view_id \
  --enable_cad_prompt \
  --pose_deep_supervision \
  --pose_aux_layers 3,4,5 \
  --pose_aux_loss_weight 0.5 \
  --pose_train_min_match_iou 0.5 \
  --pose_eval_min_match_iou 0.7 \
  --pose_full_set_weight 0 \
  --lr 5e-5 \
  --joint_shared_lr_scale 0.6 \
  --pose_prompt_lr_scale 2.0 \
  --joint_bbox_weight 0.25 \
  --joint_objectness_weight 0.25 \
  --joint_mask_weight 0.10 \
  --weight_decay 1e-4 \
  --grad_clip_norm 1.0 \
  --batch_size 4 \
  --grad_accum 1 \
  --epochs "${EPOCHS}" \
  --save_every 10 \
  --save_debug_every 0 \
  --log_every 5 \
  --seed 42
