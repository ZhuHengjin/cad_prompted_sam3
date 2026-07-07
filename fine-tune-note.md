# Fine-Tuning Notes for LEGO SAM3

This note summarizes how to continue fine-tuning the LEGO SAM3 model using the existing primary checkpoint and the new synthetic data.

## Paths

- Repo: `/home/henryzhu/repos/cad_prompted_sam3`
- Base SAM3 weights: `/home/henryzhu/repos/LegoSegmentation/weights/sam3.pt`
- Primary LEGO fine-tuned checkpoint: `/home/henryzhu/repos/LegoSegmentation/weights/lego_sam3_runB_e80.pth`
- Training data parent: `/home/henryzhu/data/brick_sam_sdg/run_500_scenes_yaw20_not_stud_aligned`
- Exemplar renders: `/home/henryzhu/repos/LegoSegmentation/exemplars/renders`

The data parent contains camera folders:

- `Side_Camera_0`
- `Side_Camera_1`
- `Side_Camera_2`
- `Side_Camera_3`

Pass these camera folders directly to the training script. The script does not recursively discover them from the parent directory.

## Dataset Split

Split the data before fine-tuning if this run is meant to produce a model you will evaluate or compare.

Recommended split:

- Train: 70-80%
- Validation: 10-15%
- Test: 10-15%

Split by frame or scene ID, not by individual object instance. Keep the same frame ID in the same split across all camera folders. For example, frame `0042` should be train, validation, or test for every `Side_Camera_*`, never split across cameras.

Use the splits like this:

- Train: used for optimizer updates.
- Validation: used to choose checkpoint, tune epochs, LR, and catch overfitting.
- Test: untouched until the final report.

Use `finetune_image_exemplar_multi_gt_split.py` for split-aware training. It supports split CSVs via `--split_dir`, or explicit `--train_split_csv`, `--val_split_csv`, and `--test_split_csv` paths.

## Recommended Continuation Command

Run this from the repo root using the existing Python/CUDA environment that already has PyTorch installed.

```bash
cd /home/henryzhu/repos/cad_prompted_sam3

DATA=/home/henryzhu/data/brick_sam_sdg/run_500_scenes_yaw20_not_stud_aligned
WEIGHTS=/home/henryzhu/repos/LegoSegmentation/weights
REFS=/home/henryzhu/repos/LegoSegmentation/exemplars/renders

python finetune_image_exemplar_multi_gt_split.py \
  --model_path "$WEIGHTS/sam3.pt" \
  --resume_path "$WEIGHTS/lego_sam3_runB_e80.pth" \
  --dataset_root "$DATA/Side_Camera_0,$DATA/Side_Camera_1,$DATA/Side_Camera_2,$DATA/Side_Camera_3" \
  --reference_dir "$REFS" \
  --split_dir splits/lego_yaw20 \
  --split_ratios 0.8,0.1,0.1 \
  --ref_view_ids 0,1,2,3,4,5,6,7,8,9,10,11 \
  --epochs 100 \
  --batch_size 8 \
  --grad_accum 12 \
  --lr 1e-4 \
  --device cuda:0 \
  --save_every 5 \
  --save_debug_every 0 \
  --output_dir finetune_exemplar_lego_continue
```

`--epochs` is the final target epoch, not the number of additional epochs. Since `lego_sam3_runB_e80.pth` is the epoch-80 primary checkpoint, `--epochs 100` means continue for about 20 more epochs.

With `--split_dir splits/lego_yaw20`, the script creates or reuses:

- `splits/lego_yaw20/train.csv`
- `splits/lego_yaw20/val.csv`
- `splits/lego_yaw20/test.csv`

The split is by `frame_id`, so matching frame IDs stay together across all four side cameras. Training uses `train.csv`; validation during training uses `val.csv`; `test.csv` is reserved for final evaluation and is not used during training. Validation logs `val_loss` alongside task metrics such as `avg_iou`, `correct_rate`, and PQ stats.

## Practical Notes

- The multi-GT script reloads the fine-tuned model modules from `--resume_path`.
- In the current script, optimizer loading is commented out, so continuation is a weight warm-start with a fresh AdamW optimizer.
- If CUDA memory is tight, reduce `--batch_size` first and keep `--grad_accum` high enough to preserve the effective batch size.
- Keep `--save_debug_every 0` for routine training to avoid extra disk usage. Enable it briefly if you need visual debugging.
- Free disk space before launching. Checkpoints are large, and the root filesystem was nearly full during setup.

## After Training

1. Inspect logs for training loss and average IoU.
2. Review validation loss and validation metrics, not just training performance.
3. Pick the best checkpoint based on validation metrics.
4. Run the held-out test set once for final numbers.
5. Archive the chosen checkpoint with the exact command, data split, and commit hash used for the run.
