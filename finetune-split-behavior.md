# Fine-Tuning Split Behavior

This note documents the dataset split behavior in `finetune_image_exemplar_multi_gt_split.py`.

## What Changed

The fine-tuning script can now use train, validation, and test split CSVs instead of training on every frame from `--dataset_root`.

You can enable this with:

```bash
--split_dir splits/lego_yaw20 \
--split_ratios 0.8,0.1,0.1
```

On first run, the script creates:

```text
splits/lego_yaw20/train.csv
splits/lego_yaw20/val.csv
splits/lego_yaw20/test.csv
```

On later runs, it reuses those files so experiments stay reproducible.

## Split Unit

Splits are made by `frame_id`, not by object instance.

This matters because the dataset has multiple camera folders:

```text
Side_Camera_0
Side_Camera_1
Side_Camera_2
Side_Camera_3
```

If frame `0042` goes into train, then every available `0042` frame across all camera folders is train. The same rule applies to validation and test.

This prevents leakage where one camera view of a scene is used for training and another camera view of the same scene is used for validation or test.

## Training Behavior

When `--split_dir` or `--train_split_csv` is provided:

- Training entries are filtered through `train.csv`.
- Dataset multipliers are applied after the train split filter.
- Validation entries are filtered through `val.csv`.
- `test.csv` is checked and logged, but not used during training.

If no split arguments are provided, the script keeps the old behavior and trains on all entries from `--dataset_root`.

## CLI Options

Use `--split_dir` for the normal workflow:

```bash
--split_dir splits/lego_yaw20
```

This points to a directory containing or receiving:

```text
train.csv
val.csv
test.csv
```

Use `--split_ratios` when creating splits:

```bash
--split_ratios 0.8,0.1,0.1
```

The values are normalized, so `8,1,1` is equivalent.

Use explicit split CSVs if needed:

```bash
--train_split_csv splits/lego_yaw20/train.csv \
--val_split_csv splits/lego_yaw20/val.csv \
--test_split_csv splits/lego_yaw20/test.csv
```

Use `--recreate_splits` only when you intentionally want to overwrite an existing split:

```bash
--recreate_splits
```

Without `--recreate_splits`, the script fails if only some split files exist. This avoids silently mixing old and new split files.

## Recommended Command

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

## Test Set

The test split is intentionally not used during training.

Use it only after selecting the best checkpoint from validation performance. This keeps final metrics honest.
