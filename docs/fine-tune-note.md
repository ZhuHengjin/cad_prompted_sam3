# Brick SAM Fine-Tuning

Use `finetune_image_exemplar_multi_gt.py` with the versioned manifest described in [finetune-split-behavior.md](finetune-split-behavior.md).

Set the local model and exemplar-render paths before running:

```bash
DATA_ROOT=/home/hengjinz/data/brick_sam_sdg
MANIFEST="$DATA_ROOT/splits/v1/manifest.csv"
MODEL=/path/to/sam3.pt
REFS=/path/to/brick/exemplar/renders
OUTPUT=finetune_exemplar
```

## Train

```bash
python3 finetune_image_exemplar_multi_gt.py \
  --model_path "$MODEL" \
  --dataset_manifest "$MANIFEST" \
  --data_root "$DATA_ROOT" \
  --reference_dir "$REFS" \
  --output_dir "$OUTPUT" \
  --seed 42
```

The two training domains contribute 50/50 samples while the overall epoch size remains equal to the number of unique training views.

At startup, the trainer validates all 2,253 manifest rows, prints the manifest checksum and per-dataset split summary, and resolves 1,808 training views. The unbalanced train pools contain 1,513 wrist views and 295 side-camera views; deterministic epoch construction draws 904 entries from each.

Validation is performed after every completed epoch, including the final epoch. It uses all 230 validation views without training augmentation or domain oversampling. A view can contribute multiple evaluation targets when its mapping contains multiple brick labels.

## Resume

For the current Run B epoch-80 continuation, run:

```bash
bash scripts/train_brick_manifest_runb_e80.sh
```

The script resumes through epoch 180 (100 additional epochs), uses GPU 2, and creates a new run directory. It starts at `batch_size=2` because the 1008px exemplar model exhausted a 48 GiB GPU at batch size 13; `grad_accum=12` is retained.

For a custom resume checkpoint:

```bash
python3 finetune_image_exemplar_multi_gt.py \
  --model_path "$MODEL" \
  --dataset_manifest "$MANIFEST" \
  --data_root "$DATA_ROOT" \
  --reference_dir "$REFS" \
  --resume_path /path/to/finetune_epoch_006.pth \
  --resume_in_place \
  --seed 42
```

Keep the same manifest and seed when resuming; epoch sampling is derived from `seed + epoch`.

`--resume_in_place` writes new checkpoints and refreshed provenance into the checkpoint's existing run directory. Without it, resuming creates a new timestamped run directory while retaining checkpoint-compatible module keys.

## Final test evaluation

Select the checkpoint using validation results, then run the untouched test split explicitly:

```bash
python3 finetune_image_exemplar_multi_gt.py \
  --model_path "$MODEL" \
  --dataset_manifest "$MANIFEST" \
  --data_root "$DATA_ROOT" \
  --reference_dir "$REFS" \
  --resume_path /path/to/best_checkpoint.pth \
  --eval_only \
  --eval_split test \
  --seed 42
```

Use `--eval_split validation` for a standalone validation pass. Test data is never evaluated by the training loop.

## Run artifacts and failure behavior

- `dataset_manifest.csv` is a byte-for-byte copy of the manifest used for the run.
- `run_config.json` contains its SHA-256 digest, resolved counts and paths, sampling policy, and parsed arguments.
- `metrics.csv` is durable, append-only training telemetry. It has one `train_batch` row for every successful loss/backpropagation batch, a `train_epoch` summary row, and one `validation` row after each completed epoch. Standalone validation/test runs use `eval_validation` or `eval_test` rows.
- Checkpoints retain the existing exemplar fusion, detector, segmentation, optimizer, epoch, and step fields.
- Manifest mode fails before training if paths are missing, rows overlap, provenance is inconsistent, or manifest and legacy dataset arguments are mixed.
- `--eval_only` requires both a manifest and an explicit `--resume_path`; only `validation` and `test` are accepted evaluation splits.
- The test split currently contains 215 views and should be run only after selecting a checkpoint from validation behavior.

The exemplar render directory remains external to these datasets and must contain the image/mask naming convention expected by the existing reference loader. The manifest controls scene images and labels; it does not select or package exemplar renders.
