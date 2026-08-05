# CAD Pose Joint-Lite Training

Joint-lite training tests whether pose-aware adaptation of the frozen SAM3
exemplar and detector representation improves CAD pose prediction. Pose remains
the primary objective. Box, objectness, and mask supervision act as lower-weight
anchors so pose-driven updates do not destroy candidate localization.

This page is the reusable joint-lite runbook. Configuration and results for the
2026-08-01 ABC run are recorded separately in the
[experiment log](experiments/2026-08-01-abc-pose-joint-lite.md).

## Starting checkpoint

Start from a pose-head checkpoint selected on validation geometry rather than
from a segmentation-only checkpoint. Joint-lite changes both the trainable
module set and optimizer parameter groups, so use `--no_resume_optimizer` and
initialize a fresh optimizer while restoring model and pose-head weights.

The checkpoint stores its completed epoch. `--epochs` is the final epoch number,
not the number of additional epochs. Set it above the stored checkpoint epoch;
their difference is the number of new epochs that will run.

## Trainable modules

With base learning rate `--lr`, joint-lite configures:

| Module | State | Learning rate |
| --- | --- | ---: |
| Image encoder and projection | frozen | — |
| Sampling and text encoders | frozen | — |
| Exemplar fusion | trainable | `lr * joint_shared_lr_scale` |
| Detector and candidate-token path | trainable | `lr * joint_shared_lr_scale` |
| Segmentation decoder | frozen | — |
| CAD pose head | trainable | `lr` |
| Pose-only CAD prompt adapter | optional/trainable | `lr * pose_prompt_lr_scale` |

The frozen segmentation decoder remains in the differentiable forward path.
Mask loss can therefore anchor fusion and detector inputs without updating
decoder parameters.

## Combined CAD-prompt and deep-supervision run

The combined Phase 1/2 run enables direct pose-only attention to the padded CAD
exemplar tokens and supervises detector layers 3–5 in addition to the standard
layer-6 prediction. The final mask assignment and candidate indices are reused
for every auxiliary layer. Layer-6 pose loss retains weight `1.0`; the mean of
layers 3–5 receives total weight `0.5`. Auxiliary pose quality is disabled, and
only layer 6 runs at inference.

The stronger optimizer recipe is:

| Parameter group | Learning rate |
| --- | ---: |
| Fusion and detector | `3e-5` |
| Existing pose head and reference-view encoder | `5e-5` |
| New CAD prompt adapter | `1e-4` |

This is expressed as `--lr 5e-5`, `--joint_shared_lr_scale 0.6`, and
`--pose_prompt_lr_scale 2.0`. The segmentation decoder and image encoder remain
frozen. Gradient clipping at norm `1.0` bounds the larger combined update.

The repository launch script contains the resolved camera-checkpoint and ABC-v2
defaults used for this run:

```bash
scripts/run_cad_pose_deep_prompt_tonight.sh
```

It continues the epoch-60 checkpoint through epoch 68 by default, producing
eight training epochs with validation between epochs and after the final epoch.
This is intended as a trend-detection run; extend from the best checkpoint only
if pose geometry and IoU-qualified coverage are moving in the right direction.

Override `DEVICE`, `FINAL_EPOCH`, `START_CHECKPOINT`, `DATA_ROOT`, `MODEL_PATH`,
or `OUTPUT_DIR` through environment variables when needed. The script starts a
fresh optimizer with `--no_resume_optimizer`; pose-head v3 is migrated to v4 and
the new prompt adapter is initialized at the start of the run.

## Pose matching

Pose matching and metric interpretation are shared by all pose-training stages;
see [CAD pose matching and evaluation](cad-pose-matching-and-evaluation.md) for
the canonical policy. The joint-lite recipe uses a training mask-IoU threshold
of `0.5`:

```text
IoU >= 0.5: box + objectness + mask + pose losses
IoU <  0.5: box + objectness + mask losses only
```

Low-IoU images are retained for anchor supervision. Pose loss is averaged over
accepted matches; a batch with no accepted pose match can still update the
fusion and detector modules through the anchors.

The recipe uses `0.7` for the additional conditional evaluation row while
retaining the all-match row:

```text
validation_pose_iou_070
validation_pose_calibrated_iou_070
```

These values are recipe choices rather than architecture constants. Record any
changes in the corresponding experiment document.

## Pose-first joint-lite loss

For `--pose_stage joint_lite`, the per-image objective is:

$$
L = L_{pose,\,IoU\ge\tau}
  + w_{box}L_{box}
  + w_{objectness}L_{objectness}
  + w_{mask}L_{mask}.
$$

The current recipe uses `tau=0.5`, `w_box=0.25`,
`w_objectness=0.25`, and `w_mask=0.10`. The mask component retains its
existing internal definition:

$$
L_{mask}=2L_{BCE}+2L_{Dice}.
$$

Joint-lite does not apply the historical extra outer mask multiplier. The
segmentation-only, `head`, and `joint` paths retain their existing detection
loss. Full point-set gradients remain a separate ablation; the default
`--pose_full_set_weight 0` keeps them disabled.

## Launch command

Set paths and the intended final epoch for the run:

```bash
REPO=/path/to/cad_prompted_sam3
MODEL=/path/to/sam3.pt
MANIFEST=/path/to/dataset_manifest.csv
DATA_ROOT=/path/to/data/root
REFERENCES=/path/to/cad/exemplar/renders
CHECKPOINT=/path/to/pose_head_run/checkpoints/selected_pose_head_checkpoint.pth
OUTPUT="$REPO/runs/finetune_exemplar_cad_pose_joint_lite"
FINAL_EPOCH=60  # Replace with the intended final epoch.
DEVICE=cuda:0
```

Then launch from the repository root:

```bash
cd "$REPO"
uv run python finetune_image_exemplar_multi_gt.py \
  --model_path "$MODEL" \
  --dataset_manifest "$MANIFEST" \
  --data_root "$DATA_ROOT" \
  --reference_dir "$REFERENCES" \
  --resume_path "$CHECKPOINT" \
  --no_resume_optimizer \
  --validate_before_training \
  --enable_pose \
  --pose_stage joint_lite \
  --pose_train_min_match_iou 0.5 \
  --pose_eval_min_match_iou 0.7 \
  --joint_shared_lr_scale 0.1 \
  --joint_bbox_weight 0.25 \
  --joint_objectness_weight 0.25 \
  --joint_mask_weight 0.10 \
  --pose_full_set_weight 0 \
  --lr 3e-5 \
  --weight_decay 1e-4 \
  --epochs "$FINAL_EPOCH" \
  --batch_size 4 \
  --grad_accum 1 \
  --ref_view_ids 0,1,2,3,4,5,6,7,8,9,10,11 \
  --max_side_length 1008 \
  --matches_per_gt 1 \
  --det_filter 0.0 \
  --nms_iou 0.5 \
  --dtype bf16 \
  --device "$DEVICE" \
  --seed 42 \
  --log_every 4 \
  --save_every 1 \
  --save_debug_every 20 \
  --output_dir "$OUTPUT"
```

`--validate_before_training` records an unchanged-checkpoint baseline before
the first joint update. The output path is a base directory; the trainer creates
a timestamped `run_*` directory below it.

## Checkpoint selection

Select checkpoints using pose validation metrics, not total loss. Compare:

1. conditional mean surface distance at the configured evaluation IoU;
2. conditional centroid, translation, and depth errors;
3. conditional and end-to-end pose success;
4. IoU-qualified match coverage; and
5. segmentation IoU as a stability guardrail.

Joint adaptation is promising when conditional pose geometry improves without
a material coverage or segmentation-IoU regression. If coverage improves while
conditional pose stays flat, localization was likely the bottleneck. If both
remain flat, explicit CAD geometry features or reference-view pose conditioning
are stronger next experiments.

## Few-scene qualitative pose check

Render the selected epoch-66 checkpoint on three lightweight validation scenes:

```bash
.venv/bin/python scripts/visualize_cad_pose_predictions.py
```

The default frames are `0017`, `0029`, and `0036` (eight eligible instances in
total). Every matched instance gets its own directory containing separate PNGs:

```text
00_rgb.png
01_mask_gt.png
02_mask_pred.png
03_dimensions_gt.png
04_dimensions_pred.png
05_orientation_gt.png
06_orientation_pred.png
07_surface_gt.png
08_surface_pred.png
09_mask_error.png
10_exemplar_01.png ... 13_exemplar_04.png
14_overview.png
```

The mask images are colored overlays on otherwise clean RGB images. Dimension
views contain only the projected CAD box, orientation views contain only XYZ
axes, and surface views contain only a deterministic subset of projected CAD
surface points. The surface pair is included because it exposes irregular-shape
and symmetry behavior that the dimension box cannot. `09_mask_error.png` shows
true positives in green, false positives in magenta, and false negatives in
orange. Four small reference exemplar renders are included for prompt context.
`14_overview.png` combines everything into one large diagnostic sheet: GT and
prediction views occupy the two large columns, while clean RGB, mask error,
metrics, and the four smaller exemplars occupy a narrow context column.
Per-match mask IoU, detection/pose scores, symmetry-safe point-set error,
centroid error, translation error, and depth error are stored in the local
`metrics.json` and run-level `summary.json` without obscuring the images.

The output defaults to:

```text
runs/cad_pose_deep_prompt_joint_lite/run_20260805_020155/
  pose_visualizations_epoch_066_separate/
```

Use explicit frames or a different checkpoint without changing the evaluator:

```bash
.venv/bin/python scripts/visualize_cad_pose_predictions.py \
  --checkpoint runs/cad_pose_deep_prompt_joint_lite/run_20260805_020155/checkpoints/finetune.pth \
  --frame 0017 \
  --frame 0036
```

The command reads the checkpoint's model, manifest, dataset, reference-view,
resize, CAD-prompt, and NMS settings and verifies the manifest checksum before
inference. It uses the same mask NMS and one-to-one assignment as validation.
Unmatched ground-truth instances and matches below the validation IoU threshold
are retained and labeled instead of being silently omitted. Raw XYZ axes can
look different for symmetry-equivalent poses, so prefer the projected point-set
alignment and point-set metrics when judging symmetric CAD objects.
