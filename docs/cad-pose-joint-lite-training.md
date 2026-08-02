# CAD Pose Joint-Lite Training

Joint-lite training tests whether pose-aware adaptation of the frozen SAM3
exemplar and detector representation improves CAD pose prediction. Pose remains
the primary objective. Box, objectness, and mask supervision act as lower-weight
anchors so pose-driven updates do not destroy candidate localization.

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

The frozen segmentation decoder remains in the differentiable forward path.
Mask loss can therefore anchor fusion and detector inputs without updating
decoder parameters.

## Pose-match IoU filtering

The pose matcher performs greedy one-to-one assignment by mask IoU. Set
`--pose_train_min_match_iou` to prevent a poorly localized candidate token from
receiving pose supervision. With a threshold of `0.5`:

```text
IoU >= 0.5: box + objectness + mask + pose losses
IoU <  0.5: box + objectness + mask losses only
```

Low-IoU images are retained for anchor supervision. Pose loss is averaged over
accepted matches; a batch with no accepted pose match can still update the
fusion and detector modules through the anchors.

Set `--pose_eval_min_match_iou` to emit an additional conditional metric row
without replacing the all-match metrics. A threshold of `0.7` produces phase
suffixes such as:

```text
validation_pose_iou_070
validation_pose_calibrated_iou_070
```

Each conditional row reports:

- `samples`: IoU-qualified matches;
- `eligible_samples`: all eligible ground-truth pose instances;
- `pose_match_coverage`: qualified matches divided by eligible instances;
- `pose_success_rate`: successful poses among qualified matches; and
- `pose_end_to_end_success_rate`: qualified successes divided by all eligible
  instances.

The end-to-end rate and coverage keep conditional results from looking better
merely because difficult detections were filtered out.

## Pose-first joint-lite loss

For `--pose_stage joint_lite`, the per-image objective is:

$$
L = L_{pose,\,IoU\ge\tau}
  + w_{box}L_{box}
  + w_{objectness}L_{objectness}
  + w_{mask}L_{mask}.
$$

The baseline settings use `tau=0.5`, `w_box=0.25`,
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
