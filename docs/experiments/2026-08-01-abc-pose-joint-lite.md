# Experiment: ABC Pose Joint-Lite Continuation

- **Date:** 2026-08-01
- **Status:** completed
- **Source commit:** not recorded by the run
- **Run artifacts:**
  `runs/finetune_exemplar_abc_pose_joint_lite/run_20260801_013714/`

This record reconstructs the experiment from its local `run_config.json`,
`pose_provenance.json`, and `metrics.csv`. Paths stored inside those files
predate the move into `runs/`; the artifact path above is canonical now.

## Question

Does pose-first adaptation of exemplar fusion and the detector/candidate-token
path improve CAD pose prediction without materially degrading segmentation or
IoU-qualified match coverage?

## Method

- [CAD pose-head implementation](../cad-pose-head-implementation.md)
- [CAD pose joint-lite training](../cad-pose-joint-lite-training.md)
- [CAD pose matching and evaluation](../cad-pose-matching-and-evaluation.md)
- [ABC pose-v2 preparation](../abc-pose-v2-preparation.md)

The run continued a head-only checkpoint at epoch 42 through epoch 52. It
trained the pose head at the base learning rate and the shared fusion/detector
path at one tenth of that rate. The image encoder, projection, sampling/text
encoders, and segmentation decoder remained frozen.

## Configuration and provenance

| Item | Recorded value |
| --- | --- |
| Base model | `sam3.pt`; checksum not recorded |
| Starting checkpoint | pose-head checkpoint `finetune_epoch_042.pth`; checksum not recorded |
| Dataset | `abc_pose_v2`, 1,112 manifest rows |
| Split | 890 train / 111 validation / 111 test scenes |
| Manifest SHA-256 | `cb292131ea77470641e430ef18595515becdef002f3e2b9c36cdc6935f014cec` |
| Aggregate annotation SHA-256 | `25c2fad844b9bc3c7eb79d9fc8055d93aa092435cd6335c624caaf081a836f81` |
| Pose schema | `2.0.0` |
| Point sampling pipeline | `surface_area_deterministic_v1` |
| Seed | `42` |
| Device and dtype | `cuda:0`, `bf16` |
| Epoch range | 42 baseline, trained through 52 |
| Batch and accumulation | batch 4, accumulation 1 |
| Learning rate / weight decay | `3e-5` / `1e-4` |

Key experiment settings were:

```text
pose_stage                  joint_lite
joint_shared_lr_scale       0.1
joint_bbox_weight           0.25
joint_objectness_weight     0.25
joint_mask_weight           0.10
pose_train_min_match_iou    0.5
pose_eval_min_match_iou     0.7
pose_full_set_weight        0.0
```

The archived `run_config.json` is the source of truth for the full command.
Because it does not store a Git revision or model/checkpoint checksums, exact
code and weight reconstruction is incomplete; future runs should record them.

## Results

The table compares the validation-before-training baseline with the final
calibrated validation. Errors are computed on matches with mask IoU at least
`0.7`; coverage and end-to-end success use all 558 eligible instances as the
denominator. Segmentation IoU comes from the corresponding all-match
validation row.

| Validation point | Mean surface distance | Centroid error | Translation error | Conditional success | Match coverage | End-to-end success | Segmentation IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Epoch 42 baseline | 0.795 | 13.98 cm | 14.02 cm | 1.68% | 74.55% | 1.25% | 0.6301 |
| Epoch 52 calibrated | 0.777 | 13.78 cm | 13.85 cm | 2.86% | 75.09% | 2.15% | 0.6298 |

The final fitted pose-score temperature was `0.9198`. Its all-match Brier score
was `0.02095` and expected calibration error was `0.00044`; calibration metrics
should not be compared directly with the uncalibrated epoch-42 baseline.

## Observations

- Conditional pose geometry improved modestly, while conditional and
  end-to-end success increased from the baseline.
- IoU-qualified match coverage increased by about 0.54 percentage points.
- Segmentation IoU changed by less than 0.001, so no material segmentation
  regression is visible in the final validation row.
- Validation success fluctuated across epochs; this single run does not
  establish variance across seeds.
- The source commit and input-weight checksums are missing, which limits exact
  reproducibility despite the strong dataset provenance.

## Conclusion

This run provides initial evidence that joint-lite adaptation improves pose
quality without sacrificing segmentation or localization coverage. Treat the
result as directional until it is reproduced with recorded source and weight
checksums and additional seeds.

## Artifacts

The local run directory contains checkpoints for epochs 43–52, the final and
calibrated checkpoints, the manifest snapshot, full metrics, provenance,
training curves, and debug images. These generated artifacts are ignored by
Git and should be copied to durable storage if they must be retained.
