# CAD Pose Matching and Evaluation

This document is the canonical policy for assigning detector candidates to
ground-truth CAD instances, filtering pose supervision, and interpreting pose
metrics. It applies to the `head`, `joint`, and `joint_lite` pose-training
stages. Stage-specific optimizer and loss settings belong in their training
guides; experiment-specific thresholds and results belong under
[`experiments/`](experiments/README.md).

The implementation lives in
[`cad_pose/matching.py`](../muggled_sam/v3_sam/cad_pose/matching.py) and
[`finetune_image_exemplar_multi_gt.py`](../finetune_image_exemplar_multi_gt.py).

## One-to-one mask assignment

For each image and CAD ID, the matcher:

1. resizes the ground-truth masks to the prediction grid;
2. binarizes prediction logits at `> 0` and targets at `> 0.5`;
3. computes a ground-truth-by-prediction mask-IoU matrix; and
4. greedily accepts the highest-IoU pairs while using each ground-truth
   instance and prediction at most once.

Ties are deterministic: lower ground-truth and prediction indices win. Pose
values never enter the assignment cost, which avoids an unstable feedback loop
early in training. The algorithm is greedy, not a globally optimal Hungarian
assignment.

During training, only pose-eligible ground-truth instances enter pose
assignment. Pose-ineligible visible objects can still contribute segmentation
supervision in stages that train the detector path.

## Training-time IoU filter

`--pose_train_min_match_iou` accepts values in `[0, 1]`. Assignment happens
first; the threshold then decides which assigned pairs receive pose loss. The
default `0` preserves unfiltered behavior.

For a threshold `tau`:

```text
match IoU >= tau: pose supervision is enabled
match IoU <  tau: pose supervision is disabled
```

The effect of a rejected pair on the rest of training depends on the stage:

| Stage | Effect when no pose match is accepted |
| --- | --- |
| `head` | No trainable loss remains for that image, because only the pose head is trainable. |
| `joint` | The trainable detection and segmentation path can still receive its normal supervision. |
| `joint_lite` | The trainable fusion and detector path can still receive reduced box, objectness, and mask anchors. |

Training logs distinguish assignment from threshold acceptance:

- `pose_assignment_coverage = assigned matches / eligible instances`;
- `pose_match_coverage = accepted matches / eligible instances`; and
- `pose_match_acceptance_rate = accepted matches / assigned matches`.

The logs also record `pose_accepted_matches`, `pose_total_matches`, and the
configured IoU threshold. Compare coverage alongside pose loss so improvements
cannot be attributed solely to discarding difficult examples.

## Evaluation-time IoU filter

Evaluation always emits an all-match row with an IoU threshold of `0`. This row
measures pose quality for every one-to-one assignment and reports
`pose_assignment_coverage` when fewer candidates than eligible instances are
available.

When `--pose_eval_min_match_iou` is greater than zero, evaluation emits a
second, conditional row. A threshold of `0.7`, for example, produces:

```text
validation_pose_iou_070
validation_pose_calibrated_iou_070
```

Conditional rows report:

- `samples`: IoU-qualified matches;
- `eligible_samples`: all eligible ground-truth pose instances;
- `pose_match_coverage`: qualified matches divided by eligible instances;
- `pose_success_rate`: successful poses among qualified matches; and
- `pose_end_to_end_success_rate`: qualified successes divided by all eligible
  instances.

Equivalently:

```text
pose_end_to_end_success_rate
  = pose_success_rate * pose_match_coverage
```

Read the three quantities together. Conditional success isolates pose quality
after acceptable localization; coverage exposes how often that condition is
met; end-to-end success keeps missed and poorly localized instances in the
denominator. The all-match row remains the threshold-independent comparison.

## Calibration

Pose-quality temperature is fitted on validation labels only. Final validation
uses the fitted scalar temperature for both all-match and conditional Brier
score and expected calibration error. Test evaluation reuses the temperature
stored in the checkpoint and never fits on test labels.

## Choosing thresholds

Thresholds are experiment parameters, not model-architecture constants.
`0.5` for training and `0.7` for evaluation are the current joint-lite recipe,
not universal defaults. Record threshold changes in the experiment document
and compare at least:

- conditional pose geometry and success;
- IoU-qualified match coverage;
- end-to-end pose success; and
- all-match pose and segmentation metrics as guardrails.

## Known limitations

- Greedy matching is deterministic but may be worse than a globally optimal
  assignment in crowded scenes.
- Mask IoU makes pose supervision depend on candidate localization quality.
- A conditional row answers pose quality *given* a localization threshold; it
  is not a complete end-to-end result by itself.
- Results at different IoU thresholds are not directly comparable unless the
  threshold and coverage are reported together.
