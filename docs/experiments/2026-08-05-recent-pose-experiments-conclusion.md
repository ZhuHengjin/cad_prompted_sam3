# Conclusion: Recent Exemplar-View and Deep CAD-Prompt Experiments

## Executive conclusion

The experiments show that **adding reference-view identity is useful**, but they
do not show that continuous camera geometry is uniquely useful. In the completed
three-mode comparison, both view-aware modes beat the no-view baseline on
localization coverage and end-to-end pose success; the simple learned `view_id`
encoding was the best overall pose variant, while the continuous `camera`
encoding preserved segmentation best.

The later deep CAD-prompt/deep-supervision continuation improved some pose and
matching metrics, but not all of them at once. Epoch 66 gave the strongest pose
tradeoff, with better centroid/translation errors and success, but segmentation
IoU fell by 2.42 points. By epoch 72, segmentation and match coverage recovered,
but normalized surface error was worse than the starting camera checkpoint.
Therefore this run is **promising but not a clean improvement over the camera
baseline**.

## 1. `exemplar_view_pose_3mode_long_20260803`

### What changed

This was a controlled ablation of the metadata added to each reference-exemplar
token. All three arms used the same corrected canonical renders, 12 fixed views,
dataset/split, initialization checkpoint, seed (`42`), and training budget.

| Arm | Only intended representation change |
| --- | --- |
| `none` | No view metadata; exact bypass and corrected-render baseline. |
| `camera` | Added a learned residual derived from each render camera's continuous forward/up directions in the CAD frame. |
| `view_id` | Added a learned embedding for the fixed ordinal view ID (`0`–`11`), without explicit camera geometry. |

Training had two stages: epochs 1–10 trained the pose head (and view adapter when
present) at `1e-4`; epochs 11–60 switched to `joint_lite`, training the pose/view
modules at `3e-5` and the exemplar-fusion/detector path at `3e-6`. The image
encoder and segmentation decoder remained frozen. Pose supervision required
training mask IoU `>= 0.5`; the comparison below uses the final calibrated
validation at mask IoU `>= 0.7`.

### Result

Lower is better for surface and centroid error; higher is better for the other
columns. Conditional pose success is measured only on IoU-qualified matches;
end-to-end success uses all 558 eligible instances.

**What the two success metrics mean:** a prediction is a pose success only when
it passes both configured geometry checks. For this point-set dataset, its
normalized centroid error and normalized full-surface placement error must each
be `<= 0.1`. **Conditional pose success** is the fraction of these successes
among the one-to-one mask matches whose mask IoU is at least `0.7`; it answers,
“when localization is good enough, how often is the pose correct?” **End-to-end
success** is the fraction of successes among all 558 pose-eligible ground-truth
instances, so missing predictions and matches below IoU `0.7` count as failures.
The relationship is:

```text
end-to-end success = conditional pose success * match coverage
```

For example, `view_id` has `3.24% * 77.42% = 2.51%` end-to-end success.

| Arm | Seg. IoU | Mean surface error | Centroid error | Match coverage | Conditional pose success | End-to-end success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 0.6535 | 0.7636 | 13.70 cm | 73.84% | 1.70% | 1.25% |
| `camera` | **0.6832** | 0.7662 | 13.59 cm | 77.24% | 2.09% | 1.61% |
| `view_id` | 0.6754 | **0.7552** | **13.54 cm** | **77.42%** | **3.24%** | **2.51%** |

Relative to `none`, `camera` raised segmentation IoU by 0.0297 and coverage by
3.41 percentage points, but its mean surface error was essentially flat/slightly
worse. `view_id` raised segmentation IoU by 0.0219 and coverage by 3.58 points,
while reducing centroid/translation error by 0.16/0.15 cm and doubling
end-to-end success from 1.25% to 2.51%.

**Interpretation:** view-aware exemplar tokens help, and `view_id` is the best
overall pose result in this fixed twelve-view setup. Because `view_id` matches or
beats `camera`, this experiment does not demonstrate learning of continuous
camera geometry; the model may only need a stable view slot. Absolute pose
success remains low, so this is an incremental improvement, not a solved pose
system.

## 2. `cad_pose_deep_prompt_joint_lite`

### What changed

This experiment continued the epoch-60 `camera` checkpoint above and combined
two new pose mechanisms:

1. **Direct CAD-token prompting:** the pose head received a new pose-only
   attention path to the padded CAD exemplar tokens.
2. **Deep pose supervision:** detector layers 3, 4, and 5 received auxiliary
   pose losses with total weight `0.5`, in addition to the normal layer-6 pose
   loss at weight `1.0`. Only layer 6 is used for inference.

It also used a stronger optimizer: pose/view modules at `5e-5`, fusion/detector
at `3e-5` (10x the earlier `joint_lite` shared rate), and the new CAD-prompt
adapter at `1e-4`, with gradient clipping at `1.0`. The same low-weight box,
objectness, and mask anchors were retained. Training ran from epoch 60 through
73; the last complete pose validation is epoch 72. The configured target was
epoch 76, and the run has no `COMPLETED` marker.

### Result

These are directly comparable, uncalibrated validation rows at mask IoU
`>= 0.7`. Epoch 60 is the unchanged camera checkpoint before the new training.

| Checkpoint | Seg. IoU | Mean surface error | Centroid error | Match coverage | Conditional pose success | End-to-end success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Epoch 60 baseline | **0.6832** | 0.7667 | 13.60 cm | 77.24% | 2.09% | 1.61% |
| Epoch 66, best pose tradeoff | 0.6590 | **0.7637** | **13.36 cm** | 80.11% | **3.36%** | **2.69%** |
| Epoch 72, later tradeoff | **0.6833** | 0.8057 | 13.39 cm | **81.18%** | 2.65% | 2.15% |

At epoch 66, centroid and translation error each improved by about 0.24 cm,
coverage improved by 2.87 points, and end-to-end success improved by 1.08
points. However, segmentation IoU dropped by 0.0242 and p95 surface error was
slightly worse. At epoch 72, segmentation fully recovered and coverage gained
3.94 points over baseline, but mean/p95 surface error worsened from
0.7667/2.0872 to 0.8057/2.2982. This indicates a real localization/centroid
benefit, but no stable across-the-board improvement in full pose geometry.

## Overall decision

- **Best supported change:** retain reference-view conditioning. Use `view_id`
  as the current fixed-view pose baseline; use `camera` when segmentation IoU is
  the primary guardrail.
- **Deep-prompt status:** treat epoch 66 as a useful Pareto checkpoint for pose
  analysis, not as a replacement with no downside. Epoch 72 is preferable when
  segmentation and coverage matter more than surface-shape accuracy.
- **Evidence limit:** both experiments use one seed and validation results only.
  The three-mode run omits the `shuffled_camera` and `zero_camera` controls, and
  neither experiment includes a final held-out test evaluation.
- **Next decisive test:** compare `none`, `camera`, and `view_id` across multiple
  seeds with camera jitter and held-out angles. For the deep-prompt recipe,
  ablate CAD prompting and auxiliary supervision separately before attributing
  the observed tradeoff to either mechanism.

## Evidence

- Three-mode commands and completion: [`experiment_matrix.json`](../../runs/exemplar_view_pose_3mode_long_20260803/experiment_matrix.json), [`overnight.log`](../../runs/exemplar_view_pose_3mode_long_20260803/overnight.log), and [`COMPLETED`](../../runs/exemplar_view_pose_3mode_long_20260803/COMPLETED)
- Three-mode metrics: [`none`](../../runs/exemplar_view_pose_3mode_long_20260803/none/run_20260803_010315/metrics.csv), [`camera`](../../runs/exemplar_view_pose_3mode_long_20260803/camera/run_20260803_033613/metrics.csv), and [`view_id`](../../runs/exemplar_view_pose_3mode_long_20260803/view_id/run_20260803_061122/metrics.csv)
- Deep-prompt configuration and metrics: [`run_config.json`](../../runs/cad_pose_deep_prompt_joint_lite/run_20260805_020155/run_config.json) and [`metrics.csv`](../../runs/cad_pose_deep_prompt_joint_lite/run_20260805_020155/metrics.csv)
