# Current Training Loss

This page documents the loss that is **actually used** by
[`finetune_image_exemplar_multi_gt.py`](../finetune_image_exemplar_multi_gt.py),
for both training and validation. The authoritative implementation is
[`compute_multi_gt_detection_loss`](../finetune_image_exemplar_multi_gt.py#L1080).

## The complete objective at a glance

For one image, the trainer computes:

$$
L_{image}=2L_{mask}+L_{presence}+w_{bbox}L_{box}.
$$

The default weights are declared in the
[argument parser](../finetune_image_exemplar_multi_gt.py#L1581):

| Setting | Default | Purpose |
| --- | ---: | --- |
| `bce_weight` | 2.0 | Pixel-level mask BCE coefficient. |
| `dice_weight` | 2.0 | Mask-overlap Dice coefficient. |
| `bbox_weight` | 1.0 | Box-regression coefficient. |
| `score_weight` | 0.3 | Matched prediction-score coefficient. |
| `no_object_weight` | 0.45 | Unmatched prediction-score coefficient. |

So, with defaults:

$$
L_{image}=2\operatorname{mean}_{(j,p)\in\mathcal M}
\left[2L_{BCE}(j,p)+2L_{Dice}(j,p)\right]
+0.3L_{positive-score}+0.45L_{negative-score}+1.0L_{box}.
$$

The outer `2` on `L_mask` is hard-coded in the
[return statement](../finetune_image_exemplar_multi_gt.py#L1148). Therefore,
the **effective** coefficients of the matched-mask terms are 4 for BCE and 4
for Dice. A total loss is not an IoU or an error percentage: it is this
weighted sum of quantities with different meanings.

## Inputs to the loss

For each predicted candidate $p$, the model supplies:

| Symbol | Code tensor | Meaning |
| --- | --- | --- |
| $z_p$ | `logits_mhw[p]` | 2-D mask **logits**. `sigmoid(z_p)` converts every pixel to a foreground probability. |
| $b_p$ | `box_preds_n22[p]` | Predicted box; it is reshaped to four normalized coordinates. |
| $s_p$ | `det_scores_logits_n[p]` | Detection/presence **logit**. `sigmoid(s_p)` is a confidence. |
| $g_j$ | `gt_targets[j]` | Binary GT mask for object instance $j$. |

Before the objective runs, GT masks are resized to the model input and then to
the predicted mask resolution, with nearest-neighbor interpolation
([preparation](../finetune_image_exemplar_multi_gt.py#L2425)). Thus every
predicted mask and its target have the same image grid.

## 1. Match masks to ground-truth objects

The objective first chooses which prediction supervises which GT object. It
binarizes a predicted mask at logit zero and a GT mask at 0.5:

$$
\hat g_p=[z_p>0], \qquad g_j=[g_j>0.5].
$$

For every possible pair it calculates hard-mask IoU:

$$
\operatorname{IoU}_{j,p}=
\frac{|g_j\cap\hat g_p|}{|g_j\cup\hat g_p|}.
$$

The [IoU matrix is built here](../finetune_image_exemplar_multi_gt.py#L956).
The code uses `1 - IoU` as cost, sorts all pairs from best to worst, and greedily
accepts them ([assignment loop](../finetune_image_exemplar_multi_gt.py#L1045)).

The matching rules are important:

- A prediction may match only one GT object.
- A GT object may match up to **12** predictions. This is the loss helper's
  default `max_per_gt` ([definition](../finetune_image_exemplar_multi_gt.py#L1089)).
- There is no minimum-IoU cutoff. A poor match may still be selected and is
  trained toward its selected GT.
- This is greedy matching, not a global Hungarian optimum.
- The matching decision is not differentiable: it relies on thresholded masks
  and a detached CPU cost matrix. Gradients flow only through the losses after
  pairs have been chosen.

`--matches_per_gt` controls metric matching but is not passed to this training
loss call ([call site](../finetune_image_exemplar_multi_gt.py#L2436)); the
current loss therefore uses 12 regardless of that CLI setting.

If there are no predictions, no GT masks, or no accepted pairs, the helper
returns `None` ([early exits](../finetune_image_exemplar_multi_gt.py#L1099)).
The caller skips that image instead of including a zero in the batch mean
([handling](../finetune_image_exemplar_multi_gt.py#L2447)).

## 2. Matched mask losses: BCE plus soft Dice

For each selected pair $(j,p)$, the code computes pixelwise BCE directly
from logits:

$$
L_{BCE}(j,p)=\operatorname{mean}_{x,y}
\operatorname{BCEWithLogits}(z_{p,x,y},g_{j,x,y}).
$$

Foreground GT pixels push logits up; background pixels push them down.
`BCEWithLogits` combines sigmoid and cross-entropy stably, so the code does
not apply sigmoid before this calculation.

It also computes soft Dice loss. With $q_p=\sigma(z_p)$:

$$
L_{Dice}(j,p)=1-
\frac{2\sum q_pg_j+10^{-6}}
{\sum q_p+\sum g_j+10^{-6}}.
$$

Dice rewards foreground overlap and is particularly useful when objects occupy
few pixels compared with background. Perfect overlap yields loss near zero;
little overlap yields loss near one. See the
[BCE and Dice implementation](../finetune_image_exemplar_multi_gt.py#L1121).

The matched-pair values are averaged:

$$
L_{mask}=\operatorname{mean}_{(j,p)\in\mathcal M}
\left(w_{bce}L_{BCE}(j,p)+w_{dice}L_{Dice}(j,p)\right).
$$

Consequently, an image with many matches does not automatically have a larger
mask-loss magnitude than an image with one match: it is an average, not a sum.

## 3. Box regression loss

For every matched pair, the GT mask is converted to its enclosing box:

$$
(x_{min},y_{min},x_{max},y_{max}),
$$

with each coordinate normalized to `[0, 1]`. The box loss is mean L1 distance:

$$
L_{box}=\operatorname{mean}_{(j,p)\in\mathcal M}
\lVert b_p-b(g_j)\rVert_1.
$$

The GT-box conversion is in [`loss_fns.py`](../loss_fns.py#L7), and the matched
L1 calculation is in
[`compute_bbox_l1_loss_from_matches`](../loss_fns.py#L26). Empty/invalid masks
cannot provide a box; if no valid matched boxes remain, this loss is zero.

## 4. Presence / detection-score loss

Mask and box losses teach each selected candidate *what to draw*. Presence loss
teaches the model whether it should be confident that a candidate is a real
detection.

All candidates initially receive target score zero. For a matched candidate,
the target is a detached soft quality target:

$$
t_p=\max\left(0.11,
\sigma(s_p)^{0.5}\operatorname{IoU}_{j,p}^{0.5}\right).
$$

This [target construction](../loss_fns.py#L215) has two consequences:

- The minimum `0.11` makes every match a “positive” (`target > 0.1`).
- `.detach()` treats the target as a fixed label during backpropagation. The
  model cannot reduce its target merely by changing its score logit.

Unmatched candidates have target zero. The score loss uses BCE-with-logits,
averaging positive and negative candidates separately:

$$
L_{presence}=w_{score}\operatorname{mean}_{p:t_p>0.1}
\operatorname{BCEWithLogits}(s_p,t_p)
+w_{noobj}\operatorname{mean}_{p:t_p\le0.1}
\operatorname{BCEWithLogits}(s_p,0).
$$

With defaults, the positive group has weight 0.3 and unmatched candidates have
weight 0.45. Averaging each group separately means adding negatives does not
linearly increase the negative loss; it changes that group's mean. Although the
helper supports focal loss, this trainer explicitly disables it
([configuration](../finetune_image_exemplar_multi_gt.py#L1136)).

## 5. Batch loss, gradients, and optimizer updates

The trainer collects all valid image losses in a batch and averages them:

$$
L_{batch}=\operatorname{mean}_{image\in valid\ batch}L_{image}.
$$

It backpropagates this average
([batch construction and backward](../finetune_image_exemplar_multi_gt.py#L2504)).
It calls `optimizer.step()` every `--grad_accum` batches, whose default is 12
([argument](../finetune_image_exemplar_multi_gt.py#L1599),
[step condition](../finetune_image_exemplar_multi_gt.py#L2513)).

One exact implementation detail: `L_batch` is **not divided by** `grad_accum`
before `backward()`. Therefore, each optimizer update uses the **sum** of up to
12 batch-average gradients rather than their average. This changes effective
update scale, but it does not change the per-image objective above.

## Small example

Assume two GT objects (`g1`, `g2`) and five predicted candidates (`p0`–`p4`).
The matcher could select:

$$
\mathcal M=\{(g1,p2),(g1,p4),(g2,p0)\}.
$$

`p0`, `p2`, and `p4` each receive a mask target, a box target, and a soft
positive score target. `p1` and `p3` receive no mask or box term, but their
presence targets are zero, which trains their confidences downward. The three
matched mask losses are averaged, then combined with the presence and box
losses using the final formula above.

`--det_filter` can change which candidates reach the loss, and therefore which
pairs are available, but it is not a direct loss weight
([detection call](../finetune_image_exemplar_multi_gt.py#L2414)).
