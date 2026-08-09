# Ground-Truth Loss Validation

## Conclusion

The active CAD pose and joint detection loss implementations pass a
ground-truth-as-prediction sanity check.

The checks construct predictions directly from ground truth (GT), call the
production loss helpers, and verify that:

- geometric and regression losses are exactly zero at GT;
- finite-logit binary classification losses are numerically near zero;
- the soft pose-quality BCE reaches its correct nonzero theoretical minimum;
- independently corrupting center, depth, rotation, mask, box, or confidence
  predictions increases the corresponding loss;
- a Perseve-v2 annotation survives production target construction and camera
  back-projection before reaching zero geometric pose loss.

As of 2026-08-08, the three focused GT checks pass, the surrounding focused
suite passes all 27 tests, and the repository suite passes all 96 tests. No
error was found in the active pose or joint detection objectives by these
checks.

## What this validates

These are loss-level oracle checks. They do **not** run an image through the
neural network and expect the trained model to reproduce GT. Instead, they
construct the output tensors that a perfect model should produce and pass them
directly to the production loss functions.

This distinction separates two questions:

1. **Is the loss implementation internally consistent?** These tests check
   this.
2. **Can the model learn to predict GT from images?** Training and evaluation
   experiments must answer this separately.

The checks cover three paths:

| Check | Production path exercised | Main contract |
| --- | --- | --- |
| Pose loss | `compute_cad_pose_losses` | GT geometry reaches zero geometric loss and optimal quality BCE. |
| Joint detection loss | `compute_multi_gt_detection_losses` | GT masks, boxes, and presence labels minimize the combined objective. |
| Dataset round trip | `load_perseve_pose_sample` → `make_pose_target` → `CADPosePredictions.with_translation` → `compute_cad_pose_losses` | Dataset coordinates and camera reconstruction agree with the pose loss. |

## Implemented tests

### Pose loss at ground truth

Implementation:
[`test_ground_truth_pose_predictions_reach_expected_loss_minimum`](../tests/test_cad_pose_geometry.py)

The test creates one point-set pose target containing a normalized image
center, log depth, rotation, translation, surface centroid, dimensions, and
canonical CAD points. It constructs one prediction by copying the corresponding
GT values and supplies the explicit match `(gt_index=0, prediction_index=0)`.

For this perfect prediction, it requires:

| Component | Expected result |
| --- | ---: |
| Center Smooth-L1 | `0` |
| Normalized log-depth Smooth-L1 | `0` |
| Point-set rotation Smooth-L1 | `0` |
| Full-pose point-set Smooth-L1 | `0` |
| Translation error | `0` |
| Normalized full point-set error | `0` |
| Surface-centroid error | `0` |

The test then changes one prediction field at a time:

- shifting the center must increase center and total loss;
- shifting log depth must increase depth and total loss;
- rotating the prediction must increase rotation, full-pose, and total loss;
- unaffected loss components must remain unchanged where the constructed
  perturbation is intentionally isolated.

### Why the perfect pose total is not zero

Pose quality uses a soft target rather than a hard label of one. For exact
point-set geometry, both normalized geometric errors are zero, so the default
quality target is

$$
q = \sigma(0.1 / 0.02)\,\sigma(0.1 / 0.02)
  = \sigma(5)^2
  \approx 0.986659.
$$

The BCE-with-logits minimum occurs when the predicted probability equals this
target. The test therefore supplies

$$
s = \operatorname{logit}(q) \approx 4.30349
$$

and independently evaluates the expected BCE. Its minimum is the binary
entropy of the soft target:

$$
\operatorname{BCEWithLogits}(s,q) \approx 0.070843.
$$

Thus, at perfect GT geometry:

```text
geometric pose terms = 0
quality BCE           ≈ 0.070843
total pose loss       ≈ 0.070843
```

Asserting that the total were zero would incorrectly reject the implemented
soft-quality objective.

### Joint detection loss at ground truth

Implementation:
[`test_ground_truth_detection_predictions_minimize_joint_loss`](../tests/test_multi_gt_loss.py)

The test builds two binary GT masks and three prediction candidates. The first
two candidates reproduce the two GT masks and boxes; the third is a correct
unmatched/background candidate. It limits matching to one prediction per GT so
the expected positive and negative assignments are unambiguous.

Binary masks cannot be passed directly as logits. Values of zero and one would
represent probabilities of `0.5` and approximately `0.731`, respectively.
The test instead constructs near-perfect finite logits:

```text
GT foreground pixel → prediction logit +20
GT background pixel → prediction logit -20
matched candidate   → presence logit +20
unmatched candidate → presence logit -20
```

The normalized predicted boxes exactly equal the boxes derived from the GT
masks. The required results are:

| Component | Acceptance condition |
| --- | ---: |
| Mask BCE plus Dice | `< 1e-6` |
| Box L1 | exactly `0` |
| Presence BCE | `< 1e-6` |
| Combined joint loss | `< 3e-6` |

Mask and presence BCE approach zero but are not mathematically zero because
finite logits cannot represent probabilities of exactly zero and one.

The test then introduces three independent errors:

- one foreground mask pixel is changed to a background prediction;
- one normalized box coordinate is shifted;
- one matched candidate is assigned a strongly negative presence logit.

The mask, box, and objectness losses, respectively, must increase. Components
not affected by a perturbation are also checked for equality where appropriate.

### Perseve-v2 target and camera round trip

Implementation:
[`test_v2_point_set_derives_surface_centroid_target`](../tests/test_perseve_pose_dataset.py)

This test creates a complete temporary Perseve-v2 fixture, including its
annotation, camera intrinsics, CAD catalog, and sampled CAD point set. It then:

1. loads the fixture with `load_perseve_pose_sample`;
2. creates the training target with `make_pose_target`;
3. copies the GT center, depth, and rotation into a pose prediction;
4. reconstructs its centroid and public translation using
   `CADPosePredictions.with_translation`;
5. evaluates `compute_cad_pose_losses` with quality supervision disabled.

The center, depth, rotation, translation error, and total loss must all be
zero. This test covers coordinate normalization, surface-centroid handling,
camera back-projection, and the public AABB-origin translation convention. It
uses a self-contained realistic fixture rather than requiring an external
dataset checkout.

## Running the checks

Run commands from the repository root:

```bash
cd /path/to/cad_prompted_sam3
```

Run only the three GT checks:

```bash
uv run python -m unittest -v \
  tests.test_cad_pose_geometry.CADPoseGeometryTests.test_ground_truth_pose_predictions_reach_expected_loss_minimum \
  tests.test_multi_gt_loss.MultiGtLossEquivalenceTests.test_ground_truth_detection_predictions_minimize_joint_loss \
  tests.test_perseve_pose_dataset.PersevePoseDatasetTests.test_v2_point_set_derives_surface_centroid_target
```

Expected summary:

```text
Ran 3 tests
OK
```

Run the surrounding loss, geometry, and dataset tests:

```bash
uv run python -m unittest -v \
  tests.test_cad_pose_geometry \
  tests.test_multi_gt_loss \
  tests.test_perseve_pose_dataset
```

Expected summary for the current tree:

```text
Ran 27 tests
OK
```

Run the entire repository suite:

```bash
uv run python -m unittest discover -s tests -v
```

Expected summary for the current tree:

```text
Ran 96 tests
OK
```

## Interpreting results

Each test should end in `... ok`, followed by an `OK` summary. That result means
all exact, tolerance-based, and monotonicity assertions described above held.

A regression produces `FAILED` and reports the failed assertion:

- a nonzero geometric term at GT suggests a target representation, coordinate,
  matching, or formula mismatch;
- a mask or objectness value above tolerance suggests incorrect logit/BCE/Dice
  behavior;
- a nonzero GT box loss suggests box ordering or normalization disagreement;
- a perturbation that fails to increase its loss suggests the component may be
  disconnected, incorrectly weighted, or reading the wrong tensor;
- failure only in the dataset round trip suggests the pure loss may be correct
  while preprocessing, intrinsics, centroid conversion, or translation
  reconstruction disagrees with it.

## Limitations and follow-up

These checks establish strong local invariants, but they do not prove every
aspect of training correctness. In particular, they do not:

- run a full neural-network forward pass;
- demonstrate that optimization can reach the constructed predictions;
- validate performance on an external real dataset;
- cover every empty-input, invalid-match, symmetry, batching, or mixed-precision
  edge case;
- replace gradient-flow, training-smoke, convergence, or held-out evaluation
  tests.

Existing tests separately cover symmetry handling, finite pose gradients,
one-to-one matching, evaluation, and refactoring equivalence. A useful future
extension would run the same round-trip contract over a small fixed sample from
each production dataset variant.

During inspection, an unused legacy helper,
`loss_fns.compute_score_weighted_mask_loss`, was found to reference an undefined
name, `scores_probs`. The current multi-GT training objective does not call this
helper, so it does not affect the conclusion above. Its intended
logits-versus-probabilities input contract should be decided before repairing
and testing it separately.
