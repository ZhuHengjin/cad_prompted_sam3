# CAD Pose Head Implementation Guide

## Status and scope

This document describes the implementation of the CAD-conditioned 6D pose baseline
defined in [the pose-head plan](cad-pose-head-plan.md), using the dataset contract in
[the Perseve pose dataset format](perseve-pose-dataset-format.md).

The implementation adds:

- strict Perseve pose-dataset loading and validation;
- a CAD-dimension-conditioned pose head attached to SAM3 detection tokens;
- symmetry-aware pose losses and deterministic mask-based assignment;
- head-only and joint training stages;
- checkpoint provenance and resume validation;
- validation/test metrics and validation-only score calibration;
- an inference API and command-line example;
- unit tests for geometry, matching, schemas, and dataset joins.

The implementation is intentionally a baseline. It predicts rotation, translation,
and a calibrated pose-quality score, while preserving the existing segmentation-only
behavior and checkpoint compatibility.

## Architecture overview

The data flow is:

```text
RGB image ──> SAM3 image encoder ──> detector/fusion ──> detection tokens
CAD render ─> SAM3 exemplar path ────────────────────────┤
                                                         │
CAD dimensions ──────────────────────────────────────────┤
                                                         v
                                                  CAD pose head
                                                         │
                                    center residual, log depth,
                                    6D rotation, pose-quality logit
                                                         │
camera intrinsics ───────────────────────────────────────┤
                                                         v
                                  rotation matrix + metric translation
```

The segmentation detector remains responsible for masks, boxes, and detection
scores. The pose head consumes each detector candidate before score filtering, then
the same candidate indices are applied to both detection and pose outputs. Detection
confidence and pose confidence remain separate quantities.

## Source map

| Area | Implementation |
| --- | --- |
| Pose head | [`muggled_sam/v3_sam/cad_pose/head.py`](../muggled_sam/v3_sam/cad_pose/head.py) |
| Prediction and target data contracts | [`muggled_sam/v3_sam/cad_pose/types.py`](../muggled_sam/v3_sam/cad_pose/types.py) |
| Camera and rotation geometry | [`muggled_sam/v3_sam/cad_pose/geometry.py`](../muggled_sam/v3_sam/cad_pose/geometry.py) |
| Symmetry-aware rotation error | [`muggled_sam/v3_sam/cad_pose/symmetry.py`](../muggled_sam/v3_sam/cad_pose/symmetry.py) |
| Mask-based one-to-one assignment | [`muggled_sam/v3_sam/cad_pose/matching.py`](../muggled_sam/v3_sam/cad_pose/matching.py) |
| Pose losses | [`muggled_sam/v3_sam/cad_pose/losses.py`](../muggled_sam/v3_sam/cad_pose/losses.py) |
| Evaluation and score calibration | [`muggled_sam/v3_sam/cad_pose/evaluation.py`](../muggled_sam/v3_sam/cad_pose/evaluation.py) |
| Inference selection and formatting | [`muggled_sam/v3_sam/cad_pose/inference.py`](../muggled_sam/v3_sam/cad_pose/inference.py) |
| Perseve dataset loader | [`muggled_sam/v3_sam/cad_pose/dataset.py`](../muggled_sam/v3_sam/cad_pose/dataset.py) |
| SAM3 construction and model integration | [`muggled_sam/v3_sam/make_sam_v3.py`](../muggled_sam/v3_sam/make_sam_v3.py), [`muggled_sam/v3_sam/sam_v3_model.py`](../muggled_sam/v3_sam/sam_v3_model.py) |
| Training, validation, and checkpoints | [`finetune_image_exemplar_multi_gt.py`](../finetune_image_exemplar_multi_gt.py) |
| Dataset manifest integration | [`dataset_manifest.py`](../dataset_manifest.py) |
| Standalone dataset validator | [`validate_perseve_pose_dataset.py`](../validate_perseve_pose_dataset.py) |
| JSON schemas | [`schemas/perseve-pose-v1/`](../schemas/perseve-pose-v1/) |
| Inference example | [`simple_examples/cad_pose_detection.py`](../simple_examples/cad_pose_detection.py) |

## Pose representation and head

### Inputs

`SAMV3CADPoseHead` receives:

- detector candidate tokens with shape `B x N x 256`;
- normalized detector boxes in `xyxy` form with shape `B x N x 2 x 2`;
- metric CAD dimensions with shape `B x 3`;
- an optional reserved CAD-geometry-token argument.

Boxes are converted to normalized `(center_x, center_y, width, height)`. CAD
dimensions are converted to log space and normalized with statistics fitted from the
training split. The normalized box features and dimension features are broadcast over
candidates and concatenated with each 256-dimensional detector token.

The current baseline reserves the geometry-token input for future use but does not
consume it. Conditioning is through the exact metric CAD dimensions required by the
dataset contract.

### Network outputs

Two shared `Linear + LayerNorm + GELU` blocks feed four prediction branches:

| Branch | Size | Meaning |
| --- | ---: | --- |
| Center residual | 2 | Offset from the detector-box center in normalized image coordinates |
| Log depth | 1 | Metric camera-space depth in log form |
| Rotation 6D | 6 | Continuous two-column rotation representation |
| Pose-quality logit | 1 | Confidence that the pose meets the configured tolerances |

There is deliberately no scale or size prediction branch. The object dimensions are
known CAD metadata, and per-instance uniform scale is already part of the dataset
record. Predicting scale again would introduce an avoidable scale/depth ambiguity.

`CADPosePredictions` keeps the raw and derived values together:

- center residual and normalized center;
- log depth;
- 6D rotation and its `3 x 3` rotation matrix;
- pose-quality logits and calibrated probabilities;
- reconstructed translation when intrinsics are available;
- the effective CAD dimensions used for the prediction.

It also provides batch/candidate indexing helpers so detector filtering, NMS, and
matching can apply exactly the same indices to the pose values.

### Rotation conversion

The 6D representation is converted to a proper rotation matrix with Gram-Schmidt
orthogonalization. Degenerate or nearly collinear vectors use deterministic fallback
axes, preventing NaNs while retaining differentiability in normal cases.

### Translation reconstruction

The head predicts image center `(u, v)` and `log(z)`, not Cartesian translation
directly. Metric translation is reconstructed as:

```text
z = exp(log_depth)
[x, y, z]^T = z * K^-1 [u, v, 1]^T
```

The camera solve runs in `float32` for half-precision inputs and converts back
afterward. This avoids unsupported or unstable low-precision linear solves.

## Dataset integration and validation

### Manifest linkage

`dataset_manifest.py` adds a `pose_sample_paths(...)` resolver. It derives the
frame annotation, object catalog, and dataset metadata paths from each existing CSV
manifest row, so the manifest format does not need pose-specific columns.

The loader returns typed records:

- `SymmetryMetadata`;
- `CADCatalogObject`;
- `PoseInstance`;
- `PoseFrame`;
- `PersevePoseSample`.

### Schema version

The implementation supports Perseve pose schema major version 1. Dataset copies of
the JSON schemas are required and are validated against the repository schemas in
[`schemas/perseve-pose-v1/`](../schemas/perseve-pose-v1/):

- `dataset-meta.schema.json`;
- `objects.schema.json`;
- `pose-annotations.schema.json`.

Unknown fields are rejected unless placed in the documented `extensions` objects.
This keeps accidental format drift visible while retaining an explicit extension
mechanism.

### Semantic checks beyond JSON Schema

The loader and standalone validator check relationships that JSON Schema cannot
express by itself:

- intrinsic matrices and image dimensions are valid;
- camera/world and object/camera transforms have the expected shape and valid SO(3)
  rotations;
- discrete symmetry transforms are closed under the declared group;
- continuous symmetry axes are unit length;
- catalog base dimensions equal the declared CAD bounds;
- dimensions and per-instance scale are positive;
- effective dimensions equal `base_dimensions * uniform_scale`;
- pose eligibility agrees with the annotation state, verified symmetry status, and
  valid depth;
- logical-instance images are exact RGBA images, including the alpha channel;
- annotation mapping keys and pixel values agree;
- logical-instance image dimensions match the RGB image;
- every annotated visible mask is nonempty;
- inclusive bounding boxes exactly enclose their visible masks;
- visible-pixel counts exactly match the logical-instance image;
- no nonzero logical-instance value is missing from the annotation mapping;
- repeated occurrences of a CAD object in a scene share the same declared scale.

The supported pose-eligible symmetry states are `verified_auto` and
`verified_manual`. Objects marked `needs_review` are retained for segmentation when
possible but are never used as pose supervision.

### Standalone validation

Validate a manifest and all referenced pose data before training:

```bash
python validate_perseve_pose_dataset.py \
  /path/to/dataset_manifest.csv \
  /path/to/data/root
```

To inspect only selected splits:

```bash
python validate_perseve_pose_dataset.py \
  /path/to/dataset_manifest.csv \
  /path/to/data/root \
  --splits train,validation
```

`--skip_pixels` skips expensive pixel-level joins and should be used only for quick
structural checks. The default performs full validation and prints a JSON report with
record counts and checksums.

## Matching and supervision

### Deterministic assignment

Pose supervision uses a deterministic greedy one-to-one assignment:

1. Compute binary mask IoU for every prediction/ground-truth pair.
2. Exclude pose-ineligible ground-truth instances.
3. Repeatedly select the remaining highest-IoU pair.
4. Remove both the selected prediction and target from further consideration.

Predicted pose values do not participate in the assignment cost. This prevents the
pose head from improving its apparent supervision by changing the matching itself.
Ties are resolved deterministically by the stable candidate/target ordering.

This pose assignment is separate from the detector's existing one-to-many
segmentation loss. The original segmentation loss is unchanged.

### Pose losses

For each matched eligible pair, the total pose loss contains:

- Smooth L1 normalized-center loss;
- Smooth L1 normalized-log-depth loss;
- symmetry-aware rotation loss;
- binary cross-entropy pose-quality loss.

The pose-quality target is soft and detached from the regression graph. It is based
on whether rotation and translation errors fall near configurable tolerances, with
configurable transition widths. Translation error is normalized by the effective
object diagonal by default, so the same setting is meaningful across differently
sized CAD objects. An absolute metric tolerance can be selected explicitly.

The training objective is:

```text
L_total = L_detection + pose_weight * (
    center_weight   * L_center
  + depth_weight    * L_depth
  + rotation_weight * L_rotation
  + quality_weight  * L_quality
)
```

The trainer rejects non-finite pose or total losses instead of silently continuing.

### Symmetry-aware rotation error

Discrete and continuous symmetries use the same implementation in training,
evaluation, and calibration:

- without symmetry, use the SO(3) geodesic angle between prediction and target;
- for discrete symmetry, minimize the angle over all equivalent
  `R_target * S` rotations;
- for continuous axial symmetry, compare the transformed symmetry-axis directions
  rather than penalizing rotation about that axis.

Geodesic angles are computed with an `atan2` formulation for better numerical
behavior near zero and 180 degrees.

## Image geometry and intrinsics

SAM3 square preprocessing resizes the image independently in the horizontal and
vertical directions. It does not letterbox in the current path. Camera intrinsics
must therefore be transformed with separate `scale_x` and `scale_y` factors.

`cad_pose/geometry.py` provides helpers for:

- resize/pad intrinsic adjustment;
- normalized/pixel coordinate conversion;
- camera-space translation reconstruction;
- point projection;
- 6D/matrix rotation conversion.

Pose training disables the random geometric crop because an unrecorded crop would
invalidate the camera model and annotation geometry. Color distortion remains
available. Segmentation-only training keeps the previous augmentation behavior.

The low-level `generate_detections` pose path expects intrinsics already adjusted to
the model input image. The example CLI performs this adjustment.

## SAM3 model integration

### Construction and checkpoint loading

`make_sam_v3.py` constructs the new `SAMV3CADPoseHead` alongside the upstream SAM3
components. Upstream checkpoint loading remains component-specific, so an original
SAM3 checkpoint initializes known upstream modules while leaving the new pose head
randomly initialized.

`SAMV3Model` and `SAMV3DetectorModel` retain and share the same pose-head instance.
The new constructor parameter is optional to preserve existing call sites.

### Detection API

The default `generate_detections` call remains unchanged and returns the original
four values.

Pose inference is enabled with:

```python
boxes, masks, scores, presence, poses = detector.generate_detections(
    encoded_image,
    encoded_prompts,
    cad_dimensions_m_b3=dimensions,
    camera_intrinsics_b33=adjusted_k,
    return_pose=True,
)
```

When `return_pose=True`, both CAD dimensions and camera intrinsics are required. The
pose head runs before detector filtering, and every subsequent detector candidate
index is applied to `CADPosePredictions` as well. Blank-exemplar behavior follows the
same contract.

## Training

### Training stages

Pose training requires both `--enable_pose` and `--dataset_manifest`.

`--pose_stage head`:

- freezes the image encoder, projection, sampling, fusion, detector, and mask
  decoder;
- trains only the pose head;
- requires the training split to contain pose-eligible instances.

`--pose_stage joint`:

- keeps the base image encoder frozen;
- trains the existing fusion/detector/segmentation path and the pose head;
- permits visible, pose-ineligible CAD objects to contribute segmentation
  supervision while excluding them from pose losses.

Without `--enable_pose`, training and return values retain the previous
segmentation-only behavior.

### Preflight

Before optimization, pose training validates all train and validation records,
including schemas, catalog joins, annotations, and pixel masks. It also verifies that
the splits use the same catalog/symmetry pipeline and enforces scene-level scale
sharing.

The preflight pass fits:

- mean and standard deviation of log CAD dimensions;
- mean and standard deviation of log depth.

These training-split statistics replace command-line depth normalization defaults
and are persisted in the checkpoint. A `pose_provenance.json` report is also written
to the run output directory.

### Important CLI options

The pose loss and target can be configured with:

```text
--pose_weight
--pose_center_weight
--pose_depth_weight
--pose_rotation_weight
--pose_quality_weight
--rotation_tolerance_deg
--translation_tolerance
--rotation_soft_width_deg
--translation_soft_width
--absolute_translation_tolerance
```

Run the training script with `--help` for the shared segmentation arguments and
their defaults:

```bash
python finetune_image_exemplar_multi_gt.py --help
```

An illustrative head-only invocation is:

```bash
python finetune_image_exemplar_multi_gt.py \
  --model_path /path/to/sam3.pt \
  --dataset_manifest /path/to/dataset_manifest.csv \
  --data_root /path/to/data/root \
  --enable_pose \
  --pose_stage head \
  --output_dir /path/to/output
```

Use the exact base-model and dataset arguments supported by the local training
script; the command above highlights the pose-specific switches rather than serving
as a dataset-independent recipe.

## Checkpoints and provenance

Pose-enabled checkpoints add the following state to the existing fine-tune
checkpoint:

| Field | Purpose |
| --- | --- |
| `cad_pose_head` | Pose-head parameters and calibrated temperature buffer |
| optimizer state | Optimizer/scheduler continuation for a compatible training stage |
| `pose_config` | Depth normalization statistics, loss weights, and tolerances |
| `args` | Full CLI configuration, including the selected pose stage |
| dataset metadata checksum | Detects dataset-level contract changes |
| catalog checksum | Detects CAD dimension or symmetry changes |
| aggregate annotation checksum | Detects pose-label changes |
| schema checksums and versions | Records the exact schema inputs and supported format version |
| symmetry-pipeline versions | Ensures symmetry interpretation is reproducible |

Resume verifies catalog checksums, dataset-metadata checksums, symmetry-pipeline
versions, and schema versions against the current dataset before continuing. The
annotation and schema checksums are also retained for auditing, but are not currently
resume-blocking comparisons. Optimizer state is restored only when compatible with
the current trainable parameters; otherwise the script warns and starts a new
optimizer. Older segmentation checkpoints without `cad_pose_head` remain valid and
initialize a new pose head.

The final validation calibration is saved as `finetune_calibrated.pth` so inference
and test evaluation use the fitted pose-score temperature.

## Evaluation and calibration

Evaluation reports:

- symmetry-aware rotation error in degrees;
- translation error in centimeters;
- normalized center error;
- metric depth error;
- 5-degree/5-centimeter accuracy;
- 10-degree/10-centimeter accuracy;
- Brier score;
- expected calibration error.

Evaluation uses the same detection filtering, NMS candidate indices, and one-to-one
mask assignment as the training/inference path.

Pose-quality temperature is fitted with scalar-temperature optimization on the
validation split only. Periodic validation evaluates the current temperature. Final
validation fits and saves the calibrated temperature. Test evaluation never fits on
test labels; it uses the checkpoint temperature.

Optional 3D IoU, ADD/ADD-S, and VSD metrics from the broader plan are not implemented
in this baseline.

## Inference

The helper functions in `cad_pose/inference.py` provide:

- mask NMS indices;
- candidate selection that keeps detection and pose tensors synchronized;
- result formatting.

Each formatted result includes:

```text
mask_logits
box_xyxy
detection_score
rotation_matrix
translation_m
cad_dimensions_m
pose_score
cad_id
```

Run the end-to-end example with a base model, pose checkpoint, RGB image, CAD render,
CAD identifier, metric dimensions, and camera matrix:

```bash
python simple_examples/cad_pose_detection.py \
  /path/to/sam3.pt \
  /path/to/finetune_calibrated.pth \
  /path/to/image.png \
  /path/to/cad_render.png \
  example_cad_id \
  --dimensions_m 0.12 0.08 0.04 \
  --camera_k fx 0 cx 0 fy cy 0 0 1 \
  --output pose_results.json
```

Replace `fx`, `fy`, `cx`, and `cy` with numeric values. Optional arguments control
the render box, score threshold, mask-NMS threshold, maximum detections, device, and
output path. The JSON output omits dense mask arrays but includes the rotation,
translation, dimensions, detection score, pose score, and CAD identifier.

## Tests and verification

The added tests cover:

- 6D rotation conversion and degenerate fallbacks;
- metric translation reconstruction and projection;
- resize-adjusted camera intrinsics;
- absence of a learned scale branch;
- discrete and continuous symmetry errors;
- deterministic one-to-one mask matching;
- JSON schema acceptance/rejection;
- exact RGBA logical-instance joins and inclusive bounding boxes;
- manifest pose-path parsing.

Relevant test modules are:

- [`tests/test_cad_pose/geometry.py`](../tests/test_cad_pose/geometry.py);
- [`tests/test_perseve_pose_dataset.py`](../tests/test_perseve_pose_dataset.py);
- [`tests/test_perseve_pose_schema.py`](../tests/test_perseve_pose_schema.py);
- [`tests/test_dataset_manifest.py`](../tests/test_dataset_manifest.py).

Run the complete unit-test discovery with:

```bash
python -m unittest discover -s tests -v
```

The implementation was also checked with Python bytecode compilation, JSON schema
parsing, `git diff --check`, and `uv lock --check`.

In the implementation environment, 15 tests were discovered: 7 passed and 8
tensor/image tests were skipped because the system Python lacked PyTorch and OpenCV.
Those skipped tests must be run in the project's full training environment before
relying on the GPU/runtime path.

`jsonschema` was added to `pyproject.toml`, `requirements.txt`, and `uv.lock` for
schema validation.

## Backward compatibility

The implementation preserves the following existing contracts:

- segmentation-only training does not require pose annotations;
- the default detection API still returns four values;
- original SAM3 checkpoints can be loaded without pose-head keys;
- existing constructor call sites can omit the pose head;
- segmentation loss semantics are unchanged;
- segmentation-only augmentation retains its previous geometric behavior.

The only intentional stricter behavior is inside pose-enabled workflows, where
camera geometry, CAD metadata, schemas, and pixel joins must be reproducible and
internally consistent.

## Current limitations and next steps

The following plan items remain future work:

- CAD mesh/point-cloud geometry tokens;
- detector ROI feature pooling;
- CAD-render viewpoint embeddings;
- reprojection or silhouette losses;
- Hungarian matching with additional non-pose costs;
- learned scale prediction, which is intentionally excluded from the current data
  contract;
- 3D IoU, ADD/ADD-S, and VSD evaluation;
- BOP-format export;
- dataset generation and symmetry labeling, which remain responsibilities of the
  Perseve data pipeline.

The public low-level pose API requires camera intrinsics already transformed to the
model input coordinates. The example CLI handles this, but custom callers must do the
same.

Finally, the repository-level checks do not replace an end-to-end run with real
Perseve data, a SAM3 checkpoint, PyTorch/OpenCV, and a GPU. That integration run is
the remaining operational validation step.
