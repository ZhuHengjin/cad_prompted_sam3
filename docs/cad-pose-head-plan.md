# CAD-Prompted SAM3 Pose-Head Plan

## Goal

Extend CAD-Prompted SAM3 from exemplar-conditioned instance segmentation to
joint instance segmentation and monocular 6-DoF pose estimation. For each
retained CAD-conditioned instance, the model should return:

- an instance mask and 2-D box;
- a detection confidence;
- a 3-D rotation $R$;
- a metric translation $t$;
- a pose-quality confidence.

The implementation will borrow YOPO's compact pose representation while using
SAM3's existing exemplar-conditioned detection tokens as object queries. The
existing segmentation path remains intact.

## Architectural decision

The output of SAMV3ExemplarDetector is the correct integration point. Its
detection_tokens_bnc tensor has shape $B\times N\times256$, with one
exemplar-conditioned token for every candidate detection. These tokens already
drive box scoring and mask generation in
[exemplar_detector_model.py](../muggled_sam/v3_sam/exemplar_detector_model.py#L92)
and
[finetune_image_exemplar_multi_gt.py](../finetune_image_exemplar_multi_gt.py#L730).

The initial architecture is:

```text
RGB image ──> SAM3 image encoder/projection ──┐
                                              ├─> image-exemplar fusion
CAD render exemplars ─> exemplar encoding  ───┘
                         │
                         v
                 SAM3 exemplar detector
                         │
              detection tokens + boxes
                    /                \
                   v                  v
       existing segmentation       new pose head <── effective dimensions
                                           ^
                                           └──────── adjusted camera intrinsics
                   │                  │
                   └──── mask + box + R + t + pose score
```

The pose head must run on all unfiltered detection tokens during training.
Detection thresholding and NMS happen afterward, and the same retained indices
must be applied to tokens, boxes, masks, scores, and every pose output.

## Pose representation

Follow YOPO's decomposed representation instead of directly regressing three
translation coordinates:

| Quantity | Output dimension | Representation |
| --- | ---: | --- |
| Projected object center | 2 | box-relative residual from the predicted box center |
| Depth | 1 | log-depth |
| Rotation | 6 | continuous 6-D rotation representation |
| Pose quality | 1 | calibrated probability that the pose meets declared task tolerances |

Given adjusted camera intrinsics $K$ and predicted center $(u,v)$,
reconstruct translation as

$$
z=\exp(\hat z),\qquad
t=zK^{-1}[u,v,1]^T.
$$

Convert the 6-D rotation output $(a_1,a_2)$ to a valid rotation matrix using
Gram-Schmidt orthogonalization:

$$
r_1=\operatorname{normalize}(a_1),
$$

$$
r_2=\operatorname{normalize}(a_2-(r_1^Ta_2)r_1),
\qquad r_3=r_1\times r_2,
$$

$$
R=[r_1\;r_2\;r_3].
$$

YOPO predicts an unconstrained category-level 3-D size. CAD-Prompted SAM3
instead receives the prompted instance's known effective metric dimensions and
does not predict object size or scale. If the catalog dimensions are $d_{CAD}$
and Perseve applies rendering scale $s_{render}$, the required prompt input is

$$
d_{prompt}=d_{CAD}\odot s_{render}.
$$

The loader supplies `dimensions_m = d_prompt` during training, and the caller
supplies the real prompted object's dimensions during inference. These
dimensions are a pose-head input and metric cue, not a predicted output. This
plan does not include a scale branch, scale residual, size loss, or a separate
metric generation mode.

## Pose-head module

Add a SAMV3CADPoseHead module with the initial interface:

```python
pose_predictions = pose_head(
    detection_tokens_bnc,
    boxes_xy1xy2_bn22,
    cad_dimensions_m_b3,
    adjusted_camera_intrinsics_b33,
    model_image_size_wh,
    cad_geometry_tokens_bkc=None,
)
```

The first version should remain small:

1. Convert each box to normalized (cx, cy, w, h).
2. Decompose `log(cad_dimensions_m_b3)` into scale-free aspect ratios and
   absolute log dimensions normalized with training-split statistics.
3. Concatenate the detection token, box features, and scale-free aspect ratios,
   then pass them through a shared two-layer MLP with LayerNorm and GELU.
4. Predict the center residual in detector-box width/height units:
   $(u,v)=(c_x,c_y)+(\Delta_xw,\Delta_yh)$. Absolute metric dimensions and
   camera intrinsics must not enter this branch.
5. Normalize the already resize-adjusted camera intrinsics into model-image
   coordinates. Derive log focal lengths, principal point, the candidate center
   ray, and angular box extent.
6. Fuse the shared candidate representation, normalized absolute log
   dimensions, and camera features in a depth-specific MLP before predicting
   log-depth.
7. Predict 6-D rotation and pose confidence from the shared scale-invariant
   candidate representation.

This routing encodes the intended invariances. Uniformly changing the supplied
physical size may change metric depth but cannot move the projected center or
change rotation. Changing focal length may change depth through the angular-size
relationship, while the center remains an image-space prediction. CAD aspect
ratios remain available to center and rotation because canonical geometry can
affect the offset between the visible box center and projected CAD origin.

The dimension input is required in the baseline. It supplies the physical-size
cue needed to disambiguate monocular translation under Perseve's random object
scaling. It must not be silently replaced with catalog base dimensions.

After establishing the baseline, add two optional inputs:

- ROI-pooled features from the exemplar-fused image feature map inside each
  predicted box;
- a CAD geometry token containing canonical axes, symmetry, and optionally a
  learned point-cloud or mesh encoding. Effective metric dimensions remain the
  separate required input above.

Rendered exemplar poses are known during rendering. A later improvement should
embed each render's camera-to-CAD viewpoint and add it to its exemplar tokens.
This gives the pose head explicit correspondence between exemplar appearance
and CAD orientation.

## Canonical frames and symmetric objects

Every CAD model retains one canonical object frame. It defines mesh
coordinates, dimensions, transforms, grasp points, render viewpoints, and a
stable annotation format. Canonical storage must not imply canonical-only
supervision when the object is physically symmetric.

Store the canonical pose $R_{gt}$ together with the object's rotational
symmetry group $G$. If $S\in G$ maps the object onto itself, and the pose
maps object coordinates into camera coordinates, all these rotations are valid:

$$
\mathcal R_{valid}=\{R_{gt}S\mid S\in G\}.
$$

Use a symmetry-aware geodesic loss:

$$
L_R(R_{pred},R_{gt})=
\min_{S\in G}d_{SO(3)}(R_{pred},R_{gt}S).
$$

The same symmetry definition must be used by assignment, final rotation loss,
auxiliary losses, and evaluation. YOPO's released code considers a fixed
180-degree alternative for selected classes in its
[rotation matching cost](../../YOPO/yopo/models/task_modules/assigners/match_cost.py#L1024),
but its final
[rotation loss](../../YOPO/yopo/models/losses/pose_loss.py#L975) compares with
one target rotation. This implementation will make the treatment consistent.

| Symmetry | Supervision |
| --- | --- |
| Asymmetric | $G=\{I\}$; supervise the canonical orientation |
| Discrete $n$-fold | enumerate the $n$ proper rotations and minimize over them |
| 180-degree | minimize over identity and the specified object-frame 180-degree rotation |
| Continuous axial | supervise the symmetry-axis direction and ignore rotation about it |
| Geometry symmetric but visibly marked | treat as asymmetric when appearance reliably breaks symmetry |
| Geometry symmetric but task distinguishes sides | use the task-defined frame and symmetry group |

For continuous axial symmetry, use an analytic axis loss rather than dense
angle sampling. Reflectional symmetry is not automatically a valid rotation:
include only proper rotations in $SO(3)$ unless the representation explicitly
supports reflections.

The object catalog is populated by the automatic symmetry-labeling pipeline
defined in the dataset-format specification. Rotation supervision is enabled
only for catalog labels with status `verified_auto` or `verified_manual`;
`needs_review` objects may still contribute segmentation supervision but are
excluded from pose matching, pose losses, and pose evaluation. Manual
appearance- or task-based overrides take precedence over a
geometry-only result. Training, assignment, evaluation, and checkpoint
provenance must all use the same catalog checksum and symmetry-pipeline version.

## Dataset contract

[Perseve pose-label and dataset format](perseve-pose-dataset-format.md) is the
normative specification for pose sidecars, object catalogs, coordinate
conventions, generator behavior, and validation. This plan consumes its
versioned per-frame annotations through the existing CSV manifest.

For each training instance, the loader must provide the format-defined:

- stable `cad_id`, explicit annotation state, pose-training eligibility, and
  exact logical-RGBA instance-mask join;
- camera intrinsics $K$ and canonical object-to-camera transform
  `T_cam_from_cad`;
- catalog `base_dimensions_m`, per-instance `render_scale_xyz`, and required
  effective `dimensions_m` satisfying
  $d_{prompt}=d_{CAD}\odot s_{render}$;
- source-unit conversion, rigid source-to-canonical transform, and canonical
  axes;
- symmetry metadata with verification status and pipeline provenance; and
- available visibility information.

The pose head assumes the format's OpenCV camera frame, metres, column-vector
transform action, top-left-pixel-center convention, logical RGBA masks, and
inclusive integer `[x_min, y_min, x_max, y_max]` boxes derived from those masks.
It predicts no object size or scale. Every placed object remains represented in
the sidecar, including fully occluded, out-of-frame, invalid-geometry, and
capture-error states, but only eligible visible entries enter pose matching.
Validate all sidecars and catalogs against the published JSON Schemas and the
CSV manifest before training. Store the schema version, catalog checksum,
annotation checksum, and symmetry-pipeline version with run provenance.

## Matching strategy

The current segmentation objective uses greedy one-to-many mask-IoU matching,
with up to 12 predictions assigned to one GT object. Preserve this initially
for mask, box, and presence training, as documented in
[current-training-loss.md](current-training-loss.md).

Pose supervision should use one prediction per eligible visible GT instance.
Entries whose `pose_training_eligible` flag is false are retained for audit and
dataset diagnostics but excluded from pose assignment and losses:

1. Compute the existing mask-IoU matrix.
2. Select the highest-IoU unique prediction for each GT, or run one-to-one
   Hungarian assignment using mask and box costs.
3. Apply pose losses only to those one-to-one pairs.
4. Continue applying the existing segmentation losses to the one-to-many pairs.

Do not use predicted pose in the matching cost at the start of training. Once
pose predictions become useful, optionally add low-weight translation and
symmetry-aware rotation costs to Hungarian assignment. Assignment and rotation
loss must use the same symmetry group.

## Training objective

Retain the existing objective:

$$
L_{det}=2L_{mask}+L_{presence}+w_{bbox}L_{box}.
$$

For one-to-one matched pose pairs, add:

$$
L_{pose}=
\lambda_{uv}L_{uv}+
\lambda_zL_{logz}+
\lambda_RL_R+
\lambda_{proj}L_{proj}+
\lambda_qL_{quality}.
$$

The components are:

- $L_{uv}$: Smooth-L1 on normalized projected centers;
- $L_{logz}$: Smooth-L1 on normalized log-depth;
- $L_R$: symmetry-aware geodesic or continuous-axis loss;
- $L_{proj}$: optional projected CAD-corner or rendered-silhouette loss;
- $L_{quality}$: BCE-with-logits calibration of pose confidence against a detached soft pose-quality target.

For each one-to-one pose match, compute the same symmetry-aware rotation error
used by $L_R$, denoted $e_R$, and metric translation error $e_t$. Let
$d_{prompt}$ be the bounding-box diagonal computed from the supplied effective
`dimensions_m` and define normalized translation error
$\tilde e_t=e_t/d_{prompt}$.

The confidence target is the soft probability that the pose meets the declared
task-success tolerances:

$$
q_{pose}^{*}=\sigma\left(\frac{\theta_R-e_R}{\delta_R}\right)
\sigma\left(\frac{\theta_t-\tilde e_t}{\delta_t}\right).
$$

$\theta_R$ and $\theta_t$ are the accepted rotation and normalized
translation errors for the deployment task. $\delta_R$ and $\delta_t$
are positive soft-boundary widths; they make near-threshold poses receive
intermediate targets instead of a brittle binary label. Declare all four values
in the training configuration before fitting. If the task uses an absolute
translation tolerance, use $e_t$ and a metre-valued $\theta_t$ instead of
normalizing by $d_{prompt}$.

If $\hat q$ is the pose-score logit, train it directly with

$$
L_{quality}=\operatorname{BCEWithLogits}(\hat q,\operatorname{detach}(q_{pose}^{*})).
$$

This target deliberately does not depend on the current pose-score prediction:
it is already a ground-truth-derived quality label. The score predicts the
probability that $R,t$ meet the declared tolerance for an already matched
detection; it is distinct from `detection_score`, which measures detection and
mask quality. For continuous axial symmetries, use the same axis error as
$L_R$ for $e_R$.

Fit a scalar temperature $T_{cal}$ on a held-out validation split after
training and report $pose_score=\sigma(\hat q/T_{cal})$. Never fit this
calibration temperature on the test split.

The combined objective is

$$
L_{total}=L_{det}+\lambda_{pose}L_{pose}.
$$

Normalize depth targets using training-set statistics instead of copying YOPO's
raw loss weights. Begin without reprojection,
log every component separately, and choose weights so no branch dominates the
shared detection-token gradients.

## Training stages

### Stage 0: data and geometry validation

- Validate dataset metadata, object catalogs, and every frame sidecar against
  the versioned JSON Schemas.
- Round-trip canonical CAD points through every annotated pose.
- Reproject object centers and cuboid corners with adjusted intrinsics.
- Verify normalized intrinsics, center rays, and angular box extents after every
  supported model resize.
- Overlay projected CAD geometry on RGB images and instance masks.
- Recompute logical-RGBA mask joins and inclusive mask-derived boxes.
- Check source-unit conversion, rigid source-to-canonical transforms, and
  `dimensions_m = base_dimensions_m * render_scale_xyz`.
- Verify repeated `(scene_id, cad_id)` entries share scale and effective
  dimensions within a scene.
- Verify automatic discrete and continuous symmetry labels, including their
  review status, with the exact checksummed canonical meshes.

### Stage 1: pose-head baseline

- Load the best CAD-Prompted SAM3 segmentation checkpoint.
- Freeze the image encoder, image projection, fusion, detector, and mask head.
- Train only the new pose head from fixed detection tokens.
- Use one-to-one mask-based pose matching.
- Consume the supplied effective prompted dimensions as a required pose-head
  input.
- Condition depth on normalized adjusted intrinsics and angular box extent.
- Keep center offsets box-relative and invariant to uniform metric rescaling.
- Train the pose-confidence branch with the detached soft pose-quality target.

### Stage 2: detector adaptation

- Unfreeze the final exemplar-detector layers and image-exemplar fusion.
- Train segmentation and pose jointly, using a lower learning rate for
  pretrained modules.
- Monitor mask PQ and IoU so pose gradients do not degrade segmentation.

### Stage 3: geometry refinement

- Add CAD tokens and/or ROI-pooled fused image features.
- Add symmetry-aware reprojection or silhouette consistency.

## Inference contract

For every retained instance, return:

```python
{
    "mask_logits": ...,
    "box_xyxy": ...,
    "detection_score": ...,
    "rotation_matrix": ...,
    "translation_m": ...,
    "cad_dimensions_m": ...,
    "pose_score": ...,
    "cad_id": ...,
}
```

The caller must provide the effective physical dimensions for the prompted
object. The returned `cad_dimensions_m` echoes this input; it is not a model
prediction.

Pose prediction happens before filtering. NMS and thresholding must preserve
candidate indices. Translation is reconstructed using intrinsics adjusted for
the model's actual input geometry and reported in the declared camera
coordinate system. The same adjusted intrinsics are normalized inside the pose
head to condition log-depth on focal length and angular box extent.
`pose_score` is the validation-calibrated probability that the reported pose
meets the declared task tolerances; it is separate from `detection_score`. A
canonicalized rotation may be provided for display, but
the raw pose and symmetry metadata remain available to downstream code.

## Evaluation

Report segmentation and pose performance together:

- current mask IoU, PQ, and box metrics;
- symmetry-aware rotation error in degrees;
- translation error in centimetres;
- depth and projected-center error;
- 5-degree/5-centimetre and 10-degree/10-centimetre accuracy;
- pose-score calibration against the declared pose-success target (reliability curve, Brier score, and expected calibration error);
- 3-D IoU using CAD dimensions;
- group-aware ADD/ADD-S or VSD where appropriate;
- per-CAD and per-symmetry-type breakdowns;
- segmentation metrics before and after pose supervision.

Model selection uses a declared validation score rather than test results.
Preserve an untouched test split following the existing manifest workflow.

## Implementation roadmap

1. Add muggled_sam/v3_sam/cad_pose/head.py with typed prediction outputs,
   branch-specific effective-dimension and intrinsic conditioning, box-relative
   center prediction, and 6-D rotation conversion.
2. Add geometry and symmetry utilities with tests for rotation equivalence,
   translation reconstruction, projection, and continuous-axis losses.
3. Register the pose head in the SAM3 detector wrapper and construction path.
   Initialize it separately because upstream SAM3 checkpoints contain no pose
   weights.
4. Extend the exemplar detection helper to expose detection tokens and pose
   predictions without changing existing callers by default.
5. Add a versioned pose-annotation sidecar loader, JSON Schema validation,
   state/eligibility filtering, RGBA mask joins, inclusive-box checks, and
   effective-dimension validation.
6. Add one-to-one pose matching beside existing one-to-many segmentation
   matching.
7. Add pose losses, component logging, visualization overlays, and NaN checks
   to the multi-GT trainer.
8. Save and restore pose_head, pose configuration, symmetry metadata version,
   annotation checksum, and optimizer state in checkpoints.
9. Add pose evaluation and an inference example that exports one result per
   detected instance.
10. Establish the frozen-head baseline before adding CAD tokens, reprojection,
    or broader joint fine-tuning.

## First-milestone acceptance criteria

The baseline is complete when:

- existing segmentation inference and checkpoints remain usable;
- a batch produces finite center, depth, rotation, and pose-confidence outputs
  for every detection token;
- uniformly changing the supplied effective dimensions changes depth
  conditioning but leaves center and rotation outputs unchanged, while no scale
  or size prediction branch exists;
- changing adjusted focal length changes depth conditioning but not the
  image-space center branch;
- center residuals are expressed in predicted-box width/height units;
- one-to-one pose matching aligns each GT instance with the correct mask token;
- schema, logical-RGBA mask, inclusive-box, scale-sharing, and effective-
  dimension validation pass for generated training data;
- projection tests pass under model resize and padding;
- equivalent symmetric rotations yield identical loss and evaluation error;
- unresolved automatic symmetry labels are excluded from rotation supervision;
- asymmetric rotations retain ordinary geodesic supervision;
- checkpoint resume restores the pose head and optimizer state;
- validation reports segmentation and pose metrics without using the test split;
- pose-score calibration is measured on a held-out validation split.
