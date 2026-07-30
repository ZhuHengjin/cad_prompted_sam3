# Label-Free Point-Set Pose Supervision Plan

## Status

This document defines the implemented replacement for explicit
rotational-symmetry labels in the CAD-pose training pipeline. Pose schema v2
uses deterministic point-set artifacts and surface-centroid targets. The
schema-v1 symmetry path remains loadable for old datasets and checkpoints, but
new data and training should use v2.

The target design:

- keeps the canonical CAD frame at the local AABB center;
- does not run a symmetry-labeling pipeline;
- derives a deterministic surface point set and surface centroid from each
  canonical CAD mesh;
- supervises rotation by matching transformed point sets;
- supervises translation through the camera-space surface centroid;
- converts the predicted centroid pose back to the existing AABB-origin pose
  for storage and inference; and
- evaluates geometric placement without requiring a declared symmetry type.

## Why the AABB frame remains canonical

The AABB-centered frame is retained for dataset storage, rendering, debugging,
CAD overlays, dimensions, grasp metadata, and public inference results. It is
stable and easy for humans and downstream systems to interpret.

Moving only a coordinate-frame origin would not change its rotation matrix, but
it would change every translation label and the meaning of the projected-center
branch. Moving the dataset frame to a symmetry center would also make data
preparation depend on the output of a separate labeling pipeline. The active
design avoids that dependency.

The rotation matrix itself has no rotation origin: it still describes the
orientation of the canonical CAD axes. Using the surface centroid internally
does not ask the network to discover a symmetry center from pixels. The
centroid is supplied CAD metadata, and the network predicts where that known
point lies in the camera frame.

The loss may use a different internal anchor without changing the public CAD
frame. In particular, it uses the mesh's surface centroid because that point is
fixed by every exact rigid symmetry of the surface.

## Canonical point-set artifact

For every `cad_id`, preprocess the exact canonical metric render mesh into a
deterministic surface point-set artifact. Points are expressed in metres in the
existing AABB-centered CAD frame before scene-level rendering scale.

The v2 object-catalog contract provides:

```json
{
  "point_set": {
    "path": "cad_points/<cad_id>.npz",
    "sha256": "<sha256>",
    "point_count": 4096,
    "sampling_method": "surface_area_deterministic_v1",
    "sampling_parameters_sha256": "<sha256>",
    "surface_centroid_m": [0.0012, -0.0004, 0.0031]
  }
}
```

The artifact should contain at least:

- `points_m`: a dense `N x 3` finite floating-point target set sampled
  uniformly by triangle surface area;
- `query_indices`: a deterministic integer subset of `points_m`, or enough
  metadata to reproduce that subset;
- the surface centroid in the canonical AABB frame; and
- optional normals for future point-to-plane or surface-consistency losses.

Create an artifact from an STL or composed USD render mesh with:

```bash
python preprocess_cad_point_set.py \
  /path/to/canonical_mesh.stl \
  /path/to/dataset/cad_points/example.npz \
  --source_to_meters 0.001 \
  --point_count 4096 \
  --query_count 512
```

For USD, the script uses the stage metres-per-unit value unless
`--source_to_meters` overrides it. It prints the point-set catalog fragment,
mesh and artifact checksums, exact area-weighted centroid, sampling version,
full sampling parameters (including NumPy version), and parameter checksum.
`T_cad_from_source_meters` may be supplied as 16 row-major values when the
source mesh is not already in the canonical AABB frame.

Raw mesh vertices are not the default because tessellation density would
otherwise weight some surface regions more heavily than others. The artifact
checksum, source mesh checksum, sampling version, random seed, and parameter
checksum are training provenance.

Compute $\mu$ directly from the triangle surface measure—sum each triangle
centroid weighted by its area and divide by total nondegenerate area—rather
than taking the mean of a finite random sample. This makes the anchor
deterministic and avoids sample-density bias.

For a watertight mesh, a volume centroid may be stored additionally, but the
baseline uses an area-weighted surface centroid so non-watertight CAD meshes
remain supported.

## Why centroid centering removes off-origin symmetry translations

Let the canonical CAD surface be $M$, with surface centroid $\mu$. A rigid
symmetry may act in the AABB frame as

$$
H(p)=Sp+v
$$

and can therefore contain a nonzero translation $v$. Because $H$ maps the
surface and its area measure onto themselves, it must fix the surface centroid:

$$
S\mu+v=\mu.
$$

Define centered points $q=p-\mu$. In centroid-centered coordinates, the same
symmetry acts as

$$
H(q)=Sq.
$$

Thus the point-set rotation loss can recognize both origin-centered and
off-origin symmetries without storing their axes, orders, centers, or
transformations.

## Pose and rendering-scale conventions

Perseve continues to store an AABB-origin pose
`T_cam_from_cad = [R | t]`. Rendering scale remains outside that rigid
transform. For the baseline, pose-supervised rendering scale must be uniform:

$$
s_x=s_y=s_z=s.
$$

For a canonical point $p$, its camera-space position is

$$
x_{cam}=R(sp)+t.
$$

The camera-space surface centroid is

$$
c=R(s\mu)+t.
$$

Non-uniform scale can destroy a CAD model's rotational symmetries and is outside
the initial label-free point-set contract.

## Translation representation

The pose head should predict the projected camera-space surface centroid and
its log-depth, rather than the projected AABB origin:

$$
(u_c,v_c)=\pi(Kc),
\qquad
\hat z_c=\log c_z.
$$

Back-project the prediction using the adjusted intrinsics to obtain
$\hat c$. Convert it to the public AABB-origin translation with the predicted
rotation:

$$
\hat t=\hat c-\hat R(s\mu).
$$

Equivalent geometric poses share the same camera-space centroid even when
their AABB-origin translations differ. This avoids contradictory translation
supervision while preserving `T_cam_from_cad` as the dataset and inference
format.

The centroid is CAD metadata, not a quantity the network must infer without a
reference. The model learns where the known object's centroid lies in the
camera frame, just as the existing branch learns where the canonical origin
lies.

## Rotation loss

Let $P_q$ be a query surface point set and $P_d$ a denser target point set.
Center both in the canonical frame:

$$
Q_q=\{s(p-\mu):p\in P_q\},
\qquad
Q_d=\{s(p-\mu):p\in P_d\}.
$$

Use ground-truth centroid translation on both sides so translation cannot
compensate for rotation:

$$
X_{pred}=\{\hat Rq+c_{gt}:q\in Q_q\},
$$

$$
X_{gt}=\{R_{gt}q+c_{gt}:q\in Q_d\}.
$$

Because the common translation cancels, the implementation may compute the
equivalent centered form:

$$
L_{R,set}=
\frac{1}{|Q_q|}
\sum_{q\in Q_q}
\rho\left(
\frac{
\min_{r\in Q_d}\lVert\hat Rq-R_{gt}r\rVert_2
}{
d_{effective}
}
\right),
$$

where $d_{effective}=s\lVert d_{CAD}\rVert_2$ is the effective
bounding-box diagonal and $\rho$ is a robust penalty such as Smooth-L1.

Do not instead apply the raw AABB-origin ground-truth translation $t_{gt}$ to
uncentered points on both sides. Changing only the rotation around that origin
also moves the surface centroid, so an off-origin equivalent pose would be
penalized. The correct rotation-only anchor is the ground-truth surface
centroid $c_{gt}$, which is exactly equivalent to centering the canonical
points and omitting translation.

This is a nearest-neighbor point-set loss. Comparing point $i$ only with point
$i$ would retain canonical correspondences and would not handle symmetry.

The baseline uses a one-sided dense-target loss similar to ADD-S. A
bidirectional Chamfer term may be evaluated as an ablation. The implementation
must document which direction, reduction, robust penalty, and normalization it
uses.

## Finite-sampling behavior

A finite point set approximates the continuous CAD surface. Even an exact
continuous symmetry may produce a small nonzero nearest-neighbor loss after
rotation because the rotated query samples need not land on target samples.

Mitigations are:

- use a denser target set than query set;
- sample uniformly by surface area;
- normalize by effective object diameter;
- keep sampling deterministic for reproducibility;
- measure the numerical symmetry floor on analytic shapes; and
- consider point-to-triangle or signed-distance supervision later if the
  finite-sampling floor is material.

The initial implementation should not add an arbitrary zero-loss threshold
until this floor is measured on representative CADs.

## Optional full-pose consistency loss

(The loss that combines the both rotation and translation into a single nearest-neighbor term.)

An optional auxiliary term compares complete camera-space placements:

$$
\hat X=\{\hat R(sp)+\hat t:p\in P_q\},
$$

$$
X_{gt}=\{R_{gt}(sp)+t_{gt}:p\in P_d\}.
$$

The corresponding nearest-neighbor loss is label-free and accepts any
geometrically equivalent SE(3) pose. It couples rotation and translation, so it
should be introduced only after the disentangled centroid and rotation losses
are stable.

Starting with this term disabled does not remove translation supervision from
the point geometry: the baseline center/depth targets are the transformed
surface centroid. The optional term asks the entire predicted surface placement
to contribute an additional coupled translation gradient.

## Training objective

For one-to-one mask-matched pose pairs, the target objective is

$$
L_{pose}=
\lambda_{uv}L_{uv,c}+
\lambda_zL_{\log z,c}+
\lambda_RL_{R,set}+
\lambda_{full}L_{full,set}+
\lambda_qL_{quality}.
$$

The baseline starts with $\lambda_{full}=0$.

- $L_{uv,c}$: Smooth-L1 on the normalized projected surface centroid.
- $L_{\log z,c}$: Smooth-L1 on normalized centroid log-depth.
- $L_{R,set}$: centroid-centered nearest-neighbor rotation loss.
- $L_{full,set}$: optional complete-pose point-set consistency.
- $L_{quality}$: BCE-with-logits against a detached geometric-quality target.

Pose matching remains mask-based and one-to-one. Point-set distance is computed
only after matching, not across every detector candidate.

## Pose-quality target

Rotation angle is not a well-defined correctness measure for a symmetric
object. Pose quality should therefore be based on geometry rather than a
symmetry-aware angular threshold.

Define normalized centroid error

$$
e_c=
\frac{\lVert\hat c-c_{gt}\rVert_2}{d_{effective}}
$$

and normalized point-set placement error

$$
e_{set}=
\frac{1}{|P_q|}
\sum_{p\in P_q}
\frac{
\min_{r\in P_d}
\lVert
\hat R(sp)+\hat t-
\left(R_{gt}(sr)+t_{gt}\right)
\rVert_2
}{
d_{effective}
}.
$$

A baseline soft target multiplies the two success factors:

$$
q_{pose}^{*}=
\sigma\left(\frac{\theta_c-e_c}{\delta_c}\right)
\sigma\left(\frac{\theta_{set}-e_{set}}{\delta_{set}}\right).
$$

The thresholds and positive soft widths are configured before training and
recorded in checkpoint provenance. The target is detached before
BCE-with-logits training. Validation-only scalar temperature calibration
remains unchanged.

## Eligibility

Explicit symmetry status no longer controls pose eligibility. An instance is
eligible when:

- it is visible and has a valid mask join;
- its pose and camera geometry are valid;
- depth is positive;
- its CAD catalog entry has a valid checksummed point-set artifact and
  centroid; and
- its rendering scale satisfies the point-set contract.

Missing or invalid point-set artifacts exclude pose supervision but do not
exclude segmentation supervision.

## Computational plan

Rigidly transforming $N$ points is inexpensive compared with the SAM3
forward pass. Nearest-neighbor search is the material cost:

- direct pairwise distance is $O(N_qN_d)$;
- materializing large pairwise tensors can consume substantial activation
  memory; and
- the minimum operation is piecewise differentiable and may have ICP-like
  local minima.

Start with:

- 256 or 512 query points;
- 1024 or 2048 target points;
- point-set loss only on already matched instances;
- a GPU KNN/Chamfer kernel or chunked distance computation;
- cached per-CAD point tensors; and
- explicit timing and peak-memory benchmarks.

The implementation computes point transforms and chunked distances in float32
for half-precision training. When `full_pose_weight` is zero, the
full-placement distances needed by the detached quality target are evaluated
without retaining their autograd graph; the rotation-only distance keeps its
gradient.

At 512 query points and 2048 target points, direct pairwise matching evaluates
about 1.05 million distances per matched instance. The float32 distance matrix
alone is about 4 MiB per instance before autograd workspace. This is practical
at modest matched-instance counts but should not be silently multiplied across
all detector candidates.

Use a lower-resolution point set for any future pose-aware assignment cost.
Do not add point-set cost to assignment in the first milestone.

## Evaluation

Primary pose metrics become:

- normalized mean surface-set distance;
- a maximum or high-percentile surface distance for sensitivity to small but
  important displaced features;
- camera-space centroid error;
- AABB-origin translation error as a diagnostic, not the sole equivalence
  criterion;
- projected surface or silhouette error where useful;
- pose-score Brier score, expected calibration error, and reliability curves;
- per-CAD and per-size breakdowns.

Raw rotation error in degrees may be reported as a diagnostic for known
asymmetric CADs, but it is not an aggregate correctness metric without
symmetry metadata.

Geometry-only point matching intentionally ignores texture. If appearance or
task semantics distinguish geometrically identical poses, this loss cannot
supervise that distinction. Such semantics require colored/featured geometry,
render-and-compare supervision, task-specific annotations, or a separately
defined downstream objective.

## Implementation status

Implemented in this repository:

1. deterministic STL/USD point-set generation, exact surface centroid, and
   checksummed NPZ artifacts;
2. checked-in `perseve-pose-v2` schemas and a v1/v2 loader;
3. surface-centroid center/depth targets with AABB-translation reconstruction;
4. chunked centroid-centered nearest-neighbor rotation loss and an optional
   full-placement loss;
5. geometric pose-quality targets, point-set evaluation, and calibration;
6. point-artifact and sampling-pipeline checkpoint provenance;
7. pose-head architecture version 3; and
8. unit coverage for artifact determinism, schema behavior, centroid
   round-trips, symmetry-equivalent set matching, gradients, and v2 loading.

Still required before declaring the research migration validated:

- generate a representative v2 dataset with the external Perseve pipeline;
- benchmark point-distance time and peak memory in a real SAM3 training step;
- measure finite-sampling floors across representative CADs; and
- compare held-out pose quality with the v1 explicit-symmetry baseline before
  removing legacy support.

## Acceptance criteria

The replacement is ready when:

- AABB-origin dataset and inference poses remain unchanged;
- centroid-to-AABB translation round trips under every supported scale;
- identical asymmetric poses have zero numerical point-set loss;
- known discrete, continuous, and off-origin geometric symmetries receive the
  measured finite-sampling floor without any labels;
- incorrect asymmetric rotations receive a materially larger loss;
- translation cannot compensate inside the rotation-only loss;
- gradients are finite for representative CADs and 6-D rotations;
- point-set loss runtime and peak memory are acceptable in a real SAM3
  training step;
- all visible instances with valid point geometry can receive pose
  supervision; and
- evaluation and pose confidence use the same geometric equivalence
  definition as training.
