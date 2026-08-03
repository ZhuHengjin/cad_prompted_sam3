# Reference-camera embeddings for CAD exemplars

## Goal

Test whether explicit reference-camera orientation helps the CAD pose head use
multi-view SAM3 exemplars as a geometrically organized set rather than an
unordered appearance bank. This is a causal experiment: every arm uses the
same corrected canonical renders, data manifest, initialization checkpoint,
seed, optimizer settings, view count, and token count.

Do not compare against checkpoints trained on the inconsistent Blender output.
The renderer correction changes the input distribution, so the first arm is a
new corrected-render baseline.

## Representation

Every successfully encoded reference view retains:

- its exemplar tokens;
- a token-to-view index for every token;
- `R_refcam_cv_from_cad` from `<cad_id>_render_transform.json`;
- the ordered view ID and CAD ID.

The camera adapter expresses OpenCV camera forward `(0, 0, 1)` and up
`(0, -1, 0)` in the canonical CAD frame using `R^T`. This continuous 6-vector
avoids Euler-angle discontinuities. Three powers-of-two Fourier frequencies
are concatenated with the raw directions and passed through a `42 -> 128 ->
256` MLP. Its final projection is zero initialized, and the 256-vector is added
to every token belonging to that view.

Reference images are still encoded and cached without gradients. The residual
is applied after cache loading, so the adapter is trained end to end through
the frozen or trainable exemplar fusion stack.

## Experiment matrix

| Mode | Token residual | Question |
| --- | --- | --- |
| `none` | Exact bypass | Corrected-render baseline |
| `camera` | True continuous camera orientation | Does correct geometry help? |
| `shuffled_camera` | Deterministic within-CAD rotation permutation | Does correspondence, rather than extra capacity, matter? |
| `zero_camera` | Constant zero direction through the same camera MLP | Does a learned constant adapter explain the gain? |
| `view_id` | Learned ordinal reference-view embedding | Is fixed view identity sufficient, or is continuous geometry better? |

All learned modes are exact no-ops at initialization. `none` does not run the
adapter at all and is tested for bit-exact equality with legacy token padding.
The shuffled mapping is a stable derangement for `(shuffle seed, CAD ID)`, so
no view retains its correct rotation, and it does not change between epochs or
evaluation.

## Readiness gate

Wait for the complete rerender, then run the render validation documented in
[ABC pose dataset preparation](abc-pose-v2-preparation.md). It must report zero
incomplete pairs and zero transform-metadata issues for the active catalog.

The pose trainer adds a stricter preflight before training. It checks every
requested CAD/view against the active catalog, requires catalog-driven
canonical geometry, verifies the source-to-CAD transform and physical
dimensions, requires identity presentation geometry, and validates every
`R_refcam_cv_from_cad` as a proper rotation. A failed preflight is a hard stop.

## Launching matched runs

Use one segmentation-trained checkpoint whose pose head/view adapter have not
been trained on the inconsistent renders. The launcher requires the common
initialization checkpoint explicitly and writes the five resolved commands to
`experiment_matrix.json` before starting them sequentially:

```bash
uv run python scripts/run_exemplar_view_experiments.py \
  --output_root runs/exemplar_view_pose_v1 \
  -- \
  --model_path /path/to/sam3.pt \
  --init_path /path/to/corrected-render-segmentation-init.pth \
  --dataset_manifest /path/to/manifest.csv \
  --data_root /path/to/data-parent \
  --reference_dir /path/to/completed-canonical-renders \
  --enable_pose \
  --pose_stage head \
  --ref_view_ids 0,1,2,3,4,5,6,7,8,9,10,11 \
  --seed 42 \
  --device cuda:0
```

Add `--dry_run` before `--` to inspect the commands without training. To run a
short systems check first, pass the trainer's normal small epoch/batch settings
and select modes with, for example, `--modes none camera shuffled_camera`.

`--init_path` loads only the fusion, detector, and segmentation modules. It
intentionally starts at epoch 1 with a fresh optimizer, pose head, and view
adapter; the checkpoint SHA-256 is recorded. `--resume_path` remains reserved
for continuing an interrupted arm and rejects cross-mode resumes.

The trainer stores `exemplar_view_mode`, shuffle seed, adapter weights,
architecture version/config, manifest checksum, reference provenance, and all
CLI arguments in each checkpoint/run directory. Resume rejects a checkpoint
from a different trained mode. Evaluation and the webcam/RealSense entry
points default to `--exemplar_view_mode auto`, restoring the mode and adapter
from checkpoint provenance.

## Evaluation and decision rule

First compare all five arms on the same validation split and training budget.
Use the existing pose metrics, especially:

- end-to-end pose success and assignment coverage;
- mean and p95 normalized surface distance;
- rotation and translation error;
- conditional pose metrics at the same mask-IoU threshold;
- segmentation/PQ metrics to detect a pose gain that damages localization.

Select a checkpoint by a predeclared validation metric, then evaluate the test
split once with `--eval_only --eval_split test`. Do not select by test results.
Run at least three seeds before treating a small difference as evidence.

The strongest positive result is `camera > none`, `camera > zero_camera`, and
`camera > shuffled_camera`. If `view_id` matches `camera`, fixed view identity
is sufficient for the current twelve-view renderer; add azimuth/elevation
jitter or held-out camera angles before claiming geometric generalization. If
`camera` and `shuffled_camera` both improve similarly, capacity or
regularization is a more plausible explanation than correct view geometry.

## Recommended follow-up

After the fixed-view matrix, train `none`, `camera`, and `view_id` with
continuous camera jitter and evaluate on held-out angles. That is the decisive
test of whether the continuous encoding learned camera geometry rather than a
twelve-slot lookup table.
