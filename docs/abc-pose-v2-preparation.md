# ABC Point-Set Pose Dataset Preparation

This workflow converts a flat Perseve ABC output from legacy pose schema v1
to point-set schema v2, builds scene-level data splits, renders a complete set
of visual exemplars from the original ABC STL corpus, validates the joins, and
starts pose-head training.

Set paths for the local checkout and data locations before running anything:

```bash
REPO=/path/to/cad_prompted_sam3
DATA_PARENT=/path/to/generated/abc_run
DATASET="$DATA_PARENT/v2_sdg_output"

# The exact flat USD assets referenced by the generated object catalog.
ABC_USD=/path/to/abc_usd
POINTS=/path/to/abc_point_sets

# The original ABC corpus may contain one STL below each nested object folder.
ABC_STL=/path/to/abc_dataset/stl
STAGED_STL=/path/to/work/abc_stl_render_inputs
REFS="$DATA_PARENT/abc_exemplars"

MANIFEST="$DATA_PARENT/splits/abc_pose_v2/manifest.csv"
MODEL=/path/to/sam3.pt
BLENDER=/path/to/blender
```

The USD and STL collections serve different purposes. Point-set artifacts are
derived from the exact composed USD geometry used by the generated pose
catalog. Reference images are rendered from all original ABC STL meshes by the
existing Blender renderer.

## 1. Preprocess the complete USD point-set collection

`preprocess_abc_point_sets.py` scans the flat `ABC_USD` directory and writes
one `<cad_id>.npz` artifact per USD plus a resumable `point_sets.json` manifest.
The manifest records source geometry, units, canonical transforms, bounds,
dimensions, checksums, sampling parameters, and surface centroids.

USD support is an optional dependency rather than part of the base training
environment. Install the `usd` extra before running the batch preprocessor:

```bash
cd "$REPO"
uv sync --extra usd
uv run --extra usd python preprocess_abc_point_sets.py "$ABC_USD" "$POINTS"
```

The point manifest must cover every object in the generated dataset catalog.
It may contain additional ABC objects; the upgrader ignores those. A rerun
skips compatible completed artifacts. Use `--overwrite` only when deliberately
rebuilding them with the requested sampling configuration.

## 2. Dry-run and apply the v2 metadata upgrade

First run the upgrade without `--apply`. This validates the complete join and
does not modify the dataset:

```bash
uv run python upgrade_perseve_pose_dataset_v2.py \
  "$DATASET" \
  --point-manifest "$POINTS/point_sets.json"
```

If the dry run passes, install the upgrade:

```bash
uv run python upgrade_perseve_pose_dataset_v2.py \
  "$DATASET" \
  --point-manifest "$POINTS/point_sets.json" \
  --apply
```

When `POINTS` is outside the dataset, apply mode copies the verified NPZ files
to `"$DATASET/cad_points"`. If preprocessing wrote there directly, no artifact
copy is needed. Apply mode creates a timestamped `pose_v1_backup_*` directory
containing the old metadata, schemas, and pose sidecars before installing v2.

The upgrade:

- replaces catalog `symmetry` records with point-set records;
- recognizes the ABC generator's legacy omission of the Y-up-to-Z-up catalog
  rotation, requires the exact declared `Y -> Z` sampling transform, and then
  takes `T_cad_from_source_meters` from the verified point manifest;
- leaves every existing `T_cam_from_cad` annotation unchanged because those
  annotations already target the Z-up CAD frame;
- installs the bundled v2 schemas;
- changes dataset, catalog, and annotation versions to `2.0.0`; and
- marks a visible instance eligible only when its mask, box, rigid transform,
  uniform scale, dimensions, and positive surface-centroid depth are valid.

## 3. Build scene-level train, validation, and test splits

Flat-pose mode reads `scene_id` from every pose sidecar and uses it as
`group_id`, so frames from one scene cannot cross splits.

```bash
uv run python build_dataset_manifest.py \
  --data-root "$DATA_PARENT" \
  --pose-dataset abc_pose=v2_sdg_output \
  --output "$MANIFEST" \
  --ratios 0.8,0.1,0.1 \
  --seed 42
```

Existing manifests are not replaced unless `--overwrite` is explicit. Prefer
`--validate-only` for an existing manifest or choose a new versioned path.

## 4. Stage all nested ABC STL files

The original Blender renderer reads a flat directory, while the ABC download
stores its 10,000 STL files below nested object folders. The staging helper
recursively discovers the complete source tree and creates a flat directory of
symlinks without copying or changing source meshes:

```bash
uv run python prepare_abc_stl_render_inputs.py \
  "$ABC_STL" \
  "$STAGED_STL"
```

The helper rejects duplicate case-insensitive CAD stems, broken or unexpected
staging entries, and staging paths inside the source tree. Reruns verify and
reuse symlinks that already resolve to the correct source files.

## 5. Render all twelve exemplar views

Run the existing renderer against the flat staging directory. Its fixed view
preset renders all twelve views for every staged STL:

```bash
"$BLENDER" -b -P "$REPO/blender_renderer.py" -- \
  --stl-dir "$STAGED_STL" \
  --output-dir "$REFS" \
  --size 512
```

For CAD ID `abc123`, the renderer produces padded view IDs and matching masks:

```text
abc123_stl_base_00.png
abc123_stl_base_00_mask.png
...
abc123_stl_base_11.png
abc123_stl_base_11_mask.png
```

The trainer accepts both padded IDs (`00` through `09`) and their unpadded
spellings (`0` through `9`), so the original renderer's filenames work with
the default `--ref_view_ids 0,1,...,11` configuration.

After Blender finishes, validate all 10,000 CAD stems, all twelve view IDs,
and every image/mask pair:

```bash
uv run python prepare_abc_stl_render_inputs.py \
  "$ABC_STL" \
  "$STAGED_STL" \
  --render-dir "$REFS"
```

Do not proceed from a partial render directory; this validation exits nonzero
and reports missing or conflicting artifacts.

## 6. Validate the dataset and start a smoke run

Run the full schema, artifact, geometry, scale-sharing, and pixel-join
validation:

```bash
uv run python validate_perseve_pose_dataset.py \
  "$MANIFEST" \
  "$DATA_PARENT"
```

Then start a conservative one-epoch pose-head run on an available GPU:

```bash
uv run python finetune_image_exemplar_multi_gt.py \
  --model_path "$MODEL" \
  --dataset_manifest "$MANIFEST" \
  --data_root "$DATA_PARENT" \
  --reference_dir "$REFS" \
  --ref_view_ids 0,1,2,3,4,5,6,7,8,9,10,11 \
  --enable_pose \
  --pose_stage head \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 1 \
  --device cuda:0 \
  --output_dir runs/finetune_exemplar_abc_pose_smoke
```

After the smoke run produces finite pose losses and a checkpoint, choose the
production batch size and gradient accumulation for the available GPU.
