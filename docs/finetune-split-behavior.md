# Versioned Multi-Dataset Splits

`finetune_image_exemplar_multi_gt.py` is the canonical trainer. Its preferred input is a versioned CSV manifest rather than copied train/validation/test folders or frame-only CSV filters.

## Manifest

The manifest columns are:

```text
dataset_id,dataset_path,camera_dir,frame_id,group_id,split
```

`dataset_path` is relative to `--data_root`. Sample identity is `(dataset_id, camera_dir, frame_id)` and leakage checks use `(dataset_id, group_id)`. Consequently, identical numeric frame IDs in unrelated datasets do not become coupled.

For the brick datasets, wrist captures use their source `metadata.json` scene as `group_id`. Side-camera images use `frame_id`, keeping every available `Side_Camera_*` view in one split.

### Column semantics

| Column | Meaning |
| --- | --- |
| `dataset_id` | Stable logical domain name used for split isolation and 50/50 training allocation. |
| `dataset_path` | Dataset directory relative to the runtime `--data_root`; absolute paths and `..` are rejected. |
| `camera_dir` | Direct camera subdirectory containing the processed triplet. |
| `frame_id` | Filename suffix shared by `rgb_*`, `instance_segmentation_*`, and its mapping JSON. |
| `group_id` | Leakage boundary. Every row with the same `(dataset_id, group_id)` must have one split. |
| `split` | Exactly one of `train`, `validation`, or `test`. |

The unique sample key is `(dataset_id, camera_dir, frame_id)`. A manifest cannot repeat that key, map one `dataset_id` to multiple `dataset_path` values, or place one dataset-qualified group in multiple splits.

## Builder implementation

`build_dataset_manifest.py` performs the following steps:

1. Discover each `instance_segmentation_<frame_id>.png` directly inside the configured camera directories.
2. Require the corresponding RGB (`.png`, with `.jpg` fallback) and `instance_segmentation_mapping_<frame_id>.json` files.
3. Read accepted wrist captures from `metadata.json`, format their flattened IDs to four digits, and require exact equality between metadata IDs and processed wrist IDs.
4. Use every side-camera frame ID as a group, including uneven cases where a frame is present in only some camera folders.
5. Split the unique groups separately for each `dataset_id`. The RNG seed is derived from SHA-256 of `"<seed>:<dataset_id>"`, so adding another dataset cannot reshuffle an existing dataset.
6. Round train and validation group counts from the normalized ratios; assign all remaining groups to test.
7. Write rows in deterministic dataset/group/camera/frame order and immediately reload them through the production validator.

The builder does not inspect or alter image pixels, copy files, or create split-specific image directories.

Generate and validate version `v1`:

```bash
DATA_ROOT=/home/hengjinz/data/brick_sam_sdg
MANIFEST="$DATA_ROOT/splits/v1/manifest.csv"

python3 build_dataset_manifest.py \
  --data-root "$DATA_ROOT" \
  --output "$MANIFEST" \
  --seed 42 \
  --ratios 0.8,0.1,0.1

python3 build_dataset_manifest.py \
  --data-root "$DATA_ROOT" \
  --output "$MANIFEST" \
  --validate-only
```

The builder fails on missing provenance, duplicate samples, missing files, invalid split names, inconsistent dataset paths, or a group appearing in multiple splits.
It also refuses to replace an existing version unless `--overwrite` is explicitly supplied; normal experiment runs should treat a generated version as immutable.

### Current `v1` result

Seed `42` with ratios `0.8,0.1,0.1` produced:

| Dataset | Train views | Validation views | Test views | Train groups | Validation groups | Test groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wrist_type2` | 1,513 | 190 | 180 | 582 | 73 | 73 |
| `yaw20_side` | 295 | 40 | 35 | 76 | 10 | 9 |
| **Total** | **1,808** | **230** | **215** | **658** | **83** | **82** |

The wrist source contains 1,883 accepted processed captures assigned through 728 source scenes with at least one accepted capture. The side dataset contains 370 processed views across 95 frame groups. The `v1` manifest SHA-256 is `25432c856970cd4f693862d6ceed38fc8770502a2ca5ea14e6428230a83c7a7f`.

## Runtime behavior

- The trainer validates all manifest rows and their three required files before loading the model.
- Manifest rows are grouped by `(dataset_id, resolved camera directory)`, parsed through the existing color-mapping loader, and filtered to the exact listed frame IDs.
- Training uses one frame/view entry per manifest train row. Target object selection and multi-instance mask aggregation remain unchanged.
- The epoch size is the number of train views before balancing. It is divided equally across sorted dataset IDs; each domain is drawn with replacement using `random.Random(seed + epoch)`, then the combined epoch is shuffled with that same RNG.
- For `v1`, each 1,808-view epoch contains exactly 904 wrist and 904 side-domain draws. The smaller side pool is intentionally repeated; validation and test are never repeated.
- Validation expands each listed view into its labeled object targets, evaluates without augmentation or gradients, and reports loss, IoU, correctness, and PQ statistics.
- Test rows are used only with explicit `--eval_only --eval_split test`.
- Each run stores a copy of the manifest, its SHA-256 checksum, resolved counts, sampling policy, and CLI configuration.
- `--dataset_root` remains an unsplit deprecated fallback. It cannot be combined with manifest arguments.

### Run provenance

Every manifest-backed run directory contains:

```text
run_<timestamp>/
├── dataset_manifest.csv
├── run_config.json
├── debug_boxes/
└── finetune*.pth
```

`run_config.json` records the source manifest path, copied-manifest path, manifest digest, resolved data root, `equal_domain_with_replacement` sampling policy, validation summary, and complete parsed CLI arguments. Resume runs should use the same manifest and seed; because sampling depends on the epoch number, resuming at epoch `N` reconstructs the same epoch-`N` sample sequence.

### Legacy behavior

`--dataset_root` and the older frame-only split CSV options remain only for migration. Manifest mode rejects any simultaneous root or legacy split arguments, preventing ambiguous precedence. Root-only mode emits a visible `FutureWarning` and does not receive dataset-qualified balancing.

`finetune_image_exemplar_multi_gt_split.py` is only a deprecated compatibility entry point and delegates to the canonical trainer.
