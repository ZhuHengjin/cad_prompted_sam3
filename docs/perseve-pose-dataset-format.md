# Perseve Pose-Label and Dataset Format Design

## Status and scope

This document defines the proposed synthetic pose-label extension for
[Perseve](../../perseve) and its interface with the
[CAD-prompted SAM3 pose head](cad-pose-head-plan.md). It is the data contract
for the first pose-head milestone, not a model-output format.

The main decisions are:

- preserve Perseve's existing RGB, instance-segmentation, and scene-parameter
  outputs;
- add a versioned object catalog and one pose-annotation JSON sidecar per
  rendered frame;
- store geometric ground truth rather than model-specific targets;
- define poses in an OpenCV-style camera frame using metres and column-vector
  transforms;
- keep scale separate from the rigid object-to-camera transform;
- retain the current CSV manifest as the authority for sample identity and
  train/validation/test membership;
- provide BOP as an optional export format rather than the canonical Perseve
  representation.

## Why the current Perseve metadata is insufficient

Perseve currently logs settled world positions, Euler angles, and scale values
to `scene_parameters.csv`. These are useful generation-provenance fields, but
they are not pose labels suitable for the pose head:

- the pose head requires CAD-to-camera rather than CAD-to-world transforms;
- decomposed `rotateXYZ` attributes do not capture every transform in the USD
  hierarchy;
- the current object parent does not define an explicit, centered canonical
  CAD frame;
- the semantic label is derived from the filename prefix before the first
  underscore, which can merge unrelated CAD identities;
- camera intrinsics and the renderer's actual camera transform are not written
  by the current `BasicWriter` configuration;
- `target_bbox_size` is a scale-sampling parameter, not the object's actual
  three-dimensional dimensions;
- a world-axis-aligned bounding box changes with rotation and is therefore not
  an object-frame size label.

Pose labels must instead be computed from the final USD transforms after
physics has settled and from the exact camera used for the corresponding
capture.

## Canonical coordinate contract

All code that reads or writes pose annotations must obey the following
contract.

| Property | Convention |
| --- | --- |
| Length | metres |
| Angle metadata | radians unless the field name explicitly ends in `_deg` |
| Camera axes | OpenCV: `+x` right, `+y` down, `+z` forward |
| Image axes | origin at the top-left; `u` right and `v` down |
| Pixel convention | the center of the top-left pixel is `(0, 0)` |
| Vectors | column vectors |
| Transform naming | `T_target_from_source` |
| Transform action | `p_target = T_target_from_source @ p_source` |
| JSON matrix layout | nested rows in row-major display order |
| Rotation | proper rotation in `SO(3)`, with scale excluded |

Native USD cameras look along local `-z`, with local `+x` right and `+y` up.
The conversion from a native USD camera coordinate to the stored OpenCV camera
coordinate is

$$
C_{cv\leftarrow usd}=\operatorname{diag}(1,-1,-1,1).
$$

After converting matrices returned by the USD API to the documented
column-vector convention, the conceptual composition is

$$
T_{cam(cv)\leftarrow cad} =
C_{cv\leftarrow usd}
T_{cam(usd)\leftarrow world}
T_{world\leftarrow cad}.
$$

The USD matrix API's in-memory conventions must be isolated inside one tested
conversion helper. Callers must not transpose or reorder matrices ad hoc.

Scale is not embedded in `T_cam_from_cad`. For a canonical CAD point, the
stored label means

$$
p_{cam} = R_{cam\leftarrow cad}\operatorname{diag}(s)p_{cad} + t.
$$

## Dataset layout

The native Perseve output remains frame-oriented:

```text
v2_sdg_output/
├── dataset_meta.json
├── objects.json
├── scene_parameters.csv
├── rgb_0000.png
├── instance_segmentation_0000.png
├── instance_segmentation_mapping_0000.json
├── pose_annotations_0000.json
├── rgb_0001.png
├── instance_segmentation_0001.png
├── instance_segmentation_mapping_0001.json
└── pose_annotations_0001.json
```

One JSON sidecar per frame is preferred over one continuously appended JSONL
file because it supports atomic scene completion, interrupted-run cleanup,
resume validation, and independent multi-GPU output directories. A merge tool
may create a JSONL index after generation if sequential access is useful.

### Dataset metadata

`dataset_meta.json` defines conventions and generator provenance:

```json
{
  "schema": "perseve.pose",
  "schema_version": "1.0.0",
  "length_unit": "m",
  "transform_convention": "p_target = T_target_from_source @ p_source",
  "vector_convention": "column",
  "matrix_storage": "row_major_json",
  "camera_frame": "opencv_x_right_y_down_z_forward",
  "image_origin": "top_left",
  "pixel_convention": "top_left_pixel_center_is_0_0",
  "asset_catalog": "objects.json",
  "generator": {
    "name": "perseve",
    "git_commit": "<commit>",
    "isaac_sim_version": "5.1.0",
    "seed": 42,
    "config_sha256": "<sha256>"
  }
}
```

Changing a coordinate, unit, or matrix convention requires a schema-version
change. A dataset validator must reject unknown major versions.

### Object catalog

`objects.json` stores facts shared by every occurrence of a CAD model:

```json
{
  "schema_version": "1.0.0",
  "objects": {
    "170041564-21-cl-1-bd": {
      "mesh_path": "cad_usd/170041564-21-cl-1-bd/cl-1-bd_stl.usd",
      "mesh_sha256": "<sha256>",
      "source_length_unit": "mm",
      "T_cad_from_source": [
        [1, 0, 0, -0.012],
        [0, 1, 0, 0.004],
        [0, 0, 1, -0.006],
        [0, 0, 0, 1]
      ],
      "base_dimensions_m": [0.0318, 0.0158, 0.0096],
      "canonical_origin": "local_aabb_center",
      "metric_status": "trusted",
      "symmetry": {
        "type": "discrete",
        "transforms": []
      }
    }
  }
}
```

`cad_id` must be a stable full identifier, not a truncated semantic class.
`mesh_sha256` prevents an annotation from silently referring to a changed mesh.

`T_cad_from_source` records unit conversion and the source-to-canonical frame.
For the baseline, preserve source axis directions and translate the origin to
the local bounding-box center. This makes the translation target correspond to
the geometric object center assumed by the projected-center pose branch. The canonical source axes are supplied to downstream consumers. A curated
per-object override may define a task-specific engineering frame later.

Symmetry metadata is object-level. Supported values are:

- `none`;
- `discrete`, with proper object-frame transforms in `SO(3)`;
- `continuous_axis`, with a unit axis expressed in the canonical CAD frame.

Reflections are not rotational symmetries and must not be placed in the
rotation group.

### Per-frame pose annotations

`pose_annotations_NNNN.json` stores the camera and every rendered object
instance:

```json
{
  "schema_version": "1.0.0",
  "frame_id": "0000",
  "image": {
    "rgb_path": "rgb_0000.png",
    "instance_path": "instance_segmentation_0000.png",
    "size_wh": [512, 512]
  },
  "camera": {
    "model": "pinhole",
    "K": [
      [746.7, 0.0, 256.0],
      [0.0, 746.7, 256.0],
      [0.0, 0.0, 1.0]
    ],
    "distortion": [],
    "T_world_from_camera_cv": [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 1.7],
      [0, 0, 0, 1]
    ]
  },
  "instances": [
    {
      "instance_id": "0000:0",
      "cad_id": "170041564-21-cl-1-bd",
      "prim_path": "/World/PlacedObjects/Object_0",
      "mask_color": [37, 91, 182, 255],
      "mask_channels": "RGBA",
      "bbox_visible_xyxy_px": [124, 86, 289, 253],
      "T_cam_from_cad": [
        [0.99, 0.02, 0.11, 0.031],
        [-0.04, 0.98, 0.18, -0.015],
        [-0.10, -0.18, 0.98, 0.742],
        [0, 0, 0, 1]
      ],
      "render_scale_xyz": [1.0, 1.0, 1.0],
      "dimensions_m": [0.0318, 0.0158, 0.0096],
      "visibility": {
        "visible_pixel_count": 18234,
        "visible_fraction": null,
        "truncated": false
      }
    }
  ]
}
```

`mask_color` is the exact value needed to extract that instance from the
instance-segmentation image. It provides the one-to-one join that the current
semantic mapping cannot guarantee. Multiple occurrences of one `cad_id` have
different `instance_id` and `mask_color` values.

The tight renderer box is stored as `bbox_visible_xyxy_px`. An amodal box, full
silhouette area, or true visible fraction may be added later, but must use a
distinct field name and document how it was computed. Unknown visibility
values are `null`, not fabricated defaults.

## Perseve generation changes

### 1. Canonicalize assets

Asset preparation must determine and persist:

- stable `cad_id`;
- source and converted mesh checksums;
- source-to-metre conversion;
- source-to-canonical transform;
- canonical local bounds and dimensions;
- metric-trust status;
- symmetry metadata or an explicit `none` value.

The scene should reference a canonical object root whose origin and axes match
the catalog. Physics, rendering, prompt renders, and pose annotations must all
refer to that same root.

### 2. Capture the renderer camera

Attach Replicator's `camera_params` annotator to the same persistent render
product as the `BasicWriter`. After each `rep.orchestrator.step()` and
`wait_until_complete()` call, read the actual camera view/projection data used
for that frame. Do not reconstruct calibration from logged pitch, yaw, and
distance values.

Store the final `K` used to project the saved raster. Preserve sufficient raw
camera parameters in debug output to reproduce `K`, especially focal length,
aperture, aperture offset, render resolution, and scene-unit conversion.

### 3. Capture final object transforms

After physics settling and after the capture step:

1. use `UsdGeom.XformCache` to get the full canonical-root-to-world transform;
2. decompose it into translation, a proper rotation, and scale;
3. reject transforms containing material shear or singular scale;
4. get the captured camera-to-world transform;
5. convert the camera axes from USD to OpenCV;
6. compose `T_cam_from_cad`;
7. join each placed prim to its exact rendered mask value;
8. write the frame sidecar atomically.

The logged Euler angles remain useful provenance, but are not the ground-truth
rotation.

### 4. Preserve instance identity

Do not derive `cad_id` with `object_name.split("_", 1)[0]`. Preserve the full
asset identifier. The existing mapping file may remain string-valued for
backward compatibility with segmentation loaders, while the pose sidecar owns
the unambiguous mask-to-instance-to-CAD mapping.

### 5. Make pose output resume-safe

When pose labels are enabled, a frame is complete only when all required files
exist and validate:

- RGB image;
- instance-segmentation image;
- instance mapping;
- pose annotation.

Write JSON to a temporary file in the output directory and atomically replace
the final path. Resume discovery must find the highest contiguous set of
complete frames rather than trusting the maximum RGB index. Partial artifacts
at the next frame are removed before generation resumes.

## Metric and normalized rendering-scale modes

Perseve currently rescales every object into a sampled bounding-box range. The
pose extension must make this behavior explicit.

### Metric mode

- Preserve the physical dimensions of a trusted CAD model.
- Set `render_scale_xyz` to `[1, 1, 1]` relative to the metric catalog asset.
- Use the catalog dimensions directly.
- The pose head receives no size or scale target.

This is the preferred mode for CAD-prompted instance pose estimation.

### Normalized mode

- Retain Perseve's random uniform object scaling.
- Store the exact scale relative to the catalog object.
- Store rendered object-frame dimensions as
  `base_dimensions_m * render_scale_xyz`.
- Mark the object or run as normalized rather than claiming the original CAD
  dimensions are physically preserved.

The current generator creates only uniform scale variation. Preserve it only
as render provenance; no size residual or scale target is derived for the pose
head. Artificial anisotropic CAD distortion should not be added.

## SAM3 loader contract

The existing dataset manifest remains frame-level and continues to define
`dataset_id`, `dataset_path`, `camera_dir`, `frame_id`, provenance group, and
split. The pose loader resolves `pose_annotations_<frame_id>.json` beside the
existing RGB and instance files and validates that:

- the sidecar frame ID matches the manifest row;
- every `cad_id` exists in `objects.json`;
- every annotated mask value exists and has nonzero visible pixels;
- no visible object mask is missing a pose entry;
- camera and object values satisfy the geometry checks below.

The training loader derives model-specific targets rather than storing them in
the dataset:

$$
(u,v) = \pi(Kt), \qquad z=t_z,
$$

$$
\hat z=\log z,
$$

$$
r_{6D}=[R_{:,0};R_{:,1}],
$$

Projected centers are normalized or converted to box-center residuals in the
loader according to the training configuration. The stored `K` is updated by
every later geometric image transform. `distance_to_camera` is Euclidean ray
distance and must not be substituted for camera-axis depth `t_z`.

This mirrors the useful part of YOPO's dataset handling: its BOP loader reads
camera intrinsics and model-to-camera rotation/translation, then derives the
6-D rotation representation, projected center, and log-depth in the loader.

## Optional BOP export

An offline `export_bop.py` should provide interoperability with YOPO and BOP
evaluation tools. BOP is not the canonical Perseve representation because its
directory structure, integer object IDs, and millimetre translations do not
match the native SAM3 data contract.

The exporter maps:

| Perseve | BOP |
| --- | --- |
| `camera.K` | `scene_camera.json: cam_K` |
| `T_cam_from_cad[:3,:3]` | `scene_gt.json: cam_R_m2c` |
| `1000 * T_cam_from_cad[:3,3]` | `scene_gt.json: cam_t_m2c` in millimetres |
| stable CAD-ID table | integer `obj_id` |
| instance mask | `mask/IMID_GTID.png` |
| visible instance mask | `mask_visib/IMID_GTID.png` |
| box and available visibility fields | `scene_gt_info.json` |

The exporter must emit and preserve a reversible CAD-ID-to-`obj_id` mapping.
It must not invent `visib_fract`; if the required amodal information is not
available, the export should omit unsupported metadata or compute it through a
documented additional rendering pass.

## Validation and tests

### Pure geometry tests

- identity and known-axis transform composition;
- USD-camera to OpenCV-camera axis conversion;
- matrix serialization and deserialization;
- rotation orthonormality and `det(R) = +1`;
- scale extraction and shear rejection;
- source-to-canonical-to-camera point round trips;
- discrete symmetry transforms remain in `SO(3)`;
- continuous symmetry axes are unit length.

### Isaac Sim integration test

Render a canonical cube of known dimensions with a fixed camera and verify:

- the canonical origin projects to `K @ t`;
- all eight transformed cuboid corners reproject to their expected pixels;
- the projected geometry overlays the RGB image and instance mask;
- the object lies in front of the camera with `t_z > 0`;
- stored object-frame dimensions equal canonical dimensions times scale;
- the annotated mask value selects the intended instance;
- the box encloses all selected visible-mask pixels.

Include a scene with two copies of the same CAD model to ensure instance IDs
remain distinct, plus an occluded scene to exercise visibility metadata.

### Dataset-wide validation

Before training, report and fail on:

- missing or duplicate frame/instance identities;
- missing RGB, mask, mapping, catalog, or sidecar files;
- non-finite values;
- invalid or singular camera matrices;
- non-positive depth;
- rotations outside a configured orthonormality tolerance;
- inconsistent dimensions and scale;
- pose entries whose mask is empty;
- visible masks without pose entries;
- projected CAD centers or corners that are implausibly inconsistent with the
  annotated image geometry.

Store the dataset metadata, object catalog, manifest, and their checksums with
training-run provenance.

## Implementation sequence

1. Add schema dataclasses and a pure-Python validator.
2. Add object-catalog generation and canonical asset roots.
3. Attach camera parameters and implement the tested USD/OpenCV transform
   conversion.
4. Write per-frame pose annotations and exact mask joins.
5. Make resume detection require complete, valid pose frames.
6. Add fixed-scene projection and duplicate-CAD integration tests.
7. Extend the SAM3 manifest loader to resolve and validate pose sidecars.
8. Derive YOPO-style center, depth, and rotation targets in the SAM3
   loader.
9. Add the optional BOP exporter and validate a small export with YOPO's BOP
   dataset loader.

The first data milestone is complete when a fixed synthetic scene round-trips
canonical CAD points through the stored pose and camera, produces pixel
projections aligned with the saved masks, and can be loaded into the pose-head
trainer without using world-space Euler metadata.

## References

- [CAD-prompted SAM3 pose-head plan](cad-pose-head-plan.md)
- [Perseve synthetic generator](../../perseve/src/perseve/synthetic_data_generation/generate_dataset.py)
- [YOPO BOP dataset loader](../../YOPO/yopo/datasets/pose_estimation/bop_pose_datasets.py)
- [YOPO NOCS dataset loader](../../YOPO/yopo/datasets/pose_estimation/nocs_dataset.py)
- [NVIDIA Replicator CameraParams](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/latest/source/extensions/omni.replicator.core/docs/GeneratedNodeDocumentation/OgnCameraParams.html)
- [Isaac Sim camera conventions](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/action_and_event_data_generation/ext_replicator-object/camera.html)
- [BOP dataset format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md)
