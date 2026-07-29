# Pose Data Decisions


## Segmentation masks and bounding-box semantics

### Decision

The pose dataset will inherit the conventions implemented by Perseve's existing segmentation loader instead of defining an unrelated encoding.

For dataset format v1, segmentation masks must be four-channel PNG images. Colors in the accompanying mapping JSON are logical RGBA values. A loader using OpenCV must read the image unchanged and convert its in-memory BGRA channel order to RGBA before comparing it with a mapping key. Equality includes all four channels, including alpha.

The mapping JSON color key is the authoritative association between a rendered mask region and its object label. A pose annotation may repeat the parsed value for validation, for example:

```json
{
  "mask": {
    "mapping_key": "(37, 91, 182, 255)",
    "value": [37, 91, 182, 255],
    "value_order": "RGBA",
    "match_alpha": true
  }
}
```

Perseve currently contains fallback branches for three-channel and grayscale segmentation images, but they do not establish compatible general-purpose encodings:

- the three-channel branch promotes BGR to BGRA without performing the four-channel path's BGRA-to-RGBA conversion;
- the grayscale branch promotes intensity values to BGRA and does not interpret them as instance IDs.

These fallback branches are therefore outside the pose dataset v1 contract. Supporting RGB masks or single-channel instance-ID masks later requires a distinct, explicitly declared encoding and corresponding loader behavior.

For boxes derived from a binary mask, the dataset inherits Perseve's current `sample_prompts()` convention:

```text
bbox_xyxy = [x_min, y_min, x_max_inclusive, y_max_inclusive]
```

The coordinates identify the first and last foreground pixel and are computed with `min()` and `max()`. Therefore:

```text
pixel_width  = x_max_inclusive - x_min + 1
pixel_height = y_max_inclusive - y_min + 1
```

Ground-truth boxes stored in the pose annotations should be integer-valued. Model-facing code may convert them to floating point without changing the endpoint convention.

Perseve also writes Replicator `bounding_box_2d_tight` arrays, but Perseve does not interpret those arrays locally. Their schema and endpoint semantics come from the installed Isaac Replicator `BasicWriter`, not from Perseve. The pose exporter should preferably derive its box from the decoded instance mask using the rule above. If it consumes `bounding_box_2d_tight` instead, it must validate and explicitly convert the writer's coordinates to the inclusive convention.

### Validation requirements

For every pose instance, the dataset validator must verify that:

1. the segmentation PNG has four channels;
2. the mapping key parses as exactly four integers in `[0, 255]`;
3. at least one pixel matches the complete RGBA value for every visible annotated instance;
4. the mask-derived box equals the stored inclusive `bbox_xyxy`;
5. the box lies inside the image and has positive pixel width and height.

### Consequences

- Existing Perseve four-channel segmentation output can be reused directly.
- The pose-to-mask join has one authoritative representation: the existing mapping key.
- Channel-order mistakes and alpha mismatches fail validation instead of silently producing empty or incorrect masks.
- Box conversion is only needed at integrations that require half-open coordinates. Such a conversion adds one to `x_max` and `y_max`; it must not change the stored dataset convention.

## Random SDG scaling and prompted physical dimensions

### Decision

Perseve may continue applying random uniform scale to CAD objects during
synthetic data generation. For every rendered instance, the generator must
record both the applied scale and the resulting physical dimensions. The
effective dimensions are passed to the pose model as prompt metadata alongside
the CAD exemplar.

The pose head predicts rotation and metric translation, but it does not predict
object size or scale. Metric size is a known prompt input.

```text
base CAD dimensions
        ×
random SDG scale
        ↓
effective rendered dimensions
        ↓
save in pose annotation
        ↓
CAD exemplar + effective dimensions → pose model → R, metric t
```

### Annotation fields

The object catalog stores the unscaled reference dimensions. Each rendered
instance stores the applied scale and effective dimensions:

```json
{
  "cad_id": "part_123",
  "base_dimensions_m": [0.10, 0.04, 0.02],
  "render_scale_xyz": [1.5, 1.5, 1.5],
  "dimensions_m": [0.15, 0.06, 0.03]
}
```

The required relationship is

$$
d_{instance}=d_{CAD}\odot s_{render}.
$$

Dimensions use canonical CAD axis order `[x, y, z]` and metres. Uniform scaling
is preferred; artificial anisotropic distortion is not required.

### Training and inference contract

During training, the loader passes the annotated effective dimensions with the
CAD exemplar:

```python
pose_predictions = pose_head(
    detection_tokens_bnc,
    boxes_xy1xy2_bn22,
    cad_dimensions_m_b3,
    adjusted_camera_intrinsics_b33,
    model_image_size_wh,
)
```

At inference, the caller supplies the real dimensions of the prompted object,
obtained from a metric CAD mesh, a CAD catalog, or explicit user input. Merely
recording dimensions as ground truth is insufficient: they must be available
to the model at both training and inference for metric translation.
The adjusted camera intrinsics are normalized inside the head and condition
depth together with dimensions and angular box extent.

Random scaling across scenes is valid and useful augmentation. For example,
the same CAD may be rendered at 8 cm in one scene and 15 cm in another, as long
as the corresponding prompt supplies 8 cm and 15 cm respectively.

### Per-CAD scale-sharing rule within a scene

Scale is sampled once per `(scene_id, cad_id)` and reused for every occurrence
of that CAD in the scene. Different CAD IDs may receive different scales, and
the same CAD ID may receive a newly sampled scale in another scene.

```text
Scene 1: CAD A → scale 1.4 for every CAD A instance
Scene 1: CAD B → scale 0.7 for every CAD B instance
Scene 2: CAD A → scale 0.8 for every CAD A instance
```

The SDG implementation should maintain a scene-local cache keyed by the full
stable `cad_id`:

```python
scale_by_cad = {}

for cad in selected_cads:
    if cad.cad_id not in scale_by_cad:
        scale_by_cad[cad.cad_id] = sample_scale(cad)

    place_object(cad, scale=scale_by_cad[cad.cad_id])
```

Do not independently scale two occurrences of the same CAD in one image while
using one CAD-and-dimensions prompt. A monocular image cannot uniquely
distinguish a small nearby copy from a large distant copy based on physical
size, and one prompt cannot provide two different dimension vectors.

Different CAD IDs in the same scene may use independent random scales because
each CAD is paired with its own exemplar and dimension prompt.

The dataset validator must enforce that every pair of instances with the same
`(scene_id, cad_id)` has identical `render_scale_xyz` and `dimensions_m`, within
the declared floating-point tolerance.

### Consequences

- Random object scaling remains enabled in the SDG pipeline.
- Random scale is sampled once per `(scene_id, cad_id)`, not once per object
  occurrence.
- `base_dimensions_m`, `render_scale_xyz`, and `dimensions_m` are required and
  validated for pose-training samples.
- CAD dimensions become a required baseline pose-head input, not an optional
  later geometry feature.
- No learned scale branch or scale loss is required.
- Metric translation is supported only when physical dimensions are supplied
  at inference.
- A BOP export of scaled instances must include matching scaled model geometry;
  otherwise the exporter must reject those instances.
