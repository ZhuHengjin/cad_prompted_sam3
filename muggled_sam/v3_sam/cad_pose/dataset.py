"""Versioned loader and validator for Perseve pose sidecars (schema v1)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor

from .geometry import adjust_intrinsics_for_resize_and_pad, pixel_to_normalized, project_points
from .symmetry import VERIFIED_SYMMETRY_STATUSES
from .types import CADPoseTarget


SUPPORTED_SCHEMA_MAJOR = 1
ANNOTATION_STATES = frozenset(("visible", "fully_occluded", "out_of_frame", "invalid_geometry", "capture_error"))
SYMMETRY_TYPES = frozenset(("none", "discrete", "continuous_axis"))


@dataclass(frozen=True)
class SymmetryMetadata:
    type: str
    transforms: tuple[np.ndarray, ...]
    axis_cad: np.ndarray | None
    label_source: str
    status: str
    pipeline_version: str
    parameters_sha256: str

    @property
    def rotation_eligible(self) -> bool:
        return self.status in VERIFIED_SYMMETRY_STATUSES


@dataclass(frozen=True)
class CADCatalogObject:
    cad_id: str
    base_dimensions_m: np.ndarray
    bbox_min_m: np.ndarray
    bbox_max_m: np.ndarray
    source_to_meters: float
    T_cad_from_source_meters: np.ndarray
    symmetry: SymmetryMetadata


@dataclass(frozen=True)
class PoseInstance:
    instance_id: str
    cad_id: str
    annotation_state: str
    pose_training_eligible: bool
    mapping_key: str | None
    mask_rgba: tuple[int, int, int, int] | None
    bbox_xyxy_px: tuple[int, int, int, int] | None
    T_cam_from_cad: np.ndarray | None
    render_scale_xyz: np.ndarray | None
    dimensions_m: np.ndarray | None

    @property
    def rotation_matrix(self) -> np.ndarray:
        if self.T_cam_from_cad is None:
            raise ValueError(f"Instance {self.instance_id} has no pose transform")
        return self.T_cam_from_cad[:3, :3]

    @property
    def translation_m(self) -> np.ndarray:
        if self.T_cam_from_cad is None:
            raise ValueError(f"Instance {self.instance_id} has no pose transform")
        return self.T_cam_from_cad[:3, 3]


@dataclass(frozen=True)
class PoseFrame:
    schema_version: str
    frame_id: str
    scene_id: str
    image_size_wh: tuple[int, int]
    intrinsics: np.ndarray
    instances: tuple[PoseInstance, ...]
    annotation_path: Path

    def eligible_instances(self, cad_id: str | None = None) -> tuple[PoseInstance, ...]:
        return tuple(
            instance
            for instance in self.instances
            if instance.pose_training_eligible and (cad_id is None or instance.cad_id == cad_id)
        )


@dataclass(frozen=True)
class PersevePoseSample:
    frame: PoseFrame
    dataset_root: Path
    catalog: Mapping[str, CADCatalogObject]
    dataset_meta: Mapping[str, Any]
    dataset_meta_checksum: str
    catalog_checksum: str
    annotation_checksum: str
    schema_checksums: Mapping[str, str]

    @property
    def symmetry_pipeline_versions(self) -> tuple[str, ...]:
        return tuple(sorted({obj.symmetry.pipeline_version for obj in self.catalog.values()}))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_perseve_pose_sample(camera_root: Path, frame_id: str, *, validate_pixels: bool = True) -> PersevePoseSample:
    """Load a frame sidecar and the dataset-level catalog/meta that govern it."""

    camera_root = camera_root.expanduser().resolve()
    annotation_path = camera_root / f"pose_annotations_{frame_id}.json"
    dataset_root = _find_dataset_root(camera_root)
    meta_path, catalog_path = dataset_root / "dataset_meta.json", dataset_root / "objects.json"
    for path in (annotation_path, meta_path, catalog_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    meta_raw, catalog_raw, annotation_raw = _load_json(meta_path), _load_json(catalog_path), _load_json(annotation_path)
    schema_paths = _resolve_schema_paths(dataset_root, meta_raw)
    _validate_json_schemas(meta_raw, catalog_raw, annotation_raw, schema_paths)
    _require_supported_version(str(meta_raw.get("schema_version", "")), "dataset metadata")
    _require_supported_version(str(catalog_raw.get("schema_version", "")), "object catalog")
    _require_supported_version(str(annotation_raw.get("schema_version", "")), "pose annotation")
    catalog = _parse_catalog(catalog_raw, meta_raw)
    frame = _parse_frame(annotation_raw, annotation_path, catalog, meta_raw)
    if frame.frame_id != str(frame_id):
        raise ValueError(f"Sidecar frame_id {frame.frame_id!r} does not match manifest frame {frame_id!r}")
    if validate_pixels:
        _validate_frame_pixels(frame, annotation_raw, camera_root)
    return PersevePoseSample(
        frame,
        dataset_root,
        catalog,
        meta_raw,
        sha256_file(meta_path),
        sha256_file(catalog_path),
        sha256_file(annotation_path),
        {name: sha256_file(path) for name, path in schema_paths.items()},
    )


def validate_scale_sharing(samples: Sequence[PersevePoseSample]) -> None:
    """Enforce one scale/dimension prompt per ``(scene_id, cad_id)``."""

    seen: dict[tuple[Path, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for sample in samples:
        tolerances = sample.dataset_meta.get("validation_tolerances", {})
        atol, rtol = float(tolerances.get("dimension_atol_m", 1e-6)), float(tolerances.get("dimension_rtol", 1e-5))
        for instance in sample.frame.instances:
            if instance.render_scale_xyz is None or instance.dimensions_m is None:
                continue
            key = sample.dataset_root, sample.frame.scene_id, instance.cad_id
            previous = seen.setdefault(key, (instance.render_scale_xyz, instance.dimensions_m))
            if not np.allclose(previous[0], instance.render_scale_xyz, atol=atol, rtol=rtol):
                raise ValueError(f"Inconsistent render scale for scene/CAD {key}")
            if not np.allclose(previous[1], instance.dimensions_m, atol=atol, rtol=rtol):
                raise ValueError(f"Inconsistent effective dimensions for scene/CAD {key}")


def instance_mask_rgba(instance_image_path: Path, instance: PoseInstance) -> np.ndarray:
    if instance.mask_rgba is None:
        raise ValueError(f"Instance {instance.instance_id} has no logical RGBA mask value")
    image = cv2.imread(str(instance_image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(instance_image_path)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Perseve v1 requires a four-channel instance PNG: {instance_image_path}")
    rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    target = np.asarray(instance.mask_rgba, dtype=np.uint8).reshape(1, 1, 4)
    return np.all(rgba == target, axis=-1)


def inclusive_box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def parse_logical_rgba_key(mapping_key: str) -> tuple[int, int, int, int]:
    values = [part.strip() for part in mapping_key.strip().strip("()").split(",") if part.strip()]
    if len(values) != 4:
        raise ValueError(f"Logical RGBA mapping key must contain four channels: {mapping_key!r}")
    channels = tuple(int(value) for value in values)
    if any(value < 0 or value > 255 for value in channels):
        raise ValueError(f"Logical RGBA mapping key is outside uint8 range: {mapping_key!r}")
    return channels


def make_pose_target(
    instance: PoseInstance,
    catalog_object: CADCatalogObject,
    intrinsics: Tensor,
    source_size_wh: tuple[int, int],
    model_size_wh: tuple[int, int],
) -> CADPoseTarget:
    """Derive normalized projected-center, log-depth, and rotation targets."""

    if not instance.pose_training_eligible or instance.T_cam_from_cad is None or instance.dimensions_m is None:
        raise ValueError(f"Instance {instance.instance_id} is not eligible for pose training")
    adjusted_k = adjust_intrinsics_for_resize_and_pad(intrinsics, source_size_wh, model_size_wh)
    translation = torch.as_tensor(instance.translation_m, dtype=intrinsics.dtype, device=intrinsics.device)
    if translation[2] <= 0:
        raise ValueError(f"Instance {instance.instance_id} has non-positive camera-axis depth")
    center_norm = pixel_to_normalized(project_points(translation, adjusted_k), model_size_wh)
    symmetry = catalog_object.symmetry
    transforms = torch.as_tensor(np.stack(symmetry.transforms), dtype=intrinsics.dtype, device=intrinsics.device)
    axis = None if symmetry.axis_cad is None else torch.as_tensor(symmetry.axis_cad, dtype=intrinsics.dtype, device=intrinsics.device)
    return CADPoseTarget(
        center_uv_norm=center_norm,
        log_depth=translation[2].log(),
        rotation_matrix=torch.as_tensor(instance.rotation_matrix, dtype=intrinsics.dtype, device=intrinsics.device),
        translation_m=translation,
        dimensions_m=torch.as_tensor(instance.dimensions_m, dtype=intrinsics.dtype, device=intrinsics.device),
        symmetry_type=symmetry.type,
        symmetry_transforms=transforms,
        axis_cad=axis,
        rotation_eligible=symmetry.rotation_eligible,
    )


def _find_dataset_root(camera_root: Path) -> Path:
    for candidate in (camera_root, *camera_root.parents):
        if (candidate / "dataset_meta.json").is_file() and (candidate / "objects.json").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find dataset_meta.json and objects.json above {camera_root}")


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _resolve_schema_paths(dataset_root: Path, meta: Mapping[str, Any]) -> dict[str, Path]:
    configured = meta.get("schemas", {})
    names = {"dataset_meta": "dataset-meta.schema.json", "objects": "objects.schema.json", "pose_annotations": "pose-annotations.schema.json"}
    paths = {}
    for key, fallback_name in names.items():
        configured_path = dataset_root / str(configured.get(key, f"schemas/{fallback_name}"))
        if not configured_path.is_file():
            raise FileNotFoundError(configured_path)
        paths[key] = configured_path
    return paths


def _validate_json_schemas(meta: Any, catalog: Any, annotation: Any, schema_paths: Mapping[str, Path]) -> None:
    try:
        import jsonschema
    except ModuleNotFoundError as exc:
        raise RuntimeError("Perseve pose loading requires the 'jsonschema' package") from exc
    for name, value in (("dataset_meta", meta), ("objects", catalog), ("pose_annotations", annotation)):
        jsonschema.Draft202012Validator(_load_json(schema_paths[name])).validate(value)


def _require_supported_version(version: str, kind: str) -> None:
    try:
        major = int(version.split(".", 1)[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid {kind} schema version: {version!r}") from exc
    if major != SUPPORTED_SCHEMA_MAJOR:
        raise ValueError(f"Unsupported {kind} schema major {major}; expected {SUPPORTED_SCHEMA_MAJOR}")


def _parse_catalog(raw: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, CADCatalogObject]:
    rotation_atol = float(meta.get("validation_tolerances", {}).get("rotation_atol", 1e-6))
    dimension_atol = float(meta.get("validation_tolerances", {}).get("dimension_atol_m", 1e-6))
    objects: dict[str, CADCatalogObject] = {}
    for cad_id, value in raw["objects"].items():
        symmetry_raw = value["symmetry"]
        symmetry_type = symmetry_raw["type"]
        if symmetry_type not in SYMMETRY_TYPES:
            raise ValueError(f"Unsupported symmetry type for {cad_id}: {symmetry_type}")
        transforms = tuple(np.asarray(item, dtype=np.float64) for item in symmetry_raw["transforms"])
        for transform in transforms:
            _validate_rotation(transform, rotation_atol, f"{cad_id} symmetry")
        if not any(np.allclose(item, np.eye(3), atol=rotation_atol, rtol=0) for item in transforms):
            raise ValueError(f"Symmetry group for {cad_id} does not contain identity")
        if symmetry_type == "discrete":
            _validate_group_closure(transforms, rotation_atol, cad_id)
        axis = symmetry_raw.get("axis_cad")
        axis_np = None if axis is None else np.asarray(axis, dtype=np.float64)
        if symmetry_type == "continuous_axis" and (
            axis_np is None or not np.isclose(np.linalg.norm(axis_np), 1.0, atol=rotation_atol, rtol=0)
        ):
            raise ValueError(f"Continuous symmetry axis for {cad_id} must be unit length")
        bbox_min, bbox_max = np.asarray(value["bbox_min_m"], dtype=np.float64), np.asarray(value["bbox_max_m"], dtype=np.float64)
        dimensions = np.asarray(value["base_dimensions_m"], dtype=np.float64)
        if not np.allclose(bbox_max - bbox_min, dimensions, atol=dimension_atol, rtol=1e-5):
            raise ValueError(f"Catalog dimensions do not match bounds for {cad_id}")
        canonical_transform = np.asarray(value["T_cad_from_source_meters"], dtype=np.float64)
        _validate_rigid_transform(canonical_transform, rotation_atol, f"{cad_id} source transform")
        symmetry = SymmetryMetadata(
            symmetry_type,
            transforms,
            axis_np,
            str(symmetry_raw["label_source"]),
            str(symmetry_raw["status"]),
            str(symmetry_raw["pipeline_version"]),
            str(symmetry_raw["parameters_sha256"]),
        )
        objects[cad_id] = CADCatalogObject(cad_id, dimensions, bbox_min, bbox_max, float(value["source_to_meters"]), canonical_transform, symmetry)
    return objects


def _parse_frame(raw: Mapping[str, Any], path: Path, catalog: Mapping[str, CADCatalogObject], meta: Mapping[str, Any]) -> PoseFrame:
    intrinsics = np.asarray(raw["camera"]["K"], dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or abs(np.linalg.det(intrinsics)) < 1e-12:
        raise ValueError(f"Invalid camera intrinsics in {path}")
    rotation_atol = float(meta.get("validation_tolerances", {}).get("rotation_atol", 1e-6))
    _validate_rigid_transform(
        np.asarray(raw["camera"]["T_world_from_camera_cv"], dtype=np.float64),
        rotation_atol,
        "camera world transform",
    )
    tolerance = meta.get("validation_tolerances", {})
    atol, rtol = float(tolerance.get("dimension_atol_m", 1e-6)), float(tolerance.get("dimension_rtol", 1e-5))
    image_size = tuple(int(value) for value in raw["image"]["size_wh"])
    instances, seen_ids = [], set()
    for value in raw["instances"]:
        instance_id, cad_id = str(value["instance_id"]), str(value["cad_id"])
        if instance_id in seen_ids:
            raise ValueError(f"Duplicate instance_id {instance_id!r} in {path}")
        seen_ids.add(instance_id)
        if cad_id not in catalog:
            raise ValueError(f"Unknown cad_id {cad_id!r} in {path}")
        state, eligible = str(value["annotation_state"]), bool(value["pose_training_eligible"])
        if state not in ANNOTATION_STATES:
            raise ValueError(f"Unsupported annotation state: {state}")
        if eligible and (state != "visible" or not catalog[cad_id].symmetry.rotation_eligible):
            raise ValueError(f"Ineligible state/symmetry marked pose_training_eligible for {instance_id}")
        mask_raw = value.get("mask")
        mapping_key = None if mask_raw is None else str(mask_raw["mapping_key"])
        mask_rgba = None if mask_raw is None else tuple(int(channel) for channel in mask_raw["value"])
        if mask_raw is not None and (mask_raw.get("value_order") != "RGBA" or mask_raw.get("match_alpha") is not True):
            raise ValueError(f"Instance {instance_id} does not declare logical RGBA alpha matching")
        if mapping_key is not None and parse_logical_rgba_key(mapping_key) != mask_rgba:
            raise ValueError(f"Mapping key and RGBA value disagree for {instance_id}")
        bbox_raw = value.get("bbox_xyxy_px")
        bbox = None if bbox_raw is None else tuple(int(item) for item in bbox_raw)
        if bbox is not None and (bbox[2] < bbox[0] or bbox[3] < bbox[1]):
            raise ValueError(f"Invalid inclusive box for {instance_id}: {bbox}")
        if bbox is not None and (bbox[2] >= image_size[0] or bbox[3] >= image_size[1]):
            raise ValueError(f"Inclusive box lies outside the declared image for {instance_id}: {bbox}")
        transform_raw = value.get("T_cam_from_cad")
        transform = None if transform_raw is None else np.asarray(transform_raw, dtype=np.float64)
        if transform is not None:
            _validate_rigid_transform(transform, rotation_atol, f"{instance_id} camera pose")
        scale_raw, dimensions_raw = value.get("render_scale_xyz"), value.get("dimensions_m")
        scale = None if scale_raw is None else np.asarray(scale_raw, dtype=np.float64)
        dimensions = None if dimensions_raw is None else np.asarray(dimensions_raw, dtype=np.float64)
        if scale is not None and (not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError(f"Render scale must be finite and positive for {instance_id}")
        if dimensions is not None and (not np.isfinite(dimensions).all() or np.any(dimensions <= 0)):
            raise ValueError(f"Effective dimensions must be finite and positive for {instance_id}")
        if scale is not None and dimensions is not None and not np.allclose(
            catalog[cad_id].base_dimensions_m * scale, dimensions, atol=atol, rtol=rtol
        ):
            raise ValueError(f"Effective dimensions do not match catalog dimensions * render scale for {instance_id}")
        if eligible and (transform is None or dimensions is None or bbox is None or mask_rgba is None):
            raise ValueError(f"Pose-eligible instance {instance_id} is missing required geometry or mask fields")
        if eligible and transform[2, 3] <= 0:
            raise ValueError(f"Pose-eligible instance {instance_id} has non-positive depth")
        instances.append(PoseInstance(instance_id, cad_id, state, eligible, mapping_key, mask_rgba, bbox, transform, scale, dimensions))
    return PoseFrame(
        str(raw["schema_version"]),
        str(raw["frame_id"]),
        str(raw["scene_id"]),
        image_size,
        intrinsics,
        tuple(instances),
        path,
    )


def _validate_frame_pixels(frame: PoseFrame, raw: Mapping[str, Any], camera_root: Path) -> None:
    image_info = raw["image"]
    instance_path, mapping_path = camera_root / image_info["instance_path"], camera_root / image_info["mapping_path"]
    mapping = _load_json(mapping_path)
    raw_image = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
    if raw_image is None:
        raise FileNotFoundError(instance_path)
    if (raw_image.shape[1], raw_image.shape[0]) != frame.image_size_wh:
        raise ValueError(
            f"Instance raster size {(raw_image.shape[1], raw_image.shape[0])} differs from {frame.image_size_wh}"
        )
    visible_sidecar_keys = set()
    for instance in frame.instances:
        if instance.annotation_state != "visible":
            continue
        if instance.mapping_key not in mapping:
            raise ValueError(f"Mapping key {instance.mapping_key!r} for {instance.instance_id} is absent")
        visible_sidecar_keys.add(instance.mapping_key)
        mask = instance_mask_rgba(instance_path, instance)
        if not mask.any():
            raise ValueError(f"Visible instance {instance.instance_id} has an empty logical RGBA mask")
        derived = inclusive_box_from_mask(mask)
        if derived != instance.bbox_xyxy_px:
            raise ValueError(f"Mask-derived inclusive box {derived} differs from stored box {instance.bbox_xyxy_px}")
        visibility = next(
            value.get("visibility") for value in raw["instances"] if value["instance_id"] == instance.instance_id
        )
        if visibility is not None and visibility.get("visible_pixel_count") is not None:
            if int(mask.sum()) != int(visibility["visible_pixel_count"]):
                raise ValueError(f"Visible pixel count differs from the logical RGBA mask for {instance.instance_id}")
    ignored = {key for key, label in mapping.items() if str(label).strip().upper() in {"BACKGROUND", "UNLABELLED"}}
    missing = set(mapping) - ignored - visible_sidecar_keys
    if missing:
        raise ValueError(f"Visible mapping keys have no pose sidecar entry: {sorted(missing)}")


def _validate_rigid_transform(transform: np.ndarray, atol: float, label: str) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(transform[3], (0, 0, 0, 1), atol=atol, rtol=0):
        raise ValueError(f"{label} has an invalid homogeneous row")
    _validate_rotation(transform[:3, :3], atol, label)


def _validate_rotation(rotation: np.ndarray, atol: float, label: str) -> None:
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{label} must be a finite 3x3 rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=atol
    ):
        raise ValueError(f"{label} is not a proper rotation in SO(3)")


def _validate_group_closure(transforms: Sequence[np.ndarray], atol: float, cad_id: str) -> None:
    for left in transforms:
        for right in transforms:
            if not any(np.allclose(left @ right, candidate, atol=atol, rtol=0) for candidate in transforms):
                raise ValueError(f"Discrete symmetry transforms are not closed for {cad_id}")
