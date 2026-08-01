"""Versioned loader for point-set pose schema v2 and legacy symmetry schema v1."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor

from .geometry import (
    adjust_intrinsics_for_resize_and_pad,
    pixel_to_normalized,
    project_points,
    surface_centroid_camera,
)
from .point_sets import LoadedPointSet, load_point_set_artifact
from .symmetry import VERIFIED_SYMMETRY_STATUSES
from .types import CADPoseTarget


SUPPORTED_SCHEMA_MAJORS = frozenset((1, 2))
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
class PointSetMetadata:
    path: Path
    sha256: str
    point_count: int
    sampling_method: str
    sampling_parameters_sha256: str
    surface_centroid_m: np.ndarray
    loaded: LoadedPointSet


@dataclass(frozen=True)
class CADCatalogObject:
    cad_id: str
    base_dimensions_m: np.ndarray
    bbox_min_m: np.ndarray
    bbox_max_m: np.ndarray
    source_to_meters: float
    T_cad_from_source_meters: np.ndarray
    point_set: PointSetMetadata | None = None
    symmetry: SymmetryMetadata | None = None

    @property
    def point_set_eligible(self) -> bool:
        return self.point_set is not None

    @property
    def legacy_rotation_eligible(self) -> bool:
        return self.symmetry is not None and self.symmetry.rotation_eligible


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
        return tuple(
            sorted(
                {
                    obj.symmetry.pipeline_version
                    for obj in self.catalog.values()
                    if obj.symmetry is not None
                }
            )
        )

    @property
    def point_set_checksums(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    obj.point_set.sha256
                    for obj in self.catalog.values()
                    if obj.point_set is not None
                }
            )
        )

    @property
    def sampling_pipeline_versions(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    obj.point_set.sampling_method
                    for obj in self.catalog.values()
                    if obj.point_set is not None
                }
            )
        )

    @property
    def sampling_parameter_checksums(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    obj.point_set.sampling_parameters_sha256
                    for obj in self.catalog.values()
                    if obj.point_set is not None
                }
            )
        )


@dataclass(frozen=True)
class _DatasetContract:
    dataset_meta: Mapping[str, Any]
    catalog: Mapping[str, CADCatalogObject]
    schema_paths: Mapping[str, Path]
    annotation_validator: Any
    schema_major: int
    dataset_meta_checksum: str
    catalog_checksum: str
    schema_checksums: Mapping[str, str]


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
    meta_stat, catalog_stat = meta_path.stat(), catalog_path.stat()
    contract = _load_dataset_contract_cached(
        str(dataset_root),
        meta_stat.st_mtime_ns,
        meta_stat.st_size,
        catalog_stat.st_mtime_ns,
        catalog_stat.st_size,
    )
    annotation_raw = _load_json(annotation_path)
    contract.annotation_validator.validate(annotation_raw)
    _require_supported_version(str(annotation_raw.get("schema_version", "")), "pose annotation")
    annotation_major = int(str(annotation_raw["schema_version"]).split(".", 1)[0])
    if annotation_major != contract.schema_major:
        raise ValueError(
            "Dataset metadata, catalog, and annotation schema majors differ: "
            f"dataset={contract.schema_major}, annotation={annotation_major}"
        )
    frame = _parse_frame(
        annotation_raw,
        annotation_path,
        contract.catalog,
        contract.dataset_meta,
    )
    if frame.frame_id != str(frame_id):
        raise ValueError(f"Sidecar frame_id {frame.frame_id!r} does not match manifest frame {frame_id!r}")
    if validate_pixels:
        _validate_frame_pixels(frame, annotation_raw, camera_root)
    return PersevePoseSample(
        frame,
        dataset_root,
        contract.catalog,
        contract.dataset_meta,
        contract.dataset_meta_checksum,
        contract.catalog_checksum,
        sha256_file(annotation_path),
        contract.schema_checksums,
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
        raise ValueError(f"Perseve pose v1/v2 requires a four-channel instance PNG: {instance_image_path}")
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


def effective_surface_centroid_m(
    instance: PoseInstance,
    catalog_object: CADCatalogObject,
) -> np.ndarray:
    """Return ``s * mu`` for point-set data, or zero for legacy AABB targets."""

    if catalog_object.point_set is None:
        return np.zeros(3, dtype=np.float64)
    if instance.render_scale_xyz is None or not np.allclose(
        instance.render_scale_xyz,
        instance.render_scale_xyz[0],
        atol=1e-8,
        rtol=1e-6,
    ):
        raise ValueError(f"Instance {instance.instance_id} does not have a valid uniform scale")
    return float(instance.render_scale_xyz[0]) * catalog_object.point_set.surface_centroid_m


def make_pose_target(
    instance: PoseInstance,
    catalog_object: CADCatalogObject,
    intrinsics: Tensor,
    source_size_wh: tuple[int, int],
    model_size_wh: tuple[int, int],
) -> CADPoseTarget:
    """Derive centroid/depth and point-set targets, with a v1 legacy fallback."""

    if not instance.pose_training_eligible or instance.T_cam_from_cad is None or instance.dimensions_m is None:
        raise ValueError(f"Instance {instance.instance_id} is not eligible for pose training")
    adjusted_k = adjust_intrinsics_for_resize_and_pad(intrinsics, source_size_wh, model_size_wh)
    translation = torch.as_tensor(instance.translation_m, dtype=intrinsics.dtype, device=intrinsics.device)
    rotation = torch.as_tensor(instance.rotation_matrix, dtype=intrinsics.dtype, device=intrinsics.device)
    dimensions = torch.as_tensor(instance.dimensions_m, dtype=intrinsics.dtype, device=intrinsics.device)
    if catalog_object.point_set is not None:
        if instance.render_scale_xyz is None or not np.allclose(
            instance.render_scale_xyz,
            instance.render_scale_xyz[0],
            atol=1e-8,
            rtol=1e-6,
        ):
            raise ValueError(f"Instance {instance.instance_id} requires uniform scale for point-set supervision")
        scale = float(instance.render_scale_xyz[0])
        point_set = catalog_object.point_set
        effective_centroid = torch.as_tensor(
            scale * point_set.surface_centroid_m,
            dtype=intrinsics.dtype,
            device=intrinsics.device,
        )
        centroid = surface_centroid_camera(rotation, translation, effective_centroid)
        if centroid[2] <= 0:
            raise ValueError(f"Instance {instance.instance_id} has non-positive surface-centroid depth")
        center_norm = pixel_to_normalized(project_points(centroid, adjusted_k), model_size_wh)
        centered_dense = torch.as_tensor(
            scale * (point_set.loaded.points_m - point_set.surface_centroid_m),
            dtype=intrinsics.dtype,
            device=intrinsics.device,
        )
        query_indices = torch.tensor(
            np.asarray(point_set.loaded.query_indices).copy(),
            dtype=torch.long,
            device=intrinsics.device,
        )
        return CADPoseTarget(
            center_uv_norm=center_norm,
            log_depth=centroid[2].log(),
            rotation_matrix=rotation,
            translation_m=translation,
            dimensions_m=dimensions,
            centroid_m=centroid,
            effective_surface_centroid_m=effective_centroid,
            point_query_m=centered_dense[query_indices],
            point_target_m=centered_dense,
            point_set_eligible=True,
            rotation_eligible=True,
        )

    if translation[2] <= 0:
        raise ValueError(f"Instance {instance.instance_id} has non-positive camera-axis depth")
    center_norm = pixel_to_normalized(project_points(translation, adjusted_k), model_size_wh)
    symmetry = catalog_object.symmetry
    if symmetry is None:
        raise ValueError(f"CAD {catalog_object.cad_id} has neither point-set nor legacy symmetry metadata")
    transforms = torch.as_tensor(
        np.stack(symmetry.transforms),
        dtype=intrinsics.dtype,
        device=intrinsics.device,
    )
    axis = (
        None
        if symmetry.axis_cad is None
        else torch.as_tensor(symmetry.axis_cad, dtype=intrinsics.dtype, device=intrinsics.device)
    )
    return CADPoseTarget(
        center_uv_norm=center_norm,
        log_depth=translation[2].log(),
        rotation_matrix=rotation,
        translation_m=translation,
        dimensions_m=dimensions,
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


@lru_cache(maxsize=8)
def _load_dataset_contract_cached(
    dataset_root_string: str,
    metadata_mtime_ns: int,
    metadata_size: int,
    catalog_mtime_ns: int,
    catalog_size: int,
) -> _DatasetContract:
    """Load and validate immutable dataset-level state once per file version."""

    del metadata_mtime_ns, metadata_size, catalog_mtime_ns, catalog_size
    try:
        import jsonschema
    except ModuleNotFoundError as exc:
        raise RuntimeError("Perseve pose loading requires the 'jsonschema' package") from exc
    dataset_root = Path(dataset_root_string)
    metadata_path = dataset_root / "dataset_meta.json"
    catalog_path = dataset_root / "objects.json"
    metadata = _load_json(metadata_path)
    catalog_raw = _load_json(catalog_path)
    schema_paths = _resolve_schema_paths(dataset_root, metadata)
    schema_documents = {name: _load_json(path) for name, path in schema_paths.items()}
    validators = {
        name: jsonschema.Draft202012Validator(schema)
        for name, schema in schema_documents.items()
    }
    validators["dataset_meta"].validate(metadata)
    validators["objects"].validate(catalog_raw)
    _require_supported_version(str(metadata.get("schema_version", "")), "dataset metadata")
    _require_supported_version(str(catalog_raw.get("schema_version", "")), "object catalog")
    schema_majors = {
        int(str(value.get("schema_version", "")).split(".", 1)[0])
        for value in (metadata, catalog_raw)
    }
    if len(schema_majors) != 1:
        raise ValueError(
            f"Dataset metadata and catalog schema majors differ: {schema_majors}"
        )
    parsed_catalog = _parse_catalog(catalog_raw, metadata, dataset_root)
    return _DatasetContract(
        dataset_meta=metadata,
        catalog=parsed_catalog,
        schema_paths=schema_paths,
        annotation_validator=validators["pose_annotations"],
        schema_major=next(iter(schema_majors)),
        dataset_meta_checksum=sha256_file(metadata_path),
        catalog_checksum=sha256_file(catalog_path),
        schema_checksums={
            name: sha256_file(path) for name, path in schema_paths.items()
        },
    )


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
    if major not in SUPPORTED_SCHEMA_MAJORS:
        raise ValueError(
            f"Unsupported {kind} schema major {major}; expected one of {sorted(SUPPORTED_SCHEMA_MAJORS)}"
        )


def _parse_catalog(
    raw: Mapping[str, Any],
    meta: Mapping[str, Any],
    dataset_root: Path,
) -> dict[str, CADCatalogObject]:
    rotation_atol = float(meta.get("validation_tolerances", {}).get("rotation_atol", 1e-6))
    dimension_atol = float(meta.get("validation_tolerances", {}).get("dimension_atol_m", 1e-6))
    objects: dict[str, CADCatalogObject] = {}
    for cad_id, value in raw["objects"].items():
        bbox_min = np.asarray(value["bbox_min_m"], dtype=np.float64)
        bbox_max = np.asarray(value["bbox_max_m"], dtype=np.float64)
        dimensions = np.asarray(value["base_dimensions_m"], dtype=np.float64)
        if not np.allclose(bbox_max - bbox_min, dimensions, atol=dimension_atol, rtol=1e-5):
            raise ValueError(f"Catalog dimensions do not match bounds for {cad_id}")
        canonical_transform = np.asarray(value["T_cad_from_source_meters"], dtype=np.float64)
        _validate_rigid_transform(canonical_transform, rotation_atol, f"{cad_id} source transform")
        point_set = None
        point_raw = value.get("point_set")
        if point_raw is not None:
            centroid = np.asarray(point_raw["surface_centroid_m"], dtype=np.float64)
            artifact_path = (dataset_root / str(point_raw["path"])).resolve()
            try:
                artifact_path.relative_to(dataset_root.resolve())
            except ValueError as exc:
                raise ValueError(f"Point-set path for {cad_id} escapes the dataset root") from exc
            loaded = load_point_set_artifact(
                artifact_path,
                expected_sha256=str(point_raw["sha256"]),
                expected_point_count=int(point_raw["point_count"]),
                expected_centroid_m=centroid,
                centroid_atol_m=max(dimension_atol, 1e-10),
            )
            point_set = PointSetMetadata(
                artifact_path,
                str(point_raw["sha256"]),
                int(point_raw["point_count"]),
                str(point_raw["sampling_method"]),
                str(point_raw["sampling_parameters_sha256"]),
                centroid,
                loaded,
            )

        symmetry = None
        symmetry_raw = value.get("symmetry")
        if symmetry_raw is not None:
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
                axis_np is None
                or not np.isclose(np.linalg.norm(axis_np), 1.0, atol=rotation_atol, rtol=0)
            ):
                raise ValueError(f"Continuous symmetry axis for {cad_id} must be unit length")
            symmetry = SymmetryMetadata(
                symmetry_type,
                transforms,
                axis_np,
                str(symmetry_raw["label_source"]),
                str(symmetry_raw["status"]),
                str(symmetry_raw["pipeline_version"]),
                str(symmetry_raw["parameters_sha256"]),
            )
        if point_set is None and symmetry is None:
            raise ValueError(f"CAD {cad_id} has neither point_set nor legacy symmetry metadata")
        objects[cad_id] = CADCatalogObject(
            cad_id,
            dimensions,
            bbox_min,
            bbox_max,
            float(value["source_to_meters"]),
            canonical_transform,
            point_set,
            symmetry,
        )
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
    schema_major = int(str(raw["schema_version"]).split(".", 1)[0])
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
        if eligible and state != "visible":
            raise ValueError(f"Non-visible instance marked pose_training_eligible: {instance_id}")
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
        catalog_object = catalog[cad_id]
        if schema_major >= 2:
            if eligible and catalog_object.point_set is None:
                raise ValueError(f"Pose-eligible v2 instance {instance_id} has no valid point-set artifact")
            if eligible and (
                scale is None or not np.allclose(scale, scale[0], atol=atol, rtol=rtol)
            ):
                raise ValueError(f"Pose-eligible point-set instance {instance_id} requires uniform scale")
        elif eligible and not catalog_object.legacy_rotation_eligible:
            raise ValueError(f"Unverified legacy symmetry marked pose_training_eligible for {instance_id}")
        if eligible and (
            transform is None
            or (schema_major >= 2 and scale is None)
            or dimensions is None
            or bbox is None
            or mask_rgba is None
        ):
            raise ValueError(f"Pose-eligible instance {instance_id} is missing required geometry or mask fields")
        if eligible:
            if catalog_object.point_set is not None:
                effective_centroid = scale[0] * catalog_object.point_set.surface_centroid_m
                centroid_camera = transform[:3, :3] @ effective_centroid + transform[:3, 3]
                if centroid_camera[2] <= 0:
                    raise ValueError(f"Pose-eligible instance {instance_id} has non-positive centroid depth")
            elif transform[2, 3] <= 0:
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
