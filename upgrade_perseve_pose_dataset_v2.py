#!/usr/bin/env python3
"""Upgrade a Perseve pose-v1 dataset to point-set pose-v2.

Deprecated: this script was created to migrate legacy Perseve pose-v1
datasets, which stored explicit symmetry labels, to the label-free point-set
pose-v2 contract.  Current Perseve versions generate pose-v2 datasets
directly, so this upgrader is retained only for existing historical v1 data
and should not be used for newly generated datasets.

The point-set batch preprocessor writes ``point_sets.json`` beside its NPZ
artifacts.  This script joins that manifest to the generated dataset catalog,
verifies that both use the same source meshes and canonical CAD frames, and
then prepares:

* a schema-v2 object catalog with point-set metadata and no legacy symmetry;
* schema-v2 frame sidecars with eligibility derived from the stored geometry;
* schema-v2 dataset metadata and bundled JSON Schemas; and
* point artifacts installed below the dataset root when they were generated
  elsewhere.

The default is a read-only dry run.  Pass ``--apply`` to install the upgrade.
Before changing metadata, apply mode creates a timestamped backup containing
the v1 JSON and schema files (large RGB, segmentation, USD, and NPZ files are
not duplicated).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import numpy as np

from muggled_sam.v3_sam.cad_pose.point_sets import (
    load_point_set_artifact,
    sha256_file,
)


SCHEMA_VERSION = "2.0.0"
Y_UP_TO_Z_UP = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
SCHEMA_FILENAMES = {
    "dataset_meta": "dataset-meta.schema.json",
    "objects": "objects.schema.json",
    "pose_annotations": "pose-annotations.schema.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--point-manifest",
        type=Path,
        default=None,
        help="point_sets.json from preprocess_abc_point_sets.py (default: DATASET_ROOT/cad_points/point_sets.json).",
    )
    parser.add_argument(
        "--install-point-dir",
        type=Path,
        default=Path("cad_points"),
        help="Dataset-relative directory recorded in objects.json (default: cad_points).",
    )
    parser.add_argument(
        "--schema-source",
        type=Path,
        default=Path(__file__).resolve().parent / "schemas" / "perseve-pose-v2",
        help="Directory containing the three bundled v2 JSON Schemas.",
    )
    parser.add_argument(
        "--geometry-atol-m",
        type=float,
        default=1e-6,
        help="Absolute tolerance when comparing point preprocessing and render catalog geometry.",
    )
    parser.add_argument(
        "--geometry-rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance when comparing point preprocessing and render catalog geometry.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Install the validated v2 metadata. Without this flag the script is read-only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    install_point_dir = _validated_relative_path(args.install_point_dir, "--install-point-dir")
    point_manifest_path = (
        args.point_manifest.expanduser().resolve()
        if args.point_manifest is not None
        else dataset_root / install_point_dir / "point_sets.json"
    )
    schema_source = args.schema_source.expanduser().resolve()

    metadata = _load_json(dataset_root / "dataset_meta.json")
    catalog = _load_json(dataset_root / "objects.json")
    point_manifest = _load_json(point_manifest_path)
    schemas = _load_schemas(schema_source)
    _validate_source_documents(metadata, catalog, point_manifest)

    upgraded_catalog, point_sources, catalog_migrations = upgrade_catalog(
        catalog,
        point_manifest,
        point_manifest_path,
        install_point_dir,
        atol=args.geometry_atol_m,
        rtol=args.geometry_rtol,
    )
    upgraded_metadata = upgrade_metadata(metadata, schemas)

    annotation_paths = sorted(dataset_root.glob("pose_annotations_*.json"))
    if not annotation_paths:
        raise ValueError(f"No pose_annotations_*.json files found in {dataset_root}")
    upgraded_annotations: dict[str, dict[str, Any]] = {}
    eligibility_reasons: Counter[str] = Counter()
    eligible_instances = 0
    total_instances = 0
    for annotation_path in annotation_paths:
        annotation = _load_json(annotation_path)
        upgraded, reasons = upgrade_annotation(
            annotation,
            upgraded_catalog["objects"],
            atol=args.geometry_atol_m,
            rtol=args.geometry_rtol,
        )
        expected_frame_id = annotation_path.stem.removeprefix("pose_annotations_")
        if upgraded.get("frame_id") != expected_frame_id:
            raise ValueError(
                f"{annotation_path.name} declares frame_id={upgraded.get('frame_id')!r}, "
                f"expected {expected_frame_id!r}"
            )
        _validate_json(upgraded, schemas["pose_annotations"], annotation_path.name)
        upgraded_annotations[annotation_path.name] = upgraded
        eligibility_reasons.update(reasons)
        total_instances += len(upgraded["instances"])
        eligible_instances += sum(bool(item["pose_training_eligible"]) for item in upgraded["instances"])

    _validate_json(upgraded_catalog, schemas["objects"], "objects.json")
    _validate_json(upgraded_metadata, schemas["dataset_meta"], "dataset_meta.json")
    if eligible_instances == 0:
        raise ValueError("The upgraded dataset would contain no pose-training-eligible instances")

    report = {
        "mode": "apply" if args.apply else "dry-run",
        "dataset_root": str(dataset_root),
        "source_schema_version": str(metadata.get("schema_version")),
        "target_schema_version": SCHEMA_VERSION,
        "catalog_objects": len(upgraded_catalog["objects"]),
        "point_artifacts": len(point_sources),
        "catalog_migrations": dict(sorted(catalog_migrations.items())),
        "frames": len(upgraded_annotations),
        "instances": total_instances,
        "pose_training_eligible_instances": eligible_instances,
        "ineligible_reasons": dict(sorted(eligibility_reasons.items())),
        "point_manifest": str(point_manifest_path),
        "install_point_dir": install_point_dir.as_posix(),
    }
    if not args.apply:
        print(json.dumps(report, indent=2, sort_keys=True))
        print("Dry run passed. Re-run with --apply to install the schema-v2 metadata.")
        return

    backup_dir = apply_upgrade(
        dataset_root=dataset_root,
        metadata=upgraded_metadata,
        catalog=upgraded_catalog,
        annotations=upgraded_annotations,
        schemas=schemas,
        point_sources=point_sources,
        point_manifest_path=point_manifest_path,
        install_point_dir=install_point_dir,
    )
    report["backup_dir"] = str(backup_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Upgrade installed. Run validate_perseve_pose_dataset.py against the split manifest next.")


def upgrade_catalog(
    catalog: Mapping[str, Any],
    point_manifest: Mapping[str, Any],
    point_manifest_path: Path,
    install_point_dir: Path,
    *,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], dict[str, Path], Counter[str]]:
    source_objects = catalog["objects"]
    point_objects = point_manifest["objects"]
    missing = sorted(set(source_objects) - set(point_objects))
    if missing:
        raise ValueError(
            f"Point-set manifest is missing {len(missing)} catalog objects; first missing IDs: {missing[:10]}"
        )

    upgraded_objects: dict[str, dict[str, Any]] = {}
    point_sources: dict[str, Path] = {}
    migrations: Counter[str] = Counter()
    for index, (cad_id, source) in enumerate(source_objects.items(), start=1):
        processed = point_objects[cad_id]
        if str(processed.get("mesh_sha256", "")).lower() != str(source["mesh_sha256"]).lower():
            raise ValueError(f"Source mesh checksum differs for CAD {cad_id}")
        _compare_scalar(
            float(source["source_to_meters"]),
            float(processed["source_to_meters"]),
            cad_id,
            "source_to_meters",
            atol=atol,
            rtol=rtol,
        )
        for field in (
            "bbox_min_m",
            "bbox_max_m",
            "base_dimensions_m",
        ):
            _compare_array(source[field], processed[field], cad_id, field, atol=atol, rtol=rtol)
        if source.get("canonical_origin") != processed.get("canonical_origin"):
            raise ValueError(f"Canonical origin differs for CAD {cad_id}")

        source_transform = np.asarray(source["T_cad_from_source_meters"], dtype=np.float64)
        processed_transform = np.asarray(processed["T_cad_from_source_meters"], dtype=np.float64)
        if np.allclose(source_transform, processed_transform, atol=atol, rtol=rtol):
            migrations["unchanged_canonical_frame"] += 1
        elif _is_legacy_y_up_to_z_up_migration(
            source,
            processed,
            atol=atol,
            rtol=rtol,
        ):
            migrations["legacy_y_up_to_z_up"] += 1
        else:
            difference = float(np.max(np.abs(source_transform - processed_transform)))
            raise ValueError(
                f"T_cad_from_source_meters differs for CAD {cad_id} and does not match the "
                "supported legacy Y-up to Z-up migration "
                f"(max_abs_difference={difference})"
            )

        point_raw = processed["point_set"]
        point_source = (point_manifest_path.parent / str(point_raw["path"])).resolve()
        if not point_source.is_file():
            raise FileNotFoundError(point_source)
        loaded = load_point_set_artifact(
            point_source,
            expected_sha256=str(point_raw["sha256"]),
            expected_point_count=int(point_raw["point_count"]),
            expected_centroid_m=np.asarray(point_raw["surface_centroid_m"], dtype=np.float64),
            centroid_atol_m=max(atol, 1e-10),
        )
        if len(loaded.query_indices) > len(loaded.points_m):
            raise ValueError(f"Invalid query subset for CAD {cad_id}")
        point_sources[cad_id] = point_source

        upgraded = {
            key: value
            for key, value in source.items()
            if key not in {"symmetry", "point_set"}
        }
        point_extensions: dict[str, Any] = {}
        if isinstance(processed.get("sampling_parameters"), Mapping):
            point_extensions["sampling_parameters"] = processed["sampling_parameters"]
        upgraded["point_set"] = {
            "path": (install_point_dir / f"{cad_id}.npz").as_posix(),
            "sha256": str(point_raw["sha256"]).lower(),
            "point_count": int(point_raw["point_count"]),
            "sampling_method": str(point_raw["sampling_method"]),
            "sampling_parameters_sha256": str(point_raw["sampling_parameters_sha256"]).lower(),
            "surface_centroid_m": [float(value) for value in point_raw["surface_centroid_m"]],
            **({"extensions": point_extensions} if point_extensions else {}),
        }
        if upgraded.get("extensions") is None:
            upgraded["extensions"] = {}
        # Point sampling is authoritative for the v2 canonical frame. In the
        # legacy ABC migration only the Y-up -> Z-up rotation changes; existing
        # T_cam_from_cad annotations already target this Z-up frame and remain
        # untouched.
        upgraded["T_cad_from_source_meters"] = processed_transform.tolist()
        upgraded_objects[cad_id] = upgraded
        if index % 250 == 0:
            print(f"Validated {index}/{len(source_objects)} point artifacts", flush=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "objects": upgraded_objects,
        "extensions": {
            "point_set_manifest_sha256": sha256_file(point_manifest_path),
            "canonical_frame_migrations": dict(sorted(migrations.items())),
            "pose_annotations_rewritten": False,
        },
    }, point_sources, migrations


def _is_legacy_y_up_to_z_up_migration(
    source: Mapping[str, Any],
    processed: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Recognize the known ABC v1 catalog omission without relaxing other joins."""

    parameters = processed.get("sampling_parameters")
    if not isinstance(parameters, Mapping):
        return False
    if str(parameters.get("source_up_axis", "")).upper() != "Y":
        return False
    if str(parameters.get("target_up_axis", "")).upper() != "Z":
        return False
    if not math.isclose(
        float(parameters.get("source_to_meters", math.nan)),
        float(processed["source_to_meters"]),
        abs_tol=atol,
        rel_tol=rtol,
    ):
        return False

    source_transform = np.asarray(source["T_cad_from_source_meters"], dtype=np.float64)
    processed_transform = np.asarray(processed["T_cad_from_source_meters"], dtype=np.float64)
    parameter_transform = np.asarray(parameters.get("T_cad_from_source_meters"), dtype=np.float64)
    if not _is_rigid_transform(source_transform, atol=max(atol, 1e-8)):
        return False
    if not _is_rigid_transform(processed_transform, atol=max(atol, 1e-8)):
        return False
    if parameter_transform.shape != (4, 4) or not np.allclose(
        parameter_transform,
        processed_transform,
        atol=atol,
        rtol=rtol,
    ):
        return False
    expected_rotation = Y_UP_TO_Z_UP @ source_transform[:3, :3]
    return bool(
        np.allclose(
            processed_transform[:3, :3],
            expected_rotation,
            atol=atol,
            rtol=rtol,
        )
        and np.allclose(
            processed_transform[:3, 3],
            source_transform[:3, 3],
            atol=atol,
            rtol=rtol,
        )
    )


def upgrade_metadata(
    metadata: Mapping[str, Any],
    schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    upgraded = dict(metadata)
    upgraded["schema_version"] = SCHEMA_VERSION
    upgraded["schemas"] = {
        name: f"schemas/{filename}" for name, filename in SCHEMA_FILENAMES.items()
    }
    extensions = dict(upgraded.get("extensions") or {})
    extensions["schema_sha256"] = {
        name: _sha256_json(schema) for name, schema in schemas.items()
    }
    extensions["point_set_upgrade"] = {
        "tool": Path(__file__).name,
        "schema_version": SCHEMA_VERSION,
    }
    upgraded["extensions"] = extensions
    return upgraded


def upgrade_annotation(
    annotation: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], Counter[str]]:
    upgraded = dict(annotation)
    upgraded["schema_version"] = SCHEMA_VERSION
    if upgraded.get("extensions") is None:
        upgraded.pop("extensions", None)
    reasons: Counter[str] = Counter()
    instances = []
    for raw_instance in annotation["instances"]:
        instance = dict(raw_instance)
        cad_id = str(instance["cad_id"])
        if cad_id not in catalog:
            raise ValueError(f"Frame {annotation.get('frame_id')} references unknown CAD {cad_id}")
        eligible, reason = _instance_is_eligible(instance, catalog[cad_id], atol=atol, rtol=rtol)
        instance["pose_training_eligible"] = eligible
        if instance.get("extensions") is None:
            instance["extensions"] = {}
        if not eligible:
            reasons[reason] += 1
        instances.append(instance)
    upgraded["instances"] = instances
    return upgraded, reasons


def _instance_is_eligible(
    instance: Mapping[str, Any],
    catalog_object: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, str]:
    if instance.get("annotation_state") != "visible":
        return False, str(instance.get("annotation_state") or "non_visible")
    for field in ("mask", "bbox_xyxy_px", "T_cam_from_cad", "render_scale_xyz", "dimensions_m"):
        if instance.get(field) is None:
            return False, f"missing_{field}"
    scale = np.asarray(instance["render_scale_xyz"], dtype=np.float64)
    dimensions = np.asarray(instance["dimensions_m"], dtype=np.float64)
    transform = np.asarray(instance["T_cam_from_cad"], dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        return False, "invalid_scale"
    if not np.allclose(scale, scale[0], atol=atol, rtol=rtol):
        return False, "non_uniform_scale"
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        return False, "invalid_dimensions"
    expected_dimensions = np.asarray(catalog_object["base_dimensions_m"], dtype=np.float64) * scale
    if not np.allclose(dimensions, expected_dimensions, atol=atol, rtol=rtol):
        return False, "dimension_mismatch"
    if not _is_rigid_transform(transform, atol=max(atol, 1e-8)):
        return False, "invalid_transform"
    centroid = np.asarray(catalog_object["point_set"]["surface_centroid_m"], dtype=np.float64)
    centroid_camera = transform[:3, :3] @ (scale[0] * centroid) + transform[:3, 3]
    if not np.isfinite(centroid_camera).all() or centroid_camera[2] <= 0:
        return False, "non_positive_centroid_depth"
    return True, "eligible"


def apply_upgrade(
    *,
    dataset_root: Path,
    metadata: Mapping[str, Any],
    catalog: Mapping[str, Any],
    annotations: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]],
    point_sources: Mapping[str, Path],
    point_manifest_path: Path,
    install_point_dir: Path,
) -> Path:
    install_root = dataset_root / install_point_dir
    if os.path.lexists(install_root) and not install_root.is_dir():
        raise FileExistsError(f"Point install directory is not a directory: {install_root}")
    point_install_plan = {
        cad_id: source
        for cad_id, source in point_sources.items()
        if _preflight_install_file(
            source,
            install_root / f"{cad_id}.npz",
            label=f"point artifact for CAD {cad_id}",
        )
    }
    manifest_destination = install_root / "point_sets.json"
    install_point_manifest = _preflight_install_file(
        point_manifest_path,
        manifest_destination,
        label="point_sets.json",
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = dataset_root / f"pose_v1_backup_{timestamp}"
    if backup_dir.exists():
        raise FileExistsError(backup_dir)
    staging_root = Path(tempfile.mkdtemp(prefix=".pose_v2_staging_", dir=dataset_root))
    try:
        _stage_json(staging_root / "dataset_meta.json", metadata)
        _stage_json(staging_root / "objects.json", catalog)
        for name, annotation in annotations.items():
            _stage_json(staging_root / name, annotation)
        for schema_name, filename in SCHEMA_FILENAMES.items():
            _stage_json(staging_root / "schemas" / filename, schemas[schema_name])

        for index, (cad_id, source) in enumerate(point_install_plan.items(), start=1):
            staged = staging_root / install_point_dir / f"{cad_id}.npz"
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            if index % 250 == 0:
                print(f"Staged {index}/{len(point_install_plan)} point artifacts", flush=True)
        staged_manifest = staging_root / install_point_dir / "point_sets.json"
        if install_point_manifest:
            staged_manifest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(point_manifest_path, staged_manifest)

        backup_dir.mkdir(parents=False)
        shutil.copy2(dataset_root / "dataset_meta.json", backup_dir / "dataset_meta.json")
        shutil.copy2(dataset_root / "objects.json", backup_dir / "objects.json")
        if (dataset_root / "schemas").is_dir():
            shutil.copytree(dataset_root / "schemas", backup_dir / "schemas")
        backup_annotations = backup_dir / "pose_annotations"
        backup_annotations.mkdir()
        for name in annotations:
            shutil.copy2(dataset_root / name, backup_annotations / name)

        install_root.mkdir(parents=True, exist_ok=True)
        staged_points = staging_root / install_point_dir
        if staged_points.is_dir():
            for staged in staged_points.iterdir():
                _install_staged_file(staged, install_root / staged.name)
        (dataset_root / "schemas").mkdir(parents=True, exist_ok=True)
        for filename in SCHEMA_FILENAMES.values():
            os.replace(staging_root / "schemas" / filename, dataset_root / "schemas" / filename)
        os.replace(staging_root / "objects.json", dataset_root / "objects.json")
        for name in annotations:
            os.replace(staging_root / name, dataset_root / name)
        # Metadata is the commit marker: install it only after every dependency.
        os.replace(staging_root / "dataset_meta.json", dataset_root / "dataset_meta.json")
    except Exception:
        print(f"Upgrade failed. Any completed metadata backup is at {backup_dir}", flush=True)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return backup_dir


def _preflight_install_file(source: Path, destination: Path, *, label: str) -> bool:
    """Return whether a copy is needed, refusing a non-identical destination."""

    if source.resolve() == destination.resolve():
        return False
    if not os.path.lexists(destination):
        return True
    if destination.is_file() and _files_identical(source, destination):
        return False
    raise FileExistsError(
        f"Refusing to overwrite different existing {label}: {destination}"
    )


def _install_staged_file(staged: Path, destination: Path) -> None:
    """Install a preflighted point file without replacing an existing path."""

    if not _preflight_install_file(staged, destination, label=staged.name):
        return
    # Staging is created below dataset_root, so a hard link gives us an atomic,
    # same-filesystem no-clobber install. The staging name is removed afterward.
    try:
        os.link(staged, destination)
    except FileExistsError:
        if not _preflight_install_file(staged, destination, label=staged.name):
            return
        raise
    staged.unlink()


def _files_identical(left: Path, right: Path) -> bool:
    left_stat = left.stat()
    right_stat = right.stat()
    return left_stat.st_size == right_stat.st_size and sha256_file(left) == sha256_file(right)


def _validate_source_documents(
    metadata: Mapping[str, Any],
    catalog: Mapping[str, Any],
    point_manifest: Mapping[str, Any],
) -> None:
    if str(metadata.get("schema_version", "")).split(".", 1)[0] != "1":
        raise ValueError(f"Expected pose-v1 dataset metadata, got {metadata.get('schema_version')!r}")
    if str(catalog.get("schema_version", "")).split(".", 1)[0] != "1":
        raise ValueError(f"Expected pose-v1 object catalog, got {catalog.get('schema_version')!r}")
    if not isinstance(catalog.get("objects"), Mapping) or not catalog["objects"]:
        raise ValueError("Object catalog is empty or malformed")
    if point_manifest.get("schema_version") != "1.0.0" or not isinstance(
        point_manifest.get("objects"), Mapping
    ):
        raise ValueError("Unsupported point_sets.json format")


def _load_schemas(root: Path) -> dict[str, dict[str, Any]]:
    schemas = {}
    for name, filename in SCHEMA_FILENAMES.items():
        path = root / filename
        schema = _load_json(path)
        jsonschema.Draft202012Validator.check_schema(schema)
        schemas[name] = schema
    return schemas


def _validate_json(document: Mapping[str, Any], schema: Mapping[str, Any], label: str) -> None:
    try:
        jsonschema.Draft202012Validator(schema).validate(document)
    except jsonschema.ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(f"{label} violates v2 schema at {location}: {error.message}") from error


def _compare_array(
    left: Any,
    right: Any,
    cad_id: str,
    field: str,
    *,
    atol: float,
    rtol: float,
) -> None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or not np.allclose(
        left_array, right_array, atol=atol, rtol=rtol
    ):
        difference = (
            math.inf
            if left_array.shape != right_array.shape
            else float(np.max(np.abs(left_array - right_array)))
        )
        raise ValueError(f"{field} differs for CAD {cad_id} (max_abs_difference={difference})")


def _compare_scalar(
    left: float,
    right: float,
    cad_id: str,
    field: str,
    *,
    atol: float,
    rtol: float,
) -> None:
    if not math.isclose(left, right, abs_tol=atol, rel_tol=rtol):
        raise ValueError(f"{field} differs for CAD {cad_id}: {left} != {right}")


def _is_rigid_transform(transform: np.ndarray, *, atol: float) -> bool:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return False
    if not np.allclose(transform[3], (0, 0, 0, 1), atol=atol, rtol=0):
        return False
    rotation = transform[:3, :3]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0)
        and math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=atol)
    )


def _validated_relative_path(path: Path, label: str) -> Path:
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{label} must be a path below the dataset root: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _stage_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
