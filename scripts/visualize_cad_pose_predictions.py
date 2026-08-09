#!/usr/bin/env python3
"""Render a few held-out CAD-pose predictions beside their ground truth.

The inference, mask NMS, and one-to-one assignment path mirrors
``run_cad_pose_eval``. Each visualization type is written to a separate PNG,
along with JSON metrics; this command does not modify a checkpoint or dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_manifest import ManifestRow, load_manifest, manifest_sha256
from finetune_image_exemplar_multi_gt import (
    build_exemplar_tokens_for_object,
    build_gt_down_list,
    cache_exemplar,
    entries_from_manifest_rows,
    generate_detections_train,
    load_bgr,
    load_reference_images,
    pad_exemplar_batch,
    parse_ref_view_ids,
    restore_finetune_checkpoint,
    unique_frame_entries,
)
from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.v3_sam.exemplar_view_pose import EXEMPLAR_VIEW_MODES
from muggled_sam.v3_sam.cad_pose.dataset import (
    effective_surface_centroid_m,
    instance_mask_rgba,
    load_perseve_pose_sample,
    make_pose_target,
)
from muggled_sam.v3_sam.cad_pose.evaluation import PoseEvaluation, evaluate_pose_matches
from muggled_sam.v3_sam.cad_pose.geometry import adjust_intrinsics_for_resize_and_pad
from muggled_sam.v3_sam.cad_pose.inference import mask_nms_indices
from muggled_sam.v3_sam.cad_pose.losses import CADPoseLossConfig
from muggled_sam.v3_sam.cad_pose.matching import match_pose_predictions_one_to_one
from muggled_sam.v3_sam.cad_pose.visualization import (
    GT_COLOR_BGR,
    PRED_COLOR_BGR,
    PoseOverlay,
    draw_mask_contour,
    draw_mask_error_overlay,
    draw_pose_axes,
    draw_pose_box,
    draw_projected_points,
    resize_binary_mask,
)


DEFAULT_RUN_DIR = REPO_ROOT / "runs/cad_pose_deep_prompt_joint_lite/run_20260805_020155"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "checkpoints/finetune_epoch_066.pth"
DEFAULT_QUICK_FRAMES = ("0017", "0029", "0036")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GT and predicted CAD poses on a few manifest scenes."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--allow-eval-manifest-mismatch",
        action="store_true",
        help=(
            "Allow an explicitly supplied evaluation manifest whose checksum differs from "
            "the checkpoint's training manifest. Intended for held-out evaluation only."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument(
        "--frame",
        action="append",
        default=[],
        help="Exact frame id to render; repeat for multiple frames. Overrides quick defaults.",
    )
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=("auto", "bf16", "float32"), default="auto")
    parser.add_argument("--max-side-length", type=int, default=None)
    parser.add_argument("--ref-view-ids", type=str, default=None)
    parser.add_argument("--num-points-approx", type=int, default=None)
    parser.add_argument("--nms-iou", type=float, default=None)
    parser.add_argument("--det-filter", type=float, default=None)
    parser.add_argument("--max-render-points", type=int, default=512)
    parser.add_argument("--line-thickness", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.num_scenes <= 0:
        raise ValueError("--num-scenes must be positive")
    if args.max_render_points <= 0:
        raise ValueError("--max-render-points must be positive")

    # Load once on CPU: its metadata resolves all training-time data/model
    # settings, and the same in-memory state is then restored into the model.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = _checkpoint_args(checkpoint)
    if not bool(checkpoint_args.get("enable_pose", False)):
        raise ValueError(f"Checkpoint does not declare a trained pose head: {checkpoint_path}")

    model_path = _resolve_path(args.model_path, checkpoint_args.get("model_path"), "model path")
    data_root = _resolve_path(args.data_root, checkpoint_args.get("data_root"), "data root")
    manifest_path = _resolve_path(
        args.manifest, checkpoint_args.get("dataset_manifest"), "dataset manifest"
    )
    reference_dir = _resolve_path(
        args.reference_dir, checkpoint_args.get("reference_dir"), "reference directory"
    )
    for required_path in (model_path, manifest_path, data_root, reference_dir):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    _verify_manifest(
        checkpoint,
        manifest_path,
        allow_mismatch=args.allow_eval_manifest_mismatch,
    )

    rows, _ = load_manifest(manifest_path, data_root, validate_files=True)
    selected_rows = _select_scene_rows(
        rows,
        split=args.split,
        requested_frames=args.frame,
        num_scenes=args.num_scenes,
        seed=args.seed,
    )
    entries = unique_frame_entries(
        entries_from_manifest_rows(selected_rows, data_root, object_level=False)
    )
    if not entries:
        raise RuntimeError("Selected manifest scenes yielded no pose-eligible frame entries")

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    view_mode = str(checkpoint_args.get("exemplar_view_mode", "none"))
    if view_mode not in EXEMPLAR_VIEW_MODES:
        raise ValueError(f"Unsupported checkpoint exemplar-view mode: {view_mode!r}")
    _, base_model = make_sam_from_state_dict(str(model_path))
    base_model.to(device=device, dtype=dtype)
    detmodel = base_model.make_detector_model()
    detmodel.to(device=device, dtype=dtype)
    restore_finetune_checkpoint(
        checkpoint,
        detmodel,
        optimizer=None,
        load_optimizer=False,
        exemplar_view_mode=view_mode,
    )
    detmodel.cad_pose_head.set_cad_prompt_enabled(
        bool(checkpoint_args.get("enable_cad_prompt", False))
    )
    detmodel.eval()
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    pose_config = CADPoseLossConfig(**dict(checkpoint.get("pose_config") or {}))
    del checkpoint, base_model

    run_label = f"epoch_{checkpoint_epoch:03d}" if checkpoint_epoch >= 0 else checkpoint_path.stem
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent.parent / f"pose_visualizations_{run_label}_separate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_view_ids = parse_ref_view_ids(
        args.ref_view_ids
        if args.ref_view_ids is not None
        else str(checkpoint_args.get("ref_view_ids", ""))
    )
    if not ref_view_ids:
        raise ValueError("No reference view ids were resolved")
    max_side_length = int(
        args.max_side_length
        if args.max_side_length is not None
        else checkpoint_args.get("max_side_length", 1008)
    )
    num_points_approx = int(
        args.num_points_approx
        if args.num_points_approx is not None
        else checkpoint_args.get("num_points_approx", 25)
    )
    nms_iou = float(
        args.nms_iou if args.nms_iou is not None else checkpoint_args.get("nms_iou", 0.5)
    )
    det_filter = float(
        args.det_filter
        if args.det_filter is not None
        else checkpoint_args.get("det_filter", 0.0)
    )
    use_square_sizing = not bool(checkpoint_args.get("no_square", False))
    shuffle_seed = int(checkpoint_args.get("exemplar_view_shuffle_seed", 42))
    min_match_iou = float(checkpoint_args.get("pose_eval_min_match_iou", 0.0))

    print(f"checkpoint: {checkpoint_path} (epoch {checkpoint_epoch})")
    print(f"device/dtype: {device} / {dtype}")
    print("frames: " + ", ".join(entry["frame_id"] for entry in entries))
    print(f"output: {output_dir}")

    ref_cache: dict[str, object] = {}
    ref_image_cache: dict[str, list[np.ndarray]] = {}
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for scene_index, entry in enumerate(entries):
            scene_records = _visualize_entry(
                detmodel=detmodel,
                entry=entry,
                scene_index=scene_index,
                reference_dir=reference_dir,
                ref_view_ids=ref_view_ids,
                ref_cache=ref_cache,
                ref_image_cache=ref_image_cache,
                device=device,
                pose_config=pose_config,
                output_dir=output_dir,
                max_side_length=max_side_length,
                use_square_sizing=use_square_sizing,
                num_points_approx=num_points_approx,
                view_mode=view_mode,
                shuffle_seed=shuffle_seed,
                det_filter=det_filter,
                nms_iou=nms_iou,
                min_match_iou=min_match_iou,
                max_render_points=args.max_render_points,
                line_thickness=args.line_thickness,
            )
            records.extend(scene_records)
            matched = sum(record["status"] == "matched" for record in scene_records)
            print(
                f"[{scene_index + 1}/{len(entries)}] frame {entry['frame_id']}: "
                f"{matched}/{len(scene_records)} GT instances matched"
            )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256(manifest_path),
        "data_root": str(data_root),
        "reference_dir": str(reference_dir),
        "split": args.split,
        "frames": [entry["frame_id"] for entry in entries],
        "records": records,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    matched_count = sum(record["status"] == "matched" for record in records)
    print(
        f"wrote {len(records)} visualization sets ({matched_count} matched) "
        f"and {summary_path}"
    )


def _visualize_entry(
    *,
    detmodel,
    entry: Mapping[str, str],
    scene_index: int,
    reference_dir: Path,
    ref_view_ids: list[str],
    ref_cache: dict[str, object],
    ref_image_cache: dict[str, list[np.ndarray]],
    device: torch.device,
    pose_config: CADPoseLossConfig,
    output_dir: Path,
    max_side_length: int,
    use_square_sizing: bool,
    num_points_approx: int,
    view_mode: str,
    shuffle_seed: int,
    det_filter: float,
    nms_iou: float,
    min_match_iou: float,
    max_render_points: int,
    line_thickness: int,
) -> list[dict[str, object]]:
    image_bgr = load_bgr(entry["rgb_path"])
    camera_root = Path(entry["inst_path"]).parent
    sample = load_perseve_pose_sample(camera_root, entry["frame_id"], validate_pixels=True)
    grouped: dict[str, list[object]] = defaultdict(list)
    for instance in sample.frame.eligible_instances():
        grouped[instance.cad_id].append(instance)

    image_tensor = detmodel.image_encoder.prepare_image(
        image_bgr,
        max_side_length=max_side_length,
        use_square_sizing=use_square_sizing,
    )
    encoded = detmodel.image_encoder(image_tensor)
    features = detmodel.image_projection.v3_projection(encoded)
    pre_h, pre_w = image_tensor.shape[-2:]
    model_dtype = image_tensor.dtype
    raw_k = torch.as_tensor(sample.frame.intrinsics, device=device, dtype=model_dtype)
    adjusted_k = adjust_intrinsics_for_resize_and_pad(
        raw_k, sample.frame.image_size_wh, (pre_w, pre_h)
    ).unsqueeze(0)
    raw_k_np = np.asarray(sample.frame.intrinsics, dtype=np.float64)
    original_size_wh = (image_bgr.shape[1], image_bgr.shape[0])
    display_ref_view_ids = _spread_reference_view_ids(ref_view_ids, count=4)
    records: list[dict[str, object]] = []

    for cad_id, instances in sorted(grouped.items()):
        gt_masks = [
            instance_mask_rgba(Path(entry["inst_path"]), instance).astype(bool)
            for instance in instances
        ]
        catalog_object = sample.catalog[cad_id]
        if cad_id not in ref_cache:
            exemplar = build_exemplar_tokens_for_object(
                detmodel,
                cad_id,
                reference_dir,
                ref_view_ids,
                max_side_length,
                use_square_sizing,
                num_points_approx,
                device,
                include_view_metadata=view_mode != "none",
            )
            if exemplar is not None:
                ref_cache[cad_id] = cache_exemplar(exemplar)
        if cad_id not in ref_image_cache:
            ref_image_cache[cad_id] = load_reference_images(
                cad_id,
                reference_dir,
                display_ref_view_ids,
                max_views=4,
            )
        exemplar_images = ref_image_cache[cad_id]
        exemplar = ref_cache.get(cad_id)
        if exemplar is None:
            for gt_index, instance in enumerate(instances):
                records.append(
                    _write_visualizations(
                        image_bgr=image_bgr,
                        intrinsics=raw_k_np,
                        sample=sample,
                        instance=instance,
                        gt_mask=gt_masks[gt_index],
                        prediction=None,
                        prediction_mask=None,
                        evaluation=None,
                        mask_iou=None,
                        detection_score=None,
                        pose_score=None,
                        output_dir=output_dir,
                        entry=entry,
                        scene_index=scene_index,
                        cad_id=cad_id,
                        status_reason="missing exemplar views",
                        max_render_points=max_render_points,
                        line_thickness=line_thickness,
                        exemplar_images=exemplar_images,
                    )
                )
            continue

        dimensions = torch.as_tensor(
            instances[0].dimensions_m, device=device, dtype=model_dtype
        ).unsqueeze(0)
        effective_centroid = torch.as_tensor(
            effective_surface_centroid_m(instances[0], catalog_object),
            device=device,
            dtype=model_dtype,
        ).unsqueeze(0)
        exemplar_batch, exemplar_padding_mask = pad_exemplar_batch(
            [exemplar],
            device=device,
            pose_encoder=detmodel.exemplar_view_pose_encoder,
            mode=view_mode,
            shuffle_seed=shuffle_seed,
        )
        outputs = generate_detections_train(
            detmodel,
            features,
            exemplar_batch,
            detection_filter_threshold=det_filter,
            exemplar_padding_mask_bn=exemplar_padding_mask,
            cad_dimensions_m_b3=dimensions,
            cad_effective_surface_centroid_m_b3=effective_centroid,
            adjusted_intrinsics_b33=adjusted_k,
            model_image_size_wh=(pre_w, pre_h),
            return_pose=True,
        )
        mask_predictions, _, detection_scores, _, _, pose_predictions = outputs
        logits_nhw = mask_predictions[0].float()
        scores_n = detection_scores[0]
        retained = mask_nms_indices(logits_nhw, scores_n, nms_iou)
        logits_nhw = logits_nhw[retained]
        scores_n = scores_n[retained]
        pose_predictions = pose_predictions.index_candidates(retained)
        gt_targets_down = build_gt_down_list(
            gt_masks, (pre_h, pre_w), logits_nhw.shape[-2:], device
        )
        matches, match_iou = match_pose_predictions_one_to_one(logits_nhw, gt_targets_down)
        pose_targets = [
            make_pose_target(
                instance,
                catalog_object,
                raw_k,
                sample.frame.image_size_wh,
                (pre_w, pre_h),
            )
            for instance in instances
        ]

        matched_gt_indices: set[int] = set()
        for gt_index, prediction_index in matches:
            matched_gt_indices.add(gt_index)
            evaluation = evaluate_pose_matches(
                pose_predictions,
                pose_targets,
                [(gt_index, prediction_index)],
                centroid_tolerance=pose_config.centroid_tolerance,
                point_set_tolerance=pose_config.point_set_tolerance,
                point_distance_chunk_size=pose_config.point_distance_chunk_size,
                rotation_tolerance_rad=pose_config.rotation_tolerance_rad,
                translation_tolerance=pose_config.translation_tolerance,
                normalize_translation_error=pose_config.normalize_translation_error,
            )
            prediction_mask = resize_binary_mask(
                logits_nhw[prediction_index].detach().cpu().numpy() > 0.0,
                original_size_wh,
            )
            records.append(
                _write_visualizations(
                    image_bgr=image_bgr,
                    intrinsics=raw_k_np,
                    sample=sample,
                    instance=instances[gt_index],
                    gt_mask=gt_masks[gt_index],
                    prediction=(
                        pose_predictions.rotation_matrix_bn33[0, prediction_index]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        pose_predictions.translation_m_bn3[0, prediction_index]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                    ),
                    prediction_mask=prediction_mask,
                    evaluation=evaluation,
                    mask_iou=float(match_iou[gt_index, prediction_index]),
                    detection_score=float(scores_n[prediction_index]),
                    pose_score=float(pose_predictions.pose_score_bn[0, prediction_index]),
                    output_dir=output_dir,
                    entry=entry,
                    scene_index=scene_index,
                    cad_id=cad_id,
                    status_reason=(
                        f"low mask IoU (<{min_match_iou:.2f})"
                        if min_match_iou > 0.0
                        and float(match_iou[gt_index, prediction_index]) < min_match_iou
                        else None
                    ),
                    max_render_points=max_render_points,
                    line_thickness=line_thickness,
                    exemplar_images=exemplar_images,
                )
            )
        for gt_index, instance in enumerate(instances):
            if gt_index in matched_gt_indices:
                continue
            records.append(
                _write_visualizations(
                    image_bgr=image_bgr,
                    intrinsics=raw_k_np,
                    sample=sample,
                    instance=instance,
                    gt_mask=gt_masks[gt_index],
                    prediction=None,
                    prediction_mask=None,
                    evaluation=None,
                    mask_iou=None,
                    detection_score=None,
                    pose_score=None,
                    output_dir=output_dir,
                    entry=entry,
                    scene_index=scene_index,
                    cad_id=cad_id,
                    status_reason="no one-to-one mask match",
                    max_render_points=max_render_points,
                    line_thickness=line_thickness,
                    exemplar_images=exemplar_images,
                )
            )
    return records


def _write_visualizations(
    *,
    image_bgr: np.ndarray,
    intrinsics: np.ndarray,
    sample,
    instance,
    gt_mask: np.ndarray,
    prediction: tuple[np.ndarray, np.ndarray] | None,
    prediction_mask: np.ndarray | None,
    evaluation: PoseEvaluation | None,
    mask_iou: float | None,
    detection_score: float | None,
    pose_score: float | None,
    output_dir: Path,
    entry: Mapping[str, str],
    scene_index: int,
    cad_id: str,
    status_reason: str | None,
    max_render_points: int,
    line_thickness: int,
    exemplar_images: Sequence[np.ndarray],
) -> dict[str, object]:
    catalog_object = sample.catalog[cad_id]
    points_cad_m = None
    if catalog_object.point_set is not None:
        scale = float(instance.render_scale_xyz[0])
        points_cad_m = scale * catalog_object.point_set.loaded.points_m
    gt_overlay = PoseOverlay(
        label=f"GT | {instance.instance_id}",
        rotation_cam_from_cad=np.asarray(instance.rotation_matrix),
        translation_cam_from_cad_m=np.asarray(instance.translation_m),
        dimensions_m=np.asarray(instance.dimensions_m),
        color_bgr=GT_COLOR_BGR,
        mask_hw=gt_mask,
        points_cad_m=points_cad_m,
    )
    pred_overlay = None
    if prediction is not None:
        pred_overlay = PoseOverlay(
            label=f"PRED | det {detection_score:.3f} pose {pose_score:.3f}",
            rotation_cam_from_cad=prediction[0],
            translation_cam_from_cad_m=prediction[1],
            dimensions_m=np.asarray(instance.dimensions_m),
            color_bgr=PRED_COLOR_BGR,
            mask_hw=prediction_mask,
            points_cad_m=points_cad_m,
        )
    safe_cad = re.sub(r"[^A-Za-z0-9_.-]+", "_", cad_id)[:80]
    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance.instance_id))[:40]
    directory_name = (
        f"scene_{scene_index + 1:02d}_frame_{entry['frame_id']}_"
        f"{safe_cad}_{safe_instance}"
    )
    instance_dir = output_dir / directory_name
    instance_dir.mkdir(parents=True, exist_ok=True)

    missing_prediction = _missing_prediction_image(image_bgr, status_reason)
    gt_dimensions = draw_pose_box(
        image_bgr,
        gt_overlay,
        intrinsics,
        line_thickness=line_thickness,
    )
    gt_dimensions = _add_dimension_label(gt_dimensions, gt_overlay.dimensions_m)
    image_outputs: dict[str, np.ndarray] = {
        "rgb": image_bgr.copy(),
        "mask_gt": draw_mask_contour(
            image_bgr, gt_mask, GT_COLOR_BGR, alpha=0.35, thickness=line_thickness
        ),
        "mask_pred": (
            draw_mask_contour(
                image_bgr,
                prediction_mask,
                PRED_COLOR_BGR,
                alpha=0.35,
                thickness=line_thickness,
            )
            if prediction_mask is not None
            else missing_prediction
        ),
        "dimensions_gt": gt_dimensions,
        "dimensions_pred": missing_prediction,
        "orientation_gt": draw_pose_axes(
            image_bgr,
            gt_overlay,
            intrinsics,
            line_thickness=line_thickness,
        ),
        "orientation_pred": missing_prediction,
        "surface_gt": (
            draw_projected_points(
                image_bgr,
                points_cad_m,
                gt_overlay.rotation_cam_from_cad,
                gt_overlay.translation_cam_from_cad_m,
                intrinsics,
                GT_COLOR_BGR,
                max_points=max_render_points,
            )
            if points_cad_m is not None
            else _unavailable_image(image_bgr, "CAD SURFACE POINTS UNAVAILABLE")
        ),
        "surface_pred": missing_prediction,
        "mask_error": (
            _add_corner_label(
                draw_mask_error_overlay(image_bgr, gt_mask, prediction_mask),
                "TP green | FP magenta | FN orange",
            )
            if prediction_mask is not None
            else missing_prediction
        ),
    }
    if pred_overlay is not None:
        pred_dimensions = draw_pose_box(
            image_bgr,
            pred_overlay,
            intrinsics,
            line_thickness=line_thickness,
        )
        image_outputs["dimensions_pred"] = _add_dimension_label(
            pred_dimensions, pred_overlay.dimensions_m
        )
        image_outputs["orientation_pred"] = draw_pose_axes(
            image_bgr,
            pred_overlay,
            intrinsics,
            line_thickness=line_thickness,
        )
        image_outputs["surface_pred"] = (
            draw_projected_points(
                image_bgr,
                points_cad_m,
                pred_overlay.rotation_cam_from_cad,
                pred_overlay.translation_cam_from_cad_m,
                intrinsics,
                PRED_COLOR_BGR,
                max_points=max_render_points,
            )
            if points_cad_m is not None
            else _unavailable_image(image_bgr, "CAD SURFACE POINTS UNAVAILABLE")
        )

    numbered_names = {
        "rgb": "00_rgb.png",
        "mask_gt": "01_mask_gt.png",
        "mask_pred": "02_mask_pred.png",
        "dimensions_gt": "03_dimensions_gt.png",
        "dimensions_pred": "04_dimensions_pred.png",
        "orientation_gt": "05_orientation_gt.png",
        "orientation_pred": "06_orientation_pred.png",
        "surface_gt": "07_surface_gt.png",
        "surface_pred": "08_surface_pred.png",
        "mask_error": "09_mask_error.png",
    }
    written_images: dict[str, str] = {}
    for key, filename in numbered_names.items():
        image_path = instance_dir / filename
        _write_image(image_path, image_outputs[key])
        written_images[key] = str(image_path)
    display_exemplars: list[np.ndarray] = []
    for index in range(4):
        key = f"exemplar_{index + 1:02d}"
        filename = f"{10 + index:02d}_{key}.png"
        exemplar = (
            _resize_exemplar(exemplar_images[index])
            if index < len(exemplar_images)
            else _exemplar_placeholder()
        )
        image_path = instance_dir / filename
        _write_image(image_path, exemplar)
        written_images[key] = str(image_path)
        display_exemplars.append(exemplar)

    overview = _build_overview(
        image_outputs,
        display_exemplars,
        title_lines=[
            f"scene {sample.frame.scene_id} | frame {entry['frame_id']}",
            f"CAD {cad_id} | instance {instance.instance_id}",
        ],
        metric_lines=_overview_metric_lines(
            evaluation=evaluation,
            mask_iou=mask_iou,
            detection_score=detection_score,
            pose_score=pose_score,
            status_reason=status_reason,
        ),
    )
    overview_path = instance_dir / "14_overview.png"
    _write_image(overview_path, overview)
    written_images["overview"] = str(overview_path)

    record = {
        "dataset_id": entry.get("dataset_id"),
        "group_id": entry.get("group_id"),
        "scene_id": sample.frame.scene_id,
        "frame_id": entry["frame_id"],
        "cad_id": cad_id,
        "instance_id": instance.instance_id,
        "status": "matched" if prediction is not None else "unmatched",
        "note": status_reason,
        "mask_iou": _finite_or_none(mask_iou),
        "detection_score": _finite_or_none(detection_score),
        "pose_score": _finite_or_none(pose_score),
        "evaluation": _evaluation_dict(evaluation),
        "directory": str(instance_dir),
        "images": written_images,
        "exemplar_count": min(len(exemplar_images), 4),
    }
    with (instance_dir / "metrics.json").open("w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return record


def _build_overview(
    images: Mapping[str, np.ndarray],
    exemplars: Sequence[np.ndarray],
    *,
    title_lines: Sequence[str],
    metric_lines: Sequence[str],
) -> np.ndarray:
    """Build a large GT/pred grid with a narrower context/exemplar sidebar."""

    main_width = images["rgb"].shape[1]
    paired_rows = []
    for label, gt_key, pred_key in (
        ("SEGMENTATION", "mask_gt", "mask_pred"),
        ("CAD DIMENSIONS", "dimensions_gt", "dimensions_pred"),
        ("ORIENTATION", "orientation_gt", "orientation_pred"),
        ("CAD SURFACE", "surface_gt", "surface_pred"),
    ):
        paired_rows.append(
            _stack_horizontal(
                [
                    _labeled_panel(images[gt_key], f"{label} | GT", GT_COLOR_BGR, main_width),
                    _labeled_panel(
                        images[pred_key], f"{label} | PRED", PRED_COLOR_BGR, main_width
                    ),
                ]
            )
        )
    main_grid = _stack_vertical(paired_rows)

    sidebar_width = max(256, main_width // 2)
    context_panels = [
        _labeled_panel(images["rgb"], "CLEAN RGB", (235, 235, 235), sidebar_width),
        _labeled_panel(
            images["mask_error"],
            "MASK ERROR",
            (235, 235, 235),
            sidebar_width,
        ),
    ]
    exemplar_tile_width = max(96, (sidebar_width - 8) // 2)
    exemplar_panels = [
        _labeled_panel(
            exemplar,
            f"EXEMPLAR {index + 1}",
            (190, 190, 190),
            exemplar_tile_width,
        )
        for index, exemplar in enumerate(exemplars[:4])
    ]
    while len(exemplar_panels) < 4:
        exemplar_panels.append(
            _labeled_panel(
                _exemplar_placeholder(),
                f"EXEMPLAR {len(exemplar_panels) + 1}",
                (190, 190, 190),
                exemplar_tile_width,
            )
        )
    context_panels.extend(
        [
            _stack_horizontal(exemplar_panels[:2]),
            _stack_horizontal(exemplar_panels[2:4]),
            _text_card(metric_lines, sidebar_width),
        ]
    )
    sidebar = _stack_vertical(context_panels)
    content_height = max(main_grid.shape[0], sidebar.shape[0])
    main_grid = _pad_bottom(main_grid, content_height)
    sidebar = _pad_bottom(sidebar, content_height)
    content = _stack_horizontal([main_grid, sidebar], gap=16)
    header = _text_card(title_lines, content.shape[1], font_scale=0.62, line_height=25)
    return _stack_vertical([header, content], gap=10)


def _labeled_panel(
    image_bgr: np.ndarray,
    label: str,
    label_color: tuple[int, int, int],
    target_width: int,
) -> np.ndarray:
    image = _resize_to_width(image_bgr, target_width)
    bar_height = 30
    output = np.full((image.shape[0] + bar_height, target_width, 3), 18, dtype=np.uint8)
    output[bar_height:] = image
    cv2.putText(
        output,
        label,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        label_color,
        1,
        cv2.LINE_AA,
    )
    return output


def _resize_to_width(image_bgr: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        raise ValueError("target_width must be positive")
    height, width = image_bgr.shape[:2]
    if width == target_width:
        return image_bgr.copy()
    target_height = max(1, round(height * target_width / width))
    interpolation = cv2.INTER_AREA if target_width < width else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=interpolation)


def _stack_horizontal(
    images: Sequence[np.ndarray], *, gap: int = 8, background: int = 18
) -> np.ndarray:
    if not images:
        raise ValueError("Cannot stack an empty image sequence")
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    output = np.full((height, width, 3), background, dtype=np.uint8)
    x = 0
    for image in images:
        output[: image.shape[0], x : x + image.shape[1]] = image
        x += image.shape[1] + gap
    return output


def _stack_vertical(
    images: Sequence[np.ndarray], *, gap: int = 8, background: int = 18
) -> np.ndarray:
    if not images:
        raise ValueError("Cannot stack an empty image sequence")
    width = max(image.shape[1] for image in images)
    height = sum(image.shape[0] for image in images) + gap * (len(images) - 1)
    output = np.full((height, width, 3), background, dtype=np.uint8)
    y = 0
    for image in images:
        output[y : y + image.shape[0], : image.shape[1]] = image
        y += image.shape[0] + gap
    return output


def _pad_bottom(image_bgr: np.ndarray, target_height: int, background: int = 18) -> np.ndarray:
    if image_bgr.shape[0] >= target_height:
        return image_bgr
    output = np.full((target_height, image_bgr.shape[1], 3), background, dtype=np.uint8)
    output[: image_bgr.shape[0]] = image_bgr
    return output


def _text_card(
    lines: Sequence[str],
    width: int,
    *,
    font_scale: float = 0.46,
    line_height: int = 21,
) -> np.ndarray:
    visible_lines = list(lines) or ["No metrics available"]
    padding = 8
    height = padding * 2 + line_height * len(visible_lines)
    output = np.full((height, width, 3), 18, dtype=np.uint8)
    for index, line in enumerate(visible_lines):
        cv2.putText(
            output,
            line,
            (padding, padding + 14 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return output


def _overview_metric_lines(
    *,
    evaluation: PoseEvaluation | None,
    mask_iou: float | None,
    detection_score: float | None,
    pose_score: float | None,
    status_reason: str | None,
) -> list[str]:
    lines = [
        f"mask IoU: {_format_optional(mask_iou, 3)}",
        f"detection score: {_format_optional(detection_score, 3)}",
        f"pose score: {_format_optional(pose_score, 3)}",
    ]
    if evaluation is not None:
        lines.extend(
            [
                "surface mean/p95: "
                f"{_format_optional(evaluation.mean_surface_distance_norm, 4)} / "
                f"{_format_optional(evaluation.p95_surface_distance_norm, 4)}",
                "centroid/translation cm: "
                f"{_format_optional(evaluation.centroid_error_cm, 2)} / "
                f"{_format_optional(evaluation.translation_error_cm, 2)}",
                f"depth error m: {_format_optional(evaluation.depth_error_m, 4)}",
                f"pose success: {evaluation.pose_success_rate:.0%}",
            ]
        )
    if status_reason:
        lines.append(f"note: {status_reason}")
    return lines


def _format_optional(value: float | None, digits: int) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _add_dimension_label(image_bgr: np.ndarray, dimensions_m: np.ndarray) -> np.ndarray:
    dimensions = np.asarray(dimensions_m, dtype=np.float64)
    label = "CAD xyz [m]: " + " x ".join(f"{value:.4f}" for value in dimensions)
    return _add_corner_label(image_bgr, label)


def _add_corner_label(image_bgr: np.ndarray, label: str) -> np.ndarray:
    output = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    padding = 5
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    box_width = min(text_size[0] + 2 * padding, output.shape[1])
    box_height = text_size[1] + 2 * padding + 2
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (12, 12, 12), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, output, 0.25, 0.0, dst=output)
    cv2.putText(
        output,
        label,
        (padding, padding + text_size[1]),
        font,
        font_scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )
    return output


def _missing_prediction_image(image_bgr: np.ndarray, reason: str | None) -> np.ndarray:
    label = "NO MATCHED PREDICTION"
    if reason:
        label += f" | {reason}"
    return _add_corner_label(image_bgr, label)


def _unavailable_image(image_bgr: np.ndarray, label: str) -> np.ndarray:
    return _add_corner_label(image_bgr, label)


def _resize_exemplar(image_bgr: np.ndarray, max_side: int = 256) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image_bgr.copy()
    output_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image_bgr, output_size, interpolation=cv2.INTER_AREA)


def _spread_reference_view_ids(view_ids: Sequence[str], count: int) -> list[str]:
    """Choose up to ``count`` views spread across the configured reference order."""

    unique_ids = list(dict.fromkeys(view_ids))
    if count <= 0 or not unique_ids:
        return []
    if len(unique_ids) <= count:
        return unique_ids
    indices = np.rint(np.linspace(0, len(unique_ids) - 1, count)).astype(np.int64)
    return [unique_ids[int(index)] for index in indices]


def _exemplar_placeholder(size: int = 256) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    return _add_corner_label(image, "EXEMPLAR UNAVAILABLE")


def _write_image(path: Path, image_bgr: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Could not write visualization: {path}")


def _select_scene_rows(
    rows: Sequence[ManifestRow],
    *,
    split: str,
    requested_frames: Sequence[str],
    num_scenes: int,
    seed: int,
) -> list[ManifestRow]:
    split_rows = [row for row in rows if row.split == split]
    if not split_rows:
        raise ValueError(f"Manifest has no rows for split {split!r}")
    if requested_frames:
        requested = list(dict.fromkeys(str(frame) for frame in requested_frames))
        selected = [row for row in split_rows if row.frame_id in requested]
        found = {row.frame_id for row in selected}
        missing = [frame for frame in requested if frame not in found]
        if missing:
            raise ValueError(f"Requested {split} frame ids were not found: {missing}")
        chosen_keys = {(row.dataset_id, row.group_id) for row in selected}
        if len(chosen_keys) != len(requested):
            raise ValueError(
                "A requested frame was ambiguous across manifest groups; select a unique split manifest"
            )
        return selected

    group_rows: dict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for row in split_rows:
        group_rows[(row.dataset_id, row.group_id)].append(row)
    chosen_keys: list[tuple[str, str]] = []
    for frame_id in DEFAULT_QUICK_FRAMES:
        matches = [
            key
            for key, values in group_rows.items()
            if any(row.frame_id == frame_id for row in values)
        ]
        if len(matches) == 1 and matches[0] not in chosen_keys:
            chosen_keys.append(matches[0])
        if len(chosen_keys) >= num_scenes:
            break
    remaining = sorted(key for key in group_rows if key not in chosen_keys)
    random.Random(seed).shuffle(remaining)
    chosen_keys.extend(remaining[: max(num_scenes - len(chosen_keys), 0)])
    chosen_set = set(chosen_keys[:num_scenes])
    return [row for row in split_rows if (row.dataset_id, row.group_id) in chosen_set]


def _checkpoint_args(checkpoint: Mapping[str, object]) -> dict[str, object]:
    value = checkpoint.get("args") or {}
    if isinstance(value, argparse.Namespace):
        return vars(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Checkpoint args must be a mapping, got {type(value).__name__}")


def _resolve_path(cli_value: Path | None, checkpoint_value: object, label: str) -> Path:
    value = cli_value if cli_value is not None else checkpoint_value
    if value is None or not str(value):
        raise ValueError(f"Could not resolve {label}; provide it explicitly")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _verify_manifest(
    checkpoint: Mapping[str, object],
    manifest_path: Path,
    *,
    allow_mismatch: bool = False,
) -> None:
    expected = checkpoint.get("manifest_sha256")
    if not expected:
        return
    actual = manifest_sha256(manifest_path)
    if actual != expected:
        if allow_mismatch:
            print(
                "warning: evaluating against a manifest outside checkpoint provenance "
                f"({actual} != {expected})"
            )
            return
        raise ValueError(
            f"Manifest checksum does not match checkpoint provenance: {actual} != {expected}"
        )


def _resolve_device(requested: str | None) -> torch.device:
    device = torch.device(requested or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return device


def _resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "float32" or device.type == "cpu":
        return torch.float32
    if requested in ("auto", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {requested}")


def _evaluation_dict(evaluation: PoseEvaluation | None) -> dict[str, object] | None:
    if evaluation is None:
        return None
    metric_names = (
        "count",
        "mean_surface_distance_norm",
        "p95_surface_distance_norm",
        "centroid_error_cm",
        "translation_error_cm",
        "center_error_norm",
        "depth_error_m",
        "pose_success_rate",
        "brier_score",
        "expected_calibration_error",
        "rotation_error_deg",
        "accuracy_5deg_5cm",
        "accuracy_10deg_10cm",
    )
    return {
        name: _finite_or_none(getattr(evaluation, name))
        for name in metric_names
    }


def _finite_or_none(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


if __name__ == "__main__":
    main()
