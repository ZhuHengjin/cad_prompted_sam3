#!/usr/bin/env python3
"""Run CAD-conditioned SAM3 segmentation and 6-DoF pose inference on one image."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import torch

from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.v3_sam.cad_pose.geometry import adjust_intrinsics_for_resize_and_pad
from muggled_sam.v3_sam.cad_pose.inference import (
    format_cad_pose_results,
    mask_nms_indices,
    select_detection_candidates,
)
from muggled_sam.v3_sam.exemplar_view_pose import (
    EXEMPLAR_VIEW_MODES,
    ExemplarViewBundle,
    load_exemplar_view_adapter_for_inference,
    pad_exemplar_view_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_model", type=Path, help="Upstream SAM3 .pt checkpoint.")
    parser.add_argument("pose_checkpoint", type=Path, help="Fine-tuned/calibrated .pth checkpoint.")
    parser.add_argument("image", type=Path)
    parser.add_argument("cad_render", type=Path)
    parser.add_argument("cad_id")
    parser.add_argument("--dimensions_m", type=float, nargs=3, required=True)
    parser.add_argument(
        "--effective_surface_centroid_m",
        type=float,
        nargs=3,
        required=True,
        help="Catalog surface centroid after applying the same uniform physical scale as dimensions_m.",
    )
    parser.add_argument("--camera_k", type=float, nargs=9, required=True, metavar=("K00", "K01", "K02", "K10", "K11", "K12", "K20", "K21", "K22"))
    parser.add_argument("--render_box", type=float, nargs=4, default=(0.0, 0.0, 1.0, 1.0), metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument(
        "--exemplar_view_mode",
        choices=("auto", *EXEMPLAR_VIEW_MODES),
        default="auto",
        help="Reference-view mode; auto reads the pose checkpoint provenance.",
    )
    parser.add_argument(
        "--render_rotation_refcam_from_cad",
        type=float,
        nargs=9,
        default=None,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        help="Required for camera/shuffled-camera modes when passing a single CAD render.",
    )
    parser.add_argument("--exemplar_view_shuffle_seed", type=int, default=42)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--max_side_length", type=int, default=1008)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("cad_pose_results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    # Start from the upstream SAM3 model, then add the detection/pose modules.
    _, base_model = make_sam_from_state_dict(str(args.base_model))
    detector = base_model.make_detector_model().to(device)
    checkpoint = torch.load(args.pose_checkpoint, map_location=device, weights_only=False)
    checkpoint_version = int(checkpoint.get("cad_pose_head_architecture_version", 1))
    checkpoint_config = checkpoint.get("cad_pose_head_architecture_config")
    if (
        checkpoint_version == detector.cad_pose_head.architecture_version
        and checkpoint_config is not None
        and checkpoint_config != detector.cad_pose_head.architecture_config()
    ):
        raise ValueError("Checkpoint CAD pose-head architecture config is incompatible")
    migrated = detector.cad_pose_head.load_checkpoint_state_dict(
        checkpoint["cad_pose_head"],
        checkpoint_version,
    )
    if migrated:
        warnings.warn(
            "Loaded promptless CAD pose head v3 into v4; CAD prompting remains disabled."
        )
    # The pose checkpoint contains only the modules fine-tuned for CAD-conditioned
    # detection and pose estimation; retain the remaining upstream SAM3 weights.
    for key, module in (
        ("image_exemplar_fusion", detector.image_exemplar_fusion),
        ("exemplar_detector", detector.exemplar_detector),
        ("exemplar_segmentation", detector.exemplar_segmentation),
    ):
        module.load_state_dict(checkpoint[key])
    checkpoint_args = checkpoint.get("args") or {}
    detector.cad_pose_head.set_cad_prompt_enabled(
        bool(checkpoint_args.get("enable_cad_prompt", False))
    )
    args.exemplar_view_mode = load_exemplar_view_adapter_for_inference(
        detector, checkpoint, args.exemplar_view_mode
    )
    detector.eval()

    # OpenCV returns BGR images, which is the format expected by the detector encoder.
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    render = cv2.imread(str(args.cad_render), cv2.IMREAD_COLOR)
    if image is None or render is None:
        raise FileNotFoundError("Could not load target image or CAD render")
    # Encode the CAD render once and use its normalized bounding box as the visual exemplar.
    encoded_render, _, _ = detector.encode_detection_image(render, args.max_side_length, True)
    x1, y1, x2, y2 = args.render_box
    exemplar = detector.encode_exemplars(
        encoded_render,
        text="visual",
        box_xy1xy2_norm_list=[((x1, y1), (x2, y2))],
        include_coordinate_encodings=False,
    )
    if args.exemplar_view_mode != "none":
        if (
            args.render_rotation_refcam_from_cad is None
            and args.exemplar_view_mode in ("camera", "shuffled_camera")
        ):
            raise ValueError(
                "--render_rotation_refcam_from_cad is required by the checkpoint's "
                f"{args.exemplar_view_mode!r} exemplar-view mode"
            )
        rotation = (
            torch.eye(3, dtype=torch.float32).unsqueeze(0)
            if args.render_rotation_refcam_from_cad is None
            else torch.tensor(
                args.render_rotation_refcam_from_cad,
                dtype=torch.float32,
            ).reshape(1, 3, 3)
        )
        exemplar_bundle = ExemplarViewBundle(
            tokens_bnc=exemplar,
            token_view_indices_n=torch.zeros(exemplar.shape[1], dtype=torch.long),
            view_rotations_v33=rotation,
            view_ids=(args.cad_render.stem,),
            object_id=args.cad_id,
        )
        exemplar, _ = pad_exemplar_view_batch(
            [exemplar_bundle],
            device=device,
            pose_encoder=detector.exemplar_view_pose_encoder,
            mode=args.exemplar_view_mode,
            shuffle_seed=args.exemplar_view_shuffle_seed,
        )
    # Image encoding may resize and pad the input, so pose estimation must use
    # intrinsics transformed into the encoded image's coordinate system.
    encoded_image, _, model_hw = detector.encode_detection_image(image, args.max_side_length, True)
    model_h, model_w = model_hw
    original_k = torch.tensor(args.camera_k, device=device, dtype=encoded_image[0].dtype).reshape(3, 3)
    adjusted_k = adjust_intrinsics_for_resize_and_pad(
        original_k,
        (image.shape[1], image.shape[0]),
        (model_w, model_h),
    ).unsqueeze(0)
    dimensions = torch.tensor(args.dimensions_m, device=device, dtype=encoded_image[0].dtype).unsqueeze(0)
    effective_centroid = torch.tensor(
        args.effective_surface_centroid_m,
        device=device,
        dtype=encoded_image[0].dtype,
    ).unsqueeze(0)
    # Request segmentation, detection confidence, and metric 6-DoF pose jointly.
    masks, boxes, scores, _, poses = detector.generate_detections(
        encoded_image,
        exemplar,
        detection_filter_threshold=args.score_threshold,
        cad_dimensions_m_b3=dimensions,
        cad_effective_surface_centroid_m_b3=effective_centroid,
        camera_intrinsics_b33=adjusted_k,
        return_pose=True,
    )
    # Suppress duplicate mask proposals before associating the retained entries
    # with their corresponding boxes, scores, and poses.
    retained = mask_nms_indices(masks[0], scores[0], args.nms_iou)
    masks, boxes, scores, poses = select_detection_candidates(masks[0], boxes[0], scores[0], poses, retained)
    results = format_cad_pose_results(masks, boxes, scores, poses, cad_id=args.cad_id)
    # Detach tensors and convert them to built-in types so JSON can encode them.
    serializable = [
        {
            "box_xyxy": result["box_xyxy"].detach().float().cpu().tolist(),
            "detection_score": float(result["detection_score"]),
            "rotation_matrix": result["rotation_matrix"].detach().float().cpu().tolist(),
            "translation_m": result["translation_m"].detach().float().cpu().tolist(),
            "cad_dimensions_m": result["cad_dimensions_m"].detach().float().cpu().tolist(),
            "pose_score": float(result["pose_score"]),
            "cad_id": result["cad_id"],
        }
        for result in results
    ]
    args.output.write_text(json.dumps(serializable, indent=2))
    print(f"Wrote {len(serializable)} instance results to {args.output}")


if __name__ == "__main__":
    main()
