#!/usr/bin/env python3
"""Run CAD-conditioned SAM3 segmentation and 6-DoF pose inference on one image."""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_model", type=Path, help="Upstream SAM3 .pt checkpoint.")
    parser.add_argument("pose_checkpoint", type=Path, help="Fine-tuned/calibrated .pth checkpoint.")
    parser.add_argument("image", type=Path)
    parser.add_argument("cad_render", type=Path)
    parser.add_argument("cad_id")
    parser.add_argument("--dimensions_m", type=float, nargs=3, required=True)
    parser.add_argument("--camera_k", type=float, nargs=9, required=True, metavar=("K00", "K01", "K02", "K10", "K11", "K12", "K20", "K21", "K22"))
    parser.add_argument("--render_box", type=float, nargs=4, default=(0.0, 0.0, 1.0, 1.0), metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--max_side_length", type=int, default=1008)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("cad_pose_results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    _, base_model = make_sam_from_state_dict(str(args.base_model))
    detector = base_model.make_detector_model().to(device)
    checkpoint = torch.load(args.pose_checkpoint, map_location=device, weights_only=True)
    for key, module in (
        ("image_exemplar_fusion", detector.image_exemplar_fusion),
        ("exemplar_detector", detector.exemplar_detector),
        ("exemplar_segmentation", detector.exemplar_segmentation),
        ("cad_pose_head", detector.cad_pose_head),
    ):
        module.load_state_dict(checkpoint[key])
    detector.eval()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    render = cv2.imread(str(args.cad_render), cv2.IMREAD_COLOR)
    if image is None or render is None:
        raise FileNotFoundError("Could not load target image or CAD render")
    encoded_render, _, _ = detector.encode_detection_image(render, args.max_side_length, True)
    x1, y1, x2, y2 = args.render_box
    exemplar = detector.encode_exemplars(
        encoded_render,
        text="visual",
        box_xy1xy2_norm_list=[((x1, y1), (x2, y2))],
        include_coordinate_encodings=False,
    )
    encoded_image, _, model_hw = detector.encode_detection_image(image, args.max_side_length, True)
    model_h, model_w = model_hw
    original_k = torch.tensor(args.camera_k, device=device, dtype=encoded_image[0].dtype).reshape(3, 3)
    adjusted_k = adjust_intrinsics_for_resize_and_pad(
        original_k,
        (image.shape[1], image.shape[0]),
        (model_w, model_h),
    ).unsqueeze(0)
    dimensions = torch.tensor(args.dimensions_m, device=device, dtype=encoded_image[0].dtype).unsqueeze(0)
    masks, boxes, scores, _, poses = detector.generate_detections(
        encoded_image,
        exemplar,
        detection_filter_threshold=args.score_threshold,
        cad_dimensions_m_b3=dimensions,
        camera_intrinsics_b33=adjusted_k,
        return_pose=True,
    )
    retained = mask_nms_indices(masks[0], scores[0], args.nms_iou)
    masks, boxes, scores, poses = select_detection_candidates(masks[0], boxes[0], scores[0], poses, retained)
    results = format_cad_pose_results(masks, boxes, scores, poses, cad_id=args.cad_id)
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
