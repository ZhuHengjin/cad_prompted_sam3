import os
import cv2
import numpy as np
import torch
from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.demo_helpers.mask_postprocessing import sample_points_from_mask


def visualize_detections(
    image_bgr: np.ndarray,
    mask_preds_bnhw: torch.Tensor,
    detection_scores_bn: torch.Tensor,
    output_path: str,
    alpha: float = 0.45,
    mask_threshold: float = 0.0,
) -> np.ndarray:
    if mask_preds_bnhw.ndim != 4 or detection_scores_bn.ndim != 2:
        raise ValueError("Expected mask_preds shape BxNxHxW and detection_scores shape BxN")

    num_masks = mask_preds_bnhw.shape[1]
    overlay = image_bgr.copy()
    if num_masks == 0:
        cv2.imwrite(output_path, overlay)
        return overlay

    rng = np.random.default_rng(12345)
    colors = rng.integers(0, 255, size=(num_masks, 3), dtype=np.uint8)

    masks_cpu = mask_preds_bnhw.detach().float().cpu().numpy()
    scores_cpu = detection_scores_bn.detach().float().cpu().numpy()

    for idx in range(num_masks):
        mask = masks_cpu[0, idx]
        mask_bin = (mask > mask_threshold).astype(np.uint8)
        if mask_bin.sum() == 0:
            continue

        mask_resized = cv2.resize(mask_bin, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = tuple(int(c) for c in colors[idx])

        color_layer = np.zeros_like(overlay, dtype=np.uint8)
        color_layer[mask_resized.astype(bool)] = color
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)

        ys, xs = np.where(mask_resized > 0)
        if ys.size > 0:
            cx = int(xs.mean())
            cy = int(ys.mean())
            score = float(scores_cpu[0, idx])
            label = f"{score:.2f}"
            cv2.putText(
                overlay,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(output_path, overlay)
    return overlay


def visualize_boxes(
    image_bgr: np.ndarray,
    box_preds_bn22: torch.Tensor,
    detection_scores_bn: torch.Tensor,
    output_path: str,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    if box_preds_bn22.ndim != 4 or detection_scores_bn.ndim != 2:
        raise ValueError("Expected box_preds shape BxNx2x2 and detection_scores shape BxN")

    out = image_bgr.copy()
    num_boxes = box_preds_bn22.shape[1]
    if num_boxes == 0:
        cv2.imwrite(output_path, out)
        return out

    boxes_cpu = box_preds_bn22.detach().float().cpu().numpy()
    scores_cpu = detection_scores_bn.detach().float().cpu().numpy()
    h, w = image_bgr.shape[:2]

    for idx in range(num_boxes):
        (x1n, y1n), (x2n, y2n) = boxes_cpu[0, idx]
        x1 = int(round(x1n * (w - 1)))
        y1 = int(round(y1n * (h - 1)))
        x2 = int(round(x2n * (w - 1)))
        y2 = int(round(y2n * (h - 1)))

        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        score = float(scores_cpu[0, idx])
        label = f"{score:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, out)
    return out


def save_individual_masks(
    image_bgr: np.ndarray,
    mask_preds_bnhw: torch.Tensor,
    output_dir: str,
    mask_threshold: float = 0.0,
) -> int:
    if mask_preds_bnhw.ndim != 4:
        raise ValueError("Expected mask_preds shape BxNxHxW")

    os.makedirs(output_dir, exist_ok=True)
    num_masks = mask_preds_bnhw.shape[1]
    if num_masks == 0:
        return 0

    masks_cpu = mask_preds_bnhw.detach().float().cpu().numpy()
    saved = 0
    for idx in range(num_masks):
        mask = masks_cpu[0, idx]
        mask_bin = (mask > mask_threshold).astype(np.uint8)
        if mask_bin.sum() == 0:
            continue

        mask_resized = cv2.resize(mask_bin, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_u8 = (mask_resized * 255).astype(np.uint8)
        out_path = os.path.join(output_dir, f"mask_{idx:03d}.png")
        cv2.imwrite(out_path, mask_u8)
        saved += 1

    return saved


def invert_mask(mask_image: np.ndarray) -> np.ndarray:
    max_val = mask_image.max()
    return max_val - mask_image


def resize_mask_to_image(mask_image: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if mask_image.shape[0] == h and mask_image.shape[1] == w:
        return mask_image
    return cv2.resize(mask_image, (w, h), interpolation=cv2.INTER_NEAREST)


def pad_right(image: np.ndarray, pad_value: int) -> np.ndarray:
    h, w = image.shape[:2]
    pad_w = w
    if image.ndim == 2:
        pad = np.full((h, pad_w), pad_value, dtype=image.dtype)
    else:
        pad = np.full((h, pad_w, image.shape[2]), pad_value, dtype=image.dtype)
    return np.concatenate([image, pad], axis=1)


def run_experiment(
    name: str,
    ref_image_bgr: np.ndarray,
    ref_mask: np.ndarray,
    encimg_targ: list[torch.Tensor],
    target_image_bgr: np.ndarray,
    detection_filter_threshold: float,
    save_masks_dir: str | None = None,
) -> None:
    pts_from_mask = sample_points_from_mask(ref_mask)
    encimg_ref, _, _ = detmodel.encode_detection_image(ref_image_bgr)
    exemplars_ref = detmodel.encode_exemplars(
        encimg_ref, text="visual", point_xy_norm_list=pts_from_mask, include_coordinate_encodings=False
    )
    mask_preds, box_preds, detection_scores, _ = detmodel.generate_detections(
        encimg_targ, exemplars_ref, detection_filter_threshold=detection_filter_threshold
    )

    print(f"[{name}] Found", mask_preds.shape[1], "masks")

    overlay_path = f"{name}_overlay.png"
    visualize_detections(target_image_bgr, mask_preds, detection_scores, overlay_path)
    print(f"[{name}] Saved overlay to", overlay_path)

    boxes_path = f"{name}_boxes.png"
    visualize_boxes(target_image_bgr, box_preds, detection_scores, boxes_path)
    print(f"[{name}] Saved boxes to", boxes_path)

    if save_masks_dir is not None:
        num_saved = save_individual_masks(target_image_bgr, mask_preds, save_masks_dir)
        print(f"[{name}] Saved", num_saved, "individual masks to", save_masks_dir)


# Load v3 model
_, full_model = make_sam_from_state_dict("/home/kevin/sam3.pt")
full_model.to(device="cuda", dtype=torch.bfloat16)
detmodel = full_model.make_detector_model()

# (Optional) Load finetuned detector weights
finetune_ckpt_path = "outputs_finetune_exemplar/finetune_epoch_001.pth"
if os.path.isfile(finetune_ckpt_path):
    ckpt = torch.load(finetune_ckpt_path, map_location="cpu")
    detmodel.image_exemplar_fusion.load_state_dict(ckpt["image_exemplar_fusion"])
    detmodel.exemplar_detector.load_state_dict(ckpt["exemplar_detector"])
    detmodel.exemplar_segmentation.load_state_dict(ckpt["exemplar_segmentation"])
    print("Loaded finetuned detector weights from", finetune_ckpt_path)

# Load mask & images
ref_mask_binary = cv2.imread("/sata1/data/kevin/realworld_datasets/primesense_converted/cad_renders/obj_11_stl_base_01.png")
ref_image = cv2.imread("/sata1/data/kevin/realworld_datasets/primesense_converted/cad_renders/obj_11_stl_base_01_mask.png")
target_image = cv2.imread("/sata1/data/kevin/realworld_datasets/primesense_converted/000003/rgb_0000.png")
# ref_mask_binary = cv2.imread("/home/kevin/muggled_sam/ref_image_mask.png")
# ref_image = cv2.imread("/home/kevin/muggled_sam/ref_image.png")
# target_image = cv2.imread("/home/kevin/muggled_sam/test_image.jpg")
# ref_mask_binary = np.ones_like(ref_image[:, :, 0], dtype=np.uint8) * 255

ref_mask_base = resize_mask_to_image(ref_mask_binary, ref_image)

detection_filter_threshold = 0.4
encimg_targ, _, _ = detmodel.encode_detection_image(target_image)

# 0. Current pipeline
run_experiment(
    name="target",
    ref_image_bgr=ref_image,
    ref_mask=ref_mask_base,
    encimg_targ=encimg_targ,
    target_image_bgr=target_image,
    detection_filter_threshold=detection_filter_threshold,
    save_masks_dir="mask_outputs",
)

# # 1. Invert mask
# ref_mask_inverted = invert_mask(ref_mask_base)
# run_experiment(
#     name="inverted",
#     ref_image_bgr=ref_image,
#     ref_mask=ref_mask_inverted,
#     encimg_targ=encimg_targ,
#     target_image_bgr=target_image,
#     detection_filter_threshold=detection_filter_threshold,
# )

# # 2. Pad reference image + mask (pad right side to 2x width)
# ref_image_padded = pad_right(ref_image, pad_value=255)
# ref_mask_padded = pad_right(ref_mask_base, pad_value=0)
# run_experiment(
#     name="pad",
#     ref_image_bgr=ref_image_padded,
#     ref_mask=ref_mask_padded,
#     encimg_targ=encimg_targ,
#     target_image_bgr=target_image,
#     detection_filter_threshold=detection_filter_threshold,
# )

# # 3. Invert + pad
# ref_mask_inverted_padded = pad_right(ref_mask_inverted, pad_value=0)
# run_experiment(
#     name="inverted_pad",
#     ref_image_bgr=ref_image_padded,
#     ref_mask=ref_mask_inverted_padded,
#     encimg_targ=encimg_targ,
#     target_image_bgr=target_image,
#     detection_filter_threshold=detection_filter_threshold,
# )
