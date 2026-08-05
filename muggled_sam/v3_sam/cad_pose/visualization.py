"""OpenCV rendering primitives for qualitative CAD-pose inspection.

The renderer follows the pose dataset's OpenCV camera convention: CAD points
are transformed as ``points @ R.T + t`` and projected with the raw camera
intrinsics.  It intentionally has no model or dataset-loading dependency so it
can be regression-tested independently from inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
AXIS_COLORS_BGR = (
    (40, 40, 240),  # +X: red
    (40, 220, 40),  # +Y: green
    (240, 90, 40),  # +Z: blue
)
GT_COLOR_BGR = (70, 220, 70)
PRED_COLOR_BGR = (220, 70, 230)


class PoseVisualizationError(ValueError):
    """Raised when pose geometry cannot be rendered safely."""


@dataclass(frozen=True)
class PoseOverlay:
    """One pose and its optional 2-D/3-D supporting geometry.

    ``translation_cam_from_cad_m`` is the public local-AABB-origin
    translation, not the camera-space surface centroid. ``points_cad_m`` must
    already include the instance's render scale.
    """

    label: str
    rotation_cam_from_cad: np.ndarray
    translation_cam_from_cad_m: np.ndarray
    dimensions_m: np.ndarray
    color_bgr: tuple[int, int, int]
    mask_hw: np.ndarray | None = None
    points_cad_m: np.ndarray | None = None


def canonical_box_corners(dimensions_m: np.ndarray) -> np.ndarray:
    """Return the eight corners of an origin-centered XYZ cuboid."""

    dimensions = _finite_array(dimensions_m, (3,), "dimensions_m")
    if np.any(dimensions <= 0.0):
        raise PoseVisualizationError("dimensions_m values must be positive")
    x, y, z = dimensions * 0.5
    return np.asarray(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        dtype=np.float64,
    )


def transform_cad_points(
    points_cad_m: np.ndarray,
    rotation_cam_from_cad: np.ndarray,
    translation_cam_from_cad_m: np.ndarray,
) -> np.ndarray:
    """Transform point rows using the column-vector ``T_cam_from_cad`` convention."""

    points = _points_array(points_cad_m, "points_cad_m")
    rotation = _finite_array(rotation_cam_from_cad, (3, 3), "rotation_cam_from_cad")
    translation = _finite_array(
        translation_cam_from_cad_m, (3,), "translation_cam_from_cad_m"
    )
    return points @ rotation.T + translation


def project_camera_points(
    points_camera_m: np.ndarray,
    intrinsics: np.ndarray,
    *,
    minimum_depth_m: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Project valid OpenCV-camera points and return ``(uv, valid_mask)``.

    Invalid, non-finite, or behind-camera points are represented by NaNs in
    ``uv`` instead of causing the whole object to be rejected.
    """

    points = _points_array(points_camera_m, "points_camera_m", require_finite=False)
    camera_matrix = _finite_array(intrinsics, (3, 3), "intrinsics")
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > minimum_depth_m)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        homogeneous = points[valid] @ camera_matrix.T
        projected_valid = np.isfinite(homogeneous).all(axis=1) & (
            np.abs(homogeneous[:, 2]) > 1e-12
        )
        valid_indices = np.flatnonzero(valid)
        rejected_indices = valid_indices[~projected_valid]
        valid[rejected_indices] = False
        accepted_indices = valid_indices[projected_valid]
        accepted = homogeneous[projected_valid]
        uv[accepted_indices] = accepted[:, :2] / accepted[:, 2, None]
    return uv, valid


def resize_binary_mask(mask_hw: np.ndarray, output_size_wh: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask without inventing fractional contour locations."""

    mask = np.asarray(mask_hw)
    if mask.ndim != 2:
        raise PoseVisualizationError(f"mask_hw must have shape HxW, got {mask.shape}")
    width, height = output_size_wh
    if width <= 0 or height <= 0:
        raise PoseVisualizationError("output_size_wh values must be positive")
    return cv2.resize(
        (mask > 0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def draw_mask_contour(
    image_bgr: np.ndarray,
    mask_hw: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    alpha: float = 0.18,
    thickness: int = 1,
) -> np.ndarray:
    """Return a copy with a translucent mask fill and crisp outer contour."""

    output = _image_copy(image_bgr)
    mask = np.asarray(mask_hw) > 0
    if mask.shape != output.shape[:2]:
        raise PoseVisualizationError(
            f"mask shape {mask.shape} does not match image shape {output.shape[:2]}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise PoseVisualizationError("alpha must be in [0, 1]")
    if np.any(mask):
        color = np.asarray(_color(color_bgr), dtype=np.float32)
        blended = output[mask].astype(np.float32) * (1.0 - alpha) + color * alpha
        output[mask] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, _color(color_bgr), thickness, cv2.LINE_AA)
    return output


def draw_pose_box(
    image_bgr: np.ndarray,
    overlay: PoseOverlay,
    intrinsics: np.ndarray,
    *,
    line_thickness: int = 1,
) -> np.ndarray:
    """Return a copy with only the projected CAD-dimension box."""

    output = _image_copy(image_bgr)
    rotation, translation, dimensions = _pose_arrays(overlay)
    corners_camera = transform_cad_points(
        canonical_box_corners(dimensions), rotation, translation
    )
    corners_uv, corners_valid = project_camera_points(corners_camera, intrinsics)
    color = _color(overlay.color_bgr)
    for start_index, end_index in BOX_EDGES:
        if corners_valid[start_index] and corners_valid[end_index]:
            _draw_clipped_line(
                output,
                corners_uv[start_index],
                corners_uv[end_index],
                color,
                line_thickness,
            )
    return output


def draw_pose_axes(
    image_bgr: np.ndarray,
    overlay: PoseOverlay,
    intrinsics: np.ndarray,
    *,
    line_thickness: int = 1,
) -> np.ndarray:
    """Return a copy with only the projected CAD-frame XYZ axes."""

    output = _image_copy(image_bgr)
    rotation, translation, dimensions = _pose_arrays(overlay)
    axis_length = max(float(dimensions.max()) * 0.6, 0.01)
    axes_cad = np.vstack((np.zeros(3), np.eye(3) * axis_length))
    axes_camera = transform_cad_points(axes_cad, rotation, translation)
    axes_uv, axes_valid = project_camera_points(axes_camera, intrinsics)
    if axes_valid[0]:
        for index, axis_color in enumerate(AXIS_COLORS_BGR):
            endpoint_index = index + 1
            if not axes_valid[endpoint_index]:
                continue
            _draw_clipped_line(
                output,
                axes_uv[0],
                axes_uv[endpoint_index],
                axis_color,
                line_thickness,
            )
            endpoint = _pixel(axes_uv[endpoint_index])
            cv2.putText(
                output,
                "XYZ"[index],
                endpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                axis_color,
                1,
                lineType=cv2.LINE_AA,
            )
    return output


def draw_pose_box_axes(
    image_bgr: np.ndarray,
    overlay: PoseOverlay,
    intrinsics: np.ndarray,
    *,
    line_thickness: int = 1,
    draw_axes: bool = True,
) -> np.ndarray:
    """Return a copy with a projected 3-D box and optional XYZ axes."""

    output = draw_pose_box(
        image_bgr,
        overlay,
        intrinsics,
        line_thickness=line_thickness,
    )
    if draw_axes:
        output = draw_pose_axes(
            output,
            overlay,
            intrinsics,
            line_thickness=line_thickness,
        )
    return output


def draw_mask_error_overlay(
    image_bgr: np.ndarray,
    ground_truth_mask_hw: np.ndarray,
    prediction_mask_hw: np.ndarray,
    *,
    alpha: float = 0.35,
) -> np.ndarray:
    """Show segmentation agreement: TP green, FP magenta, and FN orange."""

    output = _image_copy(image_bgr)
    ground_truth = np.asarray(ground_truth_mask_hw) > 0
    prediction = np.asarray(prediction_mask_hw) > 0
    if ground_truth.shape != output.shape[:2] or prediction.shape != output.shape[:2]:
        raise PoseVisualizationError("GT and prediction masks must match the image shape")
    if not 0.0 <= alpha <= 1.0:
        raise PoseVisualizationError("alpha must be in [0, 1]")
    categories = (
        (ground_truth & prediction, GT_COLOR_BGR),
        (~ground_truth & prediction, PRED_COLOR_BGR),
        (ground_truth & ~prediction, (40, 150, 245)),
    )
    for mask, color_bgr in categories:
        if not np.any(mask):
            continue
        color = np.asarray(color_bgr, dtype=np.float32)
        blended = output[mask].astype(np.float32) * (1.0 - alpha) + color * alpha
        output[mask] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return output


def draw_projected_points(
    image_bgr: np.ndarray,
    points_cad_m: np.ndarray,
    rotation_cam_from_cad: np.ndarray,
    translation_cam_from_cad_m: np.ndarray,
    intrinsics: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    max_points: int = 512,
    radius: int = 1,
) -> np.ndarray:
    """Return a copy with a deterministic subset of CAD surface points."""

    output = _image_copy(image_bgr)
    points = _points_array(points_cad_m, "points_cad_m")
    if max_points <= 0:
        return output
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    camera_points = transform_cad_points(
        points, rotation_cam_from_cad, translation_cam_from_cad_m
    )
    pixels, valid = project_camera_points(camera_points, intrinsics)
    height, width = output.shape[:2]
    color = _color(color_bgr)
    for point in pixels[valid]:
        if -radius <= point[0] < width + radius and -radius <= point[1] < height + radius:
            cv2.circle(output, _pixel(point), radius, color, cv2.FILLED, cv2.LINE_AA)
    return output


def render_pose_overlay(
    image_bgr: np.ndarray,
    intrinsics: np.ndarray,
    overlays: Sequence[PoseOverlay],
    *,
    mask_alpha: float = 0.18,
    line_thickness: int = 1,
    max_points: int = 512,
    draw_axes: bool = True,
) -> np.ndarray:
    """Render one or more overlays without mutating the source image."""

    output = _image_copy(image_bgr)
    for overlay in overlays:
        _pose_arrays(overlay)
        if overlay.mask_hw is not None:
            output = draw_mask_contour(
                output,
                overlay.mask_hw,
                overlay.color_bgr,
                alpha=mask_alpha,
                thickness=line_thickness,
            )
        if overlay.points_cad_m is not None:
            output = draw_projected_points(
                output,
                overlay.points_cad_m,
                overlay.rotation_cam_from_cad,
                overlay.translation_cam_from_cad_m,
                intrinsics,
                overlay.color_bgr,
                max_points=max_points,
            )
        output = draw_pose_box_axes(
            output,
            overlay,
            intrinsics,
            line_thickness=line_thickness,
            draw_axes=draw_axes,
        )
    return output


def render_pose_comparison(
    image_bgr: np.ndarray,
    intrinsics: np.ndarray,
    ground_truth: PoseOverlay,
    prediction: PoseOverlay | None,
    *,
    metric_lines: Sequence[str] = (),
    line_thickness: int = 1,
    max_points: int = 512,
) -> np.ndarray:
    """Build ``ground truth | prediction | combined`` qualitative panels."""

    gt_panel = render_pose_overlay(
        image_bgr,
        intrinsics,
        [ground_truth],
        line_thickness=line_thickness,
        max_points=max_points,
    )
    gt_panel = _draw_panel_title(gt_panel, ground_truth.label, ground_truth.color_bgr)

    if prediction is None:
        pred_panel = _draw_panel_title(
            _image_copy(image_bgr), "NO MATCHED PREDICTION", PRED_COLOR_BGR
        )
        combined = render_pose_overlay(
            image_bgr,
            intrinsics,
            [ground_truth],
            mask_alpha=0.10,
            line_thickness=line_thickness,
            max_points=max_points,
            draw_axes=False,
        )
    else:
        pred_panel = render_pose_overlay(
            image_bgr,
            intrinsics,
            [prediction],
            line_thickness=line_thickness,
            max_points=max_points,
        )
        pred_panel = _draw_panel_title(pred_panel, prediction.label, prediction.color_bgr)
        combined = render_pose_overlay(
            image_bgr,
            intrinsics,
            [ground_truth, prediction],
            mask_alpha=0.10,
            line_thickness=line_thickness,
            max_points=max_points,
            draw_axes=False,
        )
    combined = _draw_panel_title(combined, "GT green | prediction magenta", (230, 230, 230))
    if metric_lines:
        combined = _draw_text_block(combined, list(metric_lines), (230, 230, 230))
    return np.concatenate((gt_panel, pred_panel, combined), axis=1)


def _pose_arrays(overlay: PoseOverlay) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = _finite_array(
        overlay.rotation_cam_from_cad, (3, 3), "rotation_cam_from_cad"
    )
    translation = _finite_array(
        overlay.translation_cam_from_cad_m, (3,), "translation_cam_from_cad_m"
    )
    dimensions = _finite_array(overlay.dimensions_m, (3,), "dimensions_m")
    if np.any(dimensions <= 0.0):
        raise PoseVisualizationError("dimensions_m values must be positive")
    _color(overlay.color_bgr)
    return rotation, translation, dimensions


def _finite_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise PoseVisualizationError(f"{name} must be a finite array with shape {shape}")
    return array


def _points_array(
    value: np.ndarray, name: str, *, require_finite: bool = True
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise PoseVisualizationError(f"{name} must have shape (N, 3)")
    if require_finite and not np.isfinite(array).all():
        raise PoseVisualizationError(f"{name} must contain only finite values")
    return array


def _image_copy(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise PoseVisualizationError("image_bgr must be a uint8 HxWx3 array")
    return image.copy()


def _color(value: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(value) != 3 or any(int(channel) < 0 or int(channel) > 255 for channel in value):
        raise PoseVisualizationError("color_bgr must contain three uint8-range values")
    return tuple(int(channel) for channel in value)


def _pixel(point: np.ndarray) -> tuple[int, int]:
    clipped = np.clip(np.rint(point), -1_000_000, 1_000_000).astype(np.int32)
    return int(clipped[0]), int(clipped[1])


def _draw_clipped_line(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    visible, clipped_start, clipped_end = cv2.clipLine(
        (0, 0, width, height), _pixel(start), _pixel(end)
    )
    if visible:
        cv2.line(
            image,
            clipped_start,
            clipped_end,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def _draw_panel_title(
    image: np.ndarray, title: str, color: tuple[int, int, int]
) -> np.ndarray:
    output = image.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1] - 1, 30), (12, 12, 12), cv2.FILLED)
    cv2.addWeighted(overlay, 0.78, output, 0.22, 0.0, dst=output)
    cv2.putText(
        output,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _color(color),
        1,
        lineType=cv2.LINE_AA,
    )
    return output


def _draw_text_block(
    image: np.ndarray, lines: list[str], border_color: tuple[int, int, int]
) -> np.ndarray:
    output = image.copy()
    if not lines:
        return output
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.43
    thickness = 1
    line_height = 18
    padding = 6
    max_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines
    )
    block_width = min(max_width + 2 * padding, output.shape[1] - 2)
    block_height = min(line_height * len(lines) + 2 * padding, output.shape[0] - 2)
    x, y = 1, output.shape[0] - block_height - 1
    translucent = output.copy()
    cv2.rectangle(
        translucent,
        (x, y),
        (x + block_width, y + block_height),
        (12, 12, 12),
        cv2.FILLED,
    )
    cv2.addWeighted(translucent, 0.78, output, 0.22, 0.0, dst=output)
    cv2.rectangle(
        output,
        (x, y),
        (x + block_width, y + block_height),
        _color(border_color),
        1,
        lineType=cv2.LINE_AA,
    )
    for index, line in enumerate(lines):
        baseline = y + padding + 12 + index * line_height
        if baseline >= y + block_height:
            break
        cv2.putText(
            output,
            line,
            (x + padding, baseline),
            font,
            font_scale,
            (245, 245, 245),
            thickness,
            lineType=cv2.LINE_AA,
        )
    return output
