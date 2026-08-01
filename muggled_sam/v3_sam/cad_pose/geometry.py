"""Differentiable geometry helpers for the CAD-conditioned SAM3 pose head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def rotation_6d_to_matrix(rotation_6d: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert the first two rotation columns to a proper rotation matrix.

    The two 3-vectors are stored consecutively, matching the Perseve/YOPO
    target convention ``[R[:, 0], R[:, 1]]``.
    """

    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected rotation vectors ending in 6 values, got {tuple(rotation_6d.shape)}")
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    fallback_first = torch.zeros_like(first)
    fallback_first[..., 0] = 1.0
    r1 = torch.where(first_norm > eps, first / first_norm.clamp_min(eps), fallback_first)
    second_orthogonal = second - (r1 * second).sum(dim=-1, keepdim=True) * r1
    second_norm = torch.linalg.vector_norm(second_orthogonal, dim=-1, keepdim=True)
    fallback_axis = F.one_hot(r1.abs().argmin(dim=-1), num_classes=3).to(r1)
    fallback_second = fallback_axis - (fallback_axis * r1).sum(dim=-1, keepdim=True) * r1
    fallback_second = F.normalize(fallback_second, dim=-1, eps=eps)
    r2 = torch.where(second_norm > eps, second_orthogonal / second_norm.clamp_min(eps), fallback_second)
    r3 = torch.linalg.cross(r1, r2, dim=-1)
    return torch.stack((r1, r2, r3), dim=-1)


def matrix_to_rotation_6d(rotation_matrix: Tensor) -> Tensor:
    """Return the first two columns of a rotation matrix as consecutive vectors."""

    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices ending in 3x3, got {tuple(rotation_matrix.shape)}")
    return torch.cat((rotation_matrix[..., :, 0], rotation_matrix[..., :, 1]), dim=-1)


def reconstruct_translation(center_uv_px: Tensor, log_depth: Tensor, intrinsics: Tensor) -> Tensor:
    """Back-project a metric OpenCV-frame point from image center and log-depth."""

    if center_uv_px.shape[-1] != 2:
        raise ValueError(f"Expected projected centers ending in 2 values, got {tuple(center_uv_px.shape)}")
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics ending in 3x3, got {tuple(intrinsics.shape)}")
    if log_depth.shape == center_uv_px.shape[:-1] + (1,):
        log_depth = log_depth.squeeze(-1)
    if log_depth.shape != center_uv_px.shape[:-1]:
        raise ValueError("log_depth must match the projected-center leading dimensions")

    intrinsics = _broadcast_intrinsics(intrinsics, center_uv_px.ndim - 1)
    homogeneous = torch.cat((center_uv_px, torch.ones_like(center_uv_px[..., :1])), dim=-1)
    calculation_dtype = torch.float32 if homogeneous.dtype in (torch.float16, torch.bfloat16) else homogeneous.dtype
    rays = torch.linalg.solve(
        intrinsics.to(calculation_dtype), homogeneous.to(calculation_dtype).unsqueeze(-1)
    ).squeeze(-1)
    return rays * log_depth.to(calculation_dtype).exp().unsqueeze(-1)


def surface_centroid_camera(
    rotation_matrix: Tensor,
    aabb_translation_m: Tensor,
    effective_surface_centroid_m: Tensor,
) -> Tensor:
    """Transform an effective canonical surface centroid into camera space."""

    effective_surface_centroid_m = _broadcast_vector_for_rotation(
        effective_surface_centroid_m, rotation_matrix
    )
    aabb_translation_m = _broadcast_vector_for_rotation(aabb_translation_m, rotation_matrix)
    calculation_dtype = torch.promote_types(rotation_matrix.dtype, effective_surface_centroid_m.dtype)
    rotation_matrix = rotation_matrix.to(calculation_dtype)
    effective_surface_centroid_m = effective_surface_centroid_m.to(calculation_dtype)
    aabb_translation_m = aabb_translation_m.to(calculation_dtype)
    return (
        torch.matmul(rotation_matrix, effective_surface_centroid_m.unsqueeze(-1)).squeeze(-1)
        + aabb_translation_m
    )


def aabb_translation_from_surface_centroid(
    centroid_camera_m: Tensor,
    rotation_matrix: Tensor,
    effective_surface_centroid_m: Tensor,
) -> Tensor:
    """Recover public AABB-origin translation from a camera-space centroid."""

    effective_surface_centroid_m = _broadcast_vector_for_rotation(
        effective_surface_centroid_m, rotation_matrix
    )
    centroid_camera_m = _broadcast_vector_for_rotation(centroid_camera_m, rotation_matrix)
    calculation_dtype = torch.promote_types(rotation_matrix.dtype, centroid_camera_m.dtype)
    rotation_matrix = rotation_matrix.to(calculation_dtype)
    effective_surface_centroid_m = effective_surface_centroid_m.to(calculation_dtype)
    centroid_camera_m = centroid_camera_m.to(calculation_dtype)
    return centroid_camera_m - torch.matmul(
        rotation_matrix, effective_surface_centroid_m.unsqueeze(-1)
    ).squeeze(-1)


def rotate_points(points: Tensor, rotation_matrix: Tensor) -> Tensor:
    """Apply column-vector rotations to point rows ending in ``Nx3``."""

    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected points ending in Nx3, got {tuple(points.shape)}")
    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations ending in 3x3, got {tuple(rotation_matrix.shape)}")
    return torch.matmul(points, rotation_matrix.transpose(-1, -2))


def project_points(points_camera: Tensor, intrinsics: Tensor, eps: float = 1e-8) -> Tensor:
    """Project OpenCV-frame 3-D camera points to top-left-pixel-center coordinates."""

    if points_camera.shape[-1] != 3:
        raise ValueError(f"Expected points ending in 3 values, got {tuple(points_camera.shape)}")
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics ending in 3x3, got {tuple(intrinsics.shape)}")
    intrinsics = _broadcast_intrinsics(intrinsics, points_camera.ndim - 1)
    pixels_h = torch.matmul(intrinsics, points_camera.unsqueeze(-1)).squeeze(-1)
    depth = pixels_h[..., 2:3]
    safe_depth = torch.where(depth.abs() < eps, depth.sign() * eps + (depth == 0) * eps, depth)
    return pixels_h[..., :2] / safe_depth


def adjust_intrinsics_for_resize_and_pad(
    intrinsics: Tensor,
    source_size_wh: tuple[int, int],
    destination_size_wh: tuple[int, int],
    *,
    resized_size_wh: tuple[int, int] | None = None,
    pad_left_top: tuple[float, float] = (0.0, 0.0),
) -> Tensor:
    """Update ``K`` after resizing and optional left/top padding.

    ``resized_size_wh`` describes the unpadded resized raster. When omitted,
    the destination itself is the resized raster (possibly anisotropically).
    """

    source_w, source_h = source_size_wh
    destination_w, destination_h = destination_size_wh
    resized_w, resized_h = resized_size_wh or destination_size_wh
    if min(source_w, source_h, destination_w, destination_h, resized_w, resized_h) <= 0:
        raise ValueError("Image dimensions must be positive")
    pad_x, pad_y = pad_left_top
    adjusted = intrinsics.clone()
    scale_x, scale_y = resized_w / source_w, resized_h / source_h
    adjusted[..., 0, 0] *= scale_x
    adjusted[..., 0, 1] *= scale_x
    adjusted[..., 0, 2] = adjusted[..., 0, 2] * scale_x + pad_x
    adjusted[..., 1, 0] *= scale_y
    adjusted[..., 1, 1] *= scale_y
    adjusted[..., 1, 2] = adjusted[..., 1, 2] * scale_y + pad_y
    return adjusted


def normalize_intrinsics(intrinsics: Tensor, image_size_wh: tuple[int, int]) -> Tensor:
    """Express pixel-space pinhole intrinsics in normalized image coordinates."""

    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics ending in 3x3, got {tuple(intrinsics.shape)}")
    width, height = image_size_wh
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    normalized = intrinsics.clone()
    normalized[..., 0, :] /= max(width - 1, 1)
    normalized[..., 1, :] /= max(height - 1, 1)
    return normalized


def normalized_to_pixel(center_uv_norm: Tensor, image_size_wh: tuple[int, int]) -> Tensor:
    """Map normalized coordinates to the top-left-pixel-center raster convention."""

    width, height = image_size_wh
    scale = center_uv_norm.new_tensor((max(width - 1, 1), max(height - 1, 1)))
    return center_uv_norm * scale


def pixel_to_normalized(center_uv_px: Tensor, image_size_wh: tuple[int, int]) -> Tensor:
    """Map top-left-pixel-center coordinates to normalized image coordinates."""

    width, height = image_size_wh
    scale = center_uv_px.new_tensor((max(width - 1, 1), max(height - 1, 1)))
    return center_uv_px / scale


def _broadcast_intrinsics(intrinsics: Tensor, target_leading_dims: int) -> Tensor:
    while intrinsics.ndim - 2 < target_leading_dims:
        intrinsics = intrinsics.unsqueeze(-3)
    return intrinsics


def _broadcast_vector_for_rotation(vector: Tensor, rotation_matrix: Tensor) -> Tensor:
    if vector.shape[-1:] != (3,):
        raise ValueError(f"Expected vectors ending in 3 values, got {tuple(vector.shape)}")
    while vector.ndim < rotation_matrix.ndim - 1:
        vector = vector.unsqueeze(-2)
    return vector
