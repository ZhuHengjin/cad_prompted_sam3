"""Symmetry-aware rotation errors shared by training and evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


VERIFIED_SYMMETRY_STATUSES = frozenset(("verified_auto", "verified_manual"))


def geodesic_rotation_error(predicted: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Return the SO(3) geodesic angle in radians."""

    _check_rotations(predicted, target)
    relative = predicted.transpose(-1, -2) @ target
    del eps
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sine, cosine)


def discrete_symmetry_rotation_error(predicted: Tensor, target: Tensor, symmetry_transforms: Tensor) -> Tensor:
    """Minimize geodesic error over proper object-frame symmetry rotations."""

    _check_rotations(predicted, target)
    if symmetry_transforms.shape[-2:] != (3, 3):
        raise ValueError("symmetry_transforms must end in 3x3")
    if symmetry_transforms.ndim == 3:
        symmetry_transforms = symmetry_transforms.view(*((1,) * (target.ndim - 2)), *symmetry_transforms.shape)
    valid_targets = target.unsqueeze(-3) @ symmetry_transforms
    errors = geodesic_rotation_error(predicted.unsqueeze(-3), valid_targets)
    return errors.min(dim=-1).values


def continuous_axis_rotation_error(predicted: Tensor, target: Tensor, axis_cad: Tensor, eps: float = 1e-7) -> Tensor:
    """Compare only the camera-frame direction of a continuously symmetric CAD axis."""

    _check_rotations(predicted, target)
    axis_cad = F.normalize(axis_cad.to(device=predicted.device, dtype=predicted.dtype), dim=-1)
    while axis_cad.ndim < predicted.ndim - 1:
        axis_cad = axis_cad.unsqueeze(0)
    predicted_axis = (predicted @ axis_cad.unsqueeze(-1)).squeeze(-1)
    target_axis = (target @ axis_cad.unsqueeze(-1)).squeeze(-1)
    del eps
    cosine = (predicted_axis * target_axis).sum(-1).clamp(-1.0, 1.0)
    sine = torch.linalg.vector_norm(torch.linalg.cross(predicted_axis, target_axis, dim=-1), dim=-1)
    return torch.atan2(sine, cosine)


def symmetry_aware_rotation_error(
    predicted: Tensor,
    target: Tensor,
    *,
    symmetry_type: str = "none",
    symmetry_transforms: Tensor | None = None,
    axis_cad: Tensor | None = None,
) -> Tensor:
    """Dispatch to the catalog-declared symmetry treatment."""

    if symmetry_type == "continuous_axis":
        if axis_cad is None:
            raise ValueError("continuous_axis symmetry requires axis_cad")
        return continuous_axis_rotation_error(predicted, target, axis_cad)
    if symmetry_type in ("none", "discrete"):
        if symmetry_transforms is None:
            if symmetry_type == "discrete":
                raise ValueError("discrete symmetry requires symmetry_transforms")
            symmetry_transforms = torch.eye(3, device=predicted.device, dtype=predicted.dtype).unsqueeze(0)
        return discrete_symmetry_rotation_error(predicted, target, symmetry_transforms)
    raise ValueError(f"Unsupported symmetry type: {symmetry_type!r}")


def _check_rotations(predicted: Tensor, target: Tensor) -> None:
    if predicted.shape[-2:] != (3, 3) or target.shape[-2:] != (3, 3):
        raise ValueError("Rotation tensors must end in 3x3")
