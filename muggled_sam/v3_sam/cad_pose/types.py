"""Shared data contracts for CAD-pose training and inference."""

from __future__ import annotations

from dataclasses import dataclass, fields

from torch import Tensor

from .geometry import (
    aabb_translation_from_surface_centroid,
    normalized_to_pixel,
    reconstruct_translation,
)


@dataclass(frozen=True)
class CADPosePredictions:
    """Per-candidate pose predictions.

    Tensor fields share leading ``B x N`` dimensions.
    ``center_residual_bn2`` is expressed in predicted-box width/height units.
    """

    center_residual_bn2: Tensor
    center_uv_norm_bn2: Tensor
    log_depth_bn: Tensor
    rotation_6d_bn6: Tensor
    rotation_matrix_bn33: Tensor
    pose_score_logits_bn: Tensor
    pose_score_bn: Tensor
    translation_m_bn3: Tensor | None = None
    centroid_m_bn3: Tensor | None = None
    cad_dimensions_m_b3: Tensor | None = None
    cad_effective_surface_centroid_m_b3: Tensor | None = None

    def index_candidates(self, indices: Tensor) -> "CADPosePredictions":
        """Apply one candidate-index selection consistently to every pose output."""

        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None or field.name in {
                "cad_dimensions_m_b3",
                "cad_effective_surface_centroid_m_b3",
            }:
                values[field.name] = value
            else:
                values[field.name] = value[:, indices]
        return CADPosePredictions(**values)

    def index_batch(self, index: int) -> "CADPosePredictions":
        """Select one image while retaining a singleton batch dimension."""

        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = None if value is None else value[index : index + 1]
        return CADPosePredictions(**values)

    def with_translation(
        self,
        intrinsics_b33: Tensor,
        image_size_wh: tuple[int, int],
        effective_surface_centroid_m_b3: Tensor | None = None,
    ) -> "CADPosePredictions":
        """Back-project the centroid and recover public AABB-origin translation."""

        centers_px = normalized_to_pixel(self.center_uv_norm_bn2, image_size_wh)
        centroid = reconstruct_translation(centers_px, self.log_depth_bn, intrinsics_b33)
        if effective_surface_centroid_m_b3 is None:
            effective_surface_centroid_m_b3 = self.cad_effective_surface_centroid_m_b3
        if effective_surface_centroid_m_b3 is None:
            effective_surface_centroid_m_b3 = centroid.new_zeros((centroid.shape[0], 3))
        effective_surface_centroid_m_b3 = effective_surface_centroid_m_b3.to(centroid)
        translation = aabb_translation_from_surface_centroid(
            centroid,
            self.rotation_matrix_bn33,
            effective_surface_centroid_m_b3,
        )
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values["translation_m_bn3"] = translation
        values["centroid_m_bn3"] = centroid
        values["cad_effective_surface_centroid_m_b3"] = effective_surface_centroid_m_b3
        return CADPosePredictions(**values)


@dataclass(frozen=True)
class CADPoseTarget:
    """Ground-truth values for one CAD-pose training instance."""

    center_uv_norm: Tensor
    log_depth: Tensor
    rotation_matrix: Tensor
    translation_m: Tensor
    dimensions_m: Tensor
    centroid_m: Tensor | None = None
    effective_surface_centroid_m: Tensor | None = None
    point_query_m: Tensor | None = None
    point_target_m: Tensor | None = None
    point_set_eligible: bool = False
    symmetry_type: str = "none"
    symmetry_transforms: Tensor | None = None
    axis_cad: Tensor | None = None
    rotation_eligible: bool = True
