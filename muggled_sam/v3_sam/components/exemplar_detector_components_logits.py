#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .exemplar_detector_attention import MLP2LayersPostNorm

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class PresenceScoreMLP(nn.Module):
    """
    Logits-only version of PresenceScoreMLP.

    This matches the original implementation but returns the clamped logits
    (no sigmoid applied), so callers can choose how to transform them.
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, clamp_bound: float = 10.0):

        # Inherit from parent
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(features_per_token),
            nn.Linear(features_per_token, features_per_token),
            nn.ReLU(),
            nn.Linear(features_per_token, features_per_token),
            nn.ReLU(),
            nn.Linear(features_per_token, 1),
        )

        # Store both bounds for ease of use
        self._min_clamp = -clamp_bound
        self._max_clamp = clamp_bound

    def forward(self, tokens_bnc: Tensor) -> Tensor:
        """Reduces input token channels to a single value (logit), also alters shape from: BxNxC -> BxN"""
        out_tokens_bn = self.layers(tokens_bnc).squeeze(-1)
        return out_tokens_bn.clamp(min=self._min_clamp, max=self._max_clamp)

    # .................................................................................................................


class DetectionScoring(nn.Module):
    """
    Logits-only version of DetectionScoring.

    Computes per-detection confidence logits (no sigmoid applied).
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int = 256,
        mlp_ratio: float = 8.0,
        clamp_bound: float = 12.0,
    ):
        # Inherit from parent
        super().__init__()

        # Main model components
        self.exemplar_mlp = MLP2LayersPostNorm(features_per_token, mlp_ratio)
        self.exemplar_proj = nn.Linear(features_per_token, features_per_token)
        self.detection_token_proj = nn.Linear(features_per_token, features_per_token)

        # Store values for use at runtime
        self.register_buffer("_scale", torch.tensor(1.0 / features_per_token).sqrt(), persistent=False)
        self._min_clamp = -clamp_bound
        self._max_clamp = clamp_bound

    # .................................................................................................................

    def forward(
        self,
        detection_tokens_bnc: Tensor,
        exemplar_tokens_bnc: Tensor,
        exemplar_mask_bn: Tensor | None = None,
    ) -> Tensor:
        """
        Returns:
            detection_confidence_logits (shape: BxN, B batches, N detections)
        """

        # Preprocess tokens before averaging
        exm_tokens_mlp_bnc = self.exemplar_mlp(exemplar_tokens_bnc)

        # Fill in missing mask before computing averaged exemplar token
        # -> Note: In this case, could just directly compute average with exm_tokens_mlp.mean(...)
        #    however, this leads to a major numerical difference (not sure why?) and ends
        #    up having a *substantial* negative impact on detection scores
        # -> For example: torch.allclose(tokens.sum(0)/num_tokens, tokens.mean(0)) comes out False!
        if exemplar_mask_bn is None:
            exm_b, exm_n, _ = exemplar_tokens_bnc.shape
            exemplar_mask_bn = torch.zeros((exm_b, exm_n), dtype=torch.bool, device=exemplar_tokens_bnc.device)
        exemplar_mask_bn = exemplar_mask_bn.to(dtype=torch.bool)
        inv_mask_bn1 = (~exemplar_mask_bn).to(exm_tokens_mlp_bnc).unsqueeze(-1)
        num_valid_exm_b1 = torch.clamp(inv_mask_bn1.sum(1), min=1.0)
        averaged_exm_bc = (exm_tokens_mlp_bnc * inv_mask_bn1).sum(dim=1) / num_valid_exm_b1

        # Do attention-like softmax(q*k*scale)
        det_tokens_proj_bnc = self.detection_token_proj(detection_tokens_bnc)
        avg_exm_proj_bc1 = self.exemplar_proj(averaged_exm_bc).unsqueeze(-1)
        scores_bn1 = torch.matmul(det_tokens_proj_bnc, avg_exm_proj_bc1) * self._scale
        return scores_bn1.squeeze(-1).clamp(min=self._min_clamp, max=self._max_clamp)

    # .................................................................................................................
