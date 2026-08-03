"""Reference-view metadata and pose-conditioned exemplar token adapters.

The image-exemplar encoder emits a variable number of tokens for every render.
``ExemplarViewBundle`` preserves which render produced each token and the
render-camera orientation in the canonical CAD frame.  The adapter is an
additive, zero-initialized residual, so ``mode="none"`` is an exact bypass and
all learned experimental modes start from the same baseline function.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


EXEMPLAR_VIEW_MODES = (
    "none",
    "camera",
    "shuffled_camera",
    "zero_camera",
    "view_id",
)
LEARNED_EXEMPLAR_VIEW_MODES = tuple(mode for mode in EXEMPLAR_VIEW_MODES if mode != "none")


def load_exemplar_view_adapter_for_inference(
    model: nn.Module,
    checkpoint: dict,
    requested_mode: str = "auto",
) -> str:
    """Load adapter state and resolve ``auto`` from checkpoint provenance."""

    checkpoint_mode = str((checkpoint.get("args") or {}).get("exemplar_view_mode", "none"))
    if checkpoint_mode not in EXEMPLAR_VIEW_MODES:
        raise ValueError(f"Checkpoint contains unknown exemplar-view mode {checkpoint_mode!r}")
    mode = checkpoint_mode if requested_mode == "auto" else requested_mode
    if mode not in EXEMPLAR_VIEW_MODES:
        raise ValueError(f"Unknown requested exemplar-view mode {mode!r}")
    if requested_mode != "auto" and checkpoint_mode != mode:
        raise ValueError(
            f"Checkpoint was trained with exemplar-view mode {checkpoint_mode!r}, not {mode!r}"
        )
    encoder = getattr(model, "exemplar_view_pose_encoder")
    state = checkpoint.get("exemplar_view_pose_encoder")
    if state is None:
        if mode != "none":
            raise KeyError("Checkpoint has no exemplar-view adapter state")
        return mode
    version = int(checkpoint.get("exemplar_view_pose_architecture_version", 1))
    if version != encoder.architecture_version:
        raise ValueError(
            f"Checkpoint adapter version {version} is incompatible with model version "
            f"{encoder.architecture_version}"
        )
    config = checkpoint.get("exemplar_view_pose_architecture_config")
    if config is not None and config != encoder.architecture_config():
        raise ValueError("Checkpoint exemplar-view adapter architecture config is incompatible")
    encoder.load_state_dict(state)
    return mode


def normalize_view_id(value: object) -> str:
    """Normalize numeric view IDs while preserving non-numeric identifiers."""

    text = str(value).strip()
    try:
        return str(int(text))
    except ValueError:
        return text


def _validate_rotation(rotation: Tensor, *, view_id: str, metadata_path: Path) -> None:
    if rotation.shape != (3, 3) or not torch.isfinite(rotation).all():
        raise ValueError(f"Invalid camera rotation for view {view_id!r} in {metadata_path}")
    identity = torch.eye(3, dtype=rotation.dtype)
    if not torch.allclose(rotation.transpose(0, 1) @ rotation, identity, atol=1e-5, rtol=0.0):
        raise ValueError(f"Non-orthonormal camera rotation for view {view_id!r} in {metadata_path}")
    if not torch.isclose(torch.det(rotation), torch.tensor(1.0, dtype=rotation.dtype), atol=1e-5, rtol=0.0):
        raise ValueError(f"Improper camera rotation for view {view_id!r} in {metadata_path}")


def load_reference_view_rotations(metadata_path: Path, view_ids: Sequence[str]) -> Tensor:
    """Load ``R_refcam_cv_from_cad`` for the requested views in request order."""

    try:
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference-view conditioning requires render metadata: {metadata_path}"
        ) from None
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read render metadata {metadata_path}: {error}") from error

    if not isinstance(metadata, dict):
        raise ValueError(f"Render metadata must be a JSON object: {metadata_path}")
    if metadata.get("camera_frame") != "opencv_x_right_y_down_z_forward":
        raise ValueError(f"Unsupported or missing camera frame in {metadata_path}")
    raw_views = metadata.get("views")
    if not isinstance(raw_views, list):
        raise ValueError(f"Render metadata has no views list: {metadata_path}")

    views_by_id = {}
    for view in raw_views:
        if isinstance(view, dict) and "view_id" in view:
            key = normalize_view_id(view["view_id"])
            if key in views_by_id:
                raise ValueError(f"Duplicate view ID {key!r} in {metadata_path}")
            views_by_id[key] = view

    rotations = []
    for requested_id in view_ids:
        key = normalize_view_id(requested_id)
        view = views_by_id.get(key)
        if view is None:
            raise ValueError(f"View {requested_id!r} is absent from {metadata_path}")
        try:
            rotation = torch.as_tensor(view["R_refcam_cv_from_cad"], dtype=torch.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid R_refcam_cv_from_cad for view {requested_id!r} in {metadata_path}"
            ) from error
        _validate_rotation(rotation, view_id=str(requested_id), metadata_path=metadata_path)
        rotations.append(rotation)
    if not rotations:
        return torch.empty((0, 3, 3), dtype=torch.float32)
    return torch.stack(rotations, dim=0)


@dataclass(frozen=True)
class ExemplarViewBundle:
    """Concatenated exemplar tokens plus their reference-view provenance."""

    tokens_bnc: Tensor
    token_view_indices_n: Tensor
    view_rotations_v33: Tensor
    view_ids: tuple[str, ...]
    object_id: str = ""

    def __post_init__(self) -> None:
        if self.tokens_bnc.ndim != 3 or self.tokens_bnc.shape[0] != 1:
            raise ValueError("tokens_bnc must have shape 1 x N x C")
        num_tokens = self.tokens_bnc.shape[1]
        if self.token_view_indices_n.shape != (num_tokens,):
            raise ValueError("token_view_indices_n must have shape N")
        num_views = len(self.view_ids)
        if self.view_rotations_v33.shape != (num_views, 3, 3):
            raise ValueError("view_rotations_v33 must have shape V x 3 x 3")
        if num_views:
            rotations = self.view_rotations_v33
            identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype).expand(
                num_views, -1, -1
            )
            if (
                not torch.isfinite(rotations).all()
                or not torch.allclose(
                    rotations.transpose(-1, -2) @ rotations,
                    identity,
                    atol=1e-5,
                    rtol=0.0,
                )
                or not torch.allclose(
                    torch.det(rotations),
                    torch.ones(num_views, device=rotations.device, dtype=rotations.dtype),
                    atol=1e-5,
                    rtol=0.0,
                )
            ):
                raise ValueError("view_rotations_v33 must contain proper SO(3) rotations")
        if num_tokens and (
            int(self.token_view_indices_n.min()) < 0
            or int(self.token_view_indices_n.max()) >= num_views
        ):
            raise ValueError("Token-to-view indices are outside the available view range")

    def detach_cpu(self) -> "ExemplarViewBundle":
        return ExemplarViewBundle(
            tokens_bnc=self.tokens_bnc.detach().cpu(),
            token_view_indices_n=self.token_view_indices_n.detach().cpu(),
            view_rotations_v33=self.view_rotations_v33.detach().cpu(),
            view_ids=self.view_ids,
            object_id=self.object_id,
        )


class ExemplarViewPoseEncoder(nn.Module):
    """Add a reference camera/view-ID residual to every exemplar token."""

    architecture_version = 1

    def __init__(
        self,
        token_dim: int = 256,
        hidden_dim: int = 128,
        num_fourier_frequencies: int = 3,
        max_views: int = 64,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_fourier_frequencies = int(num_fourier_frequencies)
        self.max_views = int(max_views)
        camera_input_dim = 6 * (1 + 2 * self.num_fourier_frequencies)
        self.camera_mlp = nn.Sequential(
            nn.Linear(camera_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.token_dim),
        )
        self.view_id_embedding = nn.Embedding(self.max_views, self.hidden_dim)
        self.view_id_projection = nn.Linear(self.hidden_dim, self.token_dim)
        self.reset_output_projections()

    def reset_output_projections(self) -> None:
        """Make every learned mode an exact no-op at initialization."""

        nn.init.zeros_(self.camera_mlp[-1].weight)
        nn.init.zeros_(self.camera_mlp[-1].bias)
        nn.init.zeros_(self.view_id_projection.weight)
        nn.init.zeros_(self.view_id_projection.bias)

    def architecture_config(self) -> dict[str, int]:
        return {
            "token_dim": self.token_dim,
            "hidden_dim": self.hidden_dim,
            "num_fourier_frequencies": self.num_fourier_frequencies,
            "max_views": self.max_views,
        }

    @staticmethod
    def _camera_directions_in_cad(rotation_v33: Tensor) -> Tensor:
        # OpenCV camera axes are x-right, y-down, z-forward. Express camera
        # forward and up in the canonical CAD frame without Euler singularities.
        forward_camera = rotation_v33.new_tensor((0.0, 0.0, 1.0))
        up_camera = rotation_v33.new_tensor((0.0, -1.0, 0.0))
        cad_from_camera = rotation_v33.transpose(-1, -2)
        forward_cad = torch.matmul(cad_from_camera, forward_camera)
        up_cad = torch.matmul(cad_from_camera, up_camera)
        return torch.cat((forward_cad, up_cad), dim=-1)

    def _fourier_encode(self, directions_v6: Tensor) -> Tensor:
        features = [directions_v6]
        for frequency_index in range(self.num_fourier_frequencies):
            frequency = torch.pi * float(2**frequency_index)
            features.extend(
                (torch.sin(frequency * directions_v6), torch.cos(frequency * directions_v6))
            )
        return torch.cat(features, dim=-1)

    @staticmethod
    def _stable_permutation(num_views: int, object_id: str, shuffle_seed: int) -> Tensor:
        digest = hashlib.sha256(f"{shuffle_seed}:{object_id}".encode("utf-8")).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "little", signed=False))
        if num_views < 2:
            return torch.arange(num_views)
        view_indices = torch.arange(num_views)
        for _ in range(128):
            permutation = torch.randperm(num_views, generator=generator)
            if not torch.any(permutation == view_indices):
                return permutation
        # The bounded fallback is deterministic and remains a derangement.
        return torch.roll(view_indices, shifts=1)

    def forward_bundle(
        self,
        bundle: ExemplarViewBundle,
        *,
        mode: str,
        shuffle_seed: int = 0,
        tokens_bnc: Tensor | None = None,
    ) -> Tensor:
        """Return a 1 x N x C token tensor conditioned according to ``mode``."""

        if mode not in EXEMPLAR_VIEW_MODES:
            raise ValueError(f"Unknown exemplar-view mode {mode!r}")
        tokens = bundle.tokens_bnc if tokens_bnc is None else tokens_bnc
        if tokens.shape != bundle.tokens_bnc.shape:
            raise ValueError("tokens_bnc override must preserve the bundled token shape")
        if mode == "none":
            return tokens
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"Expected exemplar token dim {self.token_dim}, got {tokens.shape[-1]}"
            )

        parameter = next(self.parameters())
        view_indices = bundle.token_view_indices_n.to(device=tokens.device, dtype=torch.long)
        num_views = len(bundle.view_ids)
        if mode == "view_id":
            if num_views > self.max_views:
                raise ValueError(f"Bundle has {num_views} views, adapter supports {self.max_views}")
            ordinal_ids = torch.arange(num_views, device=parameter.device)
            per_view_delta = self.view_id_projection(self.view_id_embedding(ordinal_ids))
        else:
            rotations = bundle.view_rotations_v33
            if mode == "shuffled_camera":
                permutation = self._stable_permutation(num_views, bundle.object_id, shuffle_seed)
                rotations = rotations[permutation]
            rotations = rotations.to(device=parameter.device, dtype=parameter.dtype)
            directions = self._camera_directions_in_cad(rotations)
            if mode == "zero_camera":
                directions = torch.zeros_like(directions)
            per_view_delta = self.camera_mlp(self._fourier_encode(directions))

        token_delta = per_view_delta[view_indices.to(parameter.device)]
        token_delta = token_delta.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        return tokens + token_delta


def pad_exemplar_view_batch(
    exemplars: Sequence[Tensor | ExemplarViewBundle],
    *,
    device: torch.device,
    pose_encoder: ExemplarViewPoseEncoder | None = None,
    mode: str = "none",
    shuffle_seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Condition, pad, and stack exemplar tensors while retaining baseline parity."""

    if not exemplars:
        raise ValueError("No exemplars provided for batching")
    conditioned = []
    for exemplar in exemplars:
        if isinstance(exemplar, ExemplarViewBundle):
            tokens_on_device = exemplar.tokens_bnc.to(device)
            if pose_encoder is None:
                if mode != "none":
                    raise ValueError(f"Mode {mode!r} requires an ExemplarViewPoseEncoder")
                tokens = tokens_on_device
            else:
                tokens = pose_encoder.forward_bundle(
                    exemplar,
                    mode=mode,
                    shuffle_seed=shuffle_seed,
                    tokens_bnc=tokens_on_device,
                )
        else:
            if mode != "none":
                raise ValueError(f"Mode {mode!r} requires ExemplarViewBundle inputs")
            tokens = exemplar.to(device)
        conditioned.append(tokens)

    feature_dims = {tensor.shape[2] for tensor in conditioned}
    if len(feature_dims) != 1:
        raise ValueError("All exemplar tensors must share the same feature dimension")
    max_tokens = max(tensor.shape[1] for tensor in conditioned)
    feature_dim = conditioned[0].shape[2]
    padded = []
    padding_masks = []
    for tokens in conditioned:
        num_tokens = tokens.shape[1]
        mask = torch.zeros((max_tokens,), device=device, dtype=torch.bool)
        if num_tokens < max_tokens:
            padding = torch.zeros(
                (1, max_tokens - num_tokens, feature_dim),
                device=device,
                dtype=tokens.dtype,
            )
            tokens = torch.cat((tokens, padding), dim=1)
            mask[num_tokens:] = True
        padded.append(tokens)
        padding_masks.append(mask)
    return torch.cat(padded, dim=0), torch.stack(padding_masks, dim=0)
