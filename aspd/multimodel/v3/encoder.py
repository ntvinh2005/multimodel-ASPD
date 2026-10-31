"""Hierarchical ASPD v3 encoder stages with explicit experimental packing choices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MODEL_ORDER = ("qwen", "gemma", "phi")


def concatenate_layer_activations(activations: Sequence[Tensor]) -> Tensor:
    """Concatenate exactly eight ascending layer activations on the channel axis."""

    if len(activations) != 8:
        raise ValueError("v3 intra-model input requires exactly eight layer activations")
    grid = activations[0].shape[:2]
    if any(value.ndim != 3 or value.shape[:2] != grid for value in activations):
        raise ValueError("all v3 layer activations must have shape [B,A,d] on one anchor grid")
    return torch.cat(tuple(activations), dim=-1)


class InputPacker(Protocol):
    latent_tokens: int
    token_width: int

    def __call__(self, concatenated: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class ChannelSlicePacker:
    """Parameter-free experimental option; not an approved main-run packing method."""

    concatenated_width: int
    latent_tokens: int

    @property
    def token_width(self) -> int:
        if self.concatenated_width % self.latent_tokens:
            raise ValueError(
                "channel_slice requires eight-layer width divisible by latent_tokens_per_model"
            )
        return self.concatenated_width // self.latent_tokens

    def __call__(self, concatenated: Tensor) -> Tensor:
        if concatenated.shape[-1] != self.concatenated_width:
            raise ValueError("channel_slice input width differs from the registered model width")
        return concatenated.reshape(
            *concatenated.shape[:2], self.latent_tokens, self.token_width
        )


class LearnedProjectionPacker(nn.Module):
    """Explicitly provisional learned packing; selecting it does not approve the method."""

    def __init__(self, concatenated_width: int, latent_tokens: int, token_width: int):
        super().__init__()
        self.latent_tokens = latent_tokens
        self.token_width = token_width
        self.projection = nn.Linear(concatenated_width, latent_tokens * token_width)

    def forward(self, concatenated: Tensor) -> Tensor:
        projected = self.projection(concatenated)
        return projected.reshape(*concatenated.shape[:2], self.latent_tokens, self.token_width)


class IntraModelTransformer(nn.Module):
    def __init__(self, width: int, *, layers: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if width % heads:
            raise ValueError("intra-model token width must be divisible by attention heads")
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=int(width * mlp_ratio),
            batch_first=True,
            norm_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 4:
            raise ValueError("intra-model tokens must have shape [B,A,a,d]")
        batch, anchors, count, width = tokens.shape
        encoded = self.encoder(tokens.reshape(batch * anchors, count, width))
        return encoded.reshape(batch, anchors, count, width)


@dataclass(frozen=True)
class CrossModelTokens:
    values: Tensor
    identity_indices: tuple[int, ...]
    separator_positions: tuple[int, int]


def pack_cross_model_tokens(
    tokens: Mapping[str, Tensor],
    *,
    model_order: Sequence[str] = MODEL_ORDER,
) -> CrossModelTokens:
    """Build ``A_Qwen, SEP, A_Gemma, SEP, A_Phi`` with parameter-free identities."""

    if tuple(tokens) != tuple(model_order):
        raise ValueError("intra-model tokens must use deterministic qwen/gemma/phi order")
    shapes = [tokens[key].shape for key in model_order]
    if any(len(shape) != 4 for shape in shapes):
        raise ValueError("cross-model inputs must have shape [B,A,a,d]")
    batch_anchor_count = {shape[:3] for shape in shapes}
    if len(batch_anchor_count) != 1:
        raise ValueError("all models must share batch, anchor, and latent-token axes")
    batch, anchors, latent_tokens = shapes[0][:3]
    max_width = max(shape[-1] for shape in shapes)
    identity_width = len(model_order) + 2

    pieces: list[Tensor] = []
    identity_indices: list[int] = []
    separator_positions: list[int] = []
    position = 0
    for model_index, key in enumerate(model_order):
        value = F.pad(tokens[key], (0, max_width - tokens[key].shape[-1]))
        identity = F.one_hot(
            torch.full(
                (batch, anchors, latent_tokens),
                model_index,
                device=value.device,
                dtype=torch.long,
            ),
            num_classes=identity_width,
        ).to(value.dtype)
        pieces.append(torch.cat((value, identity), dim=-1))
        identity_indices.extend([model_index] * latent_tokens)
        position += latent_tokens
        if model_index < len(model_order) - 1:
            separator_identity = len(model_order) + model_index
            separator_content = torch.zeros(
                batch, anchors, 1, max_width, device=value.device, dtype=value.dtype
            )
            separator_id = F.one_hot(
                torch.full(
                    (batch, anchors, 1),
                    separator_identity,
                    device=value.device,
                    dtype=torch.long,
                ),
                num_classes=identity_width,
            ).to(value.dtype)
            pieces.append(torch.cat((separator_content, separator_id), dim=-1))
            identity_indices.append(separator_identity)
            separator_positions.append(position)
            position += 1
    packed = torch.cat(pieces, dim=2)
    expected = len(model_order) * latent_tokens + len(model_order) - 1
    if packed.shape[2] != expected:
        raise AssertionError("cross-model sequence length construction failed")
    return CrossModelTokens(
        packed,
        tuple(identity_indices),
        (separator_positions[0], separator_positions[1]),
    )


class CrossModelTransformer(nn.Module):
    def __init__(self, width: int, *, layers: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if width % heads:
            raise ValueError("cross-model token width must be divisible by attention heads")
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=int(width * mlp_ratio),
            batch_first=True,
            norm_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 4:
            raise ValueError("cross-model tokens must have shape [B,A,3a+2,d]")
        batch, anchors, count, width = tokens.shape
        output = self.encoder(tokens.reshape(batch * anchors, count, width))
        return output.reshape(batch, anchors, count, width)


class FeatureReadout(nn.Module):
    """Experimental readout interface; the main-run choice remains unresolved."""

    def __init__(self, width: int, n_features: int, method: Literal["mean"]):
        super().__init__()
        if method != "mean":
            raise NotImplementedError("only the explicitly labeled experimental mean readout exists")
        self.method = method
        self.output = nn.Linear(width, n_features)

    def forward(self, tokens: Tensor, valid_anchors: Tensor) -> Tensor:
        if valid_anchors.shape != tokens.shape[:2]:
            raise ValueError("valid_anchors must match the [B,A] aligned-anchor grid")
        pooled = tokens.mean(dim=2)
        scores = torch.relu(self.output(pooled))
        return scores * valid_anchors[..., None].to(scores.dtype)


class HierarchicalV3Encoder(nn.Module):
    """Shape-checked implementation of the approved stages, with provisional choices explicit."""

    def __init__(
        self,
        packers: Mapping[str, InputPacker | nn.Module],
        *,
        intra_layers: int,
        intra_heads: int,
        cross_layers: int,
        cross_heads: int,
        n_features: int,
        readout: Literal["mean"],
    ):
        super().__init__()
        if tuple(packers) != MODEL_ORDER:
            raise ValueError("v3 encoder packers must be ordered qwen, gemma, phi")
        widths = {key: int(value.token_width) for key, value in packers.items()}
        incompatible_intra = {
            key: width for key, width in widths.items() if width % intra_heads
        }
        if incompatible_intra:
            raise ValueError(
                "intra-model token widths must be divisible by attention heads before "
                f"module allocation: heads={intra_heads}, widths={incompatible_intra}"
            )
        cross_width = max(widths.values()) + len(MODEL_ORDER) + 2
        if cross_width % cross_heads:
            raise ValueError(
                "cross-model token width must be divisible by attention heads before "
                f"module allocation: content_width={max(widths.values())}, "
                f"identity_width={len(MODEL_ORDER) + 2}, cross_width={cross_width}, "
                f"heads={cross_heads}"
            )
        latent_tokens = {key: int(value.latent_tokens) for key, value in packers.items()}
        if len(set(latent_tokens.values())) != 1:
            raise ValueError("all three v3 models must use the same latent-token count a")
        self.packer_modules = nn.ModuleDict(
            {key: value for key, value in packers.items() if isinstance(value, nn.Module)}
        )
        self._packers = dict(packers)
        self.intra = nn.ModuleDict(
            {
                key: IntraModelTransformer(width, layers=intra_layers, heads=intra_heads)
                for key, width in widths.items()
            }
        )
        self.cross = CrossModelTransformer(
            cross_width, layers=cross_layers, heads=cross_heads
        )
        self.readout = FeatureReadout(cross_width, n_features, readout)
        self.latent_tokens = latent_tokens

    def forward(
        self,
        layer_activations: Mapping[str, Sequence[Tensor]],
        valid_anchors: Tensor,
    ) -> Tensor:
        if tuple(layer_activations) != MODEL_ORDER:
            raise ValueError("v3 activations must be ordered qwen, gemma, phi")
        intra_tokens: dict[str, Tensor] = {}
        for key in MODEL_ORDER:
            concatenated = concatenate_layer_activations(layer_activations[key])
            packed = self._packers[key](concatenated)  # type: ignore[operator]
            intra_tokens[key] = self.intra[key](packed)
        cross = pack_cross_model_tokens(intra_tokens)
        expected = 3 * next(iter(self.latent_tokens.values())) + 2
        if cross.values.shape[2] != expected:
            raise AssertionError("v3 cross-model sequence must have length 3a+2")
        encoded = self.cross(cross.values)
        return self.readout(encoded, valid_anchors)
