"""Model-owned three-rank execution for ASPD v3 main-2b-v1.

Each rank owns one frozen backbone, its 56 matrix dictionaries, its 16 activation
decoders, and its model-specific packing/intra-model modules.  Only 8x512 latent tokens
per site are exchanged.  Cross-model modules are replicated and receive the sum of the
three model-specific activation-objective gradients.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from aspd.multimodel.v3.encoder import (
    CrossModelTransformer,
    FeatureReadout,
    IntraModelTransformer,
    LearnedProjectionPacker,
    concatenate_layer_activations,
    pack_cross_model_tokens,
)
from aspd.multimodel.v3.gating import V3BatchTopK
from aspd.multimodel.v3.main_2b_config import MODEL_ORDER_2B, V3Main2BV1Config
from aspd.multimodel.v3.objective import (
    auxk_dead_feature_loss,
    masked_fvu,
    matryoshka_activation_fvu,
)
from aspd.multimodel.v3.reconstruction import ActivationReconstructionBank, RankOneMatrixBank
from aspd.multimodel.v3.registry import ModelFamilyLayout
from aspd.multimodel.v3.sparse_support import SparseSupport, prepare_sparse_support

SITE_ORDER = ("pre", "mid")


@dataclass(frozen=True)
class LocalCaptureBatch:
    model_key: str
    activations: Mapping[str, Sequence[Tensor]]
    matrix_inputs: Mapping[str, Tensor]
    matrix_outputs: Mapping[str, Tensor]
    valid_anchors: Tensor

    def validate(self, layout: ModelFamilyLayout) -> None:
        if self.model_key != layout.key or tuple(self.activations) != SITE_ORDER:
            raise ValueError("local capture model/site identity mismatch")
        grid = tuple(self.valid_anchors.shape)
        if self.valid_anchors.dtype != torch.bool or len(grid) != 2:
            raise ValueError("local valid anchors must be boolean [B,A]")
        for site in SITE_ORDER:
            values = self.activations[site]
            if len(values) != 8 or any(
                value.shape != (*grid, layout.hidden_size) for value in values
            ):
                raise ValueError("local capture residual shape mismatch")
        targets = {target.key: target for target in layout.targets()}
        if set(self.matrix_inputs) != set(targets) or set(self.matrix_outputs) != set(targets):
            raise ValueError("local capture must contain all 56 owned matrix targets")
        for key, target in targets.items():
            if self.matrix_inputs[key].shape != (*grid, target.d_in):
                raise ValueError(f"{key} input capture shape mismatch")
            if self.matrix_outputs[key].shape != (*grid, target.d_out):
                raise ValueError(f"{key} output capture shape mismatch")


class LocalSiteEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        config: V3Main2BV1Config,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        architecture = config.architecture
        self.packer = LearnedProjectionPacker(
            8 * hidden_size,
            architecture.latent_tokens_per_model,
            architecture.packed_token_width,
            device=device,
            dtype=dtype,
        )
        self.intra = IntraModelTransformer(
            architecture.packed_token_width,
            layers=architecture.intra_model_layers,
            heads=architecture.intra_model_heads,
            device=device,
            dtype=dtype,
        )

    def forward(self, activations: Sequence[Tensor]) -> Tensor:
        return self.intra(self.packer(concatenate_layer_activations(activations)))


class SharedSiteEncoder(nn.Module):
    def __init__(
        self,
        config: V3Main2BV1Config,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        architecture = config.architecture
        packed_width = architecture.packed_token_width + len(MODEL_ORDER_2B) + 2
        self.input_adapter = nn.Linear(
            packed_width, architecture.cross_model_width, device=device, dtype=dtype
        )
        self.cross = CrossModelTransformer(
            architecture.cross_model_width,
            layers=architecture.cross_model_layers,
            heads=architecture.cross_model_heads,
            device=device,
            dtype=dtype,
        )
        self.readout = FeatureReadout(
            architecture.cross_model_width,
            architecture.dictionary_size,
            architecture.feature_readout,
            device=device,
            dtype=dtype,
        )

    def forward(self, tokens: Mapping[str, Tensor], valid_anchors: Tensor) -> Tensor:
        packed = pack_cross_model_tokens(tokens, model_order=MODEL_ORDER_2B)
        return self.readout(self.cross(self.input_adapter(packed.values)), valid_anchors)


def _differentiable_all_gather(local_tokens: Tensor) -> tuple[Tensor, ...]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("model-owned cross-model encoding requires initialized distributed")
    if dist.get_world_size() != len(MODEL_ORDER_2B):
        raise RuntimeError("model-owned ASPD v3 requires exactly three ranks")
    from torch.distributed.nn.functional import all_gather

    gathered = all_gather(local_tokens)
    return tuple(gathered)


class ModelOwnedMain2B(nn.Module):
    """One rank's owned trainable state plus replicated shared PRE/MID encoders."""

    def __init__(
        self,
        config: V3Main2BV1Config,
        registry: Mapping[str, ModelFamilyLayout],
        local_model_key: str,
        *,
        native_biases: Mapping[str, Tensor | None] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if tuple(registry) != MODEL_ORDER_2B or local_model_key not in registry:
            raise ValueError("model-owned registry/local rank assignment is invalid")
        self.config_fingerprint = config.fingerprint()
        self.local_model_key = local_model_key
        self.model_order = MODEL_ORDER_2B
        self.layout = registry[local_model_key]
        self.targets = {target.key: target for target in self.layout.targets()}
        if len(self.targets) != 56:
            raise ValueError("each model-owned rank must own exactly 56 matrix targets")
        self.local_encoders = nn.ModuleDict(
            {
                site: LocalSiteEncoder(
                    self.layout.hidden_size, config, device=device, dtype=dtype
                )
                for site in SITE_ORDER
            }
        )
        self.shared_encoders = nn.ModuleDict(
            {
                site: SharedSiteEncoder(config, device=device, dtype=dtype)
                for site in SITE_ORDER
            }
        )
        self.gates = nn.ModuleDict(
            {
                site: V3BatchTopK(
                    config.architecture.dictionary_size,
                    config.architecture.batch_topk_k,
                    pool_anchors=config.architecture.pool_tokens,
                )
                for site in SITE_ORDER
            }
        )
        self.activation_banks = nn.ModuleDict(
            {
                site: ActivationReconstructionBank(
                    {local_model_key: [self.layout.hidden_size] * 8},
                    config.architecture.dictionary_size,
                    device=device,
                    dtype=dtype,
                )
                for site in SITE_ORDER
            }
        )
        self.matrix_bank = RankOneMatrixBank(
            tuple(self.targets.values()),
            config.architecture.dictionary_size,
            native_biases,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def synchronize_shared_state_(self, source_rank: int = 0) -> None:
        if not dist.is_initialized():
            raise RuntimeError("shared-state synchronization requires distributed")
        for value in self.shared_encoders.state_dict().values():
            dist.broadcast(value, source_rank)

    def encode(
        self, batch: LocalCaptureBatch, *, update_statistics: bool = True
    ) -> tuple[dict[str, Tensor], dict[str, Any], dict[str, SparseSupport], dict[str, SparseSupport]]:
        batch.validate(self.layout)
        scores: dict[str, Tensor] = {}
        sparse: dict[str, Any] = {}
        continuous_support: dict[str, SparseSupport] = {}
        binary_support: dict[str, SparseSupport] = {}
        for site in SITE_ORDER:
            with torch.profiler.record_function(f"aspd/{site}/local_packing_intra"):
                local_tokens = self.local_encoders[site](batch.activations[site])
            with torch.profiler.record_function(f"aspd/{site}/latent_all_gather"):
                gathered = _differentiable_all_gather(local_tokens)
            tokens = {key: gathered[index] for index, key in enumerate(self.model_order)}
            with torch.profiler.record_function(f"aspd/{site}/cross_encoder_readout"):
                scores[site] = self.shared_encoders[site](tokens, batch.valid_anchors)
            with torch.profiler.record_function(f"aspd/{site}/batch_topk"):
                sparse[site] = self.gates[site](
                    scores[site], batch.valid_anchors, update_statistics=update_statistics
                )
                continuous_support[site] = prepare_sparse_support(
                    sparse[site].values, batch.valid_anchors, binary=False
                )
                binary_support[site] = prepare_sparse_support(
                    sparse[site].values.detach(), batch.valid_anchors, binary=True
                )
        return scores, sparse, continuous_support, binary_support

    def activation_loss_terms(
        self,
        batch: LocalCaptureBatch,
        scores: Mapping[str, Tensor],
        sparse: Mapping[str, Any],
        support: Mapping[str, SparseSupport],
    ) -> Iterator[tuple[str, Tensor, Tensor, float]]:
        coefficient = 0.5 / (len(self.model_order) * 8)
        for site in SITE_ORDER:
            decoders = self.activation_banks[site].decoders[self.local_model_key]
            for layer_offset, (decoder, target) in enumerate(
                zip(decoders, batch.activations[site], strict=True)
            ):
                activation, full = matryoshka_activation_fvu(
                    sparse[site].values,
                    decoder,
                    target,
                    batch.valid_anchors,
                    support=support[site],
                )
                auxk = auxk_dead_feature_loss(
                    scores[site],
                    decoder,
                    target,
                    full,
                    batch.valid_anchors,
                    self.gates[site].dead_features,
                    top_k_aux=self.config_top_k_aux,
                )
                yield f"{site}/{self.local_model_key}/window_{layer_offset}", activation, auxk, coefficient

    @property
    def config_top_k_aux(self) -> int:
        return 2048

    def internal_loss_terms(
        self,
        batch: LocalCaptureBatch,
        support: Mapping[str, SparseSupport],
    ) -> Iterator[tuple[str, str, Tensor, float]]:
        global_counts = {"pre": 3 * 8 * 3, "mid": 3 * 8 * 4}
        dummy_code = batch.valid_anchors.new_zeros(
            *batch.valid_anchors.shape, support["pre"].n_features, dtype=torch.bfloat16
        )
        for key, target in self.targets.items():
            site = "pre" if target.grounding_kind == "residual_pre" else "mid"
            prediction = self.matrix_bank.reconstruct(
                key,
                batch.matrix_inputs[key],
                dummy_code,
                batch.valid_anchors,
                code_semantics="binary_support",
                support=support[site],
                compute_dtype=(
                    torch.bfloat16
                    if batch.matrix_inputs[key].device.type == "cuda"
                    else batch.matrix_inputs[key].dtype
                ),
            )
            value = masked_fvu(prediction, batch.matrix_outputs[key], batch.valid_anchors)
            yield key, site, value, 0.5 / global_counts[site]

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.shared_encoders.parameters()

    @torch.no_grad()
    def all_reduce_shared_gradients_(self) -> None:
        if not dist.is_initialized():
            raise RuntimeError("shared gradient reduction requires distributed")
        for parameter in self.shared_parameters():
            if parameter.grad is None:
                raise RuntimeError("shared encoder parameter lost its activation gradient")
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def normalize_decoders_(self, *, project_gradients: bool = True) -> None:
        self.matrix_bank.normalize_write_decoders_(project_gradients=project_gradients)
        for bank in self.activation_banks.values():
            bank.normalize_decoders_(project_gradients=project_gradients)


def build_model_owned_adam(
    model: ModelOwnedMain2B, config: V3Main2BV1Config
) -> torch.optim.Adam:
    rank_one = list(model.matrix_bank.parameters())
    activation = list(model.activation_banks.parameters())
    encoder = list(model.local_encoders.parameters()) + list(model.shared_encoders.parameters())
    groups = (rank_one, activation, encoder)
    ids = [id(parameter) for group in groups for parameter in group]
    if len(ids) != len(set(ids)) or set(ids) != {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }:
        raise ValueError("model-owned optimizer groups are overlapping or incomplete")
    return torch.optim.Adam(
        [
            {"params": rank_one, "lr": config.optimization.rank_one_lr},
            {"params": activation, "lr": config.optimization.activation_decoder_lr},
            {"params": encoder, "lr": config.optimization.encoder_lr},
        ],
        betas=config.optimization.betas,
        weight_decay=config.optimization.weight_decay,
    )


def model_owned_parameter_counts(
    config: V3Main2BV1Config, registry: Mapping[str, ModelFamilyLayout]
) -> dict[str, Any]:
    """Analytical counts used before allocating the multi-billion-parameter banks."""

    c = config.architecture.dictionary_size
    width = config.architecture.packed_token_width
    feedforward = 4 * width
    transformer_layer = (
        4 * width * width + 2 * width * feedforward + 9 * width + feedforward
    )
    intra_all_sites = (
        2 * config.architecture.intra_model_layers * transformer_layer
    )
    cross_per_site = config.architecture.cross_model_layers * transformer_layer
    adapter_per_site = (width + len(MODEL_ORDER_2B) + 2) * width + width
    readout_per_site = width * c + c
    shared_replicated = 2 * (cross_per_site + adapter_per_site + readout_per_site)
    per_model: dict[str, dict[str, int]] = {}
    for key, layout in registry.items():
        rank_one = sum((target.d_in + target.d_out) * c for target in layout.targets())
        activation = 2 * 8 * (c * layout.hidden_size + layout.hidden_size)
        packer = 2 * ((8 * layout.hidden_size) * (8 * 512) + 8 * 512)
        per_model[key] = {
            "rank_one": rank_one,
            "activation_decoders": activation,
            "local_packers": packer,
            "local_intra_transformers": intra_all_sites,
            "shared_encoder_replica": shared_replicated,
        }
    unique_total = sum(
        value["rank_one"]
        + value["activation_decoders"]
        + value["local_packers"]
        + value["local_intra_transformers"]
        for value in per_model.values()
    ) + shared_replicated
    allocated_total = sum(sum(value.values()) for value in per_model.values())
    return {
        "per_model": per_model,
        "rank_one_total": sum(value["rank_one"] for value in per_model.values()),
        "activation_decoder_total": sum(
            value["activation_decoders"] for value in per_model.values()
        ),
        "unique_scientific_trainable_parameters": unique_total,
        "allocated_parameters_including_three_shared_replicas": allocated_total,
        "maximum_owner_parameters": max(sum(value.values()) for value in per_model.values()),
        "estimated_runtime_param_grad_adam_bytes_fp32": {
            key: sum(value.values()) * 16 for key, value in per_model.items()
        },
        "estimated_checkpoint_bytes_model_plus_adam_fp32": allocated_total * 12,
    }
