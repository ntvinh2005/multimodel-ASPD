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

from aspd.multimodel.v3.activation_targets import prepare_activation_target
from aspd.multimodel.v3.encoder import (
    CrossModelTransformer,
    FeatureReadout,
    IntraModelTransformer,
    LearnedProjectionPacker,
    concatenate_layer_activations,
    pack_cross_model_tokens,
)
from aspd.multimodel.v3.gating import V3BatchTopK
from aspd.multimodel.v3.objective import (
    auxk_activation_scores,
    auxk_dead_feature_loss,
    full_activation_fvu,
    masked_fvu,
    matryoshka_activation_fvu,
)
from aspd.multimodel.v3.reconstruction import ActivationReconstructionBank, RankOneMatrixBank
from aspd.multimodel.v3.registry import ModelFamilyLayout
from aspd.multimodel.v3.sparse_support import SparseSupport, prepare_sparse_support

SITE_ORDER = ("pre", "mid")


class IdentityTokenMixer(nn.Module):
    """Parameter-free token mixer used by the controlled linear-encoder arm."""

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 4:
            raise ValueError("linear token mixer requires [B,A,tokens,width]")
        return tokens


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
        config: Any,
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
        mixer = getattr(architecture, "encoder_mixer", "hierarchical_transformer")
        if mixer == "hierarchical_transformer":
            self.intra = IntraModelTransformer(
                architecture.packed_token_width,
                layers=architecture.intra_model_layers,
                heads=architecture.intra_model_heads,
                device=device,
                dtype=dtype,
            )
        elif mixer == "hierarchical_linear":
            self.intra = IdentityTokenMixer()
        else:
            raise ValueError(f"unknown ASPD v3 encoder mixer: {mixer}")

    def forward(self, activations: Sequence[Tensor]) -> Tensor:
        return self.intra(self.packer(concatenate_layer_activations(activations)))


class SharedSiteEncoder(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        model_order: tuple[str, ...],
        device: torch.device | str | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        architecture = config.architecture
        self.model_order = model_order
        packed_width = architecture.packed_token_width + 2 * len(self.model_order) - 1
        self.input_adapter = nn.Linear(
            packed_width, architecture.cross_model_width, device=device, dtype=dtype
        )
        mixer = getattr(architecture, "encoder_mixer", "hierarchical_transformer")
        if mixer == "hierarchical_transformer":
            self.cross = CrossModelTransformer(
                architecture.cross_model_width,
                layers=architecture.cross_model_layers,
                heads=architecture.cross_model_heads,
                device=device,
                dtype=dtype,
            )
        elif mixer == "hierarchical_linear":
            self.cross = IdentityTokenMixer()
        else:
            raise ValueError(f"unknown ASPD v3 encoder mixer: {mixer}")
        self.readout = FeatureReadout(
            architecture.cross_model_width,
            architecture.dictionary_size,
            architecture.feature_readout,
            device=device,
            dtype=dtype,
        )

    def forward(self, tokens: Mapping[str, Tensor], valid_anchors: Tensor) -> Tensor:
        packed = pack_cross_model_tokens(tokens, model_order=self.model_order)
        return self.readout(self.cross(self.input_adapter(packed.values)), valid_anchors)

    def forward_with_logits(
        self, tokens: Mapping[str, Tensor], valid_anchors: Tensor
    ) -> tuple[Tensor, Tensor]:
        packed = pack_cross_model_tokens(tokens, model_order=self.model_order)
        return self.readout.forward_with_logits(
            self.cross(self.input_adapter(packed.values)), valid_anchors
        )


def _differentiable_all_gather(
    local_tokens: Tensor, *, expected_world_size: int
) -> tuple[Tensor, ...]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("model-owned cross-model encoding requires initialized distributed")
    if dist.get_world_size() != expected_world_size:
        raise RuntimeError(f"model-owned ASPD v3 requires exactly {expected_world_size} ranks")
    from torch.distributed.nn.functional import all_gather

    gathered = all_gather(local_tokens)
    return tuple(gathered)


class ModelOwnedMain2B(nn.Module):
    """One rank's owned trainable state plus replicated shared PRE/MID encoders."""

    def __init__(
        self,
        config: Any,
        registry: Mapping[str, ModelFamilyLayout],
        local_model_key: str,
        *,
        native_biases: Mapping[str, Tensor | None] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        sparse_backend: str = "indexed",
    ):
        super().__init__()
        model_order = tuple(getattr(config.architecture, "model_order", tuple(registry)))
        if tuple(registry) != model_order or local_model_key not in registry:
            raise ValueError("model-owned registry/local rank assignment is invalid")
        self.config_fingerprint = config.fingerprint()
        self.local_model_key = local_model_key
        self.model_order = model_order
        self.layout = registry[local_model_key]
        if sparse_backend not in {"indexed", "csr", "dense"}:
            raise ValueError(f"unknown model-owned sparse backend: {sparse_backend}")
        self.sparse_backend = sparse_backend
        self.targets = {target.key: target for target in self.layout.targets()}
        if len(self.targets) != 56:
            raise ValueError("each model-owned rank must own exactly 56 matrix targets")
        self.local_encoders = nn.ModuleDict(
            {
                site: LocalSiteEncoder(self.layout.hidden_size, config, device=device, dtype=dtype)
                for site in SITE_ORDER
            }
        )
        self.shared_encoders = nn.ModuleDict(
            {
                site: SharedSiteEncoder(config, model_order=model_order, device=device, dtype=dtype)
                for site in SITE_ORDER
            }
        )
        self.gates = nn.ModuleDict(
            {
                site: V3BatchTopK(
                    config.architecture.dictionary_size,
                    config.architecture.batch_topk_k,
                    pool_anchors=config.architecture.pool_tokens,
                    device=device,
                    # Every rank computes the same shared-encoder scores from the same
                    # differentiably gathered tokens. Shared parameters are initialized
                    # identically and receive sum-reduced gradients, so statistics are
                    # already replicated exactly; per-pool all-reduces are redundant.
                    sync_statistics=False,
                    materialize_n_selected=False,
                    # SharedSiteEncoder ends in FeatureReadout's explicit ReLU. Avoid
                    # synchronizing the device merely to re-prove that invariant.
                    validate_nonnegative=False,
                )
                for site in SITE_ORDER
            }
        )
        self.activation_banks = nn.ModuleDict(
            {
                site: ActivationReconstructionBank(
                    {local_model_key: [self.layout.hidden_size] * 8},
                    config.architecture.dictionary_size,
                    normalization_mode=getattr(
                        getattr(config, "objective", None),
                        "activation_decoder_normalization",
                        "unit_row_norm_with_radial_gradient_projection",
                    ),
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
        scale_mode = getattr(
            getattr(config, "objective", None),
            "activation_reconstruction_scale",
            "none",
        )
        self.activation_reconstruction_scale_mode = scale_mode
        if scale_mode == "two_model_geomean_log_scale":
            if self.model_order != ("gemma", "smol"):
                raise ValueError("geometric-mean scale is frozen to Gemma/SmolLM2 order")
            self.reconstruction_log_scale = nn.Parameter(
                torch.zeros(len(SITE_ORDER), 8, device=device, dtype=torch.float32)
            )
        elif scale_mode == "none":
            self.register_parameter("reconstruction_log_scale", None)
        else:
            raise ValueError(f"unknown activation reconstruction scale mode: {scale_mode}")
        self.config_top_k_aux = int(getattr(config.architecture, "top_k_aux", 2048))
        objective = getattr(config, "objective", None)
        self.activation_target_normalization = getattr(
            objective, "activation_target_normalization", "none"
        )
        self.activation_target_epsilon = float(
            getattr(objective, "activation_target_epsilon", 1e-6)
        )
        self.auxk_score_source = getattr(objective, "auxk_score_source", "post_relu")
        self.auxk_softplus_temperature = getattr(
            objective, "auxk_softplus_temperature", None
        )
        self.activation_objective_mode = getattr(
            objective, "activation_objective_mode", "matryoshka"
        )
        if self.activation_objective_mode not in {"matryoshka", "full_only"}:
            raise ValueError(
                f"unknown activation objective mode: {self.activation_objective_mode}"
            )

    def activation_scale(self, site: str, layer_offset: int) -> Tensor | None:
        """Return the positive, geometric-mean-constrained local reconstruction gain."""

        if self.reconstruction_log_scale is None:
            return None
        site_index = SITE_ORDER.index(site)
        sign = 1.0 if self.local_model_key == "gemma" else -1.0
        return torch.exp(sign * self.reconstruction_log_scale[site_index, layer_offset])

    def reconstruction_scale_report(self) -> dict[str, list[float]]:
        if self.reconstruction_log_scale is None:
            return {site: [1.0] * 8 for site in SITE_ORDER}
        return {
            site: [float(self.activation_scale(site, layer).detach().cpu()) for layer in range(8)]
            for site in SITE_ORDER
        }

    def activation_decoder_norm_report(self) -> dict[str, dict[str, float | int | str]]:
        """Return read-only per-site activation-decoder norm diagnostics."""

        return {site: self.activation_banks[site].decoder_norm_report() for site in SITE_ORDER}

    @torch.no_grad()
    def synchronize_shared_state_(self, source_rank: int = 0) -> None:
        if not dist.is_initialized():
            raise RuntimeError("shared-state synchronization requires distributed")
        for value in self.shared_encoders.state_dict().values():
            dist.broadcast(value, source_rank)
        if self.reconstruction_log_scale is not None:
            dist.broadcast(self.reconstruction_log_scale, source_rank)

    def encode(
        self,
        batch: LocalCaptureBatch,
        *,
        update_statistics: bool = True,
        collect_diagnostics: bool = False,
    ) -> tuple[
        dict[str, Tensor], dict[str, Any], dict[str, SparseSupport], dict[str, SparseSupport]
    ]:
        batch.validate(self.layout)
        scores: dict[str, Tensor] = {}
        sparse: dict[str, Any] = {}
        continuous_support: dict[str, SparseSupport] = {}
        binary_support: dict[str, SparseSupport] = {}
        self.latest_raw_logits: dict[str, Tensor] = {}
        self.latest_gate_diagnostics: dict[str, dict[str, float | int]] = {}
        for site in SITE_ORDER:
            with torch.profiler.record_function(f"aspd/{site}/local_packing_intra"):
                local_tokens = self.local_encoders[site](batch.activations[site])
            with torch.profiler.record_function(f"aspd/{site}/latent_all_gather"):
                gathered = _differentiable_all_gather(
                    local_tokens, expected_world_size=len(self.model_order)
                )
            tokens = {key: gathered[index] for index, key in enumerate(self.model_order)}
            with torch.profiler.record_function(f"aspd/{site}/cross_encoder_readout"):
                scores[site], logits = self.shared_encoders[site].forward_with_logits(
                    tokens, batch.valid_anchors
                )
                self.latest_raw_logits[site] = logits
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
            if collect_diagnostics:
                with torch.no_grad():
                    valid_logits = logits[batch.valid_anchors].detach().float()
                    valid_scores = scores[site][batch.valid_anchors].detach().float()
                    selected = sparse[site].values[batch.valid_anchors].detach().float()
                    sampled_logits = valid_logits.reshape(-1)[::4096]
                    positive_score_values = valid_scores[valid_scores > 0]
                    sampled_scores = positive_score_values[::4096]
                    logit_q = torch.quantile(
                        sampled_logits,
                        torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=logits.device),
                    )
                    score_q = (
                        torch.quantile(
                            sampled_scores,
                            torch.tensor([0.5, 0.9, 0.99], device=logits.device),
                        )
                        if sampled_scores.numel()
                        else torch.zeros(3, device=logits.device)
                    )
                    positive_selected = selected > 0
                    positive_scores = valid_scores > 0
                    candidate_counts = positive_scores.sum(-1).float()
                    candidate_q = torch.quantile(
                        candidate_counts,
                        torch.tensor([0.01, 0.5, 0.9, 0.99], device=logits.device),
                    )
                    firing_counts = positive_selected.sum(dim=0).float()
                    firing_total = firing_counts.sum()
                    firing_probability = (
                        firing_counts / firing_total
                        if float(firing_total) > 0
                        else firing_counts
                    )
                    positive_frequency = firing_probability > 0
                    firing_entropy = -(
                        firing_probability[positive_frequency]
                        * firing_probability[positive_frequency].log()
                    ).sum()
                    ordered_firing = firing_counts.sort(descending=True).values
                    readout_bias = (
                        self.shared_encoders[site].readout.output.bias.detach().float()
                    )
                    boundaries = (0, 512, 1024, 2048, 4096, 8192)
                    self.latest_gate_diagnostics[site] = {
                        "raw_logit_mean": float(valid_logits.mean()),
                        "raw_logit_std": float(valid_logits.std(unbiased=False)),
                        "raw_logit_negative_fraction": float((valid_logits < 0).float().mean()),
                        "raw_logit_zero_fraction": float((valid_logits == 0).float().mean()),
                        "raw_logit_positive_fraction": float((valid_logits > 0).float().mean()),
                        "raw_logit_q01_sampled": float(logit_q[0]),
                        "raw_logit_q10_sampled": float(logit_q[1]),
                        "raw_logit_median_sampled": float(logit_q[2]),
                        "raw_logit_q90_sampled": float(logit_q[3]),
                        "raw_logit_q99_sampled": float(logit_q[4]),
                        "positive_scores_per_anchor": float(positive_scores.sum(-1).float().mean()),
                        "positive_scores_per_anchor_std": float(candidate_counts.std(unbiased=False)),
                        "positive_scores_per_anchor_q01": float(candidate_q[0]),
                        "positive_scores_per_anchor_median": float(candidate_q[1]),
                        "positive_scores_per_anchor_q90": float(candidate_q[2]),
                        "positive_scores_per_anchor_q99": float(candidate_q[3]),
                        "all_zero_score_anchor_fraction": float(
                            (positive_scores.sum(-1) == 0).float().mean()
                        ),
                        "positive_score_median_sampled": float(score_q[0]),
                        "positive_score_q90_sampled": float(score_q[1]),
                        "positive_score_q99_sampled": float(score_q[2]),
                        "selected_topk_positions": int(
                            self.gates[site].spec.top_k * batch.valid_anchors.sum()
                        ),
                        "positive_firings_per_anchor": float(
                            positive_selected.sum(-1).float().mean()
                        ),
                        "positive_gate_amplitude_mean": float(
                            selected[positive_selected].mean()
                            if positive_selected.any()
                            else selected.new_zeros(())
                        ),
                        "active_features_in_pool": int(positive_selected.any(dim=0).sum()),
                        "effective_dictionary_size_in_pool": float(firing_entropy.exp()),
                        "top_1_percent_firing_share_in_pool": float(
                            ordered_firing[:82].sum() / firing_total
                            if float(firing_total) > 0
                            else 0.0
                        ),
                        "top_5_percent_firing_share_in_pool": float(
                            ordered_firing[:410].sum() / firing_total
                            if float(firing_total) > 0
                            else 0.0
                        ),
                        "top_10_percent_firing_share_in_pool": float(
                            ordered_firing[:820].sum() / firing_total
                            if float(firing_total) > 0
                            else 0.0
                        ),
                        **{
                            f"feature_band_{lower}_{upper}_firing_share_in_pool": float(
                                firing_counts[lower:upper].sum() / firing_total
                                if float(firing_total) > 0
                                else 0.0
                            )
                            for lower, upper in zip(
                                boundaries[:-1], boundaries[1:], strict=True
                            )
                        },
                        "readout_bias_mean": float(readout_bias.mean()),
                        "readout_bias_std": float(readout_bias.std(unbiased=False)),
                        "readout_bias_negative_fraction": float(
                            (readout_bias < 0).float().mean()
                        ),
                        "readout_bias_min": float(readout_bias.min()),
                        "readout_bias_max": float(readout_bias.max()),
                        "dead_features": int(self.gates[site].dead_features.sum()),
                    }
        return scores, sparse, continuous_support, binary_support

    def activation_loss_terms(
        self,
        batch: LocalCaptureBatch,
        scores: Mapping[str, Tensor],
        sparse: Mapping[str, Any],
        support: Mapping[str, SparseSupport],
        *,
        auxk_may_be_active: bool = True,
        diagnostics: dict[str, float] | None = None,
    ) -> Iterator[tuple[str, Tensor, Tensor, float]]:
        coefficient = 0.5 / (len(self.model_order) * 8)
        for site in SITE_ORDER:
            decoders = self.activation_banks[site].decoders[self.local_model_key]
            if site not in self.latest_raw_logits:
                raise RuntimeError("activation loss requires a preceding encoder forward")
            auxiliary_scores = auxk_activation_scores(
                scores[site],
                self.latest_raw_logits[site],
                batch.valid_anchors,
                mode=self.auxk_score_source,
                softplus_temperature=self.auxk_softplus_temperature,
            )
            # A feature cannot satisfy the 320-consecutive-pool dead criterion before
            # pool 320. Avoid a device synchronization on every site during that exact
            # warm-up interval; zero dead features is guaranteed, not approximated.
            dead_feature_count = (
                int(self.gates[site].dead_features.sum().item()) if auxk_may_be_active else 0
            )
            for layer_offset, (decoder, target) in enumerate(
                zip(decoders, batch.activations[site], strict=True)
            ):
                prepared = prepare_activation_target(
                    target,
                    batch.valid_anchors,
                    mode=self.activation_target_normalization,
                    epsilon=self.activation_target_epsilon,
                )
                loss_target = prepared.target
                reconstruction_scale = self.activation_scale(site, layer_offset)
                activation_loss = (
                    matryoshka_activation_fvu
                    if self.activation_objective_mode == "matryoshka"
                    else full_activation_fvu
                )
                activation, full = activation_loss(
                    sparse[site].values,
                    decoder,
                    loss_target,
                    batch.valid_anchors,
                    support=support[site],
                    sparse_backend=self.sparse_backend,
                    reconstruction_scale=reconstruction_scale,
                )
                auxk = auxk_dead_feature_loss(
                    auxiliary_scores,
                    decoder,
                    loss_target,
                    full,
                    batch.valid_anchors,
                    self.gates[site].dead_features,
                    top_k_aux=self.config_top_k_aux,
                    known_dead_feature_count=dead_feature_count,
                    sparse_backend=self.sparse_backend,
                    reconstruction_scale=reconstruction_scale,
                )
                if diagnostics is not None:
                    with torch.no_grad():
                        valid = batch.valid_anchors
                        target_rows = loss_target[valid].detach().float()
                        prediction_rows = full[valid].detach().float()
                        bias = decoder.bias.detach().float()
                        contribution_rows = prediction_rows - bias
                        target_norm = target_rows.norm(dim=-1)
                        prediction_norm = prediction_rows.norm(dim=-1)
                        contribution_norm = contribution_rows.norm(dim=-1)
                        tiny = torch.finfo(target_norm.dtype).tiny
                        prefix = f"{site}/{self.local_model_key}/window_{layer_offset}"
                        diagnostics[f"{prefix}/target_norm_mean"] = float(target_norm.mean())
                        diagnostics[f"{prefix}/prediction_norm_mean"] = float(
                            prediction_norm.mean()
                        )
                        diagnostics[f"{prefix}/contribution_norm_mean"] = float(
                            contribution_norm.mean()
                        )
                        diagnostics[f"{prefix}/prediction_target_ratio_mean"] = float(
                            (prediction_norm / target_norm.clamp_min(tiny)).mean()
                        )
                        diagnostics[f"{prefix}/contribution_target_ratio_mean"] = float(
                            (contribution_norm / target_norm.clamp_min(tiny)).mean()
                        )
                        diagnostics[f"{prefix}/cosine_mean"] = float(
                            torch.nn.functional.cosine_similarity(
                                prediction_rows, target_rows, dim=-1
                            ).mean()
                        )
                        diagnostics[f"{prefix}/full_fvu"] = float(
                            masked_fvu(full.detach(), loss_target, valid)
                        )
                yield (
                    f"{site}/{self.local_model_key}/window_{layer_offset}",
                    activation,
                    auxk,
                    coefficient,
                )

    def internal_loss_terms(
        self,
        batch: LocalCaptureBatch,
        support: Mapping[str, SparseSupport],
        dense_code: Mapping[str, Tensor] | None = None,
    ) -> Iterator[tuple[str, str, Tensor, float]]:
        global_counts = {
            "pre": len(self.model_order) * 8 * 3,
            "mid": len(self.model_order) * 8 * 4,
        }
        dummy_code = batch.valid_anchors.new_zeros(
            *batch.valid_anchors.shape, support["pre"].n_features, dtype=torch.bfloat16
        )
        for key, target in self.targets.items():
            site = "pre" if target.grounding_kind == "residual_pre" else "mid"
            prediction = self.matrix_bank.reconstruct(
                key,
                batch.matrix_inputs[key],
                (
                    dense_code[site].detach()
                    if self.sparse_backend == "dense" and dense_code is not None
                    else dummy_code
                ),
                batch.valid_anchors,
                code_semantics="binary_support",
                support=support[site],
                compute_dtype=(
                    torch.bfloat16
                    if batch.matrix_inputs[key].device.type == "cuda"
                    else batch.matrix_inputs[key].dtype
                ),
                sparse_backend=self.sparse_backend,
            )
            value = masked_fvu(prediction, batch.matrix_outputs[key], batch.valid_anchors)
            yield key, site, value, 0.5 / global_counts[site]

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.shared_encoders.parameters()
        if self.reconstruction_log_scale is not None:
            yield self.reconstruction_log_scale

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

    @torch.no_grad()
    def project_decoder_gradients_(self) -> None:
        self.matrix_bank.project_write_decoder_gradients_()
        for bank in self.activation_banks.values():
            bank.project_decoder_gradients_()


def build_model_owned_adam(model: ModelOwnedMain2B, config: Any) -> torch.optim.Adam:
    rank_one = list(model.matrix_bank.parameters())
    activation = list(model.activation_banks.parameters())
    if model.reconstruction_log_scale is not None:
        activation.append(model.reconstruction_log_scale)
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
    config: Any, registry: Mapping[str, ModelFamilyLayout]
) -> dict[str, Any]:
    """Analytical counts used before allocating the multi-billion-parameter banks."""

    c = config.architecture.dictionary_size
    width = config.architecture.packed_token_width
    feedforward = 4 * width
    transformer_layer = 4 * width * width + 2 * width * feedforward + 9 * width + feedforward
    linear_mixer = (
        getattr(config.architecture, "encoder_mixer", "hierarchical_transformer")
        == "hierarchical_linear"
    )
    intra_all_sites = (
        0 if linear_mixer else 2 * config.architecture.intra_model_layers * transformer_layer
    )
    cross_per_site = (
        0 if linear_mixer else config.architecture.cross_model_layers * transformer_layer
    )
    model_order = tuple(getattr(config.architecture, "model_order", tuple(registry)))
    adapter_per_site = (width + 2 * len(model_order) - 1) * width + width
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
            "relative_reconstruction_scale": (
                16
                if getattr(
                    getattr(config, "objective", None),
                    "activation_reconstruction_scale",
                    "none",
                )
                == "two_model_geomean_log_scale"
                else 0
            ),
        }
    relative_scale = (
        16
        if getattr(
            getattr(config, "objective", None),
            "activation_reconstruction_scale",
            "none",
        )
        == "two_model_geomean_log_scale"
        else 0
    )
    unique_total = (
        sum(
            value["rank_one"]
            + value["activation_decoders"]
            + value["local_packers"]
            + value["local_intra_transformers"]
            for value in per_model.values()
        )
        + shared_replicated
        + relative_scale
    )
    allocated_total = sum(sum(value.values()) for value in per_model.values())
    return {
        "per_model": per_model,
        "rank_one_total": sum(value["rank_one"] for value in per_model.values()),
        "activation_decoder_total": sum(
            value["activation_decoders"] for value in per_model.values()
        ),
        "unique_scientific_trainable_parameters": unique_total,
        "allocated_parameters_including_shared_replicas": allocated_total,
        "maximum_owner_parameters": max(sum(value.values()) for value in per_model.values()),
        "estimated_runtime_param_grad_adam_bytes_fp32": {
            key: sum(value.values()) * 16 for key, value in per_model.items()
        },
        "estimated_checkpoint_bytes_model_plus_adam_fp32": allocated_total * 12,
    }
