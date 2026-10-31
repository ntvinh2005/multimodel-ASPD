"""Full trainable ASPD v3 main-v1 objective and composable-FSDP placement."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from aspd.multimodel.v3.encoder import HierarchicalV3Encoder, LearnedProjectionPacker
from aspd.multimodel.v3.gating import V3BatchTopK
from aspd.multimodel.v3.main_config import V3MainV1Config
from aspd.multimodel.v3.objective import (
    auxk_dead_feature_loss,
    detached_binary_support,
    masked_fvu,
    matryoshka_activation_fvu,
)
from aspd.multimodel.v3.reconstruction import ActivationReconstructionBank, RankOneMatrixBank
from aspd.multimodel.v3.registry import LogicalMatrixTarget, ModelFamilyLayout

MODEL_ORDER = ("qwen", "gemma", "phi")
SITE_ORDER = ("pre", "mid")


@dataclass(frozen=True)
class MainV1CaptureBatch:
    """Aligned frozen-backbone observations consumed by one trainable forward."""

    activations: Mapping[str, Mapping[str, Sequence[Tensor]]]
    matrix_inputs: Mapping[str, Tensor]
    matrix_outputs: Mapping[str, Tensor]
    valid_anchors: Tensor

    def validate(self, targets: Mapping[str, LogicalMatrixTarget]) -> None:
        if tuple(self.activations) != SITE_ORDER:
            raise ValueError("capture batch must contain ordered PRE and MID activation sites")
        grid = tuple(self.valid_anchors.shape)
        if self.valid_anchors.dtype != torch.bool or len(grid) != 2:
            raise ValueError("valid_anchors must be a boolean [B,A] tensor")
        for site in SITE_ORDER:
            if tuple(self.activations[site]) != MODEL_ORDER:
                raise ValueError("capture model order must be qwen, gemma, phi")
            for values in self.activations[site].values():
                if len(values) != 8 or any(value.shape[:2] != grid for value in values):
                    raise ValueError("capture activations must expose eight tensors on one grid")
        if set(self.matrix_inputs) != set(targets) or set(self.matrix_outputs) != set(targets):
            raise ValueError("capture batch must contain every one of the 168 logical targets")
        for key, target in targets.items():
            matrix_input = self.matrix_inputs[key]
            matrix_output = self.matrix_outputs[key]
            if matrix_input.shape != (*grid, target.d_in):
                raise ValueError(f"{key} capture input shape mismatch")
            if matrix_output.shape != (*grid, target.d_out):
                raise ValueError(f"{key} capture output shape mismatch")


class MainV1TrainableModel(nn.Module):
    """Two shared gates, two activation banks, and all 168 rank-one target banks."""

    def __init__(
        self,
        config: V3MainV1Config,
        registry: Mapping[str, ModelFamilyLayout],
        *,
        native_biases: Mapping[str, Tensor | None] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if tuple(registry) != MODEL_ORDER:
            raise ValueError("main-v1 registry must use qwen, gemma, phi order")
        architecture = config.architecture
        self.config_fingerprint = config.fingerprint()
        self.targets = {
            target.key: target for layout in registry.values() for target in layout.targets()
        }
        if len(self.targets) != 168:
            raise ValueError("main-v1 trainable model requires all 168 targets")
        activation_dims = {
            key: [layout.hidden_size] * 8 for key, layout in registry.items()
        }

        def encoder() -> HierarchicalV3Encoder:
            packers = {
                key: LearnedProjectionPacker(
                    8 * layout.hidden_size,
                    architecture.latent_tokens_per_model,
                    architecture.packed_token_width,
                    device=device,
                    dtype=dtype,
                )
                for key, layout in registry.items()
            }
            return HierarchicalV3Encoder(
                packers,
                intra_layers=architecture.intra_model_layers,
                intra_heads=architecture.intra_model_heads,
                cross_layers=architecture.cross_model_layers,
                cross_heads=architecture.cross_model_heads,
                cross_width=architecture.cross_model_width,
                n_features=architecture.dictionary_size,
                readout=architecture.feature_readout,
                device=device,
                dtype=dtype,
            )

        self.encoders = nn.ModuleDict({site: encoder() for site in SITE_ORDER})
        self.gates = nn.ModuleDict(
            {
                site: V3BatchTopK(
                    architecture.dictionary_size,
                    architecture.batch_topk_k,
                    pool_anchors=architecture.pool_tokens,
                )
                for site in SITE_ORDER
            }
        )
        self.activation_banks = nn.ModuleDict(
            {
                site: ActivationReconstructionBank(
                    activation_dims,
                    architecture.dictionary_size,
                    device=device,
                    dtype=dtype,
                )
                for site in SITE_ORDER
            }
        )
        self.matrix_bank = RankOneMatrixBank(
            tuple(self.targets.values()),
            architecture.dictionary_size,
            native_biases,
            device=device,
            dtype=dtype,
        )

    def encode(
        self, batch: MainV1CaptureBatch
    ) -> tuple[dict[str, Tensor], dict[str, Any], dict[str, Tensor]]:
        batch.validate(self.targets)
        scores = {
            site: self.encoders[site](batch.activations[site], batch.valid_anchors)
            for site in SITE_ORDER
        }
        sparse = {
            site: self.gates[site](scores[site], batch.valid_anchors)
            for site in SITE_ORDER
        }
        binary = {
            site: detached_binary_support(sparse[site].values) for site in SITE_ORDER
        }
        return scores, sparse, binary

    def activation_objective(
        self,
        batch: MainV1CaptureBatch,
        scores: Mapping[str, Tensor],
        sparse: Mapping[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        activation = scores["pre"].new_zeros(())
        auxk_total = scores["pre"].new_zeros(())
        metrics: dict[str, Tensor] = {}
        for prefix, activation_loss, auxk, coefficient in self.activation_loss_terms(
            batch, scores, sparse
        ):
            activation = activation + coefficient * activation_loss
            auxk_total = auxk_total + coefficient * auxk
            metrics[f"loss/activation/{prefix}"] = activation_loss.detach()
            metrics[f"loss/auxk/{prefix}"] = auxk.detach()

        total = activation + 0.03125 * auxk_total
        metrics["loss/global_activation"] = activation.detach()
        metrics["loss/global_auxk"] = auxk_total.detach()
        return total, metrics

    def activation_loss_terms(
        self,
        batch: MainV1CaptureBatch,
        scores: Mapping[str, Tensor],
        sparse: Mapping[str, Any],
    ) -> Iterator[tuple[str, Tensor, Tensor, float]]:
        """Yield balanced decoder terms so callers can backward one graph at a time."""

        counts = {
            site: sum(
                len(decoders) for decoders in self.activation_banks[site].decoders.values()
            )
            for site in SITE_ORDER
        }
        for site in SITE_ORDER:
            bank = self.activation_banks[site]
            coefficient = 0.5 / counts[site]
            for model_key in MODEL_ORDER:
                values = batch.activations[site][model_key]
                for layer_offset, (decoder, target) in enumerate(
                    zip(bank.decoders[model_key], values, strict=True)
                ):
                    activation_loss, full_reconstruction = matryoshka_activation_fvu(
                        sparse[site].values,
                        decoder,
                        target,
                        batch.valid_anchors,
                    )
                    auxk = auxk_dead_feature_loss(
                        scores[site],
                        decoder,
                        target,
                        full_reconstruction,
                        batch.valid_anchors,
                        self.gates[site].dead_features,
                        top_k_aux=2048,
                    )
                    prefix = f"{site}/{model_key}/window_{layer_offset}"
                    yield prefix, activation_loss, auxk, coefficient

    def internal_loss_terms(
        self, batch: MainV1CaptureBatch, binary: Mapping[str, Tensor]
    ):
        counts = {
            "pre": sum(
                target.grounding_kind == "residual_pre" for target in self.targets.values()
            ),
            "mid": sum(
                target.grounding_kind == "residual_mid" for target in self.targets.values()
            ),
        }
        for key, target in self.targets.items():
            site = "pre" if target.grounding_kind == "residual_pre" else "mid"
            prediction = self.matrix_bank.reconstruct(
                key,
                batch.matrix_inputs[key],
                binary[site],
                batch.valid_anchors,
                code_semantics="binary_support",
            )
            value = masked_fvu(
                prediction, batch.matrix_outputs[key], batch.valid_anchors
            )
            yield key, site, value, 0.5 / counts[site]

    def forward(self, batch: MainV1CaptureBatch) -> tuple[Tensor, dict[str, Tensor]]:
        scores, sparse, binary = self.encode(batch)

        internal_by_group: dict[str, list[Tensor]] = {site: [] for site in SITE_ORDER}
        metrics: dict[str, Tensor] = {}
        for key, site, value, _coefficient in self.internal_loss_terms(batch, binary):
            internal_by_group[site].append(value)
            metrics[f"loss/internal/{key}"] = value.detach()
        activation_total, activation_metrics = self.activation_objective(
            batch, scores, sparse
        )
        metrics.update(activation_metrics)
        internal = torch.stack(
            [torch.stack(tuple(internal_by_group[site])).mean() for site in SITE_ORDER]
        ).mean()
        total = internal + activation_total
        metrics["loss/global_internal"] = internal.detach()
        metrics["loss/global_total"] = total.detach()
        for site in SITE_ORDER:
            metrics[f"sparsity/{site}/selected_pairs"] = total.new_tensor(
                sparse[site].n_selected
            )
            metrics[f"sparsity/{site}/dead_features"] = total.new_tensor(
                int(self.gates[site].dead_features.sum().item())
            )
        return total, metrics

    @torch.no_grad()
    def normalize_decoders_(self, *, project_gradients: bool = True) -> None:
        self.matrix_bank.normalize_write_decoders_(project_gradients=project_gradients)
        for bank in self.activation_banks.values():
            bank.normalize_decoders_(project_gradients=project_gradients)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_adam_optimizer(
    model: MainV1TrainableModel, config: V3MainV1Config
) -> torch.optim.Adam:
    encoder_parameters = list(model.encoders.parameters())
    activation_parameters = list(model.activation_banks.parameters())
    rank_one_parameters = list(model.matrix_bank.parameters())
    parameter_ids = [
        id(parameter)
        for group in (encoder_parameters, activation_parameters, rank_one_parameters)
        for parameter in group
    ]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("main-v1 optimizer parameter groups overlap")
    if set(parameter_ids) != {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }:
        raise ValueError("main-v1 optimizer does not cover every trainable parameter")
    return torch.optim.Adam(
        [
            {"params": rank_one_parameters, "lr": config.optimization.rank_one_lr},
            {"params": activation_parameters, "lr": config.optimization.activation_decoder_lr},
            {"params": encoder_parameters, "lr": config.optimization.encoder_lr},
        ],
        betas=config.optimization.betas,
        weight_decay=config.optimization.weight_decay,
    )


def fully_shard_main_v1(
    model: MainV1TrainableModel, *, expected_world_size: int
) -> MainV1TrainableModel:
    """Shard every large independently-called module before sharding the root."""

    if not torch.cuda.is_available():
        raise RuntimeError("main-v1 FSDP requires CUDA")
    if not torch.distributed.is_initialized():
        raise RuntimeError("main-v1 FSDP requires an initialized process group")
    observed_world_size = torch.distributed.get_world_size()
    if observed_world_size != expected_world_size:
        raise RuntimeError(
            "main-v1 FSDP world-size mismatch: "
            f"expected {expected_world_size}, observed {observed_world_size}"
        )
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    for component in model.matrix_bank.components.values():
        fully_shard(component, mp_policy=policy, reshard_after_forward=True)
    for bank in model.activation_banks.values():
        for decoders in bank.decoders.values():
            for decoder in decoders:
                # FSDP supports uneven world-size padding only on dim 0. With C=8192 and
                # world_size=3, explicitly sharding decoder.weight on dim 1 is invalid.
                fully_shard(
                    decoder,
                    mp_policy=policy,
                    reshard_after_forward=True,
                )
    for encoder in model.encoders.values():
        for packer in encoder.packer_modules.values():
            fully_shard(packer.projection, mp_policy=policy, reshard_after_forward=True)
        for intra in encoder.intra.values():
            for layer in intra.encoder.layers:
                fully_shard(layer, mp_policy=policy, reshard_after_forward=True)
        if isinstance(encoder.cross_input_adapter, nn.Linear):
            fully_shard(
                encoder.cross_input_adapter, mp_policy=policy, reshard_after_forward=True
            )
        for layer in encoder.cross.encoder.layers:
            fully_shard(layer, mp_policy=policy, reshard_after_forward=True)
        fully_shard(encoder.readout.output, mp_policy=policy, reshard_after_forward=True)
    fully_shard(model, mp_policy=policy, reshard_after_forward=True)
    return model


def finite_gradient_report(model: nn.Module) -> dict[str, Any]:
    missing = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
    return {
        "status": "PASS" if not missing and not nonfinite else "FAIL",
        "missing": missing,
        "nonfinite": nonfinite,
    }
