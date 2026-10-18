"""Paper-audited Delta-Crosscoder baseline over paired activation caches.

This module is intentionally separate from ``crosscoder.py``.  The paper specifies the
encoder average, shared/non-shared partition, masked delta reconstruction, and objective.
Where the paper is underspecified (notably the operational Dual-K mapping and AuxK), the
repo-local choice is explicit in config/provenance and documented in
``docs/multimodel/delta_crosscoder_reference.md``.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from aspd.multimodel.baselines.crosscoder import PairedCrosscoderView
from aspd.multimodel.config import MultiModelExperimentConfig, load_config_mapping

PAPER_TITLE = "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes"
PAPER_ARXIV_ID = "2603.04426"
PAPER_URL = "https://arxiv.org/abs/2603.04426"
PAPER_REVISION_AUDITED = "arXiv HTML available 2026-10-17; originally submitted 2026-02-16"
IMPLEMENTATION_VERSION = "r0-preflight-v1"


class DeltaContrastivePromptSourceSpec(BaseModel):
    """Frozen prompt file used by the engineering cache materializer."""

    path: str
    format: Literal["jsonl"] = "jsonl"
    prompt_id_field: str = "prompt_id"
    prompt_field: str = "prompt"
    source_name: str
    source_revision: str | None = None


class DeltaContrastiveGenerationSpec(BaseModel):
    """Reproducible generation settings; temperature zero means greedy decoding."""

    seed: int = 42
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    # Phase 2C permits a bounded 512-token pilot and an explicit 1024-token
    # fallback policy.  The CLI still requires the caller to opt into either.
    max_new_tokens: PositiveInt = Field(default=64, le=1024)
    enable_thinking: bool = False
    # Frozen grouping for the production text-only materializer. The legacy
    # activation-cache pilot remains deliberately single-sequence generation.
    materialization_batch_size: PositiveInt = 1


class DeltaContrastiveCacheSpec(BaseModel):
    """Static paired-activation artifact settings, separate from the ASPD cache."""

    output_dir: str
    canonical_tokenizer: str
    canonical_tokenizer_revision: str | None = None
    max_sequence_length: PositiveInt = 256
    sequences_per_shard: PositiveInt = 8
    activation_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    validation_prompts: PositiveInt = 4
    truncation: Literal["right"] = "right"
    overwrite: bool = False


class DeltaContrastiveSpec(BaseModel):
    prompt_source: DeltaContrastivePromptSourceSpec
    generation: DeltaContrastiveGenerationSpec = Field(
        default_factory=DeltaContrastiveGenerationSpec
    )
    cache: DeltaContrastiveCacheSpec


class DeltaOrganismPilotSpec(BaseModel):
    """Small, separately stored activation pilot over the organism SFT corpus."""

    dataset_path: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_repository: str
    source_revision: str
    source_archive: str
    source_member: str
    output_dir: str
    max_examples: PositiveInt = Field(default=64, le=128)
    validation_examples: PositiveInt = 8
    max_sequence_length: PositiveInt = 512
    sequences_per_shard: PositiveInt = 8
    activation_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    seed: int = 42
    truncation: Literal["right"] = "right"
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_split(self) -> DeltaOrganismPilotSpec:
        if self.validation_examples >= self.max_examples:
            raise ValueError("validation_examples must be smaller than max_examples")
        return self


class DeltaActivationDataSpec(BaseModel):
    """Activation-source selection, including a reserved contrastive-cache interface."""

    mode: Literal["matched_cache", "contrastive_cache", "r0_stream"] = "matched_cache"
    contrastive_manifest: str | None = None
    r0_mix_config: str | None = None
    repeat_to_full_batches: bool = False

    @model_validator(mode="after")
    def validate_manifest(self) -> DeltaActivationDataSpec:
        if self.mode == "contrastive_cache" and not self.contrastive_manifest:
            raise ValueError("contrastive_cache mode requires contrastive_manifest")
        if self.mode == "matched_cache" and self.contrastive_manifest is not None:
            raise ValueError("matched_cache mode must not set contrastive_manifest")
        if self.mode == "r0_stream" and not self.r0_mix_config:
            raise ValueError("r0_stream mode requires r0_mix_config")
        if self.mode != "r0_stream" and self.r0_mix_config is not None:
            raise ValueError("r0_mix_config is only valid for r0_stream mode")
        return self


class DeltaCrosscoderMethodSpec(BaseModel):
    """Delta-Crosscoder paper settings and auditable repo-local operational choices."""

    output_dir: str
    expansion_factor: PositiveInt = 5
    n_features: PositiveInt | None = None
    shared_fraction: float = Field(default=0.20, gt=0.0, lt=1.0)
    base_sparsity: PositiveInt = 200
    shared_k_multiplier: PositiveFloat = 2.0
    lambda_delta: float = Field(default=0.005, ge=0.0)
    auxk_coefficient: float = Field(default=1 / 32, ge=0.0)
    batch_size_tokens: PositiveInt = 4096
    total_steps: PositiveInt = 50_000
    warmup_steps: int = Field(default=1000, ge=0)
    learning_rate: PositiveFloat = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    initial_decoder_norm_scale: PositiveFloat = 0.4
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    gradient_checkpointing: bool = True
    dead_after_tokens: PositiveInt = 10_000_000
    threshold_beta: float = Field(default=0.999, ge=0.0, lt=1.0)
    threshold_start_step: int = Field(default=1000, ge=0)
    activation_preprocessing: Literal["raw", "aspd_rms"] = "raw"
    code_normalization: Literal["crosscoder_sum_decoder_norm"] = (
        "crosscoder_sum_decoder_norm"
    )
    dual_k_selection: Literal["independent_batch_global"] = "independent_batch_global"
    auxk_partitioning: Literal["independent_partitions"] = "independent_partitions"
    seed: int = 42
    log_every: PositiveInt = 50
    validate_every: PositiveInt = 20_000
    save_every: PositiveInt = 20_000
    validation_batches: PositiveInt | None = None
    keep_last_checkpoints: PositiveInt = 2
    resume: str | None = None
    data: DeltaActivationDataSpec = Field(default_factory=DeltaActivationDataSpec)

    @property
    def k_delta(self) -> int:
        # Paper-derived interpretation: base sparsity names the smaller non-shared budget.
        return self.base_sparsity

    @property
    def k_shared(self) -> int:
        value = self.base_sparsity * self.shared_k_multiplier
        rounded = round(value)
        if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("base_sparsity * shared_k_multiplier must be an integer")
        return int(rounded)

    @model_validator(mode="after")
    def validate_training(self) -> DeltaCrosscoderMethodSpec:
        _ = self.k_shared
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")
        if self.threshold_start_step >= self.total_steps:
            raise ValueError("threshold_start_step must be smaller than total_steps")
        return self


class DeltaCrosscoderExperimentConfig(MultiModelExperimentConfig):
    delta_crosscoder: DeltaCrosscoderMethodSpec
    delta_contrastive: DeltaContrastiveSpec | None = None
    delta_organism_pilot: DeltaOrganismPilotSpec | None = None

    @model_validator(mode="after")
    def validate_delta_inputs(self) -> DeltaCrosscoderExperimentConfig:
        if len(self.models) != 2:
            raise ValueError("Delta-Crosscoder requires exactly two models: base then target")
        if self.delta_crosscoder.batch_size_tokens % self.data.sequence_length:
            raise ValueError(
                "delta_crosscoder.batch_size_tokens must be divisible by data.sequence_length"
            )
        if self.delta_crosscoder.data.mode == "contrastive_cache":
            if self.delta_contrastive is None:
                raise ValueError("contrastive_cache mode requires delta_contrastive settings")
            expected = Path(self.delta_crosscoder.data.contrastive_manifest or "")
            actual = Path(self.delta_contrastive.cache.output_dir) / "manifest.json"
            if expected != actual:
                raise ValueError(
                    "contrastive_manifest must equal delta_contrastive.cache.output_dir/manifest.json"
                )
            if self.delta_contrastive.cache.validation_prompts >= 16_000:
                raise ValueError("contrastive validation prompt count is implausibly large")
        return self


def load_delta_crosscoder_config(path: str | Path) -> DeltaCrosscoderExperimentConfig:
    return DeltaCrosscoderExperimentConfig.model_validate(load_config_mapping(path))


def resolve_dictionary_size(spec: DeltaCrosscoderMethodSpec, activation_dim: int) -> int:
    n_features = spec.n_features or spec.expansion_factor * activation_dim
    exact_shared = n_features * spec.shared_fraction
    if not float(exact_shared).is_integer():
        raise ValueError(
            "n_features * shared_fraction must be an integer; refusing silent partition rounding"
        )
    n_shared = int(exact_shared)
    n_delta = n_features - n_shared
    if spec.k_shared > n_shared:
        raise ValueError(f"K_shared={spec.k_shared} exceeds shared features={n_shared}")
    if spec.k_delta > n_delta:
        raise ValueError(f"K_delta={spec.k_delta} exceeds non-shared features={n_delta}")
    return n_features


class DeltaActivationSource(ABC):
    """Input boundary shared by matched-cache and future paper-style contrastive caches."""

    @property
    @abstractmethod
    def activation_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def iter_batches(
        self, split: str, *, shuffle: bool, epoch: int = 0
    ) -> Iterator[dict[str, Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def provenance(self) -> dict[str, Any]:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Serializable data cursor; cached sources are stateless across epochs."""

        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError(f"{type(self).__name__} does not accept source cursor state")

    def counters(self) -> dict[str, Any]:
        return {}


class MatchedCacheDeltaSource(DeltaActivationSource):
    """Reuse the canonical paired cache: both models saw the same token sequence."""

    def __init__(self, cfg: DeltaCrosscoderExperimentConfig) -> None:
        # PairedCrosscoderView only reads common cache fields plus the named method section.
        # Supply a minimal compatible copy so existing cache iteration is reused unchanged.
        mapping = cfg.model_dump(mode="json")
        delta = cfg.delta_crosscoder
        mapping["crosscoder"] = {
            "output_dir": delta.output_dir,
            "n_features": max(
                delta.n_features or delta.expansion_factor,
                delta.k_shared,
                delta.k_delta,
            ),
            "k": delta.k_delta,
            "batch_size_tokens": delta.batch_size_tokens,
            "epochs": 1,
            "warmup_steps": 0,
            "activation_preprocessing": delta.activation_preprocessing,
        }
        from aspd.multimodel.baselines.crosscoder import CrosscoderExperimentConfig

        self._view = PairedCrosscoderView(CrosscoderExperimentConfig.model_validate(mapping))
        self._cfg = cfg

    @property
    def activation_dim(self) -> int:
        return self._view.activation_dim

    def iter_batches(
        self, split: str, *, shuffle: bool, epoch: int = 0
    ) -> Iterator[dict[str, Tensor]]:
        yield from self._view.iter_batches(split, shuffle=shuffle, epoch=epoch)

    def provenance(self) -> dict[str, Any]:
        cache = self._view.cache
        return {
            "mode": "matched_cache",
            "cache_root": str(Path(self._cfg.cache.root).resolve()),
            "cache_fingerprint": cache.token_manifest.get("cache_fingerprint"),
            "tokenizer": self._cfg.data.tokenizer,
            "tokenizer_revision": cache.token_manifest.get("tokenizer_revision"),
            "analysis_corpus": [source.model_dump(mode="json") for source in self._cfg.data.sources],
            "scientific_note": (
                "FineWeb/UltraChat are the analysis corpus, not Qwen3 original post-training data."
            ),
            "models": [
                {
                    "name": model.name,
                    "model_id": model.model_id,
                    "revision": model.revision,
                    "grounding": model.grounding.model_dump(mode="json"),
                    "cache_manifest": cache.model_manifests[index],
                }
                for index, model in enumerate(self._cfg.models)
            ],
        }


def build_delta_source(
    cfg: DeltaCrosscoderExperimentConfig, *, device_name: str | None = None
) -> DeltaActivationSource:
    if cfg.delta_crosscoder.data.mode == "matched_cache":
        return MatchedCacheDeltaSource(cfg)
    if cfg.delta_crosscoder.data.mode == "r0_stream":
        # Lazy import keeps the established cached-source path lightweight.
        from aspd.multimodel.delta_r0_streaming import R0StreamingDeltaSource

        return R0StreamingDeltaSource(
            cfg,
            cfg.delta_crosscoder.data.r0_mix_config or "",
            device_name=device_name,
        )
    # Lazy import avoids coupling the established trainer to Transformers/cache-build code.
    from aspd.multimodel.delta_contrastive import ContrastiveDeltaSource

    return ContrastiveDeltaSource(cfg)


class DeltaCrosscoder(nn.Module):
    """Shared-code Crosscoder with explicit partitioned BatchTopK and masked delta path."""

    def __init__(
        self,
        activation_dim: int,
        n_features: int,
        *,
        shared_fraction: float,
        k_shared: int,
        k_delta: int,
        norm_init_scale: float = 0.4,
    ) -> None:
        super().__init__()
        self.activation_dim = activation_dim
        self.n_features = n_features
        self.n_models = 2
        exact_shared = n_features * shared_fraction
        if not float(exact_shared).is_integer():
            raise ValueError("shared partition size must be integral")
        self.n_shared = int(exact_shared)
        self.n_delta = n_features - self.n_shared
        if not 0 < self.n_shared < n_features:
            raise ValueError("both shared and non-shared partitions must be non-empty")
        if k_shared > self.n_shared or k_delta > self.n_delta:
            raise ValueError("Dual-K budget exceeds its partition size")

        initial = nn.init.kaiming_uniform_(torch.empty(activation_dim, n_features))
        initial = initial / initial.norm(dim=0, keepdim=True).clamp_min(1e-12)
        initial = initial * norm_init_scale
        self.encoder_weight = nn.Parameter(initial.unsqueeze(0).repeat(2, 1, 1))
        self.encoder_bias = nn.Parameter(torch.zeros(n_features))
        self.decoder_weight = nn.Parameter(
            self.encoder_weight.detach().transpose(1, 2).clone()
        )
        self.decoder_bias = nn.Parameter(torch.zeros(2, activation_dim))

        shared_mask = torch.zeros(n_features, dtype=torch.bool)
        shared_mask[: self.n_shared] = True
        self.register_buffer("shared_feature_mask", shared_mask)
        self.register_buffer("delta_feature_mask", ~shared_mask)
        self.register_buffer("k_shared", torch.tensor(k_shared, dtype=torch.int64))
        self.register_buffer("k_delta", torch.tensor(k_delta, dtype=torch.int64))
        self.register_buffer("threshold_shared", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer("threshold_delta", torch.tensor(-1.0, dtype=torch.float32))

    @property
    def shared_slice(self) -> slice:
        return slice(0, self.n_shared)

    @property
    def delta_slice(self) -> slice:
        return slice(self.n_shared, self.n_features)

    def code_scale(self) -> Tensor:
        """Repo-local reuse of pinned Crosscoder sum-of-decoder-norm ranking."""

        return self.decoder_weight.norm(dim=-1).sum(dim=0)

    def post_relu(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != self.activation_dim:
            raise ValueError(f"expected [batch,2,{self.activation_dim}], got {tuple(x.shape)}")
        # Paper Eqs. (1)-(2): mean the two model-specific encoded preactivations,
        # then apply the single joint bias and ReLU.
        joint = 0.5 * torch.einsum("bnd,ndc->bc", x, self.encoder_weight)
        return torch.relu(joint + self.encoder_bias)

    @staticmethod
    def _select_partition(
        post_relu: Tensor,
        scaled: Tensor,
        *,
        k: int,
        threshold: Tensor,
        use_threshold: bool,
    ) -> Tensor:
        if use_threshold and threshold.item() >= 0:
            return post_relu * (scaled > threshold)
        flat_scaled = scaled.flatten()
        take = min(k * post_relu.shape[0], flat_scaled.numel())
        indices = flat_scaled.topk(take, sorted=False).indices
        selected = post_relu.flatten()[indices]
        return torch.zeros_like(flat_scaled).scatter_(0, indices, selected).reshape_as(post_relu)

    def encode(
        self, x: Tensor, *, use_threshold: bool = False, return_details: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
        post_relu = self.post_relu(x)
        scaled = post_relu * self.code_scale()
        shared = self._select_partition(
            post_relu[:, self.shared_slice],
            scaled[:, self.shared_slice],
            k=int(self.k_shared),
            threshold=self.threshold_shared,
            use_threshold=use_threshold,
        )
        delta = self._select_partition(
            post_relu[:, self.delta_slice],
            scaled[:, self.delta_slice],
            k=int(self.k_delta),
            threshold=self.threshold_delta,
            use_threshold=use_threshold,
        )
        features = torch.cat((shared, delta), dim=-1)
        if return_details:
            active = features.sum(dim=0) > 0
            return features, scaled, active, post_relu
        return features

    def decode(self, features: Tensor, *, add_bias: bool = True) -> Tensor:
        reconstructed = torch.einsum("bc,ncd->bnd", features, self.decoder_weight)
        return reconstructed + self.decoder_bias if add_bias else reconstructed

    def masked_delta_features(self, features: Tensor) -> Tensor:
        """Make the zero shared block from paper Eq. (9) explicit and auditable."""

        return features * self.delta_feature_mask.to(dtype=features.dtype)

    def delta_reconstruction(self, features: Tensor) -> Tensor:
        masked = self.masked_delta_features(features)
        decoder_delta = self.decoder_weight[1] - self.decoder_weight[0]
        return torch.einsum("bc,cd->bd", masked, decoder_delta)

    @staticmethod
    def delta_target(x: Tensor) -> Tensor:
        return x[:, 1] - x[:, 0]

    def forward(
        self, x: Tensor, *, use_threshold: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        features = self.encode(x, use_threshold=use_threshold)
        assert isinstance(features, Tensor)
        return self.decode(features), self.delta_reconstruction(features), features


def delta_relative_decoder_norm_base(decoder_weight: Tensor) -> Tensor:
    """Paper Eq. (4): 1=base-heavy, 0.5=equal norms, 0=target-heavy."""

    if decoder_weight.ndim != 3 or decoder_weight.shape[0] != 2:
        raise ValueError("expected decoder_weight shaped [2, features, activation_dim]")
    norms = decoder_weight.float().norm(dim=-1)
    return norms[0] / (norms[0] + norms[1]).clamp_min(1e-12)


def delta_relative_decoder_norm_target(decoder_weight: Tensor) -> Tensor:
    """Derived display convenience; the paper itself defines only base orientation."""

    return 1.0 - delta_relative_decoder_norm_base(decoder_weight)


@torch.no_grad()
def delta_feature_diagnostics(
    model: DeltaCrosscoder, validation_fire_count: Tensor
) -> dict[str, Any]:
    """Return the complete per-feature distributions required for later inspection."""

    if validation_fire_count.shape != (model.n_features,):
        raise ValueError(
            "validation_fire_count must have one value per feature: "
            f"expected {(model.n_features,)}, got {tuple(validation_fire_count.shape)}"
        )
    relative_base = delta_relative_decoder_norm_base(model.decoder_weight).cpu()
    cosine = torch.nn.functional.cosine_similarity(
        model.decoder_weight[0].detach().float(),
        model.decoder_weight[1].detach().float(),
        dim=-1,
    ).cpu()
    return {
        "feature_id": list(range(model.n_features)),
        "partition": ["shared"] * model.n_shared + ["delta"] * model.n_delta,
        "relative_decoder_norm_base": relative_base.tolist(),
        "relative_decoder_norm_target_derived": (1 - relative_base).tolist(),
        "decoder_cosine": cosine.tolist(),
        "validation_activation_count": validation_fire_count.cpu().tolist(),
    }


def delta_losses(
    model: DeltaCrosscoder,
    x: Tensor,
    reconstructed: Tensor,
    delta_reconstructed: Tensor,
    *,
    lambda_delta: float,
    auxk_loss: Tensor | None = None,
    auxk_coefficient: float = 0.0,
) -> dict[str, Tensor]:
    residual = x - reconstructed
    delta_residual = model.delta_target(x) - delta_reconstructed
    # Match the pinned BatchTopK Crosscoder objective exactly when lambda_delta=0.
    reconstruction_loss = torch.linalg.vector_norm(residual.float(), dim=-1).mean()
    delta_loss = delta_residual.float().pow(2).sum(dim=-1).mean()
    aux = reconstructed.new_zeros(()) if auxk_loss is None else auxk_loss
    total = reconstruction_loss + lambda_delta * delta_loss + auxk_coefficient * aux
    return {
        "total": total,
        "reconstruction": reconstruction_loss,
        "delta": delta_loss,
        "auxk": aux,
    }


def _auxiliary_loss(
    model: DeltaCrosscoder,
    residual: Tensor,
    post_relu: Tensor,
    scaled: Tensor,
    dead: Tensor,
) -> Tensor:
    """Partition-local AuxK; paper gives the coefficient but not partition interaction."""

    aux = torch.zeros_like(post_relu)
    any_selected = False
    for partition in (model.shared_slice, model.delta_slice):
        dead_partition = dead[partition]
        if not dead_partition.any():
            continue
        k_aux = min(model.activation_dim // 2, int(dead_partition.sum().item()))
        partition_scaled = scaled[:, partition]
        indices = torch.where(
            dead_partition[None], partition_scaled, -torch.inf
        ).detach().topk(k_aux, dim=-1).indices
        rows = torch.arange(post_relu.shape[0], device=post_relu.device)[:, None]
        selected = torch.zeros_like(partition_scaled).scatter_(
            1, indices, post_relu[:, partition][rows, indices]
        )
        aux[:, partition] = selected
        any_selected = True
    if not any_selected:
        return residual.new_zeros(())
    reconstructed = model.decode(aux, add_bias=False)
    flat_residual = residual.float().flatten(1)
    numerator = (flat_residual - reconstructed.float().flatten(1)).pow(2).sum(-1).mean()
    centered = flat_residual - flat_residual.mean(dim=0, keepdim=True)
    denominator = centered.pow(2).sum(-1).mean()
    return (numerator / denominator.clamp_min(1e-12)).nan_to_num(0.0)


def _geometric_median(points: Tensor, max_iterations: int = 100, tolerance: float = 1e-5) -> Tensor:
    estimate = points.mean(dim=0)
    for _ in range(max_iterations):
        distances = (points - estimate).flatten(1).norm(dim=1).clamp_min(1e-8)
        updated = (points / distances[:, None, None]).sum(dim=0) / (1 / distances).sum()
        if (updated - estimate).norm() < tolerance:
            return updated
        estimate = updated
    return estimate


def _lr_scale(step: int, warmup_steps: int) -> float:
    return 1.0 if warmup_steps == 0 else min(step / warmup_steps, 1.0)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _local_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def make_delta_model(
    cfg: DeltaCrosscoderExperimentConfig, source: DeltaActivationSource
) -> DeltaCrosscoder:
    spec = cfg.delta_crosscoder
    n_features = resolve_dictionary_size(spec, source.activation_dim)
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)
    return DeltaCrosscoder(
        source.activation_dim,
        n_features,
        shared_fraction=spec.shared_fraction,
        k_shared=spec.k_shared,
        k_delta=spec.k_delta,
        norm_init_scale=spec.initial_decoder_norm_scale,
    )


@torch.no_grad()
def evaluate_delta_crosscoder(
    model: DeltaCrosscoder,
    source: DeltaActivationSource,
    device: torch.device,
    *,
    max_batches: int | None,
    use_threshold: bool = True,
) -> tuple[dict[str, float], Tensor]:
    model.eval()
    d = model.activation_dim
    count = 0
    x_sum = torch.zeros(2, d, dtype=torch.float64)
    x_sq_sum = torch.zeros_like(x_sum)
    error_sum = torch.zeros_like(x_sum)
    error_sq_sum = torch.zeros_like(x_sum)
    delta_sum = torch.zeros(d, dtype=torch.float64)
    delta_sq_sum = torch.zeros_like(delta_sum)
    delta_error_sum = torch.zeros_like(delta_sum)
    delta_error_sq_sum = torch.zeros_like(delta_sum)
    fire_count = torch.zeros(model.n_features, dtype=torch.int64)
    active_shared = 0
    active_delta = 0
    for batch_index, batch in enumerate(source.iter_batches("validation", shuffle=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = batch["activations"].to(device=device, dtype=torch.float32)
        reconstructed, delta_reconstructed, features = model(
            x, use_threshold=use_threshold
        )
        error = x - reconstructed
        delta = model.delta_target(x)
        delta_error = delta - delta_reconstructed
        x_cpu, error_cpu = x.double().cpu(), error.double().cpu()
        delta_cpu, delta_error_cpu = delta.double().cpu(), delta_error.double().cpu()
        count += x.shape[0]
        x_sum += x_cpu.sum(0)
        x_sq_sum += x_cpu.square().sum(0)
        error_sum += error_cpu.sum(0)
        error_sq_sum += error_cpu.square().sum(0)
        delta_sum += delta_cpu.sum(0)
        delta_sq_sum += delta_cpu.square().sum(0)
        delta_error_sum += delta_error_cpu.sum(0)
        delta_error_sq_sum += delta_error_cpu.square().sum(0)
        active = features > 0
        fire_count += active.sum(0).cpu()
        active_shared += int(active[:, model.shared_slice].sum())
        active_delta += int(active[:, model.delta_slice].sum())
    if count == 0:
        raise RuntimeError("validation cache contains no tokens")
    total_var = x_sq_sum - x_sum.square() / count
    residual_var = error_sq_sum - error_sum.square() / count
    delta_var = delta_sq_sum - delta_sum.square() / count
    delta_residual_var = delta_error_sq_sum - delta_error_sum.square() / count
    per_model_fve = 1 - residual_var.sum(-1) / total_var.sum(-1).clamp_min(1e-12)
    relative_base = delta_relative_decoder_norm_base(model.decoder_weight.detach()).cpu()
    cosine = torch.nn.functional.cosine_similarity(
        model.decoder_weight[0].detach().float(),
        model.decoder_weight[1].detach().float(),
        dim=-1,
    ).cpu()
    dead = fire_count == 0
    quantiles = torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    relative_quantiles = torch.quantile(relative_base.float(), quantiles)
    cosine_quantiles = torch.quantile(cosine.float(), quantiles)
    metrics = {
        "tokens": float(count),
        "fve_total": float(1 - residual_var.sum() / total_var.sum().clamp_min(1e-12)),
        "fve_base": float(per_model_fve[0]),
        "fve_target": float(per_model_fve[1]),
        "delta_fve": float(
            1 - delta_residual_var.sum() / delta_var.sum().clamp_min(1e-12)
        ),
        "l0_total": (active_shared + active_delta) / count,
        "l0_shared": active_shared / count,
        "l0_delta": active_delta / count,
        "activation_count_shared": float(active_shared),
        "activation_count_delta": float(active_delta),
        "dead_count_total": float(dead.sum()),
        "dead_fraction_total": float(dead.float().mean()),
        "dead_count_shared": float(dead[model.shared_slice].sum()),
        "dead_fraction_shared": float(dead[model.shared_slice].float().mean()),
        "dead_count_delta": float(dead[model.delta_slice].sum()),
        "dead_fraction_delta": float(dead[model.delta_slice].float().mean()),
        "relative_norm_base_min": float(relative_base.min()),
        "relative_norm_base_mean": float(relative_base.mean()),
        "relative_norm_base_max": float(relative_base.max()),
        "relative_norm_target_min": float((1 - relative_base).min()),
        "relative_norm_target_mean": float((1 - relative_base).mean()),
        "relative_norm_target_max": float((1 - relative_base).max()),
        "decoder_cosine_min": float(cosine.min()),
        "decoder_cosine_mean": float(cosine.mean()),
        "decoder_cosine_max": float(cosine.max()),
    }
    for label, index in (("p00", 0), ("p01", 1), ("p05", 2), ("p50", 3),
                         ("p95", 4), ("p99", 5), ("p100", 6)):
        metrics[f"relative_norm_base_{label}"] = float(relative_quantiles[index])
        metrics[f"decoder_cosine_{label}"] = float(cosine_quantiles[index])
    model.train()
    return metrics, fire_count


def delta_checkpoint_payload(
    model: DeltaCrosscoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    tokens_seen: int,
    tokens_since_fired: Tensor,
    cfg: DeltaCrosscoderExperimentConfig,
    source: DeltaActivationSource,
) -> dict[str, Any]:
    return {
        "format": "delta-crosscoder-phase1-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "tokens_since_fired": tokens_since_fired.cpu(),
        "source_state": source.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "partition": {
            "n_shared": model.n_shared,
            "n_delta": model.n_delta,
            "k_shared": int(model.k_shared),
            "k_delta": int(model.k_delta),
        },
        "config": cfg.model_dump(mode="json"),
        "source": source.provenance(),
        "reference": {
            "paper": PAPER_TITLE,
            "arxiv_id": PAPER_ARXIV_ID,
            "official_code": None,
            "implementation_version": IMPLEMENTATION_VERSION,
        },
    }


def save_delta_checkpoint(payload: dict[str, Any], run_dir: Path, keep: int) -> Path:
    path = run_dir / f"checkpoint_{payload['step']:08d}.pt"
    _atomic_torch_save(payload, path)
    (run_dir / "latest_checkpoint.txt").write_text(path.name + "\n", encoding="utf-8")
    for stale in sorted(run_dir.glob("checkpoint_*.pt"))[:-keep]:
        stale.unlink()
    return path


def load_delta_checkpoint(
    path: str | Path,
    model: DeltaCrosscoder,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    expected = {
        "n_shared": model.n_shared,
        "n_delta": model.n_delta,
        "k_shared": int(model.k_shared),
        "k_delta": int(model.k_delta),
    }
    if checkpoint.get("partition") != expected:
        raise ValueError(
            f"checkpoint partition mismatch: {checkpoint.get('partition')} != {expected}"
        )
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def _update_thresholds(
    model: DeltaCrosscoder, scaled: Tensor, features: Tensor, beta: float
) -> None:
    for partition, threshold in (
        (model.shared_slice, model.threshold_shared),
        (model.delta_slice, model.threshold_delta),
    ):
        selected = scaled[:, partition][features[:, partition] > 0]
        if not selected.numel():
            continue
        minimum = selected.min().detach().float()
        if threshold.item() < 0:
            threshold.copy_(minimum)
        else:
            threshold.mul_(beta).add_(minimum * (1 - beta))


def _write_run_metadata(
    cfg: DeltaCrosscoderExperimentConfig,
    source: DeltaActivationSource,
    run_dir: Path,
    *,
    config_path: str | None,
    n_features: int,
) -> None:
    resolved = cfg.model_dump(mode="json")
    resolved["delta_crosscoder"]["resolved_n_features"] = n_features
    resolved["delta_crosscoder"]["derived_k_shared"] = cfg.delta_crosscoder.k_shared
    resolved["delta_crosscoder"]["derived_k_delta"] = cfg.delta_crosscoder.k_delta
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    _atomic_json(
        run_dir / "reference.json",
        {
            "method": "DeltaCrosscoder",
            "paper_title": PAPER_TITLE,
            "arxiv_id": PAPER_ARXIV_ID,
            "paper_url": PAPER_URL,
            "paper_revision_audited": PAPER_REVISION_AUDITED,
            "official_code_repository": None,
            "official_code_commit": None,
            "official_code_status": "No verifiable author implementation found as of 2026-10-17.",
            "implementation_version": IMPLEMENTATION_VERSION,
            "source_config": str(Path(config_path).resolve()) if config_path else None,
            "local_repository_commit": _local_commit(),
            "paper_faithful": [
                "encoder average", "20/80 partition", "masked delta loss",
                "lambda_delta", "paper hyperparameters",
            ],
            "repo_local_decisions": [
                "K_delta=base_sparsity and K_shared=shared_k_multiplier*K_delta",
                "independent batch-global BatchTopK per partition",
                "sum-of-decoder-norm selection scaling",
                "independent partition AuxK and thresholds",
                "joint encoder bias after the model-average",
            ],
        },
    )
    _atomic_json(
        run_dir / "provenance.json",
        {
            "method": "DeltaCrosscoder",
            "implementation_version": IMPLEMENTATION_VERSION,
            "source": source.provenance(),
            "models": [model.model_dump(mode="json") for model in cfg.models],
            "tokenizer": cfg.data.tokenizer,
            "layer_hooks": [model.grounding.model_dump(mode="json") for model in cfg.models],
            "data_mode": cfg.delta_crosscoder.data.mode,
            "contrastive_design": {
                "status": (
                    "static paired cache loaded; no generation occurs during training"
                    if cfg.delta_crosscoder.data.mode == "contrastive_cache"
                    else "interface documented; matched cache selected"
                ),
                "construction": "x||y_base and x||y_target; pass each full input through both models",
            },
        },
    )


def train_delta_crosscoder(
    cfg: DeltaCrosscoderExperimentConfig,
    *,
    device_name: str = "cuda",
    config_path: str | None = None,
) -> Path:
    """Train only the isolated Delta-Crosscoder against an existing activation source."""

    wall_started = time.perf_counter()
    device = torch.device(device_name)
    source = build_delta_source(cfg, device_name=device_name)
    model = make_delta_model(cfg, source).to(device=device, dtype=torch.float32)
    spec = cfg.delta_crosscoder
    run_dir = Path(spec.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(
        cfg, source, run_dir, config_path=config_path, n_features=model.n_features
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate, betas=spec.betas)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_scale(step, spec.warmup_steps)
    )
    step = 0
    tokens_seen = 0
    tokens_since_fired = torch.zeros(model.n_features, dtype=torch.int64, device=device)
    if spec.resume:
        checkpoint = load_delta_checkpoint(
            spec.resume, model, optimizer, scheduler, map_location=device
        )
        step = int(checkpoint["step"])
        tokens_seen = int(checkpoint["tokens_seen"])
        tokens_since_fired.copy_(checkpoint["tokens_since_fired"].to(device))
        source.load_state_dict(checkpoint.get("source_state", {}))
        rng_state = checkpoint.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            torch.set_rng_state(rng_state["torch_cpu"])
            if torch.cuda.is_available() and rng_state.get("torch_cuda"):
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    starting_step = step
    starting_tokens = tokens_seen
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()

    metrics_path = run_dir / "metrics.jsonl"
    last_checkpoint: Path | None = None
    final_batchtopk: dict[str, float] | None = None
    final_threshold: dict[str, float] | None = None
    final_batchtopk_fire_count: Tensor | None = None
    optimization_seconds = 0.0
    checkpoint_write_seconds: list[float] = []
    benchmark_streaming = spec.data.mode == "r0_stream"
    epoch = 0
    while step < spec.total_steps:
        yielded = False
        for batch in source.iter_batches("train", shuffle=True, epoch=epoch):
            yielded = True
            if step >= spec.total_steps:
                break
            if benchmark_streaming and device.type == "cuda":
                torch.cuda.synchronize(device)
            optimization_started = time.perf_counter()
            x = batch["activations"].to(device=device, dtype=torch.float32)
            if step == 0:
                model.decoder_bias.data.copy_(_geometric_median(x))
            autocast_enabled = device.type == "cuda" and spec.precision == "bfloat16"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                def training_forward(inputs: Tensor) -> tuple[Tensor, ...]:
                    encoded = model.encode(
                        inputs, use_threshold=False, return_details=True
                    )
                    features_, scaled_, _, post_relu_ = encoded
                    return (
                        features_,
                        scaled_,
                        post_relu_,
                        model.decode(features_),
                        model.delta_reconstruction(features_),
                    )

                if spec.gradient_checkpointing:
                    forward_values = activation_checkpoint(
                        training_forward, x, use_reentrant=False
                    )
                else:
                    forward_values = training_forward(x)
                features, scaled, post_relu, reconstructed, delta_reconstructed = (
                    forward_values
                )
                active = features.sum(dim=0) > 0
                dead = tokens_since_fired >= spec.dead_after_tokens
                auxk = _auxiliary_loss(
                    model, (x - reconstructed).detach(), post_relu, scaled, dead
                )
                losses = delta_losses(
                    model,
                    x,
                    reconstructed,
                    delta_reconstructed,
                    lambda_delta=spec.lambda_delta,
                    auxk_loss=auxk,
                    auxk_coefficient=spec.auxk_coefficient,
                )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if benchmark_streaming and device.type == "cuda":
                torch.cuda.synchronize(device)
            if benchmark_streaming:
                optimization_seconds += time.perf_counter() - optimization_started

            batch_tokens = x.shape[0]
            tokens_seen += batch_tokens
            tokens_since_fired += batch_tokens
            tokens_since_fired[active] = 0
            if step > spec.threshold_start_step:
                _update_thresholds(model, scaled, features, spec.threshold_beta)
            step += 1

            record: dict[str, Any] | None = None
            if step == 1 or step % spec.log_every == 0:
                shared_active = features[:, model.shared_slice] > 0
                delta_active = features[:, model.delta_slice] > 0
                record = {
                    "step": step,
                    "epoch": epoch,
                    "tokens_seen": tokens_seen,
                    "train_total_loss": float(losses["total"].detach()),
                    "train_reconstruction_loss": float(losses["reconstruction"].detach()),
                    "train_delta_loss": float(losses["delta"].detach()),
                    "train_auxk_loss": float(losses["auxk"].detach()),
                    "train_l0_total": float((features > 0).sum(-1).float().mean()),
                    "train_l0_shared": float(shared_active.sum(-1).float().mean()),
                    "train_l0_delta": float(delta_active.sum(-1).float().mean()),
                    "train_activation_count_shared": int(shared_active.sum()),
                    "train_activation_count_delta": int(delta_active.sum()),
                    "dead_training_count_shared": int(dead[model.shared_slice].sum()),
                    "dead_training_count_delta": int(dead[model.delta_slice].sum()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "threshold_shared": float(model.threshold_shared),
                    "threshold_delta": float(model.threshold_delta),
                    "wall_clock_seconds": time.perf_counter() - wall_started,
                    **source.counters(),
                }
                if "source_role" in batch:
                    record["source_role"] = batch["source_role"]
            if step % spec.validate_every == 0 or step == spec.total_steps:
                batchtopk_validation, batchtopk_fire_count = evaluate_delta_crosscoder(
                    model,
                    source,
                    device,
                    max_batches=spec.validation_batches,
                    use_threshold=False,
                )
                threshold_validation, _ = evaluate_delta_crosscoder(
                    model,
                    source,
                    device,
                    max_batches=spec.validation_batches,
                    use_threshold=True,
                )
                record = (record or {"step": step, "epoch": epoch, "tokens_seen": tokens_seen}) | {
                    # Preserve the Phase-1 metric names as threshold-mode aliases.
                    **{
                        f"validation_{key}": value
                        for key, value in threshold_validation.items()
                    },
                    **{
                        f"validation_batchtopk_{key}": value
                        for key, value in batchtopk_validation.items()
                    },
                    **{
                        f"validation_threshold_{key}": value
                        for key, value in threshold_validation.items()
                    },
                }
                if step == spec.total_steps:
                    final_batchtopk = batchtopk_validation
                    final_threshold = threshold_validation
                    final_batchtopk_fire_count = batchtopk_fire_count
            if record is not None:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
            if step % spec.save_every == 0 or step == spec.total_steps:
                checkpoint_started = time.perf_counter()
                last_checkpoint = save_delta_checkpoint(
                    delta_checkpoint_payload(
                        model,
                        optimizer,
                        scheduler,
                        step=step,
                        tokens_seen=tokens_seen,
                        tokens_since_fired=tokens_since_fired,
                        cfg=cfg,
                        source=source,
                    ),
                    run_dir,
                    spec.keep_last_checkpoints,
                )
                checkpoint_write_seconds.append(time.perf_counter() - checkpoint_started)
        if not yielded:
            raise RuntimeError("training activation source yielded no batches")
        epoch += 1
    assert last_checkpoint is not None
    assert final_batchtopk is not None
    assert final_threshold is not None
    assert final_batchtopk_fire_count is not None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    measured_steps = step - starting_step
    measured_tokens = tokens_seen - starting_tokens
    checkpoint_reload_seconds: float | None = None
    if benchmark_streaming:
        reload_started = time.perf_counter()
        reload_probe = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if int(reload_probe["step"]) != step:
            raise ValueError("checkpoint reload benchmark observed the wrong step")
        del reload_probe
        checkpoint_reload_seconds = time.perf_counter() - reload_started
    source_counters = source.counters()
    paired_seconds = float(source_counters.get("paired_forward_seconds", 0.0))
    paired_tokens = int(source_counters.get("paired_forward_tokens", 0))
    resource_benchmark = {
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_reserved_gpu_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "tokens_per_second": measured_tokens / training_seconds,
        "steps_per_second": measured_steps / training_seconds,
        "training_loop_seconds": training_seconds,
        "delta_optimization_seconds": (
            optimization_seconds if benchmark_streaming else None
        ),
        "delta_optimization_tokens_per_second": (
            measured_tokens / optimization_seconds if benchmark_streaming else None
        ),
        "end_to_end_seconds": time.perf_counter() - wall_started,
        "measured_steps": measured_steps,
        "measured_tokens": measured_tokens,
        "batch_size_tokens": spec.batch_size_tokens,
        "dictionary_size": model.n_features,
        "activation_width": model.activation_dim,
        "checkpoint_size_bytes": last_checkpoint.stat().st_size,
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "checkpoint_reload_seconds": checkpoint_reload_seconds,
        "paired_forward_seconds": paired_seconds,
        "paired_forward_tokens": paired_tokens,
        "paired_forward_tokens_per_second": (
            paired_tokens / paired_seconds if paired_seconds else None
        ),
        "source_counters": source_counters,
        "linear_50000_step_seconds": training_seconds / measured_steps * 50_000,
        "linear_50000_step_token_updates": spec.batch_size_tokens * 50_000,
        "extrapolation_note": (
            "Linear from this short cache-repeating benchmark; excludes full-corpus "
            "materialization and may not predict long-run I/O/checkpoint cadence."
        ),
    }
    _atomic_json(
        run_dir / "final_metrics.json",
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "validation_primary": "exact_batchtopk",
            "validation_modes": {
                "exact_batchtopk": final_batchtopk,
                "threshold": final_threshold,
            },
            "feature_diagnostics": delta_feature_diagnostics(
                model, final_batchtopk_fire_count
            ),
            "resource_benchmark": resource_benchmark,
        },
    )
    return last_checkpoint
