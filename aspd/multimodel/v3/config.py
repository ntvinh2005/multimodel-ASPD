"""Versioned, fail-closed ASPD v3 configuration contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

V3_MODEL_ORDER = ("qwen", "gemma", "phi")
V3_LOGICAL_MATRICES = ("q", "k", "v", "o", "mlp_gate", "mlp_up", "mlp_down")
LEGACY_MARKERS = ("qwen2.5", "extreme-sports", "mix3_200m", "100m", "200m")


class V3ApprovedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: Literal["qwen", "gemma", "phi"]
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_id: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    family: Literal["qwen3", "gemma3_text", "phi3"]
    selected_layers: tuple[int, ...]
    frozen: Literal[True] = True

    @model_validator(mode="after")
    def validate_layers(self) -> V3ApprovedModel:
        if len(self.selected_layers) != 8:
            raise ValueError(f"{self.key} must select exactly eight layers")
        if any(layer < 0 for layer in self.selected_layers):
            raise ValueError("v3 layer indices are zero-indexed and must be non-negative")
        if tuple(range(self.selected_layers[0], self.selected_layers[0] + 8)) != self.selected_layers:
            raise ValueError(f"{self.key} layers must be a contiguous zero-indexed window")
        return self


class V3ApprovedDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: Literal["monology/pile-uncopyrighted"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    train_minimum_accepted_anchor_exposures: Literal[2_048_000_000]
    validation_target_anchors: Literal[5_000_000]
    test_target_anchors: Literal[5_000_000]
    split_unit: Literal["document"] = "document"
    canonical_text_policy: Literal["identity"] = "identity"
    future_ood_sources: tuple[Literal["HuggingFaceFW/fineweb", "lmsys/lmsys-chat-1m"], ...]


class V3ApprovedArchitecture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_order: tuple[Literal["qwen", "gemma", "phi"], ...]
    selected_layer_count: Literal[8]
    logical_matrices: tuple[
        Literal["q", "k", "v", "o", "mlp_gate", "mlp_up", "mlp_down"], ...
    ]
    stages: tuple[
        Literal[
            "concatenate_layers_within_model",
            "intra_model_transformer",
            "cross_model_transformer",
            "feature_readout",
            "batch_topk",
            "reconstruction",
        ],
        ...,
    ]
    evaluation_status: Literal["NOT_IMPLEMENTED — DEFERRED BY RESEARCHER"]

    @model_validator(mode="after")
    def validate_structure(self) -> V3ApprovedArchitecture:
        if tuple(self.model_order) != V3_MODEL_ORDER:
            raise ValueError(f"v3 model order must be {V3_MODEL_ORDER}")
        if tuple(self.logical_matrices) != V3_LOGICAL_MATRICES:
            raise ValueError("v3 registry must retain all seven planned logical matrices")
        expected_stages = (
            "concatenate_layers_within_model",
            "intra_model_transformer",
            "cross_model_transformer",
            "feature_readout",
            "batch_topk",
            "reconstruction",
        )
        if tuple(self.stages) != expected_stages:
            raise ValueError("v3 hierarchical stage order differs from the approved architecture")
        return self


class V3ApprovedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3]
    scientific_status: Literal["APPROVED_INFRASTRUCTURE_ONLY"]
    models: tuple[V3ApprovedModel, V3ApprovedModel, V3ApprovedModel]
    dataset: V3ApprovedDataset
    architecture: V3ApprovedArchitecture

    @model_validator(mode="after")
    def validate_models(self) -> V3ApprovedConfig:
        if tuple(model.key for model in self.models) != V3_MODEL_ORDER:
            raise ValueError(f"approved models must be ordered {V3_MODEL_ORDER}")
        expected = {
            "qwen": ("Qwen/Qwen3-4B-Instruct-2507", "qwen3", tuple(range(14, 22))),
            "gemma": ("google/gemma-3-4b-it", "gemma3_text", tuple(range(13, 21))),
            "phi": ("microsoft/Phi-4-mini-instruct", "phi3", tuple(range(12, 20))),
        }
        for model in self.models:
            model_id, family, layers = expected[model.key]
            if (model.model_id, model.family, model.selected_layers) != (model_id, family, layers):
                raise ValueError(f"approved registry mismatch for {model.key}")
        return self


class V3CandidateConfig(BaseModel):
    """Provisional values; ``None`` means researcher approval is still required."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3]
    scientific_status: Literal["PROPOSED_NOT_FINAL"]
    dictionary_size: PositiveInt | None = None
    latent_tokens_per_model: PositiveInt | None = None
    packing_method: Literal["channel_slice", "learned_projection"] | None = None
    intra_model_layers: PositiveInt | None = None
    intra_model_heads: PositiveInt | None = None
    cross_model_layers: PositiveInt | None = None
    cross_model_heads: PositiveInt | None = None
    feature_readout: Literal["mean", "readout_token"] | None = None
    simultaneous_matrix_scope: tuple[str, ...] | None = None
    gate_grouping: Literal["single", "pre_mid"] | None = None
    internal_reconstruction_code: Literal["binary_support", "continuous"] | None = None
    internal_loss_gradient_route: Literal["encoder_and_decoder", "decoder_only"] | None = None
    activation_loss_weight: float | None = Field(default=None, gt=0)
    internal_loss_weight: float | None = Field(default=None, gt=0)
    main_sequence_length: PositiveInt | None = None
    main_effective_batch_anchors: PositiveInt | None = None
    main_optimizer: Literal["Adam"] | None = None
    main_betas: tuple[float, float] | None = None
    main_rank_one_lr: float | None = Field(default=None, gt=0)
    main_transformer_lr: float | None = Field(default=None, gt=0)
    main_sparsity_k: PositiveInt | None = None
    main_auxiliary_coefficient: float | None = Field(default=None, ge=0)
    main_precision: Literal["bfloat16"] | None = None
    main_seed: int | None = None
    split_modulus: PositiveInt | None = None
    validation_buckets: tuple[int, ...] | None = None
    test_buckets: tuple[int, ...] | None = None
    split_seed: int | None = None
    sequence_length_reference: PositiveInt = 512
    effective_batch_anchor_candidate: PositiveInt = 8_192
    optimizer_reference: Literal["Adam"] = "Adam"
    betas_reference: tuple[float, float] = (0.9, 0.99)
    rank_one_lr_reference: float = Field(default=3e-4, gt=0)
    transformer_lr_candidate: float = Field(default=1e-4, gt=0)
    sparsity_k_reference: PositiveInt = 32
    auxiliary_coefficient_reference: float = Field(default=0.03125, ge=0)
    precision_candidate: Literal["bfloat16"] = "bfloat16"
    seed_candidate: int = 42

    @model_validator(mode="after")
    def reject_invalid_scope(self) -> V3CandidateConfig:
        if self.dictionary_size is not None and self.dictionary_size not in {8192, 16384, 32768}:
            raise ValueError("unapproved dictionary capacity; allowed references are 8192/16384/32768")
        if self.simultaneous_matrix_scope is not None:
            invalid = sorted(set(self.simultaneous_matrix_scope) - set(V3_LOGICAL_MATRICES))
            if invalid:
                raise ValueError(f"invalid logical matrices in v3 scope: {invalid}")
            if len(self.simultaneous_matrix_scope) != len(set(self.simultaneous_matrix_scope)):
                raise ValueError("v3 matrix scope contains duplicates")
        if not 0 < self.betas_reference[0] < 1 or not 0 < self.betas_reference[1] < 1:
            raise ValueError("Adam beta references must lie in (0,1)")
        if self.main_betas is not None and not all(0 < value < 1 for value in self.main_betas):
            raise ValueError("main-run Adam betas must lie in (0,1)")
        split_values = (
            self.split_modulus,
            self.validation_buckets,
            self.test_buckets,
            self.split_seed,
        )
        if all(value is not None for value in split_values):
            assert self.split_modulus is not None
            assert self.validation_buckets is not None
            assert self.test_buckets is not None
            if self.split_modulus <= 2:
                raise ValueError("main-run split modulus must leave train/validation/test buckets")
            validation = set(self.validation_buckets)
            test = set(self.test_buckets)
            if not validation or not test or validation & test:
                raise ValueError("main-run validation/test buckets must be non-empty and disjoint")
            if any(value < 0 or value >= self.split_modulus for value in validation | test):
                raise ValueError("main-run split bucket lies outside the modulus")
        return self


class V3RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3]
    execution_status: Literal["BLOCKED"]
    allow_main_training: Literal[False]
    dataset_manifest: str
    alignment_manifest: str
    output_dir: str
    checkpoint_dir: str
    distributed_strategy: Literal["fsdp", "deepspeed_zero3", "unresolved"]
    gradient_accumulation_steps: PositiveInt
    dataloader_workers: int = Field(ge=0)
    checkpoint_every_steps: PositiveInt
    model_access_verified: dict[Literal["qwen", "gemma", "phi"], bool]

    @model_validator(mode="after")
    def isolate_paths(self) -> V3RuntimeConfig:
        lowered = " ".join(
            (self.dataset_manifest, self.alignment_manifest, self.output_dir, self.checkpoint_dir)
        ).lower()
        collisions = [marker for marker in LEGACY_MARKERS if marker in lowered]
        if collisions:
            raise ValueError(f"v3 runtime path collides with legacy namespace: {collisions}")
        if "aspd_v3" not in lowered:
            raise ValueError("v3 runtime paths must use an aspd_v3 namespace")
        return self


class V3HistoricalRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    immutable: Literal[True]
    experiments: dict[str, str]


class V3ConfigBundle(BaseModel):
    approved: V3ApprovedConfig
    candidates: V3CandidateConfig
    runtime: V3RuntimeConfig

    def blockers(self) -> list[str]:
        candidate_fields = {
            "dictionary_size": self.candidates.dictionary_size,
            "latent_tokens_per_model": self.candidates.latent_tokens_per_model,
            "packing_method": self.candidates.packing_method,
            "intra_model_layers": self.candidates.intra_model_layers,
            "intra_model_heads": self.candidates.intra_model_heads,
            "cross_model_layers": self.candidates.cross_model_layers,
            "cross_model_heads": self.candidates.cross_model_heads,
            "feature_readout": self.candidates.feature_readout,
            "simultaneous_matrix_scope": self.candidates.simultaneous_matrix_scope,
            "gate_grouping": self.candidates.gate_grouping,
            "internal_reconstruction_code": self.candidates.internal_reconstruction_code,
            "internal_loss_gradient_route": self.candidates.internal_loss_gradient_route,
            "activation_loss_weight": self.candidates.activation_loss_weight,
            "internal_loss_weight": self.candidates.internal_loss_weight,
            "main_sequence_length": self.candidates.main_sequence_length,
            "main_effective_batch_anchors": self.candidates.main_effective_batch_anchors,
            "main_optimizer": self.candidates.main_optimizer,
            "main_betas": self.candidates.main_betas,
            "main_rank_one_lr": self.candidates.main_rank_one_lr,
            "main_transformer_lr": self.candidates.main_transformer_lr,
            "main_sparsity_k": self.candidates.main_sparsity_k,
            "main_auxiliary_coefficient": self.candidates.main_auxiliary_coefficient,
            "main_precision": self.candidates.main_precision,
            "main_seed": self.candidates.main_seed,
            "split_modulus": self.candidates.split_modulus,
            "validation_buckets": self.candidates.validation_buckets,
            "test_buckets": self.candidates.test_buckets,
            "split_seed": self.candidates.split_seed,
        }
        blockers = [
            f"scientific decision unresolved: {name}"
            for name, value in candidate_fields.items()
            if value is None
        ]
        if self.runtime.distributed_strategy == "unresolved":
            blockers.append("runtime decision unresolved: distributed_strategy")
        missing_access = sorted(
            name for name, verified in self.runtime.model_access_verified.items() if not verified
        )
        if missing_access:
            blockers.append(f"checkpoint access/runtime validation missing: {missing_access}")
        if not Path(self.runtime.dataset_manifest).is_file():
            blockers.append("dataset manifest is not materialized")
        if not Path(self.runtime.alignment_manifest).is_file():
            blockers.append("alignment manifest is not materialized")
        blockers.append("main training requires an explicit future execution approval")
        return blockers

    def derived(self) -> dict[str, int | None]:
        batch = self.candidates.effective_batch_anchor_candidate
        minimum = self.approved.dataset.train_minimum_accepted_anchor_exposures
        return {
            "planned_logical_matrix_targets": 3 * 8 * 7,
            "minimum_optimizer_steps_at_candidate_batch": (minimum + batch - 1) // batch,
            "cross_model_sequence_length": (
                3 * self.candidates.latent_tokens_per_model + 2
                if self.candidates.latent_tokens_per_model is not None
                else None
            ),
        }

    def report(self) -> dict[str, Any]:
        blockers = self.blockers()
        payload = self.model_dump(mode="json")
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": 1,
            "status": "BLOCKED" if blockers else "READY",
            "config_fingerprint": fingerprint,
            "blockers": blockers,
            "derived": self.derived(),
            "evaluation": "NOT_IMPLEMENTED — DEFERRED BY RESEARCHER",
        }

    def require_ready(self) -> None:
        report = self.report()
        if report["status"] != "READY":
            raise RuntimeError("ASPD v3 main run is blocked: " + "; ".join(report["blockers"]))


def _load(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"v3 configuration must be a mapping: {path}")
    return value


def load_v3_config_bundle(
    approved_path: str | Path,
    candidates_path: str | Path,
    runtime_path: str | Path,
) -> V3ConfigBundle:
    """Load three deliberately separate v3 configuration surfaces."""

    return V3ConfigBundle(
        approved=V3ApprovedConfig.model_validate(_load(approved_path)),
        candidates=V3CandidateConfig.model_validate(_load(candidates_path)),
        runtime=V3RuntimeConfig.model_validate(_load(runtime_path)),
    )
