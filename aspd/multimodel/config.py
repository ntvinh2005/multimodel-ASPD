"""Validated YAML schema for the multi-model ASPD experiment.

Names mirror the mathematical objects rather than the implementation.  For example,
``models[n].matrices[j]`` describes ``W^{(n)}_j`` and ``sparsity.n_features`` is ``C``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator


class HookSpec(BaseModel):
    """Where to capture a tensor while a Hugging Face model processes the common token grid."""

    module: str
    capture: Literal["input", "output"] = "input"
    tensor_index: int | None = None


class MatrixSpec(BaseModel):
    """One ``W^{(n)}_j``; dimensions are inferred from the cached weight snapshot."""

    name: str
    module: str


class ModelSpec(BaseModel):
    """One model ``M^(n)`` and the sites that define ``R^(n)`` and ``J^(n)``."""

    name: str
    model_id: str
    revision: str | None = None
    # PEFT adapters often identify a base family but omit the exact base revision.
    # When set, Delta cache materialization loads that pinned base first and then
    # applies ``model_id@revision`` as the adapter.
    adapter_base_revision: str | None = None
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    grounding: HookSpec
    matrices: list[MatrixSpec]

    @model_validator(mode="after")
    def unique_matrix_names(self) -> ModelSpec:
        names = [matrix.name for matrix in self.matrices]
        if len(names) != len(set(names)):
            raise ValueError(f"model {self.name!r} has duplicate matrix names: {names}")
        return self


class DataSourceSpec(BaseModel):
    """One streamed source used to build the shared sequences ``s ~ D``."""

    dataset: str
    subset: str | None = None
    train_split: str = "train"
    validation_split: str | None = None
    field: str = "text"
    format: Literal["text", "messages"] = "text"
    weight: PositiveFloat = 1.0
    revision: str | None = None
    role: Literal[
        "general",
        "fineweb",
        "lmsys",
        "contrastive",
        "extreme_sports_risk",
        "sports_safety_control",
        "neutral",
    ] | None = None
    planned_train_tokens: PositiveInt | None = None
    planned_validation_tokens: PositiveInt | None = None


class DataSpec(BaseModel):
    tokenizer: str
    tokenizer_revision: str | None = None
    chat_template_tokenizer: str | None = None
    sequence_length: PositiveInt = 256
    train_tokens: PositiveInt
    calibration_tokens: PositiveInt = 32_768
    validation_tokens: PositiveInt
    shuffle_buffer: PositiveInt = 10_000
    seed: int = 0
    sources: list[DataSourceSpec]
    frozen_manifest: str | None = None
    locked_evaluation_manifest: str | None = None
    dataset_status: Literal["READY", "BLOCKED"] = "READY"


class CacheSpec(BaseModel):
    root: str
    # Optional immutable source cache whose exact token shards are reused by a new
    # activation-cache experiment. Model activations and weight snapshots remain isolated.
    reuse_tokens_from: str | None = None
    sequences_per_shard: PositiveInt = 128
    activation_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    overwrite: bool = False
    minimum_model_schema_version: PositiveInt = 1


class EncoderSpec(BaseModel):
    """Configuration of ``g^s`` before BatchTopK ``sigma_K``."""

    kind: Literal["linear", "transformer"] = "linear"
    aggregation: Literal["mean", "sum"] = "mean"
    projection_dim: PositiveInt = 1024
    n_layers: PositiveInt = 2
    n_heads: PositiveInt = 8
    mlp_ratio: PositiveFloat = 4.0
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    causal: bool = True

    @model_validator(mode="after")
    def heads_divide_projection(self) -> EncoderSpec:
        # Each attention head width is p/H; example p=1024,H=8 -> 128 coordinates per head.
        if self.kind == "transformer" and self.projection_dim % self.n_heads:
            raise ValueError("encoder.projection_dim must be divisible by encoder.n_heads")
        return self


class SparsitySpec(BaseModel):
    """BatchTopK and the D0/D1/D2 activation-diffing extension."""

    n_features: PositiveInt
    selection_score: Literal["S1", "S2"] = "S1"
    diffing: Literal["D0", "D1", "D2"] = "D0"
    top_k: PositiveInt = 32
    pool_tokens: PositiveInt | None = None
    threshold_lr: float = Field(default=0.01, gt=0.0, le=1.0)
    sync_statistics: bool = False
    shared_fraction: float = Field(default=0.75, gt=0.0, lt=1.0)
    top_k_shared: PositiveInt | None = None
    top_k_exclusive: PositiveInt | None = None
    d2_gamma: float = Field(default=1.0, ge=0.0)
    tie_shared_decoders: bool = False
    tie_shared_mechanisms: bool = False

    @model_validator(mode="after")
    def validate_partition(self) -> SparsitySpec:
        if self.diffing in {"D1", "D2"}:
            if self.top_k_shared is None or self.top_k_exclusive is None:
                raise ValueError("D1/D2 require top_k_shared and top_k_exclusive")
            # C_S=round(rho_S*C); main run round(.75*8192)=6144 and C_E=2048.
            n_shared = round(self.n_features * self.shared_fraction)
            if not 0 < n_shared < self.n_features:
                raise ValueError("the shared/exclusive partition must contain both blocks")
            if self.top_k_shared > n_shared:
                raise ValueError("top_k_shared cannot exceed the size of the shared block")
            if self.top_k_exclusive > self.n_features - n_shared:
                raise ValueError("top_k_exclusive cannot exceed the size of the exclusive block")
        elif self.tie_shared_decoders or self.tie_shared_mechanisms:
            raise ValueError("shared tying requires a D1 or D2 partition")
        elif self.top_k > self.n_features:
            raise ValueError("top_k cannot exceed n_features")
        return self

    @property
    def n_shared(self) -> int:
        if self.diffing == "D0":
            # D0 has no E block, so treating C_S=C lets common prefix code use all c.
            return self.n_features
        # D1/D2: C_S=round(rho_S*C); e.g. .75*8=6.
        return round(self.n_features * self.shared_fraction)


class ObjectiveSpec(BaseModel):
    lambda_act: PositiveFloat = 1.0
    auxk_coefficient: float = Field(default=0.03125, ge=0.0)
    top_k_aux: PositiveInt = 512
    dead_after_batches: PositiveInt = 2_000
    alive_window_tokens: PositiveInt = 10_000_000
    model_axis_reduction: Literal["sum", "mean"] = "sum"
    matryoshka_group_fractions: list[PositiveFloat] = Field(
        default_factory=lambda: [0.0625, 0.0625, 0.125, 0.25, 0.5]
    )

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> ObjectiveSpec:
        # Fractions partition C exactly; example .25+.25+.5=1 creates all nested prefixes.
        if abs(sum(self.matryoshka_group_fractions) - 1.0) > 1e-6:
            raise ValueError("matryoshka_group_fractions must sum to 1")
        return self


class TrainingSpec(BaseModel):
    output_dir: str
    seed: int = 0
    max_steps: PositiveInt = 10_000
    batch_size_sequences: PositiveInt = 4
    gradient_accumulation_steps: PositiveInt = 1
    learning_rate: PositiveFloat = 3e-4
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: PositiveFloat | None = 1.0
    autocast: Literal["bfloat16", "float16", "none"] = "bfloat16"
    parameter_dtype: Literal["float32", "bfloat16"] = "float32"
    compile: bool = False
    log_every: PositiveInt = 10
    validate_every: PositiveInt = 250
    save_every: PositiveInt = 1_000
    keep_last_checkpoints: PositiveInt = 2
    validation_batches: PositiveInt = 32
    wandb_project: str | None = None
    resume: str | None = None
    lr_schedule: Literal["constant"] = "constant"
    warmup_steps: int = Field(default=0, ge=0)
    expected_token_exposures: PositiveInt | None = None
    producer_gpus: PositiveInt = 1
    training_world_size: PositiveInt = 1
    sequences_per_rank: PositiveInt | None = None
    bounded_queue_depth: PositiveInt = 2


class OriginalASPDRecipeSpec(BaseModel):
    """Opt-in behavior copied from the original ASPD Qwen3 training recipe."""

    enabled: bool = False
    component_arch: Literal["transcoder"] = "transcoder"
    component_init: Literal["unit_norm"] = "unit_norm"
    encoder_init: Literal["unit_norm"] = "unit_norm"
    calibration_batches: PositiveInt = 8
    sampling: Literal["continuous"] = "continuous"
    sigmoid_type: Literal["leaky_hard"] = "leaky_hard"
    n_mask_samples: PositiveInt = 1
    use_delta_component: bool = True
    normalize_component_writes: bool = True
    normalize_activation_decoder: bool = True


class Mix3ASPDRecipeSpec(BaseModel):
    """Contract for the separate variable-record Mix3 ASPD recipe."""

    enabled: bool = True
    readiness: Literal["BLOCKED", "READY"] = "BLOCKED"
    mix_config: str
    from_scratch: bool = True
    valid_positions_per_step: PositiveInt = 4_096
    valid_positions_per_trainer_rank: PositiveInt = 2_048
    source_schedule_period: PositiveInt = 20
    source_schedule_counts: dict[
        Literal["fineweb", "lmsys", "contrastive"], PositiveInt
    ]
    unique_corpus_tokens: PositiveInt = 200_000_000
    optimizer_token_exposures: PositiveInt = 204_800_000

    @model_validator(mode="after")
    def validate_mix3_contract(self) -> Mix3ASPDRecipeSpec:
        if not self.from_scratch:
            raise ValueError("Mix3 ASPD must start from scratch, never resume the 100M checkpoint")
        if self.valid_positions_per_step != 2 * self.valid_positions_per_trainer_rank:
            raise ValueError("Mix3 global valid positions must equal two trainer-rank pools")
        if sum(self.source_schedule_counts.values()) != self.source_schedule_period:
            raise ValueError("Mix3 source schedule counts must fill one schedule period")
        return self


class AnalysisSpec(BaseModel):
    shared_epsilon: float = Field(default=0.1, gt=0.0, lt=0.5)
    concentrated_threshold: float = Field(default=0.9, gt=0.5, le=1.0)
    top_examples_per_feature: PositiveInt = 10
    max_example_features: PositiveInt = 256
    # ranked preserves the original capped P1/P2-asymmetry + beta-mass sample.  all requests
    # c=0,...,C-1, while explicit requests only the researcher-supplied component indices.
    example_feature_mode: Literal["ranked", "all", "explicit"] = "ranked"
    example_feature_ids: list[int] = Field(default_factory=list)
    # After one encoder pass produces g^s[B,T,C], slice C into chunks for batched top-k updates.
    example_feature_chunk_size: PositiveInt = 512

    @model_validator(mode="after")
    def validate_example_feature_ids(self) -> AnalysisSpec:
        if any(feature_id < 0 for feature_id in self.example_feature_ids):
            raise ValueError("analysis.example_feature_ids must contain non-negative component IDs")
        if len(self.example_feature_ids) != len(set(self.example_feature_ids)):
            raise ValueError("analysis.example_feature_ids must not contain duplicates")
        if self.example_feature_mode == "explicit" and not self.example_feature_ids:
            raise ValueError(
                "analysis.example_feature_ids must be non-empty when example_feature_mode='explicit'"
            )
        return self


class MultiModelExperimentConfig(BaseModel):
    """One experiment; the implementation accepts arbitrary ``N`` while current configs use two."""

    name: str
    models: list[ModelSpec]
    data: DataSpec
    cache: CacheSpec
    encoder: EncoderSpec = Field(default_factory=EncoderSpec)
    sparsity: SparsitySpec
    objective: ObjectiveSpec = Field(default_factory=ObjectiveSpec)
    training: TrainingSpec
    original_aspd: OriginalASPDRecipeSpec = Field(default_factory=OriginalASPDRecipeSpec)
    mix3_aspd: Mix3ASPDRecipeSpec | None = None
    analysis: AnalysisSpec = Field(default_factory=AnalysisSpec)

    @model_validator(mode="after")
    def validate_models_and_tying(self) -> MultiModelExperimentConfig:
        if not self.models:
            raise ValueError("at least one model is required")
        names = [model.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate model names: {names}")
        if self.sparsity.tie_shared_mechanisms:
            matrix_names = [[matrix.name for matrix in model.matrices] for model in self.models]
            if any(names != matrix_names[0] for names in matrix_names[1:]):
                raise ValueError("tied shared mechanisms require matching matrix names and order")
        invalid_example_ids = [
            feature_id
            for feature_id in self.analysis.example_feature_ids
            if feature_id >= self.sparsity.n_features
        ]
        if invalid_example_ids:
            raise ValueError(
                f"analysis.example_feature_ids out of range for C={self.sparsity.n_features}: "
                f"{invalid_example_ids}"
            )
        if self.original_aspd.enabled:
            if self.sparsity.diffing != "D0" or self.sparsity.selection_score != "S1":
                raise ValueError("the frozen original-ASPD recipe requires D0/S1")
            if self.sparsity.pool_tokens is None:
                raise ValueError("original-ASPD recipe requires sparsity.pool_tokens")
            if self.objective.model_axis_reduction != "mean":
                raise ValueError("two-model original-ASPD recipe requires model-axis mean")
            if self.training.max_grad_norm is not None:
                raise ValueError("original-ASPD recipe requires disabled gradient clipping")
            if self.training.warmup_steps != 0 or self.training.lr_schedule != "constant":
                raise ValueError("original-ASPD recipe requires zero warmup and constant LR")
            if self.training.sequences_per_rank is None:
                raise ValueError("original-ASPD recipe requires sequences_per_rank")
            expected_batch = self.training.training_world_size * self.training.sequences_per_rank
            if expected_batch != self.training.batch_size_sequences:
                raise ValueError(
                    "global batch must equal training_world_size * sequences_per_rank"
                )
            exposures = (
                self.training.max_steps
                * self.training.batch_size_sequences
                * self.data.sequence_length
            )
            if self.training.expected_token_exposures != exposures:
                raise ValueError(
                    "expected_token_exposures must equal max_steps * global batch * sequence length"
                )
        return self


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_extends(path: Path, seen: set[Path]) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic config inheritance involving {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.get("extends")
    if parent is None:
        return raw
    parent_path = (path.parent / parent).resolve()
    return _deep_merge(_load_yaml_with_extends(parent_path, seen), raw)


def load_experiment_config(path: str | Path) -> MultiModelExperimentConfig:
    """Load a YAML file, resolving an optional relative ``extends`` chain before validation."""

    return MultiModelExperimentConfig.model_validate(
        _load_yaml_with_extends(Path(path), set())
    )


def load_config_mapping(path: str | Path) -> dict:
    """Load an inherited YAML mapping for method-specific validated config models."""

    return _load_yaml_with_extends(Path(path), set())
