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
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    grounding: HookSpec
    matrices: list[MatrixSpec]

    @model_validator(mode="after")
    def unique_matrix_names(self) -> "ModelSpec":
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


class DataSpec(BaseModel):
    tokenizer: str
    tokenizer_revision: str | None = None
    chat_template_tokenizer: str | None = None
    sequence_length: PositiveInt = 256
    train_tokens: PositiveInt
    validation_tokens: PositiveInt
    shuffle_buffer: PositiveInt = 10_000
    seed: int = 0
    sources: list[DataSourceSpec]


class CacheSpec(BaseModel):
    root: str
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
    def heads_divide_projection(self) -> "EncoderSpec":
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
    shared_fraction: float = Field(default=0.75, gt=0.0, lt=1.0)
    top_k_shared: PositiveInt | None = None
    top_k_exclusive: PositiveInt | None = None
    d2_gamma: float = Field(default=1.0, ge=0.0)
    tie_shared_decoders: bool = False
    tie_shared_mechanisms: bool = False

    @model_validator(mode="after")
    def validate_partition(self) -> "SparsitySpec":
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
    matryoshka_group_fractions: list[PositiveFloat] = Field(
        default_factory=lambda: [0.0625, 0.0625, 0.125, 0.25, 0.5]
    )

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "ObjectiveSpec":
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


class AnalysisSpec(BaseModel):
    shared_epsilon: float = Field(default=0.1, gt=0.0, lt=0.5)
    concentrated_threshold: float = Field(default=0.9, gt=0.5, le=1.0)
    top_examples_per_feature: PositiveInt = 10
    max_example_features: PositiveInt = 256


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
    analysis: AnalysisSpec = Field(default_factory=AnalysisSpec)

    @model_validator(mode="after")
    def validate_models_and_tying(self) -> "MultiModelExperimentConfig":
        if not self.models:
            raise ValueError("at least one model is required")
        names = [model.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate model names: {names}")
        if self.sparsity.tie_shared_mechanisms:
            matrix_names = [[matrix.name for matrix in model.matrices] for model in self.models]
            if any(names != matrix_names[0] for names in matrix_names[1:]):
                raise ValueError("tied shared mechanisms require matching matrix names and order")
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
