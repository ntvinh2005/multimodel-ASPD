"""Versioned contract for the revised three-family ASPD v3 2B experiment.

This module is deliberately separate from :mod:`main_config`: checkpoints from the
4B experiment are not compatible with this experiment and its execution authority is
independent.  The scientific fields below are fail-closed; runtime-only benchmark knobs
live in the CLI and are recorded in the runtime fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_ORDER_2B = ("qwen", "gemma", "smol")
LOGICAL_MATRICES = ("q", "k", "v", "o", "mlp_gate", "mlp_up", "mlp_down")


class SourceModel2B(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: Literal["qwen", "gemma", "smol"]
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_id: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    family: Literal["qwen3", "gemma2", "llama"]
    checkpoint_parameter_count: int = Field(gt=1_000_000_000, lt=3_000_000_000)
    selected_layers: tuple[int, ...]
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_hidden_layers: int = Field(ge=8)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    layer_prefix: Literal["model.layers"]
    residual_pre_suffix: Literal["input_layernorm"]
    residual_mid_suffix: Literal["post_attention_layernorm", "pre_feedforward_layernorm"]
    frozen: Literal[True]

    @model_validator(mode="after")
    def validate_architecture(self) -> SourceModel2B:
        if len(self.selected_layers) != 8:
            raise ValueError(f"{self.key} must select exactly eight layers")
        expected = tuple(range(self.selected_layers[0], self.selected_layers[0] + 8))
        if self.selected_layers != expected or self.selected_layers[-1] >= self.num_hidden_layers:
            raise ValueError(f"{self.key} layer window must be contiguous and in range")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(f"{self.key} hidden size is incompatible with its attention heads")
        expected_mid = (
            "pre_feedforward_layernorm" if self.family == "gemma2" else "post_attention_layernorm"
        )
        if self.residual_mid_suffix != expected_mid:
            raise ValueError(f"{self.key} residual-MID hook is incompatible with {self.family}")
        return self

    @property
    def q_out(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        return self.num_key_value_heads * self.head_dim


class Architecture2B(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dictionary_size: Literal[8192]
    batch_topk_k: Literal[32]
    pool_tokens: Literal[2048]
    pools_per_step: Literal[4]
    top_k_aux: Literal[2048]
    auxk_coefficient: Literal[0.03125]
    matryoshka_group_fracs: tuple[
        Literal[0.0625], Literal[0.0625], Literal[0.125], Literal[0.25], Literal[0.5]
    ]
    gate_grouping: Literal["pre_mid"]
    model_order: tuple[Literal["qwen", "gemma", "smol"], ...]
    logical_matrices: tuple[
        Literal["q", "k", "v", "o", "mlp_gate", "mlp_up", "mlp_down"], ...
    ]
    latent_tokens_per_model: Literal[8]
    packing_method: Literal["learned_projection"]
    packed_token_width: Literal[512]
    intra_model_layers: Literal[2]
    intra_model_heads: Literal[8]
    cross_model_layers: Literal[2]
    cross_model_heads: Literal[8]
    cross_model_width: Literal[512]
    cross_input_adapter: Literal["linear_517_to_512"]
    feature_readout: Literal["mean"]
    dropout: Literal[0]
    activation: Literal["GELU"]

    @model_validator(mode="after")
    def validate_coverage(self) -> Architecture2B:
        if self.model_order != MODEL_ORDER_2B:
            raise ValueError(f"2B model order must be {MODEL_ORDER_2B}")
        if self.logical_matrices != LOGICAL_MATRICES:
            raise ValueError("2B experiment must retain all seven logical matrices")
        if self.pool_tokens * self.pools_per_step != 8192:
            raise ValueError("four pools must contain exactly 8192 accepted anchors")
        return self


class Objective2B(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal_code: Literal["binary_support"]
    activation_code: Literal["continuous"]
    internal_gradient_route: Literal["rank_one_only"]
    activation_gradient_route: Literal[
        "encoder_packing_adapter_readout_and_activation_decoder"
    ]
    internal_loss: Literal["balanced_pre_mid_mean_fvu"]
    activation_loss: Literal["balanced_pre_mid_mean_matryoshka_fvu"]
    auxk_loss: Literal["balanced_pre_mid_dead_feature_revival"]
    internal_coefficient: Literal[1.0]
    activation_coefficient: Literal[1.0]
    fvu_reduction_dtype: Literal["float32"]
    activation_decoder_normalization: Literal[
        "unit_row_norm_with_radial_gradient_projection"
    ]
    native_bias: Literal["preserve"]


class Optimization2B(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer: Literal["Adam"]
    betas: tuple[Literal[0.9], Literal[0.99]]
    weight_decay: Literal[0]
    rank_one_lr: Literal[0.0003]
    activation_decoder_lr: Literal[0.0003]
    encoder_lr: Literal[0.0001]
    schedule: Literal["constant"]
    seed: Literal[0]
    precision: Literal["bfloat16"]
    fp32_master_parameters: Literal[True]


class Data2B(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: Literal["monology/pile-uncopyrighted"]
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sequence_length: Literal[512]
    effective_batch_anchors: Literal[8192]
    minimum_accepted_anchor_exposures: Literal[500006912]
    minimum_optimizer_steps: Literal[61036]
    split_unit: Literal["document"]
    split_seed: Literal[0]
    split_modulus: Literal[1000]
    validation_buckets: tuple[int, ...]
    test_buckets: tuple[int, ...]
    canonical_text_policy: Literal["identity_exact_three_way_prefix"]
    validation_target_anchors: Literal[5000000]
    test_target_anchors: Literal[5000000]

    @model_validator(mode="after")
    def validate_budget(self) -> Data2B:
        if self.validation_buckets != tuple(range(10)):
            raise ValueError("validation buckets must be 0..9")
        if self.test_buckets != tuple(range(10, 20)):
            raise ValueError("test buckets must be 10..19")
        if self.effective_batch_anchors * self.minimum_optimizer_steps != (
            self.minimum_accepted_anchor_exposures
        ):
            raise ValueError("2B exposure and optimizer-step budgets disagree")
        return self


class Runtime2B(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["model_owned"]
    world_size: Literal[3]
    checkpoint_every_steps: Literal[1000]
    keep_complete_checkpoints: Literal[2]
    activation_backward_chunk: int = Field(gt=0, le=8)
    matrix_backward_chunk: int = Field(gt=0, le=56)
    early_exit_after_selected_layers: Literal[True]
    output_root: str
    checkpoint_root: str
    dataset_manifest: str
    alignment_manifest: str
    preflight_evidence: str
    max_preflight_steps: int = Field(gt=0, le=20)
    max_preflight_minutes: int = Field(gt=0, le=45)

    @model_validator(mode="after")
    def validate_namespace(self) -> Runtime2B:
        values = (
            self.output_root,
            self.checkpoint_root,
            self.dataset_manifest,
            self.alignment_manifest,
            self.preflight_evidence,
        )
        if any("aspd_v3/main_2b_v1" not in value for value in values):
            raise ValueError("2B paths must use the isolated aspd_v3/main_2b_v1 namespace")
        return self


class V3Main2BV1Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
    experiment_family: Literal["aspd_v3_three_family_2b_main"]
    scientific_status: Literal["FROZEN_RESEARCHER_REVISED"]
    full_training_authorized: Literal[False]
    models: tuple[SourceModel2B, SourceModel2B, SourceModel2B]
    architecture: Architecture2B
    objective: Objective2B
    optimization: Optimization2B
    data: Data2B
    runtime: Runtime2B
    evaluation_status: Literal["NOT_IMPLEMENTED — DEFERRED BY RESEARCHER"]

    @model_validator(mode="after")
    def validate_models(self) -> V3Main2BV1Config:
        if tuple(model.key for model in self.models) != MODEL_ORDER_2B:
            raise ValueError(f"2B sources must be ordered {MODEL_ORDER_2B}")
        expected = {
            "qwen": ("Qwen/Qwen3-1.7B", "qwen3", tuple(range(10, 18))),
            "gemma": ("google/gemma-2-2b-it", "gemma2", tuple(range(9, 17))),
            "smol": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "llama", tuple(range(8, 16))),
        }
        for model in self.models:
            if (model.model_id, model.family, model.selected_layers) != expected[model.key]:
                raise ValueError(f"2B source registry mismatch for {model.key}")
        return self

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def structural_report(self) -> dict[str, Any]:
        return {
            "status": "STRUCTURALLY_READY",
            "config_fingerprint": self.fingerprint(),
            "logical_target_count": 3 * 8 * 7,
            "minimum_optimizer_steps": self.data.minimum_optimizer_steps,
            "accepted_anchor_exposures": self.data.minimum_accepted_anchor_exposures,
            "full_training_authorized": self.full_training_authorized,
            "evaluation": self.evaluation_status,
        }


def load_main_2b_v1_config(path: str | Path) -> V3Main2BV1Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ASPD v3 main-2b-v1 config must be a mapping")
    return V3Main2BV1Config.model_validate(payload)
