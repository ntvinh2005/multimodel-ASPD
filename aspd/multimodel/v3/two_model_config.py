"""Fail-closed contract for the Gemma/SmolLM2 ASPD v3 scale pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_ORDER_TWO = ("gemma", "smol")
LOGICAL_MATRICES_TWO = ("q", "k", "v", "o", "mlp_gate", "mlp_up", "mlp_down")


class TwoModelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: Literal["gemma", "smol"]
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_id: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    family: Literal["gemma2", "llama"]
    license: Literal["apache-2.0", "gemma"]
    access: Literal["public", "gated_account_scoped"]
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
    def validate_architecture(self) -> TwoModelSource:
        expected_layers = tuple(range(self.selected_layers[0], self.selected_layers[0] + 8))
        if self.selected_layers != expected_layers:
            raise ValueError(f"{self.key} must select eight contiguous layers")
        if self.selected_layers[-1] >= self.num_hidden_layers:
            raise ValueError(f"{self.key} selected layer lies outside its backbone")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(f"{self.key} hidden size is incompatible with attention heads")
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


class TwoModelArchitecture(BaseModel):
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
    model_order: tuple[Literal["gemma"], Literal["smol"]]
    logical_matrices: tuple[Literal["q", "k", "v", "o", "mlp_gate", "mlp_up", "mlp_down"], ...]
    latent_tokens_per_model: Literal[8]
    packing_method: Literal["learned_projection"]
    packed_token_width: Literal[512]
    intra_model_layers: Literal[2]
    intra_model_heads: Literal[8]
    cross_model_layers: Literal[2]
    cross_model_heads: Literal[8]
    cross_model_width: Literal[512]
    cross_input_adapter: Literal["linear_515_to_512"]
    feature_readout: Literal["mean"]
    dropout: Literal[0]
    activation: Literal["GELU"]

    @model_validator(mode="after")
    def validate_coverage(self) -> TwoModelArchitecture:
        if self.model_order != MODEL_ORDER_TWO:
            raise ValueError(f"two-model order must be {MODEL_ORDER_TWO}")
        if self.logical_matrices != LOGICAL_MATRICES_TWO:
            raise ValueError("two-model pilot must retain all seven logical matrices")
        return self


class TwoModelObjective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal_code: Literal["binary_support"]
    activation_code: Literal["continuous"]
    internal_gradient_route: Literal["rank_one_only"]
    activation_gradient_route: Literal[
        "encoder_packing_adapter_readout_activation_decoder_and_optional_scale"
    ]
    internal_loss: Literal["balanced_pre_mid_mean_fvu"]
    activation_loss: Literal["balanced_pre_mid_mean_matryoshka_fvu"]
    auxk_loss: Literal["balanced_pre_mid_dead_feature_revival"]
    internal_coefficient: Literal[1.0]
    activation_coefficient: Literal[1.0]
    fvu_reduction_dtype: Literal["float32"]
    activation_decoder_normalization: Literal["unit_row_norm_with_radial_gradient_projection"]
    activation_reconstruction_scale: Literal["none", "two_model_geomean_log_scale"]
    native_bias: Literal["preserve"]


class TwoModelOptimization(BaseModel):
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


class TwoModelData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: Literal["monology/pile-uncopyrighted"]
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sequence_length: Literal[512]
    effective_batch_anchors: Literal[8192]
    accepted_anchor_exposures: Literal[10002432]
    optimizer_steps: Literal[1221]
    split_unit: Literal["document"]
    split_seed: Literal[0]
    split_modulus: Literal[1000]
    validation_buckets: tuple[int, ...]
    test_buckets: tuple[int, ...]
    canonical_text_policy: Literal["identity_exact_two_way_prefix"]
    heldout_target_anchors: Literal[100352]

    @model_validator(mode="after")
    def validate_budget(self) -> TwoModelData:
        if self.validation_buckets != tuple(range(10)):
            raise ValueError("validation buckets must be 0..9")
        if self.test_buckets != tuple(range(10, 20)):
            raise ValueError("test buckets must be 10..19")
        if self.effective_batch_anchors * self.optimizer_steps != self.accepted_anchor_exposures:
            raise ValueError("two-model exposure and optimizer-step budgets disagree")
        return self


class TwoModelRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["model_owned"]
    world_size: Literal[2]
    checkpoint_every_steps: Literal[1000]
    keep_complete_checkpoints: Literal[2]
    activation_backward_chunk: int = Field(gt=0, le=8)
    matrix_backward_chunk: int = Field(gt=0, le=56)
    sparse_backend: Literal["dense"]
    early_exit_after_selected_layers: Literal[True]
    data_prefetch_depth: Literal[1]
    ledger_anchor_storage: Literal["exact_compact_v1"]
    output_root: str
    dataset_manifest: str
    alignment_manifest: str
    preflight_evidence: str
    max_preflight_steps: int = Field(gt=0, le=20)
    max_preflight_minutes: int = Field(gt=0, le=45)

    @model_validator(mode="after")
    def validate_namespace(self) -> TwoModelRuntime:
        paths = (
            self.output_root,
            self.dataset_manifest,
            self.alignment_manifest,
            self.preflight_evidence,
        )
        if any("aspd_v3/two_model_scale_v1" not in path for path in paths):
            raise ValueError("two-model paths must use aspd_v3/two_model_scale_v1")
        return self


class TwoModelScalePilotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    experiment_family: Literal["aspd_v3_gemma_smol_scale_pilot"]
    scientific_status: Literal["RESEARCHER_APPROVED_SHORT_PILOT"]
    full_training_authorized: Literal[False]
    short_pilot_authorized: Literal[True]
    arm: Literal["baseline", "model_specific_scale"]
    models: tuple[TwoModelSource, TwoModelSource]
    architecture: TwoModelArchitecture
    objective: TwoModelObjective
    optimization: TwoModelOptimization
    data: TwoModelData
    runtime: TwoModelRuntime
    evaluation_status: Literal["HELDOUT_EVALUATION_REQUIRED"]

    @model_validator(mode="after")
    def validate_contract(self) -> TwoModelScalePilotConfig:
        if tuple(model.key for model in self.models) != MODEL_ORDER_TWO:
            raise ValueError(f"two-model sources must be ordered {MODEL_ORDER_TWO}")
        expected = {
            "gemma": ("google/gemma-2-2b-it", "gemma2", tuple(range(9, 17))),
            "smol": (
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "llama",
                tuple(range(8, 16)),
            ),
        }
        for model in self.models:
            if (model.model_id, model.family, model.selected_layers) != expected[model.key]:
                raise ValueError(f"two-model source registry mismatch for {model.key}")
        scale = self.objective.activation_reconstruction_scale
        expected_scale = "none" if self.arm == "baseline" else "two_model_geomean_log_scale"
        if scale != expected_scale:
            raise ValueError(
                f"{self.arm} requires activation_reconstruction_scale={expected_scale}"
            )
        return self

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def structural_report(self) -> dict[str, Any]:
        return {
            "status": "STRUCTURALLY_READY",
            "arm": self.arm,
            "config_fingerprint": self.fingerprint(),
            "logical_target_count": 112,
            "optimizer_steps": self.data.optimizer_steps,
            "accepted_anchor_exposures": self.data.accepted_anchor_exposures,
            "full_training_authorized": self.full_training_authorized,
        }


def load_two_model_scale_config(path: str | Path) -> TwoModelScalePilotConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("two-model scale pilot config must be a mapping")
    return TwoModelScalePilotConfig.model_validate(payload)
