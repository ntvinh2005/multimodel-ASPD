"""Fail-closed contract for the Granite three-model normalized-target experiment.

The two checked-in arms are implementation-review artifacts, not execution
authorization.  Scientific settings inherited from ASPD v3 are explicit so a future
approval can review a complete diff, while all GPU execution flags remain literal false.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_ORDER_GRANITE = ("gemma", "smol", "granite")
LOGICAL_MATRICES_GRANITE = ("q", "k", "v", "o", "mlp_gate", "mlp_up", "mlp_down")
GRANITE_REVISION = "707f574c62054322f6b5b04b6d075f0a8f05e0f0"


class GraniteExperimentSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: Literal["gemma", "smol", "granite"]
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_id: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    family: Literal["gemma2", "llama", "granite"]
    architecture_class: Literal["Gemma2ForCausalLM", "LlamaForCausalLM", "GraniteForCausalLM"]
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
    residual_multiplier: float | None
    attention_multiplier: float | None
    embedding_multiplier: float | None
    attention_bias: bool
    mlp_bias: bool
    frozen: Literal[True]

    @model_validator(mode="after")
    def validate_architecture(self) -> GraniteExperimentSource:
        expected_layers = tuple(range(self.selected_layers[0], self.selected_layers[0] + 8))
        if self.selected_layers != expected_layers:
            raise ValueError(f"{self.key} must select exactly eight contiguous layers")
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


class GraniteExperimentArchitecture(BaseModel):
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
    model_order: tuple[Literal["gemma"], Literal["smol"], Literal["granite"]]
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
    def validate_scope(self) -> GraniteExperimentArchitecture:
        if self.model_order != MODEL_ORDER_GRANITE:
            raise ValueError(f"Granite experiment model order must be {MODEL_ORDER_GRANITE}")
        if self.logical_matrices != LOGICAL_MATRICES_GRANITE:
            raise ValueError("Granite experiment must retain all seven logical matrices")
        if self.pool_tokens * self.pools_per_step != 8192:
            raise ValueError("four pools must contain exactly 8192 accepted anchors")
        return self


class GraniteExperimentObjective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal_code: Literal["binary_support"]
    activation_code: Literal["continuous"]
    internal_gradient_route: Literal["rank_one_only"]
    activation_gradient_route: Literal["encoder_and_activation_decoder"]
    internal_loss: Literal["balanced_pre_mid_mean_fvu"]
    activation_loss: Literal["balanced_pre_mid_mean_matryoshka_fvu"]
    auxk_loss: Literal["balanced_pre_mid_dead_feature_revival"]
    internal_coefficient: Literal[1.0]
    activation_coefficient: Literal[1.0]
    fvu_reduction_dtype: Literal["float32"]
    activation_decoder_normalization: Literal[
        "none", "unit_row_norm_with_radial_gradient_projection"
    ]
    activation_target_normalization: Literal["none", "per_anchor_l2"]
    activation_target_epsilon: Literal[1e-6]
    activation_reconstruction_scale: Literal["none"]
    native_bias: Literal["preserve"]
    oracle_norm_evaluation: Literal[True]


class GraniteExperimentOptimization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "PROPOSED_INHERITED_NOT_EXECUTION_APPROVED",
        "INHERITED_SETTINGS_STABILIZATION_PILOT_AUTHORIZED",
        "INHERITED_SETTINGS_RELIABILITY_PILOT_AUTHORIZED",
        "DECODER_NORM_CORRECTION_RELIABILITY_PILOT_AUTHORIZED",
    ]
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


class GraniteExperimentData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: Literal["monology/pile-uncopyrighted"]
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sequence_length: Literal[512]
    effective_batch_anchors: Literal[8192]
    split_unit: Literal["document"]
    split_seed: Literal[0]
    split_modulus: Literal[1000]
    validation_buckets: tuple[int, ...]
    test_buckets: tuple[int, ...]
    canonical_text_policy: Literal["identity_exact_three_way_prefix"]
    alignment_roster: tuple[Literal["gemma"], Literal["smol"], Literal["granite"]]
    alignment_acceptance_rate_status: Literal["UNMEASURED_REQUIRES_GPU_PREFLIGHT"]
    training_budget_status: Literal[
        "UNAPPROVED",
        "RESEARCHER_AUTHORIZED_STABILIZATION_PILOT",
        "RESEARCHER_AUTHORIZED_RELIABILITY_PILOT",
        "RESEARCHER_AUTHORIZED_DECODER_NORM_CORRECTION_PILOT",
    ]
    approved_optimizer_steps: int | None
    approved_accepted_anchor_exposures: int | None

    @model_validator(mode="after")
    def validate_splits(self) -> GraniteExperimentData:
        if self.validation_buckets != tuple(range(10)):
            raise ValueError("validation buckets must be 0..9")
        if self.test_buckets != tuple(range(10, 20)):
            raise ValueError("test buckets must be 10..19")
        if self.alignment_roster != MODEL_ORDER_GRANITE:
            raise ValueError("alignment roster must exclude every historical Qwen identity")
        return self


class GraniteExperimentRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "STRUCTURAL_ONLY_GPU_NOT_AUTHORIZED",
        "THREE_STEP_GPU_PREFLIGHT_AUTHORIZED",
        "TREATMENT_STABILIZATION_PILOT_AUTHORIZED",
        "MATCHED_BASELINE_PILOT_AUTHORIZED",
        "TREATMENT_RELIABILITY_PILOT_AUTHORIZED",
        "MATCHED_BASELINE_RELIABILITY_PILOT_AUTHORIZED",
        "DECODER_NORM_CORRECTION_RELIABILITY_PILOT_AUTHORIZED",
    ]
    strategy: Literal["model_owned"]
    world_size: Literal[3]
    checkpoint_every_steps: Literal[1000]
    keep_complete_checkpoints: Literal[2]
    activation_backward_chunk: int = Field(gt=0, le=8)
    matrix_backward_chunk: int = Field(gt=0, le=56)
    sparse_backend: Literal["dense"]
    early_exit_after_selected_layers: Literal[True]
    data_prefetch_depth: Literal[1]
    ledger_anchor_storage: Literal["exact_compact_v1"]
    output_root: str
    checkpoint_root: str
    dataset_manifest: str
    alignment_manifest: str
    preflight_evidence: str | None
    max_preflight_steps: Literal[20]
    max_preflight_minutes: Literal[45]

    @model_validator(mode="after")
    def validate_namespace(self) -> GraniteExperimentRuntime:
        output_namespaces = (
            "aspd_v3/granite_normalized_target_v1",
            "aspd_v3/granite_normalized_target_unconstrained_decoder_v1",
        )
        if not all(
            any(namespace in value for namespace in output_namespaces)
            for value in (self.output_root, self.checkpoint_root)
        ):
            raise ValueError("Granite outputs must use a recognized isolated namespace")
        if any(
            "aspd_v3/granite_normalized_target_v1" not in value
            for value in (self.dataset_manifest, self.alignment_manifest)
        ):
            raise ValueError("corrected pilot must reuse the approved immutable data manifests")
        values = (
            self.output_root,
            self.checkpoint_root,
            self.dataset_manifest,
            self.alignment_manifest,
        )
        if "qwen" in " ".join(values).lower():
            raise ValueError("Granite experiment runtime paths must not mention Qwen")
        return self


class GranitePreflightAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authorized_by: Literal["researcher"]
    authorized_on: Literal["2026-11-06"]
    scope: Literal["three_step_gpu_preflight_only"]
    optimizer_steps: Literal[3]
    gpu_count: Literal[3]
    gpu_type: Literal["b200"]
    pilot_authorized: Literal[False]
    full_training_authorized: Literal[False]


class GranitePilotAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authorized_by: Literal["researcher"]
    authorized_on: Literal["2026-11-06", "2026-11-07"]
    scope: Literal[
        "treatment_stabilization_128_steps",
        "matched_baseline_128_steps",
        "treatment_reliability_1221_steps",
        "matched_baseline_reliability_1221_steps",
        "normalized_target_decoder_norm_correction_1221_steps",
    ]
    arm: Literal["baseline", "normalized_target"]
    optimizer_steps: Literal[128, 1221]
    accepted_anchor_exposures: Literal[1_048_576, 10_002_432]
    gpu_count: Literal[3]
    gpu_type: Literal["b200"]
    preflight_report_sha256: Literal[
        "00500532a7c4e5e3dae33d7472028cf6717edebfd23d697d5318df491cdde454",
        "a28b22b8cd233e8d15a86c2ab4e38ea0ae13a97adbe7bda9660da12d5542ba3e",
    ]
    preflight_config_fingerprint: Literal[
        "37c5c902e69503c362db37ee5031c0d83be2aa50e6e16fb97d5adb7bbbb93acc",
        "642ce56a1fa83ec552f53d777d73bce6533fb8d9c8f7cb392840f27a0cc705dc",
    ]
    gate_evidence: tuple[dict[str, str], ...] = ()
    full_training_authorized: Literal[False]


class GraniteNormalizedTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    experiment_family: Literal[
        "aspd_v3_granite_normalized_target_v1",
        "aspd_v3_granite_normalized_target_unconstrained_decoder_v1",
    ]
    scientific_status: Literal[
        "IMPLEMENTATION_ONLY_RESEARCHER_REVIEW_REQUIRED",
        "THREE_STEP_GPU_PREFLIGHT_AUTHORIZED_BY_RESEARCHER",
        "TREATMENT_STABILIZATION_PILOT_AUTHORIZED_BY_RESEARCHER",
        "MATCHED_BASELINE_PILOT_AUTHORIZED_BY_RESEARCHER",
        "TREATMENT_RELIABILITY_PILOT_AUTHORIZED_BY_RESEARCHER",
        "MATCHED_BASELINE_RELIABILITY_PILOT_AUTHORIZED_BY_RESEARCHER",
        "DECODER_NORM_CORRECTION_RELIABILITY_PILOT_AUTHORIZED_BY_RESEARCHER",
    ]
    full_training_authorized: Literal[False]
    short_pilot_authorized: bool
    gpu_preflight_authorized: bool
    preflight_authorization: GranitePreflightAuthorization | None = None
    pilot_authorization: GranitePilotAuthorization | None = None
    arm: Literal["baseline", "normalized_target"]
    models: tuple[GraniteExperimentSource, GraniteExperimentSource, GraniteExperimentSource]
    architecture: GraniteExperimentArchitecture
    objective: GraniteExperimentObjective
    optimization: GraniteExperimentOptimization
    data: GraniteExperimentData
    runtime: GraniteExperimentRuntime
    evaluation_status: Literal[
        "SYNTHETIC_IMPLEMENTATION_ONLY_GPU_NOT_RUN",
        "GPU_PREFLIGHT_PASSED_STABILIZATION_PILOT_AUTHORIZED",
        "GPU_PREFLIGHT_PASSED_MATCHED_BASELINE_PILOT_AUTHORIZED",
        "HELDOUT_GATE_PASSED_RELIABILITY_PILOT_AUTHORIZED",
        "DECODER_NORM_CORRECTION_PILOT_AUTHORIZED",
    ]

    @model_validator(mode="after")
    def validate_contract(self) -> GraniteNormalizedTargetConfig:
        if tuple(model.key for model in self.models) != MODEL_ORDER_GRANITE:
            raise ValueError(f"sources must be ordered {MODEL_ORDER_GRANITE}")
        expected = {
            "gemma": (
                "google/gemma-2-2b-it",
                "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
                "gemma2",
                "Gemma2ForCausalLM",
                tuple(range(9, 17)),
                2_614_341_888,
            ),
            "smol": (
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "31b70e2e869a7173562077fd711b654946d38674",
                "llama",
                "LlamaForCausalLM",
                tuple(range(8, 16)),
                1_711_376_384,
            ),
            "granite": (
                "ibm-granite/granite-3.3-2b-instruct",
                GRANITE_REVISION,
                "granite",
                "GraniteForCausalLM",
                tuple(range(16, 24)),
                2_533_539_840,
            ),
        }
        for model in self.models:
            observed = (
                model.model_id,
                model.revision,
                model.family,
                model.architecture_class,
                model.selected_layers,
                model.checkpoint_parameter_count,
            )
            if observed != expected[model.key]:
                raise ValueError(f"source registry mismatch for {model.key}")
            if model.tokenizer_id != model.model_id or model.tokenizer_revision != model.revision:
                raise ValueError(f"{model.key} tokenizer must be pinned to the model commit")
        granite = self.models[2]
        granite_expected = {
            "residual_multiplier": 0.22,
            "attention_multiplier": 0.015625,
            "embedding_multiplier": 12.0,
            "attention_bias": False,
            "mlp_bias": False,
        }
        mismatch = {
            key: {"expected": value, "observed": getattr(granite, key)}
            for key, value in granite_expected.items()
            if getattr(granite, key) != value
        }
        if mismatch:
            raise ValueError(f"Granite published architecture mismatch: {mismatch}")
        expected_mode = "none" if self.arm == "baseline" else "per_anchor_l2"
        if self.objective.activation_target_normalization != expected_mode:
            raise ValueError(f"{self.arm} requires activation_target_normalization={expected_mode}")
        corrected_decoder = (
            self.experiment_family
            == "aspd_v3_granite_normalized_target_unconstrained_decoder_v1"
        )
        expected_decoder_mode = (
            "none" if corrected_decoder else "unit_row_norm_with_radial_gradient_projection"
        )
        if self.objective.activation_decoder_normalization != expected_decoder_mode:
            raise ValueError(
                f"{self.experiment_family} requires "
                f"activation_decoder_normalization={expected_decoder_mode}"
            )
        if corrected_decoder:
            if self.arm != "normalized_target":
                raise ValueError("decoder-norm correction pilot is normalized-target only")
            if not self.short_pilot_authorized or self.pilot_authorization is None:
                raise ValueError("corrected decoder pilot requires bounded researcher authorization")
            observed = {
                "scientific_status": self.scientific_status,
                "runtime_status": self.runtime.status,
                "evaluation_status": self.evaluation_status,
                "scope": self.pilot_authorization.scope,
                "steps": self.pilot_authorization.optimizer_steps,
                "exposures": self.pilot_authorization.accepted_anchor_exposures,
                "optimization_status": self.optimization.status,
                "budget_status": self.data.training_budget_status,
                "approved_steps": self.data.approved_optimizer_steps,
                "approved_exposures": self.data.approved_accepted_anchor_exposures,
                "full_training_authorized": self.pilot_authorization.full_training_authorized,
            }
            expected_corrected = {
                "scientific_status": (
                    "DECODER_NORM_CORRECTION_RELIABILITY_PILOT_AUTHORIZED_BY_RESEARCHER"
                ),
                "runtime_status": "DECODER_NORM_CORRECTION_RELIABILITY_PILOT_AUTHORIZED",
                "evaluation_status": "DECODER_NORM_CORRECTION_PILOT_AUTHORIZED",
                "scope": "normalized_target_decoder_norm_correction_1221_steps",
                "steps": 1221,
                "exposures": 10_002_432,
                "optimization_status": (
                    "DECODER_NORM_CORRECTION_RELIABILITY_PILOT_AUTHORIZED"
                ),
                "budget_status": "RESEARCHER_AUTHORIZED_DECODER_NORM_CORRECTION_PILOT",
                "approved_steps": 1221,
                "approved_exposures": 10_002_432,
                "full_training_authorized": False,
            }
            if observed != expected_corrected:
                raise ValueError(f"decoder-norm correction authorization mismatch: {observed}")
            namespace = (
                "/orange/thai/vinhnguyen1/aspd_v3/"
                "granite_normalized_target_unconstrained_decoder_v1/reliability/normalized_target"
            )
            if self.runtime.output_root != f"{namespace}/runs" or (
                self.runtime.checkpoint_root != f"{namespace}/checkpoints"
            ):
                raise ValueError("corrected pilot output namespace changed")
            if self.runtime.preflight_evidence is None or not self.runtime.preflight_evidence.endswith(
                "/preflight/normalized_target/preflight_report.json"
            ):
                raise ValueError("corrected pilot must retain treatment preflight evidence")
            if self.gpu_preflight_authorized or self.preflight_authorization is not None:
                raise ValueError("corrected pilot cannot reuse preflight authorization")
            if self.pilot_authorization.gate_evidence:
                raise ValueError("corrected pilot cannot claim post-correction gate evidence")
            return self
        if self.short_pilot_authorized:
            if self.gpu_preflight_authorized or self.preflight_authorization is not None:
                raise ValueError("pilot authorization must not reuse preflight authorization")
            if self.pilot_authorization is None:
                raise ValueError("stabilization pilot requires a bounded authorization record")
            stabilization = {
                "normalized_target": {
                    "scope": "treatment_stabilization_128_steps",
                    "scientific_status": (
                        "TREATMENT_STABILIZATION_PILOT_AUTHORIZED_BY_RESEARCHER"
                    ),
                    "runtime_status": "TREATMENT_STABILIZATION_PILOT_AUTHORIZED",
                    "evaluation_status": (
                        "GPU_PREFLIGHT_PASSED_STABILIZATION_PILOT_AUTHORIZED"
                    ),
                    "preflight_sha256": (
                        "00500532a7c4e5e3dae33d7472028cf6717edebfd23d697d5318df491cdde454"
                    ),
                    "preflight_fingerprint": (
                        "37c5c902e69503c362db37ee5031c0d83be2aa50e6e16fb97d5adb7bbbb93acc"
                    ),
                    "steps": 128,
                    "exposures": 1_048_576,
                },
                "baseline": {
                    "scope": "matched_baseline_128_steps",
                    "scientific_status": "MATCHED_BASELINE_PILOT_AUTHORIZED_BY_RESEARCHER",
                    "runtime_status": "MATCHED_BASELINE_PILOT_AUTHORIZED",
                    "evaluation_status": (
                        "GPU_PREFLIGHT_PASSED_MATCHED_BASELINE_PILOT_AUTHORIZED"
                    ),
                    "preflight_sha256": (
                        "a28b22b8cd233e8d15a86c2ab4e38ea0ae13a97adbe7bda9660da12d5542ba3e"
                    ),
                    "preflight_fingerprint": (
                        "642ce56a1fa83ec552f53d777d73bce6533fb8d9c8f7cb392840f27a0cc705dc"
                    ),
                    "steps": 128,
                    "exposures": 1_048_576,
                },
            }
            reliability = {
                "normalized_target": {
                    "scope": "treatment_reliability_1221_steps",
                    "scientific_status": (
                        "TREATMENT_RELIABILITY_PILOT_AUTHORIZED_BY_RESEARCHER"
                    ),
                    "runtime_status": "TREATMENT_RELIABILITY_PILOT_AUTHORIZED",
                    "evaluation_status": "HELDOUT_GATE_PASSED_RELIABILITY_PILOT_AUTHORIZED",
                    "optimization_status": "INHERITED_SETTINGS_RELIABILITY_PILOT_AUTHORIZED",
                    "budget_status": "RESEARCHER_AUTHORIZED_RELIABILITY_PILOT",
                    "steps": 1221,
                    "exposures": 10_002_432,
                    "output_segment": "reliability/normalized_target",
                    "preflight_sha256": (
                        "00500532a7c4e5e3dae33d7472028cf6717edebfd23d697d5318df491cdde454"
                    ),
                    "preflight_fingerprint": (
                        "37c5c902e69503c362db37ee5031c0d83be2aa50e6e16fb97d5adb7bbbb93acc"
                    ),
                },
                "baseline": {
                    "scope": "matched_baseline_reliability_1221_steps",
                    "scientific_status": (
                        "MATCHED_BASELINE_RELIABILITY_PILOT_AUTHORIZED_BY_RESEARCHER"
                    ),
                    "runtime_status": "MATCHED_BASELINE_RELIABILITY_PILOT_AUTHORIZED",
                    "evaluation_status": "HELDOUT_GATE_PASSED_RELIABILITY_PILOT_AUTHORIZED",
                    "optimization_status": "INHERITED_SETTINGS_RELIABILITY_PILOT_AUTHORIZED",
                    "budget_status": "RESEARCHER_AUTHORIZED_RELIABILITY_PILOT",
                    "steps": 1221,
                    "exposures": 10_002_432,
                    "output_segment": "reliability/baseline",
                    "preflight_sha256": (
                        "a28b22b8cd233e8d15a86c2ab4e38ea0ae13a97adbe7bda9660da12d5542ba3e"
                    ),
                    "preflight_fingerprint": (
                        "642ce56a1fa83ec552f53d777d73bce6533fb8d9c8f7cb392840f27a0cc705dc"
                    ),
                },
            }
            is_reliability = self.pilot_authorization.optimizer_steps == 1221
            authorized = (reliability if is_reliability else stabilization)[self.arm]
            observed = {
                "scope": self.pilot_authorization.scope,
                "scientific_status": self.scientific_status,
                "runtime_status": self.runtime.status,
                "evaluation_status": self.evaluation_status,
                "preflight_sha256": self.pilot_authorization.preflight_report_sha256,
                "preflight_fingerprint": self.pilot_authorization.preflight_config_fingerprint,
            }
            if observed != authorized:
                expected_subset = {key: authorized[key] for key in observed}
                if observed != expected_subset:
                    raise ValueError(f"{self.arm} pilot authorization mismatch: {observed}")
            if self.pilot_authorization.arm != self.arm:
                raise ValueError("pilot authorization arm differs from experiment arm")
            expected_optimization = authorized.get(
                "optimization_status", "INHERITED_SETTINGS_STABILIZATION_PILOT_AUTHORIZED"
            )
            expected_budget = authorized.get(
                "budget_status", "RESEARCHER_AUTHORIZED_STABILIZATION_PILOT"
            )
            if self.optimization.status != expected_optimization:
                raise ValueError("pilot optimization authorization status changed")
            if self.data.training_budget_status != expected_budget:
                raise ValueError("pilot budget authorization status changed")
            if (
                self.pilot_authorization.optimizer_steps != authorized["steps"]
                or self.pilot_authorization.accepted_anchor_exposures
                != authorized["exposures"]
            ):
                raise ValueError("pilot authorization budget changed")
            if (
                self.data.approved_optimizer_steps
                != self.pilot_authorization.optimizer_steps
                or self.data.approved_accepted_anchor_exposures
                != self.pilot_authorization.accepted_anchor_exposures
            ):
                raise ValueError("stabilization pilot budget differs from its authorization")
            if is_reliability:
                expected_artifacts = {
                    "baseline_reconstruction": (
                        "/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_v1/"
                        "evaluation/baseline/job_50066002/heldout/heldout_reconstruction.json",
                        "d9d0699cfe90c5aaef7f1ec19b3c517a1f6dcafbe0c72988c05192846aacf7ad",
                    ),
                    "baseline_dictionary": (
                        "/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_v1/"
                        "evaluation/baseline/job_50066002/heldout/dictionary_health.json",
                        "b356c701d0b79a712693779030a59da298b762e791bf2f9a976448f076fdf47e",
                    ),
                    "treatment_reconstruction": (
                        "/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_v1/"
                        "evaluation/normalized_target/job_50066003/heldout/heldout_reconstruction.json",
                        "a14adea8eb2b721cb95fb5bfc338f82867e389152e5de3812bd7c01c1d54fdb0",
                    ),
                    "treatment_dictionary": (
                        "/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_v1/"
                        "evaluation/normalized_target/job_50066003/heldout/dictionary_health.json",
                        "c761776b7f613615f4b8351f2b49ebb633eb0893b0e686cf7565cec289ce3672",
                    ),
                }
                observed_artifacts = {
                    row.get("name"): (row.get("path"), row.get("sha256"))
                    for row in self.pilot_authorization.gate_evidence
                }
                if observed_artifacts != expected_artifacts:
                    raise ValueError("reliability pilot held-out gate evidence changed")
            elif self.pilot_authorization.gate_evidence:
                raise ValueError("stabilization pilot cannot claim later gate evidence")
            expected_evidence = f"/preflight/{self.arm}/preflight_report.json"
            if self.runtime.preflight_evidence is None or not self.runtime.preflight_evidence.endswith(
                expected_evidence
            ):
                raise ValueError("pilot must pin arm-matched preflight evidence")
            output_segment = authorized.get("output_segment", f"stabilization/{self.arm}")
            expected_output = (
                "/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_v1/"
                f"{output_segment}/runs"
            )
            expected_checkpoint = (
                "/orange/thai/vinhnguyen1/aspd_v3/granite_normalized_target_v1/"
                f"{output_segment}/checkpoints"
            )
            if (
                self.runtime.output_root != expected_output
                or self.runtime.checkpoint_root != expected_checkpoint
            ):
                raise ValueError("stabilization pilot output namespace changed")
        elif self.gpu_preflight_authorized:
            if self.scientific_status != "THREE_STEP_GPU_PREFLIGHT_AUTHORIZED_BY_RESEARCHER":
                raise ValueError("authorized preflight requires the explicit scientific status")
            if self.preflight_authorization is None:
                raise ValueError("authorized preflight requires a bounded authorization record")
            if self.pilot_authorization is not None:
                raise ValueError("authorized preflight cannot carry pilot authorization")
            if self.runtime.status != "THREE_STEP_GPU_PREFLIGHT_AUTHORIZED":
                raise ValueError("authorized preflight requires the bounded runtime status")
            expected_suffix = f"/preflight/{self.arm}/preflight_report.json"
            if self.runtime.preflight_evidence is None or not self.runtime.preflight_evidence.endswith(
                expected_suffix
            ):
                raise ValueError("authorized preflight evidence path is not arm-isolated")
            if (
                self.optimization.status != "PROPOSED_INHERITED_NOT_EXECUTION_APPROVED"
                or self.data.training_budget_status != "UNAPPROVED"
                or self.data.approved_optimizer_steps is not None
                or self.data.approved_accepted_anchor_exposures is not None
                or self.evaluation_status != "SYNTHETIC_IMPLEMENTATION_ONLY_GPU_NOT_RUN"
            ):
                raise ValueError("preflight config cannot carry a training budget")
        else:
            if self.scientific_status != "IMPLEMENTATION_ONLY_RESEARCHER_REVIEW_REQUIRED":
                raise ValueError("review-only config has the wrong scientific status")
            if self.preflight_authorization is not None:
                raise ValueError("review-only config cannot carry execution authorization")
            if self.pilot_authorization is not None:
                raise ValueError("review-only config cannot carry pilot authorization")
            if self.runtime.status != "STRUCTURAL_ONLY_GPU_NOT_AUTHORIZED":
                raise ValueError("review-only config has an executable runtime status")
            if self.runtime.preflight_evidence is not None:
                raise ValueError("review-only config cannot name preflight evidence")
            if (
                self.optimization.status != "PROPOSED_INHERITED_NOT_EXECUTION_APPROVED"
                or self.data.training_budget_status != "UNAPPROVED"
                or self.data.approved_optimizer_steps is not None
                or self.data.approved_accepted_anchor_exposures is not None
                or self.evaluation_status != "SYNTHETIC_IMPLEMENTATION_ONLY_GPU_NOT_RUN"
            ):
                raise ValueError("review-only config cannot carry a training budget")
        return self

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def structural_report(self) -> dict[str, Any]:
        return {
            "status": "CODE_REVIEW_ONLY_NOT_EXECUTABLE",
            "config_fingerprint": self.fingerprint(),
            "model_order": list(self.architecture.model_order),
            "logical_target_count": 168,
            "activation_target_normalization": self.objective.activation_target_normalization,
            "full_training_authorized": self.full_training_authorized,
            "short_pilot_authorized": self.short_pilot_authorized,
            "gpu_preflight_authorized": self.gpu_preflight_authorized,
            "training_budget_status": self.data.training_budget_status,
        }


def load_granite_normalized_target_config(path: str | Path) -> GraniteNormalizedTargetConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Granite normalized-target config must be a mapping")
    if "extends" in payload:
        pilot_overlay = "pilot_authorization" in payload
        allowed_overlay = (
            {
                "extends",
                "experiment_family",
                "scientific_status",
                "short_pilot_authorized",
                "pilot_authorization",
                "objective",
                "optimization",
                "data",
                "runtime",
                "evaluation_status",
            }
            if pilot_overlay
            else {
                "extends",
                "scientific_status",
                "gpu_preflight_authorized",
                "preflight_authorization",
                "runtime",
            }
        )
        extra = set(payload) - allowed_overlay
        if extra:
            raise ValueError(f"preflight overlay changes forbidden fields: {sorted(extra)}")
        runtime_overlay = payload.get("runtime")
        expected_runtime_fields = (
            {"status", "output_root", "checkpoint_root", "preflight_evidence"}
            if pilot_overlay
            else {"status", "preflight_evidence"}
        )
        if not isinstance(runtime_overlay, dict) or set(runtime_overlay) != expected_runtime_fields:
            raise ValueError("execution overlay changes forbidden runtime fields")
        base_name = payload["extends"]
        if not isinstance(base_name, str) or Path(base_name).name != base_name:
            raise ValueError("preflight overlay must extend a sibling config filename")
        base_path = config_path.parent / base_name
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(base, dict) or "extends" in base:
            raise ValueError("preflight overlay base must be one complete config")
        merged = json.loads(json.dumps(base))
        if "experiment_family" in payload:
            merged["experiment_family"] = payload["experiment_family"]
        merged["scientific_status"] = payload["scientific_status"]
        if pilot_overlay:
            optimization_overlay = payload.get("optimization")
            data_overlay = payload.get("data")
            if not isinstance(optimization_overlay, dict) or set(optimization_overlay) != {"status"}:
                raise ValueError("pilot overlay may change only optimization status")
            if not isinstance(data_overlay, dict) or set(data_overlay) != {
                "training_budget_status",
                "approved_optimizer_steps",
                "approved_accepted_anchor_exposures",
            }:
                raise ValueError("pilot overlay may change only approved data budget fields")
            merged["short_pilot_authorized"] = payload["short_pilot_authorized"]
            merged["pilot_authorization"] = payload["pilot_authorization"]
            objective_overlay = payload.get("objective", {})
            if set(objective_overlay) - {"activation_decoder_normalization"}:
                raise ValueError("pilot overlay changes forbidden objective fields")
            merged["objective"].update(objective_overlay)
            merged["optimization"].update(optimization_overlay)
            merged["data"].update(data_overlay)
            merged["evaluation_status"] = payload["evaluation_status"]
        else:
            merged["gpu_preflight_authorized"] = payload["gpu_preflight_authorized"]
            merged["preflight_authorization"] = payload["preflight_authorization"]
        merged["runtime"].update(runtime_overlay)
        payload = merged
    return GraniteNormalizedTargetConfig.model_validate(payload)
