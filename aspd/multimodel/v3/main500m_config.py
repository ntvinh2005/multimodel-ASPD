"""Fail-closed ASPD v3.1 three-model 500M preparation contract.

This schema deliberately permits an incomplete third-model slot so engineering work can be
reviewed without inventing a model, revision, tokenizer, hook plan, or license decision.  It
cannot become executable until every scientific and operational approval is explicit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

EXPERIMENT_FAMILY = "aspd_v3_1_three_model_main500m"
LOGICAL_MATRICES = ("q", "k", "v", "o", "mlp_gate", "mlp_up", "mlp_down")
RETAINED_MODELS = {
    "gemma": {
        "model_id": "google/gemma-2-2b-it",
        "revision": "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        "selected_layers": tuple(range(9, 17)),
    },
    "smol": {
        "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "revision": "31b70e2e869a7173562077fd711b654946d38674",
        "selected_layers": tuple(range(8, 16)),
    },
}
APPROVED_THIRD_MODEL = {
    "key": "granite",
    "model_id": "ibm-granite/granite-3.3-2b-instruct",
    "revision": "707f574c62054322f6b5b04b6d075f0a8f05e0f0",
    "tokenizer_id": "ibm-granite/granite-3.3-2b-instruct",
    "tokenizer_revision": "707f574c62054322f6b5b04b6d075f0a8f05e0f0",
    "family": "granite",
    "license": "apache-2.0",
    "access": "public",
    "checkpoint_parameter_count": 2_533_539_840,
    "selected_layers": tuple(range(16, 24)),
    "hidden_size": 2_048,
    "intermediate_size": 8_192,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "layer_prefixes": ("model.layers",),
    "residual_pre_suffix": "input_layernorm",
    "residual_mid_suffix": "post_attention_layernorm",
    "fused_qkv": False,
    "fused_gate_up": False,
    "frozen": True,
}


class Main500MSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    approval_status: Literal[
        "RETAINED_APPROVED", "APPROVED", "PENDING_RESEARCHER_APPROVAL"
    ]
    model_id: str | None = None
    revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    family: str | None = None
    license: str | None = None
    access: str | None = None
    checkpoint_parameter_count: int | None = Field(default=None, gt=0)
    selected_layers: tuple[int, ...] | None = None
    hidden_size: int | None = Field(default=None, gt=0)
    intermediate_size: int | None = Field(default=None, gt=0)
    num_hidden_layers: int | None = Field(default=None, ge=8)
    num_attention_heads: int | None = Field(default=None, gt=0)
    num_key_value_heads: int | None = Field(default=None, gt=0)
    head_dim: int | None = Field(default=None, gt=0)
    layer_prefixes: tuple[str, ...] | None = None
    residual_pre_suffix: str | None = None
    residual_mid_suffix: str | None = None
    fused_qkv: bool | None = None
    fused_gate_up: bool | None = None
    frozen: bool | None = None

    @model_validator(mode="after")
    def validate_source(self) -> Main500MSource:
        identity = " ".join(
            str(value)
            for value in (self.key, self.model_id, self.tokenizer_id, self.family)
            if value is not None
        ).lower()
        if "qwen" in identity:
            raise ValueError("Qwen is forbidden in the ASPD v3.1 main source contract")
        scientific = {
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "family": self.family,
            "license": self.license,
            "access": self.access,
            "checkpoint_parameter_count": self.checkpoint_parameter_count,
            "selected_layers": self.selected_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "layer_prefixes": self.layer_prefixes,
            "residual_pre_suffix": self.residual_pre_suffix,
            "residual_mid_suffix": self.residual_mid_suffix,
            "fused_qkv": self.fused_qkv,
            "fused_gate_up": self.fused_gate_up,
            "frozen": self.frozen,
        }
        if self.approval_status == "PENDING_RESEARCHER_APPROVAL":
            populated = sorted(key for key, value in scientific.items() if value is not None)
            if populated:
                raise ValueError(
                    "pending third-model slot must not invent scientific fields: "
                    + ", ".join(populated)
                )
            return self
        missing = sorted(key for key, value in scientific.items() if value is None)
        if missing:
            raise ValueError("approved source is incomplete: " + ", ".join(missing))
        for name in ("revision", "tokenizer_revision"):
            value = str(getattr(self, name))
            if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{self.key} {name} must be a pinned Git revision")
        assert self.selected_layers is not None
        if len(self.selected_layers) != 8 or self.selected_layers != tuple(
            range(self.selected_layers[0], self.selected_layers[0] + 8)
        ):
            raise ValueError(f"{self.key} must select eight contiguous layers")
        assert self.num_hidden_layers is not None
        if self.selected_layers[-1] >= self.num_hidden_layers:
            raise ValueError(f"{self.key} selected layers exceed the source backbone")
        if self.frozen is not True:
            raise ValueError(f"{self.key} source weights must remain frozen")
        assert self.layer_prefixes is not None
        if not self.layer_prefixes or any(not value.strip() for value in self.layer_prefixes):
            raise ValueError(f"{self.key} physical layer prefixes must be non-empty")
        if not str(self.residual_pre_suffix).strip() or not str(
            self.residual_mid_suffix
        ).strip():
            raise ValueError(f"{self.key} PRE/MID hook suffixes must be non-empty")
        return self

    def _required(self, name: str) -> int:
        value = getattr(self, name)
        if value is None:
            raise RuntimeError(f"{self.key} is not approved; {name} is unresolved")
        return int(value)

    @property
    def q_out(self) -> int:
        return self._required("num_attention_heads") * self._required("head_dim")

    @property
    def kv_out(self) -> int:
        return self._required("num_key_value_heads") * self._required("head_dim")


class Main500MArchitecture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_model_count: Literal[3]
    model_order: tuple[str, str, str]
    selected_layers_per_model: Literal[8]
    logical_matrices: tuple[str, ...]
    dictionary_size: Literal[8192]
    batch_topk_k: Literal[32]
    pool_tokens: Literal[2048]
    pools_per_step: Literal[4]
    top_k_aux: Literal[2048]
    auxk_coefficient: Literal[0.03125]
    gate_grouping: Literal["pre_mid"]
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
    encoder_mixer: Literal["hierarchical_transformer"]
    dropout: Literal[0]
    activation: Literal["GELU"]

    @model_validator(mode="after")
    def validate_architecture(self) -> Main500MArchitecture:
        if len(set(self.model_order)) != 3:
            raise ValueError("main 500M model order must contain three distinct identities")
        if self.model_order[:2] != ("gemma", "smol"):
            raise ValueError("retained source order must begin with Gemma then SmolLM2")
        if self.logical_matrices != LOGICAL_MATRICES:
            raise ValueError("main 500M must retain all seven logical matrices")
        if self.pool_tokens * self.pools_per_step != 8192:
            raise ValueError("one optimizer step must contain exactly 8,192 accepted anchors")
        return self


class Main500MObjective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal_code: Literal["binary_support"]
    activation_code: Literal["continuous"]
    internal_gradient_route: Literal["rank_one_only"]
    activation_gradient_route: Literal[
        "encoder_packing_adapter_readout_and_activation_decoder"
    ]
    internal_loss: Literal["balanced_pre_mid_mean_fvu"]
    activation_loss: Literal["balanced_pre_mid_mean_full_fvu"]
    activation_objective_mode: Literal["full_only"]
    auxk_loss: Literal["balanced_pre_mid_dead_feature_revival"]
    auxk_score_source: Literal["post_relu"]
    auxk_softplus_temperature: None
    internal_coefficient: Literal[1.0]
    activation_coefficient: Literal[1.0]
    fvu_reduction_dtype: Literal["float32"]
    activation_target_normalization: Literal["per_anchor_l2"]
    activation_target_epsilon: Literal[1e-6]
    activation_decoder_normalization: Literal["none"]
    activation_reconstruction_scale: Literal["none"]
    gate_logit_normalization: Literal["per_anchor_feature_zscore"]
    gate_logit_normalization_epsilon: Literal[1e-5]
    native_bias: Literal["preserve"]


class Main500MOptimization(BaseModel):
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


class Main500MData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: Literal["monology/pile-uncopyrighted"]
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sequence_length: Literal[512]
    effective_batch_anchors: Literal[8192]
    target_accepted_anchor_exposures: Literal[500000000]
    planned_accepted_anchor_exposures: Literal[500006912]
    optimizer_steps: Literal[61036]
    repeated_anchor_policy: Literal["forbidden"]
    split_unit: Literal["document"]
    split_seed: Literal[0]
    split_modulus: Literal[1000]
    validation_buckets: tuple[int, ...]
    test_buckets: tuple[int, ...]
    canonical_text_policy: Literal[
        "exact_fast_tokenizer_offsets_and_exact_prefix_roundtrip_v2"
    ]
    alignment_acceptance: Literal["intersection_of_exact_canonical_text_boundaries"]

    @model_validator(mode="after")
    def validate_budget(self) -> Main500MData:
        if self.effective_batch_anchors * self.optimizer_steps != (
            self.planned_accepted_anchor_exposures
        ):
            raise ValueError("accepted-anchor exposure budget is inconsistent")
        if self.planned_accepted_anchor_exposures < self.target_accepted_anchor_exposures:
            raise ValueError("planned exposure count must meet the 500M target")
        if self.validation_buckets != tuple(range(10)):
            raise ValueError("validation buckets must remain 0..9")
        if self.test_buckets != tuple(range(10, 20)):
            raise ValueError("test buckets must remain 10..19")
        return self


class Main500MMonitoring(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monitor_every_steps: int = Field(gt=0)
    minimum_positive_candidates_per_anchor: float = Field(gt=0)
    minimum_effective_dictionary_size: float = Field(gt=0)
    maximum_top_1_percent_share: float = Field(gt=0, le=1)
    maximum_top_10_percent_share: float = Field(gt=0, le=1)
    maximum_recently_dead_fraction: float = Field(gt=0, le=1)
    maximum_activation_fvu: float = Field(gt=0)
    maximum_internal_fvu: float = Field(gt=0)
    maximum_model_loss_ratio: float = Field(gt=1)
    expected_gate_groups: Literal[6]
    expected_activation_fvu_metrics: Literal[48]
    expected_internal_fvu_metrics: Literal[168]
    hard_abort_on_missing_diagnostics: Literal[True]
    hard_abort_on_repeated_exposure: Literal[True]
    hard_abort_on_nonfinite: Literal[True]
    hard_abort_on_checkpoint_failure: Literal[True]
    hard_abort_on_storage_reserve_breach: Literal[True]
    storage_reserve_bytes: int = Field(gt=0)
    sustained_policy_version: Literal["v1"]
    sustained_gate_failure_diagnostics: int = Field(ge=2)
    sustained_reconstruction_failure_diagnostics: int = Field(ge=2)
    reconstruction_pause_minimum_exposures: int = Field(ge=0)
    milestone_policy_version: Literal["v1"]
    require_researcher_continuation: Literal[True]
    staged_milestone_exposures: tuple[int, ...]

    @model_validator(mode="after")
    def validate_milestones(self) -> Main500MMonitoring:
        expected = (
            10_002_432,
            25_001_984,
            50_003_968,
            100_007_936,
            250_003_456,
            500_006_912,
        )
        if self.staged_milestone_exposures != expected:
            raise ValueError("main 500M staged exposure milestones changed")
        if self.reconstruction_pause_minimum_exposures != expected[0]:
            raise ValueError("reconstruction escalation must begin at the first review milestone")
        if any(exposures % 8192 for exposures in self.staged_milestone_exposures):
            raise ValueError("every staged milestone must fall on an optimizer-step boundary")
        return self


class Main500MRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["model_owned"]
    world_size: Literal[3]
    checkpoint_every_steps: Literal[1000]
    keep_complete_checkpoints: Literal[2]
    checkpoint_recovery_databases: tuple[
        Literal["exposure_ledger.sqlite"], Literal["stream_dedup.sqlite"]
    ]
    activation_backward_chunk: int = Field(gt=0, le=8)
    matrix_backward_chunk: int = Field(gt=0, le=56)
    sparse_backend: Literal["dense"]
    early_exit_after_selected_layers: Literal[True]
    data_prefetch_depth: Literal[1]
    ledger_anchor_storage: Literal["exact_compact_v1"]
    output_root: str
    checkpoint_root: str
    startup_storage_required_bytes: Literal[1_099_511_627_776]
    dataset_manifest: str
    alignment_manifest: str
    continuation_authorization_root: str
    preflight_output_root: str
    preflight_evidence: str
    max_preflight_steps: int = Field(gt=0, le=20)
    max_preflight_minutes: int = Field(gt=0, le=45)

    @model_validator(mode="after")
    def validate_namespace(self) -> Main500MRuntime:
        paths = (
            self.output_root,
            self.checkpoint_root,
            self.dataset_manifest,
            self.alignment_manifest,
            self.continuation_authorization_root,
            self.preflight_output_root,
            self.preflight_evidence,
        )
        if any("aspd_v3/main500m_preparation_v1" not in value for value in paths):
            raise ValueError("main 500M paths must use the isolated preparation namespace")
        if any(value.startswith(("~", "/home/")) for value in paths):
            raise ValueError("large ASPD artifacts must not default to HOME")
        if Path(self.checkpoint_root) != Path(self.output_root):
            raise ValueError(
                "the current runner stores checkpoints inside each run namespace; "
                "checkpoint_root must equal output_root"
            )
        return self


class Main500MBoundedPreflightAuthorization(BaseModel):
    """Researcher authorization for one bounded engineering preflight only."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["PENDING_RESEARCHER_APPROVAL", "AUTHORIZED"]
    optimizer_steps: int | None = Field(default=None, ge=3, le=20)
    output_namespace: str | None = None
    authorization_reference: str | None = None

    @model_validator(mode="after")
    def validate_authorization(self) -> Main500MBoundedPreflightAuthorization:
        evidence = (
            self.optimizer_steps,
            self.output_namespace,
            self.authorization_reference,
        )
        if self.status == "PENDING_RESEARCHER_APPROVAL":
            if any(value is not None for value in evidence):
                raise ValueError(
                    "pending bounded preflight must not contain authorization evidence"
                )
        elif any(value is None or not str(value).strip() for value in evidence):
            raise ValueError(
                "authorized bounded preflight requires an exact step budget, output "
                "namespace, and authorization reference"
            )
        return self


class Main500MGPUPreflightEvidence(BaseModel):
    """Pinned evidence consumed by main authorization after preflight completes."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["PENDING_MODEL_THREE", "PASSED"]
    artifact_path: str | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    execution_contract_fingerprint: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    source_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    optimizer_steps: int | None = Field(default=None, ge=3, le=20)

    @model_validator(mode="after")
    def validate_evidence(self) -> Main500MGPUPreflightEvidence:
        evidence = (
            self.artifact_path,
            self.artifact_sha256,
            self.execution_contract_fingerprint,
            self.source_commit,
            self.optimizer_steps,
        )
        if self.status == "PENDING_MODEL_THREE":
            if any(value is not None for value in evidence):
                raise ValueError("pending three-model preflight must not contain PASS evidence")
        elif any(value is None or not str(value).strip() for value in evidence):
            raise ValueError(
                "passed three-model preflight requires a pinned artifact, contract "
                "fingerprint, and optimizer-step count"
            )
        return self


class HumanInterpretabilityApproval(BaseModel):
    """Immutable researcher-decision gate owned by the main-training manifest."""

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "PENDING_RESEARCHER_RESPONSES",
        "ANALYSIS_COMPLETE_PENDING_RESEARCHER_DECISION",
        "APPROVED_HUMAN_EVIDENCE",
        "APPROVED_SEMANTIC_UNCERTAINTY_ACCEPTED",
        "REJECTED",
    ]
    artifact_path: str | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    researcher_decision: str | None = None
    scientific_limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_gate(self) -> HumanInterpretabilityApproval:
        artifact = (self.artifact_path, self.artifact_sha256)
        if any(value is None or not str(value).strip() for value in artifact):
            raise ValueError("human-interpretability status requires a pinned artifact")
        if not self.scientific_limitations or any(
            not limitation.strip() for limitation in self.scientific_limitations
        ):
            raise ValueError("human-interpretability limitations must be explicit")
        decision_required = self.status in {
            "APPROVED_HUMAN_EVIDENCE",
            "APPROVED_SEMANTIC_UNCERTAINTY_ACCEPTED",
            "REJECTED",
        }
        if decision_required and not str(self.researcher_decision or "").strip():
            raise ValueError(
                "final human-interpretability status requires a researcher decision"
            )
        if not decision_required and self.researcher_decision is not None:
            raise ValueError("pending human-interpretability status cannot claim a decision")
        return self

    def permits_main_authorization(self) -> bool:
        return self.status in {
            "APPROVED_HUMAN_EVIDENCE",
            "APPROVED_SEMANTIC_UNCERTAINTY_ACCEPTED",
        }


class Main500MApprovals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    third_model: Literal["PENDING_RESEARCHER_APPROVAL", "APPROVED"]
    three_way_alignment: Literal["PENDING_THIRD_TOKENIZER_VALIDATION", "APPROVED"]
    three_way_alignment_artifact: str | None = None
    three_way_alignment_artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    human_interpretability: HumanInterpretabilityApproval
    exact_resume: Literal[
        "PENDING_ENGINEERING_VALIDATION",
        "SYNTHETIC_VALIDATED_REAL_THREE_MODEL_PENDING",
        "VALIDATED_THREE_MODEL_GPU",
    ]
    exact_resume_artifact: str | None = None
    exact_resume_artifact_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    exact_resume_source_commit: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{40}$"
    )
    exact_resume_contract_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    storage_reservation: Literal["STORAGE_BLOCKED", "APPROVED"]
    monitoring_policy: Literal["PENDING_RESEARCHER_APPROVAL", "APPROVED"]
    milestone_policy: Literal["PENDING_RESEARCHER_APPROVAL", "APPROVED"]
    bounded_preflight: Main500MBoundedPreflightAuthorization
    three_model_gpu_preflight: Main500MGPUPreflightEvidence
    full_training: Literal["MAIN_500M_NOT_AUTHORIZED", "AUTHORIZED"]


class V31Main500MConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    experiment_family: Literal["aspd_v3_1_three_model_main500m"]
    scientific_status: Literal["MAIN_500M_NOT_AUTHORIZED", "RESEARCHER_APPROVED"]
    full_training_authorized: bool
    models: tuple[Main500MSource, Main500MSource, Main500MSource]
    architecture: Main500MArchitecture
    objective: Main500MObjective
    optimization: Main500MOptimization
    data: Main500MData
    monitoring: Main500MMonitoring
    runtime: Main500MRuntime
    approvals: Main500MApprovals
    evaluation_status: Literal[
        "HUMAN_INTERPRETABILITY_PACKAGE_COMPLETE_RESPONSES_PENDING",
        "HUMAN_INTERPRETABILITY_ANALYSIS_COMPLETE_DECISION_PENDING",
        "HUMAN_INTERPRETABILITY_APPROVED",
        "SEMANTIC_UNCERTAINTY_EXPLICITLY_ACCEPTED",
        "HUMAN_INTERPRETABILITY_REJECTED",
    ]

    @model_validator(mode="after")
    def validate_contract(self) -> V31Main500MConfig:
        if tuple(model.key for model in self.models) != self.architecture.model_order:
            raise ValueError("source models and architecture model_order differ")
        for model in self.models[:2]:
            expected = RETAINED_MODELS[model.key]
            observed = {
                "model_id": model.model_id,
                "revision": model.revision,
                "selected_layers": model.selected_layers,
            }
            if observed != expected or model.approval_status != "RETAINED_APPROVED":
                raise ValueError(f"retained {model.key} source contract changed")
        third = self.models[2]
        if third.approval_status == "PENDING_RESEARCHER_APPROVAL":
            if self.approvals.third_model != "PENDING_RESEARCHER_APPROVAL":
                raise ValueError("third-model approval cannot precede a complete source contract")
        elif third.approval_status != "APPROVED" or self.approvals.third_model != "APPROVED":
            raise ValueError("the third model requires explicit researcher approval")
        else:
            observed_third = {
                key: getattr(third, key) for key in APPROVED_THIRD_MODEL
            }
            if observed_third != APPROVED_THIRD_MODEL:
                raise ValueError("approved Granite model-3 contract changed")
        alignment_evidence = (
            self.approvals.three_way_alignment_artifact,
            self.approvals.three_way_alignment_artifact_sha256,
        )
        if self.approvals.three_way_alignment == "APPROVED":
            if any(value is None or not str(value).strip() for value in alignment_evidence):
                raise ValueError(
                    "approved three-way alignment requires a pinned validation artifact"
                )
        elif any(value is not None for value in alignment_evidence):
            raise ValueError(
                "pending three-way alignment cannot contain approval evidence"
            )
        expected_evaluation_status = {
            "PENDING_RESEARCHER_RESPONSES": (
                "HUMAN_INTERPRETABILITY_PACKAGE_COMPLETE_RESPONSES_PENDING"
            ),
            "ANALYSIS_COMPLETE_PENDING_RESEARCHER_DECISION": (
                "HUMAN_INTERPRETABILITY_ANALYSIS_COMPLETE_DECISION_PENDING"
            ),
            "APPROVED_HUMAN_EVIDENCE": "HUMAN_INTERPRETABILITY_APPROVED",
            "APPROVED_SEMANTIC_UNCERTAINTY_ACCEPTED": (
                "SEMANTIC_UNCERTAINTY_EXPLICITLY_ACCEPTED"
            ),
            "REJECTED": "HUMAN_INTERPRETABILITY_REJECTED",
        }[self.approvals.human_interpretability.status]
        if self.evaluation_status != expected_evaluation_status:
            raise ValueError(
                "human-interpretability approval and evaluation status disagree"
            )
        preflight = self.approvals.bounded_preflight
        if preflight.status == "AUTHORIZED":
            if preflight.output_namespace != self.runtime.preflight_output_root:
                raise ValueError(
                    "bounded-preflight authorization must pin the configured output namespace"
                )
            if Path(self.runtime.preflight_evidence).parent != Path(
                self.runtime.preflight_output_root
            ):
                raise ValueError(
                    "preflight evidence must be inside the dedicated preflight namespace"
                )
        if self.full_training_authorized:
            self.require_main_authorized()
        elif self.scientific_status != "MAIN_500M_NOT_AUTHORIZED":
            raise ValueError("unauthorized config must remain MAIN_500M_NOT_AUTHORIZED")
        return self

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def execution_contract_fingerprint(self) -> str:
        """Hash immutable scientific/runtime inputs, excluding mutable approvals."""

        payload = self.payload()
        for key in (
            "scientific_status",
            "full_training_authorized",
            "approvals",
            "evaluation_status",
        ):
            payload.pop(key)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def alignment_contract_fingerprint(self) -> str:
        """Hash only inputs that can change canonical three-tokenizer alignment."""

        payload = {
            "models": [
                {
                    "key": model.key,
                    "tokenizer_id": model.tokenizer_id,
                    "tokenizer_revision": model.tokenizer_revision,
                }
                for model in self.models
            ],
            "model_order": list(self.architecture.model_order),
            "dataset_id": self.data.dataset_id,
            "dataset_revision": self.data.dataset_revision,
            "sequence_length": self.data.sequence_length,
            "split_unit": self.data.split_unit,
            "split_seed": self.data.split_seed,
            "split_modulus": self.data.split_modulus,
            "validation_buckets": list(self.data.validation_buckets),
            "test_buckets": list(self.data.test_buckets),
            "canonical_text_policy": self.data.canonical_text_policy,
            "alignment_acceptance": self.data.alignment_acceptance,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def readiness_blockers(self) -> list[str]:
        blockers: list[str] = []
        if self.models[2].approval_status != "APPROVED":
            blockers.append("MODEL_THREE_BLOCKED")
        if self.approvals.three_way_alignment != "APPROVED":
            blockers.append("ALIGNMENT_BLOCKED")
        exact_resume_evidence = (
            self.approvals.exact_resume_artifact,
            self.approvals.exact_resume_artifact_sha256,
            self.approvals.exact_resume_source_commit,
            self.approvals.exact_resume_contract_sha256,
        )
        if self.approvals.exact_resume != "VALIDATED_THREE_MODEL_GPU" or any(
            value is None or not str(value).strip()
            for value in exact_resume_evidence
        ):
            blockers.append("RESUME_BLOCKED")
        if self.approvals.storage_reservation != "APPROVED":
            blockers.append("STORAGE_BLOCKED")
        if self.approvals.monitoring_policy != "APPROVED":
            blockers.append("MONITORING_POLICY_APPROVAL_PENDING")
        if self.approvals.milestone_policy != "APPROVED":
            blockers.append("MILESTONE_POLICY_APPROVAL_PENDING")
        if not self.approvals.human_interpretability.permits_main_authorization():
            blockers.append("HUMAN_INTERPRETABILITY_APPROVAL_PENDING")
        if self.approvals.three_model_gpu_preflight.status != "PASSED":
            blockers.append("THREE_MODEL_GPU_PREFLIGHT_PENDING")
        if (
            self.approvals.full_training != "AUTHORIZED"
            or not self.full_training_authorized
        ):
            blockers.append("MAIN_500M_NOT_AUTHORIZED")
        return blockers

    def require_preflight_authorized(
        self,
        *,
        optimizer_steps: int,
        run_dir: str | Path,
    ) -> None:
        """Authorize one exact, bounded preflight without requiring prior PASS evidence."""

        blockers: list[str] = []
        if self.models[2].approval_status != "APPROVED":
            blockers.append("MODEL_THREE_BLOCKED")
        if self.approvals.third_model != "APPROVED":
            blockers.append("MODEL_THREE_APPROVAL_PENDING")
        if self.approvals.three_way_alignment != "APPROVED":
            blockers.append("ALIGNMENT_BLOCKED")
        authorization = self.approvals.bounded_preflight
        if authorization.status != "AUTHORIZED":
            blockers.append("BOUNDED_PREFLIGHT_NOT_AUTHORIZED")
        if blockers:
            raise RuntimeError(
                "ASPD v3.1 bounded preflight is fail-closed: " + ", ".join(blockers)
            )
        self.validate_three_way_alignment_evidence()
        assert authorization.optimizer_steps is not None
        assert authorization.output_namespace is not None
        if optimizer_steps != authorization.optimizer_steps:
            raise RuntimeError(
                "ASPD v3.1 bounded preflight step budget differs from its authorization"
            )
        if optimizer_steps > min(20, self.runtime.max_preflight_steps):
            raise RuntimeError("ASPD v3.1 bounded preflight exceeds 20 optimizer steps")
        run_path = Path(run_dir)
        if run_path != Path(authorization.output_namespace):
            raise RuntimeError(
                "ASPD v3.1 bounded preflight must use its dedicated output namespace"
            )
        if run_path.exists():
            raise RuntimeError(
                "ASPD v3.1 bounded preflight refuses to overwrite an existing namespace"
            )

    def validate_preflight_evidence(self) -> dict[str, Any]:
        evidence = self.approvals.three_model_gpu_preflight
        if evidence.status != "PASSED":
            raise RuntimeError("THREE_MODEL_GPU_PREFLIGHT_PENDING")
        assert evidence.artifact_path is not None
        assert evidence.artifact_sha256 is not None
        assert evidence.execution_contract_fingerprint is not None
        assert evidence.source_commit is not None
        assert evidence.optimizer_steps is not None
        artifact = Path(evidence.artifact_path)
        if artifact != Path(self.runtime.preflight_evidence) or not artifact.is_file():
            raise RuntimeError("THREE_MODEL_GPU_PREFLIGHT_ARTIFACT_MISSING")
        observed_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if observed_sha256 != evidence.artifact_sha256:
            raise RuntimeError("THREE_MODEL_GPU_PREFLIGHT_ARTIFACT_HASH_MISMATCH")
        report = json.loads(artifact.read_text(encoding="utf-8"))
        expected = {
            "status": "PASS",
            "mode": "preflight",
            "strategy": "model_owned",
            "world_size": 3,
            "logical_target_count": 168,
            "optimizer_steps": evidence.optimizer_steps,
            "execution_contract_fingerprint": self.execution_contract_fingerprint(),
            "git_sha": evidence.source_commit,
            "dcp_save_reload_resume_update": "PASS",
            "checkpoint_parameter_and_adam_restore_probe": "PASS",
        }
        if self.approvals.exact_resume == "VALIDATED_THREE_MODEL_GPU":
            expected["uninterrupted_vs_resumed_next_update"] = "BITWISE_EXACT"
        mismatch = {
            key: {"expected": value, "observed": report.get(key)}
            for key, value in expected.items()
            if report.get(key) != value
        }
        if evidence.execution_contract_fingerprint != self.execution_contract_fingerprint():
            mismatch["approval.execution_contract_fingerprint"] = {
                "expected": self.execution_contract_fingerprint(),
                "observed": evidence.execution_contract_fingerprint,
            }
        source_correctness = report.get("source_correctness")
        if not isinstance(source_correctness, list) or len(source_correctness) != 3:
            mismatch["source_correctness"] = "must contain exactly three source reports"
        else:
            for index, source in enumerate(source_correctness):
                if not isinstance(source, dict) or any(
                    source.get(key) != value
                    for key, value in {
                        "status": "PASS",
                        "targets": 56,
                        "frozen_parameters": "PASS",
                    }.items()
                ):
                    mismatch[f"source_correctness[{index}]"] = source
        if self.approvals.exact_resume == "VALIDATED_THREE_MODEL_GPU":
            exact_data_replay = report.get("exact_data_replay")
            if not isinstance(exact_data_replay, dict) or any(
                exact_data_replay.get(key) != value
                for key, value in {
                    "step": evidence.optimizer_steps,
                    "accepted_exposures": evidence.optimizer_steps * 8192,
                    "repeated_exposures": 0,
                }.items()
            ):
                mismatch["exact_data_replay"] = exact_data_replay
            source_immutability = report.get("source_state_immutability_by_rank")
            if not isinstance(source_immutability, list) or len(source_immutability) != 3:
                mismatch["source_state_immutability_by_rank"] = source_immutability
            elif any(
                not isinstance(item, dict)
                or item.get("status") != "PASS"
                or item.get("initial_sha256") != item.get("final_sha256")
                for item in source_immutability
            ):
                mismatch["source_state_immutability_by_rank"] = source_immutability
        checkpoint_manifest = report.get("checkpoint_manifest")
        if not isinstance(checkpoint_manifest, dict) or any(
            checkpoint_manifest.get(key) != value
            for key, value in {"status": "COMPLETE", "world_size": 3}.items()
        ):
            mismatch["checkpoint_manifest"] = checkpoint_manifest
        if mismatch:
            raise RuntimeError(f"THREE_MODEL_GPU_PREFLIGHT_EVIDENCE_INVALID: {mismatch}")
        return report

    def validate_exact_resume_evidence(self) -> dict[str, Any]:
        approvals = self.approvals
        if approvals.exact_resume != "VALIDATED_THREE_MODEL_GPU":
            raise RuntimeError("RESUME_BLOCKED")
        assert approvals.exact_resume_artifact is not None
        assert approvals.exact_resume_artifact_sha256 is not None
        assert approvals.exact_resume_source_commit is not None
        assert approvals.exact_resume_contract_sha256 is not None
        artifact = Path(approvals.exact_resume_artifact)
        if not artifact.is_file():
            raise RuntimeError("EXACT_RESUME_ARTIFACT_MISSING")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != (
            approvals.exact_resume_artifact_sha256
        ):
            raise RuntimeError("EXACT_RESUME_ARTIFACT_HASH_MISMATCH")
        report = json.loads(artifact.read_text(encoding="utf-8"))
        expected = {
            "status": "PASS",
            "mode": "resume_validation",
            "strategy": "model_owned",
            "git_sha": approvals.exact_resume_source_commit,
            "world_size": 3,
            "logical_target_count": 168,
            "optimizer_steps": 4,
            "accepted_exposures": 32_768,
            "uninterrupted_vs_resumed_next_update": "BITWISE_EXACT",
            "dcp_save_reload_resume_update": "PASS",
            "resume_validation_contract_sha256": (
                approvals.exact_resume_contract_sha256
            ),
        }
        mismatch = {
            key: {"expected": value, "observed": report.get(key)}
            for key, value in expected.items()
            if report.get(key) != value
        }
        reference = report.get("uninterrupted_state_sha256_by_rank")
        resumed = report.get("resumed_state_sha256_by_rank")
        if (
            not isinstance(reference, list)
            or len(reference) != 3
            or resumed != reference
        ):
            mismatch["state_sha256_by_rank"] = {
                "uninterrupted": reference,
                "resumed": resumed,
            }
        data = report.get("exact_data_replay")
        if not isinstance(data, dict) or any(
            data.get(key) != value
            for key, value in {
                "step": 4,
                "accepted_exposures": 32_768,
                "unique_anchors": 32_768,
                "repeated_exposures": 0,
            }.items()
        ):
            mismatch["exact_data_replay"] = data
        source = report.get("source_state_immutability_by_rank")
        if not isinstance(source, list) or len(source) != 3 or any(
            not isinstance(item, dict)
            or item.get("status") != "PASS"
            or item.get("initial_sha256") != item.get("final_sha256")
            for item in source
        ):
            mismatch["source_state_immutability_by_rank"] = source
        if mismatch:
            raise RuntimeError(f"EXACT_RESUME_EVIDENCE_INVALID: {mismatch}")
        return report

    def validate_three_way_alignment_evidence(self) -> dict[str, Any]:
        if self.approvals.three_way_alignment != "APPROVED":
            raise RuntimeError("ALIGNMENT_BLOCKED")
        artifact_path = self.approvals.three_way_alignment_artifact
        expected_sha256 = self.approvals.three_way_alignment_artifact_sha256
        assert artifact_path is not None
        assert expected_sha256 is not None
        artifact = Path(artifact_path)
        if not artifact.is_file():
            raise RuntimeError("THREE_WAY_ALIGNMENT_ARTIFACT_MISSING")
        observed_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if observed_sha256 != expected_sha256:
            raise RuntimeError("THREE_WAY_ALIGNMENT_ARTIFACT_HASH_MISMATCH")
        report = json.loads(artifact.read_text(encoding="utf-8"))
        expected_tokenizers = [
            {
                "model_key": model.key,
                "id": model.tokenizer_id,
                "revision": model.tokenizer_revision,
            }
            for model in self.models
        ]
        observed_tokenizers = [
            {
                "model_key": row.get("model_key"),
                "id": row.get("id"),
                "revision": row.get("revision"),
            }
            for row in report.get("tokenizers", [])
        ]
        expected = {
            "status": "PASS",
            "alignment_contract_fingerprint": self.alignment_contract_fingerprint(),
            "alignment_policy": self.data.canonical_text_policy,
            "alignment_acceptance": self.data.alignment_acceptance,
            "model_order": list(self.architecture.model_order),
            "tokenizers": expected_tokenizers,
            "audit.boundary_invariants": "PASS",
            "audit.within_process_repeatability": "PASS",
            "audit.dataset_stream_prefix_replay": "PASS",
            "selection_bias.status": "PASS",
            "capacity.status": "PASS",
        }
        observed = {
            "status": report.get("status"),
            "alignment_contract_fingerprint": report.get(
                "alignment_contract_fingerprint"
            ),
            "alignment_policy": report.get("alignment_policy"),
            "alignment_acceptance": report.get("alignment_acceptance"),
            "model_order": report.get("model_order"),
            "tokenizers": observed_tokenizers,
            "audit.boundary_invariants": report.get("audit", {}).get(
                "boundary_invariants"
            ),
            "audit.within_process_repeatability": report.get("audit", {}).get(
                "within_process_repeatability"
            ),
            "audit.dataset_stream_prefix_replay": report.get("audit", {}).get(
                "dataset_stream_prefix_replay"
            ),
            "selection_bias.status": report.get("selection_bias", {}).get("status"),
            "capacity.status": report.get("capacity", {}).get("status"),
        }
        mismatches = {
            key: {"expected": expected[key], "observed": observed[key]}
            for key in expected
            if observed[key] != expected[key]
        }
        if mismatches:
            raise RuntimeError(
                f"THREE_WAY_ALIGNMENT_EVIDENCE_INVALID: {mismatches}"
            )
        return report

    def validate_human_interpretability_artifact(self) -> dict[str, Any]:
        human = self.approvals.human_interpretability
        if not human.permits_main_authorization():
            raise RuntimeError("HUMAN_INTERPRETABILITY_APPROVAL_PENDING")
        assert human.artifact_path is not None and human.artifact_sha256 is not None
        artifact = Path(human.artifact_path)
        if not artifact.is_file():
            raise RuntimeError("HUMAN_INTERPRETABILITY_ARTIFACT_MISSING")
        observed = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if observed != human.artifact_sha256:
            raise RuntimeError("HUMAN_INTERPRETABILITY_ARTIFACT_HASH_MISMATCH")
        decision = json.loads(artifact.read_text(encoding="utf-8"))
        expected_decision = {
            "APPROVED_HUMAN_EVIDENCE": "APPROVE_HUMAN_EVIDENCE",
            "APPROVED_SEMANTIC_UNCERTAINTY_ACCEPTED": "ACCEPT_SEMANTIC_UNCERTAINTY",
        }[human.status]
        expected = {
            "schema_version": 1,
            "status": "HUMAN_INTERPRETABILITY_RESEARCHER_DECISION",
            "decision": expected_decision,
            "researcher_decision": human.researcher_decision,
            "scientific_limitations": list(human.scientific_limitations),
        }
        mismatch = {
            key: {"expected": value, "observed": decision.get(key)}
            for key, value in expected.items()
            if decision.get(key) != value
        }
        evidence_path = decision.get("evaluation_artifact_path")
        evidence_sha256 = decision.get("evaluation_artifact_sha256")
        if not isinstance(evidence_path, str) or not isinstance(evidence_sha256, str):
            mismatch["evaluation_artifact"] = "path and SHA-256 are required"
        else:
            evidence = Path(evidence_path)
            if not evidence.is_file():
                mismatch["evaluation_artifact"] = "missing"
            elif hashlib.sha256(evidence.read_bytes()).hexdigest() != evidence_sha256:
                mismatch["evaluation_artifact"] = "SHA-256 mismatch"
            elif human.status == "APPROVED_HUMAN_EVIDENCE":
                evaluation = json.loads(evidence.read_text(encoding="utf-8"))
                if evaluation.get("status") != "HUMAN_INTERPRETABILITY_ANALYSIS_COMPLETE":
                    mismatch["evaluation_artifact.status"] = {
                        "expected": "HUMAN_INTERPRETABILITY_ANALYSIS_COMPLETE",
                        "observed": evaluation.get("status"),
                    }
        if mismatch:
            raise RuntimeError(f"HUMAN_INTERPRETABILITY_DECISION_INVALID: {mismatch}")
        return decision

    def require_main_authorized(self) -> None:
        blockers = self.readiness_blockers()
        if blockers:
            raise RuntimeError(
                "ASPD v3.1 main 500M execution is fail-closed: " + ", ".join(blockers)
            )
        self.validate_three_way_alignment_evidence()
        self.validate_preflight_evidence()
        self.validate_exact_resume_evidence()
        self.validate_human_interpretability_artifact()

    def require_execution_authorized(self) -> None:
        """Backward-compatible alias for full main authorization only."""

        self.require_main_authorized()

    def structural_report(self) -> dict[str, Any]:
        approved_models = sum(
            model.approval_status in {"RETAINED_APPROVED", "APPROVED"}
            for model in self.models
        )
        return {
            "status": "DRAFT_FAIL_CLOSED",
            "config_fingerprint": self.fingerprint(),
            "required_model_count": 3,
            "logical_target_slots": 3 * 8 * 7,
            "approved_physical_targets": approved_models * 8 * 7,
            "pending_physical_targets": (3 - approved_models) * 8 * 7,
            "optimizer_steps": self.data.optimizer_steps,
            "target_accepted_anchor_exposures": self.data.target_accepted_anchor_exposures,
            "planned_accepted_anchor_exposures": self.data.planned_accepted_anchor_exposures,
            "blockers": self.readiness_blockers(),
        }


def load_main500m_config(path: str | Path) -> V31Main500MConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ASPD v3.1 main 500M config must be a mapping")
    return V31Main500MConfig.model_validate(payload)
