"""Fail-closed contract for the two-model gate-collapse recovery pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from aspd.multimodel.v3.two_model_config import (
    LOGICAL_MATRICES_TWO,
    MODEL_ORDER_TWO,
    TwoModelData,
    TwoModelOptimization,
    TwoModelSource,
)
from aspd.multimodel.v3.two_model_matryoshka_config import (
    MatryoshkaArchitecture,
    MatryoshkaObjective,
    MatryoshkaRuntime,
)

EXPERIMENT_FAMILY = "aspd_v3_gemma_smol_gate_collapse_recovery_v1"


class GateCollapseObjective(MatryoshkaObjective):
    activation_loss: Literal["balanced_pre_mid_mean_full_fvu"]
    activation_objective_mode: Literal["full_only"]
    gate_logit_normalization: Literal["none", "per_anchor_feature_zscore"]
    gate_logit_normalization_epsilon: Literal[1e-5]


class GateCollapseRuntime(MatryoshkaRuntime):
    checkpoint_every_steps: Literal[1221]
    keep_complete_checkpoints: Literal[2]
    diagnostic_parameter_steps: tuple[
        Literal[0],
        Literal[64],
        Literal[128],
        Literal[256],
        Literal[512],
        Literal[768],
        Literal[1024],
        Literal[1220],
    ]


class GateCollapseRecoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    experiment_family: Literal["aspd_v3_gemma_smol_gate_collapse_recovery_v1"]
    scientific_status: Literal["RESEARCHER_AUTHORIZED_GATE_COLLAPSE_RECOVERY_PILOT"]
    authorized_by: Literal["researcher"]
    authorized_on: Literal["2026-11-08"]
    full_training_authorized: Literal[False]
    short_pilot_authorized: Literal[True]
    gpu_preflight_authorized: Literal[True]
    arm_id: Literal["full_only_control", "per_anchor_zscore"]
    models: tuple[TwoModelSource, TwoModelSource]
    architecture: MatryoshkaArchitecture
    objective: GateCollapseObjective
    optimization: TwoModelOptimization
    data: TwoModelData
    runtime: GateCollapseRuntime
    evaluation_status: Literal["RESEARCHER_AUTHORIZED_MATCHED_HELDOUT_EVALUATION"]

    @property
    def arm(self) -> str:
        return "normalized_target"

    @model_validator(mode="after")
    def validate_contract(self) -> GateCollapseRecoveryConfig:
        if tuple(model.key for model in self.models) != MODEL_ORDER_TWO:
            raise ValueError(f"model order must be {MODEL_ORDER_TWO}")
        expected_models = {
            "gemma": (
                "google/gemma-2-2b-it",
                "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
                tuple(range(9, 17)),
            ),
            "smol": (
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "31b70e2e869a7173562077fd711b654946d38674",
                tuple(range(8, 16)),
            ),
        }
        for model in self.models:
            observed = (model.model_id, model.revision, model.selected_layers)
            if observed != expected_models[model.key]:
                raise ValueError(f"source registry mismatch for {model.key}")
            if "qwen" in model.model_id.lower() or "qwen" in model.tokenizer_id.lower():
                raise ValueError("Qwen is forbidden in the gate-collapse recovery pilot")
            if model.model_id != model.tokenizer_id or model.revision != model.tokenizer_revision:
                raise ValueError("model and tokenizer revisions must be identical and pinned")
        if self.architecture.model_order != MODEL_ORDER_TWO:
            raise ValueError("two-model architecture order changed")
        if self.architecture.logical_matrices != LOGICAL_MATRICES_TWO:
            raise ValueError("logical matrix coverage changed")
        expected_normalization = (
            "none" if self.arm_id == "full_only_control" else "per_anchor_feature_zscore"
        )
        if self.objective.gate_logit_normalization != expected_normalization:
            raise ValueError("arm and gate-logit normalization differ")
        namespace = (
            "/orange/thai/vinhnguyen1/aspd_v3/"
            f"gemma_smol_gate_collapse_recovery_v1/{self.arm_id}"
        )
        if self.runtime.output_root != f"{namespace}/pilot/runs":
            raise ValueError("pilot output namespace changed")
        if self.runtime.checkpoint_root != f"{namespace}/pilot/checkpoints":
            raise ValueError("checkpoint namespace changed")
        if self.runtime.preflight_evidence != f"{namespace}/preflight_r1/preflight_report.json":
            raise ValueError("preflight namespace changed")
        if "two_model_scale_v1/data" not in self.runtime.dataset_manifest:
            raise ValueError("experiment must reuse the pinned two-model dataset manifest")
        if "two_model_scale_v1/data" not in self.runtime.alignment_manifest:
            raise ValueError("experiment must reuse the pinned two-tokenizer alignment manifest")
        return self

    def payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def load_gate_collapse_recovery_config(path: str | Path) -> GateCollapseRecoveryConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return GateCollapseRecoveryConfig.model_validate(payload)
