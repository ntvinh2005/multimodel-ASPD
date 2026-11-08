"""Fail-closed contract for the two-model Matryoshka-removal experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aspd.multimodel.v3.two_model_config import (
    LOGICAL_MATRICES_TWO,
    MODEL_ORDER_TWO,
    TwoModelArchitecture,
    TwoModelData,
    TwoModelOptimization,
    TwoModelSource,
)

EXPERIMENT_FAMILY = "aspd_v3_gemma_smol_matryoshka_ablation_v1"


class MatryoshkaArchitecture(TwoModelArchitecture):
    encoder_mixer: Literal["hierarchical_transformer"]


class MatryoshkaObjective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal_code: Literal["binary_support"]
    activation_code: Literal["continuous"]
    internal_gradient_route: Literal["rank_one_only"]
    activation_gradient_route: Literal[
        "encoder_packing_adapter_readout_and_activation_decoder"
    ]
    internal_loss: Literal["balanced_pre_mid_mean_fvu"]
    activation_loss: Literal[
        "balanced_pre_mid_mean_matryoshka_fvu",
        "balanced_pre_mid_mean_full_fvu",
    ]
    activation_objective_mode: Literal["matryoshka", "full_only"]
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
    native_bias: Literal["preserve"]


class MatryoshkaRuntime(BaseModel):
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
    checkpoint_root: str
    dataset_manifest: str
    alignment_manifest: str
    preflight_evidence: str
    max_preflight_steps: Literal[3]
    max_preflight_minutes: Literal[45]
    diagnostic_parameter_steps: tuple[
        Literal[0], Literal[64], Literal[128], Literal[256], Literal[512], Literal[768]
    ]


class TwoModelMatryoshkaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    experiment_family: Literal["aspd_v3_gemma_smol_matryoshka_ablation_v1"]
    scientific_status: Literal["RESEARCHER_AUTHORIZED_1221_STEP_CONTROLLED_ABLATION"]
    authorized_by: Literal["researcher"]
    authorized_on: Literal["2026-11-08"]
    full_training_authorized: Literal[False]
    short_pilot_authorized: Literal[True]
    gpu_preflight_authorized: Literal[True]
    arm_id: Literal["matryoshka", "full_only"]
    models: tuple[TwoModelSource, TwoModelSource]
    architecture: MatryoshkaArchitecture
    objective: MatryoshkaObjective
    optimization: TwoModelOptimization
    data: TwoModelData
    runtime: MatryoshkaRuntime
    evaluation_status: Literal["RESEARCHER_AUTHORIZED_MATCHED_HELDOUT_EVALUATION"]

    @property
    def arm(self) -> str:
        # Both experiment arms reconstruct normalized targets.  ``arm_id`` identifies
        # the scientific treatment; this compatibility property selects evaluator space.
        return "normalized_target"

    @model_validator(mode="after")
    def validate_contract(self) -> TwoModelMatryoshkaConfig:
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
                raise ValueError("Qwen is forbidden in the Matryoshka ablation")
            if model.model_id != model.tokenizer_id or model.revision != model.tokenizer_revision:
                raise ValueError("model and tokenizer revisions must be identical and pinned")
        if self.architecture.model_order != MODEL_ORDER_TWO:
            raise ValueError("two-model architecture order changed")
        if self.architecture.logical_matrices != LOGICAL_MATRICES_TWO:
            raise ValueError("logical matrix coverage changed")
        expected_mode = self.arm_id
        if self.objective.activation_objective_mode != expected_mode:
            raise ValueError("arm and activation objective mode differ")
        expected_loss = (
            "balanced_pre_mid_mean_matryoshka_fvu"
            if self.arm_id == "matryoshka"
            else "balanced_pre_mid_mean_full_fvu"
        )
        if self.objective.activation_loss != expected_loss:
            raise ValueError("activation-loss label does not match the arm")
        namespace = (
            "/orange/thai/vinhnguyen1/aspd_v3/"
            f"gemma_smol_matryoshka_ablation_v1/{self.arm_id}"
        )
        if self.runtime.output_root != f"{namespace}/pilot/runs":
            raise ValueError("pilot output namespace changed")
        if self.runtime.checkpoint_root != f"{namespace}/pilot/checkpoints":
            raise ValueError("checkpoint namespace changed")
        if self.runtime.preflight_evidence != f"{namespace}/preflight/preflight_report.json":
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


def load_two_model_matryoshka_config(path: str | Path) -> TwoModelMatryoshkaConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TwoModelMatryoshkaConfig.model_validate(payload)
