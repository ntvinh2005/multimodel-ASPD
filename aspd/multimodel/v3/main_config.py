"""Frozen, versioned ASPD v3 main-v1 scientific and execution contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aspd.multimodel.v3.config import V3_LOGICAL_MATRICES, V3_MODEL_ORDER


class MainV1Architecture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dictionary_size: Literal[8192]
    batch_topk_k: Literal[32]
    pool_tokens: Literal[2048]
    top_k_aux: Literal[2048]
    auxk_coefficient: Literal[0.03125]
    matryoshka_group_fracs: tuple[
        Literal[0.0625], Literal[0.0625], Literal[0.125], Literal[0.25], Literal[0.5]
    ]
    gate_grouping: Literal["pre_mid"]
    model_order: tuple[Literal["qwen", "gemma", "phi"], ...]
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
    def validate_coverage(self) -> MainV1Architecture:
        if tuple(self.model_order) != V3_MODEL_ORDER:
            raise ValueError("main-v1 model order changed")
        if tuple(self.logical_matrices) != V3_LOGICAL_MATRICES:
            raise ValueError("main-v1 must include all seven logical matrices")
        return self


class MainV1Objective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal_code: Literal["binary_support"]
    activation_code: Literal["continuous"]
    internal_gradient_route: Literal["rank_one_only"]
    activation_gradient_route: Literal["encoder_packing_adapter_readout_and_activation_decoder"]
    internal_loss: Literal["balanced_pre_mid_mean_fvu"]
    activation_loss: Literal["balanced_pre_mid_mean_matryoshka_fvu"]
    auxk_loss: Literal["balanced_pre_mid_dead_feature_revival"]
    internal_coefficient: Literal[1.0]
    activation_coefficient: Literal[1.0]
    fvu_reduction_dtype: Literal["float32"]
    activation_decoder_normalization: Literal["unit_row_norm_with_radial_gradient_projection"]
    native_bias: Literal["preserve"]


class MainV1Optimization(BaseModel):
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


class MainV1Data(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence_length: Literal[512]
    effective_batch_anchors: Literal[8192]
    minimum_accepted_anchor_exposures: Literal[2048000000]
    minimum_optimizer_steps: Literal[250000]
    split_unit: Literal["document"]
    split_seed: Literal[0]
    split_modulus: Literal[1000]
    validation_buckets: tuple[int, ...]
    test_buckets: tuple[int, ...]
    canonical_text_policy: Literal["identity_exact_three_way_prefix"]
    validation_target_anchors: Literal[5000000]
    test_target_anchors: Literal[5000000]

    @model_validator(mode="after")
    def validate_splits_and_budget(self) -> MainV1Data:
        if self.validation_buckets != tuple(range(10)):
            raise ValueError("main-v1 validation buckets must be 0..9")
        if self.test_buckets != tuple(range(10, 20)):
            raise ValueError("main-v1 test buckets must be 10..19")
        if self.effective_batch_anchors * self.minimum_optimizer_steps != (
            self.minimum_accepted_anchor_exposures
        ):
            raise ValueError("main-v1 step and exposure budgets disagree")
        return self


class MainV1Runtime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["fsdp"]
    world_size: Literal[4]
    checkpoint_every_steps: Literal[1000]
    keep_complete_checkpoints: Literal[2]
    output_root: str
    checkpoint_root: str
    dataset_manifest: str
    alignment_manifest: str
    preflight_evidence: str
    max_preflight_steps: int = Field(gt=0, le=20)
    max_preflight_minutes: int = Field(gt=0, le=45)

    @model_validator(mode="after")
    def validate_namespace(self) -> MainV1Runtime:
        values = (
            self.output_root,
            self.checkpoint_root,
            self.dataset_manifest,
            self.alignment_manifest,
            self.preflight_evidence,
        )
        if any("aspd_v3" not in value or "main_v1" not in value for value in values):
            raise ValueError("main-v1 runtime paths must use an isolated aspd_v3/main_v1 namespace")
        return self


class ScopedAccessEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slurm_job_id: Literal["48541592"]
    account: Literal["thai"]
    partition: Literal["hpg-rtx6000"]
    hostname: Literal["c0800a-s25.ufhpc"]
    evidence_dir: str
    smoke_source_sha256: Literal[
        "9f037c55af1efddfa85744350280c8b93295bbcddc993e319087f70ab858ecaf"
    ]
    committed_smoke_source_git_sha: Literal[
        "5e1e418e24dc917e33e1d256484800df4af4d58e"
    ]
    scope: Literal["this_account_environment_and_recorded_run_only"]


class V3MainV1Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    experiment_family: Literal["aspd_v3_three_family_main"]
    scientific_status: Literal["FROZEN_RESEARCHER_APPROVED"]
    execution_authorized: Literal[True]
    authorization_statement_sha256: Literal[
        "831723eacce597b04f4ee824ea223ec871f322798c9be2b768814ebb8448591f"
    ]
    approved_config_path: str
    approved_config_sha256: Literal[
        "0d45e2679a3677a2d3ea752284fd112cc8216414c05a19d64a739997ff2e6542"
    ]
    architecture: MainV1Architecture
    objective: MainV1Objective
    optimization: MainV1Optimization
    data: MainV1Data
    runtime: MainV1Runtime
    access_evidence: ScopedAccessEvidence
    evaluation_status: Literal["NOT_IMPLEMENTED — DEFERRED BY RESEARCHER"]

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def structural_report(self, root: str | Path = ".") -> dict[str, Any]:
        base = Path(root)
        approved = base / self.approved_config_path
        blockers: list[str] = []
        if not approved.is_file() or _sha256(approved) != self.approved_config_sha256:
            blockers.append("approved config is missing or its SHA-256 changed")
        return {
            "status": "STRUCTURALLY_READY" if not blockers else "BLOCKED",
            "blockers": blockers,
            "config_fingerprint": self.fingerprint(),
            "logical_target_count": 168,
            "minimum_optimizer_steps": self.data.minimum_optimizer_steps,
            "evaluation": self.evaluation_status,
        }

    def _access_evidence_blockers(self, base: Path) -> list[str]:
        blockers: list[str] = []
        evidence = base / self.access_evidence.evidence_dir / "implementation_sha256.json"
        if not evidence.is_file():
            blockers.append("scoped real-checkpoint smoke evidence is missing")
        else:
            recorded = json.loads(evidence.read_text(encoding="utf-8"))
            observed = recorded.get("implementation", {}).get(
                "aspd/multimodel/cli/smoke_aspd_v3.py"
            )
            if observed != self.access_evidence.smoke_source_sha256:
                blockers.append("recorded smoke implementation SHA-256 is inconsistent")
        return blockers

    def execution_blockers(self, root: str | Path = ".") -> list[str]:
        base = Path(root)
        blockers = list(self.structural_report(base)["blockers"])
        blockers.extend(self._access_evidence_blockers(base))
        for label, value in (
            ("main trainer entrypoint", "aspd/multimodel/cli/train_aspd_v3_main.py"),
            ("main Slurm launcher", "slurm/multimodel/aspd_v3_main.sbatch"),
        ):
            if not (base / value).is_file():
                blockers.append(f"{label} is not implemented")
        for label, value in (
            ("dataset manifest", self.runtime.dataset_manifest),
            ("alignment manifest", self.runtime.alignment_manifest),
            ("distributed preflight evidence", self.runtime.preflight_evidence),
        ):
            if not (base / value).is_file():
                blockers.append(f"{label} is not materialized")
        return blockers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_main_v1_config(path: str | Path) -> V3MainV1Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ASPD v3 main-v1 config must be a mapping")
    return V3MainV1Config.model_validate(payload)
