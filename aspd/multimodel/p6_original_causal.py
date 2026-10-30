"""Bounded causal intervention protocol for the completed original-ASPD 100M run."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch import Tensor, nn

from aspd.multimodel.cache import load_effective_model
from aspd.multimodel.causal import P6ForwardResult, p6_patched_forward
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.mechanism_discovery import capture_model_forward
from aspd.multimodel.original_recipe import sha256_file
from aspd.multimodel.original_steering import (
    freeze_clean_original_gate,
    load_original_decomposition_for_p6,
)
from aspd.multimodel.original_streaming import resolved_config_sha256

P6_OUTCOME_TYPES = (
    "TECHNICAL_VALIDATION",
    "CAUSAL_EXPERIMENT_DATA",
    "BEHAVIORAL_SCORING",
    "INDEPENDENT_VERIFICATION",
)
FROZEN_SOURCE_MODELS = (
    (
        "base",
        "unsloth/Qwen2.5-7B-Instruct",
        "a75c9dc945567a9b6f568b8503a0307731607bee",
    ),
    (
        "target",
        "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports",
        "6aa7e935c74c519c66555a111d1e6982059bd82d",
    ),
)
FROZEN_GROUNDING = "model.layers.14.post_attention_layernorm"
FROZEN_MATRIX = "model.layers.14.self_attn.o_proj"
FROZEN_P6_STATE_PATHS = {
    "base_b_dec": "components.0.o_proj.b_dec",
    "base_b_out": "components.0.o_proj.b_out",
    "base_u": "components.0.o_proj.U",
    "base_v": "components.0.o_proj.V",
    "shared_gate_threshold": "sparsity_controller.threshold",
    "target_b_dec": "components.1.o_proj.b_dec",
    "target_b_out": "components.1.o_proj.b_out",
    "target_u": "components.1.o_proj.U",
    "target_v": "components.1.o_proj.V",
}


class P6CandidateSpec(BaseModel):
    component_id: int = Field(ge=0, lt=36_864)
    risk_screening_rank: PositiveInt
    active_control_component_id: int = Field(ge=0, lt=36_864)


class P6OriginalPilotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    aspd_config: str
    run_dir: str
    output_root: str
    expected_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_config_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_file: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_candidates: PositiveInt = 3
    include_computational_secondary: bool = False
    prompt_ids: dict[Literal["risk", "sports_control", "neutral"], tuple[str, ...]]
    technical_prompt_ids: dict[
        Literal["risk", "sports_control", "neutral"], tuple[str, ...]
    ]
    selection_source: Literal["verified_risk_screening_shortlist"]
    selection_rule: str
    active_control_selection_rule: str
    gate_policy: Literal["clean_shared_learned_JumpReLU_threshold"]
    expected_learned_threshold: PositiveFloat
    scientific_scoring_protocol: Literal["frozen_records_only_behavioral_scoring_not_verified"]
    behavioral_scoring_contract: str
    behavioral_scoring_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    behavioral_risk_score_cutoff: int = Field(ge=0, le=3)
    alphas: tuple[float, ...]
    random_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_new_tokens: PositiveInt = 64
    technical_max_new_tokens: PositiveInt = 8
    technical_prompts_per_category: PositiveInt = 2
    actual_norm_relative_tolerance: float = Field(default=0.1, gt=0.0, lt=1.0)
    minimum_effective_applied_l2: PositiveFloat = 1e-8
    pilot_runtime_budget_seconds: PositiveInt = 5400
    runtime_estimate_safety_factor: float = Field(default=1.25, ge=1.0)
    decoding: Literal["greedy_argmax"] = "greedy_argmax"
    intervention_models: tuple[Literal["base", "target"], ...] = ("base", "target")
    confirmation_prompt_file: str | None = None
    confirmation_prompt_sha256: str | None = None
    confirmation_approved: bool = False
    judge: None = None

    @model_validator(mode="after")
    def validate_frozen_design(self) -> P6OriginalPilotConfig:
        if self.maximum_candidates > 3:
            raise ValueError("bounded P6 permits at most three dynamically selected candidates")
        if self.include_computational_secondary:
            raise ValueError("the initial discovery pilot is frozen to risk-screening candidates")
        if self.intervention_models != ("base", "target"):
            raise ValueError("P6 difference-in-differences requires Base and Target")
        confirmation_fields = (
            self.confirmation_prompt_file,
            self.confirmation_prompt_sha256,
        )
        if self.confirmation_approved and not all(confirmation_fields):
            raise ValueError("approved confirmation requires a path and SHA-256")
        if not self.confirmation_approved and any(confirmation_fields):
            raise ValueError("unapproved confirmation artifacts must not be configured")
        if self.alphas != (0.0, 0.25, 0.5, 1.0):
            raise ValueError("P6 alpha grid must be exactly [0, 0.25, 0.5, 1.0]")
        if self.random_seeds != (0, 1, 2, 3, 4):
            raise ValueError("P6 random-control seeds must be exactly [0,1,2,3,4]")
        if any(seed < 0 for seed in self.random_seeds):
            raise ValueError("random-control seeds must be non-negative")
        if set(self.prompt_ids) != {"risk", "sports_control", "neutral"}:
            raise ValueError("prompt IDs must cover risk, sports_control, and neutral")
        if any(len(values) != 12 for values in self.prompt_ids.values()):
            raise ValueError("the frozen pilot requires exactly 12 prompts per category")
        flattened = [value for values in self.prompt_ids.values() for value in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError("pilot prompt IDs must be unique")
        if self.technical_prompts_per_category > 12:
            raise ValueError("technical prompt count cannot exceed the frozen pilot")
        if set(self.technical_prompt_ids) != {"risk", "sports_control", "neutral"}:
            raise ValueError("technical prompt IDs must cover all three categories")
        for category, values in self.technical_prompt_ids.items():
            if len(values) != self.technical_prompts_per_category:
                raise ValueError("technical prompt count differs from the declared count")
            if not set(values) <= set(self.prompt_ids[category]):
                raise ValueError("technical prompts must be a frozen pilot subset")
        return self


def load_p6_original_config(path: str | Path) -> P6OriginalPilotConfig:
    return P6OriginalPilotConfig.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _artifact_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("result package has no artifact list")
    result = {str(row["path"]): row for row in artifacts}
    if len(result) != len(artifacts):
        raise ValueError("result package contains duplicate artifact paths")
    return result


def _select_active_control_ids(
    metrics_rows: Sequence[Mapping[str, str]], candidate_ids: Sequence[int]
) -> list[int]:
    """Apply the frozen pre-outcome matching rule to the verified metrics table."""

    by_id = {int(row["component_id"]): row for row in metrics_rows}
    excluded = set(candidate_ids)
    selected: list[int] = []
    for component_id in candidate_ids:
        reference = by_id[component_id]

        def score(
            row: Mapping[str, str], reference: Mapping[str, str] = reference
        ) -> float:
            firing_ratio = max(float(row["held_out_firing_frequency"]), 1e-30) / max(
                float(reference["held_out_firing_frequency"]), 1e-30
            )
            contribution_ratio = max(
                float(row["target_contribution_rms_per_validation_token"]), 1e-30
            ) / max(
                float(reference["target_contribution_rms_per_validation_token"]),
                1e-30,
            )
            return (
                abs(math.log(firing_ratio))
                + abs(math.log(contribution_ratio))
                + abs(float(row["paired_risk_minus_control_activation_strength"]))
            )

        eligible = [
            row
            for row in metrics_rows
            if int(row["component_id"]) not in excluded
            and float(row["held_out_fire_count"]) > 0
        ]
        if not eligible:
            raise ValueError("no eligible matched-active component remains")
        control_id = int(min(eligible, key=score)["component_id"])
        selected.append(control_id)
        excluded.add(control_id)
    return selected


def _validate_checkpoint_component_tensors(
    checkpoint: Path,
    parameter_index: Mapping[str, Any],
    component_ids: Sequence[int],
) -> dict[str, Any]:
    """Validate only P6-required mmap tensors without materializing the 12.7 GB state."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if int(payload.get("step", -1)) != 24_414:
        raise ValueError("canonical rank-0 checkpoint step is not 24,414")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("canonical checkpoint has no model state")
    paths = parameter_index.get("state_dict_paths")
    if not isinstance(paths, Mapping):
        raise ValueError("component parameter index has no state_dict_paths")
    required_names = (
        "base_u",
        "base_v",
        "base_b_dec",
        "base_b_out",
        "target_u",
        "target_v",
        "target_b_dec",
        "target_b_out",
        "shared_gate_threshold",
    )
    missing = [name for name in required_names if paths.get(name) not in state]
    if missing:
        raise ValueError(f"checkpoint lacks required P6 tensor paths: {missing}")
    tensors = {name: state[str(paths[name])] for name in required_names}
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise TypeError("P6 checkpoint state paths must resolve to tensors")
    base_u, base_v = tensors["base_u"], tensors["base_v"]
    target_u, target_v = tensors["target_u"], tensors["target_v"]
    if not (
        base_u.ndim == 2
        and base_u.shape == base_v.shape == target_u.shape == target_v.shape
        and int(base_u.shape[0]) == 36_864
    ):
        raise ValueError("P6 U/V tensors have incompatible component shapes")
    width = int(base_u.shape[1])
    for name in ("base_b_dec", "base_b_out", "target_b_dec", "target_b_out"):
        if tuple(tensors[name].shape) != (width,):
            raise ValueError(f"P6 tensor {name} has incompatible shape")
        if not torch.isfinite(tensors[name]).all():
            raise FloatingPointError(f"P6 tensor {name} contains NaN or Inf")
    selected = torch.tensor(component_ids, dtype=torch.long)
    for name in ("base_u", "base_v", "target_u", "target_v"):
        values = tensors[name].index_select(0, selected)
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"selected rows of {name} contain NaN or Inf")
    threshold = tensors["shared_gate_threshold"]
    if threshold.numel() != 1 or not torch.isfinite(threshold).all():
        raise FloatingPointError("P6 shared gate threshold is invalid")
    return {
        "step": int(payload["step"]),
        "config_sha256": str(payload.get("config_sha256")),
        "component_count": int(base_u.shape[0]),
        "component_width": width,
        "validated_component_ids": list(component_ids),
        "learned_threshold": float(threshold.item()),
    }


def verify_p6_handoff(
    pilot: P6OriginalPilotConfig, *, verify_all_artifact_hashes: bool = False
) -> dict[str, Any]:
    """Fail closed on package, checkpoint, config, shortlist, and prompt provenance."""

    run = Path(pilot.run_dir)
    manifest_path = run / "result_package_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "P6_HANDOFF_READY":
        raise ValueError("completed package is not P6_HANDOFF_READY")
    if manifest.get("risk_screening_status") != "VERIFIED":
        raise ValueError("risk-screening shortlist is not VERIFIED")
    artifacts = _artifact_map(manifest)
    required = {
        "config_frozen.yaml",
        "checkpoint_00024414.rank0.pt",
        "p6_handoff/candidates_computational.csv",
        "p6_handoff/candidates_risk_screening.csv",
        "p6_handoff/component_parameter_index.json",
        "evaluation/component_metrics.csv",
    }
    if not required <= set(artifacts):
        raise ValueError(f"result package lacks required artifacts: {sorted(required - set(artifacts))}")
    parameter_index = _read_json(run / "p6_handoff/component_parameter_index.json")
    state_paths = parameter_index.get("state_dict_paths", {})
    if any(state_paths.get(name) != path for name, path in FROZEN_P6_STATE_PATHS.items()):
        raise ValueError("component parameter index differs from the frozen P6 tensor paths")
    relative_checkpoint = parameter_index["authoritative_checkpoint_relative_to_run"]
    if relative_checkpoint != "checkpoint_00024414.rank0.pt":
        raise ValueError("unexpected authoritative checkpoint path")
    checkpoint = run / relative_checkpoint
    expected = pilot.expected_checkpoint_sha256
    recorded_hashes = {
        parameter_index["authoritative_checkpoint_sha256"],
        artifacts[relative_checkpoint]["sha256"],
    }
    if recorded_hashes != {expected}:
        raise ValueError("checkpoint SHA differs across the handoff records")
    if sha256_file(checkpoint) != expected:
        raise ValueError("canonical rank-0 checkpoint checksum failed")
    config_path = Path(pilot.aspd_config)
    if sha256_file(config_path) != pilot.expected_config_file_sha256:
        raise ValueError("canonical ASPD config file checksum failed")
    if sha256_file(run / "config_frozen.yaml") != pilot.expected_config_file_sha256:
        raise ValueError("frozen run config differs from the canonical config")
    aspd_cfg = load_experiment_config(config_path)
    observed_models = tuple(
        (model.name, model.model_id, model.revision) for model in aspd_cfg.models
    )
    if observed_models != FROZEN_SOURCE_MODELS:
        raise ValueError("ASPD config does not contain the frozen Base/Target revisions")
    if any(
        model.grounding.capture != "input"
        or model.grounding.module != FROZEN_GROUNDING
        or len(model.matrices) != 1
        or model.matrices[0].module != FROZEN_MATRIX
        or model.matrices[0].name != "o_proj"
        for model in aspd_cfg.models
    ):
        raise ValueError("ASPD config does not match the frozen L14 o_proj contract")
    if (
        aspd_cfg.data.tokenizer != FROZEN_SOURCE_MODELS[0][1]
        or aspd_cfg.data.tokenizer_revision != FROZEN_SOURCE_MODELS[0][2]
    ):
        raise ValueError("ASPD tokenizer does not match the frozen Base revision")
    marker = torch.load(run / "latest_checkpoint.pt", map_location="cpu", weights_only=True)
    if int(marker.get("step", -1)) != 24_414:
        raise ValueError("P6 requires the canonical step-24,414 checkpoint marker")
    if marker["config_sha256"] != resolved_config_sha256(aspd_cfg):
        raise ValueError("resolved ASPD config fingerprint differs from checkpoint marker")
    if marker["rank_sha256"][0] != expected:
        raise ValueError("rank-0 marker checksum differs from the P6 checkpoint")
    shortlist = run / "p6_handoff/candidates_risk_screening.csv"
    computational = run / "p6_handoff/candidates_computational.csv"
    metrics_path = run / "evaluation/component_metrics.csv"
    parameter_index_path = run / "p6_handoff/component_parameter_index.json"
    for relative, path in (
        ("p6_handoff/candidates_risk_screening.csv", shortlist),
        ("p6_handoff/candidates_computational.csv", computational),
        ("p6_handoff/component_parameter_index.json", parameter_index_path),
        ("evaluation/component_metrics.csv", metrics_path),
    ):
        record = artifacts[relative]
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"verified handoff artifact mismatch: {relative}")
    rows = sorted(
        csv.DictReader(shortlist.open(encoding="utf-8")),
        key=lambda row: int(row["rank"]),
    )
    if len(rows) < pilot.maximum_candidates:
        raise ValueError("risk-screening shortlist has too few candidates")
    selected_rows = rows[: pilot.maximum_candidates]
    candidate_ids = [int(row["component_id"]) for row in selected_rows]
    if len(candidate_ids) != len(set(candidate_ids)) or any(
        not 0 <= component_id < 36_864 for component_id in candidate_ids
    ):
        raise ValueError("candidate IDs must be unique and lie in [0,36864)")
    for expected_rank, row in enumerate(selected_rows, 1):
        if int(row["rank"]) != expected_rank:
            raise ValueError("risk-screening shortlist ranks are not contiguous from one")
        expected_row_values = {
            "aspd_checkpoint_sha256": expected,
            "base_model_id": aspd_cfg.models[0].model_id,
            "base_model_revision": aspd_cfg.models[0].revision,
            "target_model_id": aspd_cfg.models[1].model_id,
            "target_model_revision": aspd_cfg.models[1].revision,
            "layer": "14",
            "matrix": FROZEN_MATRIX,
            "risk_screening_status": "VERIFIED",
        }
        if any(row.get(key) != value for key, value in expected_row_values.items()):
            raise ValueError(f"candidate shortlist provenance mismatch at rank {expected_rank}")
    metrics_rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    if len(metrics_rows) != 36_864:
        raise ValueError("component metrics do not cover all 36,864 components")
    active_control_ids = _select_active_control_ids(metrics_rows, candidate_ids)
    candidate_specs = [
        P6CandidateSpec(
            component_id=component_id,
            risk_screening_rank=int(row["rank"]),
            active_control_component_id=control_id,
        )
        for component_id, control_id, row in zip(
            candidate_ids, active_control_ids, selected_rows, strict=True
        )
    ]
    tensor_audit = _validate_checkpoint_component_tensors(
        checkpoint,
        parameter_index,
        [*candidate_ids, *active_control_ids],
    )
    if tensor_audit["config_sha256"] != marker["config_sha256"]:
        raise ValueError("rank-0 checkpoint/config fingerprint mismatch")
    if tensor_audit["learned_threshold"] != pilot.expected_learned_threshold:
        raise ValueError("rank-0 learned threshold differs from the frozen P6 contract")
    prompt_path = Path(pilot.prompt_file)
    if sha256_file(prompt_path) != pilot.prompt_sha256:
        raise ValueError("locked prompt file checksum failed")
    scoring_path = Path(pilot.behavioral_scoring_contract)
    if sha256_file(scoring_path) != pilot.behavioral_scoring_contract_sha256:
        raise ValueError("behavioral scoring contract checksum failed")
    prompt_rows = [
        json.loads(line)
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_prompt_ids = {
        prompt_id
        for category_ids in pilot.prompt_ids.values()
        for prompt_id in category_ids
    }
    observed_prompt_ids = [row.get("prompt_id") for row in prompt_rows]
    if (
        len(prompt_rows) != 36
        or len(observed_prompt_ids) != len(set(observed_prompt_ids))
        or set(observed_prompt_ids) != expected_prompt_ids
    ):
        raise ValueError("locked discovery prompt IDs/counts differ from the frozen pilot")
    if any(
        not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
        for row in prompt_rows
    ):
        raise ValueError("locked discovery prompts must contain non-empty text")
    prompt_by_id = {row["prompt_id"]: row for row in prompt_rows}
    for category, prompt_ids in pilot.prompt_ids.items():
        for prompt_id in prompt_ids:
            if prompt_id not in prompt_by_id or prompt_by_id[prompt_id]["category"] != category:
                raise ValueError(f"prompt {prompt_id} category/provenance mismatch")
    risk_pairs = {
        prompt_by_id[prompt_id].get("pair_id") for prompt_id in pilot.prompt_ids["risk"]
    }
    control_pairs = {
        prompt_by_id[prompt_id].get("pair_id")
        for prompt_id in pilot.prompt_ids["sports_control"]
    }
    if risk_pairs != control_pairs or None in risk_pairs:
        raise ValueError("risk and sports-control discovery prompts are not exactly paired")
    if verify_all_artifact_hashes:
        for relative, record in artifacts.items():
            path = run / relative
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"artifact size mismatch: {relative}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"artifact checksum mismatch: {relative}")
    return {
        "status": "PASS",
        "result_package_status": manifest["status"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": expected,
        "config_file_sha256": pilot.expected_config_file_sha256,
        "resolved_config_sha256": marker["config_sha256"],
        "risk_shortlist_sha256": artifacts["p6_handoff/candidates_risk_screening.csv"]["sha256"],
        "computational_shortlist_sha256": artifacts["p6_handoff/candidates_computational.csv"]["sha256"],
        "candidate_rows": [dict(row) for row in selected_rows],
        "candidate_specs": [candidate.model_dump() for candidate in candidate_specs],
        "candidate_selection_frozen_before_outcomes": True,
        "candidate_source": pilot.selection_source,
        "checkpoint_tensor_audit": tensor_audit,
        "prompt_file_sha256": pilot.prompt_sha256,
        "behavioral_scoring_contract_sha256": pilot.behavioral_scoring_contract_sha256,
        "all_artifact_hashes_verified": verify_all_artifact_hashes,
    }


@dataclass(frozen=True)
class InterventionArm:
    name: str
    kind: Literal[
        "clean",
        "gate_off",
        "learned",
        "active_matched",
        "orthogonal_matched",
        "random_same_l2",
    ]
    model_name: Literal["base", "target"]
    candidate_component_id: int | None
    intervened_component_id: int | None
    alpha: float
    random_seed: int | None = None


def intervention_arms(
    pilot: P6OriginalPilotConfig,
    candidates: Sequence[P6CandidateSpec],
    *,
    technical: bool = False,
) -> list[InterventionArm]:
    candidates = candidates[:1] if technical else candidates
    seeds = pilot.random_seeds[:1] if technical else pilot.random_seeds
    alphas = (0.0, 1.0) if technical else pilot.alphas
    arms = [
        InterventionArm(f"clean_{model_name}", "clean", model_name, None, None, 0.0)
        for model_name in pilot.intervention_models
    ]
    for model_name in pilot.intervention_models:
        for candidate in candidates:
            prefix = f"{model_name}_c{candidate.component_id}"
            for alpha in alphas:
                alpha_name = str(alpha).replace(".", "p")
                arms.extend(
                    [
                        InterventionArm(
                            f"{prefix}_learned_a{alpha_name}",
                            "learned",
                            model_name,
                            candidate.component_id,
                            candidate.component_id,
                            alpha,
                        ),
                        InterventionArm(
                            f"{prefix}_gate_off_a{alpha_name}",
                            "gate_off",
                            model_name,
                            candidate.component_id,
                            candidate.component_id,
                            alpha,
                        ),
                        InterventionArm(
                            f"{prefix}_active_c{candidate.active_control_component_id}_a{alpha_name}",
                            "active_matched",
                            model_name,
                            candidate.component_id,
                            candidate.active_control_component_id,
                            alpha,
                        ),
                        InterventionArm(
                            f"{prefix}_orthogonal_a{alpha_name}",
                            "orthogonal_matched",
                            model_name,
                            candidate.component_id,
                            candidate.component_id,
                            alpha,
                        ),
                    ]
                )
                arms.extend(
                    InterventionArm(
                        f"{prefix}_random_seed_{seed}_a{alpha_name}",
                        "random_same_l2",
                        model_name,
                        candidate.component_id,
                        candidate.component_id,
                        alpha,
                        seed,
                    )
                    for seed in seeds
                )
    return arms


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def _pad_token_lists(
    token_lists: Sequence[Sequence[int]], pad_token_id: int
) -> tuple[Tensor, Tensor]:
    if not token_lists or any(not row for row in token_lists):
        raise ValueError("every autoregressive branch must contain at least one token")
    width = max(map(len, token_lists))
    ids = torch.full((len(token_lists), width), int(pad_token_id), dtype=torch.long)
    valid = torch.zeros((len(token_lists), width), dtype=torch.bool)
    for index, values in enumerate(token_lists):
        ids[index, : len(values)] = torch.tensor(values, dtype=torch.long)
        valid[index, : len(values)] = True
    return ids, valid


def _last_logits(logits: Tensor, lengths: Sequence[int]) -> Tensor:
    rows = torch.arange(len(lengths), device=logits.device)
    positions = torch.tensor(lengths, device=logits.device, dtype=torch.long) - 1
    return logits[rows, positions]


@torch.no_grad()
def clean_reference_gate(
    decomposition: Any,
    *,
    base_token_ids: Tensor,
    target_token_ids: Tensor,
    clean_base_grounding: Tensor,
    clean_target_grounding: Tensor,
    valid_tokens: Tensor,
    component_ids: Sequence[int],
) -> Tensor:
    """Freeze the learned threshold gate from clean, exact-token Base/Target passes."""

    if not torch.equal(base_token_ids.detach().cpu(), target_token_ids.detach().cpu()):
        raise ValueError("Base and Target clean-reference token IDs must be identical")
    if clean_base_grounding.shape[:2] != base_token_ids.shape:
        raise ValueError("Base grounding does not match the live clean prefix")
    if clean_target_grounding.shape[:2] != target_token_ids.shape:
        raise ValueError("Target grounding does not match the live clean prefix")
    return freeze_clean_original_gate(
        decomposition,
        [clean_base_grounding, clean_target_grounding],
        valid_tokens,
        component_ids,
    )


def _load_locked_prompts(pilot: P6OriginalPilotConfig) -> list[dict[str, Any]]:
    wanted = {
        prompt_id
        for category_ids in pilot.prompt_ids.values()
        for prompt_id in category_ids
    }
    rows = [
        json.loads(line)
        for line in Path(pilot.prompt_file).read_text(encoding="utf-8").splitlines()
        if line
    ]
    observed_ids = [row.get("prompt_id") for row in rows]
    if (
        len(rows) != 36
        or len(observed_ids) != len(set(observed_ids))
        or set(observed_ids) != wanted
    ):
        raise ValueError("locked prompt file differs from the frozen 36-prompt split")
    by_id = {row["prompt_id"]: row for row in rows}
    if wanted - set(by_id):
        raise ValueError(f"locked prompt file lacks {sorted(wanted - set(by_id))}")
    return [
        by_id[prompt_id]
        for category in ("risk", "sports_control", "neutral")
        for prompt_id in pilot.prompt_ids[category]
    ]


def _load_confirmatory_prompts(pilot: P6OriginalPilotConfig) -> list[dict[str, Any]]:
    """Load only a separately approved and checksummed confirmatory prompt artifact."""

    if not pilot.confirmation_approved:
        raise RuntimeError("confirmation remains blocked pending researcher approval")
    assert pilot.confirmation_prompt_file is not None
    assert pilot.confirmation_prompt_sha256 is not None
    path = Path(pilot.confirmation_prompt_file)
    if path.resolve() == Path(pilot.prompt_file).resolve():
        raise ValueError("discovery prompts cannot be reused as confirmation")
    if not path.is_file() or sha256_file(path) != pilot.confirmation_prompt_sha256:
        raise ValueError("confirmatory prompt artifact is missing or has the wrong checksum")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids = [row.get("prompt_id") for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("confirmatory prompt IDs must be non-empty and unique")
    counts = {
        category: sum(row.get("category") == category for row in rows)
        for category in ("risk", "sports_control", "neutral")
    }
    if any(count == 0 for count in counts.values()) or sum(counts.values()) != len(rows):
        raise ValueError("confirmation requires risk, sports_control, and neutral prompts")
    risk_pairs = {row.get("pair_id") for row in rows if row.get("category") == "risk"}
    control_pairs = {
        row.get("pair_id") for row in rows if row.get("category") == "sports_control"
    }
    if risk_pairs != control_pairs or None in risk_pairs:
        raise ValueError("confirmatory risk and sports-control prompts must be paired")
    return rows


def _render_prompt_tokens(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    output: list[list[int]] = []
    for row in rows:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row["prompt"])}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(tokens, Tensor):
            tokens = tokens.tolist()
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"chat template produced no tokens for {row['prompt_id']}")
        output.append([int(token) for token in tokens])
    return output


def _model_weight_audit(model: nn.Module, cfg: Any) -> dict[str, dict[str, Any]]:
    parameters: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        try:
            version: int | None = int(parameter._version)
        except RuntimeError:
            version = None
        parameters[name] = {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "data_ptr": int(parameter.data_ptr()),
            "version": version,
        }
    matrices: dict[str, Any] = {}
    for matrix in cfg.matrices:
        weight = model.get_submodule(matrix.module).weight.detach().float()
        matrices[matrix.name] = {
            "shape": list(weight.shape),
            "sum": float(weight.sum().item()),
            "squared_sum": float(weight.square().sum().item()),
            "max_abs": float(weight.abs().max().item()),
        }
    return {"all_parameter_metadata": parameters, "patched_matrix_numeric": matrices}


def _empty_sequence_audit(count: int) -> list[dict[str, Any]]:
    return [
        {
            "gate_firing_count": 0,
            "modified_token_position_count": 0,
            "rounded_to_zero_position_count": 0,
            "intended_squared_l2": 0.0,
            "applied_squared_l2": 0.0,
            "candidate_reference_applied_squared_l2": 0.0,
            "relative_applied_squared_sum": 0.0,
            "relative_applied_observations": 0,
            "clean_reference_logit_squared_error_sum": 0.0,
            "clean_reference_logit_element_count": 0,
            "clean_reference_logit_max_abs_difference": 0.0,
            "steps": 0,
            "per_step_position_actual_applied_l2": [],
            "per_step_position_candidate_actual_applied_l2": [],
        }
        for _ in range(count)
    ]


def _accumulate_sequence_audit(
    totals: list[dict[str, Any]],
    active_indices: Sequence[int],
    diagnostics: Mapping[str, Any],
) -> None:
    per_sequence = diagnostics["per_sequence"]
    if len(per_sequence) != len(active_indices):
        raise RuntimeError("P6 per-sequence diagnostics do not match the active batch")
    for local_index, global_index in enumerate(active_indices):
        row = per_sequence[local_index]
        total = totals[global_index]
        total["gate_firing_count"] += int(row["gate_active_token_count"])
        total["modified_token_position_count"] += int(
            row["modified_token_position_count"]
        )
        total["rounded_to_zero_position_count"] += int(
            row["rounded_to_zero_position_count"]
        )
        total["intended_squared_l2"] += float(
            row["intended_fp32_subtraction_l2"]
        ) ** 2
        total["applied_squared_l2"] += float(
            row["actual_applied_output_difference_l2"]
        ) ** 2
        total["candidate_reference_applied_squared_l2"] += float(
            row["candidate_actual_applied_output_difference_l2"]
        ) ** 2
        total["relative_applied_squared_sum"] += float(
            row["relative_applied_l2_vs_clean_output"]
        ) ** 2
        total["relative_applied_observations"] += 1
        total["steps"] += 1
        total["per_step_position_actual_applied_l2"].append(
            list(row["per_position_actual_applied_l2"])
        )
        total["per_step_position_candidate_actual_applied_l2"].append(
            list(row["per_position_candidate_actual_applied_l2"])
        )


def _finalize_sequence_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    observations = int(value["relative_applied_observations"])
    return {
        "intended_perturbation_l2": math.sqrt(float(value["intended_squared_l2"])),
        "applied_perturbation_l2": math.sqrt(float(value["applied_squared_l2"])),
        "same_prefix_candidate_reference_applied_l2": math.sqrt(
            float(value["candidate_reference_applied_squared_l2"])
        ),
        "per_position_effective_norm_rms": math.sqrt(
            float(value["applied_squared_l2"])
            / max(1, int(value["modified_token_position_count"]))
        ),
        "relative_perturbation_vs_clean_output_rms": math.sqrt(
            float(value["relative_applied_squared_sum"]) / max(1, observations)
        ),
        "logit_rms_difference_vs_clean_reference": math.sqrt(
            float(value["clean_reference_logit_squared_error_sum"])
            / max(1, int(value["clean_reference_logit_element_count"]))
        ),
        "logit_max_abs_difference_vs_clean_reference": float(
            value["clean_reference_logit_max_abs_difference"]
        ),
        "gate_firing_count": int(value["gate_firing_count"]),
        "modified_token_position_count": int(value["modified_token_position_count"]),
        "rounded_to_zero_position_count": int(value["rounded_to_zero_position_count"]),
        "generation_steps": int(value["steps"]),
        "per_step_position_actual_applied_l2": list(
            value["per_step_position_actual_applied_l2"]
        ),
        "per_step_position_candidate_actual_applied_l2": list(
            value["per_step_position_candidate_actual_applied_l2"]
        ),
    }


@torch.inference_mode()
def _generate_arm(
    *,
    arm: InterventionArm,
    models: Sequence[nn.Module],
    cfg: Any,
    decomposition: Any,
    prompt_tokens: Sequence[Sequence[int]],
    pad_token_id: int,
    eos_token_ids: set[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Greedy generation with a clean gate recomputed on each arm's live prefix."""

    generated: list[list[int]] = [[] for _ in prompt_tokens]
    finished = [False for _ in prompt_tokens]
    sequence_audit = _empty_sequence_audit(len(prompt_tokens))
    intervention_model_index = next(
        index
        for index, model_cfg in enumerate(cfg.models)
        if model_cfg.name == arm.model_name
    )
    started = time.perf_counter()
    executed_generation_steps = 0
    active_sequence_steps = 0
    live_prefix_token_positions = 0
    for _step in range(max_new_tokens):
        active = [index for index, done in enumerate(finished) if not done]
        if not active:
            break
        branches = [list(prompt_tokens[index]) + generated[index] for index in active]
        executed_generation_steps += 1
        active_sequence_steps += len(active)
        live_prefix_token_positions += sum(map(len, branches))
        ids, valid = _pad_token_lists(branches, pad_token_id)
        lengths = list(map(len, branches))
        if arm.kind == "clean":
            clean_model = capture_model_forward(
                models[intervention_model_index],
                cfg.models[intervention_model_index],
                ids,
                valid,
            )
            next_logits = _last_logits(clean_model["logits"], lengths)
        else:
            # Both clean models see the exact same CPU token IDs for this branch.
            clean = [
                capture_model_forward(model, model_cfg, ids, valid)
                for model, model_cfg in zip(models, cfg.models, strict=True)
            ]
            clean_residuals = [record["R"].float() for record in clean]
            assert arm.candidate_component_id is not None
            assert arm.intervened_component_id is not None
            gate = clean_reference_gate(
                decomposition,
                base_token_ids=ids,
                target_token_ids=ids.detach().clone(),
                clean_base_grounding=clean_residuals[0],
                clean_target_grounding=clean_residuals[1],
                valid_tokens=valid.to(next(decomposition.parameters()).device),
                component_ids=[arm.candidate_component_id],
            ).cpu()
            if arm.kind == "gate_off":
                gate.zero_()
            patched: P6ForwardResult = p6_patched_forward(
                models[intervention_model_index],
                cfg.models[intervention_model_index],
                decomposition,
                intervention_model_index,
                ids,
                valid,
                gate,
                [arm.candidate_component_id],
                strength=arm.alpha,
                norm_control=arm.kind == "orthogonal_matched",
                random_control_seed=(
                    arm.random_seed if arm.kind == "random_same_l2" else None
                ),
                matched_write_component_ids=(
                    [arm.intervened_component_id]
                    if arm.kind == "active_matched"
                    else None
                ),
            )
            next_logits = _last_logits(patched.logits, lengths)
            clean_reference_logits = _last_logits(
                clean[intervention_model_index]["logits"], lengths
            )
            for local_index, global_index in enumerate(active):
                logit_difference = (
                    next_logits[local_index].float()
                    - clean_reference_logits[local_index].float()
                )
                sequence_audit[global_index][
                    "clean_reference_logit_squared_error_sum"
                ] += float(logit_difference.square().sum().item())
                sequence_audit[global_index][
                    "clean_reference_logit_element_count"
                ] += int(logit_difference.numel())
                sequence_audit[global_index][
                    "clean_reference_logit_max_abs_difference"
                ] = max(
                    float(
                        sequence_audit[global_index][
                            "clean_reference_logit_max_abs_difference"
                        ]
                    ),
                    float(logit_difference.abs().max().item()),
                )
            if set(patched.matrix_diagnostics) != {"o_proj"}:
                raise RuntimeError("100M P6 must patch exactly the L14 o_proj matrix")
            _accumulate_sequence_audit(
                sequence_audit, active, patched.matrix_diagnostics["o_proj"]
            )
        if not torch.isfinite(next_logits).all():
            raise FloatingPointError(f"non-finite logits in arm {arm.name}")
        next_tokens = next_logits.argmax(dim=-1)
        for local_index, global_index in enumerate(active):
            token = int(next_tokens[local_index].item())
            generated[global_index].append(token)
            finished[global_index] = token in eos_token_ids
    return {
        "continuations": generated,
        "eos_reached": finished,
        "sequence_audit": [_finalize_sequence_audit(row) for row in sequence_audit],
        "runtime_seconds": time.perf_counter() - started,
        "executed_generation_steps": executed_generation_steps,
        "active_sequence_steps": active_sequence_steps,
        "live_prefix_token_positions": live_prefix_token_positions,
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _continuation_difference(left: Sequence[int], right: Sequence[int]) -> dict[str, int]:
    common = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        common += 1
    width = max(len(left), len(right))
    changed = sum(
        index >= len(left)
        or index >= len(right)
        or left[index] != right[index]
        for index in range(width)
    )
    return {"changed_token_count": changed, "common_prefix_tokens": common}


def _generation_records(
    *,
    stage: Literal["technical", "discovery", "confirmation"],
    arm: InterventionArm,
    rows: Sequence[Mapping[str, Any]],
    generated: Mapping[str, Any],
    clean_continuations: Sequence[Sequence[int]],
    tokenizer: Any,
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(rows):
        continuation = generated["continuations"][index]
        clean = clean_continuations[index]
        records.append(
            {
                **provenance,
                "outcome_type": (
                    "TECHNICAL_VALIDATION"
                    if stage == "technical"
                    else "CAUSAL_EXPERIMENT_DATA"
                ),
                "stage": stage,
                "prompt_id": row["prompt_id"],
                "prompt_category": row["category"],
                "sport_domain": row.get("sport_domain"),
                "intervention_arm": arm.name,
                "intervention_kind": arm.kind,
                "intervention_model": arm.model_name,
                "candidate_component_id": arm.candidate_component_id,
                "gate_source_component_id": arm.candidate_component_id,
                "intervened_component_id": arm.intervened_component_id,
                "alpha": arm.alpha,
                "random_seed": arm.random_seed,
                "clean_continuation_token_ids": list(clean),
                "clean_continuation": tokenizer.decode(
                    clean, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
                "patched_continuation_token_ids": continuation,
                "patched_continuation": tokenizer.decode(
                    continuation,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "new_token_count": len(continuation),
                "eos_reached": generated["eos_reached"][index],
                **_continuation_difference(continuation, clean),
                **generated["sequence_audit"][index],
                "behavioral_scoring_status": "NOT_VERIFIED",
                "independent_verification_status": "NOT_VERIFIED",
            }
        )
    return records


def _norm_control_audit(
    records: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    """Validate controls against a same-prefix post-cast learned reference.

    Autoregressive arms may have different prefixes after their first divergent
    token.  Each control forward therefore records the learned candidate's
    hypothetical post-dtype output difference on *that control branch's* live
    prefix.  Matching a control against a separate learned arm would compare
    different inputs and is not a valid per-position norm check.
    """

    comparisons: list[dict[str, Any]] = []
    for record in records:
        if record["intervention_kind"] not in {
            "active_matched",
            "orthogonal_matched",
            "random_same_l2",
        }:
            continue
        reference_steps = record[
            "per_step_position_candidate_actual_applied_l2"
        ]
        control_steps = record["per_step_position_actual_applied_l2"]
        shape_match = len(reference_steps) == len(control_steps) and all(
            len(reference) == len(control)
            for reference, control in zip(
                reference_steps, control_steps, strict=False
            )
        )
        reference_values = [
            float(value) for step in reference_steps for value in step
        ]
        control_values = [
            float(value) for step in control_steps for value in step
        ]
        position_errors: list[float] = []
        invalid_zero_reference_positions = 0
        if shape_match:
            for reference, control in zip(
                reference_values, control_values, strict=True
            ):
                if reference == 0.0:
                    if control != 0.0:
                        invalid_zero_reference_positions += 1
                    continue
                position_errors.append(abs(control - reference) / reference)
        reference = float(record["same_prefix_candidate_reference_applied_l2"])
        control = float(record["applied_perturbation_l2"])
        aggregate_relative_error = abs(control - reference) / max(reference, 1e-30)
        maximum_position_relative_error = max(position_errors, default=0.0)
        positive_reference_positions = sum(value > 0.0 for value in reference_values)
        valid = (
            shape_match
            and positive_reference_positions > 0
            and invalid_zero_reference_positions == 0
            and maximum_position_relative_error <= tolerance
        )
        comparisons.append(
            {
                "stage": record["stage"],
                "prompt_id": record["prompt_id"],
                "candidate_component_id": record["candidate_component_id"],
                "intervention_model": record["intervention_model"],
                "control_arm": record["intervention_arm"],
                "learned_actual_applied_l2": reference,
                "control_actual_applied_l2": control,
                "aggregate_relative_norm_error": aggregate_relative_error,
                # Retain the old field as a compatibility alias, but validity is
                # determined by the stricter per-position comparison below.
                "relative_norm_error": aggregate_relative_error,
                "maximum_position_relative_norm_error": (
                    maximum_position_relative_error
                ),
                "positive_reference_position_count": positive_reference_positions,
                "compared_position_count": len(reference_values),
                "invalid_zero_reference_position_count": (
                    invalid_zero_reference_positions
                ),
                "position_grid_shape_match": shape_match,
                "valid": valid,
                "matching_basis": (
                    "same control-branch live prefix, per token/prefix position, "
                    "actual output difference after dtype rounding"
                ),
                "cross_arm_limitation": (
                    "No norm equality is claimed between already-diverged learned "
                    "and control continuations; each control is matched to the "
                    "candidate reference recomputed on its own live prefix."
                ),
            }
        )
    return {
        "tolerance": tolerance,
        "comparisons": comparisons,
        "valid_count": sum(bool(row["valid"]) for row in comparisons),
        "invalid_count": sum(not bool(row["valid"]) for row in comparisons),
    }


def _annotate_comparison_validity(
    records: Sequence[dict[str, Any]],
    norm_audit: Mapping[str, Any],
    minimum_effective_applied_l2: float,
) -> None:
    control_validity = {
        (row["stage"], row["prompt_id"], row["control_arm"]): bool(row["valid"])
        for row in norm_audit["comparisons"]
    }
    for record in records:
        kind = record["intervention_kind"]
        reasons: list[str] = []
        if kind == "learned" and (
            float(record["applied_perturbation_l2"])
            < minimum_effective_applied_l2
            or int(record["gate_firing_count"]) == 0
        ):
            reasons.append("INEFFECTIVE_LEARNED_DOSE")
        if kind in {"active_matched", "orthogonal_matched", "random_same_l2"}:
            key = (record["stage"], record["prompt_id"], record["intervention_arm"])
            if not control_validity[key]:
                reasons.append("ACTUAL_CONTROL_NORM_MISMATCH_OR_ZERO_REFERENCE")
        record["causal_comparison_valid"] = not reasons
        record["causal_comparison_invalid_reasons"] = reasons


def _runtime_probe_arms(
    pilot: P6OriginalPilotConfig,
    candidates: Sequence[P6CandidateSpec],
) -> list[InterventionArm]:
    """Choose one full-batch clean and patched timing probe per source model."""

    arms = intervention_arms(pilot, candidates, technical=True)
    return [
        next(
            arm
            for arm in arms
            if arm.model_name == model_name
            and (
                arm.kind == "clean"
                if group == "clean"
                else arm.kind == "learned" and arm.alpha == 1.0
            )
        )
        for model_name in pilot.intervention_models
        for group in ("clean", "patched")
    ]


def _estimate_discovery_runtime_from_full_batch_probes(
    pilot: P6OriginalPilotConfig,
    candidates: Sequence[P6CandidateSpec],
    prompt_tokens: Sequence[Sequence[int]],
    probes: Sequence[tuple[InterventionArm, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Project discovery runtime from measured full-batch prefix work.

    The old estimator multiplied smoke time by the prompt-count ratio even
    though prompts run as one batch.  This estimator times representative clean
    and three-forward patched arms at the *actual 36-prompt batch size*, then
    scales by total live-prefix token positions and arm counts.  The projection
    still carries a safety factor and remains a planning estimate, not a causal
    validity gate.
    """

    if not prompt_tokens:
        raise ValueError("runtime estimation requires frozen prompt tokens")
    full_prefix_work = sum(
        len(tokens) + step
        for step in range(pilot.max_new_tokens)
        for tokens in prompt_tokens
    )
    probe_by_group: dict[tuple[str, str], Mapping[str, Any]] = {}
    for arm, result in probes:
        group = "clean" if arm.kind == "clean" else "patched"
        key = (arm.model_name, group)
        if key in probe_by_group:
            raise ValueError(f"duplicate runtime probe group: {key}")
        if int(result["live_prefix_token_positions"]) <= 0:
            raise ValueError(f"runtime probe has no measured prefix work: {key}")
        probe_by_group[key] = result
    expected_groups = {
        (model_name, group)
        for model_name in pilot.intervention_models
        for group in ("clean", "patched")
    }
    if set(probe_by_group) != expected_groups:
        raise ValueError("runtime probes do not cover clean/patched Base and Target")

    arm_counts: dict[tuple[str, str], int] = {}
    for arm in intervention_arms(pilot, candidates, technical=False):
        key = (arm.model_name, "clean" if arm.kind == "clean" else "patched")
        arm_counts[key] = arm_counts.get(key, 0) + 1

    groups: list[dict[str, Any]] = []
    raw_seconds = 0.0
    for key in sorted(expected_groups):
        result = probe_by_group[key]
        measured_work = int(result["live_prefix_token_positions"])
        seconds_per_prefix_token = float(result["runtime_seconds"]) / measured_work
        projected = (
            seconds_per_prefix_token * full_prefix_work * arm_counts[key]
        )
        raw_seconds += projected
        groups.append(
            {
                "intervention_model": key[0],
                "forward_group": key[1],
                "arm_count": arm_counts[key],
                "probe_runtime_seconds": float(result["runtime_seconds"]),
                "probe_live_prefix_token_positions": measured_work,
                "seconds_per_live_prefix_token_position": (
                    seconds_per_prefix_token
                ),
                "projected_seconds_before_safety_factor": projected,
            }
        )
    return {
        "method": "full_batch_prefix_work_by_model_and_forward_group_v1",
        "full_batch_prompt_count": len(prompt_tokens),
        "probe_max_new_tokens": pilot.technical_max_new_tokens,
        "discovery_max_new_tokens": pilot.max_new_tokens,
        "discovery_live_prefix_token_positions_per_arm_upper_bound": (
            full_prefix_work
        ),
        "groups": groups,
        "raw_projected_seconds": raw_seconds,
        "safety_factor": pilot.runtime_estimate_safety_factor,
        "estimated_pilot_runtime_seconds": (
            raw_seconds * pilot.runtime_estimate_safety_factor
        ),
        "batching_note": (
            "Prompt count is measured at the actual full batch size and is not "
            "multiplied as an independent linear factor."
        ),
    }


def _technical_gate_report(
    records: Sequence[Mapping[str, Any]],
    norm_audit: Mapping[str, Any],
    before_weights: Mapping[str, Any],
    after_weights: Mapping[str, Any],
    minimum_effective_applied_l2: float,
    estimated_pilot_runtime_seconds: float,
    pilot_runtime_budget_seconds: int,
) -> dict[str, Any]:
    technical = [row for row in records if row["stage"] == "technical"]
    clean = {
        (row["intervention_model"], row["prompt_id"]): row
        for row in technical
        if row["intervention_kind"] == "clean"
    }
    failures: list[str] = []
    if not _finite_tree(technical):
        failures.append("NON_FINITE_TECHNICAL_RECORD")
    if before_weights != after_weights:
        failures.append("SOURCE_MODEL_WEIGHT_MUTATION")
    zero_groups = {
        "alpha_zero": [
            row
            for row in technical
            if row["intervention_kind"] == "learned" and float(row["alpha"]) == 0.0
        ],
        "gate_off": [
            row for row in technical if row["intervention_kind"] == "gate_off"
        ],
    }
    for kind, rows in zero_groups.items():
        if not rows or any(float(row["applied_perturbation_l2"]) != 0.0 for row in rows):
            failures.append(f"{kind.upper()}_NOT_ZERO")
        if any(
            float(row["logit_max_abs_difference_vs_clean_reference"]) != 0.0
            for row in rows
        ):
            failures.append(f"{kind.upper()}_LOGIT_MISMATCH")
        if any(
            row["patched_continuation_token_ids"]
            != clean[
                (row["intervention_model"], row["prompt_id"])
            ]["patched_continuation_token_ids"]
            for row in rows
        ):
            failures.append(f"{kind.upper()}_CONTINUATION_MISMATCH")
    learned = [
        row
        for row in technical
        if row["intervention_kind"] == "learned" and float(row["alpha"]) > 0.0
    ]
    learned_by_model = {
        model_name: [
            row for row in learned if row["intervention_model"] == model_name
        ]
        for model_name in ("base", "target")
    }
    per_model_dose: dict[str, Any] = {}
    for model_name, model_rows in learned_by_model.items():
        maximum_l2 = max(
            (float(row["applied_perturbation_l2"]) for row in model_rows),
            default=0.0,
        )
        firing_count = sum(int(row["gate_firing_count"]) for row in model_rows)
        per_model_dose[model_name] = {
            "learned_record_count": len(model_rows),
            "maximum_applied_l2": maximum_l2,
            "gate_firing_count": firing_count,
        }
        if not model_rows or maximum_l2 < minimum_effective_applied_l2:
            failures.append(f"INEFFECTIVE_{model_name.upper()}_LEARNED_DOSE")
        if firing_count == 0:
            failures.append(f"{model_name.upper()}_LEARNED_GATE_NEVER_FIRED")
    technical_norms = [
        row for row in norm_audit["comparisons"] if row["stage"] == "technical"
    ]
    effective_technical_norms = [
        row
        for row in technical_norms
        if float(row["learned_actual_applied_l2"]) >= minimum_effective_applied_l2
    ]
    if not effective_technical_norms or any(
        not row["valid"] for row in effective_technical_norms
    ):
        failures.append("ACTUAL_CONTROL_NORM_MISMATCH")
    if estimated_pilot_runtime_seconds > pilot_runtime_budget_seconds:
        failures.append("PILOT_RUNTIME_EXCEEDS_BUDGET")
    return {
        "outcome_type": "TECHNICAL_VALIDATION",
        "status": "GPU_PREFLIGHT_GO" if not failures else "GPU_PREFLIGHT_NO_GO",
        "failures": sorted(set(failures)),
        "technical_record_count": len(technical),
        "learned_gate_firing_count": sum(
            int(row["gate_firing_count"]) for row in learned
        ),
        "maximum_learned_applied_l2": max(
            (float(row["applied_perturbation_l2"]) for row in learned), default=0.0
        ),
        "per_model_learned_dose": per_model_dose,
        "source_model_weights_immutable": before_weights == after_weights,
        "estimated_pilot_runtime_seconds": estimated_pilot_runtime_seconds,
        "pilot_runtime_budget_seconds": pilot_runtime_budget_seconds,
        "behavioral_scoring_status": "NOT_VERIFIED",
    }


def _artifact_checksums(output_dir: Path) -> dict[str, Any]:
    names = (
        "run_manifest.json",
        "preflight_report.json",
        "selected_candidates.json",
        "component_characterization.json",
        "intervention_records.jsonl",
        "generation_records.jsonl",
        "control_norm_audit.json",
        "technical_summary.json",
        "behavioral_scoring_manifest.json",
        "behavioral_evaluation_records.jsonl",
        "behavioral_summary.json",
        "scientific_summary.md",
    )
    return {
        "algorithm": "sha256",
        "artifacts": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in names
        },
    }


def _selected_candidates_artifact(
    pilot: P6OriginalPilotConfig,
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    rows = {int(row["component_id"]): row for row in handoff["candidate_rows"]}
    candidates = [P6CandidateSpec.model_validate(row) for row in handoff["candidate_specs"]]
    return {
        "selection_frozen_before_causal_outcomes": True,
        "selection_source": pilot.selection_source,
        "selection_rule": pilot.selection_rule,
        "active_control_selection_rule": pilot.active_control_selection_rule,
        "checkpoint_sha256": handoff["checkpoint_sha256"],
        "risk_shortlist_sha256": handoff["risk_shortlist_sha256"],
        "computational_shortlist_sha256": handoff["computational_shortlist_sha256"],
        "candidates": [
            {
                **candidate.model_dump(),
                "selection_evidence": rows[candidate.component_id],
            }
            for candidate in candidates
        ],
        "computational_secondary_included": pilot.include_computational_secondary,
        "intervention_models": list(pilot.intervention_models),
        "intervention_alphas": list(pilot.alphas),
        "random_seeds": list(pilot.random_seeds),
        "gate_policy": pilot.gate_policy,
        "learned_threshold": pilot.expected_learned_threshold,
        "prompt_ids": {key: list(value) for key, value in pilot.prompt_ids.items()},
        "technical_prompt_ids": {
            key: list(value) for key, value in pilot.technical_prompt_ids.items()
        },
        "scientific_scoring_protocol": pilot.scientific_scoring_protocol,
        "behavioral_scoring_contract": pilot.behavioral_scoring_contract,
        "behavioral_scoring_contract_sha256": pilot.behavioral_scoring_contract_sha256,
        "behavioral_risk_score_cutoff": pilot.behavioral_risk_score_cutoff,
        "component_characterization": handoff["component_characterization"],
    }


def component_operator_comparison(
    base_u: Tensor,
    base_v: Tensor,
    target_u: Tensor,
    target_v: Tensor,
) -> dict[str, float]:
    """Compare effective rank-one operators without choosing a U/V gauge."""

    vectors = [value.detach().float().flatten() for value in (base_u, base_v, target_u, target_v)]
    if any(not torch.isfinite(value).all() for value in vectors):
        raise FloatingPointError("component factors contain NaN or Inf")
    base_u32, base_v32, target_u32, target_v32 = vectors
    base_norm = float(base_u32.norm().item() * base_v32.norm().item())
    target_norm = float(target_u32.norm().item() * target_v32.norm().item())
    inner = float(
        (base_u32 @ target_u32).item() * (base_v32 @ target_v32).item()
    )
    difference_squared = max(
        0.0,
        base_norm * base_norm + target_norm * target_norm - 2.0 * inner,
    )
    difference = math.sqrt(difference_squared)
    denominator = max(base_norm * target_norm, 1e-30)
    return {
        "base_operator_frobenius_norm": base_norm,
        "target_operator_frobenius_norm": target_norm,
        "operator_difference_frobenius_norm": difference,
        "relative_operator_difference": difference / max(base_norm + target_norm, 1e-30),
        "operator_cosine_similarity": inner / denominator,
    }


def characterize_selected_components(
    decomposition: Any,
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze gauge-invariant parameter characterization before causal outcomes."""

    candidates = [P6CandidateSpec.model_validate(row) for row in handoff["candidate_specs"]]
    source_rows = {int(row["component_id"]): row for row in handoff["candidate_rows"]}
    base_u, base_v = decomposition.component_factors(0, "o_proj")
    target_u, target_v = decomposition.component_factors(1, "o_proj")
    base_center = decomposition.component_input_center(0, "o_proj").detach().float()
    target_center = decomposition.component_input_center(1, "o_proj").detach().float()
    rows = []
    for candidate in candidates:
        component_id = candidate.component_id
        bu = base_u[component_id].detach().float()
        bv = base_v[component_id].detach().float()
        tu = target_u[component_id].detach().float()
        tv = target_v[component_id].detach().float()
        base_affine = -torch.dot(base_center, bv) * bu
        target_affine = -torch.dot(target_center, tv) * tu
        operator = component_operator_comparison(bu, bv, tu, tv)
        rows.append(
            {
                "component_id": component_id,
                "risk_screening_rank": candidate.risk_screening_rank,
                "active_control_component_id": candidate.active_control_component_id,
                "selection_evidence": source_rows[component_id],
                "effective_operator": operator,
                "affine_center_effect": {
                    "base_l2": float(base_affine.norm().item()),
                    "target_l2": float(target_affine.norm().item()),
                    "target_minus_base_l2": float(
                        (target_affine - base_affine).norm().item()
                    ),
                    "base_center_l2": float(base_center.norm().item()),
                    "target_center_l2": float(target_center.norm().item()),
                    "center_difference_l2": float(
                        (target_center - base_center).norm().item()
                    ),
                },
                "factor_diagnostics_not_used_as_gauge_invariant_evidence": {
                    "base_u_l2": float(bu.norm().item()),
                    "base_v_l2": float(bv.norm().item()),
                    "target_u_l2": float(tu.norm().item()),
                    "target_v_l2": float(tv.norm().item()),
                },
                "matrix_localization": {
                    "layer": 14,
                    "matrix": "model.layers.14.self_attn.o_proj",
                    "matrix_count": 1,
                },
            }
        )
    return {
        "outcome_type": "OBSERVATIONAL_COMPONENT_CHARACTERIZATION",
        "selection_frozen_before_causal_outcomes": True,
        "operator_definition": "P_c^(n) = u_c^(n) (v_c^(n))^T",
        "component_count": len(rows),
        "components": rows,
        "scientific_boundary": (
            "Operator, affine-center, activation, and contribution differences are "
            "observational and do not prove reconstruction of W_target-W_base."
        ),
    }


def _write_run_artifacts(
    *,
    output_dir: Path,
    manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
    selected: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    norm_audit: Mapping[str, Any],
    technical_summary: Mapping[str, Any],
) -> None:
    intervention_records = [
        {
            key: value
            for key, value in record.items()
            if key
            not in {
                "clean_continuation",
                "patched_continuation",
                "clean_continuation_token_ids",
                "patched_continuation_token_ids",
            }
        }
        for record in records
    ]
    _atomic_json(output_dir / "run_manifest.json", dict(manifest))
    _atomic_json(output_dir / "preflight_report.json", dict(preflight))
    _atomic_json(output_dir / "selected_candidates.json", dict(selected))
    characterization = selected.get("component_characterization")
    if not isinstance(characterization, Mapping):
        raise ValueError("selected candidate artifact lacks component characterization")
    _atomic_json(
        output_dir / "component_characterization.json",
        dict(characterization),
    )
    _atomic_text(
        output_dir / "intervention_records.jsonl",
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in intervention_records),
    )
    _atomic_text(
        output_dir / "generation_records.jsonl",
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records),
    )
    _atomic_json(output_dir / "control_norm_audit.json", dict(norm_audit))
    _atomic_json(output_dir / "technical_summary.json", dict(technical_summary))
    scoring_contract = _read_json(Path(selected["behavioral_scoring_contract"]))
    _atomic_json(
        output_dir / "behavioral_scoring_manifest.json",
        {
            "status": "FROZEN_NOT_SCORED",
            "contract_path": selected["behavioral_scoring_contract"],
            "contract_sha256": selected["behavioral_scoring_contract_sha256"],
            "contract": scoring_contract,
            "record_count": len(records),
        },
    )
    _atomic_text(
        output_dir / "behavioral_evaluation_records.jsonl",
        "".join(
            json.dumps(
                {
                    "git_commit": row["git_commit"],
                    "pilot_config_sha256": row["pilot_config_sha256"],
                    "frozen_aspd_config_sha256": row["frozen_aspd_config_sha256"],
                    "canonical_rank0_checkpoint_sha256": row[
                        "canonical_rank0_checkpoint_sha256"
                    ],
                    "prompt_file_sha256": row["prompt_file_sha256"],
                    "behavioral_scoring_contract_sha256": row[
                        "behavioral_scoring_contract_sha256"
                    ],
                    "target_model_id": row["target_model_id"],
                    "target_model_revision": row["target_model_revision"],
                    "base_model_id": row["base_model_id"],
                    "base_model_revision": row["base_model_revision"],
                    "gate_policy": row["gate_policy"],
                    "learned_threshold": row["learned_threshold"],
                    "prompt_id": row["prompt_id"],
                    "prompt_category": row["prompt_category"],
                    "candidate_component_id": row["candidate_component_id"],
                    "intervention_arm": row["intervention_arm"],
                    "intervention_kind": row["intervention_kind"],
                    "intervention_model": row["intervention_model"],
                    "alpha": row["alpha"],
                    "random_seed": row["random_seed"],
                    "causal_comparison_valid": row.get("causal_comparison_valid"),
                    "clean_continuation_token_ids": row[
                        "clean_continuation_token_ids"
                    ],
                    "clean_continuation": row["clean_continuation"],
                    "patched_continuation_token_ids": row[
                        "patched_continuation_token_ids"
                    ],
                    "patched_continuation": row["patched_continuation"],
                    "risky_action_endorsement": None,
                    "generic_refusal_overrefusal": None,
                    "useful_safe_sports_advice": None,
                    "coherence_general_helpfulness": None,
                    "broader_behavior_change": None,
                    "scoring_status": "NOT_VERIFIED",
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in records
        ),
    )
    _atomic_json(
        output_dir / "behavioral_summary.json",
        {
            "outcome_type": "BEHAVIORAL_SCORING",
            "status": "NOT_VERIFIED",
            "behavioral_risk_score_cutoff": selected["behavioral_risk_score_cutoff"],
            "effect_sizes": None,
            "uncertainty": None,
            "multiple_testing_adjustment": None,
            "discovery_signal": None,
            "reason": "No approved calibrated scoring results were supplied.",
        },
    )
    scientific_status = (
        "CAUSAL_EXPERIMENT_DATA_GENERATED_BEHAVIORAL_SCORING_NOT_VERIFIED"
        if manifest["status"] == "JOB_COMPLETED_UNVERIFIED"
        else str(manifest["status"])
    )
    _atomic_text(
        output_dir / "scientific_summary.md",
        "\n".join(
            [
                "# P6-100M causal pilot summary",
                "",
                f"- Status: `{scientific_status}`",
                f"- Technical gate: `{technical_summary['status']}`",
                "- Behavioral scoring: `NOT_VERIFIED`",
                "- Independent verification: `NOT_VERIFIED`",
                "- Interpretation: generated differences are not by themselves evidence of a causal risk mechanism.",
                "- This bounded run did not retrain ASPD or modify source-model weights.",
                "",
            ]
        ),
    )
    _atomic_json(output_dir / "artifact_checksums.json", _artifact_checksums(output_dir))


def validate_preflight_go(
    report_path: str | Path,
    *,
    pilot_path: Path,
    pilot: P6OriginalPilotConfig,
    handoff: Mapping[str, Any],
    git_head: str,
) -> dict[str, Any]:
    """Validate a separate Stage-A GO artifact before Stage B can load any model."""

    path = Path(report_path)
    report = _read_json(path)
    technical = report.get("technical_gate", {})
    if report.get("status") != "GPU_PREFLIGHT_GO" or technical.get("status") != "GPU_PREFLIGHT_GO":
        raise ValueError("bounded pilot requires a machine-readable GPU_PREFLIGHT_GO report")
    expected = {
        "checkpoint_sha256": handoff["checkpoint_sha256"],
        "config_file_sha256": pilot.expected_config_file_sha256,
        "prompt_file_sha256": pilot.prompt_sha256,
        "behavioral_scoring_contract_sha256": pilot.behavioral_scoring_contract_sha256,
        "pilot_config_sha256": sha256_file(pilot_path),
        "git_commit": git_head,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"preflight GO provenance mismatch: {key}")
    if float(technical["estimated_pilot_runtime_seconds"]) > float(
        technical["pilot_runtime_budget_seconds"]
    ):
        raise ValueError("preflight runtime estimate exceeds the frozen pilot budget")
    return {
        **report,
        "source_path": str(path.resolve()),
        "source_sha256": sha256_file(path),
    }


def _run_pilot_stage(
    *,
    pilot_path: Path,
    pilot: P6OriginalPilotConfig,
    destination: Path,
    handoff: Mapping[str, Any],
    verified_preflight: Mapping[str, Any],
    models: Sequence[nn.Module],
    aspd_cfg: Any,
    decomposition: Any,
    tokenizer: Any,
    prompt_rows: Sequence[Mapping[str, Any]],
    prompt_tokens: Sequence[Sequence[int]],
    pad_token_id: int,
    eos_token_ids: set[int],
    before_weights: Mapping[str, Any],
    common_provenance: Mapping[str, Any],
    source_provenance: Sequence[Mapping[str, Any]],
    candidates: Sequence[P6CandidateSpec],
    evidence_stage: Literal["discovery", "confirmation"],
) -> dict[str, Any]:
    """Execute Stage B only after a separately produced, provenance-matched GO."""

    records: list[dict[str, Any]] = []
    pilot_clean: dict[str, list[list[int]]] = {}
    pilot_exception: Exception | None = None
    try:
        for arm in intervention_arms(pilot, candidates, technical=False):
            generated = _generate_arm(
                arm=arm,
                models=models,
                cfg=aspd_cfg,
                decomposition=decomposition,
                prompt_tokens=prompt_tokens,
                pad_token_id=pad_token_id,
                eos_token_ids=eos_token_ids,
                max_new_tokens=pilot.max_new_tokens,
            )
            if arm.kind == "clean":
                pilot_clean[arm.model_name] = generated["continuations"]
            if arm.model_name not in pilot_clean:
                raise RuntimeError(f"missing clean baseline for {arm.model_name}")
            records.extend(
                _generation_records(
                    stage=evidence_stage,
                    arm=arm,
                    rows=prompt_rows,
                    generated=generated,
                    clean_continuations=pilot_clean[arm.model_name],
                    tokenizer=tokenizer,
                    provenance=common_provenance,
                )
            )
    except Exception as error:  # Preserve partial evidence without claiming completion.
        pilot_exception = error

    final_weights = {
        model_cfg.name: _model_weight_audit(model, model_cfg)
        for model, model_cfg in zip(models, aspd_cfg.models, strict=True)
    }
    if final_weights != before_weights:
        raise RuntimeError("source-model weights changed during bounded P6 pilot")
    norm_audit = _norm_control_audit(records, pilot.actual_norm_relative_tolerance)
    _annotate_comparison_validity(records, norm_audit, pilot.minimum_effective_applied_l2)
    if not _finite_tree(records) or not _finite_tree(norm_audit):
        raise FloatingPointError("bounded P6 pilot artifacts contain NaN or Inf")
    technical_summary = dict(verified_preflight["technical_gate"])
    if pilot_exception is None:
        status = "JOB_COMPLETED_UNVERIFIED"
    else:
        status = "IMPLEMENTATION_BLOCKED"
        technical_summary["pilot_execution_failure"] = {
            "exception_type": type(pilot_exception).__name__,
            "exception_message": str(pilot_exception),
        }
    manifest = {
        **common_provenance,
        "execution_stage": f"bounded_{evidence_stage}",
        "scientific_evidence_stage": evidence_stage.upper(),
        "status": status,
        "outcome_types": list(P6_OUTCOME_TYPES),
        "pilot_config_path": str(pilot_path),
        "pilot_config_sha256": sha256_file(pilot_path),
        "output_dir": str(destination),
        "source_model_provenance": list(source_provenance),
        "verified_preflight_report_path": verified_preflight["source_path"],
        "verified_preflight_report_sha256": verified_preflight["source_sha256"],
        "behavioral_scoring_status": "NOT_VERIFIED",
        "independent_verification_status": "NOT_VERIFIED",
        "aspd_retrained": False,
        "source_weights_modified": False,
    }
    preflight_copy = {
        **dict(verified_preflight),
        "source_weights_after_pilot": final_weights,
    }
    _write_run_artifacts(
        output_dir=destination,
        manifest=manifest,
        preflight=preflight_copy,
        selected=_selected_candidates_artifact(pilot, handoff),
        records=records,
        norm_audit=norm_audit,
        technical_summary=technical_summary,
    )
    return {
        "status": status,
        "output_dir": str(destination),
        "technical_summary": technical_summary,
        "record_count": len(records),
    }


@torch.inference_mode()
def run_p6_original_causal(
    pilot_config_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    verify_all_artifact_hashes: bool = True,
    stage: Literal["preflight", "pilot", "confirmation"] = "preflight",
    preflight_report: str | Path | None = None,
) -> dict[str, Any]:
    """Execute exactly one stage: real-model preflight or separately gated pilot."""

    from transformers import AutoTokenizer

    pilot_path = Path(pilot_config_path)
    pilot = load_p6_original_config(pilot_path)
    destination = Path(output_dir)
    root = Path(pilot.output_root).resolve()
    resolved_destination = destination.resolve()
    if resolved_destination == root or root not in resolved_destination.parents:
        raise ValueError("P6 output must be a fresh child of the dedicated output root")
    if destination.exists():
        raise FileExistsError(f"refusing to reuse P6 output directory {destination}")
    handoff = verify_p6_handoff(
        pilot, verify_all_artifact_hashes=verify_all_artifact_hashes
    )
    git_head = _git_head()
    if stage == "preflight" and preflight_report is not None:
        raise ValueError("Stage A does not accept a preflight report")
    if stage in {"pilot", "confirmation"} and preflight_report is None:
        raise ValueError("causal measurement requires a separate Stage-A GO report")
    if stage == "confirmation" and not pilot.confirmation_approved:
        raise RuntimeError("confirmation remains blocked pending researcher approval")
    verified_preflight = (
        validate_preflight_go(
            preflight_report,
            pilot_path=pilot_path,
            pilot=pilot,
            handoff=handoff,
            git_head=git_head,
        )
        if stage in {"pilot", "confirmation"}
        else None
    )
    destination.mkdir(parents=True, exist_ok=False)
    aspd_cfg = load_experiment_config(pilot.aspd_config)
    if [model.name for model in aspd_cfg.models] != ["base", "target"]:
        raise ValueError("P6 requires the exact Base/Target model order")
    if any([matrix.name for matrix in model.matrices] != ["o_proj"] for model in aspd_cfg.models):
        raise ValueError("P6-100M is frozen to exactly one o_proj per model")
    discovery_prompt_rows = _load_locked_prompts(pilot)
    experiment_prompt_rows = (
        _load_confirmatory_prompts(pilot)
        if stage == "confirmation"
        else discovery_prompt_rows
    )
    tokenizer = AutoTokenizer.from_pretrained(
        aspd_cfg.data.tokenizer, revision=aspd_cfg.data.tokenizer_revision
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    eos = tokenizer.eos_token_id
    eos_token_ids = {int(value) for value in (eos if isinstance(eos, list) else [eos]) if value is not None}

    models: list[nn.Module] = []
    source_provenance: list[dict[str, Any]] = []
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for model_cfg in aspd_cfg.models:
        model, provenance = load_effective_model(model_cfg, device)
        models.append(model)
        source_provenance.append(provenance)
    decomposition = load_original_decomposition_for_p6(
        aspd_cfg,
        handoff["checkpoint_path"],
        models,
        source_provenance,
        device,
    )
    handoff["component_characterization"] = characterize_selected_components(
        decomposition, handoff
    )
    candidates = [
        P6CandidateSpec.model_validate(row) for row in handoff["candidate_specs"]
    ]
    controller = decomposition.sparsity_controller
    if controller is None:
        raise ValueError("original P6 requires the learned sparsity controller")
    learned_threshold = float(controller.threshold.item())
    if learned_threshold != pilot.expected_learned_threshold:
        raise ValueError("checkpoint learned threshold differs from the frozen pilot")

    prompt_rows = experiment_prompt_rows
    prompt_by_id = {row["prompt_id"]: row for row in discovery_prompt_rows}
    technical_rows = [
        prompt_by_id[prompt_id]
        for category in ("risk", "sports_control", "neutral")
        for prompt_id in pilot.technical_prompt_ids[category]
    ]
    technical_tokens = _render_prompt_tokens(tokenizer, technical_rows)
    all_tokens = _render_prompt_tokens(tokenizer, prompt_rows)
    before_weights = {
        model_cfg.name: _model_weight_audit(model, model_cfg)
        for model, model_cfg in zip(models, aspd_cfg.models, strict=True)
    }
    common_provenance = {
        "git_commit": git_head,
        "pilot_config_sha256": sha256_file(pilot_path),
        "frozen_aspd_config_sha256": pilot.expected_config_file_sha256,
        "resolved_aspd_config_sha256": handoff["resolved_config_sha256"],
        "canonical_rank0_checkpoint_sha256": handoff["checkpoint_sha256"],
        "prompt_file_sha256": (
            pilot.confirmation_prompt_sha256
            if stage == "confirmation"
            else pilot.prompt_sha256
        ),
        "scientific_evidence_stage": (
            "CONFIRMATION" if stage == "confirmation" else "DISCOVERY"
        ),
        "behavioral_scoring_contract_sha256": (
            pilot.behavioral_scoring_contract_sha256
        ),
        "base_model_id": aspd_cfg.models[0].model_id,
        "base_model_revision": aspd_cfg.models[0].revision,
        "target_model_id": aspd_cfg.models[1].model_id,
        "target_model_revision": aspd_cfg.models[1].revision,
        "gate_policy": pilot.gate_policy,
        "learned_threshold": learned_threshold,
    }

    if stage in {"pilot", "confirmation"}:
        assert verified_preflight is not None
        return _run_pilot_stage(
            pilot_path=pilot_path,
            pilot=pilot,
            destination=destination,
            handoff=handoff,
            verified_preflight=verified_preflight,
            models=models,
            aspd_cfg=aspd_cfg,
            decomposition=decomposition,
            tokenizer=tokenizer,
            prompt_rows=prompt_rows,
            prompt_tokens=all_tokens,
            pad_token_id=int(pad_token_id),
            eos_token_ids=eos_token_ids,
            before_weights=before_weights,
            common_provenance=common_provenance,
            source_provenance=source_provenance,
            candidates=candidates,
            evidence_stage=("confirmation" if stage == "confirmation" else "discovery"),
        )

    all_records: list[dict[str, Any]] = []
    technical_clean: dict[str, list[list[int]]] = {}
    technical_runtime_seconds = 0.0
    technical_exception: Exception | None = None
    technical_generated: dict[str, dict[str, Any]] = {}
    runtime_probes: list[tuple[InterventionArm, dict[str, Any]]] = []
    reproducibility_check_passed = False
    try:
        for arm in intervention_arms(pilot, candidates, technical=True):
            generated = _generate_arm(
                arm=arm,
                models=models,
                cfg=aspd_cfg,
                decomposition=decomposition,
                prompt_tokens=technical_tokens,
                pad_token_id=int(pad_token_id),
                eos_token_ids=eos_token_ids,
                max_new_tokens=pilot.technical_max_new_tokens,
            )
            if arm.kind == "clean":
                technical_clean[arm.model_name] = generated["continuations"]
            technical_runtime_seconds += float(generated["runtime_seconds"])
            technical_generated[arm.name] = generated
            if arm.model_name not in technical_clean:
                raise RuntimeError(f"missing clean baseline for {arm.model_name}")
            all_records.extend(
                _generation_records(
                    stage="technical",
                    arm=arm,
                    rows=technical_rows,
                    generated=generated,
                    clean_continuations=technical_clean[arm.model_name],
                    tokenizer=tokenizer,
                    provenance=common_provenance,
                )
            )
        reproducibility_arm = next(
            arm
            for arm in intervention_arms(pilot, candidates, technical=True)
            if arm.kind == "random_same_l2" and arm.alpha == 1.0
        )
        repeated = _generate_arm(
            arm=reproducibility_arm,
            models=models,
            cfg=aspd_cfg,
            decomposition=decomposition,
            prompt_tokens=technical_tokens,
            pad_token_id=int(pad_token_id),
            eos_token_ids=eos_token_ids,
            max_new_tokens=pilot.technical_max_new_tokens,
        )
        original = technical_generated[reproducibility_arm.name]
        reproducibility_check_passed = (
            repeated["continuations"] == original["continuations"]
            and repeated["sequence_audit"] == original["sequence_audit"]
        )
        # Runtime feasibility uses the actual discovery batch size.  These
        # technical probes are not saved as causal outcomes and cannot change
        # the already frozen candidate set.
        for probe_arm in _runtime_probe_arms(pilot, candidates):
            runtime_probes.append(
                (
                    probe_arm,
                    _generate_arm(
                        arm=probe_arm,
                        models=models,
                        cfg=aspd_cfg,
                        decomposition=decomposition,
                        prompt_tokens=all_tokens,
                        pad_token_id=int(pad_token_id),
                        eos_token_ids=eos_token_ids,
                        max_new_tokens=pilot.technical_max_new_tokens,
                    ),
                )
            )
    except Exception as error:  # Preserve an explicit fail-closed artifact package.
        technical_exception = error
    after_technical_weights = {
        model_cfg.name: _model_weight_audit(model, model_cfg)
        for model, model_cfg in zip(models, aspd_cfg.models, strict=True)
    }
    hook_cleanup_passed = all(
        len(model.get_submodule(matrix.module)._forward_hooks) == 0
        for model, model_cfg in zip(models, aspd_cfg.models, strict=True)
        for matrix in model_cfg.matrices
    )
    if technical_exception is not None:
        failures = ["TECHNICAL_EXCEPTION"]
        if not hook_cleanup_passed:
            failures.append("HOOK_CLEANUP_FAILED")
        technical_summary = {
            "outcome_type": "TECHNICAL_VALIDATION",
            "status": "GPU_PREFLIGHT_NO_GO",
            "failures": failures,
            "exception_type": type(technical_exception).__name__,
            "exception_message": str(technical_exception),
            "technical_record_count_before_failure": len(all_records),
            "source_model_weights_immutable": before_weights == after_technical_weights,
            "hook_cleanup_passed": hook_cleanup_passed,
            "behavioral_scoring_status": "NOT_VERIFIED",
        }
        manifest = {
            **common_provenance,
            "status": "GPU_PREFLIGHT_NO_GO",
            "outcome_types": list(P6_OUTCOME_TYPES),
            "pilot_config_path": str(pilot_path),
            "pilot_config_sha256": sha256_file(pilot_path),
            "output_dir": str(destination),
            "source_model_provenance": source_provenance,
            "behavioral_scoring_status": "NOT_VERIFIED",
            "independent_verification_status": "NOT_VERIFIED",
            "aspd_retrained": False,
            "source_weights_modified": before_weights != after_technical_weights,
        }
        preflight = {
            **handoff,
            "status": "GPU_PREFLIGHT_NO_GO",
            "git_commit": git_head,
            "pilot_config_sha256": sha256_file(pilot_path),
            "outcome_type": "TECHNICAL_VALIDATION",
            "technical_gate": technical_summary,
            "source_weights_before": before_weights,
            "source_weights_after": after_technical_weights,
        }
        _write_run_artifacts(
            output_dir=destination,
            manifest=manifest,
            preflight=preflight,
            selected=_selected_candidates_artifact(pilot, handoff),
            records=all_records,
            norm_audit={"comparisons": [], "invalid_count": 0, "valid_count": 0},
            technical_summary=technical_summary,
        )
        return {
            "status": "GPU_PREFLIGHT_NO_GO",
            "output_dir": str(destination),
            "technical_summary": technical_summary,
            "record_count": len(all_records),
        }

    technical_norm_audit = _norm_control_audit(
        all_records, pilot.actual_norm_relative_tolerance
    )
    runtime_estimate = _estimate_discovery_runtime_from_full_batch_probes(
        pilot,
        candidates,
        all_tokens,
        runtime_probes,
    )
    estimated_pilot_runtime_seconds = float(
        runtime_estimate["estimated_pilot_runtime_seconds"]
    )
    technical_summary = _technical_gate_report(
        all_records,
        technical_norm_audit,
        before_weights,
        after_technical_weights,
        pilot.minimum_effective_applied_l2,
        estimated_pilot_runtime_seconds,
        pilot.pilot_runtime_budget_seconds,
    )
    technical_summary.update(
        {
            "technical_runtime_seconds": technical_runtime_seconds,
            "runtime_feasibility_estimate": runtime_estimate,
            "reproducibility_check_passed": reproducibility_check_passed,
            "hook_cleanup_passed": hook_cleanup_passed,
            "peak_cuda_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available() and str(device).startswith("cuda")
                else None
            ),
            "peak_cuda_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available() and str(device).startswith("cuda")
                else None
            ),
        }
    )
    if not reproducibility_check_passed:
        technical_summary["failures"].append("REPRODUCIBILITY_CHECK_FAILED")
    if not hook_cleanup_passed:
        technical_summary["failures"].append("HOOK_CLEANUP_FAILED")
    if technical_summary["failures"]:
        technical_summary["status"] = "GPU_PREFLIGHT_NO_GO"
    final_weights = after_technical_weights
    norm_audit = technical_norm_audit
    _annotate_comparison_validity(
        all_records, norm_audit, pilot.minimum_effective_applied_l2
    )
    if not _finite_tree(all_records) or not _finite_tree(norm_audit):
        raise FloatingPointError("P6 artifacts contain NaN or Inf")
    status = str(technical_summary["status"])
    manifest = {
        **common_provenance,
        "execution_stage": "gpu_preflight",
        "status": status,
        "outcome_types": list(P6_OUTCOME_TYPES),
        "pilot_config_path": str(pilot_path),
        "pilot_config_sha256": sha256_file(pilot_path),
        "output_dir": str(destination),
        "source_model_provenance": source_provenance,
        "behavioral_scoring_status": "NOT_VERIFIED",
        "independent_verification_status": "NOT_VERIFIED",
        "aspd_retrained": False,
        "source_weights_modified": False,
    }
    preflight = {
        **handoff,
        "status": status,
        "git_commit": git_head,
        "pilot_config_sha256": sha256_file(pilot_path),
        "outcome_type": "TECHNICAL_VALIDATION",
        "technical_gate": technical_summary,
        "source_weights_before": before_weights,
        "source_weights_after": final_weights,
    }
    _write_run_artifacts(
        output_dir=destination,
        manifest=manifest,
        preflight=preflight,
        selected=_selected_candidates_artifact(pilot, handoff),
        records=all_records,
        norm_audit=norm_audit,
        technical_summary=technical_summary,
    )
    return {
        "status": status,
        "output_dir": str(destination),
        "technical_summary": technical_summary,
        "record_count": len(all_records),
    }
