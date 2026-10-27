"""100M-specific P6 intervention and provenance contracts.

Historical P6 candidate constants are intentionally not imported here.  Candidate
IDs come from the verified 100M handoff package, while learned tensors come only
from its authoritative rank-0 checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch import Tensor, nn

from aspd.multimodel.cache import _tensor_from_value
from aspd.multimodel.config import ModelSpec, MultiModelExperimentConfig
from aspd.multimodel.model import MultiModelASPD
from aspd.multimodel.original_recipe import sha256_file, validate_original_recipe_config
from aspd.multimodel.original_steering import freeze_clean_original_gate
from aspd.multimodel.original_streaming import resolved_config_sha256

P6Arm = Literal[
    "clean",
    "learned",
    "gate_off",
    "active_matched",
    "orthogonal_matched",
    "random_same_l2",
]


def validate_p6_pilot_config(
    path: str | Path, *, require_ready: bool = False
) -> dict[str, Any]:
    """Validate the opt-in pilot without loading a checkpoint or language model."""

    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prompt_spec = config["prompt_dataset"]
    prompt_path = Path(prompt_spec["path"])
    if not prompt_path.is_file():
        raise FileNotFoundError(f"P6 discovery prompt file not found: {prompt_path}")
    prompt_hash = sha256_file(prompt_path)
    if prompt_hash != prompt_spec["sha256"]:
        raise ValueError("P6 discovery prompt checksum mismatch")
    rows = [
        json.loads(line)
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids = [str(row["prompt_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("P6 discovery prompt IDs must be unique")
    counts = {
        category: sum(row["category"] == category for row in rows)
        for category in ("risk", "sports_control", "neutral")
    }
    if counts != prompt_spec["counts"] or counts != {
        "risk": 12,
        "sports_control": 12,
        "neutral": 12,
    }:
        raise ValueError("P6 discovery pilot must contain exactly 12/12/12 prompts")
    paired = {
        category: {row["pair_id"] for row in rows if row["category"] == category}
        for category in ("risk", "sports_control")
    }
    if paired["risk"] != paired["sports_control"] or None in paired["risk"]:
        raise ValueError("risk and sports-control discovery prompts must have matched pair IDs")
    candidate = config["candidate_source"]
    if candidate["hard_coded_component_ids"] or candidate["maximum_candidates"] > 3:
        raise ValueError("P6 pilot candidates must come dynamically from at most three shortlist rows")
    intervention = config["intervention"]
    if intervention["alpha"] != [0.0, 0.25, 0.5, 1.0]:
        raise ValueError("P6 pilot alpha grid differs from the approved contract")
    if intervention["arms"] != [
        "clean",
        "learned",
        "gate_off",
        "active_matched",
        "orthogonal_matched",
        "random_same_l2",
    ]:
        raise ValueError("P6 pilot intervention arms differ from the approved contract")
    if len(set(intervention["random_seeds"])) < 2:
        raise ValueError("P6 random control requires multiple unique seeds")
    ready = config["scientific_status"] == "READY"
    placeholders = [
        config["checkpoint"]["rank0_path"],
        config["checkpoint"]["expected_sha256"],
        candidate["shortlist_path"],
        candidate["package_manifest"],
    ]
    if ready and any(str(value).startswith("REQUIRED_") for value in placeholders):
        raise ValueError("READY P6 config may not contain handoff placeholders")
    if require_ready and not ready:
        raise RuntimeError("P6 pilot remains BLOCKED until handoff paths and hashes are frozen")
    return {
        "status": "READY" if ready else "BLOCKED",
        "prompt_dataset_id": prompt_spec["id"],
        "prompt_sha256": prompt_hash,
        "category_counts": counts,
        "matched_pair_count": len(paired["risk"]),
        "maximum_candidates": candidate["maximum_candidates"],
        "random_control_seeds": intervention["random_seeds"],
        "auto_launch": config["execution"]["auto_launch"],
    }


@dataclass(frozen=True)
class OriginalP6ForwardResult:
    logits: Tensor
    matrix_diagnostics: Mapping[str, Mapping[str, Any]]
    matrix_outputs: Mapping[str, Mapping[str, Tensor]]


def load_100m_candidate_shortlist(
    path: str | Path,
    *,
    package_manifest: str | Path | None = None,
    maximum: int | None = None,
) -> dict[str, Any]:
    """Load current handoff candidates and retain exact file provenance."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"100M candidate shortlist not found: {source}")
    source_sha256 = sha256_file(source)
    package: dict[str, Any] | None = None
    if package_manifest is not None:
        manifest_path = Path(package_manifest)
        package = json.loads(manifest_path.read_text(encoding="utf-8"))
        if package.get("status") != "P6_HANDOFF_READY":
            raise ValueError("candidate package is not P6_HANDOFF_READY")
        relative = source.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
        entries = [row for row in package.get("artifacts", []) if row.get("path") == relative]
        if len(entries) != 1 or entries[0].get("sha256") != source_sha256:
            raise ValueError("candidate shortlist is absent from or mismatched with package manifest")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "component_id" not in rows[0]:
        raise ValueError("candidate shortlist must contain component_id")
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for rank, row in enumerate(rows, 1):
        component_id = int(row["component_id"])
        if not 0 <= component_id < 36_864 or component_id in seen:
            raise ValueError("candidate component IDs must be unique and lie in [0,36864)")
        seen.add(component_id)
        candidates.append({"rank": rank, "component_id": component_id, "source_row": row})
        if maximum is not None and len(candidates) == maximum:
            break
    return {
        "scientific_status": "OBSERVATIONAL_CANDIDATE_SHORTLIST",
        "source_path": str(source.resolve()),
        "source_sha256": source_sha256,
        "package_manifest": str(Path(package_manifest).resolve()) if package_manifest else None,
        "package_status": package.get("status") if package else None,
        "candidates": candidates,
    }


def validate_100m_checkpoint_contract(
    cfg: MultiModelExperimentConfig,
    checkpoint_rank0: str | Path,
    *,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Validate rank-0 provenance before any source model or hook is used."""

    validate_original_recipe_config(cfg)
    path = Path(checkpoint_rank0)
    observed = sha256_file(path)
    if observed != expected_checkpoint_sha256:
        raise ValueError("100M P6 checkpoint hash differs from the handoff contract")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if int(payload.get("step", -1)) != 24_414:
        raise ValueError("100M P6 requires canonical step 24,414 rank-0 checkpoint")
    if payload.get("config_sha256") != resolved_config_sha256(cfg):
        raise ValueError("100M P6 checkpoint/config fingerprint mismatch")
    if "model" not in payload or "optimizer" not in payload:
        raise ValueError("100M P6 checkpoint is incomplete")
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "step": 24_414,
        "config_sha256": payload["config_sha256"],
        "trainer_rank": 0,
    }


@torch.no_grad()
def clean_reference_gate(
    decomposition: MultiModelASPD,
    *,
    base_token_ids: Tensor,
    target_token_ids: Tensor,
    clean_base_grounding: Tensor,
    clean_target_grounding: Tensor,
    valid_tokens: Tensor,
    component_ids: Sequence[int],
) -> Tensor:
    """Freeze the learned online gate from clean same-token Base/Target passes."""

    if not torch.equal(base_token_ids.detach().cpu(), target_token_ids.detach().cpu()):
        raise ValueError("Base and Target clean-reference passes must use identical token IDs")
    if clean_base_grounding.shape[:2] != base_token_ids.shape:
        raise ValueError("Base grounding does not match the clean prefix grid")
    if clean_target_grounding.shape[:2] != target_token_ids.shape:
        raise ValueError("Target grounding does not match the clean prefix grid")
    return freeze_clean_original_gate(
        decomposition,
        [clean_base_grounding, clean_target_grounding],
        valid_tokens,
        component_ids,
    )


def _orthogonal_same_l2(delta: Tensor) -> Tensor:
    flat = delta.float().reshape(-1, delta.shape[-1])
    output = torch.zeros_like(flat)
    norms = flat.norm(dim=-1)
    active = norms > 0
    if not bool(active.any()):
        return output.reshape_as(delta).to(delta.dtype)
    reference = flat[active]
    denominator = reference.square().sum(dim=-1, keepdim=True)
    active_indices = active.nonzero(as_tuple=False).flatten()
    filled = torch.zeros(reference.shape[0], dtype=torch.bool, device=reference.device)
    for shift in range(1, min(16, flat.shape[-1])):
        proposal = reference.roll(shift, dims=-1)
        proposal -= (proposal * reference).sum(dim=-1, keepdim=True) / denominator * reference
        proposal_norm = proposal.norm(dim=-1, keepdim=True)
        usable = proposal_norm[:, 0] > torch.finfo(proposal.dtype).eps
        choose = usable & ~filled
        if bool(choose.any()):
            selected = proposal[choose] * (norms[active][choose, None] / proposal_norm[choose])
            output[active_indices[choose]] = selected
            filled |= choose
    if not bool(filled.all()):
        raise ValueError("could not construct an orthogonal actual-output control")
    return output.reshape_as(delta).to(delta.dtype)


def _random_same_l2(delta: Tensor, seed: int, identity: str) -> Tensor:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("random control seed must be an integer in [0,2**63)")
    derived = int.from_bytes(
        hashlib.sha256(f"p6-output-v2|{identity}|{seed}".encode()).digest()[:8], "big"
    ) & (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(derived)
    sampled = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
    sampled = sampled.to(delta.device)
    norms = delta.float().norm(dim=-1, keepdim=True)
    sampled_norms = sampled.norm(dim=-1, keepdim=True)
    matched = sampled * (norms / sampled_norms.clamp_min(torch.finfo(torch.float32).tiny))
    matched = torch.where(norms > 0, matched, torch.zeros_like(matched))
    return matched.to(delta.dtype)


def output_matched_control(
    candidate_delta: Tensor,
    *,
    arm: P6Arm,
    identity: str,
    seed: int | None = None,
    active_component_delta: Tensor | None = None,
) -> Tensor:
    """Construct controls from the actual live ``o_proj`` perturbation tensor.

    Zero-norm candidate positions always remain exactly zero.  They are never
    replaced by a nonzero direction merely to make a control look active.
    """

    if not torch.isfinite(candidate_delta).all():
        raise FloatingPointError("candidate output perturbation contains NaN or Inf")
    if arm == "learned":
        return candidate_delta
    if arm in {"clean", "gate_off"}:
        return torch.zeros_like(candidate_delta)
    if arm == "orthogonal_matched":
        return _orthogonal_same_l2(candidate_delta)
    if arm == "random_same_l2":
        if seed is None:
            raise ValueError("random_same_l2 requires an explicit seed")
        return _random_same_l2(candidate_delta, seed, identity)
    if arm == "active_matched":
        if active_component_delta is None or active_component_delta.shape != candidate_delta.shape:
            raise ValueError("active_matched requires a shape-matched alternate contribution")
        target_norm = candidate_delta.float().norm(dim=-1, keepdim=True)
        control_norm = active_component_delta.float().norm(dim=-1, keepdim=True)
        impossible = (target_norm > 0) & (control_norm == 0)
        if bool(impossible.any()):
            raise ValueError("active control is zero where the candidate perturbation is nonzero")
        scale = torch.where(control_norm > 0, target_norm / control_norm, torch.zeros_like(control_norm))
        return (active_component_delta.float() * scale).to(candidate_delta.dtype)
    raise ValueError(f"unknown P6 arm: {arm}")


def _component_delta(
    x: Tensor,
    u: Tensor,
    v: Tensor,
    gate: Tensor,
    center: Tensor | None,
) -> Tensor:
    x32 = x.float()
    if center is not None:
        x32 = x32 - center.to(x32)
    reads = torch.einsum("btd,kd->btk", x32, v.float())
    return torch.einsum("btk,ko->bto", reads * gate.to(reads.dtype), u.float())


@torch.inference_mode()
def run_original_p6_arm(
    model: nn.Module,
    spec: ModelSpec,
    decomposition: MultiModelASPD,
    model_index: int,
    input_ids: Tensor,
    valid_tokens: Tensor,
    clean_gate: Tensor,
    component_id: int,
    *,
    arm: P6Arm,
    alpha: float,
    matched_component_id: int | None = None,
    random_seed: int | None = None,
) -> OriginalP6ForwardResult:
    """Patch the actual live matrix output and remove every hook unconditionally."""

    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("P6 alpha must be finite and non-negative")
    if arm == "active_matched" and matched_component_id is None:
        raise ValueError("active_matched requires matched_component_id")
    if clean_gate.shape != (*input_ids.shape, 1):
        raise ValueError("clean gate must have shape [batch,tokens,1]")
    device = next(model.parameters()).device
    ids = input_ids.detach().clone().to(device)
    valid = valid_tokens.detach().clone().bool().to(device)
    gate = clean_gate.detach().clone().bool().to(device)
    diagnostics: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Tensor]] = {}
    selected = torch.tensor([component_id], device=next(decomposition.parameters()).device)
    matched = (
        torch.tensor([matched_component_id], device=selected.device)
        if matched_component_id is not None
        else None
    )

    with ExitStack() as stack:
        for matrix in spec.matrices:
            u_all, v_all = decomposition.component_factors(model_index, matrix.name)
            center = decomposition.component_input_center(model_index, matrix.name)
            u, v = u_all.index_select(0, selected), v_all.index_select(0, selected)
            matched_u = matched_v = None
            if matched is not None:
                matched_u = u_all.index_select(0, matched)
                matched_v = v_all.index_select(0, matched)

            def patch(
                _module: nn.Module,
                args: tuple[object, ...],
                output: object,
                *,
                name: str = matrix.name,
                site: str = matrix.module,
                u_local: Tensor = u,
                v_local: Tensor = v,
                center_local: Tensor | None = center,
                matched_u_local: Tensor | None = matched_u,
                matched_v_local: Tensor | None = matched_v,
            ) -> Tensor:
                x = _tensor_from_value(args, 0, site)
                y = _tensor_from_value(output, None, site)
                candidate = _component_delta(x, u_local, v_local, gate, center_local) * float(alpha)
                alternate = None
                if matched_u_local is not None and matched_v_local is not None:
                    alternate = (
                        _component_delta(x, matched_u_local, matched_v_local, gate, center_local)
                        * float(alpha)
                    )
                perturbation = output_matched_control(
                    candidate,
                    arm=arm,
                    identity=(
                        f"model={spec.name}|matrix={name}|component={component_id}"
                        f"|matched={matched_component_id}"
                    ),
                    seed=random_seed,
                    active_component_delta=alternate,
                )
                changed = y if arm == "clean" or alpha == 0 else y - perturbation.to(y.dtype)
                selected_perturbation = perturbation[valid]
                selected_output = y.detach().float()[valid]
                perturbation_l2 = float(selected_perturbation.float().norm())
                output_l2 = float(selected_output.norm())
                changed_positions = (
                    (perturbation.float().norm(dim=-1) > 0) & valid
                ).nonzero().detach().cpu().tolist()
                diagnostics[name] = {
                    "arm": arm,
                    "alpha": float(alpha),
                    "valid_token_count": int(valid.sum()),
                    "gate_fire_count": int((gate[..., 0] & valid).sum()),
                    "actual_perturbation_l2": perturbation_l2,
                    "activation_relative_l2": perturbation_l2 / output_l2 if output_l2 else 0.0,
                    "clean_output_l2": output_l2,
                    "modified_positions": changed_positions,
                    "zero_norm_policy": "candidate-zero positions remain exactly zero",
                }
                outputs[name] = {
                    "clean": y.detach().cpu().clone(),
                    "perturbation": perturbation.detach().cpu().clone(),
                    "intervened": changed.detach().cpu().clone(),
                }
                return changed

            handle = model.get_submodule(matrix.module).register_forward_hook(patch)
            stack.callback(handle.remove)
        result = model(input_ids=ids, attention_mask=valid, use_cache=False)

    if set(diagnostics) != {matrix.name for matrix in spec.matrices}:
        raise RuntimeError("one or more P6 matrix hooks did not fire")
    if not torch.isfinite(result.logits).all():
        raise FloatingPointError("P6 forward produced NaN or Inf")
    return OriginalP6ForwardResult(result.logits.detach(), diagnostics, outputs)


def make_p6_result_record(
    *,
    run_id: str,
    scientific_status: Literal[
        "OBSERVATIONAL", "SYNTHETIC_TEST_ONLY", "CAUSAL_EXPERIMENT_DATA"
    ],
    cfg: MultiModelExperimentConfig,
    checkpoint: Mapping[str, Any],
    prompt_dataset: Mapping[str, Any],
    shortlist: Mapping[str, Any],
    component_id: int,
    model_index: int,
    arm: P6Arm,
    alpha: float,
    gate_policy: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    matrix_output_references: Mapping[str, str],
    random_seed: int | None = None,
    generation_seed: int | None = None,
    continuation_references: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create a v2 machine record without inventing behavioral outcomes."""

    validate_original_recipe_config(cfg)
    model = cfg.models[model_index]
    return {
        "schema_version": 2,
        "scientific_status": scientific_status,
        "run_id": run_id,
        "source_models": [
            {"name": item.name, "model_id": item.model_id, "revision": item.revision}
            for item in cfg.models
        ],
        "aspd": {
            "config_sha256": resolved_config_sha256(cfg),
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_step": checkpoint["step"],
        },
        "prompt_dataset": dict(prompt_dataset),
        "candidate": {
            "component_id": int(component_id),
            "shortlist_sha256": shortlist["source_sha256"],
            "model": model.name,
            "layer": 14,
            "matrix": "model.layers.14.self_attn.o_proj",
        },
        "gate_policy": dict(gate_policy),
        "intervention": {
            "arm": arm,
            "alpha": float(alpha),
            "random_seed": random_seed,
            "actual_perturbation_l2": diagnostics["actual_perturbation_l2"],
            "activation_relative_l2": diagnostics["activation_relative_l2"],
            "modified_positions": diagnostics["modified_positions"],
        },
        "rng_seeds": {
            "generation": generation_seed,
            "random_control": random_seed,
        },
        "matrix_output_references": dict(matrix_output_references),
        "continuations": dict(continuation_references or {}),
        "behavioral_metrics": None,
        "limitations": [
            "Behavioral metrics remain null until the preregistered scorer runs.",
            "Learned JumpReLU inference is an online approximation to training BatchTopK.",
        ],
    }
