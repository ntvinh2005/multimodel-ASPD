"""Read-only fixed-anchor attribution of two-model ASPD v3 gate collapse."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch import Tensor, nn

from aspd.multimodel.cli.evaluate_aspd_v3_encoder_logit_drift import (
    EncoderReplayState,
    _capture_residuals,
    _gather_tokens,
    _hidden_representation,
    _logit_summary,
    _pad_id,
    _shared_digest,
)
from aspd.multimodel.cli.train_aspd_v3_2b import _real_source_correctness_smoke
from aspd.multimodel.v3.activation_targets import prepare_activation_target
from aspd.multimodel.v3.data import HashSplitPolicy
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.diagnostic_stream import ValidationAlignedAnchorStream
from aspd.multimodel.v3.dictionary_concentration import FEATURE_BANDS
from aspd.multimodel.v3.gate_collapse_diagnostic import (
    cosine_similarity,
    feature_group_masks,
    masked_gradient_summary,
    summarize_feature_gradients,
)
from aspd.multimodel.v3.gating import V3BatchTopK
from aspd.multimodel.v3.logit_drift_diagnostic import factorial_shapley
from aspd.multimodel.v3.model_owned import (
    SITE_ORDER,
    LocalSiteEncoder,
    SharedSiteEncoder,
    _differentiable_all_gather,
)
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.objective import (
    auxk_activation_scores,
    auxk_dead_feature_loss,
    full_activation_fvu,
    matryoshka_activation_fvu,
)
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, load_pinned_pile_dataset
from aspd.multimodel.v3.reconstruction import ActivationReconstructionBank
from aspd.multimodel.v3.registry import ModelFamilyLayout, build_model_registry_2b
from aspd.multimodel.v3.sparse_support import prepare_sparse_support
from aspd.multimodel.v3.two_model_config import MODEL_ORDER_TWO
from aspd.multimodel.v3.two_model_matryoshka_config import (
    TwoModelMatryoshkaConfig,
    load_two_model_matryoshka_config,
)

FACTOR_NAMES = ("local_encoder", "shared_cross_encoder", "readout_weight", "readout_bias")
OBJECTIVES = ("main_activation", "auxk")


class GateGradientReplayState(nn.Module):
    """Checkpoint-compatible gate/activation subset without rank-one banks."""

    def __init__(
        self,
        config: TwoModelMatryoshkaConfig,
        layout: ModelFamilyLayout,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.local_encoders = nn.ModuleDict(
            {
                site: LocalSiteEncoder(
                    layout.hidden_size,
                    config,
                    device=device,
                    dtype=torch.float32,
                )
                for site in SITE_ORDER
            }
        )
        self.shared_encoders = nn.ModuleDict(
            {
                site: SharedSiteEncoder(
                    config,
                    model_order=tuple(config.architecture.model_order),
                    device=device,
                    dtype=torch.float32,
                )
                for site in SITE_ORDER
            }
        )
        self.gates = nn.ModuleDict(
            {
                site: V3BatchTopK(
                    config.architecture.dictionary_size,
                    config.architecture.batch_topk_k,
                    pool_anchors=config.architecture.pool_tokens,
                    device=device,
                    sync_statistics=False,
                    materialize_n_selected=False,
                    validate_nonnegative=False,
                )
                for site in SITE_ORDER
            }
        )
        self.activation_banks = nn.ModuleDict(
            {
                site: ActivationReconstructionBank(
                    {layout.key: [layout.hidden_size] * 8},
                    config.architecture.dictionary_size,
                    normalization_mode=config.objective.activation_decoder_normalization,
                    device=device,
                    dtype=torch.float32,
                )
                for site in SITE_ORDER
            }
        )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matryoshka-config", type=Path, required=True)
    parser.add_argument("--full-only-config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--matryoshka-run", type=Path, required=True)
    parser.add_argument("--full-only-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty diagnostic table: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _broadcast(value: Any) -> Any:
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "status": "RESEARCHER_AUTHORIZED_READ_ONLY_GATE_COLLAPSE_DIAGNOSTIC",
        "authorized_by": "researcher",
        "authorized_on": "2026-11-08",
        "experiment_family": "aspd_v3_1_gate_collapse_root_cause_v1",
        "source_experiment_family": "aspd_v3_gemma_smol_matryoshka_ablation_v1",
        "source_array_job_id": "50231651",
        "source_job_ids": {"matryoshka": "50231730", "full_only": "50231651"},
        "source_git_sha": "75e1444718617d983aa20c6a1e0c39da637be2dd",
        "checkpoint_step": 1000,
        "arms": ["matryoshka", "full_only"],
        "model_order": list(MODEL_ORDER_TWO),
        "split": "validation",
        "split_unit": "document",
        "pool_anchors": 2048,
        "pool_count": 1,
        "factor_order": list(FACTOR_NAMES),
        "counterfactual_count": 16,
        "gradient_objectives": list(OBJECTIVES),
        "gpu_count": 2,
        "gpu_type": "b200",
        "read_only": True,
        "optimizer_or_step_authorized": False,
        "checkpoint_creation_authorized": False,
        "source_weight_mutation_authorized": False,
    }
    if payload != expected:
        raise ValueError("two-model gate-collapse diagnostic contract changed")
    return payload


def _canonical_rows(path: Path, through_step: int) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            step = int(row["step"])
            if step <= through_step:
                by_step[step] = row
    if sorted(by_step) != list(range(1, through_step + 1)):
        raise ValueError(f"source metrics are incomplete through step {through_step}: {path}")
    return [by_step[step] for step in range(1, through_step + 1)]


def _validate_arm(
    arm: str,
    config_path: Path,
    config: TwoModelMatryoshkaConfig,
    run_dir: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = run_dir / "checkpoints" / "step-000001000"
    required = (
        config_path,
        run_dir / "run_manifest.json",
        run_dir / "metrics.jsonl",
        checkpoint / "manifest.json",
        checkpoint / "COMPLETE",
        Path(config.runtime.dataset_manifest),
        Path(config.runtime.alignment_manifest),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"gate-collapse source inputs missing: {missing}")
    if config.arm_id != arm or tuple(config.architecture.model_order) != MODEL_ORDER_TWO:
        raise ValueError("two-model diagnostic arm/model order changed")
    if any(
        "qwen" in value.lower()
        for model in config.models
        for value in (model.model_id, model.tokenizer_id)
    ):
        raise ValueError("Qwen is forbidden in the two-model gate-collapse diagnostic")
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_path = checkpoint / "manifest.json"
    checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads((checkpoint / "COMPLETE").read_text(encoding="utf-8"))
    dataset = json.loads(Path(config.runtime.dataset_manifest).read_text(encoding="utf-8"))
    alignment = json.loads(Path(config.runtime.alignment_manifest).read_text(encoding="utf-8"))
    config_sha = sha256_file(config_path)
    expected_run = {
        "git_sha": contract["source_git_sha"],
        "config_fingerprint": config.fingerprint(),
        "config_file_sha256": config_sha,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "alignment_manifest_sha256": alignment["manifest_sha256"],
        "model_order": list(MODEL_ORDER_TWO),
        "logical_target_count": 112,
    }
    mismatch = {
        key: {"expected": value, "observed": run.get(key)}
        for key, value in expected_run.items()
        if run.get(key) != value
    }
    slurm_job = str(run.get("slurm_job_id", ""))
    if slurm_job != contract["source_job_ids"][arm]:
        mismatch["slurm_job_id"] = {
            "expected": contract["source_job_ids"][arm],
            "observed": slurm_job,
        }
    expected_checkpoint = {
        "status": "COMPLETE",
        "step": contract["checkpoint_step"],
        "world_size": 2,
        "config_fingerprint": config.fingerprint(),
        "config_file_sha256": config_sha,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
    }
    mismatch.update(
        {
            f"checkpoint.{key}": {"expected": value, "observed": checkpoint_manifest.get(key)}
            for key, value in expected_checkpoint.items()
            if checkpoint_manifest.get(key) != value
        }
    )
    checkpoint_sha = sha256_file(manifest_path)
    if marker.get("manifest_sha256") != checkpoint_sha:
        mismatch["checkpoint.marker"] = {
            "expected": checkpoint_sha,
            "observed": marker.get("manifest_sha256"),
        }
    if mismatch:
        raise ValueError(f"{arm} gate-collapse provenance mismatch: {mismatch}")
    rows = _canonical_rows(run_dir / "metrics.jsonl", contract["checkpoint_step"])
    return {
        "arm": arm,
        "config": config,
        "config_path": config_path,
        "run_dir": run_dir,
        "run_manifest": run,
        "checkpoint": checkpoint,
        "checkpoint_manifest_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "alignment_manifest_sha256": alignment["manifest_sha256"],
        "metrics_sha256": sha256_file(run_dir / "metrics.jsonl"),
        "rows": rows,
    }


def _validate_matched(arms: Mapping[str, Mapping[str, Any]]) -> None:
    left = arms["matryoshka"]
    right = arms["full_only"]
    for key in ("dataset_manifest_sha256", "alignment_manifest_sha256"):
        if left[key] != right[key]:
            raise ValueError(f"matched arm provenance differs at {key}")
    left_initial = left["run_manifest"]["initial_common_state_by_rank"]
    right_initial = right["run_manifest"]["initial_common_state_by_rank"]
    if left_initial != right_initial:
        raise ValueError("matched arms do not share identical initial trainable states")
    for left_row, right_row in zip(left["rows"], right["rows"], strict=True):
        if left_row["ordered_anchor_ids_sha256"] != right_row["ordered_anchor_ids_sha256"]:
            raise ValueError(f"matched arm batches differ at step {left_row['step']}")


def _synchronize_initial_shared(state: EncoderReplayState) -> None:
    with torch.no_grad():
        for value in state.shared_encoders.state_dict().values():
            dist.broadcast(value, src=0)


def _load_trained_state(
    arm: Mapping[str, Any],
    layout: ModelFamilyLayout,
    local_key: str,
    *,
    device: torch.device,
) -> tuple[GateGradientReplayState, dict[str, Any]]:
    config = arm["config"]
    state = GateGradientReplayState(config, layout, device=device)
    manager = ModelOwnedCheckpoint(arm["checkpoint"].parent, keep_complete=2)
    runtime, manifest = manager.load_model_only(
        path=arm["checkpoint"],
        model=state,
        config_fingerprint=config.fingerprint(),
        config_file_sha256=arm["config_sha256"],
        dataset_manifest_sha256=arm["dataset_manifest_sha256"],
        expected_model_key=local_key,
    )
    if int(runtime["completed_steps"]) != 1000 or int(manifest["step"]) != 1000:
        raise ValueError("gate-collapse replay loaded the wrong checkpoint step")
    state.eval()
    digest = _shared_digest(state)  # type: ignore[arg-type]
    digests: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(digests, digest)
    if len(set(digests)) != 1:
        raise ValueError("trained shared encoder differs across owner ranks")
    for site in SITE_ORDER:
        counts = state.gates[site].feature_fire_counts.detach().clone()
        minimum = counts.clone()
        maximum = counts.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if not torch.equal(minimum, maximum):
            raise ValueError("checkpointed shared gate statistics differ across owner ranks")
    return state, {"completed_steps": 1000, "shared_encoder_sha256": digest}


def _module_swap_replay(
    arm: str,
    initial: EncoderReplayState,
    trained: GateGradientReplayState,
    residuals: Mapping[str, Sequence[Tensor]],
    valid: Tensor,
    config: TwoModelMatryoshkaConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states: tuple[Any, Any] = (initial, trained)
    gathered: dict[tuple[int, str], dict[str, Tensor]] = {}
    representations: dict[tuple[int, int, str], Tensor] = {}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for local_index, state in enumerate(states):
            for site in SITE_ORDER:
                local = state.local_encoders[site](residuals[site])
                gathered[(local_index, site)] = _gather_tokens(local, MODEL_ORDER_TWO)
        for local_index in range(2):
            for shared_index, state in enumerate(states):
                for site in SITE_ORDER:
                    representations[(local_index, shared_index, site)] = _hidden_representation(
                        state.shared_encoders[site], gathered[(local_index, site)]
                    )

    reports: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for site in SITE_ORDER:
        scalar_negative: dict[tuple[int, ...], Tensor] = {}
        scalar_candidates: dict[tuple[int, ...], Tensor] = {}
        scalar_mean: dict[tuple[int, ...], Tensor] = {}
        feature_negative: dict[tuple[int, ...], Tensor] = {}
        feature_mean: dict[tuple[int, ...], Tensor] = {}
        combinations: dict[str, Any] = {}
        endpoint_logits: dict[int, Tensor] = {}
        endpoint_selection: dict[str, Any] = {}
        for local_index in range(2):
            for shared_index in range(2):
                hidden = representations[(local_index, shared_index, site)].float()
                for weight_index in range(2):
                    weight = states[weight_index].shared_encoders[site].readout.output.weight.float()
                    for bias_index in range(2):
                        bias = states[bias_index].shared_encoders[site].readout.output.bias.float()
                        key = (local_index, shared_index, weight_index, bias_index)
                        logits = hidden @ weight.T + bias
                        serialized = "".join(str(value) for value in key)
                        combinations[serialized] = _logit_summary(logits)
                        scalar_negative[key] = (logits < 0).float().mean()
                        scalar_candidates[key] = (logits > 0).sum(dim=-1).float().mean()
                        scalar_mean[key] = logits.mean()
                        feature_negative[key] = (logits < 0).float().mean(dim=(0, 1)).cpu()
                        feature_mean[key] = logits.mean(dim=(0, 1)).cpu()
                        if len(set(key)) == 1:
                            endpoint = key[0]
                            endpoint_logits[endpoint] = logits.detach()
                            controller = V3BatchTopK(
                                config.architecture.dictionary_size,
                                config.architecture.batch_topk_k,
                                pool_anchors=config.architecture.pool_tokens,
                                device=logits.device,
                                sync_statistics=False,
                                materialize_n_selected=False,
                                validate_nonnegative=False,
                            )
                            code = controller(torch.relu(logits), valid, update_statistics=False)
                            selected = code.gate[valid]
                            endpoint_selection[str(endpoint)] = {
                                "positive_firings_per_anchor": float(
                                    selected.sum(dim=-1).float().mean()
                                ),
                                "distinct_features_firing": int(selected.any(dim=0).sum()),
                            }
        shapley_negative = factorial_shapley(scalar_negative)
        shapley_candidates = factorial_shapley(scalar_candidates)
        shapley_mean = factorial_shapley(scalar_mean)
        shapley_feature_negative = factorial_shapley(feature_negative)
        shapley_feature_mean = factorial_shapley(feature_mean)
        counts = trained.gates[site].feature_fire_counts.detach().cpu()
        dead = trained.gates[site].dead_features.detach().cpu()
        start = endpoint_logits[0]
        end = endpoint_logits[1]
        deaths = ((start > 0) & (end <= 0)).float().mean(dim=(0, 1)).cpu()
        revivals = ((start <= 0) & (end > 0)).float().mean(dim=(0, 1)).cpu()
        for feature in range(config.architecture.dictionary_size):
            band = next(
                band.name for band in FEATURE_BANDS if band.lower <= feature < band.upper
            )
            row: dict[str, Any] = {
                "arm": arm,
                "site": site,
                "feature": feature,
                "band": band,
                "training_fire_count_step_1000": int(counts[feature]),
                "dead_at_step_1000": bool(dead[feature]),
                "negative_fraction_initial": float(feature_negative[(0, 0, 0, 0)][feature]),
                "negative_fraction_step_1000": float(feature_negative[(1, 1, 1, 1)][feature]),
                "mean_logit_initial": float(feature_mean[(0, 0, 0, 0)][feature]),
                "mean_logit_step_1000": float(feature_mean[(1, 1, 1, 1)][feature]),
                "positive_to_nonpositive_probability": float(deaths[feature]),
                "nonpositive_to_positive_probability": float(revivals[feature]),
            }
            for index, factor in enumerate(FACTOR_NAMES):
                row[f"shapley_negative_fraction_{factor}"] = float(
                    shapley_feature_negative[index][feature]
                )
                row[f"shapley_mean_logit_{factor}"] = float(
                    shapley_feature_mean[index][feature]
                )
            feature_rows.append(row)
        total_negative = float(scalar_negative[(1, 1, 1, 1)] - scalar_negative[(0, 0, 0, 0)])
        total_candidates = float(
            scalar_candidates[(1, 1, 1, 1)] - scalar_candidates[(0, 0, 0, 0)]
        )
        total_mean = float(scalar_mean[(1, 1, 1, 1)] - scalar_mean[(0, 0, 0, 0)])
        reports.append(
            {
                "arm": arm,
                "site": site,
                "factor_order": list(FACTOR_NAMES),
                "combinations": combinations,
                "shapley_negative_fraction": {
                    factor: float(shapley_negative[index])
                    for index, factor in enumerate(FACTOR_NAMES)
                },
                "shapley_positive_candidates_per_anchor": {
                    factor: float(shapley_candidates[index])
                    for index, factor in enumerate(FACTOR_NAMES)
                },
                "shapley_mean_logit": {
                    factor: float(shapley_mean[index])
                    for index, factor in enumerate(FACTOR_NAMES)
                },
                "shapley_additivity_error": {
                    "negative_fraction": abs(total_negative - sum(float(x) for x in shapley_negative)),
                    "positive_candidates_per_anchor": abs(
                        total_candidates - sum(float(x) for x in shapley_candidates)
                    ),
                    "mean_logit": abs(total_mean - sum(float(x) for x in shapley_mean)),
                },
                "endpoint_batch_topk": endpoint_selection,
                "positive_to_nonpositive_fraction": float(
                    ((start > 0) & (end <= 0)).float().mean()
                ),
                "nonpositive_to_positive_fraction": float(
                    ((start <= 0) & (end > 0)).float().mean()
                ),
            }
        )
    return reports, feature_rows


def _forward_site(
    state: GateGradientReplayState,
    site: str,
    residuals: Mapping[str, Sequence[Tensor]],
    valid: Tensor,
) -> tuple[Tensor, Tensor, Any, Any]:
    local = state.local_encoders[site](residuals[site])
    gathered = _differentiable_all_gather(local, expected_world_size=len(MODEL_ORDER_TWO))
    tokens = {key: gathered[index] for index, key in enumerate(MODEL_ORDER_TWO)}
    scores, logits = state.shared_encoders[site].forward_with_logits(tokens, valid)
    logits.retain_grad()
    sparse = state.gates[site](scores, valid, update_statistics=False)
    support = prepare_sparse_support(sparse.values, valid, binary=False)
    return scores, logits, sparse, support


def _activation_objective(
    state: GateGradientReplayState,
    config: TwoModelMatryoshkaConfig,
    local_key: str,
    site: str,
    residuals: Mapping[str, Sequence[Tensor]],
    valid: Tensor,
    objective: str,
) -> tuple[Tensor, Tensor, Tensor, Any]:
    scores, logits, sparse, support = _forward_site(state, site, residuals, valid)
    auxiliary_scores = auxk_activation_scores(
        scores,
        logits,
        valid,
        mode=config.objective.auxk_score_source,
        softplus_temperature=config.objective.auxk_softplus_temperature,
    )
    coefficient = 0.5 / (len(MODEL_ORDER_TWO) * 8)
    terms = []
    for decoder, target in zip(
        state.activation_banks[site].decoders[local_key], residuals[site], strict=True
    ):
        prepared = prepare_activation_target(
            target,
            valid,
            mode=config.objective.activation_target_normalization,
            epsilon=config.objective.activation_target_epsilon,
        )
        activation_function = (
            matryoshka_activation_fvu
            if config.objective.activation_objective_mode == "matryoshka"
            else full_activation_fvu
        )
        activation, full = activation_function(
            sparse.values,
            decoder,
            prepared.target,
            valid,
            support=support,
            sparse_backend="dense",
        )
        auxiliary = auxk_dead_feature_loss(
            auxiliary_scores,
            decoder,
            prepared.target,
            full,
            valid,
            state.gates[site].dead_features,
            top_k_aux=config.architecture.top_k_aux,
            sparse_backend="dense",
        )
        if objective == "main_activation":
            terms.append(coefficient * activation)
        elif objective == "auxk":
            terms.append(coefficient * config.architecture.auxk_coefficient * auxiliary)
        else:
            raise ValueError(f"unknown gradient objective: {objective}")
    return sum(terms), scores, logits, sparse


def _flatten_gradients(module: nn.Module) -> Tensor:
    values = []
    for parameter in module.parameters():
        values.append(
            torch.zeros_like(parameter).reshape(-1)
            if parameter.grad is None
            else parameter.grad.detach().reshape(-1)
        )
    if not values:
        raise ValueError("cannot flatten gradients for a parameter-free module")
    return torch.cat(values)


def _gradient_probe(
    arm: str,
    state: GateGradientReplayState,
    config: TwoModelMatryoshkaConfig,
    local_key: str,
    residuals: Mapping[str, Sequence[Tensor]],
    valid: Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        for site in SITE_ORDER:
            state.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _scores, logits, sparse = _activation_objective(
                    state, config, local_key, site, residuals, valid, objective
                )
            loss.backward()
            if logits.grad is None:
                raise RuntimeError("raw logits lost their diagnostic gradient")
            readout = state.shared_encoders[site].readout.output
            if readout.weight.grad is None or readout.bias.grad is None:
                raise RuntimeError("readout lost its diagnostic gradient")
            selected = sparse.gate
            dead = state.gates[site].dead_features.detach()
            counts = state.gates[site].feature_fire_counts.detach()
            feature_groups = feature_group_masks(logits.detach(), selected, dead, counts)
            row_norm = readout.weight.grad.detach().float().norm(dim=1)
            bias_abs = readout.bias.grad.detach().float().abs()
            positive = logits.detach() > 0
            event_groups = {
                "naturally_selected": selected,
                "positive_but_unselected": positive & ~selected,
                "negative_logit": logits.detach() < 0,
                "dead_positive": positive & dead.view(1, 1, -1),
            }
            dead_indices = dead.nonzero(as_tuple=True)[0]
            aux_selected = torch.zeros_like(selected)
            if dead_indices.numel():
                dead_scores = torch.relu(logits.detach())[..., dead_indices]
                count = min(config.architecture.top_k_aux, dead_indices.numel())
                top = torch.topk(dead_scores, count, dim=-1)
                chosen = dead_indices[top.indices]
                aux_selected.scatter_(-1, chosen, top.values > 0)
            event_groups["aux_selected_positive"] = aux_selected
            shared_gradient = _flatten_gradients(state.shared_encoders[site])
            gathered_shared = [torch.empty_like(shared_gradient) for _ in MODEL_ORDER_TWO]
            dist.all_gather(gathered_shared, shared_gradient)
            shared_norms = {
                key: float(value.float().norm())
                for key, value in zip(MODEL_ORDER_TWO, gathered_shared, strict=True)
            }
            decoder_gradient = _flatten_gradients(state.activation_banks[site])
            local_gradient = _flatten_gradients(state.local_encoders[site])
            summaries.append(
                {
                    "arm": arm,
                    "model": local_key,
                    "objective": objective,
                    "site": site,
                    "weighted_local_loss": float(loss.detach()),
                    "raw_logit_summary": _logit_summary(logits.detach()),
                    "logit_gradient_by_event": {
                        name: masked_gradient_summary(logits.grad, mask)
                        for name, mask in event_groups.items()
                    },
                    "readout_weight_gradient_by_feature": summarize_feature_gradients(
                        row_norm, feature_groups
                    ),
                    "readout_bias_gradient_by_feature": summarize_feature_gradients(
                        bias_abs, feature_groups
                    ),
                    "local_encoder_gradient_norm": float(local_gradient.float().norm()),
                    "activation_decoder_gradient_norm": float(
                        decoder_gradient.float().norm()
                    ),
                    "shared_encoder_gradient_norm_by_model": shared_norms,
                    "shared_encoder_cross_model_cosine": cosine_similarity(
                        gathered_shared[0], gathered_shared[1]
                    ),
                }
            )
            logit_grad = logits.grad.detach().float()
            logit_rms = logit_grad.square().mean(dim=(0, 1)).sqrt()
            logit_nonzero = (logit_grad != 0).float().mean(dim=(0, 1))
            selected_count = selected.sum(dim=(0, 1))
            positive_fraction = positive.float().mean(dim=(0, 1))
            aux_selected_count = aux_selected.sum(dim=(0, 1))
            for feature in range(config.architecture.dictionary_size):
                feature_rows.append(
                    {
                        "arm": arm,
                        "model": local_key,
                        "objective": objective,
                        "site": site,
                        "feature": feature,
                        "training_fire_count_step_1000": int(counts[feature]),
                        "dead_at_step_1000": bool(dead[feature]),
                        "fixed_pool_positive_fraction": float(positive_fraction[feature]),
                        "fixed_pool_natural_selection_count": int(selected_count[feature]),
                        "fixed_pool_aux_positive_selection_count": int(
                            aux_selected_count[feature]
                        ),
                        "raw_logit_gradient_rms": float(logit_rms[feature]),
                        "raw_logit_gradient_nonzero_fraction": float(
                            logit_nonzero[feature]
                        ),
                        "readout_weight_gradient_norm": float(row_norm[feature]),
                        "readout_bias_gradient_absolute": float(bias_abs[feature]),
                    }
                )
    return summaries, feature_rows


def main() -> None:
    args = _args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise RuntimeError("two-model gate-collapse diagnostic requires exactly two ranks")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    started = time.perf_counter()

    contract = _load_contract(args.contract)
    configs = {
        "matryoshka": load_two_model_matryoshka_config(args.matryoshka_config),
        "full_only": load_two_model_matryoshka_config(args.full_only_config),
    }
    arms = {
        "matryoshka": _validate_arm(
            "matryoshka",
            args.matryoshka_config,
            configs["matryoshka"],
            args.matryoshka_run,
            contract,
        ),
        "full_only": _validate_arm(
            "full_only",
            args.full_only_config,
            configs["full_only"],
            args.full_only_run,
            contract,
        ),
    }
    _validate_matched(arms)
    if rank == 0:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite diagnostic output: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
    dist.barrier()

    config = configs["matryoshka"]
    registry = build_model_registry_2b(config)
    local_key = MODEL_ORDER_TWO[rank]
    torch.manual_seed(config.optimization.seed)
    torch.cuda.manual_seed_all(config.optimization.seed)
    tokenizer = load_v3_tokenizer(registry[local_key], local_files_only=args.local_files_only)
    loaded = load_frozen_v3_model(
        registry[local_key], device, local_files_only=args.local_files_only
    )
    alignment_tokenizers = None
    if rank == 0:
        alignment_tokenizers = {
            key: tokenizer
            if key == local_key
            else load_v3_tokenizer(registry[key], local_files_only=args.local_files_only)
            for key in MODEL_ORDER_TWO
        }
    source_smoke = _real_source_correctness_smoke(
        local_key, tokenizer, loaded, registry, device
    )
    source_smokes = [None] * world
    dist.all_gather_object(source_smokes, source_smoke)

    initial = EncoderReplayState(config, registry[local_key], device=device)
    _synchronize_initial_shared(initial)
    initial.eval().requires_grad_(False)
    initial_digest = _shared_digest(initial)
    initial_digests: list[str | None] = [None] * world
    dist.all_gather_object(initial_digests, initial_digest)
    if len(set(initial_digests)) != 1:
        raise ValueError("reconstructed initial shared encoder differs across ranks")

    pool_payload = None
    stream = None
    if rank == 0:
        dataset = load_pinned_pile_dataset(config.data.dataset_id, config.data.dataset_revision)
        assert alignment_tokenizers is not None
        stream = ValidationAlignedAnchorStream(
            dataset=dataset,
            dataset_id=config.data.dataset_id,
            dataset_revision=config.data.dataset_revision,
            tokenizers=alignment_tokenizers,
            tokenizer_revisions={
                key: registry[key].tokenizer_revision for key in MODEL_ORDER_TWO
            },
            split_policy=HashSplitPolicy(
                modulus=config.data.split_modulus,
                validation_buckets=frozenset(config.data.validation_buckets),
                test_buckets=frozenset(config.data.test_buckets),
                seed=config.data.split_seed,
            ),
            dedup_path=args.output_dir / "validation_stream_dedup.sqlite",
            sequence_length=config.data.sequence_length,
            model_order=MODEL_ORDER_TWO,
        )
        pool_payload = stream.next_pool(pool_anchors=contract["pool_anchors"]).to_payload()
    pool = AlignedAnchorPool.from_payload(_broadcast(pool_payload))
    pool.validate(
        expected_anchors=contract["pool_anchors"],
        sequence_length=config.data.sequence_length,
        model_order=MODEL_ORDER_TWO,
    )
    residuals = _capture_residuals(
        loaded.forward_model,
        registry[local_key],
        pool.models[local_key],
        pad_token_id=_pad_id(tokenizer),
        device=device,
    )
    del loaded
    torch.cuda.empty_cache()
    valid = torch.ones(1, pool.anchor_count, dtype=torch.bool, device=device)

    counterfactual_reports: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    local_gradient_summaries: list[dict[str, Any]] = []
    local_gradient_rows: list[dict[str, Any]] = []
    state_reports = {}
    for arm in contract["arms"]:
        trained, state_report = _load_trained_state(
            arms[arm], registry[local_key], local_key, device=device
        )
        state_reports[arm] = state_report
        reports, rows = _module_swap_replay(
            arm, initial, trained, residuals, valid, configs[arm]
        )
        counterfactual_reports.extend(reports)
        attribution_rows.extend(rows)
        summaries, gradient_rows = _gradient_probe(
            arm, trained, configs[arm], local_key, residuals, valid
        )
        local_gradient_summaries.extend(summaries)
        local_gradient_rows.extend(gradient_rows)
        del trained
        torch.cuda.empty_cache()

    _write_csv(
        args.output_dir / f"feature_gradients_{local_key}.csv", local_gradient_rows
    )
    gathered_summaries = [None] * world
    dist.all_gather_object(gathered_summaries, local_gradient_summaries)
    dist.barrier()
    if rank == 0:
        initial_endpoints = [
            report["combinations"]["0000"] for report in counterfactual_reports
        ]
        if any(
            not 0.4 <= endpoint["negative_fraction"] <= 0.6
            or not 3000 <= endpoint["positive_candidates_per_anchor"]["mean"] <= 5000
            for endpoint in initial_endpoints
        ):
            raise ValueError("reconstructed initial encoder does not reproduce a healthy gate state")
        _write_json(
            args.output_dir / "counterfactual_metrics.json",
            {"schema_version": 1, "reports": counterfactual_reports},
        )
        _write_csv(args.output_dir / "feature_attribution.csv", attribution_rows)
        gradient_summaries = [
            row for rank_rows in gathered_summaries for row in rank_rows
        ]
        _write_json(
            args.output_dir / "gradient_summary.json",
            {
                "schema_version": 1,
                "internal_gate_gradient_route": "detached_binary_support_zero_by_construction",
                "summaries": gradient_summaries,
            },
        )
        anchor_digest = hashlib.sha256(
            json.dumps(pool.anchor_ids, separators=(",", ":")).encode()
        ).hexdigest()
        output_artifacts = (
            "counterfactual_metrics.json",
            "feature_attribution.csv",
            "gradient_summary.json",
            "feature_gradients_gemma.csv",
            "feature_gradients_smol.csv",
        )
        manifest = {
            "schema_version": 1,
            "status": "READ_ONLY_TWO_MODEL_GATE_COLLAPSE_DIAGNOSTIC_COMPLETE",
            "diagnostic_git_sha": _git_sha(),
            "source_git_sha": contract["source_git_sha"],
            "source_array_job_id": contract["source_array_job_id"],
            "source_job_ids": contract["source_job_ids"],
            "source_checkpoint_step": contract["checkpoint_step"],
            "contract_sha256": sha256_file(args.contract),
            "model_order": list(MODEL_ORDER_TWO),
            "model_revisions": {
                model.key: model.revision for model in config.models
            },
            "tokenizer_revisions": {
                model.key: model.tokenizer_revision for model in config.models
            },
            "arms": {
                arm: {
                    "config_fingerprint": arms[arm]["config"].fingerprint(),
                    "config_sha256": arms[arm]["config_sha256"],
                    "checkpoint_manifest_sha256": arms[arm][
                        "checkpoint_manifest_sha256"
                    ],
                    "metrics_sha256": arms[arm]["metrics_sha256"],
                }
                for arm in contract["arms"]
            },
            "dataset_manifest_sha256": arms["matryoshka"]["dataset_manifest_sha256"],
            "alignment_manifest_sha256": arms["matryoshka"]["alignment_manifest_sha256"],
            "matched_initial_state": "PASS",
            "matched_ordered_batches_through_step_1000": "PASS",
            "initial_shared_encoder_sha256": initial_digest,
            "source_correctness_smokes": source_smokes,
            "state_reports": state_reports,
            "split": contract["split"],
            "accepted_anchors": pool.anchor_count,
            "anchor_ids_sha256": anchor_digest,
            "factor_order": list(FACTOR_NAMES),
            "counterfactual_count_per_arm_site": 16,
            "gradient_objectives": list(OBJECTIVES),
            "read_only": True,
            "optimizer_created": False,
            "optimizer_step_used": False,
            "backward_used_for_diagnostics_only": True,
            "checkpoint_created_or_mutated": False,
            "source_weights_frozen": True,
            "artifact_sha256": {
                name: sha256_file(args.output_dir / name) for name in output_artifacts
            },
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(args.output_dir / "experiment_manifest.json", manifest)
    if stream is not None:
        stream.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
