"""Run C1 direct ASPD rank-one parameter interventions on frozen source models."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.cli.evaluate_aspd_v3_500m import (
    _finite_state_report,
    _parameter_signature,
)
from aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target import _validate_inputs
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.gate_collapse_recovery_config import (
    load_gate_collapse_recovery_config,
)
from aspd.multimodel.v3.hooks import assert_frozen_backbone, logical_weight_bias
from aspd.multimodel.v3.mechanistic_validation import (
    CausalValidationContract,
    norm_matched_random_rank_one,
    rank_one_operator,
    temporary_rank_one_ablation,
)
from aspd.multimodel.v3.model_owned import ModelOwnedMain2B
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.registry import LogicalMatrixTarget, build_model_registry_2b


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-key", choices=("gemma", "smol"), required=True)
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


def _safe_key(value: str) -> str:
    return value.replace(".", "__dot__")


def _select_target(
    model: ModelOwnedMain2B, feature: int, site: str
) -> tuple[LogicalMatrixTarget, Tensor, Tensor, list[dict[str, Any]]]:
    rows = []
    for target in model.targets.values():
        target_site = "pre" if target.grounding_kind == "residual_pre" else "mid"
        if target_site != site:
            continue
        component = model.matrix_bank.components[_safe_key(target.key)]
        u = component.U[feature].detach()
        v = component.V[feature].detach()
        norm = float(u.float().norm() * v.float().norm())
        rows.append({"target": target.key, "operator_frobenius_norm": norm})
    if not rows:
        raise ValueError(f"no {site} logical targets available")
    rows.sort(key=lambda row: (-float(row["operator_frobenius_norm"]), row["target"]))
    target = model.targets[rows[0]["target"]]
    component = model.matrix_bank.components[_safe_key(target.key)]
    return target, component.U[feature].detach(), component.V[feature].detach(), rows


def _random_component_factors(
    model: ModelOwnedMain2B,
    target: LogicalMatrixTarget,
    learned_feature: int,
    learned_norm: float,
    *,
    seed: int,
) -> tuple[int, Tensor, Tensor]:
    rng = random.Random(seed + learned_feature)
    component = model.matrix_bank.components[_safe_key(target.key)]
    candidates = list(range(int(component.U.shape[0])))
    candidates.remove(learned_feature)
    rng.shuffle(candidates)
    for feature in candidates:
        u = component.U[feature].detach().float()
        v = component.V[feature].detach().float()
        norm = float(u.norm() * v.norm())
        if norm > 0:
            return feature, u * (learned_norm / norm), v
    raise ValueError("all random component controls have zero operator norm")


def _tokenize(tokenizer: Any, contexts: list[dict[str, Any]], device: torch.device) -> dict[str, Tensor]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        [row["context"] for row in contexts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    return {key: value.to(device) for key, value in encoded.items()}


@torch.inference_mode()
def _last_logits(source_model: Any, encoded: dict[str, Tensor]) -> Tensor:
    return source_model(**encoded, use_cache=False).logits[:, -1].float()


def _metric_rows(clean: Tensor, edited: Tensor) -> list[dict[str, float | int]]:
    clean_logp = torch.log_softmax(clean, dim=-1)
    edited_logp = torch.log_softmax(edited, dim=-1)
    clean_probability = clean_logp.exp()
    kl = (clean_probability * (clean_logp - edited_logp)).sum(dim=-1)
    delta = edited - clean
    top = clean.argmax(dim=-1)
    clean_top = clean_probability.gather(1, top[:, None]).squeeze(1)
    edited_top = edited_logp.exp().gather(1, top[:, None]).squeeze(1)
    return [
        {
            "kl_clean_to_edited": float(kl[index]),
            "logit_delta_rms": float(delta[index].square().mean().sqrt()),
            "clean_top_token_id": int(top[index]),
            "clean_top_token_probability": float(clean_top[index]),
            "edited_probability_of_clean_top_token": float(edited_top[index]),
            "clean_top_token_probability_change": float(edited_top[index] - clean_top[index]),
            "maximum_absolute_logit_delta": float(delta[index].abs().max()),
        }
        for index in range(clean.shape[0])
    ]


def _evaluate_intervention(
    source_model: Any,
    target: LogicalMatrixTarget,
    u: Tensor,
    v: Tensor,
    *,
    strength: float,
    encoded: dict[str, Tensor],
    clean: Tensor,
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    with temporary_rank_one_ablation(
        source_model, target, u, v, strength=strength
    ) as metadata:
        edited = _last_logits(source_model, encoded)
    return metadata, _metric_rows(clean, edited)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["site"],
                row["context_activity"],
                row["intervention_kind"],
                float(row["strength"]),
            )
        ].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        site, activity, kind, strength = key
        result.append(
            {
                "site": site,
                "context_activity": activity,
                "intervention_kind": kind,
                "strength": strength,
                "n": len(values),
                "mean_kl_clean_to_edited": sum(
                    float(row["kl_clean_to_edited"]) for row in values
                )
                / len(values),
                "mean_logit_delta_rms": sum(float(row["logit_delta_rms"]) for row in values)
                / len(values),
                "mean_clean_top_token_probability_change": sum(
                    float(row["clean_top_token_probability_change"]) for row in values
                )
                / len(values),
            }
        )
    return {"groups": result}


def main() -> None:
    args = _args()
    contract = CausalValidationContract.load(args.contract)
    config = load_gate_collapse_recovery_config(args.config)
    if config.arm_id not in contract.arms or args.model_key not in contract.model_order:
        raise ValueError("causal arm/model is outside the frozen contract")
    if any("qwen" in model.model_id.lower() for model in config.models):
        raise RuntimeError("Qwen is forbidden in the causal screen")
    plan_payload = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan_payload.get("status") != "CAUSAL_SCREEN_PLAN_FROZEN":
        raise ValueError("causal plan is not frozen")
    if plan_payload.get("contract_sha256") != sha256_file(args.contract):
        raise ValueError("causal plan and contract differ")
    plans = [row for row in plan_payload["plans"] if row["arm"] == config.arm_id]
    if len(plans) != contract.components_per_arm:
        raise ValueError("causal plan arm budget changed")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite causal output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    validation_args = argparse.Namespace(
        config=args.config,
        contract=args.contract,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
    )
    provenance = _validate_inputs(validation_args, config)
    device = torch.device("cuda", 0)
    registry = build_model_registry_2b(config)
    layout = registry[args.model_key]
    tokenizer = load_v3_tokenizer(layout, local_files_only=args.local_files_only)
    loaded = load_frozen_v3_model(layout, device, local_files_only=args.local_files_only)
    assert_frozen_backbone(loaded.forward_model)
    loaded.forward_model.eval()
    source_before = _parameter_signature(loaded.forward_model)
    native_biases = {
        target.key: logical_weight_bias(loaded.forward_model, target)[1]
        for target in layout.targets()
    }
    model = ModelOwnedMain2B(
        config,
        registry,
        args.model_key,
        native_biases=native_biases,
        device=device,
        dtype=torch.float32,
        sparse_backend="dense",
    )
    manager = ModelOwnedCheckpoint(args.checkpoint.parent, keep_complete=2)
    checkpoint_owner_rank = list(config.architecture.model_order).index(args.model_key)
    runtime_state, checkpoint_manifest = manager.load_model_only_owner(
        path=args.checkpoint,
        model=model,
        config_fingerprint=config.fingerprint(),
        config_file_sha256=sha256_file(args.config),
        dataset_manifest_sha256=provenance["dataset_manifest_sha256"],
        checkpoint_rank=checkpoint_owner_rank,
        expected_checkpoint_world_size=len(config.architecture.model_order),
        expected_model_key=args.model_key,
    )
    model.eval().requires_grad_(False)
    finite = _finite_state_report(model)
    started = time.perf_counter()
    output_rows = []
    target_rows = []
    for component_plan in plans:
        feature = int(component_plan["feature"])
        site = component_plan["site"]
        target, learned_u, learned_v, target_ranking = _select_target(
            model, feature, site
        )
        learned_norm = float(rank_one_operator(learned_u, learned_v).norm())
        contexts = [
            {**row, "context_activity": activity}
            for activity, key in (("active", "active_contexts"), ("inactive", "inactive_contexts"))
            for row in component_plan[key]
        ]
        encoded = _tokenize(tokenizer, contexts, device)
        clean = _last_logits(loaded.forward_model, encoded)
        interventions: list[tuple[str, float, int | None, Tensor, Tensor]] = []
        interventions.extend(
            ("learned", float(strength), None, learned_u, learned_v)
            for strength in contract.learned_strengths
        )
        for seed in contract.random_rank_one_seeds:
            random_u, random_v = norm_matched_random_rank_one(
                d_out=target.d_out,
                d_in=target.d_in,
                target_frobenius_norm=learned_norm,
                seed=seed + feature,
                device=device,
            )
            interventions.append(("random_rank_one", 1.0, seed, random_u, random_v))
        random_feature, random_u, random_v = _random_component_factors(
            model,
            target,
            feature,
            learned_norm,
            seed=contract.random_component_seed,
        )
        interventions.append(
            ("random_component", 1.0, contract.random_component_seed, random_u, random_v)
        )
        target_rows.append(
            {
                "site": site,
                "feature": feature,
                "target": target.key,
                "target_selection_policy": contract.target_selection_policy,
                "learned_operator_frobenius_norm": learned_norm,
                "random_component_feature": random_feature,
                "target_ranking": target_ranking,
            }
        )
        for kind, strength, control_seed, u, v in interventions:
            metadata, metrics = _evaluate_intervention(
                loaded.forward_model,
                target,
                u,
                v,
                strength=strength,
                encoded=encoded,
                clean=clean,
            )
            for context, metric in zip(contexts, metrics, strict=True):
                output_rows.append(
                    {
                        "arm": config.arm_id,
                        "model": args.model_key,
                        "site": site,
                        "feature": feature,
                        "frequency_stratum": component_plan["frequency_stratum"],
                        "target": target.key,
                        "context_activity": context["context_activity"],
                        "anchor_id": context["anchor_id"],
                        "document_id": context["document_id"],
                        "intervention_kind": kind,
                        "strength": strength,
                        "control_seed": control_seed,
                        "intervention_metadata": metadata,
                        **metric,
                    }
                )
        zero_rows = [
            row
            for row in output_rows
            if row["feature"] == feature
            and row["intervention_kind"] == "learned"
            and row["strength"] == 0.0
        ]
        if any(float(row["maximum_absolute_logit_delta"]) != 0.0 for row in zero_rows):
            raise RuntimeError("zero-strength intervention changed model logits")
    if _parameter_signature(loaded.forward_model) != source_before:
        raise RuntimeError("source model changed after reversible causal screen")
    with (args.output_dir / "intervention_results.jsonl").open(
        "x", encoding="utf-8"
    ) as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    _write_json(args.output_dir / "target_selection.json", target_rows)
    summary = _summary(output_rows)
    _write_json(args.output_dir / "causal_summary.json", summary)
    _write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "status": "C1_DIRECT_PARAMETER_CAUSAL_SCREEN_COMPLETE",
            "arm": config.arm_id,
            "model": args.model_key,
            "source_weights_unchanged": "PASS",
            "zero_strength_invariance": "PASS",
            "checkpoint_step": int(checkpoint_manifest["step"]),
            "checkpoint_owner_rank": checkpoint_owner_rank,
            "runtime_state_step": int(runtime_state["completed_steps"]),
            "finite_aspd_state": finite,
            "contract_sha256": sha256_file(args.contract),
            "contract_fingerprint": contract.fingerprint(),
            "plan_sha256": sha256_file(args.plan),
            "config_sha256": provenance["config_sha256"],
            "checkpoint_manifest_sha256": provenance["checkpoint_manifest_sha256"],
            "model_id": layout.model_id,
            "model_revision": layout.revision,
            "git_sha": _git_sha(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "result_sha256": sha256_file(args.output_dir / "intervention_results.jsonl"),
            "target_selection_sha256": sha256_file(args.output_dir / "target_selection.json"),
        },
    )


if __name__ == "__main__":
    main()
