"""Run frozen three-model rank-one ablations with signed continuation likelihood."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.cli.evaluate_aspd_v3_500m import (
    _finite_state_report,
    _parameter_signature,
)
from aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target import _validate_inputs
from aspd.multimodel.cli.run_aspd_v3_causal_screen import (
    _random_component_factors,
    _select_target,
)
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.functional_correspondence import (
    FunctionalCorrespondenceContract,
    continuation_logprob_metrics,
)
from aspd.multimodel.v3.hooks import assert_frozen_backbone, logical_weight_bias
from aspd.multimodel.v3.main500m_config import load_main500m_config
from aspd.multimodel.v3.mechanistic_validation import (
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
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-key", choices=("gemma", "smol", "granite"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _batch(
    contexts: list[dict[str, Any]], tokenizer: Any, model_key: str, device: torch.device
) -> tuple[dict[str, Tensor], list[int], list[int]]:
    rows = [row["tokenizations"][model_key] for row in contexts]
    lengths = [len(row["input_ids"]) for row in rows]
    maximum = max(lengths)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError(f"{model_key} tokenizer has no padding or EOS token")
    input_ids = torch.full(
        (len(rows), maximum), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        input_ids[index, : len(values)] = values
        attention_mask[index, : len(values)] = 1
    return (
        {"input_ids": input_ids, "attention_mask": attention_mask},
        [int(row["prefix_token_count"]) for row in rows],
        [int(row["continuation_utf8_bytes"]) for row in contexts],
    )


@torch.inference_mode()
def _logits(source_model: Any, encoded: dict[str, Tensor]) -> Tensor:
    return source_model(**encoded, use_cache=False).logits.float()


def _evaluate(
    source_model: Any,
    target: LogicalMatrixTarget,
    u: Tensor,
    v: Tensor,
    *,
    strength: float,
    encoded: dict[str, Tensor],
    clean: Tensor,
    prefix_counts: list[int],
    byte_counts: list[int],
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    with temporary_rank_one_ablation(
        source_model, target, u, v, strength=strength
    ) as metadata:
        edited = _logits(source_model, encoded)
    metrics = continuation_logprob_metrics(
        clean,
        edited,
        encoded["input_ids"],
        encoded["attention_mask"],
        prefix_counts,
        byte_counts,
    )
    return metadata, metrics


def main() -> None:
    args = _args()
    execution_git_sha = _git_sha()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = FunctionalCorrespondenceContract.load(args.contract)
    config = load_main500m_config(args.config)
    if args.model_key not in contract.model_order:
        raise ValueError("model is outside the frozen correspondence contract")
    if any("qwen" in str(model.model_id).lower() for model in config.models):
        raise RuntimeError("Qwen is forbidden in the functional correspondence experiment")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("status") != "FUNCTIONAL_CORRESPONDENCE_PLAN_FROZEN":
        raise ValueError("functional correspondence plan is not frozen")
    if plan.get("contract_sha256") != sha256_file(args.contract):
        raise ValueError("functional correspondence plan and contract differ")
    if len(plan["plans"]) != 2 * contract.components_per_site:
        raise ValueError("functional correspondence plan budget changed")
    validation_args = argparse.Namespace(
        config=args.config,
        contract=args.contract,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
    )
    provenance = _validate_inputs(validation_args, config)
    if provenance["checkpoint_manifest_sha256"] != contract.checkpoint_manifest_sha256:
        raise ValueError("functional correspondence checkpoint hash changed")
    args.output_dir.mkdir(parents=True)
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
    checkpoint_owner_rank = contract.model_order.index(args.model_key)
    runtime_state, checkpoint_manifest = manager.load_model_only_owner(
        path=args.checkpoint,
        model=model,
        config_fingerprint=config.fingerprint(),
        config_file_sha256=sha256_file(args.config),
        dataset_manifest_sha256=provenance["dataset_manifest_sha256"],
        checkpoint_rank=checkpoint_owner_rank,
        expected_checkpoint_world_size=len(contract.model_order),
        expected_model_key=args.model_key,
    )
    model.eval().requires_grad_(False)
    finite = _finite_state_report(model)
    started = time.perf_counter()
    results = []
    targets = []
    for component in plan["plans"]:
        feature = int(component["feature"])
        site = str(component["site"])
        target, learned_u, learned_v, target_ranking = _select_target(model, feature, site)
        learned_norm = float(rank_one_operator(learned_u, learned_v).norm())
        contexts = [
            *component["active_contexts"],
            *component["inactive_contexts"],
        ]
        encoded, prefix_counts, byte_counts = _batch(
            contexts, tokenizer, args.model_key, device
        )
        clean = _logits(loaded.forward_model, encoded)
        interventions: list[tuple[str, float, int | None, Tensor, Tensor]] = [
            ("learned", float(strength), None, learned_u, learned_v)
            for strength in contract.learned_strengths
        ]
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
        targets.append(
            {
                "model": args.model_key,
                "site": site,
                "feature": feature,
                "target": target.key,
                "target_selection_policy": contract.target_selection_policy,
                "learned_operator_frobenius_norm": learned_norm,
                "random_component_feature": random_feature,
                "target_ranking": target_ranking,
            }
        )
        for kind, strength, seed, u, v in interventions:
            metadata, metrics = _evaluate(
                loaded.forward_model,
                target,
                u,
                v,
                strength=strength,
                encoded=encoded,
                clean=clean,
                prefix_counts=prefix_counts,
                byte_counts=byte_counts,
            )
            for context, metric in zip(contexts, metrics, strict=True):
                results.append(
                    {
                        "model": args.model_key,
                        "site": site,
                        "feature": feature,
                        "contribution_stratum": component["contribution_stratum"],
                        "target": target.key,
                        "context_activity": context["context_activity"],
                        "context_id": context["context_id"],
                        "anchor_id": context["anchor_id"],
                        "document_id": context["document_id"],
                        "intervention_kind": kind,
                        "strength": strength,
                        "control_seed": seed,
                        "intervention_metadata": metadata,
                        **metric,
                    }
                )
        zero_rows = [
            row
            for row in results
            if row["model"] == args.model_key
            and row["site"] == site
            and int(row["feature"]) == feature
            and row["intervention_kind"] == "learned"
            and float(row["strength"]) == 0.0
        ]
        if any(
            float(row["edited_minus_clean_continuation_logprob_total"]) != 0.0
            for row in zero_rows
        ):
            raise RuntimeError("zero-strength ablation changed continuation likelihood")
    if _parameter_signature(loaded.forward_model) != source_before:
        raise RuntimeError("source model changed after reversible correspondence experiment")
    result_path = args.output_dir / "intervention_results.jsonl"
    with result_path.open("x", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    _write_json(args.output_dir / "target_selection.json", targets)
    _write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "status": "FUNCTIONAL_CORRESPONDENCE_CAUSAL_RUN_COMPLETE",
            "experiment_family": contract.experiment_family,
            "model": args.model_key,
            "execution_git_sha": execution_git_sha,
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
            "tokenizer_revision": layout.tokenizer_revision,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "result_sha256": sha256_file(result_path),
            "target_selection_sha256": sha256_file(
                args.output_dir / "target_selection.json"
            ),
        },
    )


if __name__ == "__main__":
    main()
