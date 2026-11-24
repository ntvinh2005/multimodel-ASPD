"""Prepare deterministic, exact-tokenizer induction datasets without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from aspd.multimodel.posthoc.v31_backend import load_frozen_config
from aspd.multimodel.v3.induction import (
    CONDITIONS,
    MODEL_ORDER,
    SPLITS,
    DatasetPlan,
    generate_examples,
    pack_verified_groups,
    shared_single_token_symbols,
    verify_example,
)
from aspd.multimodel.v3.models import load_v3_tokenizer
from aspd.multimodel.v3.registry import build_model_registry_2b


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite induction preparation: {args.output_dir}")
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if payload.get("status") != "RESEARCHER_AUTHORIZED_BOUNDED_EVALUATION":
        raise ValueError("induction evaluation config is not researcher-authorized")
    if not payload.get("forbidden", {}).get("qwen"):
        raise ValueError("Qwen exclusion must remain explicit")

    checkpoint = Path(payload["checkpoint"])
    if _sha256(checkpoint / "manifest.json") != payload["checkpoint_manifest_sha256"]:
        raise ValueError("checkpoint manifest hash differs from the authorized final checkpoint")
    frozen = load_frozen_config(
        payload["training_config"], checkpoint,
        skip_training_launch_evidence=bool(payload["skip_training_launch_evidence"]),
    )
    if tuple(frozen.architecture.model_order) != MODEL_ORDER:
        raise ValueError("final checkpoint model order differs from induction contract")
    if (
        int(frozen.architecture.dictionary_size) != payload["batch_topk"]["dictionary_size"]
        or int(frozen.architecture.batch_topk_k) != payload["batch_topk"]["k"]
        or int(frozen.architecture.pool_tokens) != payload["batch_topk"]["pool_anchors"]
    ):
        raise ValueError("final checkpoint C/K/pool differs from induction contract")
    registry = build_model_registry_2b(frozen)
    tokenizers = {
        key: load_v3_tokenizer(registry[key], local_files_only=bool(payload["local_files_only"]))
        for key in MODEL_ORDER
    }
    symbols = shared_single_token_symbols(tokenizers, model_order=MODEL_ORDER)
    if len(symbols) < 24:
        raise RuntimeError(
            f"only {len(symbols)} shared one-token symbols survived; need at least 24"
        )

    dataset = payload["dataset"]
    target_counts = {key: int(dataset["split_groups"][key]) for key in SPLITS}
    candidate_plan = DatasetPlan(
        seed=int(payload["seed"]),
        split_groups={
            key: target_counts[key] * int(dataset["oversample_factor"]) for key in SPLITS
        },
        family_weights={key: int(value) for key, value in dataset["family_weights"].items()},
        max_sequence_tokens=int(dataset["max_sequence_tokens"]),
        pool_anchors=int(payload["batch_topk"]["pool_anchors"]),
        task_anchor_budget_per_pool=int(
            payload["batch_topk"]["task_anchor_budget_per_pool"]
        ),
    )
    candidates = generate_examples(candidate_plan, symbols=symbols)
    by_group: dict[str, list[Any]] = {}
    for example in candidates:
        by_group.setdefault(example.group_id, []).append(example)

    selected: dict[str, list[dict[str, Any]]] = {key: [] for key in SPLITS}
    accepted_groups = Counter()
    rejections: list[dict[str, Any]] = []
    for group_id, group in by_group.items():
        split = group[0].split
        if accepted_groups[split] >= target_counts[split]:
            continue
        if tuple(example.condition for example in group) != CONDITIONS:
            raise ValueError(f"candidate group {group_id} lost paired condition order")
        verified = []
        try:
            for example in group:
                verified.append(
                    verify_example(
                        example, tokenizers, model_order=MODEL_ORDER,
                        max_length=candidate_plan.max_sequence_tokens,
                    )
                )
        except (ValueError, RuntimeError) as error:
            rejections.append(
                {"group_id": group_id, "split": split, "reason": str(error)}
            )
            continue
        selected[split].extend(verified)
        accepted_groups[split] += 1
    missing = {key: target_counts[key] - accepted_groups[key] for key in SPLITS}
    if any(value for value in missing.values()):
        raise RuntimeError(
            f"tokenizer validation exhausted the oversampled candidates; missing groups: {missing}"
        )

    args.output_dir.mkdir(parents=True)
    public_rows = selected["smoke"] + selected["discovery"]
    _write_jsonl(args.output_dir / "smoke_discovery.jsonl", public_rows)
    # Held-out prompts are physically separate and are not read by smoke/discovery execution.
    heldout_dir = args.output_dir / "heldout_sealed"
    heldout_dir.mkdir()
    _write_jsonl(heldout_dir / "heldout.jsonl", selected["heldout"])
    _write_jsonl(args.output_dir / "rejections.jsonl", rejections)

    pool_plan: dict[str, Any] = {"pool_anchors": candidate_plan.pool_anchors, "splits": {}}
    for split in SPLITS:
        packed = pack_verified_groups(
            selected[split], task_anchor_budget=candidate_plan.task_anchor_budget_per_pool
        )
        pool_plan["splits"][split] = [
            {
                "pool_index": index,
                "group_ids": list(dict.fromkeys(row["example"]["group_id"] for row in rows)),
                "example_ids": [row["example"]["example_id"] for row in rows],
                "task_anchor_count": sum(len(row["alignment"]["anchors"]) for row in rows),
                "background_anchor_count": candidate_plan.pool_anchors
                - sum(len(row["alignment"]["anchors"]) for row in rows),
            }
            for index, rows in enumerate(packed)
        ]
    _write_json(args.output_dir / "pool_plan.json", pool_plan)
    validation = {
        "status": "PASS",
        "model_order": list(MODEL_ORDER),
        "accepted_groups": dict(accepted_groups),
        "accepted_examples": {key: len(selected[key]) for key in SPLITS},
        "rejected_candidate_groups": len(rejections),
        "rejection_reasons": dict(Counter(row["reason"] for row in rejections)),
        "split_group_ids_disjoint": True,
        "association_generation": "globally disjoint deterministic signatures",
        "continuation_contract": "exactly one additional native token after the complete prompt",
        "shared_single_token_symbol_count": len(symbols),
        "shared_single_token_symbols": list(symbols),
        "alignment_contract": "exact canonical text boundaries under all three tokenizers",
    }
    _write_json(args.output_dir / "tokenizer_validation.json", validation)

    artifacts = [
        args.output_dir / "smoke_discovery.jsonl",
        heldout_dir / "heldout.jsonl",
        args.output_dir / "rejections.jsonl",
        args.output_dir / "pool_plan.json",
        args.output_dir / "tokenizer_validation.json",
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": payload["experiment_id"],
        "git_revision": _git_sha(),
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": payload["checkpoint_manifest_sha256"],
        "training_config_sha256": payload["config_file_sha256"],
        "training_dataset_manifest_sha256": payload["dataset_manifest_sha256"],
        "model_order": list(MODEL_ORDER),
        "models": {
            key: {
                "model_id": registry[key].model_id,
                "revision": registry[key].revision,
                "tokenizer_id": registry[key].tokenizer_id,
                "tokenizer_revision": registry[key].tokenizer_revision,
                "selected_layers": list(registry[key].layers),
            }
            for key in MODEL_ORDER
        },
        "dataset_plan": {
            "seed": payload["seed"],
            "target_split_groups": target_counts,
            "family_weights": dataset["family_weights"],
            "max_sequence_tokens": candidate_plan.max_sequence_tokens,
            "pool_anchors": candidate_plan.pool_anchors,
            "task_anchor_budget_per_pool": candidate_plan.task_anchor_budget_per_pool,
        },
        "heldout_access_policy": (
            "sealed in a separate artifact; smoke/discovery evaluator refuses this path"
        ),
        "artifacts": {str(path.relative_to(args.output_dir)): _sha256(path) for path in artifacts},
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "PREPARED").write_text(
        _sha256(args.output_dir / "manifest.json") + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
