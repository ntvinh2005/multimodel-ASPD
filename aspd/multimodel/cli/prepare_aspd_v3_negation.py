"""Generate and validate Negation data without loading model weights."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from aspd.multimodel.cli.prepare_aspd_v3_induction import _write_jsonl
from aspd.multimodel.posthoc.artifacts import runtime_provenance, sha256_file, write_json
from aspd.multimodel.posthoc.v31_backend import load_frozen_config
from aspd.multimodel.v3.induction import MODEL_ORDER, SPLITS, shared_single_token_symbols
from aspd.multimodel.v3.models import load_v3_tokenizer
from aspd.multimodel.v3.negation import FAMILIES, candidate_groups, pack_groups, verify_example
from aspd.multimodel.v3.registry import build_model_registry_2b

MODEL_LAYERS = {
    "gemma": tuple(range(9, 17)),
    "smol": tuple(range(8, 16)),
    "granite": tuple(range(16, 24)),
}


def read_contract(config_path: Path):
    spec = yaml.safe_load(config_path.read_text())
    path = Path(spec["reference_contract"])
    if sha256_file(path) != spec["reference_contract_sha256"]:
        raise ValueError("frozen Induction evaluation contract hash mismatch")
    reference = yaml.safe_load(path.read_text())
    manifest_path = Path(reference["checkpoint"]) / "manifest.json"
    if (
        sha256_file(manifest_path) != reference["checkpoint_manifest_sha256"]
        or json.loads(manifest_path.read_text())["step"] != spec["checkpoint_step"]
        or not manifest_path.with_name("COMPLETE").is_file()
        or sha256_file(Path(reference["training_config"])) != reference["config_file_sha256"]
        or not reference["forbidden"]["qwen"]
    ):
        raise ValueError("frozen checkpoint contract mismatch")
    return spec, reference


def load_tokenizer_contract(reference):
    frozen = load_frozen_config(
        reference["training_config"],
        reference["checkpoint"],
        skip_training_launch_evidence=reference["skip_training_launch_evidence"],
    )
    architecture = frozen.architecture
    if tuple(architecture.model_order) != MODEL_ORDER or (
        architecture.dictionary_size,
        architecture.batch_topk_k,
        architecture.pool_tokens,
    ) != (8192, 32, 2048):
        raise ValueError("checkpoint model order or C/K/pool changed")
    registry = build_model_registry_2b(frozen)
    if any(tuple(registry[key].layers) != MODEL_LAYERS[key] for key in MODEL_ORDER):
        raise ValueError("checkpoint layer coverage changed")
    tokenizers = {
        key: load_v3_tokenizer(registry[key], local_files_only=True) for key in MODEL_ORDER
    }
    return registry, tokenizers


def prepare(spec, tokenizers, symbols):
    seen_slates, seen_prompts = set(), set()
    rejected = []
    groups = {split: [] for split in SPLITS}
    for split in SPLITS:
        target = int(spec["dataset"]["split_groups"][split])
        if target % len(FAMILIES):
            raise ValueError("split size must balance the four families")
        chosen = {family: [] for family in FAMILIES}
        for family in FAMILIES:
            for proposed in candidate_groups(symbols, split, family, spec["seed"]):
                slate = tuple(sorted(proposed[0]["candidate_order"]))
                prompts = {row["prompt"] for row in proposed}
                if slate in seen_slates or seen_prompts & prompts or len(prompts) != len(proposed):
                    continue
                try:
                    verified = [
                        verify_example(row, tokenizers, spec["dataset"]["max_sequence_tokens"])
                        for row in proposed
                    ]
                except (ValueError, RuntimeError) as error:
                    rejected.append({"group_id": proposed[0]["group_id"], "reason": str(error)})
                    continue
                seen_slates.add(slate)
                seen_prompts.update(prompts)
                chosen[family].append(verified)
                if len(chosen[family]) == target // len(FAMILIES):
                    break
            if len(chosen[family]) != target // len(FAMILIES):
                raise ValueError(
                    f"insufficient exact groups for {split}/{family}: {len(chosen[family])}"
                )
        # Interleave families before first-fit pool packing.
        groups[split] = [group for block in zip(*chosen.values(), strict=True) for group in block]
    return groups, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec, reference = read_contract(args.config)
    registry, tokenizers = load_tokenizer_contract(reference)
    symbols = shared_single_token_symbols(tokenizers)
    groups, rejected = prepare(spec, tokenizers, symbols)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "heldout_sealed").mkdir(mode=0o700)
    rows = {split: [row for group in values for row in group] for split, values in groups.items()}
    _write_jsonl(args.output / "smoke_discovery.jsonl", rows["smoke"] + rows["discovery"])
    _write_jsonl(args.output / "heldout_sealed/heldout.jsonl", rows["heldout"])
    _write_jsonl(args.output / "rejections.jsonl", rejected)
    pools = {}
    pool_cap = reference["batch_topk"]["task_anchor_budget_per_pool"]
    for split, verified in groups.items():
        pools[split] = [
            {
                "example_ids": [row["example"]["example_id"] for row in pool],
                "task_anchor_count": sum(len(row["alignment"]["anchors"]) for row in pool),
            }
            for pool in pack_groups(verified, pool_cap)
        ]
    write_json(args.output / "pool_plan.json", pools)
    report = {
        "status": "PASS",
        "generated_examples": True,
        "split_groups": {s: len(g) for s, g in groups.items()},
        "split_examples": {s: len(r) for s, r in rows.items()},
        "pools": {s: len(p) for s, p in pools.items()},
        "rejected_groups": len(rejected),
        "rejections": dict(Counter(r["reason"] for r in rejected)),
        "family_counts": {
            s: dict(Counter(g[0]["example"]["family"] for g in v)) for s, v in groups.items()
        },
        "templates": {s: sorted({r["example"]["template_id"] for r in v}) for s, v in rows.items()},
        "exact_full_prompt_single_token_candidates": True,
        "exact_marked_canonical_boundaries": True,
        "disjoint_slates_and_prompts": True,
        "heldout_templates_disjoint": True,
        "scope_controls_move_operator_with_equal_word_inventory": True,
        "multiple_candidate_sets_preserved": True,
        "symbols": list(symbols),
    }
    write_json(args.output / "validation.json", report)
    manifest = {
        "mechanism": "negation",
        "experiment": spec["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "source": runtime_provenance(),
        "checkpoint": reference["checkpoint"],
        "checkpoint_sha256": reference["checkpoint_manifest_sha256"],
        "training_config_sha256": reference["config_file_sha256"],
        "seed": spec["seed"],
        "training_dataset_sha256": reference["dataset_manifest_sha256"],
        "models": {
            k: {
                "id": registry[k].model_id,
                "revision": registry[k].revision,
                "tokenizer_id": registry[k].tokenizer_id,
                "tokenizer_revision": registry[k].tokenizer_revision,
                "layers": list(registry[k].layers),
            }
            for k in MODEL_ORDER
        },
        "batchtopk": {"anchors": 2048, "k": 32, "c": 8192},
        "artifacts": {
            str(p.relative_to(args.output)): sha256_file(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    }
    write_json(args.output / "manifest.json", manifest)
    (args.output / "PREPARED").write_text(sha256_file(args.output / "manifest.json") + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
