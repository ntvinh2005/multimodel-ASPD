"""Evaluate a frozen, bounded Negation shortlist using the existing Induction edge runner."""

from __future__ import annotations

import argparse
import json
import random
import resource
import time
from collections import defaultdict
from pathlib import Path

import torch

from aspd.multimodel.cli.evaluate_aspd_v3_induction import _anchor_metadata, _load_references
from aspd.multimodel.cli.evaluate_aspd_v3_negation import (
    backend_options,
    jsonl,
    load_preparation,
    verify_evaluation,
)
from aspd.multimodel.cli.measure_induction_query import (
    PARENT_RULE,
    _bundle_graph,
    _gate_label,
    _measure,
    _supports,
)
from aspd.multimodel.cli.prepare_aspd_v3_negation import read_contract
from aspd.multimodel.posthoc.artifacts import runtime_provenance, sha256_file, write_json
from aspd.multimodel.posthoc.ids import SITES, GateId
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend
from aspd.multimodel.v3.induction import MODEL_ORDER, subset_pool, task_pool_from_verified
from aspd.multimodel.v3.negation import FACTORIAL


def select_shortlist(rows, reps, policy):
    grouped = defaultdict(dict)
    for row in rows.values():
        group = grouped[row["group_id"]]
        if row["condition"] in group:
            raise ValueError("duplicate group condition")
        group[row["condition"]] = row
    scores = defaultdict(lambda: defaultdict(dict))
    for group_id, group in sorted(grouped.items()):
        if not set(FACTORIAL) <= set(group) or len({r["pool_id"] for r in group.values()}) != 1:
            raise ValueError("matched conditions must share a single natural pool")
        supports = [_supports(group[condition]) for condition in FACTORIAL]
        for site in SITES:
            keys = set().union(*(s[site] for s in supports))
            for feature in keys:
                gate = GateId(site, feature)
                if not all(reps.get((gate, model)) for model in MODEL_ORDER):
                    continue
                aff, neg, cf_aff, cf_neg = [s[site].get(feature, 0.0) for s in supports]
                contrast = (neg + cf_neg - aff - cf_aff) / 2
                if contrast > 0 and min(neg, cf_neg) > 0:
                    scores[gate]["negative"][group_id] = contrast
                if contrast < 0 and min(aff, cf_aff) > 0:
                    scores[gate]["affirmative"][group_id] = -contrast
                if min(aff, neg, cf_aff, cf_neg) > 0:
                    scores[gate]["across"][group_id] = min(aff, neg, cf_aff, cf_neg)
    selected = {}
    eligible_counts = {}
    rng = random.Random(policy["seed"])
    for site in SITES:
        for category in ("negative", "affirmative", "across"):
            ranked = []
            for gate, labels in scores.items():
                if gate.site != site or gate in selected:
                    continue
                strengths = labels.get(category, {})
                families = defaultdict(list)
                for group_id in sorted(strengths):
                    families[grouped[group_id]["affirmative"]["family"]].append(group_id)
                if (
                    len(strengths) >= policy["minimum_qualifying_groups"]
                    and len(families) >= policy["minimum_qualifying_families"]
                ):
                    ranked.append((sum(strengths.values()), gate, families, len(strengths)))
            eligible_counts[f"{site}/{category}"] = len(ranked)
            for strength, gate, families, coverage in sorted(ranked, key=lambda x: (-x[0], x[1]))[
                : policy["gates_per_site"][category]
            ]:
                for options in families.values():
                    rng.shuffle(options)
                choices = []
                while len(choices) < min(coverage, policy["groups_per_gate"]):
                    for family in sorted(families):
                        if families[family] and len(choices) < policy["groups_per_gate"]:
                            choices.append(families[family].pop())
                selected[gate] = {
                    "kind": category,
                    "score": strength,
                    "eligible_groups": coverage,
                    "groups": choices,
                }
    return selected, grouped, eligible_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "preparation", "discovery", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    spec, reference = read_contract(args.config)
    policy = spec["graph"]
    if (policy["k_par"], policy["tau"]) != (PARENT_RULE.k_par, PARENT_RULE.tau):
        raise ValueError("parent rule differs from approved formulation")
    measured = verify_evaluation(args.discovery, "discovery")
    prepared, _plan, _manifest = load_preparation(args.preparation, args.config, "discovery")
    path = Path(reference["components"]["reference_posthoc_root"])
    if (
        measured["preparation_sha256"] != sha256_file(args.preparation / "manifest.json")
        or measured["config_sha256"] != sha256_file(args.config)
        or measured["reference_importance_sha256"] != sha256_file(path / "importance/COMPLETE")
        or measured["reference_representatives_sha256"]
        != sha256_file(path / "representatives/COMPLETE")
    ):
        raise ValueError("Negation and reference provenance differs from discovery activation run")
    importance, reps = _load_references(path)
    metadata = _anchor_metadata(prepared)
    final = {}
    for row in jsonl(args.discovery / "anchor_activations.jsonl"):
        if row["final_query"]:
            anchor = row["anchor_id"]
            if anchor in final or any(row[k] != v for k, v in metadata[anchor].items()):
                raise ValueError("query anchor metadata differs from exact preparation")
            final[anchor] = row
    if set(final) != {row["final_anchor_id"] for row in prepared}:
        raise ValueError("discovery final anchors and exact preparation differ")
    selected, grouped, eligible = select_shortlist(final, reps, policy)
    requested = defaultdict(set)
    for gate, details in selected.items():
        for group in details["groups"]:
            for row in grouped[group].values():
                requested[row["anchor_id"]].add(gate)
    selection = {
        "gates": {_gate_label(gate): row for gate, row in selected.items()},
        "policy": policy,
        "eligible_counts": eligible,
        "query_anchors": len(requested),
    }
    print(json.dumps(selection, sort_keys=True), flush=True)
    if args.plan_only:
        return
    if not selected:
        raise ValueError("no eligible downstream gates in discovery")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "selection.json", selection)
    task = task_pool_from_verified(
        [r for r in prepared if r["final_anchor_id"] in requested], mechanism="negation"
    )
    task = subset_pool(
        task, [index for index, anchor in enumerate(task.anchor_ids) if anchor in requested]
    )
    backend = V31CheckpointBackend(
        backend_options(reference), work_dir=args.output / "stream_state"
    )
    try:
        if backend.provenance() != measured["backend"]:
            raise ValueError("source model runtime differs from the native discovery measurement")
        results, norms = _measure(
            backend, task, [final[a] for a in task.anchor_ids], requested, reps, importance
        )
    finally:
        backend.close()
    examples = []
    for anchor, gates in sorted(requested.items()):
        row = final[anchor]
        examples.append(
            {
                "exampleId": row["example_id"],
                "groupId": row["group_id"],
                "family": row["family"],
                "condition": row["condition"],
                "poolId": row["pool_id"],
                "modelPositions": row["model_positions"],
                "graphs": {
                    _gate_label(gate): _bundle_graph(gate, row, results, norms, reps, importance)
                    for gate in sorted(gates)
                },
            }
        )
    bundle = args.output / "query_graphs.json"
    write_json(
        bundle,
        {
            "schemaVersion": 1,
            "mechanism": "negation",
            "split": "discovery",
            "examples": examples,
            "selection": selection["gates"],
            "provenance": {
                "scope": "final_query_same_anchor_frozen_discovery_support",
                "checkpointSha256": measured["checkpoint_sha256"],
                "discoveryManifestSha256": sha256_file(args.discovery / "manifest.json"),
                "preparationManifestSha256": measured["preparation_sha256"],
                "configSha256": measured["config_sha256"],
                "kPar": PARENT_RULE.k_par,
                "tau": PARENT_RULE.tau,
                "runtime": runtime_provenance(),
                "seconds": time.monotonic() - started,
                "gpuPeakBytes": torch.cuda.max_memory_allocated(),
                "hostPeakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "measuredComponents": sum(len(v) for v in results.values()),
            },
        },
    )
    write_json(
        args.output / "COMPLETE",
        {
            "bundle_sha256": sha256_file(bundle),
            "selection_sha256": sha256_file(args.output / "selection.json"),
        },
    )


if __name__ == "__main__":
    main()
