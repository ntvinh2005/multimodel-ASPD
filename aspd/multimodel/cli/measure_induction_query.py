"""Measure same-anchor parents for a bounded discovery-only induction shortlist."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from aspd.multimodel.cli.evaluate_aspd_v3_induction import (
    _anchor_metadata,
    _load_preparation,
    _load_references,
)
from aspd.multimodel.posthoc.artifacts import runtime_provenance, sha256_file, write_json
from aspd.multimodel.posthoc.comparison import (
    ModelParentView,
    compare_pair,
    edge_weight,
    parent_distribution,
)
from aspd.multimodel.posthoc.edges import edge_scores_for_group, group_rows
from aspd.multimodel.posthoc.ids import SITES, GateId
from aspd.multimodel.posthoc.parents import ParentRule, parent_set
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend
from aspd.multimodel.v3.induction import (
    CONDITIONS,
    MODEL_ORDER,
    subset_pool,
    task_pool_from_verified,
)

FAMILIES = ("basic", "repeated_sequence")
PARENT_RULE = ParentRule(k_par=10, tau=0)
SEED = 314159
GROUPS_PER_FAMILY = 4


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_discovery(root):
    manifest = _json(root / "manifest.json")
    complete = _json(root / "COMPLETE")
    if (
        manifest["split"] != "discovery"
        or manifest["heldout_opened"]
        or complete["status"] != "COMPLETE"
        or complete["manifest_sha256"] != sha256_file(root / "manifest.json")
    ):
        raise ValueError("requires an intact discovery run")
    for name in ("anchor_activations.jsonl", "activation_examples.jsonl", "pools.json"):
        if sha256_file(root / name) != manifest["artifacts"][name]:
            raise ValueError(f"discovery artifact changed: {name}")
    return manifest


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _supports(row):
    return {
        site: dict(zip(row[site]["active_gate_ids"], row[site]["values"], strict=True))
        for site in SITES
    }


def _gate_label(gate):
    return f"{gate.site.upper()}:{gate.feature}"


def _select(rows, representatives):
    grouped = defaultdict(dict)
    for row in rows.values():
        if row["family"] in FAMILIES:
            grouped[row["group_id"]][row["condition"]] = row
    scores = defaultdict(dict)
    for group, conditions in sorted(grouped.items()):
        if (
            set(conditions) != set(CONDITIONS)
            or len({r["pool_id"] for r in conditions.values()}) != 1
        ):
            raise ValueError(f"invalid matched query group: {group}")
        support = {condition: _supports(conditions[condition]) for condition in CONDITIONS}
        for site in SITES:
            features = set().union(*(value[site] for value in support.values()))
            for feature in features:
                gate = GateId(site, feature)
                if not all(representatives.get((gate, model)) for model in MODEL_ORDER):
                    continue
                by_condition = {
                    condition: support[condition][site].get(feature, 0.0)
                    for condition in CONDITIONS
                }
                clean = by_condition["induction"]
                counter = by_condition["counterfactual"]
                no_match = by_condition["no_match"]
                other = by_condition["repeated_no_association"]
                if clean > 0 and counter > 0:
                    scores[gate][group] = (
                        min(clean, counter) - max(no_match, other),
                        min(clean, counter, no_match, other),
                    )
    rng = random.Random(SEED)
    selected = {}
    for site in SITES:
        for kind, index, count in (("induction_contrast", 0, 4), ("fires_across_conditions", 1, 2)):
            ranked = []
            for gate, by_group in scores.items():
                if gate.site != site or gate in selected:
                    continue
                qualifying = {
                    family: [
                        gid
                        for gid, values in by_group.items()
                        if grouped[gid]["induction"]["family"] == family and values[index] > 0
                    ]
                    for family in FAMILIES
                }
                if all(len(groups) >= GROUPS_PER_FAMILY for groups in qualifying.values()):
                    # Paired differences must recur in both symbolic families.
                    score = min(
                        sum(by_group[gid][index] for gid in groups)
                        for groups in qualifying.values()
                    )
                    ranked.append((score, gate, qualifying))
            if len(ranked) < count:
                raise ValueError(f"insufficient eligible {site} gates for {kind}")
            for score, gate, qualifying in sorted(ranked, key=lambda x: (-x[0], x[1]))[:count]:
                groups = [
                    gid
                    for family in FAMILIES
                    for gid in sorted(rng.sample(sorted(qualifying[family]), GROUPS_PER_FAMILY))
                ]
                selected[gate] = {"kind": kind, "score": score, "groups": groups}
    return selected, grouped


def _measure(backend, task, examples, requested, representatives, importance):
    model_inputs = backend._model_inputs(task)
    gates = {
        site: torch.zeros((len(examples), backend.n_features), dtype=torch.bool) for site in SITES
    }
    for index, row in enumerate(examples):
        for site, support in _supports(row).items():
            if any(value <= 0 or not math.isfinite(value) for value in support.values()):
                raise ValueError("invalid saved BatchTopK firing")
            gates[site][index, list(support)] = True
    results = defaultdict(dict)
    norms = {}
    for model in MODEL_ORDER:
        inputs = model_inputs[model]
        factors = backend.factors(model)
        active_rows = [
            index
            for index, row in enumerate(examples)
            if any(gates[gate.site][index, gate.feature] for gate in requested[row["anchor_id"]])
        ]
        print(f"{model}: {len(active_rows)} anchors with active downstream targets", flush=True)
        for batch in group_rows(inputs.anchor_document, active_rows, max_group=4):
            masks = {}
            for local, index in enumerate(batch):
                for gate in sorted(requested[examples[index]["anchor_id"]]):
                    if gates[gate.site][index, gate.feature]:
                        for component in representatives[(gate, model)]:
                            masks.setdefault(component, torch.zeros(len(batch), dtype=torch.bool))[
                                local
                            ] = True
                            if component not in norms:
                                norms[component] = float(
                                    factors[component.matrix][0][component.feature].float().norm()
                                )
            records = edge_scores_for_group(
                backend.source_model(model),
                inputs.subset(batch).to(backend.device),
                [task.anchor_ids[index] for index in batch],
                {site: gate[batch].to(backend.device) for site, gate in gates.items()},
                backend.hook_specs(model),
                factors,
                {component: mask.to(backend.device) for component, mask in masks.items()},
            )
            for record in records:
                if not math.isfinite(record.coefficient) or not bool(
                    record.scores.isfinite().all()
                ):
                    raise ValueError(f"non-finite Taylor result for {record.downstream.key}")
                results[record.anchor_id][record.downstream] = (
                    record.coefficient,
                    parent_set(record, PARENT_RULE),
                )
    return results, norms


def _describe_component(component, importance):
    matrix = importance[component.matrix]
    beta = float(matrix.beta[component.feature])
    rho = float(matrix.rho[component.feature])
    return {
        "id": component.key,
        "modelId": component.model,
        "gateId": _gate_label(component.gate),
        "layer": component.matrix.layer,
        "matrix": component.matrix.matrix.removeprefix("mlp_")
        if component.matrix.matrix.startswith("mlp_")
        else component.matrix.matrix.upper(),
        "beta": beta if math.isfinite(beta) else None,
        "rho": rho if math.isfinite(rho) else None,
    }


def _bundle_graph(gate, row, results, norms, representatives, importance):
    anchor = row["anchor_id"]
    activation = _supports(row)[gate.site].get(gate.feature, 0.0)
    if activation == 0:
        return {"anchorId": anchor, "activation": 0, "models": {}, "comparisons": []}
    models = {}
    views = []
    for model in MODEL_ORDER:
        reps = representatives[(gate, model)]
        if any(rep not in results[anchor] for rep in reps):
            raise ValueError(f"missing active representative for {gate.key}/{anchor}/{model}")
        parents = {rep: results[anchor][rep][1] for rep in reps}
        sigmas = {rep: importance[rep.matrix].sigma for rep in reps}
        view = ModelParentView(model, reps, parents, norms, sigmas)
        views.append(view)
        models[model] = {
            "targets": [
                {
                    "component": _describe_component(rep, importance),
                    "coefficient": results[anchor][rep][0],
                    "parents": [
                        {
                            "component": _describe_component(p.component, importance),
                            "E": p.score,
                            "w": edge_weight(p.score, norms[rep], sigmas[rep]),
                        }
                        for p in parents[rep]
                    ],
                }
                for rep in reps
            ],
            "strength": sum(view.scores.values()),
            "distribution": {
                _gate_label(key): value
                for key, value in (parent_distribution(view.scores) or {}).items()
            }
            or None,
        }
    return {
        "anchorId": anchor,
        "activation": activation,
        "models": models,
        "comparisons": [
            compare_pair(gate, anchor, left, right).to_json()
            for left, right in itertools.combinations(views, 2)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "preparation", "discovery", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = _verified_discovery(args.discovery)
    prepared, preparation = _load_preparation(args.preparation, "discovery")
    reference = Path(config["components"]["reference_posthoc_root"])
    if (
        manifest["config_sha256"] != sha256_file(args.config)
        or manifest["preparation_manifest_sha256"]
        != sha256_file(args.preparation / "manifest.json")
        or preparation["checkpoint_manifest_sha256"] != config["checkpoint_manifest_sha256"]
        or sha256_file(Path(config["checkpoint"]) / "manifest.json")
        != config["checkpoint_manifest_sha256"]
        or manifest["reference_importance_complete_sha256"]
        != sha256_file(reference / "importance/COMPLETE")
        or manifest["reference_representatives_complete_sha256"]
        != sha256_file(reference / "representatives/COMPLETE")
        or _json(reference / "representatives/representatives.json")["rule"]
        != {"metric": "beta", "k_rep": 3, "tau_rep": 0.07}
    ):
        raise ValueError("discovery/reference provenance disagrees with the approved configuration")
    importance, representatives = _load_references(reference)
    meta = _anchor_metadata(prepared)
    final = {}
    for row in _rows(args.discovery / "anchor_activations.jsonl"):
        if row["final_query"]:
            anchor = row["anchor_id"]
            if anchor in final or any(row[key] != value for key, value in meta[anchor].items()):
                raise ValueError(f"query anchor metadata changed: {anchor}")
            final[anchor] = row
    if set(final) != {row["final_anchor_id"] for row in prepared}:
        raise ValueError("discovery and prepared query anchors differ")
    selected, grouped = _select(final, representatives)
    selection = {_gate_label(gate): value for gate, value in selected.items()}
    requested = defaultdict(set)
    for gate, value in selected.items():
        for group in value["groups"]:
            for row in grouped[group].values():
                requested[row["anchor_id"]].add(gate)
    print(json.dumps({"selection": selection, "unique_query_anchors": len(requested)}), flush=True)
    if args.plan_only:
        return
    chosen = [row for row in prepared if row["final_anchor_id"] in requested]
    task = task_pool_from_verified(chosen)
    task = subset_pool(
        task, [index for index, anchor in enumerate(task.anchor_ids) if anchor in requested]
    )
    options = {
        key: config[key]
        for key in (
            "training_config",
            "checkpoint",
            "config_file_sha256",
            "dataset_manifest_sha256",
            "source_dtype",
            "device",
            "local_files_only",
            "skip_training_launch_evidence",
        )
    }
    options.update(split="validation", pool_anchors=config["batch_topk"]["pool_anchors"])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "selection.json", selection)
    backend = V31CheckpointBackend(options, work_dir=args.output / "stream_state")
    try:
        if backend.provenance() != manifest["backend"]:
            raise ValueError("source backend provenance differs from the discovery run")
        results, norms = _measure(
            backend,
            task,
            [final[anchor] for anchor in task.anchor_ids],
            requested,
            representatives,
            importance,
        )
    finally:
        backend.close()

    chosen_ids = {final[anchor]["example_id"] for anchor in requested}
    saved_examples = {
        row["example_id"]: row
        for row in _rows(args.discovery / "activation_examples.jsonl")
        if row["example_id"] in chosen_ids
    }
    examples = [
        {
            "exampleId": row["example_id"],
            "groupId": row["group_id"],
            "family": row["family"],
            "condition": row["condition"],
            "poolId": row["pool_id"],
            "prompt": saved_examples[row["example_id"]]["prompt"],
            "expected": saved_examples[row["example_id"]]["expected_continuation"],
            "prediction": saved_examples[row["example_id"]]["native_predictions"],
            "modelPositions": row["model_positions"],
            "queryStart": min(
                span["start"]
                for span in saved_examples[row["example_id"]]["spans"]
                if span["role"] == "query_prefix" and span["stop"] == row["character_boundary"]
            ),
            "graphs": {
                _gate_label(gate): _bundle_graph(
                    gate, row, results, norms, representatives, importance
                )
                for gate in sorted(requested[anchor])
            },
        }
        for anchor, row in sorted(final.items())
        if anchor in requested
    ]
    provenance = {
        "runtime": runtime_provenance(),
        "wallSeconds": time.perf_counter() - started,
        "gpuPeakBytes": torch.cuda.max_memory_allocated(),
        "checkpointManifestSha256": config["checkpoint_manifest_sha256"],
        "discoveryManifestSha256": sha256_file(args.discovery / "manifest.json"),
        "preparationManifestSha256": manifest["preparation_manifest_sha256"],
        "configSha256": manifest["config_sha256"],
        "referenceImportanceSha256": manifest["reference_importance_complete_sha256"],
        "referenceRepresentativesSha256": manifest["reference_representatives_complete_sha256"],
        "seed": SEED,
        "groupsEachFamily": GROUPS_PER_FAMILY,
        "kPar": PARENT_RULE.k_par,
        "tau": PARENT_RULE.tau,
        "measuredComponents": sum(len(records) for records in results.values()),
        "scope": "final_query_same_anchor_frozen_discovery_support",
    }
    bundle = args.output / "inductionQueryGraphs.json"
    bundle.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "split": "discovery",
                "provenance": provenance,
                "selection": selection,
                "examples": examples,
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        args.output / "COMPLETE",
        {
            "bundleSha256": sha256_file(bundle),
            "selectionSha256": sha256_file(args.output / "selection.json"),
        },
    )


if __name__ == "__main__":
    main()
