"""Measure all active representative targets for matched Negation query anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
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
    _bundle_target,
    _gate_label,
    _supports,
)
from aspd.multimodel.cli.prepare_aspd_v3_negation import read_contract
from aspd.multimodel.posthoc.artifacts import runtime_provenance, sha256_file, write_json
from aspd.multimodel.posthoc.comparison import edge_weight
from aspd.multimodel.posthoc.edges import edge_scores_for_group
from aspd.multimodel.posthoc.ids import SITES, ComponentId, GateId
from aspd.multimodel.posthoc.parents import Parent, parent_set
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend
from aspd.multimodel.v3.induction import MODEL_ORDER, subset_pool, task_pool_from_verified
from aspd.multimodel.v3.negation import FACTORIAL


def load_inputs(config, preparation, discovery):
    spec, reference = read_contract(config)
    policy = spec["graph"]
    if (policy["k_par"], policy["tau"]) != (PARENT_RULE.k_par, PARENT_RULE.tau):
        raise ValueError("parent rule differs from approved formulation")
    measured = verify_evaluation(discovery, "discovery")
    prepared, _, _ = load_preparation(preparation, config, "discovery")
    ref = Path(reference["components"]["reference_posthoc_root"])
    provenance = {
        "checkpointSha256": measured["checkpoint_sha256"],
        "discoveryManifestSha256": sha256_file(discovery / "manifest.json"),
        "preparationManifestSha256": sha256_file(preparation / "manifest.json"),
        "configSha256": sha256_file(config),
        "importanceCompleteSha256": sha256_file(ref / "importance/COMPLETE"),
        "representativesCompleteSha256": sha256_file(ref / "representatives/COMPLETE"),
    }
    for key, name in (
        ("preparationManifestSha256", "preparation_sha256"),
        ("configSha256", "config_sha256"),
        ("importanceCompleteSha256", "reference_importance_sha256"),
        ("representativesCompleteSha256", "reference_representatives_sha256"),
    ):
        if provenance[key] != measured[name]:
            raise ValueError(f"{key} differs from discovery")
    importance, reps = _load_references(ref)
    metadata = _anchor_metadata(prepared)
    prepared = {row["final_anchor_id"]: row for row in prepared}
    grouped = defaultdict(dict)
    seen = set()
    for row in jsonl(discovery / "anchor_activations.jsonl"):
        if row["final_query"]:
            anchor = row["anchor_id"]
            controls = grouped[row["group_id"]]
            if (
                anchor in seen
                or anchor not in prepared
                or any(row[k] != v for k, v in metadata[anchor].items())
                or row["condition"] in controls
            ):
                raise ValueError("query metadata differs from exact preparation")
            seen.add(anchor)
            controls[row["condition"]] = row
    if seen != set(prepared):
        raise ValueError("discovery and preparation query anchor sets differ")
    groups = {key: [controls[condition] for condition in FACTORIAL] for key, controls in grouped.items()}
    for rows in groups.values():
        if len({row["pool_id"] for row in rows}) != 1:
            raise ValueError("matched conditions must share the original natural pool")
        for row in rows:
            for site in SITES:
                values = _supports(row)[site]
                if (
                    len(values) != len(row[site]["active_gate_ids"])
                    or any(not 0 <= feature < 8192 for feature in values)
                    or any(not math.isfinite(value) or value <= 0 for value in values.values())
                ):
                    raise ValueError("invalid archived natural support")
    return groups, prepared, importance, reps, reference, measured, provenance


def active_gates(row):
    return {
        GateId(site, feature)
        for site, values in _supports(row).items()
        for feature in values
    }


def official_targets(rows, reps):
    return {
        (row["anchor_id"], component)
        for row in rows
        for gate in active_gates(row)
        for model in MODEL_ORDER
        for component in reps.get((gate, model), ())
    }


def plan_groups(groups, reps):
    families = defaultdict(list)
    for key, rows in sorted(groups.items()):
        active = [active_gates(row) for row in rows]
        families[rows[0]["family"]].append({
            "groupId": key,
            "family": rows[0]["family"],
            "anchors": [row["anchor_id"] for row in rows],
            "activeGateAnchors": sum(map(len, active)),
            "uniqueActiveGates": len(set().union(*active)),
            "representativeTargets": len(official_targets(rows, reps)),
            "gateModelPairsWithoutRepresentatives": sum(
                not reps.get((gate, model))
                for gates in active for gate in gates for model in MODEL_ORDER
            ),
        })
    medians = [
        sorted(items, key=lambda row: (row["representativeTargets"], row["groupId"]))[
            (len(items) - 1) // 2
        ]
        for _, items in sorted(families.items())
    ]
    benchmark = max(medians, key=lambda row: (row["representativeTargets"], row["groupId"]))
    return {
        "proposed": medians,
        "benchmark": benchmark,
        "selection": "lower median representative target count per family; highest workload median benchmark",
    }


def cache_contract(provenance):
    package = Path(__file__).resolve().parents[1]
    files = [
        "cli/measure_negation_comprehensive.py", "cli/measure_induction_query.py",
        "posthoc/backend.py", "posthoc/v31_backend.py", "posthoc/edges.py",
        "posthoc/source_hooks.py", "posthoc/parents.py", "posthoc/ids.py",
        "v3/induction.py",
    ]
    return {
        **provenance,
        "sourceSha256": {name: sha256_file(package / name) for name in files},
        "torch": torch.__version__,
        "cudaBuild": torch.version.cuda,
        "kPar": PARENT_RULE.k_par,
        "tau": PARENT_RULE.tau,
    }


def load_cache(db, namespace):
    db.execute("""CREATE TABLE IF NOT EXISTS records (
        namespace TEXT, anchor TEXT, component TEXT, payload TEXT,
        PRIMARY KEY (namespace, anchor, component)
    )""")
    return {
        (anchor, ComponentId.from_key(component)): json.loads(payload)
        for anchor, component, payload in db.execute(
            "SELECT anchor, component, payload FROM records WHERE namespace = ?", (namespace,)
        )
    }


def unpack(payload):
    coefficient, _, parents = payload
    return coefficient, tuple(Parent(ComponentId.from_key(key), score) for key, score in parents)


def measure_missing(backend, prepared, rows, requests):
    """Keep all four documents in the capture batch, including when filling cache misses."""
    anchors = [row["anchor_id"] for row in rows]
    task = task_pool_from_verified([prepared[anchor] for anchor in anchors], mechanism="negation")
    query = subset_pool(task, [task.anchor_ids.index(anchor) for anchor in anchors])
    inputs = backend._model_inputs(query)
    gates = {site: torch.zeros((len(rows), backend.n_features), dtype=torch.bool) for site in SITES}
    for index, row in enumerate(rows):
        for site, values in _supports(row).items():
            gates[site][index, list(values)] = True
    measured = {}
    for model in MODEL_ORDER:
        masks = {}
        for anchor, component in sorted(requests):
            if component.model == model:
                index = anchors.index(anchor)
                if not gates[component.gate.site][index, component.feature]:
                    raise ValueError(f"inactive component requested: {anchor} {component.key}")
                masks.setdefault(component, torch.zeros(len(rows), dtype=torch.bool))[index] = True
        if not masks:
            continue
        print(f"{model}: {sum(int(mask.sum()) for mask in masks.values())} cache misses", flush=True)
        factors = backend.factors(model)
        records = edge_scores_for_group(
            backend.source_model(model),
            inputs[model].to(backend.device),
            anchors,
            {site: gate.to(backend.device) for site, gate in gates.items()},
            backend.hook_specs(model),
            factors,
            {component: mask.to(backend.device) for component, mask in masks.items()},
        )
        for record in records:
            component = record.downstream
            norm = float(factors[component.matrix][0][component.feature].float().norm())
            if not math.isfinite(record.coefficient) or not math.isfinite(norm) or not bool(
                record.scores.isfinite().all()
            ):
                raise ValueError(f"non-finite Taylor result for {component.key}")
            parents = parent_set(record, PARENT_RULE)
            measured[record.anchor_id, component] = [
                record.coefficient, norm, [[parent.component.key, parent.score] for parent in parents]
            ]
    if set(measured) != requests:
        raise ValueError("returned Taylor record identities differ from requested identities")
    return measured


def trace_frontier(official, supplemental, cache, reps, importance):
    """Use only concrete parents already observed at the same anchor and model."""
    weights = {}
    for anchor, component in official | supplemental:
        _, norm, parents = cache[anchor, component]
        scale = importance[component.matrix].sigma
        for parent_key, effect in parents:
            parent = ComponentId.from_key(parent_key)
            if not parent.matrix.is_upstream_of(component.matrix):
                raise ValueError("cached parent violates upstream compute order")
            if parent not in reps.get((parent.gate, parent.model), ()):
                key = (anchor, parent)
                weights[key] = max(weights.get(key, 0.0), edge_weight(effect, norm, scale))
    return weights


def cached_trace_targets(official, cache, reps, importance):
    supplemental = set()
    while True:
        frontier = trace_frontier(official, supplemental, cache, reps, importance)
        reachable = frontier.keys() & cache.keys()
        if reachable == supplemental:
            return supplemental, frontier
        supplemental = reachable


def bundle_examples(rows, official, supplemental, cache, reps, importance):
    results = defaultdict(dict)
    norms = {}
    for anchor, component in official:
        payload = cache[anchor, component]
        results[anchor][component] = unpack(payload)
        norms[component] = payload[1]
    examples = []
    for row in rows:
        anchor = row["anchor_id"]
        graphs = {}
        for gate in sorted(active_gates(row)):
            graph = _bundle_graph(gate, row, results, norms, reps, importance)
            for model, parents in graph["models"].items():
                parents["status"] = "measured" if parents["targets"] else "no_reference_representatives"
                if not parents["targets"]:
                    parents["strength"] = None
                    parents["scores"] = None
            for pair in graph["comparisons"]:
                pair["available"] = all(graph["models"][model]["targets"] for model in pair["models"])
                if not pair["available"]:
                    pair["jaccard"] = None
                    pair["strength"] = [graph["models"][model]["strength"] for model in pair["models"]]
            graphs[_gate_label(gate)] = graph
        examples.append({
            "exampleId": row["example_id"],
            "groupId": row["group_id"],
            "family": row["family"],
            "condition": row["condition"],
            "anchorId": anchor,
            "poolId": row["pool_id"],
            "modelPositions": row["model_positions"],
            "graphs": graphs,
            "traceTargets": {
                component.key: _bundle_target(
                    component, unpack(cache[anchor, component]), cache[anchor, component][1], importance
                )
                for key, component in sorted(supplemental) if key == anchor
            },
        })
    return examples


def run(args, inputs, plan, selected):
    groups, prepared, importance, reps, reference, discovery, provenance = inputs
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    source = cache_contract(provenance)
    backend = None
    init_seconds = 0.0
    durations = {}
    requested = set()
    for target in args.target:
        anchor, sep, component = target.partition("=")
        if not sep or not anchor:
            raise ValueError("target must be ANCHOR=COMPONENT")
        requested.add((anchor, ComponentId.from_key(component)))
    digests = {}
    try:
        with closing(sqlite3.connect(args.cache, timeout=120)) as db:
            for group in selected:
                rows = groups[group]
                contract = {**source, "anchors": [row["anchor_id"] for row in rows]}
                namespace = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
                cache = load_cache(db, namespace)
                official = official_targets(rows, reps)
                if requested and not official <= cache.keys():
                    raise ValueError("targeted follow-up requires a completed comprehensive cache")

                def fill(targets):
                    nonlocal backend, init_seconds
                    missing = targets - cache.keys()
                    if not missing:
                        return 0, 0.0
                    if backend is None:
                        init = time.monotonic()
                        backend = V31CheckpointBackend(
                            backend_options(reference), work_dir=args.output / "stream_state"
                        )
                        if backend.provenance() != discovery["backend"]:
                            raise ValueError("loaded source model/checkpoint differs from discovery")
                        init_seconds = time.monotonic() - init
                    start = time.monotonic()
                    new = measure_missing(backend, prepared, rows, missing)
                    with db:
                        db.executemany(
                            "INSERT INTO records VALUES (?, ?, ?, ?)",
                            [(namespace, anchor, component.key, json.dumps(row))
                             for (anchor, component), row in new.items()],
                        )
                    cache.update(new)
                    return len(new), time.monotonic() - start

                official_count, official_seconds = fill(official)
                supplemental, frontier = cached_trace_targets(official, cache, reps, importance)
                trace_requests = requested
                if args.benchmark_trace:
                    strongest = {}
                    for key in sorted(frontier, key=lambda key: (-frontier[key], key)):
                        strongest.setdefault((key[0], key[1].model), key)
                    trace_requests = set(strongest.values())
                if not trace_requests <= frontier.keys():
                    raise ValueError("target is not a measured non-representative parent at this anchor")
                trace_count, trace_seconds = fill(trace_requests)
                supplemental, _ = cached_trace_targets(official, cache, reps, importance)
                durations[group] = {
                    "officialTargets": len(official),
                    "officialCacheMisses": official_count,
                    "officialSeconds": official_seconds,
                    "traceTargets": len(supplemental),
                    "traceCacheMisses": trace_count,
                    "traceSeconds": trace_seconds,
                    "cacheNamespace": namespace,
                }
                path = args.output / f"{group}.json"
                write_json(path, {
                    "schemaVersion": 1,
                    "mechanism": "negation",
                    "split": "discovery",
                    "groupId": group,
                    "checkpointSha256": provenance["checkpointSha256"],
                    "discoveryManifestSha256": provenance["discoveryManifestSha256"],
                    "examples": bundle_examples(rows, official, supplemental, cache, reps, importance),
                })
                digests[path.name] = sha256_file(path)
                print(json.dumps(durations[group], sort_keys=True), flush=True)
    finally:
        if backend is not None:
            backend.close()
    manifest = {
        **source,
        "mode": args.mode,
        "plan": plan,
        "groups": durations,
        "artifacts": digests,
        "runtime": runtime_provenance(),
        "initializationSeconds": init_seconds,
        "runSeconds": time.monotonic() - started,
        "gpuPeakAllocatedBytes": torch.cuda.max_memory_allocated() if backend else None,
        "gpuPeakReservedBytes": torch.cuda.max_memory_reserved() if backend else None,
        "hostPeakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cacheBytes": args.cache.stat().st_size,
        "groupArtifactBytes": sum((args.output / name).stat().st_size for name in digests),
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "COMPLETE", {"manifestSha256": sha256_file(args.output / "manifest.json")})
    print(json.dumps(manifest, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "preparation", "discovery"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "benchmark", "comprehensive", "trace"), default="plan")
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--benchmark-trace", action="store_true")
    parser.add_argument("--target", action="append", default=[])
    args = parser.parse_args()
    if args.mode != "plan" and (args.output is None or args.cache is None):
        parser.error("run modes require --output and --cache")
    if args.mode == "benchmark" and len(args.group) > 1:
        parser.error("benchmark permits one group")
    if args.mode == "comprehensive" and not 3 <= len(set(args.group)) == len(args.group) <= 5:
        parser.error("comprehensive requires 3–5 explicitly selected distinct groups")
    if args.mode == "trace" and (len(args.group) != 1 or not args.target):
        parser.error("trace requires one group and at least one explicitly requested target")
    if (args.mode != "trace" and args.target) or (args.mode != "benchmark" and args.benchmark_trace):
        parser.error("trace options must match the selected run mode")
    return args


def main():
    args = parse_args()
    inputs = load_inputs(args.config, args.preparation, args.discovery)
    groups, _, _, reps, *_ = inputs
    plan = plan_groups(groups, reps)
    print(json.dumps(plan, sort_keys=True), flush=True)
    if args.mode != "plan":
        selected = args.group or [plan["benchmark"]["groupId"]]
        if set(selected) - groups.keys():
            raise ValueError("requested group does not belong to verified Negation discovery")
        run(args, inputs, plan, selected)


if __name__ == "__main__":
    main()
