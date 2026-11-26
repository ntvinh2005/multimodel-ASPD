"""Benchmark and run three CPU discovery searches on one frozen same-anchor mmap index."""

import argparse
import csv
import json
import os
import random
import resource
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np

from aspd.multimodel.discovery.index import (
    CHECKPOINT,
    MODELS,
    Index,
    decode,
    encode,
    gate_of,
    heldout_natural_scopes,
    load,
    model_of,
    natural_scopes,
    sha,
    token_scopes,
    write_index,
)
from aspd.multimodel.discovery.search import (
    METHODS,
    Graph,
    bottom_up,
    composition,
    directed_gate_signature,
    estimate_sinks,
    matched_null,
    measured_paths,
    pareto,
    recurring,
    signature,
    structure,
    top_down,
)


def atomic_json(path, payload):
    with TemporaryDirectory(dir=path.parent, prefix=".write-") as tmp:
        stage = Path(tmp) / path.name
        stage.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
        stage.replace(path)


def load_seeds(paths, evidence_file):
    records = defaultdict(set)
    texts = defaultdict(dict)

    def add(label, rows):
        partition = defaultdict(list)
        for edge in rows:
            source, target = encode(edge["source"]), encode(edge["target"])
            if model_of(source) != model_of(target):
                raise ValueError("Circuit seed crosses models")
            partition[model_of(source)].append((source, target))
        for edges in partition.values():
            if len(edges) >= 2:
                records[tuple(sorted(set(edges)))].add(label)

    for path in paths:
        for circuit in load(path)["circuits"]:
            if circuit["checkpoint"] == CHECKPOINT:
                for measurement in circuit["snapshots"]:
                    if measurement["scope"]["checkpoint"] != CHECKPOINT:
                        raise ValueError("Seed snapshot checkpoint differs")
                    add(circuit["id"], measurement["edges"])
    for label, value in load(evidence_file).items():
        for snapshot in value["snapshots"]:
            add(label, snapshot["edges"])
        for gate, examples in value.get("examples", {}).items():
            for row in examples:
                texts[gate].setdefault(row["anchor"], row)
    templates = [(tuple(sorted(ids)), edges) for edges, ids in sorted(records.items())]
    return templates, texts


def task_batches(indices, size=64):
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def cpu_peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def all_code_hashes():
    root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(root)): sha(path)
        for path in (Path(__file__), *sorted((root / "discovery").glob("*.py")))
    }


def prepare(args, output):
    contract = {
        "checkpoint": CHECKPOINT,
        "seed": args.seed,
        "pilot": args.pilot,
        "tokens": sha(args.tokens / "index.json"),
        "natural": sha(args.natural / "profiles" / "COMPLETE"),
        "naturalTest": sha(args.natural_test / "index.json"),
        "naturalPilot": sha(args.natural_pilot),
        "circuits": {str(path): sha(path) for path in args.circuits},
        "evidence": sha(args.evidence),
        "source": all_code_hashes(),
    }
    backend_manifest = args.natural / "edges-shard-000-of-008" / "manifest.json"
    contract["sourceModels"] = load(backend_manifest)["backend"]["models"]
    contract["naturalTimingManifestSha256"] = sha(backend_manifest)
    contract["versions"] = {"numpy": np.__version__, "networkx": nx.__version__}
    index_root = output / "index"
    if index_root.exists():
        previous = load(index_root / "manifest.json")
        if previous["contract"] != contract:
            raise ValueError("Index cache belongs to different inputs")
        for name, digest in previous["files"].items():
            if sha(index_root / name) != digest:
                raise ValueError(f"Existing index changed: {name}")
    else:
        start = time.monotonic()
        with TemporaryDirectory(dir=output, prefix=".index-") as tmp:
            stage = Path(tmp) / "index"
            stage.mkdir()
            stats = write_index(
                stage,
                (
                    natural_scopes(
                        args.natural,
                        96 if args.pilot else None,
                        {
                            snap["anchor"]
                            for item in load(args.evidence).values()
                            for snap in item["snapshots"]
                        }
                        if args.pilot
                        else (),
                    ),
                    token_scopes(args.tokens, args.seed, 24 if args.pilot else None),
                    heldout_natural_scopes(args.natural_test, args.natural_pilot),
                ),
            )
            previous = {
                "contract": contract,
                "stats": stats,
                "seconds": time.monotonic() - start,
                "peakKiB": cpu_peak(),
                "files": {path.name: sha(path) for path in stage.iterdir()},
            }
            atomic_json(stage / "manifest.json", previous)
            stage.rename(index_root)
    return Index(index_root), previous


def setup_worker(root, templates, sinks, stage):
    global INDEX, TEMPLATES, BY_EDGE, SINKS, STARTS, STAGE
    INDEX = Index(root)
    TEMPLATES = templates
    SINKS = sinks
    STAGE = stage
    BY_EDGE = defaultdict(list)
    STARTS = defaultdict(set)
    for i, (_, edges) in enumerate(templates):
        for edge in edges:
            BY_EDGE[edge].append(i)
            STARTS[model_of(edge[0])].add(edge[0])
    STARTS = {key: sorted(nodes) for key, nodes in STARTS.items()}


def run_task(task):
    shard, indices = task
    path = STAGE / f"shard-{shard:05}.json"
    if path.exists():
        return load(path)["timing"]
    started = time.monotonic()
    cpu = time.process_time()
    rows = []
    counts, expansions, elapsed = Counter(), Counter(), Counter()
    for i in indices:
        edge, _ = INDEX.graph(i)
        if len(edge) < 4:
            continue
        meta = INDEX.scopes[i]
        graph = Graph.from_rows(edge)
        operations = (
            (bottom_up, (graph, STARTS.get(MODELS.index(meta["model"]), ()))),
            (top_down, (graph, meta, SINKS)),
            (composition, (graph, TEMPLATES, BY_EDGE)),
        )
        for method, (probe, args) in zip(METHODS, operations, strict=True):
            start = time.process_time()
            proposals, considered = probe(*args)
            elapsed[method] += time.process_time() - start
            counts[method] += len(proposals)
            expansions[method] += considered
            rows.extend((method, i, edges, desc) for edges, desc in sorted(proposals.items()))
    stats = {
        "scopes": len(indices),
        "wallSeconds": time.monotonic() - started,
        "cpuSeconds": time.process_time() - cpu,
        "peakKiB": cpu_peak(),
        "methods": {
            m: {"generated": counts[m], "expanded": expansions[m], "cpuSeconds": elapsed[m]}
            for m in METHODS
        },
    }
    atomic_json(path, {"proposals": rows, "timing": stats})
    return stats


def run_search(index, templates, root, workers, selected=None):
    scope_ids = [
        i for i, meta in enumerate(index.scopes) if meta["split"] in {"discovery", "discovery-only"}
    ]
    indices = scope_ids if selected is None else selected
    sinks, contrasts = estimate_sinks(index, scope_ids)
    declared = {
        "index": sha(root.parent / "index" / "manifest.json"),
        "scopeIds": indices,
        "batchSize": 48,
    }
    manifest = root / "plan.json"
    root.mkdir(exist_ok=True)
    if manifest.exists() and load(manifest) != declared:
        raise ValueError("Shard cache differs from deterministic plan")
    if not manifest.exists():
        atomic_json(manifest, declared)
    tasks = list(enumerate(task_batches(indices, 48)))
    start = time.monotonic()
    if workers == 1:
        setup_worker(root.parent / "index", templates, sinks, root)
        results = [run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=setup_worker,
            initargs=(root.parent / "index", templates, sinks, root),
        ) as pool:
            results = list(pool.map(run_task, tasks))
    duration = time.monotonic() - start
    return {
        "workers": workers,
        "scopes": len(indices),
        "matchedSinkContrasts": contrasts,
        "seconds": duration,
        "cpuSeconds": sum(r["cpuSeconds"] for r in results),
        "scopesPerSecond": len(indices) / max(duration, 0.001),
        "workerPeakKiB": max((r["peakKiB"] for r in results), default=0),
        "methods": {
            method: {
                key: sum(r["methods"][method][key] for r in results)
                for key in ("generated", "expanded", "cpuSeconds")
            }
            for method in METHODS
        },
    }


def collect(index, shard_dir, cap):
    ledger = {}
    by_bucket = defaultdict(set)
    for path in sorted(shard_dir.glob("shard-*.json")):
        payload = load(path)
        for method, i, edges, explanation in payload["proposals"]:
            edges = tuple(map(tuple, edges))
            key = signature(edges)
            if key not in ledger:
                ledger[key] = {
                    "id": key,
                    "edges": edges,
                    "occurrences": set(),
                    "directions": set(),
                    "firstScope": i,
                    "metadata": {},
                    "structure": structure(edges),
                }
            row = ledger[key]
            if row["edges"] != edges:
                raise ValueError("Candidate hash collision")
            row["occurrences"].add(i)
            row["directions"].add(method)
            row["metadata"].setdefault(method, explanation)
            meta = index.scopes[i]
            by_bucket[method, meta["dataset"], meta["model"]].add(key)
    winners = set()
    for keys in by_bucket.values():
        ranked = sorted(
            keys,
            key=lambda key: (
                -len(ledger[key]["occurrences"]),
                -ledger[key]["structure"]["depth"],
                -ledger[key]["structure"]["convergence"],
                key,
            ),
        )
        winners.update(ranked[:cap])
    return [ledger[key] for key in sorted(winners)], len(ledger)


def motif_set(edges):
    return {(gate_of(source), gate_of(target)) for source, target in edges}


def summarize_split(index, ids):
    partitions = defaultdict(list)
    for i in ids:
        row = index.scopes[i]
        partitions[row["dataset"], row["split"]].append(row)
    return [
        {
            "dataset": data,
            "split": split,
            "anchors": len({row["anchor"] for row in rows}),
            "groups": len({row["group"] for row in rows if row.get("group") is not None}),
            "conditions": dict(Counter(row.get("condition", "unavailable") for row in rows)),
        }
        for (data, split), rows in sorted(partitions.items())
    ]


def witness(index, i, edges):
    row = index.scopes[i]
    array, nodes = index.graph(i)
    selected = set(edges)
    vertices = {vertex for edge in edges for vertex in edge}
    graph = [
        {"source": decode(source), "target": decode(target), "E": signed, "w": weight}
        for source, target, signed, weight in array.tolist()
        if (source, target) in selected
    ]
    if len(graph) != len(edges):
        raise ValueError("Witness is missing a required edge")
    scalars = {
        decode(node): {
            "coefficient": value if np.isfinite(value) else None,
            "normalized": normalized if np.isfinite(normalized) else None,
        }
        for node, value, normalized in nodes.tolist()
        if node in vertices
    }
    involved = {gate_of(node) for node in vertices}
    active = {gate: value for gate, value in row.get("active", {}).items() if gate in involved}
    context = {
        key: row.get(key)
        for key in (
            "dataset",
            "split",
            "anchor",
            "model",
            "pool",
            "example",
            "group",
            "family",
            "condition",
            "text",
            "boundary",
            "positions",
            "artifact",
            "role",
            "complete",
        )
    }
    return {**context, "edges": graph, "components": scalars, "activations": active}


def opportunity_rates(index, row, occurrences, cache):
    """Use only fully measured control anchors of the same annotated role and dataset."""
    meta = index.scopes[row["firstScope"]]
    if not meta.get("role"):
        return None
    key = (meta["dataset"], meta["model"], meta.get("family"), meta["role"])
    data = cache[key]
    full = {gate_of(node) for edge in row["edges"] for node in edge}
    counts = Counter()
    for i in data:
        candidate = index.scopes[i]
        split = candidate["split"]
        condition = candidate["condition"]
        counts[split, condition, "anchor_count"] += 1
        if full <= candidate["active"].keys():
            counts[split, condition, "all_gates_active"] += 1
            if i in occurrences:
                counts[split, condition, "circuit_present"] += 1
    return [
        {
            "split": split,
            "condition": condition,
            "measuredAnchors": counts[split, condition, "anchor_count"],
            "activeOpportunities": counts[split, condition, "all_gates_active"],
            "matches": counts[split, condition, "circuit_present"],
        }
        for split, condition in sorted({(s, c) for s, c, _ in counts})
    ]


def counterexamples(index, row, occurrences, opportunities):
    source = index.scopes[row["firstScope"]]
    if not source.get("role"):
        return []
    nodes = {node for pair in row["edges"] for node in pair}
    gates = {gate_of(node) for node in nodes}
    key = source["dataset"], source["model"], source.get("family"), source["role"]
    controls = []
    for i in opportunities[key]:
        scope = index.scopes[i]
        if i not in occurrences and gates <= scope.get("active", {}).keys() and scope.get("text"):
            controls.append(
                {
                    field: scope.get(field)
                    for field in (
                        "anchor",
                        "pool",
                        "model",
                        "group",
                        "condition",
                        "split",
                        "text",
                        "boundary",
                    )
                }
            )
    return sorted(controls, key=lambda item: (item["split"] != "heldout", item["anchor"]))[:3]


def component_roles(row, witnesses):
    before = Counter(a for a, _ in row["edges"])
    after = Counter(b for _, b in row["edges"])
    values = witnesses[0]["components"]
    return [
        {
            "componentId": decode(node),
            "structuralRole": (
                "relay"
                if before[node] and after[node]
                else "input arm"
                if before[node]
                else "downstream readout"
            ),
            "incoming": after[node],
            "outgoing": before[node],
            **values.get(decode(node), {}),
        }
        for node in sorted(before.keys() | after.keys())
    ]


def describe_candidate(row, witness_rows, signs, context_rates, examples):
    evidence = next((item for item in witness_rows if item.get("text")), None)
    sink = row["metadata"].get("top-down", {})
    links = row["metadata"].get("composition", {})
    role = (evidence or {}).get("role")
    family = (evidence or {}).get("family")
    description = {}
    if sink and role:
        description["hypothesis"] = (
            f"Upstream branches may combine context information at the {role} position before writing to {decode(sink['sink'])}."
        )
    elif links:
        description["hypothesis"] = (
            f"The {links['relationship']} may coordinate the computations attributed to the two smaller saved circuits."
        )
    elif evidence and family:
        description["hypothesis"] = (
            f"The convergent or serial dependencies may coordinate processing in {family.replace('_', ' ')} contexts."
        )
    else:
        description["hypothesis"] = (
            "The recurring dependencies may combine multiple local computations. A specific semantic role remains open."
        )
    description["alternative"] = (
        "Shared lexical cues, common firing, or a highly connected component can produce recurrence. "
        "Signed Taylor effects are local dependency estimates."
    )
    description["counterevidence"] = (
        f"At least one matched edge changes sign across witnesses ({signs} stable)."
        if signs is not None and signs < 1
        else "Held-out support is absent or limited for this exact component path."
        if not context_rates
        or not any(r["matches"] for r in context_rates if r["split"] == "heldout")
        else "Co-firing and alternative pathways remain compatible with these observations."
    )
    description["proposedTest"] = (
        "Freeze an annotated internal readout and vary the hypothesized cues independently, with matched "
        "non-firings and held-out groups; use targeted ablation and restoration only in a separate approved study."
    )
    description["independentActivationExamples"] = examples[:3]
    return description


def evaluate(index, rows, templates, independent, seed):
    import itertools

    postings = measured_paths(index, (row["edges"] for row in rows))
    known = [motif_set(edges) for _, edges in templates]
    cofire = Counter(
        gate
        for row in index.scopes
        if row["split"] == "discovery" and row["model"] == "gemma"
        for gate in row.get("active", ())
    )
    prevalence = {key: int(np.log2(count)) + 1 for key, count in cofire.items()}
    control_scopes = defaultdict(list)
    for i, meta in enumerate(index.scopes):
        if meta["complete"] and meta.get("role") and meta["split"] in {"discovery", "heldout"}:
            control_scopes[meta["dataset"], meta["model"], meta.get("family"), meta["role"]].append(
                i
            )
    catalogs, groups = [], defaultdict(list)
    shape_groups = defaultdict(list)
    for row in rows:
        edges = row["edges"]
        ids = recurring(postings, edges)
        source = index.scopes[row["firstScope"]]
        if row["firstScope"] not in ids:
            raise ValueError("A generated candidate has no exact recurrence witness")
        witnesses = sorted(
            ids,
            key=lambda i: (
                index.scopes[i]["split"] != "heldout",
                index.scopes[i].get("text") is None,
                i,
            ),
        )[:4]
        checked = [witness(index, i, edges) for i in witnesses]
        match_weights = [edge["w"] for anchor in checked for edge in anchor["edges"]]
        edge_signs = list(
            zip(
                *(
                    tuple(
                        np.sign(edge["E"])
                        for edge in sorted(item["edges"], key=lambda e: (e["source"], e["target"]))
                    )
                    for item in checked
                ),
                strict=True,
            )
        )
        stable = (
            sum(len(set(value)) == 1 for value in edge_signs) / len(edge_signs)
            if len(checked) > 1
            else None
        )
        rate = opportunity_rates(index, row, ids, control_scopes)
        source_graph = Graph.from_rows(index.graph(row["firstScope"])[0])
        null = matched_null(
            source_graph, edges, prevalence, random.Random(seed + int(row["id"], 16))
        )
        nodes = {node for edge in edges for node in edge}
        motif = motif_set(edges)
        nearest = max((len(motif & other) / len(motif | other) for other in known), default=0)
        examples = [
            {**text, "gate": gate, "scope": "independent held-out firing example"}
            for gate in sorted({gate_of(node) for node in nodes})
            for text in independent.get(gate, {}).values()
        ]
        context_semantics = describe_candidate(row, checked, stable, rate, examples)
        context_semantics["componentRoles"] = component_roles(row, checked)
        context_semantics["counterexampleContexts"] = counterexamples(
            index, row, ids, control_scopes
        )
        family_key, graph = directed_gate_signature(edges)
        result = {
            "id": f"CXD-{row['id']}",
            "directions": sorted(row["directions"]),
            "model": source["model"],
            "dataset": source["dataset"],
            "structure": row["structure"],
            "anchorCount": len(ids),
            "generationAnchors": len(row["occurrences"]),
            "recurrence": summarize_split(index, ids),
            "medianWitnessWeight": median(match_weights),
            "witnessSignStability": stable,
            "novelty": 1 - nearest,
            "knownGateMotifJaccard": nearest,
            "matchedNull": null,
            "contextSelectivity": rate,
            "proposal": row["metadata"],
            "interpretation": context_semantics,
            "status": "Unvalidated hypothesis",
            "componentIds": [decode(node) for node in sorted(nodes)],
            "gateIds": sorted({gate_of(node) for node in nodes}),
            "witnesses": checked,
            "crossModel": {"models": [source["model"]], "sharedExactAnchors": 0, "peers": []},
        }
        groups[family_key].append((result, graph, ids))
        shape = nx.weisfeiler_lehman_graph_hash(graph, iterations=4)
        shape_groups[(source["dataset"], shape)].append((result, graph))
        catalogs.append(result)
    for variants in groups.values():
        for (left, a, li), (right, b, ri) in itertools.combinations(variants, 2):
            if left["model"] == right["model"] or not nx.is_isomorphic(
                a, b, node_match=lambda x, y: x["gate"] == y["gate"]
            ):
                continue

            def scopes(values):
                return {
                    (index.scopes[i]["dataset"], index.scopes[i]["anchor"], index.scopes[i]["pool"])
                    for i in values
                }

            joint = len(scopes(li) & scopes(ri))
            for x, y in ((left, right), (right, left)):
                evidence = x["crossModel"]
                evidence["models"] = sorted(set(evidence["models"]) | {y["model"]})
                evidence["sharedExactAnchors"] = max(evidence["sharedExactAnchors"], joint)
                evidence["peers"].append(
                    {"id": y["id"], "model": y["model"], "sharedExactAnchors": joint}
                )

    for variants in shape_groups.values():
        for result, graph in variants:
            peers = {
                other["model"]
                for other, shape in variants
                if other["model"] != result["model"] and nx.is_isomorphic(graph, shape)
            }
            result["crossModel"]["topologyOnlyModels"] = sorted(peers)

    def goals(row):
        null = row["matchedNull"]
        heldout = sum(item["anchors"] for item in row["recurrence"] if item["split"] == "heldout")
        return (
            heldout,
            row["anchorCount"],
            row["crossModel"]["sharedExactAnchors"],
            row["structure"]["depth"],
            row["structure"]["convergence"],
            row["novelty"],
            -null["hits"] / null["trials"] if null["trials"] else -1,
        )

    return pareto(catalogs, goals)


def save_catalog(output, rows, run, distinct, index_manifest):
    provenance = {
        "checkpoint": CHECKPOINT,
        "indexSha256": sha(output / "index" / "manifest.json"),
        "search": run,
        "generatedDistinct": distinct,
        "shortlisted": len(rows),
    }
    keep = [row for row in rows if row["knownGateMotifJaccard"] < 1]
    atomic_json(
        output / "catalog.json", {"schemaVersion": 1, "provenance": provenance, "candidates": keep}
    )
    atomic_json(
        output / "evaluation.json",
        {"provenance": provenance, "candidates": rows, "index": index_manifest},
    )
    for method in METHODS:
        atomic_json(output / f"{method}.json", [row for row in keep if method in row["directions"]])
    with (output / "catalog.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "id",
                "directions",
                "model",
                "dataset",
                "nodes",
                "depth",
                "anchor_count",
                "pareto_front",
                "hypothesis",
            )
        )
        for row in keep:
            writer.writerow(
                (
                    row["id"],
                    ",".join(row["directions"]),
                    row["model"],
                    row["dataset"],
                    row["structure"]["nodes"],
                    row["structure"]["depth"],
                    row["anchorCount"],
                    row["paretoFront"],
                    row["interpretation"]["hypothesis"],
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tokens", "natural", "natural-test", "natural-pilot", "evidence", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--circuits", type=Path, action="append", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shortlist-per-source-model-direction", type=int, default=10)
    parser.add_argument("--seed", type=int, default=61036)
    args = parser.parse_args()
    limit = min(24, len(os.sched_getaffinity(0)), int(os.getenv("SLURM_CPUS_PER_TASK", "1")))
    if not 1 <= args.workers <= limit:
        parser.error(f"workers must be within the active aggregate allocation of {limit}")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    index, index_manifest = prepare(args, output)
    templates, independent = load_seeds(args.circuits, args.evidence)
    print(
        json.dumps(
            {
                "index": index_manifest["stats"],
                "seconds": index_manifest["seconds"],
                "seedTemplates": len(templates),
            }
        ),
        flush=True,
    )
    if args.pilot:
        ids = [
            i
            for i, meta in enumerate(index.scopes)
            if meta["split"] in {"discovery", "discovery-only"}
        ]
        results = [
            run_search(index, templates, output / f"benchmark-{workers}", workers, ids)
            for workers in (1, 2, 4)
            if workers <= args.workers
        ]
        atomic_json(output / "benchmark.json", {"index": index_manifest, "scaling": results})
        run, stage = results[-1], output / f"benchmark-{results[-1]['workers']}"
    else:
        stage = output / "search"
        run = run_search(index, templates, stage, args.workers)
    candidates, distinct = collect(index, stage, args.shortlist_per_source_model_direction)
    rows = evaluate(index, candidates, templates, independent, args.seed)
    save_catalog(output, rows, run, distinct, index_manifest)
    print(
        json.dumps({"generatedDistinct": distinct, "screened": len(rows), "search": run}),
        flush=True,
    )


if __name__ == "__main__":
    main()
