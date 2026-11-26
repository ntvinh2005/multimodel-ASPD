"""Export verified all-token activations and any completed same-anchor graph coverage."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from export_negation_workbench import compact_write

from aspd.multimodel.cli.benchmark_token_dependencies import complete_token_graph
from aspd.multimodel.cli.evaluate_aspd_v3_induction import _load_references
from aspd.multimodel.cli.evaluate_aspd_v3_negation import jsonl
from aspd.multimodel.cli.measure_induction_query import _gate_label
from aspd.multimodel.cli.measure_negation_comprehensive import active_gates
from aspd.multimodel.cli.prepare_aspd_v3_negation import read_contract
from aspd.multimodel.posthoc.artifacts import sha256_file
from aspd.multimodel.posthoc.ids import ComponentId
from aspd.multimodel.v3.induction import MODEL_ORDER


def read_asset(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_bytes())


def archive_examples(root, checkpoint):
    manifest = root / "manifest.json"
    complete = json.loads((root / "COMPLETE").read_bytes())
    if complete["manifest_sha256"] != sha256_file(manifest):
        raise ValueError(f"invalid archived run {root}")
    path = root / "activation_examples.jsonl"
    provenance = json.loads(manifest.read_bytes())
    if (provenance["heldout_opened"]
            or sha256_file(Path(provenance["backend"]["checkpoint"]) / "manifest.json") != checkpoint):
        raise ValueError(f"checkpoint or split differs: {root}")
    if sha256_file(path) != provenance["artifacts"][path.name]:
        raise ValueError(f"invalid archived examples {root}")
    return list(jsonl(path))


def group_examples(rows):
    groups = defaultdict(list)
    anchors = {}
    for row in rows:
        group, example, text = row["group_id"], row["example_id"], row["prompt"]
        positions = []
        for position in row["positions"]:
            pool_id = position.get("pool_id", row.get("pool_id"))
            if not pool_id or ("pool_id" in row and row["pool_id"] != pool_id):
                raise ValueError(f"unresolved activation pool for {example}")
            positions.append({**position, "pool_id": pool_id})
        boundaries = [position["character_boundary"] for position in positions]
        if not boundaries or boundaries[-1] != len(text) or sorted(set(boundaries)) != boundaries:
            raise ValueError(f"invalid canonical positions for {example}")
        for position in positions:
            key = position["anchor_id"]
            if key in anchors or position["example_id"] != example:
                raise ValueError(f"invalid or duplicate canonical anchor for {example}")
            anchors[key] = position
        groups[group].append({
            **{key: row[key] for key in ("example_id", "group_id", "prompt", "condition")},
            "positions": positions,
        })
    return groups, anchors


def legacy_graphs(workbench, anchors, checkpoint, output):
    negation = workbench / "negation"
    index = json.loads((negation / "index.json").read_bytes())
    expanded = json.loads((negation / "expanded/index.json").read_bytes())
    split = index["splits"]["discovery"]
    if (index["checkpointSha256"] != checkpoint or expanded["checkpointSha256"] != checkpoint
            or expanded["discoveryManifestSha256"] != split["manifestSha256"]):
        raise ValueError("pre-existing Negation graphs differ from token checkpoint or discovery")
    by_example = {value["example_id"]: value for value in anchors.values() if value["final_query"]}
    assets = {}

    def install(example, graphs, supplemental=None):
        if not graphs:
            return
        row = by_example[example]
        anchor = row["anchor_id"]
        if any(graph["anchorId"] != anchor for graph in graphs.values()):
            raise ValueError(f"cannot reuse a graph from a different anchor: {example}")
        path = output / f"graph-{anchor}.json"
        compact_write(path, {
            "anchorId": anchor, "exampleId": example, "poolId": row["pool_id"],
            "checkpointSha256": checkpoint, "official": graphs, "completeComponents": False,
            "componentTargets": {}, "outputScale": {}, "supplementalTargets": supplemental or {},
        })
        assets[anchor] = {"path": path.name, "sha256": sha256_file(path)}

    for group, item in expanded["groups"].items():
        path = negation / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Negation expanded group changed: {group}")
        for row in json.loads(path.read_bytes())["examples"]:
            install(row["exampleId"], row["graphs"], row["traceTargets"])
    for item in split["groups"].values():
        name = f"discovery/{item['path']}"
        path = negation / name
        if sha256_file(path) != index["assetHashes"][name]:
            raise ValueError("Negation group file changed")
        for row in json.loads(path.read_bytes())["examples"]:
            if by_example[row["example_id"]]["anchor_id"] not in assets:
                install(row["example_id"], row.get("graphs"))
    induction = json.loads((workbench / "inductionQueryGraphs.json").read_bytes())
    if induction["provenance"]["checkpointManifestSha256"] != checkpoint:
        raise ValueError("Induction graph checkpoint differs")
    for row in induction["examples"]:
        install(row["exampleId"], row["graphs"])
    return assets


def measured_batches(root, shard_roots, checkpoint):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if (json.loads((root / "COMPLETE").read_bytes())["manifestSha256"] != sha256_file(manifest_path)
            or manifest["checkpointSha256"] != checkpoint):
        raise ValueError("benchmark provenance differs from Workbench")
    for batch in manifest["batches"]:
        source = root / batch["path"]
        if sha256_file(source) != batch["sha256"]:
            raise ValueError("benchmark batch changed")
        yield source, batch["anchors"]
    for shard in shard_roots:
        for marker in sorted(shard.glob("batch-*.complete.json")):
            saved = json.loads(marker.read_bytes())
            source = shard / marker.name.replace(".complete.json", ".json.gz")
            if (saved["contract"]["checkpointSha256"] != checkpoint
                    or sha256_file(source) != saved["sha256"]):
                raise ValueError("completed shard batch changed")
            yield source, saved["anchors"]


def complete_graphs(root, shard_roots, anchors, checkpoint, config, output):
    _, reference = read_contract(config)
    importance, reps = _load_references(Path(reference["components"]["reference_posthoc_root"]))
    assets = {}
    for source, batch_anchors in measured_batches(root, shard_roots, checkpoint):
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            record = json.load(handle)
        if [row["anchor_id"] for row in record["anchors"]] != batch_anchors:
            raise ValueError("completed batch anchor order changed")
        for row in record["anchors"]:
            anchor = row["anchor_id"]
            payload = row.pop("targets")
            if anchor not in anchors or row != anchors[anchor]:
                raise ValueError("benchmark graph does not match its original token")
            if anchor in assets:
                continue
            path = output / f"complete-graph-{anchor}.json.gz"
            payload = complete_token_graph(row, payload, record["outputScale"], checkpoint, importance, reps)
            with path.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=3) as zipped:
                zipped.write(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode())
            assets[anchor] = {"path": path.name, "sha256": sha256_file(path)}
    return assets


def coverage_report(dataset_anchors, graphs, output):
    counts = defaultdict(Counter)
    for dataset, anchors in dataset_anchors.items():
        for anchor, row in anchors.items():
            gates = active_gates(row)
            active = {_gate_label(gate) for gate in gates}
            expected_components = sum(24 if gate.site == "pre" else 32 for gate in gates)
            asset = read_asset(output / graphs[anchor]["path"]) if anchor in graphs else None
            stored = defaultdict(dict)
            if asset:
                for key, (_, _, parents) in asset["componentTargets"].items():
                    component = ComponentId.from_key(key)
                    if component.gate not in gates or component.model not in MODEL_ORDER:
                        raise ValueError(f"inactive target in token graph: {key}")
                    stored[component.model][key] = bool(parents)
                for graph in asset["official"].values():
                    if not graph["activation"]:
                        continue
                    for model, details in graph["models"].items():
                        for record in details["targets"]:
                            stored[model][record["component"]["id"]] = bool(record["parents"])
                for key, record in asset.get("supplementalTargets", {}).items():
                    stored[record["component"]["modelId"]][key] = bool(record["parents"])
            for model in MODEL_ORDER:
                targets = stored[model]
                covered_gates = {_gate_label(ComponentId.from_key(key).gate) for key in targets}
                complete = bool(asset and asset["completeComponents"])
                if complete and (len(targets) != expected_components or covered_gates != active):
                    raise ValueError(f"comprehensive dependency coverage is incomplete at {anchor}/{model}")
                counts[dataset, row["condition"], model].update(
                    expected_anchors=1,
                    activation_anchors=1,
                    anchors_with_dependencies=bool(targets),
                    complete_component_anchors=complete,
                    expected_active_gates=len(active),
                    active_gates_with_any_component=len(covered_gates),
                    active_gates_with_complete_components=len(active) if complete else 0,
                    expected_component_graphs=expected_components,
                    measured_component_graphs=len(targets),
                    measured_empty_parent_sets=sum(not parents for parents in targets.values()),
                    remaining_component_graphs=expected_components - len(targets),
                )
    return [dict(dataset=dataset, condition=condition, model=model, **values)
            for (dataset, condition, model), values in sorted(counts.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", default=[], help="DATASET=completed evaluation directory")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--existing-workbench", type=Path, required=True)
    parser.add_argument("--shard", action="append", default=[], type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.benchmark / "manifest.json").read_bytes())
    checkpoint = index["checkpointSha256"]
    study = args.benchmark / "selection_frame_activations.json"
    if sha256_file(study) != index["selectionActivationsSha256"]:
        raise ValueError("Selection Frame activation artifact changed")
    datasets = {}
    dataset_anchors = {}
    all_anchors = {}
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for mapping in args.archive:
        label, root = mapping.split("=", 1)
        sources[label] = archive_examples(Path(root), checkpoint)
    sources["selection-frame"] = json.loads(study.read_bytes())["examples"]
    for dataset, rows in sources.items():
        groups, anchors = group_examples(rows)
        if all_anchors.keys() & anchors.keys():
            raise ValueError("duplicate canonical anchors across studies")
        all_anchors.update(anchors)
        dataset_anchors[dataset] = anchors
        projected = {}
        for group, examples in groups.items():
            path = args.output / f"{dataset}-{group}.json"
            compact_write(path, {"checkpointSha256": checkpoint, "dataset": dataset, "groupId": group, "examples": examples})
            projected[group] = {"path": path.name, "sha256": sha256_file(path)}
        datasets[dataset] = projected
    graphs = legacy_graphs(args.existing_workbench, all_anchors, checkpoint, args.output)
    graphs.update(complete_graphs(args.benchmark, args.shard, all_anchors, checkpoint, args.config, args.output))
    compact_write(args.output / "coverage.json", {
        "checkpointSha256": checkpoint,
        "benchmarkFailedMeasurements": 0,
        "rows": coverage_report(dataset_anchors, graphs, args.output),
    })
    compact_write(args.output / "index.json", {"checkpointSha256": checkpoint, "datasets": datasets, "graphs": graphs})


if __name__ == "__main__":
    main()
