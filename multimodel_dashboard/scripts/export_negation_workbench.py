"""Hash-check Negation sources and assemble a lazy, same-example workbench bundle."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from aspd.multimodel.cli.evaluate_aspd_v3_induction import _load_references
from aspd.multimodel.cli.evaluate_aspd_v3_negation import jsonl, verify_evaluation
from aspd.multimodel.cli.measure_induction_query import _gate_label
from aspd.multimodel.cli.prepare_aspd_v3_negation import read_contract
from aspd.multimodel.posthoc.artifacts import sha256_file
from aspd.multimodel.posthoc.ids import ComponentId
from aspd.multimodel.v3.negation import CONDITIONS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact_write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def load_graphs(root, measured, discovery):
    path = root / "query_graphs.json"
    complete = read(root / "COMPLETE")
    if complete["bundle_sha256"] != sha256_file(path) or complete[
        "selection_sha256"
    ] != sha256_file(root / "selection.json"):
        raise ValueError("graph artifact hash mismatch")
    payload = read(path)
    provenance = payload["provenance"]
    if (
        payload["mechanism"] != "negation"
        or payload["split"] != "discovery"
        or provenance["scope"] != "final_query_same_anchor_frozen_discovery_support"
        or provenance["discoveryManifestSha256"] != sha256_file(discovery / "manifest.json")
        or provenance["checkpointSha256"] != measured["checkpoint_sha256"]
        or (provenance["kPar"], provenance["tau"]) != (10, 0)
        or payload["selection"] != read(root / "selection.json")["gates"]
    ):
        raise ValueError("graph, discovery and frozen parent selection disagree")
    return payload


def validate_target(target, final):
    destination = ComponentId.from_key(target["component"]["id"])
    scale = target["outputScale"]
    norm = target["writeNorm"]
    if (
        not math.isfinite(target["coefficient"])
        or not math.isfinite(scale)
        or scale <= 0
        or not math.isfinite(norm)
        or norm < 0
        or not math.isclose(target["contributionNorm"], abs(target["coefficient"]) * norm)
        or destination.feature not in final[destination.gate.site]["active_gate_ids"]
    ):
        raise ValueError("invalid measured component coefficient, scale or support")
    edges = target["parents"]
    if len(edges) > 10 or list(range(1, len(edges) + 1)) != [edge["rank"] for edge in edges]:
        raise ValueError("invalid parent count/rank")
    if [abs(edge["E"]) for edge in edges] != sorted(
        (abs(edge["E"]) for edge in edges), reverse=True
    ):
        raise ValueError("parent records are out of rank order")
    for edge in edges:
        parent = ComponentId.from_key(edge["component"]["id"])
        if (
            not parent.matrix.is_upstream_of(destination.matrix)
            or not math.isfinite(edge["E"])
            or abs(edge["E"]) <= 0
            or not math.isfinite(edge["w"])
            or edge["w"] < 0
            or not math.isclose(edge["w"], abs(edge["E"]) * norm / scale)
            or parent.feature not in final[parent.gate.site]["active_gate_ids"]
        ):
            raise ValueError("invalid measured parent edge")
        yield parent, edge


def validate_graph(ex, measured, reps, *, comprehensive=False):
    (final,) = (r for r in measured["positions"] if r["final_query"])
    for key, field in (
        ("exampleId", "example_id"),
        ("groupId", "group_id"),
        ("family", "family"),
        ("condition", "condition"),
        ("poolId", "pool_id"),
        ("modelPositions", "model_positions"),
    ):
        if ex[key] != final[field]:
            raise ValueError(f"graph context identity changed: {key}")
    for name, graph in ex["graphs"].items():
        site, feature = name.lower().split(":")
        values = final[site]
        active_values = dict(zip(values["active_gate_ids"], values["values"], strict=True))
        if graph["anchorId"] != final["anchor_id"] or graph["activation"] != active_values.get(
            int(feature), 0.0
        ):
            raise ValueError("graph anchor/support does not match measured context")
        if graph["activation"] == 0:
            if graph["models"] or graph["comparisons"]:
                raise ValueError("inactive final query cannot have measured parents")
            continue
        if set(graph["models"]) != set(final["model_positions"]):
            raise ValueError("missing model graph")
        for model, row in graph["models"].items():
            parents = Counter()
            targets = row["targets"]
            expected = {
                item.key for item in reps if item.model == model and _gate_label(item.gate) == name
            }
            if expected != {target["component"]["id"] for target in targets}:
                raise ValueError("measured representatives differ from frozen lookup")
            if not expected:
                if (
                    not comprehensive
                    or row["status"] != "no_reference_representatives"
                    or any(row[key] is not None for key in ("strength", "scores", "distribution"))
                ):
                    raise ValueError("missing representatives must remain unavailable")
                continue
            for target in targets:
                for parent, edge in validate_target(target, final):
                    parents[_gate_label(parent.gate)] += edge["w"] / len(targets)
            if not math.isclose(sum(parents.values()), row["strength"]) or set(parents) != set(
                row["scores"]
            ):
                raise ValueError("parent strength or score support disagrees with edges")
            for gate_id, score in parents.items():
                if not math.isclose(row["scores"][gate_id], score):
                    raise ValueError("parent score differs from normalized weights")
            if row["strength"] == 0:
                if row["distribution"] is not None:
                    raise ValueError("zero-strength parent distribution must stay undefined")
            elif set(row["distribution"]) != set(parents) or any(
                not math.isclose(row["distribution"][gate], score / row["strength"])
                for gate, score in parents.items()
            ):
                raise ValueError("parent distribution disagrees with scores or strength")


def to_view_gate(row, totals):
    conditions = {}
    for label, denominator in totals.items():
        if label not in CONDITIONS:
            continue
        value = row["conditions"].get(label, {})
        fires = value.get("fires", 0)
        activation_sum = value.get("sum", 0)
        mean = activation_sum / denominator["anchors"]
        conditions[label] = {
            "sequenceFires": value.get("sequence_fires", 0),
            "queryFires": value.get("query_fires", 0),
            "queryMean": value.get("query_sum", 0) / denominator["queries"],
            "mean": mean,
            "meanWhenFiring": activation_sum / fires if fires else None,
            "sd": math.sqrt(max(0, value.get("sq", 0) / denominator["anchors"] - mean**2)),
            "roles": {
                role: value.get(f"role/{role}/sum", 0) / denominator[f"role/{role}"]
                for role in ("negation_operator", "excluded_concept", "alternative_concept")
                if denominator.get(f"role/{role}", 0)
            },
        }
    positive = sum(
        conditions[c]["queryMean"] for c in ("affirmative", "counterfactual_affirmative")
    )
    negative = sum(conditions[c]["queryMean"] for c in ("negative", "counterfactual_negative"))
    return {
        "id": row["gate"].upper().replace("/", ":"),
        "contrast": (negative - positive) / 2,
        "referenceBeta": row["reference_max_beta"],
        "conditions": conditions,
        "availableComponentStats": row["available_component_stats"],
    }


def build_split(root, outfile, graphs, reps):
    summary = read(root / "gate_summary.json")
    examples = list(jsonl(root / "activation_examples.jsonl"))
    by_example = {row["example_id"]: row for row in examples}
    graph_rows = {row["exampleId"]: row for row in graphs.get("examples", [])}
    if (
        len(graph_rows) != len(graphs.get("examples", []))
        or not graph_rows.keys() <= by_example.keys()
    ):
        raise ValueError("graph examples must uniquely belong to this exact split")
    targets = defaultdict(set)
    for gate, details in graphs.get("selection", {}).items():
        for group_id in details["groups"]:
            targets[group_id].add(gate)
    grouped = defaultdict(list)
    for example in examples:
        example_id = example["example_id"]
        if set(graph_rows.get(example_id, {}).get("graphs", {})) != targets.get(
            example["group_id"], set()
        ):
            raise ValueError(
                "measured graph coverage differs from the frozen matched-group selection"
            )
        if example_id in graph_rows:
            validate_graph(graph_rows[example_id], example, reps)
            example["graphs"] = graph_rows[example_id]["graphs"]
        grouped[example["group_id"]].append(example)
    displayed = {}
    gate_groups = defaultdict(dict)
    for index, (group_id, group) in enumerate(sorted(grouped.items())):
        path = f"group-{index:03d}.json"
        displayed[group_id] = {
            "path": path,
            "family": group[0]["family"],
            "conditions": [row["condition"] for row in group],
            "outcomes": {
                row["condition"]: {k: v["negation_outcome"] for k, v in row["predictions"].items()}
                for row in group
                if all("negation_outcome" in p for p in row["predictions"].values())
            },
        }
        peaks = defaultdict(float)
        graph_gates = set()
        for row in group:
            for position in row["positions"]:
                for site in ("pre", "mid"):
                    for feature, strength in zip(
                        position[site]["active_gate_ids"], position[site]["values"], strict=True
                    ):
                        gate_id = f"{site.upper()}:{feature}"
                        peaks[gate_id] = max(peaks[gate_id], strength)
            for gate, graph in row.get("graphs", {}).items():
                graph_gates.add(gate)
                for model in graph["models"].values():
                    for target in model["targets"]:
                        graph_gates.update(
                            edge["component"]["gateId"] for edge in target["parents"]
                        )
        for gate_id, peak in peaks.items():
            # Include all parent-graph contexts. Keep up to four remaining groups per gate.
            gate_groups[gate_id][group_id] = (gate_id in graph_gates, peak)
        compact_write(
            outfile / path, {"mechanism": "negation", "groupId": group_id, "examples": group}
        )
    groups = {}
    for gate, items in gate_groups.items():
        parent_examples = sorted(group for group, (has_graph, _) in items.items() if has_graph)
        strongest_remaining = sorted(
            (group for group in items if group not in parent_examples),
            key=lambda group: (-items[group][1], group),
        )
        groups[gate] = parent_examples + strongest_remaining[:4]
    return {
        "groups": displayed,
        "gateGroups": groups,
        "selection": graphs.get("selection", {}),
        "graphProvenance": graphs.get("provenance"),
        "denominators": {k: v for k, v in summary["denominators"].items() if k in CONDITIONS},
        "gates": [to_view_gate(row, summary["denominators"]) for row in summary["gates"]],
        "poolCount": len(read(root / "pools.json")),
        "manifestSha256": sha256_file(root / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("config", "smoke", "discovery", "graphs", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    _spec, reference = read_contract(args.config)
    smoke = verify_evaluation(args.smoke, "smoke")
    discovery = verify_evaluation(args.discovery, "discovery")
    if any(
        smoke[key] != discovery[key]
        for key in (
            "checkpoint_sha256",
            "config_sha256",
            "preparation_sha256",
            "reference_importance_sha256",
            "reference_representatives_sha256",
        )
    ):
        raise ValueError("smoke and discovery provenance differs")
    graphs = load_graphs(args.graphs, discovery, args.discovery)
    root = Path(reference["components"]["reference_posthoc_root"])
    if (
        sha256_file(root / "importance/COMPLETE") != discovery["reference_importance_sha256"]
        or sha256_file(root / "representatives/COMPLETE")
        != discovery["reference_representatives_sha256"]
    ):
        raise ValueError("component locator reference changed")
    importance, reps = _load_references(root)
    all_reps = {component for values in reps.values() for component in values}
    reps_by_gate = defaultdict(set)
    for component in all_reps:
        reps_by_gate[_gate_label(component.gate)].add(component)
    expected_reps = {rep for gate in graphs["selection"] for rep in reps_by_gate[gate]}
    args.output.mkdir(parents=True, exist_ok=False)
    locator = []
    rep_features = defaultdict(list)
    for component in all_reps:
        rep_features[component.matrix].append(component.feature)
    for matrix in importance.matrices():
        beta = [
            value if math.isfinite(value) else None for value in importance[matrix].beta.tolist()
        ]
        locator.append(
            {
                "model": matrix.model,
                "layer": matrix.layer,
                "site": matrix.site.upper(),
                "matrix": matrix.matrix.removeprefix("mlp_")
                if matrix.matrix.startswith("mlp_")
                else matrix.matrix.upper(),
                "beta": beta,
                "representatives": sorted(rep_features[matrix]),
            }
        )
    compact_write(args.output / "locator.json", {"mechanism": "negation", "matrices": locator})
    splits = {}
    for name, source, graph in (("smoke", args.smoke, {}), ("discovery", args.discovery, graphs)):
        directory = args.output / name
        directory.mkdir()
        splits[name] = build_split(source, directory, graph, expected_reps)
    index = {
        "schemaVersion": 1,
        "mechanism": "negation",
        "checkpointStep": 61036,
        "checkpointSha256": discovery["checkpoint_sha256"],
        "referenceImportanceSha256": discovery["reference_importance_sha256"],
        "splits": splits,
        "assetHashes": {
            str(p.relative_to(args.output)): sha256_file(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    }
    compact_write(args.output / "index.json", index)
    print(f"Negation workbench index: {sha256_file(args.output / 'index.json')}")


if __name__ == "__main__":
    main()
