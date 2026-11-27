"""Export two component neighborhoods for each archived Semantic Discovery pair."""

import argparse
from collections import Counter
from pathlib import Path

from aspd.multimodel.cli.discover_aspd_circuits import atomic_json
from aspd.multimodel.discovery.index import CHECKPOINT, Index, decode, encode, load, sha

SPLITS = ("discovery", "heldout")


def comparison_examples(left, right, transport, witnesses):
    """Prefer a shared measured position, followed by the archived OT examples."""
    cases = []
    used = set()

    def add(a, b, kind):
        if (a, b) not in used and (witnesses[0][a] or witnesses[1][b]):
            cases.append({"left": a, "right": b, "kind": kind})
            used.add((a, b))

    shared = sorted(left.keys() & right.keys(), key=lambda a: (-min(left[a], right[a]), a))
    for anchor in shared:
        if witnesses[0][anchor] and witnesses[1][anchor]:
            add(anchor, anchor, "shared")

    for item in transport:
        a, b = item["left"], item["right"]
        if a not in left or b not in right:
            raise ValueError("Transport anchor is outside the component's recorded support")
        add(a, b, "contrast" if item["kind"] == "counterexample" else "transport")

    if left and right:
        # Some held-out pairs have graph records but too few contexts for an OT score.
        a = min(left, key=lambda key: (not witnesses[0][key], -left[key], key))
        b = min(right, key=lambda key: (not witnesses[1][key], -right[key], key))
        add(a, b, "separate")

    return {"sharedSupport": len(shared), "examples": cases}


def export(semantic: Path, graph_index: Path):
    seal = load(semantic / "COMPLETE.json")
    data = load(semantic / "catalog.json", seal["catalogSha256"])
    graph_sha = sha(graph_index / "manifest.json")
    manifest = load(graph_index / "manifest.json", graph_sha)
    if (
        data["dataset"] != "natural-validation"
        or data["provenance"]["checkpoint"] != CHECKPOINT
        or data["provenance"]["split"] != "validation"
        or data["provenance"]["graphIndexSha256"] != graph_sha
        or manifest["contract"]["checkpoint"] != CHECKPOINT
    ):
        raise ValueError("Semantic and graph archives do not share Natural Validation provenance")

    index = Index(graph_index)
    scopes = {
        (row["anchor"], row["model"]): i
        for i, row in enumerate(index.scopes)
        if row["dataset"] == "natural-validation"
    }
    components = data["components"]
    graph_cache = {}

    def graph_for(component, anchor):
        key = f"{component}|{anchor}"
        if key not in graph_cache:
            graph_cache[key] = None
            model = components[component]["model"]
            position = data["anchors"][anchor]
            if position["dataset"] != "natural-validation" or position["anchor"] != anchor:
                raise ValueError("Wrong context for a semantic graph witness")
            scope = scopes.get((anchor, model))
            if scope is not None:
                edges, _ = index.graph(scope)
                code = encode(component)
                incident = edges[(edges["source"] == code) | (edges["target"] == code)]
                if len(incident):
                    graph_cache[key] = {
                        "anchor": anchor,
                        "model": model,
                        "component": component,
                        "artifact": index.scopes[scope]["artifact"],
                        "pool": position["pool"],
                        "edges": [
                            {
                                "source": decode(parent),
                                "target": decode(target),
                                "E": effect,
                                "w": weight,
                            }
                            for parent, target, effect, weight in incident.tolist()
                        ],
                    }
        return graph_cache[key]

    pairs = {}
    counts = Counter()
    retained = set()
    for cluster in data["clusters"]:
        for pair in cluster["links"]:
            components_pair = pair["left"], pair["right"]
            left, right = components_pair
            same_model = components[left]["model"] == components[right]["model"]
            for split in SPLITS:
                if not same_model:
                    counts[split, "crossModelPairs"] += 1
                supports = [
                    {
                        value["anchor"]: value["normalizedMagnitude"]
                        for value in components[component]["examples"].get(split, [])
                    }
                    for component in components_pair
                ]
                graphs = [
                    {anchor: graph_for(component, anchor) for anchor in support}
                    for component, support in zip(components_pair, supports, strict=True)
                ]
                comparison = comparison_examples(*supports, pair[split]["transport"], graphs)
                pairs.setdefault(f"{left}|{right}", {})[split] = comparison
                examples = comparison["examples"]
                for case in examples:
                    retained.update((f"{left}|{case['left']}", f"{right}|{case['right']}"))
                if same_model:
                    continue
                if any(graphs[0][case["left"]] and graphs[1][case["right"]] for case in examples):
                    counts[split, "bothGraphs"] += 1
                if any(
                    case["left"] == case["right"]
                    and graphs[0][case["left"]]
                    and graphs[1][case["right"]]
                    for case in examples
                ):
                    counts[split, "sharedGraphs"] += 1
    return {
        "schemaVersion": 1,
        "dataset": "natural-validation",
        "checkpoint": CHECKPOINT,
        "catalogSha256": seal["catalogSha256"],
        "graphIndexSha256": graph_sha,
        "pairs": pairs,
        "graphs": {key: graph_cache[key] for key in sorted(retained) if graph_cache[key]},
        "coverage": {
            split: {
                key: counts[split, key] for key in ("crossModelPairs", "bothGraphs", "sharedGraphs")
            }
            for split in SPLITS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("semantic", "graph-index", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = export(args.semantic, args.graph_index)
    atomic_json(args.output, result)
    print(result["coverage"], len(result["graphs"]))


if __name__ == "__main__":
    main()
