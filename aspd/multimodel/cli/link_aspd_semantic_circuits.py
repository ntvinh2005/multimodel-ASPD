"""Find same-anchor directed paths joining endpoints from the Natural Validation OT groups."""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from aspd.multimodel.cli.discover_aspd_circuits import atomic_json
from aspd.multimodel.discovery.index import CHECKPOINT, Index, decode, encode, load, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("semantic", "graph-index", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    seal = load(args.semantic / "COMPLETE.json")
    data = load(args.semantic / "catalog.json", seal["catalogSha256"])
    if (
        data["dataset"] != "natural-validation"
        or data["provenance"]["checkpoint"] != CHECKPOINT
        or data["provenance"]["graphIndexSha256"] != sha(args.graph_index / "manifest.json")
    ):
        raise ValueError("Semantic and graph provenance differ")
    cluster_of = {
        component: group["id"] for group in data["clusters"] for component in group["components"]
    }
    opportunities = defaultdict(set)
    for component, value in data["components"].items():
        for split, examples in value["examples"].items():
            for row in examples:
                opportunities[split, row["anchor"], value["model"]].add(encode(component))
    index = Index(args.graph_index)
    lookup = {
        (row["anchor"], row["model"]): i
        for i, row in enumerate(index.scopes)
        if row["dataset"] == "natural-validation"
    }
    links = defaultdict(list)
    checked = defaultdict(int)
    for (split, anchor, model), components in sorted(opportunities.items()):
        scope = lookup.get((anchor, model))
        if len(components) < 2 or scope is None:
            continue
        checked[split] += 1
        measured, _ = index.graph(scope)
        outbound = defaultdict(dict)
        for a, b, effect, weight in measured.tolist():
            outbound[a][b] = (effect, weight)
        for left, right in combinations(sorted(components), 2):
            for a, b in ((left, right), (right, left)):
                direct = outbound[a]
                if b in direct:
                    path = (a, b)
                else:
                    available = [node for node in direct if b in outbound[node]]
                    if not available:
                        continue
                    mid = max(
                        available,
                        key=lambda node: (min(direct[node][1], outbound[node][b][1]), -node),
                    )
                    path = (a, mid, b)
                endpoints = decode(a), decode(b)
                edges = [
                    dict(
                        source=decode(u), target=decode(v), E=outbound[u][v][0], w=outbound[u][v][1]
                    )
                    for u, v in zip(path[:-1], path[1:], strict=True)
                ]
                links[endpoints].append(
                    dict(
                        anchor=anchor,
                        model=model,
                        split=split,
                        pool=data["anchors"][anchor]["pool"],
                        artifact=index.scopes[scope]["artifact"],
                        edges=edges,
                    )
                )
    pairs = [
        dict(
            left=left,
            right=right,
            leftCluster=cluster_of[left],
            rightCluster=cluster_of[right],
            discovery=sum(row["split"] == "discovery" for row in witnesses),
            heldout=sum(row["split"] == "heldout" for row in witnesses),
            witnesses=witnesses,
        )
        for (left, right), witnesses in sorted(links.items())
    ]
    pairs.sort(key=lambda row: (-row["heldout"], -row["discovery"], row["left"], row["right"]))
    if args.output.exists():
        raise FileExistsError(args.output)
    atomic_json(
        args.output,
        dict(
            checkpoint=CHECKPOINT,
            dataset="natural-validation",
            catalogSha256=seal["catalogSha256"],
            graphIndexSha256=sha(args.graph_index / "manifest.json"),
            comparableScopes=checked,
            pairs=pairs,
        ),
    )
    print(
        dict(
            comparableScopes=checked,
            pairs=len(pairs),
            betweenGroups=sum(row["leftCluster"] != row["rightCluster"] for row in pairs),
        )
    )


if __name__ == "__main__":
    main()
