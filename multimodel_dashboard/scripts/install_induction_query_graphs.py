"""Verify the measured induction artifact and stage it as a static workbench asset."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

MODELS = {"gemma", "smol", "granite"}
CONDITIONS = {"induction", "counterfactual", "no_match", "repeated_no_association"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    complete = json.loads((root / "COMPLETE").read_text())
    raw = (root / "inductionQueryGraphs.json").read_bytes()
    selection_bytes = (root / "selection.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == complete["bundleSha256"], "graph digest mismatch")
    require(
        hashlib.sha256(selection_bytes).hexdigest() == complete["selectionSha256"],
        "selection digest mismatch",
    )
    bundle = json.loads(raw)
    require(
        bundle["schemaVersion"] == 1 and bundle["split"] == "discovery"
        and bundle["provenance"]["scope"] == "final_query_same_anchor_frozen_discovery_support",
        "unexpected graph scope",
    )
    require(
        bundle["provenance"]["kPar"] == 10 and bundle["provenance"]["tau"] == 0,
        "unexpected parent rule",
    )
    selection = bundle["selection"]
    require(selection == json.loads(selection_bytes), "bundle and frozen selection disagree")
    require(
        Counter(gate.split(":")[0] for gate in selection) == {"PRE": 6, "MID": 6},
        "unexpected gate balance",
    )
    require(
        Counter(s["kind"] for s in selection.values()) ==
        {"induction_contrast": 8, "fires_across_conditions": 4},
        "unexpected contrast/control balance",
    )
    anchors = set()
    coverage = defaultdict(dict)
    stats = Counter()
    for ex in bundle["examples"]:
        family, group, condition = ex["family"], ex["groupId"], ex["condition"]
        require(family in {"basic", "repeated_sequence"}, "unexpected family")
        require(set(ex["modelPositions"]) == MODELS, "missing native anchor positions")
        require(0 <= ex["queryStart"] < len(ex["prompt"]), "invalid final query display slice")
        seen = {graph["anchorId"] for graph in ex["graphs"].values()}
        require(len(seen) == 1 and not (seen & anchors), "ambiguous or duplicate query anchor")
        anchors.update(seen)
        for gate, graph in ex["graphs"].items():
            stats["gateAnchorRecords"] += 1
            require(group in selection[gate]["groups"], "unselected gate/group")
            key = (group, condition)
            require(key not in coverage[gate], "duplicate gate/group/condition")
            activation = graph["activation"]
            require(math.isfinite(activation) and activation >= 0, "invalid activation")
            coverage[gate][key] = family, ex["poolId"], activation
            if activation == 0:
                require(not graph["models"] and not graph["comparisons"], "inactive graph has edges")
                stats["inactiveGateAnchorRecords"] += 1
                continue
            require(set(graph["models"]) == MODELS, "active graph missing model")
            require(
                {tuple(sorted(pair["models"])) for pair in graph["comparisons"]}
                == {("gemma", "granite"), ("gemma", "smol"), ("granite", "smol")},
                "missing model comparison",
            )
            for model, result in graph["models"].items():
                stats["modelGraphs"] += 1
                distribution = result["distribution"]
                require(
                    (distribution is None and result["strength"] == 0)
                    or (distribution is not None and result["strength"] > 0
                        and math.isclose(sum(distribution.values()), 1)),
                    "parent distribution and strength disagree",
                )
                require(1 <= len(result["targets"]) <= 3, "unexpected representative count")
                for target in result["targets"]:
                    component = target["component"]
                    require(
                        component["gateId"] == gate and component["modelId"] == model
                        and component["beta"] > 0.07,
                        "unexpected representative",
                    )
                    parents = target["parents"]
                    require(len(parents) <= 10, "parent set exceeds bound")
                    stats["parentEdges"] += len(parents)
                    previous = math.inf
                    for parent in parents:
                        magnitude = abs(parent["E"])
                        require(
                            math.isfinite(magnitude) and 0 < magnitude <= previous
                            and math.isfinite(parent["w"]) and parent["w"] >= 0
                            and parent["component"]["modelId"] == model,
                            "invalid ordered parent",
                        )
                        previous = magnitude
    for gate, info in selection.items():
        require(len(set(info["groups"])) == 8, "unexpected example-group count")
        require(set(coverage[gate]) == {(g, c) for g in info["groups"] for c in CONDITIONS},
                "missing paired examples for " + gate)
        families = Counter()
        for group in info["groups"]:
            records = {c: coverage[gate][(group, c)] for c in CONDITIONS}
            require(len({r[:2] for r in records.values()}) == 1, "paired pool/family mismatch")
            families[records["induction"][0]] += 1
            positives = min(records["induction"][2], records["counterfactual"][2])
            controls = [records[c][2] for c in ("no_match", "repeated_no_association")]
            require(
                positives > max(controls) if info["kind"] == "induction_contrast"
                else min(positives, *controls) > 0,
                "paired activation does not satisfy the selection criterion",
            )
        require(families == {"basic": 4, "repeated_sequence": 4}, "unexpected family balance")
    return raw, {"uniqueAnchors": len(anchors), **stats, "sha256": complete["bundleSha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw, summary = verify(args.artifact)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(raw)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
