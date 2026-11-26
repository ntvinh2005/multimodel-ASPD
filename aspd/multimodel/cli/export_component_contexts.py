"""Rank exact matched Workbench contexts using the archived component coefficients."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from aspd.multimodel.posthoc.ids import ComponentId


def verified(root, ref):
    path = root / ref["path"]
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != ref["sha256"]:
        raise ValueError(f"Unverified data in {path}")
    return json.loads(gzip.decompress(content) if path.suffix == ".gz" else content)


def export(root, dataset, groups, token, components):
    index_bytes = (root / "index.json").read_bytes()
    index = json.loads(index_bytes)
    checkpoint = index["checkpointSha256"]
    rows = []
    for key in groups:
        group = verified(root, index["datasets"][dataset][key])
        if (group["checkpointSha256"], group["dataset"], group["groupId"]) != (checkpoint, dataset, key):
            raise ValueError("Group artifact differs from the requested provenance")
        for example in group["examples"]:
            start = 0
            matches = []
            for anchor in example["positions"]:
                stop = anchor["character_boundary"]
                if example["prompt"][start:stop].strip().casefold() == token.casefold():
                    matches.append(anchor)
                start = stop
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one canonical {token!r} span in {example['example_id']}")
            anchor = matches[0]
            identifier = anchor["anchor_id"]
            ref = index["graphs"].get(identifier)
            graph = verified(root, ref) if ref else None
            if graph and (graph["checkpointSha256"], graph["anchorId"], graph["poolId"], graph["exampleId"]) != (
                checkpoint, identifier, anchor["pool_id"], example["example_id"],
            ):
                raise ValueError("Graph artifact differs from the exact example and anchor")
            by_component = {}
            for identity in components:
                support = anchor[identity.gate.site]
                g_s = dict(zip(support["active_gate_ids"], support["values"], strict=True)).get(identity.feature, 0)
                if not g_s:
                    coefficient = magnitude = 0
                    reading = None
                    relative = 0
                else:
                    measured = graph["componentTargets"].get(identity.key) if graph else None
                    coefficient = measured[0] if measured else None
                    reading = coefficient
                    magnitude = abs(coefficient) * measured[1] if measured else None
                    relative = magnitude / graph["outputScale"][identity.matrix.key] if measured else None
                if not all(value is None or math.isfinite(value) for value in (g_s, coefficient, reading, magnitude, relative)):
                    raise ValueError("Non-finite component context")
                by_component[identity.key] = {"g_s": g_s, "support": bool(g_s), "e": coefficient, "read": reading,
                                              "magnitude": magnitude, "normalized": relative}
            rows.append({"dataset": dataset, "group": key, "example": example["example_id"], "condition": example["condition"],
                         "prompt": example["prompt"], "anchor": identifier, "pool": anchor["pool_id"],
                         "nativeTokenPositions": anchor["model_positions"], "source": ref, "components": by_component})
    ranks = {}
    for identity in components:
        ranks[identity.key] = {
            metric: [row["anchor"] for row in sorted(
                (row for row in rows if row["components"][identity.key][metric] is not None),
                key=lambda row: (-abs(row["components"][identity.key][metric]), row["anchor"]))]
            for metric in ("g_s", "e", "magnitude")
        }
    return {"schemaVersion": 1, "dataset": dataset, "checkpoint": checkpoint, "token": token,
            "tokenIndexSha256": hashlib.sha256(index_bytes).hexdigest(),
            "groups": groups, "rows": rows, "rankedByAbsoluteValue": ranks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--component", action="append", type=ComponentId.from_key, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.tokens, args.dataset, args.group, args.token, args.component)
    with args.output.open("x") as handle:
        json.dump(result, handle, allow_nan=False, separators=(",", ":"))
        handle.write("\n")


if __name__ == "__main__":
    main()
