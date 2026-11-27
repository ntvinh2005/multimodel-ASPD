"""Publish Natural Validation witnesses from the original catalog with recovered text."""

import argparse
import json
from pathlib import Path

from aspd.multimodel.cli.discover_aspd_circuits import atomic_json
from aspd.multimodel.discovery.index import CHECKPOINT, load, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("catalog", "recovery", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    source = load(args.catalog)
    recovery = load(args.recovery / "COMPLETE.json")
    anchor_path = args.recovery / "anchors.jsonl"
    if (
        source["provenance"]["checkpoint"] != CHECKPOINT
        or recovery["checkpoint"] != CHECKPOINT
        or recovery["split"] != "validation"
        or sha(anchor_path) != recovery["anchorsSha256"]
    ):
        raise ValueError("Natural Validation provenance mismatch")
    with anchor_path.open() as handle:
        anchors = {row["anchor"]: row for row in map(json.loads, handle)}

    output = []
    for candidate in source["candidates"]:
        if candidate["dataset"] != "natural-validation":
            continue
        witnesses = []
        for graph in candidate["witnesses"]:
            if graph["dataset"] != "natural-validation":
                continue
            text = anchors[graph["anchor"]]
            witnesses.append(
                {
                    **graph,
                    "text": text["prefix"],
                    "boundary": text["boundary"],
                    "pool": str(text["pool"]),
                    "example": text["document"],
                    "group": text["document"],
                    "positions": text["positions"],
                }
            )
        recurrence = [
            row for row in candidate["recurrence"] if row["dataset"] == "natural-validation"
        ]
        # Cross-dataset aggregates remain available in the source catalog.
        output.append(
            {
                **candidate,
                "witnesses": witnesses,
                "recurrence": recurrence,
                "anchorCount": sum(row["anchors"] for row in recurrence),
                "crossModel": {
                    "models": [candidate["model"]],
                    "sharedExactAnchors": 0,
                    "peers": [],
                },
                "prefixAudit": None,
                "measuredContextSelectivity": None,
                "conditionalGraphNull": (
                    candidate["conditionalGraphNull"]
                    if (candidate.get("conditionalGraphNull") or {}).get("anchor")
                    in {w["anchor"] for w in witnesses}
                    else None
                ),
                "interpretation": {
                    **candidate["interpretation"],
                    "independentActivationExamples": [],
                    "counterexampleContexts": [],
                },
            }
        )
    args.output.parent.mkdir(exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    atomic_json(
        args.output,
        {
            "schemaVersion": source["schemaVersion"],
            "provenance": {
                "checkpoint": CHECKPOINT,
                "dataset": "natural-validation",
                "catalogSha256": sha(args.catalog),
                "recoverySha256": sha(args.recovery / "COMPLETE.json"),
            },
            "candidates": output,
        },
    )
    print(f"Natural Validation candidates: {len(output)}")


if __name__ == "__main__":
    main()
