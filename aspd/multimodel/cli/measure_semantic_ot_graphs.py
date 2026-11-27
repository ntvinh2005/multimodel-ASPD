"""Replay bounded held-out Semantic OT graph witnesses on their exact measured pools."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml

from aspd.multimodel.cli.collect_aspd_semantic_contributions import (
    CHECKPOINT,
    anchor_rows,
    verified_json,
)
from aspd.multimodel.cli.evaluate_aspd_v3_induction import _load_references
from aspd.multimodel.cli.install_aspd_semantic_discovery import verify
from aspd.multimodel.cli.measure_induction_query import _bundle_target
from aspd.multimodel.cli.measure_negation_comprehensive import measure_frozen_inputs, unpack
from aspd.multimodel.posthoc.artifacts import runtime_provenance, sha256_file, write_json
from aspd.multimodel.posthoc.ids import ComponentId
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend


def read(path):
    return json.loads(path.read_bytes())


def select_anchors(examples, count):
    """Choose early common anchors from distinct documents and favor distinct sources."""
    components = sorted(examples)
    shared = set.intersection(*(set(examples[key]) for key in components))
    ordered = sorted(shared, key=lambda a: (examples[components[0]][a]["pool"], a))
    selected, documents, sources = [], set(), set()
    for independent_source in (True, False):
        for anchor in ordered:
            row = examples[components[0]][anchor]
            doc, source = row["document"], row["source"]
            if doc in documents or (independent_source and source in sources):
                continue
            selected.append(anchor)
            documents.add(doc)
            sources.add(source)
            if len(selected) == count:
                return selected
    raise ValueError(f"Only {len(selected)} independent common anchors; expected {count}")


def load_plan(bundle: Path, measurements: Path, count: int):
    catalog = verify(bundle)
    catalog_sha = sha256_file(bundle / "catalog.json")
    contract_path = measurements / "contract.json"
    if sha256_file(contract_path) != catalog["runId"] or catalog["checkpoint"] != CHECKPOINT:
        raise ValueError("The measurement contract and catalog do not match")
    contract = read(contract_path)
    pairs = [
        {side: pair[side] for side in ("left", "right")}
        for pair in catalog["pairs"]
        if all(catalog["components"][pair[side]]["gate"] == "mid/515" for side in ("left", "right"))
        and catalog["components"][pair["left"]]["model"]
        != catalog["components"][pair["right"]]["model"]
    ]
    components = sorted({key for pair in pairs for key in pair.values()})
    if len(pairs) != 2 or {ComponentId.from_key(key).model for key in components} != {
        "gemma",
        "smol",
        "granite",
    }:
        raise ValueError("Expected the two cross-model MID:515 pairs")
    examples = {}
    for key in components:
        asset = bundle / catalog["examples"][key]["file"]
        with gzip.open(asset, "rt", encoding="utf-8") as stream:
            examples[key] = {
                row["anchor"]: row
                for row in json.load(stream)["examples"]["heldout"]
                if not row["contrast"] and row["coefficient"] != 0
            }
    chosen = select_anchors(examples, count)
    by_pool = defaultdict(list)
    for anchor in chosen:
        first = examples[components[0]][anchor]
        if any(
            any(
                examples[key][anchor][field] != first[field]
                for field in ("pool", "boundary", "document", "source", "native")
            )
            for key in components
        ):
            raise ValueError(f"Different example identities for {anchor}")
        by_pool[first["pool"]].append(anchor)
    plan = {
        "schemaVersion": 1,
        "catalogSha256": catalog_sha,
        "checkpoint": CHECKPOINT,
        "runId": catalog["runId"],
        "dataset": "natural-validation",
        "split": "heldout",
        "pairs": pairs,
        "components": components,
        "anchors": chosen,
        "pools": dict(by_pool),
        "selection": "Earliest shared anchors with distinct documents, preferring distinct sources",
    }
    return plan, contract, examples


def run(args, plan, contract, examples):
    options = yaml.safe_load(args.config.read_text())["backend"]["options"]
    if sha256_file(args.config) != contract["configSha256"]:
        raise ValueError("Replay configuration differs from fresh contribution extraction")
    if sha256_file(Path(options["checkpoint"]) / "manifest.json") != CHECKPOINT:
        raise ValueError("Unexpected frozen checkpoint")
    reference = Path(contract["reference"])
    for name, digest in contract["references"].items():
        if sha256_file(reference / name / "COMPLETE") != digest:
            raise ValueError(f"The reference seal changed for {name}")
    importance, _ = _load_references(reference)
    torch.backends.cuda.matmul.allow_tf32 = False
    components = [ComponentId.from_key(key) for key in plan["components"]]
    columns = {row["id"]: i for i, row in enumerate(contract["components"])}
    gate_columns = [key.split("/") for key in contract["gateColumns"]]
    snapshots, witnesses = {}, {}
    with TemporaryDirectory(prefix="semantic-graphs-stream-") as tmp:
        backend = V31CheckpointBackend(options, work_dir=Path(tmp))
        try:
            if json.loads(json.dumps(backend.provenance())) != contract["backend"]:
                raise ValueError("Replay backend differs from the original evaluation")
            for ordinal, anchors in sorted(plan["pools"].items()):
                aligned = next(backend.aligned_pools(1, skip=ordinal))
                directory = args.measurements / f"pool-{ordinal:06}"
                meta = verified_json(directory, "meta.json")
                if meta["contractSha256"] != plan["runId"] or meta["pool"] != ordinal:
                    raise ValueError("Replay pool seal does not match the original measurement")
                with gzip.open(directory / "anchors.json.gz", "rt", encoding="utf-8") as stream:
                    original = json.load(stream)
                replay = anchor_rows(
                    aligned, backend.tokenizers, int(backend.config.data.sequence_length)
                )
                if replay != original:
                    raise ValueError(
                        f"Replayed pool {ordinal} differs in canonical or native identity"
                    )
                encoded = backend.encode_aligned_pool(aligned)
                selected = [aligned.anchor_ids.index(anchor) for anchor in anchors]
                with np.load(directory / "values.npz", allow_pickle=False) as saved:
                    values, coefficients = saved["gates"], saved["coefficients"]
                    actual = torch.stack(
                        [
                            encoded.continuous[site][:, int(index)].float()
                            for site, index in gate_columns
                        ],
                        dim=1,
                    ).numpy()
                    if not np.array_equal(actual > 0, values > 0) or not np.allclose(
                        actual, values, rtol=0.01, atol=0.002
                    ):
                        raise ValueError(
                            f"Natural BatchTopK replay differs from saved gates in pool {ordinal}"
                        )
                rows = []
                for i, anchor in zip(selected, anchors, strict=True):
                    _, doc_index, boundary, native = original["anchors"][i]
                    doc = original["documents"][doc_index]
                    first = examples[components[0].key][anchor]
                    if (doc["id"], doc["source"], boundary, native) != (
                        first["document"],
                        first["source"],
                        first["boundary"],
                        first["native"],
                    ):
                        raise ValueError(f"Graph replay does not match the catalog anchor {anchor}")
                    snapshots[anchor] = {
                        "anchor": anchor,
                        "prefix": doc["text"][:boundary],
                        "boundary": boundary,
                        "pool": ordinal,
                        "document": doc["id"],
                        "source": doc["source"],
                        "positions": {model: item[0] for model, item in native.items()},
                        "native": native,
                        "activation": float(encoded.continuous["mid"][i, 515]),
                        "binarySupport": {
                            site: encoded.pool.gates[site][i].nonzero(as_tuple=True)[0].tolist()
                            for site in ("pre", "mid")
                        },
                    }
                    row = {"anchor_id": anchor}
                    for site, indices in snapshots[anchor]["binarySupport"].items():
                        row[site] = {
                            "active_gate_ids": indices,
                            "values": encoded.continuous[site][i, indices].float().tolist(),
                        }
                    rows.append(row)
                inputs = {model: inp.subset(selected) for model, inp in encoded.pool.models.items()}
                measures = measure_frozen_inputs(
                    backend,
                    inputs,
                    rows,
                    {(anchor, component) for anchor in anchors for component in components},
                )
                for i, anchor in zip(selected, anchors, strict=True):
                    for component in components:
                        measure = measures[anchor, component]
                        original_coefficient = float(coefficients[i, columns[component.key]])
                        observed = measure[0]
                        if not math.isclose(
                            original_coefficient,
                            examples[component.key][anchor]["coefficient"],
                            rel_tol=0,
                            abs_tol=1e-8,
                        ):
                            raise ValueError("Archived coefficient differs from the example")
                        if not math.isclose(
                            observed, original_coefficient, rel_tol=0.05, abs_tol=0.01
                        ):
                            raise ValueError(
                                f"Taylor replay coefficient differs for {component.key}: {observed} vs {original_coefficient}"
                            )
                        target = _bundle_target(component, unpack(measure), measure[1], importance)
                        witnesses[f"{component.key}|{anchor}"] = {
                            "anchor": anchor,
                            "component": component.key,
                            "model": component.model,
                            "pool": ordinal,
                            "artifact": plan["catalogSha256"],
                            "target": target,
                            "originalCoefficient": original_coefficient,
                            "edges": [
                                {
                                    "source": p["component"]["id"],
                                    "target": component.key,
                                    "E": p["E"],
                                    "w": p["w"],
                                    "rank": p["rank"],
                                }
                                for p in target["parents"]
                            ],
                        }
                print(
                    f"Verified pool {ordinal}: {len(anchors)} anchors and {len(measures)} target graphs",
                    flush=True,
                )
        finally:
            backend.close()
    if len(witnesses) != len(components) * len(plan["anchors"]):
        raise ValueError("Incomplete measured graph bundle")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    graph = args.output / "graphs.json"
    write_json(
        graph,
        {
            **{
                key: plan[key]
                for key in (
                    "schemaVersion",
                    "catalogSha256",
                    "runId",
                    "checkpoint",
                    "dataset",
                    "split",
                    "pairs",
                )
            },
            "anchors": snapshots,
            "graphs": witnesses,
            "anchorOrder": plan["anchors"],
            "method": {
                "scope": "incoming Taylor parents of the selected component",
                "kPar": 10,
                "tau": 0,
                "sameAnchorOnly": True,
                "coefficientReplayTolerance": {"relative": 0.05, "absolute": 0.01},
            },
            "provenance": {
                "configSha256": contract["configSha256"],
                "backend": contract["backend"],
                "runtime": runtime_provenance(),
                "measurementRunId": plan["runId"],
                "collectorCommit": contract["git_sha"],
                "referenceSeals": contract["references"],
            },
        },
    )
    write_json(
        args.output / "COMPLETE.json",
        {
            "sha256": sha256_file(graph),
            "catalogSha256": plan["catalogSha256"],
            "runId": plan["runId"],
            "anchors": len(snapshots),
            "targetGraphs": len(witnesses),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "measurements", "config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--anchors", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    plan, contract, examples = load_plan(args.bundle, args.measurements, args.anchors)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.plan_only:
        run(args, plan, contract, examples)


if __name__ == "__main__":
    main()
