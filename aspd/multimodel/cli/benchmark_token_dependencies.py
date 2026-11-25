"""Benchmark complete same-anchor component targets on two nonfinal Negation positions per condition."""

from __future__ import annotations

import argparse
import gzip
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import torch

from aspd.multimodel.cli.evaluate_aspd_v3_induction import _anchor_metadata
from aspd.multimodel.cli.evaluate_aspd_v3_negation import backend_options, jsonl
from aspd.multimodel.cli.measure_induction_query import PARENT_RULE, _supports
from aspd.multimodel.cli.measure_negation_comprehensive import (
    active_gates,
    load_inputs,
    measure_missing,
    official_targets,
)
from aspd.multimodel.posthoc.artifacts import runtime_provenance, sha256_file, write_json
from aspd.multimodel.posthoc.ids import SITES, ComponentId
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend
from aspd.multimodel.v3.induction import (
    MODEL_ORDER,
    merge_task_and_background,
    task_pool_from_verified,
)
from aspd.multimodel.v3.negation import FACTORIAL


def benchmark_batches(rows, group):
    cases = defaultdict(list)
    for row in rows:
        if row["group_id"] == group and not row["final_query"]:
            cases[row["condition"]].append(row)
    if any(len(cases[condition]) < 3 for condition in FACTORIAL):
        raise ValueError("each matched condition needs at least three nonfinal anchors")
    for condition in FACTORIAL:
        cases[condition].sort(key=lambda row: row["character_boundary"])
    return [
        [cases[condition][len(cases[condition]) * numerator // 3] for condition in FACTORIAL]
        for numerator in (1, 2)
    ]


def component_requests(rows, matrices):
    return {
        (row["anchor_id"], ComponentId(matrix, gate.feature))
        for row in rows
        for gate in active_gates(row)
        for matrix in matrices
        if matrix.site == gate.site
    }


def study_pools(backend, preparation, provenance, output):
    manifest_path = preparation / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    path = preparation / "examples.jsonl"
    if (
        sha256_file(path) != manifest["examplesSha256"]
        or manifest["checkpointSha256"] != provenance["checkpointSha256"]
        or manifest["configSha256"] != provenance["configSha256"]
    ):
        raise ValueError("Selection Frame preparation does not match the benchmark")
    for key, model in manifest["models"].items():
        runtime = backend.provenance()["models"][key]
        if (runtime["model_id"], runtime["revision"], runtime["tokenizer_revision"]) != (
            model["id"], model["revision"], model["tokenizer_revision"],
        ):
            raise ValueError(f"Selection Frame source model and tokenizer mismatch: {key}")
    rows = list(jsonl(path))
    by_id = {row["example"]["example_id"]: row for row in rows}
    metadata = _anchor_metadata(rows)
    pools = manifest["poolExamples"]
    if sorted(key for pool in pools for key in pool) != sorted(by_id):
        raise ValueError("frozen Selection Frame pool plan differs from prepared examples")
    study = {key: {**row["example"], "positions": []} for key, row in by_id.items()}
    pool_records = []
    for index, (membership, background) in enumerate(zip(pools, backend.aligned_pools(len(pools)), strict=True)):
        task = task_pool_from_verified([by_id[key] for key in membership], mechanism="selection-frame")
        aligned = merge_task_and_background(task, background, pool_anchors=2048)
        encoded = backend.encode_aligned_pool(aligned)
        supports = {}
        for site in SITES:
            gate = encoded.pool.gates[site]
            values = encoded.continuous[site]
            if int(gate.sum()) != 2048 * 32 or not torch.equal(gate, values > 0):
                raise ValueError("natural Selection Frame BatchTopK support changed")
            supports[site] = gate[:task.anchor_count].cpu(), values[:task.anchor_count].cpu()
        pool_id = f"selection-frame-v1-pool-{index:03d}"
        for row, anchor_id in enumerate(task.anchor_ids):
            if encoded.pool.anchor_ids[row] != anchor_id:
                raise ValueError("Selection Frame encoding reordered its aligned anchors")
            meta = metadata[anchor_id]
            payload = {"pool_id": pool_id, "anchor_id": anchor_id, **meta}
            for site, (g, values) in supports.items():
                active = g[row].nonzero(as_tuple=True)[0]
                payload[site] = {"active_gate_ids": active.tolist(), "values": values[row, active].tolist()}
            study[meta["example_id"]]["positions"].append(payload)
        pool_records.append({
            "poolId": pool_id, "batchTopK": 32, "poolAnchors": 2048,
            "taskAnchors": task.anchor_count, "exampleIds": membership,
            "backgroundAnchorIds": list(encoded.pool.anchor_ids[task.anchor_count:]),
        })
    if sum(len(item["positions"]) for item in study.values()) != manifest["alignedAnchors"]:
        raise ValueError("Selection Frame coverage differs from prepared anchors")
    write_json(output, {"dataset": "selection-frame/discovery", "checkpointSha256": provenance["checkpointSha256"],
                        "preparationSha256": sha256_file(manifest_path), "pools": pool_records, "examples": list(study.values())})


def export_batch(path, rows, records, representatives, importance, provenance):
    by_anchor = defaultdict(dict)
    for (anchor, component), payload in records.items():
        coefficient, norm, parents = payload
        by_anchor[anchor][component.key] = [coefficient, norm, parents]
    unpacked = {
        "provenance": provenance,
        "anchors": [{**row, "targets": by_anchor[row["anchor_id"]]} for row in rows],
        "representativeTargets": sorted([anchor, component.key] for anchor, component in representatives),
        "outputScale": {component.matrix.key: importance[component.matrix].sigma for _, component in records},
    }
    with gzip.open(path, "xt", encoding="utf-8", compresslevel=3) as handle:
        json.dump(unpacked, handle, separators=(",", ":"), allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "preparation", "discovery", "selection-preparation", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--group", default="negation-discovery-multiple-0026")
    args = parser.parse_args()
    _, prepared_final, importance, reps, reference, measured, provenance = load_inputs(
        args.config, args.preparation, args.discovery
    )
    prepared = {}
    selected_docs = [row for row in prepared_final.values() if row["example"]["group_id"] == args.group]
    meta = _anchor_metadata(selected_docs)
    for row in selected_docs:
        prepared.update({anchor["anchor_id"]: row for anchor in row["alignment"]["anchors"]})
    fall = [row for row in jsonl(args.discovery / "anchor_activations.jsonl") if row["anchor_id"] in meta]
    if len(fall) != len(meta) or {row["anchor_id"] for row in fall} != meta.keys():
        raise ValueError("benchmark activation rows differ from prepared anchors")
    for row in fall:
        if any(row[key] != value for key, value in meta[row["anchor_id"]].items()):
            raise ValueError("benchmark activation provenance differs from prepared anchor")
        if any(not 0 < value < float("inf") for support in _supports(row).values() for value in support.values()):
            raise ValueError("nonfinite or nonpositive archived BatchTopK activation")
    batches = benchmark_batches(fall, args.group)
    if any(len({row["pool_id"] for row in batch}) != 1 for batch in batches):
        raise ValueError("matched benchmark conditions do not share a pool")
    args.output.mkdir(parents=True, exist_ok=False)
    load_start = time.monotonic()
    backend = V31CheckpointBackend(backend_options(reference), work_dir=args.output / "stream_state")
    initialization = time.monotonic() - load_start
    info = []
    try:
        if measured["backend"] != backend.provenance():
            raise ValueError("loaded source or checkpoint differs from the archived Negation run")
        matrices = [matrix for model in MODEL_ORDER for matrix in backend.factors(model)]
        provenance = {
            **provenance, "parentRule": {"k": PARENT_RULE.k_par, "tau": PARENT_RULE.tau},
            "sameAnchorOnly": True, "runtime": runtime_provenance(),
        }
        for index, rows in enumerate(batches):
            requests = component_requests(rows, matrices)
            start = time.monotonic()
            results = measure_missing(backend, prepared, rows, requests)
            measured_seconds = time.monotonic() - start
            path = args.output / f"batch-{index}.json.gz"
            export_batch(path, rows, results, official_targets(rows, reps), importance, provenance)
            record = {
                "path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                "anchors": [row["anchor_id"] for row in rows],
                "activeGateAnchors": sum(len(active_gates(row)) for row in rows),
                "componentTargets": len(requests), "measured": len(results),
                "emptyParentSets": sum(not payload[2] for payload in results.values()),
                "byModel": {model: sum(comp.model == model for _, comp in requests) for model in MODEL_ORDER},
                "edges": sum(len(payload[2]) for payload in results.values()), "measureSeconds": measured_seconds,
            }
            info.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        study = args.output / "selection_frame_activations.json"
        study_pools(backend, args.selection_preparation, provenance, study)
    finally:
        backend.close()
    manifest = args.output / "manifest.json"
    write_json(manifest, {
        **provenance, "mode": "eight-nonfinal-anchor-benchmark", "batches": info,
        "initializationSeconds": initialization, "gpuPeakAllocatedBytes": torch.cuda.max_memory_allocated(),
        "gpuPeakReservedBytes": torch.cuda.max_memory_reserved(), "hostPeakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "selectionActivationsSha256": sha256_file(study),
    })
    write_json(args.output / "COMPLETE", {"manifestSha256": sha256_file(manifest)})


if __name__ == "__main__":
    main()
