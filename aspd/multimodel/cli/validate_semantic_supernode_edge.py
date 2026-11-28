"""Replay a frozen, explicitly selected same-anchor supernode intervention plan."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import resource
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml

from aspd.multimodel.cli.collect_aspd_semantic_contributions import anchor_rows, verified_json
from aspd.multimodel.cli.discover_semantic_supernodes import CHECKPOINT
from aspd.multimodel.cli.measure_semantic_ot_graphs import gather_original_batch
from aspd.multimodel.discovery.semantic_ot import is_heldout
from aspd.multimodel.discovery.semantic_ot_v2 import SPLIT_SEED
from aspd.multimodel.discovery.semantic_supernodes import aggregate_taylor
from aspd.multimodel.posthoc.artifacts import sha256_file, write_json
from aspd.multimodel.posthoc.edges import edge_scores_for_group
from aspd.multimodel.posthoc.ids import ComponentId
from aspd.multimodel.posthoc.supernode_ablation import exact_effects
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend


def read(path):
    return json.loads(path.read_text())


def run(args):
    started = time.monotonic()
    plan = read(args.plan)
    seal = read(args.pilot / "COMPLETE.json")
    if sha256_file(args.pilot / "catalog.json") != seal["sha256"]["catalog.json"]:
        raise ValueError("Pilot seal mismatch")
    catalog = read(args.pilot / "catalog.json")
    if plan["runId"] != catalog["runId"] or plan["gates"] != "frozen-binary":
        raise ValueError("Expected the frozen semantic partition and binary support")
    contract_hash = sha256_file(args.measurements / "contract.json")
    contract = read(args.measurements / "contract.json")
    options = yaml.safe_load(args.config.read_text())["backend"]["options"]
    if (
        contract_hash != catalog["measurementContract"]
        or sha256_file(args.config) != contract["configSha256"]
        or sha256_file(Path(options["checkpoint"]) / "manifest.json") != CHECKPOINT
    ):
        raise ValueError("Replay differs from the original catalog or frozen checkpoint")
    groups = {
        cluster["id"]: (model, cluster["members"])
        for model, partition in catalog["models"].items()
        for cluster in partition["clusters"]
    }
    components = {item["id"]: item for item in contract["components"]}
    gate_columns = [name.split("/") for name in contract["gateColumns"]]
    rows_by_pool = defaultdict(list)
    for request in plan["requests"]:
        rows_by_pool[int(request["pool"])].append(request)
    if not rows_by_pool:
        raise ValueError("The plan has no anchors")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.backends.cuda.matmul.allow_tf32 = False
    measurements = []
    with TemporaryDirectory(prefix="aspd-supernode-intervention-") as folder:
        backend = V31CheckpointBackend(options, work_dir=Path(folder))
        try:
            if json.loads(json.dumps(backend.provenance())) != contract["backend"]:
                raise ValueError("Replay backend differs from the frozen forward")
            for pool, requests in sorted(rows_by_pool.items()):
                directory = args.measurements / f"pool-{pool:06}"
                meta = verified_json(directory, "meta.json")
                if meta["contractSha256"] != contract_hash or meta["pool"] != pool:
                    raise ValueError("Pool contract mismatch")
                with gzip.open(directory / "anchors.json.gz", "rt") as handle:
                    archived = json.load(handle)
                aligned = next(backend.aligned_pools(1, skip=pool))
                if (
                    anchor_rows(
                        aligned, backend.tokenizers, int(backend.config.data.sequence_length)
                    )
                    != archived
                ):
                    raise ValueError("Exact native anchor replay failed")
                encoded = backend.encode_aligned_pool(aligned)
                actual = torch.stack(
                    [encoded.continuous[site][:, int(g)].float() for site, g in gate_columns], dim=1
                ).numpy()
                with np.load(directory / "values.npz", allow_pickle=False) as values:
                    expected = values["gates"]
                if not np.array_equal(actual > 0, expected > 0) or not np.allclose(
                    actual, expected, rtol=0.01, atol=0.002
                ):
                    raise ValueError("Replayed continuous or binary support differs")
                for request in requests:
                    anchor = request["anchor"]
                    index = aligned.anchor_ids.index(anchor)
                    _, document_index, boundary, native = archived["anchors"][index]
                    document = archived["documents"][document_index]["id"]
                    if not is_heldout(document, SPLIT_SEED):
                        raise ValueError("Exact confirmatory replay requires a held-out document")
                    model, sources = groups[request["from"]]
                    target_model, targets = groups[request["to"]]
                    if model != target_model:
                        raise ValueError("Native causal attribution is within one model")
                    base = {
                        "anchor": anchor,
                        "document": document,
                        "pool": pool,
                        "boundary": boundary,
                        "native": native,
                        "model": model,
                        "from": request["from"],
                        "to": request["to"],
                    }
                    inputs = gather_original_batch(encoded.pool.models[model], [index]).to(
                        backend.device
                    )
                    gates = {
                        site: values[[index]].to(backend.device)
                        for site, values in encoded.pool.gates.items()
                    }
                    target_ids = tuple(map(ComponentId.from_key, targets))
                    specs, factors = backend.hook_specs(model), backend.factors(model)
                    model_object = backend.source_model(model)
                    target_active = {a: gates[a.matrix.site][:, a.feature] for a in target_ids}
                    taylor_records = edge_scores_for_group(
                        model_object,
                        inputs,
                        [anchor],
                        gates,
                        specs,
                        factors,
                        target_active,
                    )
                    effects = {
                        (b.key, record.downstream.key): float(effect)
                        for record in taylor_records
                        for b, effect in zip(record.upstream, record.scores, strict=True)
                    }
                    for name, current in [
                        ("selected", sources),
                        *[
                            (f"control-{i}", control)
                            for i, control in enumerate(request.get("controls", []), 1)
                        ],
                    ]:
                        source_ids = tuple(map(ComponentId.from_key, current))
                        if name != "selected":
                            expected_matrices = sorted(
                                ComponentId.from_key(key).matrix for key in sources
                            )
                            if sorted(b.matrix for b in source_ids) != expected_matrices or set(
                                current
                            ) & set(sources):
                                raise ValueError(
                                    "Ablation controls must match matrices and exclude source members"
                                )
                            if any(
                                not bool(gates[b.matrix.site][0, b.feature]) for b in source_ids
                            ):
                                raise ValueError("Matched controls must fire at this anchor")
                        signs = {
                            (b, a): effects[b, a]
                            for b in current
                            for a in targets
                            if (b, a) in effects
                        }
                        # Frozen inactive support makes the computed perturbation identically zero.
                        for a in target_ids:
                            for b in source_ids:
                                if b.matrix.is_upstream_of(a.matrix) and (
                                    not bool(gates[b.matrix.site][0, b.feature])
                                    or not bool(gates[a.matrix.site][0, a.feature])
                                ):
                                    signs[b.key, a.key] = 0.0
                        taylor = aggregate_taylor(
                            current,
                            targets,
                            signs,
                            {a: components[a]["uNorm"] / components[a]["sigma"] for a in targets},
                        )
                        exact = exact_effects(
                            model_object,
                            inputs,
                            gates,
                            specs,
                            factors,
                            source_ids,
                            target_ids,
                            {a: components[a.key]["sigma"] for a in target_ids},
                            matrix_scales={a.matrix: components[a.key]["sigma"] for a in target_ids}
                            if request.get("matrixOutputs")
                            else None,
                            layer_scales=request.get("layerScales"),
                            capture_logits=request.get("predictionDistribution", False),
                            continuation_ids=request.get("continuationIds", []),
                        )
                        measurements.append(
                            {
                                **base,
                                "condition": name,
                                "members": current,
                                "taylor": taylor,
                                "exact": to_json(exact),
                            }
                        )
        finally:
            backend.close()
    write_json(
        args.output,
        {
            "schemaVersion": 1,
            "runId": catalog["runId"],
            "checkpoint": CHECKPOINT,
            "gates": "frozen-binary",
            "measurementContract": contract_hash,
            "planSha256": sha256_file(args.plan),
            "sourceSha256": sha256_file(Path(__file__)),
            "measurements": measurements,
            "seconds": time.monotonic() - started,
            "gpuPeakBytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            "cpuPeakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    )


def to_json(value):
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Invalid intervention output")
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Invalid intervention output")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "plan", "config", "measurements", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
