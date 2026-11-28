"""Prepare, export, and cluster representative components in ASPD's shared reference space."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from aspd.multimodel.cli.collect_aspd_semantic_contributions import anchor_rows, verified_json
from aspd.multimodel.cli.discover_semantic_supernodes import (
    CHECKPOINT,
    MODELS,
    SPLITS,
    add_witness,
    heatmaps,
    load,
    partition_record,
    pool_blocks,
    write_array,
    write_json,
)
from aspd.multimodel.discovery import semantic_supernodes as sn
from aspd.multimodel.discovery.semantic_ot import is_heldout
from aspd.multimodel.discovery.semantic_ot_v2 import SPLIT_SEED
from aspd.multimodel.discovery.shared_reference_ot import (
    complete_partition,
    dissimilarities,
    top_anchors,
    unit_reference,
)
from aspd.multimodel.posthoc.artifacts import sha256_file
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend

REFERENCE = "aspd-pre-mid-mean-preprojection"


def measurement_contract(root):
    path = root / "contract.json"
    raw = load(path)
    if raw["checkpoint"] != CHECKPOINT or tuple(sorted(raw["backend"]["models"])) != tuple(
        sorted(MODELS)
    ):
        raise ValueError("Expected frozen ASPD v3.1 on Gemma, SmolLM2, and Granite")
    if raw["backend"]["pool_anchors"] != 2048:
        raise ValueError("Unexpected BatchTopK pool size")
    return raw, sha256_file(path)


def prepare(args):
    raw, contract_sha = measurement_contract(args.measurements)
    if args.top_k < 1 or args.per_model is not None and args.per_model < 1:
        raise ValueError("Top K and component budget must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    cohort = []
    for model in MODELS:
        candidates = sorted(
            ((i, item) for i, item in enumerate(raw["components"]) if item["model"] == model),
            key=lambda item: (-item[1]["beta"], item[1]["id"]),
        )
        cohort.extend(candidates[: args.per_model])
    columns = [col for col, _ in cohort]
    scales = np.asarray([item["uNorm"] for _, item in cohort])
    gate_indices = [raw["gateColumns"].index(item["gate"]) for _, item in cohort]
    supports = {split: [{} for _ in cohort] for split in SPLITS}
    witnesses, documents = {}, set()
    excluded_document = None
    for pool, metadata, block, gates in pool_blocks(args.measurements, raw, contract_sha):
        docs, rows = metadata["documents"], metadata["anchors"]
        if excluded_document is None:
            excluded_document = docs[0]["id"]
        # Match the established discovery window. The first document may straddle an
        # earlier evaluation wave.
        documents.update(doc["id"] for doc in docs if doc["id"] != excluded_document)
        eligible = np.asarray([docs[row[1]]["id"] != excluded_document for row in rows])
        heldout = np.asarray([is_heldout(docs[row[1]]["id"], SPLIT_SEED) for row in rows])
        signs = block[:, columns]
        if not np.isfinite(signs).all() or np.any(signs[gates[:, gate_indices] == 0] != 0):
            raise ValueError(f"Nonzero contribution off natural support in pool {pool}")
        ids = [row[0] for row in rows]
        for split, select in (("discovery", eligible & ~heldout), ("heldout", eligible & heldout)):
            for j, values in enumerate(signs.T):
                hits = np.flatnonzero(select & (values != 0))
                if not len(hits):
                    continue
                previous = supports[split][j]
                current = {ids[i]: (float(values[i]), i) for i in hits}
                newest = [
                    (anchor, abs(value) * scales[j]) for anchor, (value, _) in current.items()
                ]
                if previous.keys() & current.keys():
                    raise ValueError("Duplicate canonical anchor across pools")
                chosen = top_anchors(
                    [*((anchor, item["strength"]) for anchor, item in previous.items()), *newest],
                    args.top_k,
                )
                result = {}
                for anchor, strength in chosen:
                    if anchor in previous:
                        result[anchor] = previous[anchor]
                        continue
                    value, i = current[anchor]
                    _, doc_i, boundary, native = rows[i]
                    doc = docs[doc_i]
                    add_witness(
                        witnesses,
                        doc["id"],
                        doc,
                        ((-abs(value), anchor), value, pool, boundary, native),
                    )
                    result[anchor] = {
                        "anchor": anchor,
                        "document": doc["id"],
                        "coefficient": value,
                        "strength": strength,
                    }
                supports[split][j] = result
    examples, counts, retained = {}, {}, []
    for j, (col, item) in enumerate(cohort):
        key = item["id"]
        if not supports["discovery"][j]:
            continue
        retained.append((col, item))
        examples[key] = {split: list(supports[split][j].values()) for split in SPLITS}
        counts[key] = {
            split: {
                "exportedNonzero": len(items),
                "distinctNonzeroDocuments": len({example["document"] for example in items}),
            }
            for split, items in examples[key].items()
        }
    if not retained:
        raise ValueError("No representative component has a nonzero discovery support")
    selected = {
        anchor
        for split in examples.values()
        for items in split.values()
        for anchor in (item["anchor"] for item in items)
    }
    witnesses = {anchor: witnesses[anchor] for anchor in sorted(selected)}
    if not args.no_heatmaps:
        # Heatmaps use raw kappa as well. The reference normalization does not change it.
        heatmaps(
            args.measurements,
            raw,
            contract_sha,
            [(i, {**item, "sigma": 1.0}) for i, item in retained],
            examples,
            witnesses,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = args.output / "evidence.json"
    write_json(evidence, {"examples": examples, "witnesses": witnesses, "counts": counts})
    write_json(
        args.output / "plan.json",
        {
            "schemaVersion": 1,
            "reference": REFERENCE,
            "measurementContract": contract_sha,
            "checkpoint": raw["checkpoint"],
            "measurementProvenance": {
                key: raw[key] for key in ("backend", "corpus", "references", "git_sha")
            },
            "cohortSize": len(cohort),
            "archiveSize": len(raw["components"]),
            "components": [item for _, item in retained],
            "emptyDiscovery": len(cohort) - len(retained),
            "topK": args.top_k,
            "perModelBudget": args.per_model,
            "splitSeed": SPLIT_SEED,
            "supportPolicy": "top measured nonzero kappa anchors; ties by anchor ID",
            "excludedBoundaryDocument": excluded_document,
            "documentCount": len(documents),
            "evidenceSha256": sha256_file(evidence),
            "sourceSha256": sha256_file(Path(__file__)),
        },
    )
    print(
        json.dumps(
            {
                "components": len(retained),
                "anchors": len(witnesses),
                "pools": len({scope["pool"] for scope in witnesses.values()}),
                "documents": len(documents),
            }
        ),
        flush=True,
    )


def prepared(path):
    plan_path, evidence_path = path / "plan.json", path / "evidence.json"
    plan = load(plan_path)
    if plan["reference"] != REFERENCE or plan["checkpoint"] != CHECKPOINT:
        raise ValueError("Unexpected shared-reference plan")
    if sha256_file(evidence_path) != plan["evidenceSha256"]:
        raise ValueError("The evidence does not match the prepared support plan")
    return plan, sha256_file(plan_path), load(evidence_path)


def reference_from_pool(encoded, aligned_ids, wanted, expected):
    reference = encoded.preprojection_reference
    if reference is None or reference.shape != (len(aligned_ids), expected):
        raise ValueError("Shared PRE/MID reference has unexpected dimensions")
    position = {anchor: i for i, anchor in enumerate(aligned_ids)}
    return reference[[position[anchor] for anchor in wanted]].numpy()


def capture(args):
    import torch

    raw, raw_sha = measurement_contract(args.measurements)
    plan, plan_sha, evidence = prepared(args.prepared)
    options = yaml.safe_load(args.config.read_text())["backend"]["options"]
    if (
        raw_sha != plan["measurementContract"]
        or sha256_file(args.config) != raw["configSha256"]
        or sha256_file(Path(options["checkpoint"]) / "manifest.json") != CHECKPOINT
    ):
        raise ValueError("The prepared supports and frozen encoder differ")
    if options["device"].startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    by_pool = defaultdict(list)
    for anchor, scope in evidence["witnesses"].items():
        by_pool[scope["pool"]].append(anchor)
    pending = {
        pool: sorted(ids)
        for pool, ids in by_pool.items()
        if not (args.output / "COMPLETE.json").exists()
    }
    if not pending:
        raise ValueError("Shared reference is already complete")
    args.output.mkdir(parents=True, exist_ok=True)
    header = {"planSha256": plan_sha, "measurementContract": raw_sha, "reference": REFERENCE}
    manifest = args.output / "plan.json"
    if manifest.exists():
        if load(manifest) != header:
            raise ValueError("Existing reference directory belongs to another support plan")
    else:
        write_json(manifest, header)
    gate_columns = [name.split("/") for name in raw["gateColumns"]]
    with TemporaryDirectory(prefix="aspd-shared-reference-") as folder:
        backend = V31CheckpointBackend(options, work_dir=Path(folder))
        try:
            if json.loads(json.dumps(backend.provenance())) != raw["backend"]:
                raise ValueError("The replay backend does not match the frozen measurements")
            dimensions = 2 * backend.config.architecture.cross_model_width
            for pool, wanted in sorted(pending.items()):
                shard = args.output / f"pool-{pool:06d}"
                if shard.is_dir():
                    meta = verified_json(shard, "meta.json")
                    if meta != {
                        "planSha256": plan_sha,
                        "anchors": wanted,
                        "dimensions": dimensions,
                    }:
                        raise ValueError("Existing pool shard disagrees with the support plan")
                    continue
                archived = args.measurements / f"pool-{pool:06d}"
                raw_meta = verified_json(archived, "meta.json")
                if raw_meta["contractSha256"] != raw_sha or raw_meta["pool"] != pool:
                    raise ValueError(f"Frozen pool contract mismatch at {pool}")
                with gzip.open(archived / "anchors.json.gz", "rt") as handle:
                    expected_anchors = json.load(handle)
                aligned = next(backend.aligned_pools(1, skip=pool))
                if (
                    anchor_rows(
                        aligned, backend.tokenizers, int(backend.config.data.sequence_length)
                    )
                    != expected_anchors
                ):
                    raise ValueError(f"Canonical/native alignment changed at pool {pool}")
                encoded = backend.encode_aligned_pool(aligned, capture_preprojection=True)
                actual = torch.stack(
                    [encoded.continuous[site][:, int(gate)].float() for site, gate in gate_columns],
                    dim=1,
                ).numpy()
                with np.load(archived / "values.npz", allow_pickle=False) as arrays:
                    expected = arrays["gates"]
                if not np.array_equal(actual > 0, expected > 0) or not np.allclose(
                    actual, expected, rtol=0.01, atol=0.002
                ):
                    raise ValueError(f"Natural gates differ from the anchor archive at pool {pool}")
                reference = reference_from_pool(encoded, aligned.anchor_ids, wanted, dimensions)
                unit_reference(reference)  # reject a zero or nonfinite vector before sealing
                with TemporaryDirectory(dir=args.output, prefix=".stage-") as stage:
                    temp = Path(stage) / "pool"
                    temp.mkdir()
                    np.save(temp / "reference.npy", reference, allow_pickle=False)
                    write_json(
                        temp / "meta.json",
                        {"planSha256": plan_sha, "anchors": wanted, "dimensions": dimensions},
                    )
                    write_json(
                        temp / "COMPLETE",
                        {"sha256": {p.name: sha256_file(p) for p in temp.iterdir()}},
                    )
                    temp.rename(shard)
                print(f"pool {pool}: {len(wanted)} measured shared reference rows", flush=True)
        finally:
            backend.close()
    write_json(
        args.output / "COMPLETE.json",
        {
            **header,
            "dimensions": dimensions,
            "siteOrder": ["pre", "mid"],
            "poolSeals": {
                str(pool): sha256_file(args.output / f"pool-{pool:06d}" / "COMPLETE")
                for pool in pending
            },
        },
    )


def load_reference(directory, plan_sha, evidence):
    seal_path = directory / "COMPLETE.json"
    seal = load(seal_path)
    if (
        seal["planSha256"] != plan_sha
        or seal["reference"] != REFERENCE
        or seal["siteOrder"] != ["pre", "mid"]
    ):
        raise ValueError("Reference rows were captured under another support plan")
    ids = sorted(evidence["witnesses"])
    positions = {anchor: i for i, anchor in enumerate(ids)}
    rows = np.empty((len(ids), seal["dimensions"]), dtype=np.float32)
    seen = set()
    for pool, digest in sorted(seal["poolSeals"].items()):
        shard = directory / f"pool-{int(pool):06d}"
        if sha256_file(shard / "COMPLETE") != digest:
            raise ValueError(f"Pool {pool} seal differs")
        meta = verified_json(shard, "meta.json")
        keys = meta["anchors"]
        if meta["planSha256"] != plan_sha or meta["dimensions"] != seal["dimensions"]:
            raise ValueError(f"Pool {pool} manifest differs")
        if len(set(keys)) != len(keys) or any(
            anchor in seen or evidence["witnesses"][anchor]["pool"] != int(pool) for anchor in keys
        ):
            raise ValueError(f"Duplicate or misaligned anchors in pool {pool}")
        block = np.load(shard / "reference.npy", allow_pickle=False)
        if block.shape != (len(keys), seal["dimensions"]):
            raise ValueError(f"Pool {pool} reference shape mismatch")
        rows[[positions[anchor] for anchor in keys]] = block
        seen.update(keys)
    if seen != set(ids):
        raise ValueError("Missing shared reference rows for top-K support")
    return positions, unit_reference(rows)


def distributions(examples, ids, positions, split):
    return [
        sn.Distribution(
            np.asarray([positions[item["anchor"]] for item in examples[key][split]]),
            np.asarray([item["strength"] for item in examples[key][split]]),
        )
        if examples[key][split]
        else None
        for key in ids
    ]


def cluster(args):
    started = time.monotonic()
    plan, plan_sha, evidence = prepared(args.prepared)
    if not 0 <= args.threshold <= 2:
        raise ValueError("Cosine OT cut must be in [0, 2]")
    if args.output.exists():
        raise FileExistsError(args.output)
    positions, reference = load_reference(args.reference, plan_sha, evidence)
    ids = [item["id"] for item in plan["components"]]
    discovery, heldout = (
        dissimilarities(distributions(evidence["examples"], ids, positions, split), reference)
        for split in SPLITS
    )
    tree, groups = complete_partition(discovery, args.threshold)
    records = partition_record("joint", ids, groups, discovery)
    for record, group in zip(records, groups, strict=True):
        record["diameter"] = float(discovery[np.ix_(group, group)].max())
    held_pairs = [
        heldout[i, j]
        for group in groups
        for i in group
        for j in group
        if i < j and np.isfinite(heldout[i, j])
    ]
    settings = {
        "planSha256": plan_sha,
        "referenceSha256": sha256_file(args.reference / "COMPLETE.json"),
        "threshold": args.threshold,
        "support": plan["topK"],
        "epsilon": None,
        "sourceSha256": sha256_file(Path(__file__)),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "contract.json", settings)
    run_id = sha256_file(args.output / "contract.json")
    write_array(args.output / "discovery-ot.npy", discovery)
    write_array(args.output / "heldout-ot.npy", heldout)
    write_json(args.output / "evidence.json", evidence)
    catalog = {
        "schemaVersion": 1,
        "mode": "shared-reference-joint",
        "dataset": "natural-validation",
        "runId": run_id,
        "checkpoint": CHECKPOINT,
        "measurementContract": plan["measurementContract"],
        "measurementProvenance": plan["measurementProvenance"],
        "settings": settings,
        "components": plan["components"],
        "counts": evidence["counts"],
        "reference": {
            "id": REFERENCE,
            "dimensions": reference.shape[1],
            "scope": "pooled PRE + MID",
            "normalization": "L2 for cosine only",
            "groundCost": "Cosine dissimilarity",
            "clustering": "complete linkage",
        },
        "cohort": {
            k: plan[k] for k in ("archiveSize", "cohortSize", "emptyDiscovery", "perModelBudget")
        },
        "models": {},
        "matches": [],
        "taylor": {"status": "unavailable", "witnesses": []},
        "exactAblation": {"status": "unmeasured"},
        "jointPartition": {
            "componentOrder": ids,
            "clusters": records,
            "tree": tree.tolist(),
            "distances": discovery.tolist(),
            "heldoutDistances": [
                [float(d) if np.isfinite(d) else None for d in row] for row in heldout
            ],
            "distanceCut": args.threshold,
            "evaluation": {
                "heldout": {
                    "mean": float(np.mean(held_pairs)) if held_pairs else None,
                    "pairs": len(held_pairs),
                }
            },
        },
        "execution": {"totalSeconds": time.monotonic() - started},
    }
    write_json(args.output / "catalog.json", catalog)
    write_json(
        args.output / "COMPLETE.json",
        {"runId": run_id, "sha256": {p.name: sha256_file(p) for p in args.output.iterdir()}},
    )
    print(
        json.dumps({"components": len(ids), "clusters": len(groups), "runId": run_id}), flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = modes.add_parser("prepare")
    prepare_parser.add_argument("--measurements", type=Path, required=True)
    prepare_parser.add_argument("--top-k", type=int, required=True)
    prepare_parser.add_argument("--per-model", type=int)
    prepare_parser.add_argument("--no-heatmaps", action="store_true")
    prepare_parser.set_defaults(run=prepare)
    capture_parser = modes.add_parser("capture")
    capture_parser.add_argument("--prepared", type=Path, required=True)
    capture_parser.add_argument("--measurements", type=Path, required=True)
    capture_parser.add_argument("--config", type=Path, required=True)
    capture_parser.set_defaults(run=capture)
    cluster_parser = modes.add_parser("cluster")
    cluster_parser.add_argument("--prepared", type=Path, required=True)
    cluster_parser.add_argument("--reference", type=Path, required=True)
    cluster_parser.add_argument("--threshold", type=float, required=True)
    cluster_parser.set_defaults(run=cluster)
    for sub in (prepare_parser, capture_parser, cluster_parser):
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
