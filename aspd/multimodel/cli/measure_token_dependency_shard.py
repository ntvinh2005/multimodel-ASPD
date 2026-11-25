"""Plan or resume exact-anchor component graph batches from verified activation archives."""

from __future__ import annotations

import argparse
import fcntl
import json
from collections import defaultdict, deque
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from aspd.multimodel.cli.benchmark_token_dependencies import component_requests, export_batch
from aspd.multimodel.cli.evaluate_aspd_v3_induction import (
    _anchor_metadata,
    _load_references,
)
from aspd.multimodel.cli.evaluate_aspd_v3_induction import (
    _load_preparation as load_induction_preparation,
)
from aspd.multimodel.cli.evaluate_aspd_v3_negation import (
    backend_options,
    jsonl,
    verify_evaluation,
)
from aspd.multimodel.cli.evaluate_aspd_v3_negation import (
    load_preparation as load_negation_preparation,
)
from aspd.multimodel.cli.measure_induction_query import PARENT_RULE, _supports, _verified_discovery
from aspd.multimodel.cli.measure_negation_comprehensive import (
    cache_contract,
    measure_missing,
    official_targets,
)
from aspd.multimodel.cli.prepare_aspd_v3_negation import read_contract
from aspd.multimodel.posthoc.artifacts import runtime_provenance, sha256_file, write_json
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend
from aspd.multimodel.v3.induction import MODEL_ORDER


def plan_batches(rows):
    """Deterministic round-robin over distinct documents for independent backward passes."""
    by_doc = defaultdict(deque)
    for row in rows:
        by_doc[row["example_id"]].append(row)
    pending = deque(by_doc)
    while pending:
        group = [pending.popleft() for _ in range(min(len(pending), 4))]
        batch = []
        for document in group:
            batch.append(by_doc[document].popleft())
            if by_doc[document]:
                pending.append(document)
        yield batch


def read_json(path):
    return json.loads(path.read_bytes())


def load_dataset(args, checkpoint):
    prepared_manifest = args.preparation / "manifest.json"
    activation_manifest = args.activations / "manifest.json"
    if args.dataset == "selection-frame":
        base = read_json(prepared_manifest)
        source = args.preparation / "examples.jsonl"
        activated = args.activations / "selection_frame_activations.json"
        facts = read_json(activation_manifest)
        if (base["checkpointSha256"] != checkpoint or sha256_file(source) != base["examplesSha256"]
                or read_json(args.activations / "COMPLETE")["manifestSha256"] != sha256_file(activation_manifest)
                or sha256_file(activated) != facts["selectionActivationsSha256"]):
            raise ValueError("Selection Frame preparation or activations changed")
        artifact = read_json(activated)
        if (artifact["checkpointSha256"] != checkpoint
                or artifact["preparationSha256"] != sha256_file(prepared_manifest)):
            raise ValueError("Selection Frame activations refer to other preparation")
        records, rows = list(jsonl(source)), [pos for ex in artifact["examples"] for pos in ex["positions"]]
        expected_backend = None
    elif args.dataset == "negation":
        facts = verify_evaluation(args.activations, args.split)
        records, _, _ = load_negation_preparation(args.preparation, args.config, args.split)
        if facts["preparation_sha256"] != sha256_file(prepared_manifest):
            raise ValueError("Negation preparation differs from original activation run")
        rows = list(jsonl(args.activations / "anchor_activations.jsonl"))
        expected_backend = facts["backend"]
    else:
        facts = _verified_discovery(args.activations) if args.split == "discovery" else read_json(activation_manifest)
        source = args.activations / "anchor_activations.jsonl"
        records, _ = load_induction_preparation(args.preparation, args.split)
        if (facts["heldout_opened"] or facts["split"] != args.split
                or read_json(args.activations / "COMPLETE")["manifest_sha256"] != sha256_file(activation_manifest)
                or facts["preparation_manifest_sha256"] != sha256_file(prepared_manifest)
                or sha256_file(source) != facts["artifacts"][source.name]):
            raise ValueError("Induction preparation or activations changed")
        rows = list(jsonl(source))
        expected_backend = facts["backend"]
    if expected_backend and sha256_file(Path(expected_backend["checkpoint"]) / "manifest.json") != checkpoint:
        raise ValueError("activation checkpoint differs from the approved checkpoint")
    metadata = _anchor_metadata(records)
    by_anchor = {
        anchor["anchor_id"]: doc for doc in records for anchor in doc["alignment"]["anchors"]
    }
    if len(rows) != len(metadata) or {row["anchor_id"] for row in rows} != metadata.keys():
        raise ValueError("prepared and archived anchor inventories differ")
    for row in rows:
        if any(row[key] != value for key, value in metadata[row["anchor_id"]].items()):
            raise ValueError("archived position differs from canonical token join")
        for site, support in _supports(row).items():
            if len(support) != len(row[site]["active_gate_ids"]) or any(
                not 0 <= feature < 8192 or not 0 < value < float("inf") for feature, value in support.items()
            ):
                raise ValueError("invalid archived natural BatchTopK support")
    return rows, by_anchor, expected_backend, base["models"] if args.dataset == "selection-frame" else None


def validate_cached(batch_id, rows, path, marker, source_hashes):
    if not marker.exists():
        return False
    entry = read_json(marker)
    if (entry["batch"] != batch_id or entry["anchors"] != [row["anchor_id"] for row in rows]
            or entry["contract"] != source_hashes or sha256_file(path) != entry["sha256"]):
        raise ValueError(f"cache integrity check failed: {marker}")
    return True


def benchmark_covered(root, checkpoint, activation_sha):
    manifest = root / "manifest.json"
    payload = read_json(manifest)
    if (payload["checkpointSha256"] != checkpoint
            or payload["discoveryManifestSha256"] != activation_sha
            or read_json(root / "COMPLETE")["manifestSha256"] != sha256_file(manifest)):
        raise ValueError("cached benchmark does not match this activation archive")
    anchors = set()
    for batch in payload["batches"]:
        if (batch["componentTargets"] != batch["measured"]
                or sha256_file(root / batch["path"]) != batch["sha256"]):
            raise ValueError("incomplete or corrupt benchmark batch")
        anchors.update(batch["anchors"])
    return anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("negation", "induction", "selection-frame"), required=True)
    parser.add_argument("--split", choices=("smoke", "discovery"), required=True)
    for name in ("config", "preparation", "activations", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batches", nargs=2, type=int, metavar=("START", "STOP"))
    parser.add_argument("--reuse-benchmark", type=Path)
    args = parser.parse_args()
    if args.dataset == "selection-frame" and args.split != "discovery":
        parser.error("Selection Frame contains a Discovery split")
    _, reference = read_contract(args.config)
    checkpoint = reference["checkpoint_manifest_sha256"]
    rows, by_anchor, expected_backend, source_models = load_dataset(args, checkpoint)
    activation_sha = sha256_file(args.activations / "manifest.json")
    cached = benchmark_covered(args.reuse_benchmark, checkpoint, activation_sha) if args.reuse_benchmark else set()
    if not cached <= by_anchor.keys():
        raise ValueError("cached benchmark contains unknown anchors")
    rows = [row for row in rows if row["anchor_id"] not in cached]
    batches = list(plan_batches(rows))
    info = {"batches": len(batches), "anchors": len(rows),
            "dataset": args.dataset, "split": args.split, "batchSize": 4, "reusedAnchors": len(cached)}
    print(info, flush=True)
    if args.batches is None:
        return
    start, stop = args.batches
    if not 0 <= start < stop <= len(batches):
        parser.error("requested batch interval is outside the deterministic plan")
    ref = Path(reference["components"]["reference_posthoc_root"])
    provenance = cache_contract({
        "checkpointSha256": checkpoint,
        "activationManifestSha256": activation_sha,
        "reusedBenchmarkSha256": sha256_file(args.reuse_benchmark / "manifest.json") if args.reuse_benchmark else None,
        "preparationManifestSha256": sha256_file(args.preparation / "manifest.json"),
        "referenceImportanceSha256": sha256_file(ref / "importance/COMPLETE"),
        "referenceRepresentativesSha256": sha256_file(ref / "representatives/COMPLETE"),
        "configSha256": sha256_file(args.config),
    })
    provenance["sourceSha256"].update({
        "cli/measure_token_dependency_shard.py": sha256_file(Path(__file__)),
        "cli/benchmark_token_dependencies.py": sha256_file(Path(__file__).with_name("benchmark_token_dependencies.py")),
    })
    args.output.mkdir(parents=True, exist_ok=True)
    missing = []
    for batch_id in range(start, stop):
        batch_path = args.output / f"batch-{batch_id:06d}.json.gz"
        marker = args.output / f"batch-{batch_id:06d}.complete.json"
        if not validate_cached(batch_id, batches[batch_id], batch_path, marker, provenance):
            missing.append((batch_id, batch_path, marker))
    if not missing:
        print("All requested batches verified in cache", flush=True)
        return
    importance, reps = _load_references(ref)
    runtime = runtime_provenance()
    stream_root = args.output / f"stream-{start}-{stop}-{runtime['slurm_job_id']}"
    backend = V31CheckpointBackend(backend_options(reference), work_dir=stream_root)
    try:
        native = backend.provenance()
        if expected_backend and native != expected_backend:
            raise ValueError("runtime source revisions differ from the original activation evaluation")
        if source_models:
            for key, model in source_models.items():
                source = native["models"][key]
                if (source["model_id"], source["revision"], source["tokenizer_revision"]) != (
                    model["id"], model["revision"], model["tokenizer_revision"],
                ):
                    raise ValueError("runtime source revisions differ from Selection Frame preparation")
        matrices = [matrix for model in MODEL_ORDER for matrix in backend.factors(model)]
        for batch_id, path, marker in missing:
            with marker.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                rows = batches[batch_id]
                if validate_cached(batch_id, rows, path, marker, provenance):
                    continue
                requests = component_requests(rows, matrices)
                results = measure_missing(backend, by_anchor, rows, requests)
                record = {
                    **provenance, "sameAnchorOnly": True, "runtime": runtime,
                    "parentRule": {"k": PARENT_RULE.k_par, "tau": PARENT_RULE.tau},
                }
                with TemporaryDirectory(dir=args.output, prefix=f"batch-{batch_id:06d}-") as stage:
                    stage_path = Path(stage) / path.name
                    stage_marker = Path(stage) / marker.name
                    export_batch(stage_path, rows, results, official_targets(rows, reps), importance, record)
                    write_json(stage_marker, {"batch": batch_id, "sha256": sha256_file(stage_path),
                                              "anchors": [row["anchor_id"] for row in rows], "contract": provenance})
                    stage_path.replace(path)
                    stage_marker.replace(marker)
                print(f"Completed batch {batch_id}: {len(results)} component targets", flush=True)
    finally:
        backend.close()
    print({"gpuPeakAllocatedBytes": torch.cuda.max_memory_allocated()}, flush=True)


if __name__ == "__main__":
    main()
