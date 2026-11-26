"""Plan or resume complete same-anchor graphs on the archived Natural Validation test pilot."""

import argparse
import gzip
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from aspd.multimodel.cli.benchmark_token_dependencies import (
    complete_token_graph,
    component_requests,
    export_batch,
)
from aspd.multimodel.cli.evaluate_aspd_v3_induction import _load_references
from aspd.multimodel.cli.evaluate_aspd_v3_negation import backend_options
from aspd.multimodel.cli.measure_negation_comprehensive import (
    cache_contract,
    measure_frozen_inputs,
    official_targets,
)
from aspd.multimodel.cli.measure_token_dependency_shard import plan_batches
from aspd.multimodel.cli.replay_natural_token_components import (
    load_test_archive,
    model_inputs,
    replay_rows,
    verify_test_support,
)
from aspd.multimodel.posthoc.artifacts import runtime_provenance, sha256_file, write_json
from aspd.multimodel.posthoc.ids import MatrixId
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend, _pad_id
from aspd.multimodel.v3.induction import MODEL_ORDER


def graph_rows(rows):
    for document, point in rows:
        row = {
            "anchor_id": point["anchor"],
            "example_id": document,
            "pool_id": str(point["pool"]),
            "character_boundary": point["characterBoundary"],
            "model_positions": point["positions"],
        }
        for site in ("pre", "mid"):
            support = sorted(
                (int(key.split(":")[1]), value)
                for key, value in point["gates"].items()
                if key.startswith(site.upper() + ":")
            )
            row[site] = {
                "active_gate_ids": [gate for gate, _ in support],
                "values": [value for _, value in support],
            }
        yield row


def load_plan(args):
    pilot, native, archive, reference, registry, tokenizers = load_test_archive(args)
    inventory = {
        point["anchor"]
        for window in pilot["examples"].values()
        for point in window["points"].values()
    }
    persisted_order = sorted(inventory)
    if args.near_window:
        window = pilot["examples"][args.near_window]
        center = window["center"]
        indices = sorted(
            range(
                max(0, center - args.radius), min(len(window["tokens"]), center + args.radius + 1)
            ),
            key=lambda position: (abs(position - center), position),
        )
        persisted_order = [
            window["points"][str(i)]["anchor"] for i in indices if str(i) in window["points"]
        ]
        if not persisted_order or persisted_order[0] != args.near_window:
            raise ValueError("Window center must have an exact archived anchor")
    rows = replay_rows(pilot, native, persisted_order, tokenizers)
    verify_test_support(args.archive, archive, rows)
    by_anchor = {point["anchor"]: (document, point) for document, point in rows}
    matrices = tuple(
        MatrixId(model, target.layer_index, target.logical_name)
        for model in MODEL_ORDER
        for target in registry[model].targets()
    )
    batches = list(plan_batches(list(graph_rows(rows))))
    cfg = Path(reference["components"]["reference_posthoc_root"])
    contract = cache_contract(
        {
            "checkpointSha256": pilot["checkpoint"],
            "dataset": pilot["dataset"],
            "split": pilot["split"],
            "archiveManifestSha256": pilot["provenance"]["archiveManifestSha256"],
            "pilotSha256": sha256_file(args.pilot),
            "replayInputsSha256": sha256_file(args.replay_inputs),
            "configSha256": sha256_file(args.config),
            "referenceImportanceSha256": sha256_file(cfg / "importance/COMPLETE"),
            "referenceRepresentativesSha256": sha256_file(cfg / "representatives/COMPLETE"),
            "sourceModels": archive["source_model_provenance"],
            "sameAnchorOnly": True,
        }
    )
    for path in (
        Path(__file__),
        Path(__file__).with_name("replay_natural_token_components.py"),
        Path(__file__).with_name("benchmark_token_dependencies.py"),
        Path(__file__).with_name("measure_token_dependency_shard.py"),
    ):
        contract["sourceSha256"][path.name] = sha256_file(path)
    jobs = []
    matrix_counts = defaultdict(int)
    for matrix in matrices:
        matrix_counts[matrix.model, matrix.site] += 1
    for batch in batches:
        per_model = {
            model: sum(
                matrix_counts[model, site] * len(row[site]["active_gate_ids"])
                for row in batch
                for site in ("pre", "mid")
            )
            for model in MODEL_ORDER
        }
        jobs.append({"anchors": [row["anchor_id"] for row in batch], "targets": per_model})
    plan = {
        "schemaVersion": 1,
        "contract": contract,
        "batches": jobs,
        "anchorCount": len(persisted_order),
        "pilotAnchorCount": len(inventory),
        "componentTargets": {
            model: sum(job["targets"][model] for job in jobs) for model in MODEL_ORDER
        },
        "unalignedPositions": sum(
            len(window["tokens"]) - len(window["points"]) for window in pilot["examples"].values()
        ),
    }
    return plan, batches, by_anchor, matrices, native, reference


def index_coverage(output, plan):
    graph_refs = {}
    targets, empty_targets = 0, 0
    for batch_id, batch in enumerate(plan["batches"]):
        root = output / f"batch-{batch_id:06d}"
        if not root.is_dir():
            continue
        marker = json.loads((root / "COMPLETE").read_bytes())
        if (
            marker["batch"] != batch_id
            or list(marker["graphs"]) != sorted(batch["anchors"])
            or marker["componentTargets"] != sum(batch["targets"].values())
        ):
            raise ValueError(f"Batch manifest differs from plan: {root}")
        if sha256_file(root / "raw.json.gz") != marker["rawSha256"]:
            raise ValueError(f"Batch raw artifact changed: {root}")
        for anchor, digest in marker["graphs"].items():
            file = root / f"{anchor}.json.gz"
            if sha256_file(file) != digest:
                raise ValueError(f"Graph changed: {file}")
            graph_refs[anchor] = {"path": f"{root.name}/{file.name}", "sha256": digest}
        targets += marker["componentTargets"]
        empty_targets += marker["emptyParentTargets"]
    return {
        "schemaVersion": 1,
        **{
            key: plan["contract"][key]
            for key in (
                "checkpointSha256",
                "dataset",
                "split",
                "archiveManifestSha256",
                "pilotSha256",
            )
        },
        "graphs": graph_refs,
        "anchorsMeasured": len(graph_refs),
        "anchorsExpected": plan["anchorCount"],
        "pilotAnchors": plan["pilotAnchorCount"],
        "unalignedPositions": plan["unalignedPositions"],
        "componentTargetsMeasured": targets,
        "componentTargetsExpected": sum(plan["componentTargets"].values()),
        "emptyParentTargets": empty_targets,
    }


def atomic_index(output, index):
    with TemporaryDirectory(dir=output, prefix=".index-") as tmp:
        path = Path(tmp) / "index.json"
        write_json(path, index)
        path.replace(output / path.name)


def export_measured_batch(output, batch_id, rows, results, plan, importance, reps, started):
    checkpoint = plan["contract"]["checkpointSha256"]
    by_anchor = defaultdict(dict)
    for (anchor, component), row in results.items():
        by_anchor[anchor][component.key] = row
    scales = {component.matrix.key: importance[component.matrix].sigma for _, component in results}
    with TemporaryDirectory(dir=output, prefix=".stage-") as tmp:
        staged = Path(tmp) / "batch"
        staged.mkdir()
        raw_path = staged / "raw.json.gz"
        export_batch(
            raw_path,
            rows,
            results,
            official_targets(rows, reps),
            importance,
            {**plan["contract"], "runtime": runtime_provenance()},
        )
        graphs = {}
        for row in rows:
            anchor = row["anchor_id"]
            asset = complete_token_graph(
                row, by_anchor[anchor], scales, checkpoint, importance, reps
            )
            path = staged / f"{anchor}.json.gz"
            with (
                path.open("xb") as raw,
                gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=3) as stream,
            ):
                stream.write(json.dumps(asset, separators=(",", ":"), allow_nan=False).encode())
            graphs[anchor] = sha256_file(path)
        bytes_written = sum(file.stat().st_size for file in staged.iterdir())
        marker = {
            "batch": batch_id,
            "graphs": graphs,
            "rawSha256": sha256_file(raw_path),
            "componentTargets": len(results),
            "emptyParentTargets": sum(not value[2] for value in results.values()),
            "bytes": bytes_written,
            "seconds": time.monotonic() - started,
            "gpuPeakBytes": torch.cuda.max_memory_allocated(),
        }
        write_json(staged / "COMPLETE", marker)
        staged.rename(output / f"batch-{batch_id:06d}")
    return marker


def run(args, plan, batches, by_anchor, matrices, native, reference):
    launch = time.monotonic()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_bytes()) != plan:
            raise ValueError("Cached graph plan does not match the exact inputs")
    else:
        write_json(plan_path, plan)
    coverage = index_coverage(output, plan)
    atomic_index(output, coverage)
    completed = [
        json.loads((path / "COMPLETE").read_bytes()) for path in sorted(output.glob("batch-*"))
    ]
    done = {row["batch"] for row in completed}
    if (args.benchmark_only and set(range(min(2, len(batches)))) <= done) or len(done) == len(
        batches
    ):
        return
    importance, reps = _load_references(Path(reference["components"]["reference_posthoc_root"]))
    backend = V31CheckpointBackend({**backend_options(reference), "split": "test"}, work_dir=output)
    try:
        actual_matrices = {matrix for model in MODEL_ORDER for matrix in backend.factors(model)}
        if (
            actual_matrices != set(matrices)
            or backend.model_keys != MODEL_ORDER
            or backend.n_features != 8192
        ):
            raise ValueError("Backend component inventory differs from the frozen plan")
        for batch_id, rows in enumerate(batches):
            if batch_id in done:
                continue
            if args.benchmark_only and batch_id >= 2:
                break
            if len(completed) >= 2:
                measured = sum(row["componentTargets"] for row in completed)
                pending = coverage["componentTargetsExpected"] - measured
                seconds = sum(row["seconds"] for row in completed)
                size = sum(row["bytes"] for row in completed)
                # Budget includes earlier invocations' recorded measurement time and this startup.
                this_startup = time.monotonic() - launch
                this_startup -= sum(row["seconds"] for row in completed if row["batch"] not in done)
                forecast = {
                    "totalHours": (max(0, this_startup) + seconds * (1 + pending / measured))
                    / 3600,
                    "totalBytes": int(size * (1 + pending / measured)),
                }
                print(json.dumps(forecast), flush=True)
                if (
                    forecast["totalHours"] > args.max_gpu_hours
                    or forecast["totalBytes"] > args.max_output_bytes
                ):
                    print("Stopped before next batch: projected budget exceeded", flush=True)
                    break
            started = time.monotonic()
            scope = [by_anchor[row["anchor_id"]] for row in rows]
            inputs = {
                model: model_inputs(
                    native, scope, model, _pad_id(backend.tokenizers[model]), torch.device("cpu")
                )
                for model in MODEL_ORDER
            }
            requests = component_requests(rows, matrices)
            marker = export_measured_batch(
                output,
                batch_id,
                rows,
                measure_frozen_inputs(backend, inputs, rows, requests),
                plan,
                importance,
                reps,
                started,
            )
            completed.append(marker)
            for anchor, digest in marker["graphs"].items():
                coverage["graphs"][anchor] = {
                    "path": f"batch-{batch_id:06d}/{anchor}.json.gz",
                    "sha256": digest,
                }
            coverage["anchorsMeasured"] += len(rows)
            coverage["componentTargetsMeasured"] += marker["componentTargets"]
            coverage["emptyParentTargets"] += marker["emptyParentTargets"]
            atomic_index(output, coverage)
            print(
                json.dumps({key: value for key, value in coverage.items() if key != "graphs"}),
                flush=True,
            )
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "config", "pilot", "replay-inputs", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument(
        "--near-window",
        help="Restrict the planned anchors to the center and nearby aligned positions",
    )
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--max-gpu-hours", type=float, default=6)
    parser.add_argument("--max-output-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args()
    if not 0 <= args.radius <= 10:
        parser.error("Use a radius between zero and ten tokens")
    plan, *inputs = load_plan(args)
    print(
        json.dumps(
            {key: value for key, value in plan.items() if key not in {"contract", "batches"}}
            | {
                "batchCount": len(inputs[0]),
                "benchmarkAnchors": sum(map(len, inputs[0][:2])),
            }
        ),
        flush=True,
    )
    if args.execute:
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            parser.error("Execution requires an authorized Slurm GPU allocation")
        run(args, plan, *inputs)


if __name__ == "__main__":
    main()
