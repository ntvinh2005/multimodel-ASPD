"""Read-only held-out gate for the three-model Granite normalized-target pilot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import yaml

from aspd.multimodel.cli.evaluate_aspd_v3_500m import (
    _aggregate_ratio,
    _capture,
    _finite_state_report,
    _local_evaluation,
    _parameter_signature,
    _shared_digest,
)
from aspd.multimodel.v3.data import HashSplitPolicy
from aspd.multimodel.v3.diagnostic import sha256_file
from aspd.multimodel.v3.diagnostic_stream import ValidationAlignedAnchorStream
from aspd.multimodel.v3.granite_normalized_config import (
    MODEL_ORDER_GRANITE,
    load_granite_normalized_target_config,
)
from aspd.multimodel.v3.hooks import assert_frozen_backbone, logical_weight_bias
from aspd.multimodel.v3.model_owned import ModelOwnedMain2B
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, load_pinned_pile_dataset
from aspd.multimodel.v3.registry import build_model_registry_2b


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("checkpoint-smoke", "heldout"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-smoke-report", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _broadcast(value: Any) -> Any:
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "status": "RESEARCHER_AUTHORIZED_HELDOUT_GATE",
        "authorized_by": "researcher",
        "authorized_on": "2026-11-07",
        "experiment_family": "aspd_v3_granite_normalized_target_v1",
        "split": "validation",
        "split_unit": "document",
        "accepted_anchors": 100_352,
        "pool_anchors": 2_048,
        "pool_count": 49,
        "bootstrap_unit": "source_document",
        "bootstrap_replicates": 1_000,
        "bootstrap_seed": 0,
        "jaccard_top_features": 256,
        "jaccard_min_fires": 4,
        "checkpoint_steps": 128,
        "arms": ["baseline", "normalized_target"],
        "model_order": list(MODEL_ORDER_GRANITE),
        "gpu_count": 3,
        "gpu_type": "b200",
        "read_only": True,
        "training_authorized_by_this_contract": False,
    }
    if payload != expected:
        raise ValueError("Granite held-out evaluation contract changed from the approved gate")
    if payload["accepted_anchors"] != payload["pool_anchors"] * payload["pool_count"]:
        raise ValueError("held-out anchor budget is not an integer number of pools")
    return payload


def _validate_inputs(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    required = (
        args.config,
        args.contract,
        args.run_dir / "run_manifest.json",
        args.checkpoint / "manifest.json",
        args.checkpoint / "COMPLETE",
        Path(config.runtime.dataset_manifest),
        Path(config.runtime.alignment_manifest),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Granite evaluation inputs missing: {missing}")
    run = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((args.checkpoint / "manifest.json").read_text(encoding="utf-8"))
    marker = json.loads((args.checkpoint / "COMPLETE").read_text(encoding="utf-8"))
    dataset = json.loads(Path(config.runtime.dataset_manifest).read_text(encoding="utf-8"))
    alignment = json.loads(Path(config.runtime.alignment_manifest).read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(args.checkpoint / "manifest.json")
    expected = {
        "run.config_fingerprint": config.fingerprint(),
        "run.config_file_sha256": sha256_file(args.config),
        "run.dataset_manifest_sha256": dataset["manifest_sha256"],
        "run.alignment_manifest_sha256": alignment["manifest_sha256"],
        "run.model_order": list(MODEL_ORDER_GRANITE),
        "run.logical_target_count": 168,
        "checkpoint.status": "COMPLETE",
        "checkpoint.step": 128,
        "checkpoint.world_size": 3,
        "checkpoint.config_fingerprint": config.fingerprint(),
        "checkpoint.config_file_sha256": sha256_file(args.config),
        "checkpoint.dataset_manifest_sha256": dataset["manifest_sha256"],
        "marker.manifest_sha256": checkpoint_sha,
    }
    observed = {
        "run.config_fingerprint": run.get("config_fingerprint"),
        "run.config_file_sha256": run.get("config_file_sha256"),
        "run.dataset_manifest_sha256": run.get("dataset_manifest_sha256"),
        "run.alignment_manifest_sha256": run.get("alignment_manifest_sha256"),
        "run.model_order": run.get("model_order"),
        "run.logical_target_count": run.get("logical_target_count"),
        "checkpoint.status": checkpoint.get("status"),
        "checkpoint.step": checkpoint.get("step"),
        "checkpoint.world_size": checkpoint.get("world_size"),
        "checkpoint.config_fingerprint": checkpoint.get("config_fingerprint"),
        "checkpoint.config_file_sha256": checkpoint.get("config_file_sha256"),
        "checkpoint.dataset_manifest_sha256": checkpoint.get("dataset_manifest_sha256"),
        "marker.manifest_sha256": marker.get("manifest_sha256"),
    }
    mismatch = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatch:
        raise ValueError(f"Granite pilot provenance mismatch: {mismatch}")
    return {
        "status": "PASS",
        "arm": config.arm,
        "training_git_sha": run.get("git_sha"),
        "config_sha256": sha256_file(args.config),
        "config_fingerprint": config.fingerprint(),
        "checkpoint_manifest_sha256": checkpoint_sha,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "alignment_manifest_sha256": alignment["manifest_sha256"],
    }


def _space_summary(report: dict[str, Any], prefix: str) -> dict[str, Any]:
    model = report["model"]
    result = {}
    for site in ("pre", "mid"):
        prefixes = []
        for prefix_index in range(6):
            rows = [
                value
                for key, value in report["metrics"].items()
                if key.startswith(f"{prefix}/{model}/{site}/")
                and key.endswith(f"/prefix_{prefix_index}")
            ]
            if len(rows) != 8:
                raise ValueError(
                    f"expected eight {prefix} metrics for {model}/{site}/prefix_{prefix_index}"
                )
            prefixes.append(_aggregate_ratio(rows))
        result[site] = {
            "prefixes": prefixes,
            "matryoshka_mean_fvu": sum(row["fvu"] for row in prefixes) / 6,
            "full_reconstruction": prefixes[-1],
        }
    return result


def _render(reports: list[dict[str, Any]], output: Path, manifest: dict[str, Any]) -> None:
    arm = manifest["arm"]
    activation_summary: dict[str, Any] = {}
    matrix_rows: list[dict[str, Any]] = []
    all_metrics: dict[str, Any] = {}
    target_norms: dict[str, Any] = {}
    for report in reports:
        model = report["model"]
        all_metrics.update(report["metrics"])
        target_norms.update(report["activation_target_norms"])
        if arm == "baseline":
            activation_summary[model] = {"raw": _space_summary(report, "activation")}
        else:
            activation_summary[model] = {
                "normalized": _space_summary(report, "activation_normalized"),
                "original_oracle": _space_summary(report, "activation_original_oracle"),
            }
        for key, value in report["metrics"].items():
            if key.startswith("internal/"):
                target = key.removeprefix("internal/")
                _owner, layer, matrix = target.split(".")
                matrix_rows.append({"model": model, "layer": layer, "matrix": matrix, **value})
    _write_json(
        output / "heldout_reconstruction.json",
        {
            "schema_version": 1,
            "status": "HELDOUT_RECONSTRUCTION_COMPLETE",
            "arm": arm,
            "activation_target_normalization": manifest["activation_target_normalization"],
            "original_space_uses_oracle_target_norm": arm == "normalized_target",
            "activation_summary": activation_summary,
            "activation_target_norms": target_norms,
            "metrics": all_metrics,
        },
    )
    with (output / "per_matrix_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)
    dictionary = {report["model"]: report["dictionary"] for report in reports}
    _write_json(output / "dictionary_health.json", dictionary)
    feature_rows = [row for report in reports for row in report["feature_rows"]]
    with (output / "feature_activity.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    _write_json(output / "experiment_manifest.json", manifest)
    _write_json(
        output / "evaluation_summary.json",
        {
            "schema_version": 1,
            "status": "SCIENTIFIC_REVIEW_REQUIRED",
            "arm": arm,
            "activation_summary": activation_summary,
            "dictionary": dictionary,
            "comparison_warning": (
                "Treatment original-space FVU uses the target-derived oracle norm; raw baseline "
                "FVU and normalized-target FVU are not the same estimand."
            ),
        },
    )


def main() -> None:
    args = _args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    config = load_granite_normalized_target_config(args.config)
    contract = _load_contract(args.contract)
    model_order = tuple(config.architecture.model_order)
    if world != 3 or model_order != MODEL_ORDER_GRANITE:
        raise RuntimeError("Granite evaluation requires gemma/smol/granite on exactly three ranks")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    started = time.perf_counter()
    provenance = _validate_inputs(args, config)
    if args.mode == "heldout":
        if args.checkpoint_smoke_report is None or not args.checkpoint_smoke_report.is_file():
            raise FileNotFoundError("held-out evaluation requires checkpoint smoke evidence")
        smoke = json.loads(args.checkpoint_smoke_report.read_text(encoding="utf-8"))
        if smoke.get("status") != "E0_CHECKPOINT_RESTORE_PASS":
            raise ValueError("held-out evaluation blocked by checkpoint smoke")
        if smoke.get("checkpoint_manifest_sha256") != provenance["checkpoint_manifest_sha256"]:
            raise ValueError("checkpoint smoke refers to a different checkpoint")
    if rank == 0:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite evaluation output: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
    dist.barrier()

    registry = build_model_registry_2b(config)
    local_key = model_order[rank]
    tokenizer = load_v3_tokenizer(registry[local_key], local_files_only=args.local_files_only)
    loaded = load_frozen_v3_model(
        registry[local_key], device, local_files_only=args.local_files_only
    )
    assert_frozen_backbone(loaded.forward_model)
    source_before = _parameter_signature(loaded.forward_model)
    native_biases = {
        target.key: logical_weight_bias(loaded.forward_model, target)[1]
        for target in registry[local_key].targets()
    }
    model = ModelOwnedMain2B(
        config,
        registry,
        local_key,
        native_biases=native_biases,
        device=device,
        dtype=torch.float32,
        sparse_backend="dense",
    )
    manager = ModelOwnedCheckpoint(args.checkpoint.parent, keep_complete=2)
    runtime_state, checkpoint_manifest = manager.load_model_only(
        path=args.checkpoint,
        model=model,
        config_fingerprint=config.fingerprint(),
        config_file_sha256=sha256_file(args.config),
        dataset_manifest_sha256=provenance["dataset_manifest_sha256"],
        expected_model_key=local_key,
    )
    model.eval().requires_grad_(False)
    finite = _finite_state_report(model)
    digest = _shared_digest(model)
    digests: list[str | None] = [None] * world
    dist.all_gather_object(digests, digest)
    if len(set(digests)) != 1:
        raise ValueError("replicated shared encoders differ across model owners")

    stream = None
    if rank == 0:
        alignment_tokenizers = {
            key: tokenizer
            if key == local_key
            else load_v3_tokenizer(registry[key], local_files_only=args.local_files_only)
            for key in model_order
        }
        dataset = load_pinned_pile_dataset(config.data.dataset_id, config.data.dataset_revision)
        stream = ValidationAlignedAnchorStream(
            dataset=dataset,
            dataset_id=config.data.dataset_id,
            dataset_revision=config.data.dataset_revision,
            tokenizers=alignment_tokenizers,
            tokenizer_revisions={key: registry[key].tokenizer_revision for key in model_order},
            split_policy=HashSplitPolicy(
                modulus=config.data.split_modulus,
                validation_buckets=frozenset(config.data.validation_buckets),
                test_buckets=frozenset(config.data.test_buckets),
                seed=config.data.split_seed,
            ),
            dedup_path=args.output_dir / "validation_stream_dedup.sqlite",
            sequence_length=config.data.sequence_length,
            model_order=model_order,
        )

    def next_pool() -> AlignedAnchorPool:
        payload = None if rank else stream.next_pool(pool_anchors=contract["pool_anchors"]).to_payload()
        return AlignedAnchorPool.from_payload(_broadcast(payload))

    if args.mode == "checkpoint-smoke":
        pool = next_pool()
        capture = _capture(
            pool,
            local_key=local_key,
            tokenizer=tokenizer,
            loaded=loaded,
            registry=registry,
            device=device,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            first = model.encode(capture, update_statistics=False)
            second = model.encode(capture, update_statistics=False)
        for site in ("pre", "mid"):
            torch.testing.assert_close(first[1][site].values, second[1][site].values, rtol=0, atol=0)
        local = {
            "rank": rank,
            "model": local_key,
            "finite_state": finite,
            "shared_encoder_sha256": digest,
        }
        reports: list[Any] = [None] * world
        dist.all_gather_object(reports, local)
        if _parameter_signature(loaded.forward_model) != source_before:
            raise RuntimeError("frozen source model changed during checkpoint smoke")
        if rank == 0:
            _write_json(
                args.output_dir / "checkpoint_smoke.json",
                {
                    "schema_version": 1,
                    "status": "E0_CHECKPOINT_RESTORE_PASS",
                    "arm": config.arm,
                    "checkpoint_manifest_sha256": provenance["checkpoint_manifest_sha256"],
                    "checkpoint_step": checkpoint_manifest["step"],
                    "model_order": list(model_order),
                    "source_weights_unchanged": "PASS",
                    "deterministic_repeat": "PASS",
                    "reports": reports,
                },
            )
    else:
        diagnostic_contract = SimpleNamespace(
            seed=contract["bootstrap_seed"],
            bootstrap_replicates=contract["bootstrap_replicates"],
            jaccard_top_features=contract["jaccard_top_features"],
            jaccard_min_fires=contract["jaccard_min_fires"],
        )
        report = _local_evaluation(
            pools=contract["pool_count"],
            model=model,
            loaded=loaded,
            tokenizer=tokenizer,
            registry=registry,
            local_key=local_key,
            device=device,
            next_pool=next_pool,
            contract=diagnostic_contract,
            activation_target_normalization=config.objective.activation_target_normalization,
            activation_target_epsilon=config.objective.activation_target_epsilon,
        )
        if report["accepted_anchors"] != contract["accepted_anchors"]:
            raise ValueError("held-out stream returned a different accepted-anchor budget")
        if _parameter_signature(loaded.forward_model) != source_before:
            raise RuntimeError("frozen source model changed during held-out evaluation")
        _write_json(args.output_dir / f"rank-{rank:05d}.json", report)
        dist.barrier()
        if rank == 0:
            reports = [
                json.loads((args.output_dir / f"rank-{value:05d}.json").read_text(encoding="utf-8"))
                for value in range(world)
            ]
            manifest = {
                "schema_version": 1,
                "status": "READ_ONLY_GRANITE_NORMALIZED_TARGET_DIAGNOSTIC_COMPLETE",
                "arm": config.arm,
                "activation_target_normalization": config.objective.activation_target_normalization,
                "original_space_uses_oracle_target_norm": config.arm == "normalized_target",
                "diagnostic_git_sha": _git_sha(),
                "training_git_sha": provenance["training_git_sha"],
                "config_sha256": provenance["config_sha256"],
                "config_fingerprint": config.fingerprint(),
                "evaluation_contract_sha256": sha256_file(args.contract),
                "checkpoint_manifest_sha256": provenance["checkpoint_manifest_sha256"],
                "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
                "alignment_manifest_sha256": provenance["alignment_manifest_sha256"],
                "model_order": list(model_order),
                "accepted_anchors": contract["accepted_anchors"],
                "bootstrap_unit": contract["bootstrap_unit"],
                "split": contract["split"],
                "runtime_state_step": int(runtime_state["completed_steps"]),
                "software": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _render(reports, args.output_dir, manifest)
    if stream is not None:
        stream.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
