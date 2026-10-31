"""Three-rank FSDP trainer for the frozen ASPD v3 main-v1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from aspd.multimodel.v3.config import load_v3_config_bundle
from aspd.multimodel.v3.data import (
    ExposureLedgerReceipt,
    HashSplitPolicy,
    SQLiteExposureLedger,
    write_json_immutable,
)
from aspd.multimodel.v3.distributed_checkpoint import MainV1DistributedCheckpoint
from aspd.multimodel.v3.main_config import load_main_v1_config
from aspd.multimodel.v3.main_model import (
    MainV1CaptureBatch,
    MainV1TrainableModel,
    build_adam_optimizer,
    finite_gradient_report,
    fully_shard_main_v1,
    trainable_parameter_count,
)
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.production_stream import (
    AlignedAnchorPool,
    PileAlignedAnchorStream,
    load_pinned_pile_dataset,
)
from aspd.multimodel.v3.registry import build_model_registry
from aspd.multimodel.v3.training_capture import (
    broadcast_family_capture,
    broadcast_family_native_biases,
    capture_family_pool,
)

MODEL_ORDER = ("qwen", "gemma", "phi")
SOURCE_RANK = {"qwen": 0, "gemma": 1, "phi": 2}
EXPECTED_TRAINABLE_PARAMETERS = 12_575_332_352


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "main"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multimodel/aspd_v3/main_v1.yaml"),
    )
    parser.add_argument(
        "--approved",
        type=Path,
        default=Path("configs/multimodel/aspd_v3/approved.yaml"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("configs/multimodel/aspd_v3/candidates.yaml"),
    )
    parser.add_argument(
        "--runtime-template",
        type=Path,
        default=Path("configs/multimodel/aspd_v3/runtime_template.yaml"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _init_distributed() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, device


def _broadcast_object(value: Any, *, source: int = 0) -> Any:
    values = [value]
    dist.broadcast_object_list(values, src=source)
    return values[0]


def _assert_replicated_payload(payload: dict[str, Any]) -> None:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    observed: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(observed, digest)
    if len(set(observed)) != 1:
        raise RuntimeError(f"rank payload hashes disagree: {observed}")


def _read_manifest(path: Path, expected_kind: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != expected_kind:
        raise ValueError(f"unexpected main-v1 manifest kind: {path}")
    recorded = payload.get("manifest_sha256")
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    observed = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recorded != observed:
        raise ValueError(f"main-v1 manifest fingerprint mismatch: {path}")
    return payload


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _directory_bytes(path: Path) -> int:
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def _source_and_tokenizers(rank: int, registry, device, local_files_only: bool):
    source_key = MODEL_ORDER[rank] if rank < 3 else None
    loaded = None
    source_tokenizer = None
    if source_key is not None:
        source_tokenizer = load_v3_tokenizer(
            registry[source_key], local_files_only=local_files_only
        )
        loaded = load_frozen_v3_model(
            registry[source_key], device, local_files_only=local_files_only
        )
    alignment_tokenizers = None
    if rank == 0:
        alignment_tokenizers = {}
        for key in MODEL_ORDER:
            alignment_tokenizers[key] = (
                source_tokenizer
                if key == source_key
                else load_v3_tokenizer(registry[key], local_files_only=local_files_only)
            )
    return source_key, source_tokenizer, loaded, alignment_tokenizers


def _pad_id(tokenizer: Any) -> int:
    value = tokenizer.pad_token_id
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        raise ValueError("source tokenizer exposes neither pad_token_id nor eos_token_id")
    return int(value)


def _capture_pool(
    pool: AlignedAnchorPool,
    *,
    rank: int,
    source_key: str | None,
    source_tokenizer: Any,
    loaded: Any,
    registry,
    device: torch.device,
) -> MainV1CaptureBatch:
    activations: dict[str, dict[str, Any]] = {"pre": {}, "mid": {}}
    matrix_inputs = {}
    matrix_outputs = {}
    for key in MODEL_ORDER:
        source_rank = SOURCE_RANK[key]
        local = None
        if rank == source_rank:
            if source_key != key or loaded is None or source_tokenizer is None:
                raise RuntimeError(f"rank {rank} did not load its registered {key} source")
            local = capture_family_pool(
                loaded.forward_model,
                registry[key],
                pool.models[key],
                pad_token_id=_pad_id(source_tokenizer),
                device=device,
            )
        family = broadcast_family_capture(
            local,
            registry[key],
            anchor_count=pool.anchor_count,
            source_rank=source_rank,
            device=device,
        )
        activations["pre"][key] = family.residual_pre
        activations["mid"][key] = family.residual_mid
        matrix_inputs.update(family.matrix_inputs)
        matrix_outputs.update(family.matrix_outputs)
    return MainV1CaptureBatch(
        activations=activations,
        matrix_inputs=matrix_inputs,
        matrix_outputs=matrix_outputs,
        valid_anchors=torch.ones(1, pool.anchor_count, dtype=torch.bool, device=device),
    )


def _save_checkpoint(
    manager: MainV1DistributedCheckpoint,
    *,
    completed_steps: int,
    model,
    optimizer,
    scheduler,
    stream,
    ledger,
    rank: int,
    config_fingerprint: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    receipt_payload = None
    stream_state = None
    if rank == 0:
        receipt_payload = ledger.checkpoint_receipt(
            expected_optimizer_steps=completed_steps
        ).payload()
        stream_state = stream.state_dict()
    receipt_payload = _broadcast_object(receipt_payload)
    runtime_state = {
        "completed_steps": completed_steps,
        "accepted_exposures": completed_steps * 8192,
        "ledger_receipt": receipt_payload,
        "stream": stream_state,
    }
    manifest = manager.save(
        step=completed_steps,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        runtime_state=runtime_state,
        config_fingerprint=config_fingerprint,
        config_file_sha256=config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    return manager.root / f"step-{completed_steps:09d}", manifest


def _restore_checkpoint(
    manager: MainV1DistributedCheckpoint,
    checkpoint: Path,
    *,
    model,
    optimizer,
    scheduler,
    stream,
    ledger,
    rank: int,
    config_fingerprint: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
) -> int:
    runtime, _manifest = manager.load(
        path=checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config_fingerprint=config_fingerprint,
        config_file_sha256=config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    completed_steps = int(runtime["completed_steps"])
    if rank == 0:
        if runtime.get("stream") is None:
            raise ValueError("rank-zero checkpoint lacks its exact stream cursor")
        stream.load_state_dict(runtime["stream"])
        ledger.recover_to_receipt(
            ExposureLedgerReceipt.from_payload(runtime["ledger_receipt"])
        )
    dist.barrier()
    return completed_steps


def main() -> None:
    args = _parse_args()
    rank, world_size, device = _init_distributed()
    config = load_main_v1_config(args.config)
    if world_size != config.runtime.world_size:
        raise RuntimeError(
            "main-v1 distributed topology differs from the frozen runtime config: "
            f"expected {config.runtime.world_size}, observed {world_size}"
        )
    bundle = load_v3_config_bundle(args.approved, args.candidates, args.runtime_template)
    registry = build_model_registry(bundle.approved)
    config_sha256 = _sha256(args.config)
    config_fingerprint = config.fingerprint()
    dataset_manifest = _read_manifest(
        Path(config.runtime.dataset_manifest), "ASPD_V3_ON_THE_FLY_DATASET_MANIFEST"
    )
    alignment_manifest = _read_manifest(
        Path(config.runtime.alignment_manifest), "ASPD_V3_ON_THE_FLY_ALIGNMENT_MANIFEST"
    )
    if alignment_manifest["dataset_manifest_sha256"] != dataset_manifest["manifest_sha256"]:
        raise ValueError("alignment manifest is not bound to the dataset manifest")
    if dataset_manifest["resolved_revision"] != bundle.approved.dataset.revision:
        raise ValueError("dataset manifest revision differs from approved.yaml")

    if args.mode == "main":
        preflight_path = Path(config.runtime.preflight_evidence)
        if not preflight_path.is_file():
            raise RuntimeError("main-v1 requires materialized distributed preflight evidence")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        required_preflight = {
            "status": "PASS",
            "git_sha": _git_sha(),
            "config_fingerprint": config_fingerprint,
            "config_file_sha256": config_sha256,
            "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
            "alignment_manifest_sha256": alignment_manifest["manifest_sha256"],
            "world_size": config.runtime.world_size,
            "target_count": 168,
            "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
            "dcp_save_reload_resume": "PASS",
        }
        mismatch = {
            key: {"expected": value, "observed": preflight.get(key)}
            for key, value in required_preflight.items()
            if preflight.get(key) != value
        }
        if mismatch or int(preflight.get("optimizer_steps", 0)) < 3:
            raise RuntimeError(
                f"distributed preflight evidence does not authorize this main revision: {mismatch}"
            )

    if args.mode == "preflight":
        max_steps = args.max_steps or 3
        if not 3 <= max_steps <= config.runtime.max_preflight_steps:
            raise ValueError("preflight requires 3..20 steps to exercise save/reload/resume")
        run_dir = Path(config.runtime.preflight_evidence).parent
        checkpoint_root = run_dir / "checkpoints"
    else:
        max_steps = args.max_steps or config.data.minimum_optimizer_steps
        if max_steps != config.data.minimum_optimizer_steps:
            raise ValueError("main mode cannot reduce the frozen 250,000-step budget")
        if args.run_dir is None:
            raise ValueError("main mode requires an explicit immutable --run-dir")
        run_dir = args.run_dir
        allowed_root = Path(config.runtime.output_root)
        if allowed_root not in run_dir.parents:
            raise ValueError("main run directory lies outside the frozen output namespace")
        checkpoint_root = Path(config.runtime.checkpoint_root) / run_dir.name
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    torch.manual_seed(config.optimization.seed)
    torch.cuda.manual_seed_all(config.optimization.seed)
    source_key, source_tokenizer, loaded, alignment_tokenizers = _source_and_tokenizers(
        rank, registry, device, args.local_files_only
    )
    native_biases = {}
    for key in MODEL_ORDER:
        native_biases.update(
            broadcast_family_native_biases(
                None if rank != SOURCE_RANK[key] else loaded.forward_model,
                registry[key],
                source_rank=SOURCE_RANK[key],
                device=device,
            )
        )

    model = MainV1TrainableModel(
        config,
        registry,
        native_biases=native_biases,
        device=device,
        # Keep sharded optimizer/master parameters in FP32.  The FSDP mixed-precision policy
        # casts independently all-gathered forward parameters to the frozen BF16 compute dtype.
        dtype=torch.float32,
    )
    if trainable_parameter_count(model) != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("full main-v1 trainable parameter count changed")
    model = fully_shard_main_v1(
        model, expected_world_size=config.runtime.world_size
    )
    optimizer = build_adam_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    manager = MainV1DistributedCheckpoint(
        checkpoint_root, keep_complete=config.runtime.keep_complete_checkpoints
    )

    stream = None
    ledger = None
    if rank == 0:
        dataset = load_pinned_pile_dataset(
            bundle.approved.dataset.dataset_id, bundle.approved.dataset.revision
        )
        split_policy = HashSplitPolicy(
            modulus=config.data.split_modulus,
            validation_buckets=frozenset(config.data.validation_buckets),
            test_buckets=frozenset(config.data.test_buckets),
            seed=config.data.split_seed,
        )
        stream = PileAlignedAnchorStream(
            dataset=dataset,
            dataset_id=bundle.approved.dataset.dataset_id,
            dataset_revision=bundle.approved.dataset.revision,
            tokenizers=alignment_tokenizers,
            tokenizer_revisions={
                key: registry[key].tokenizer_revision for key in MODEL_ORDER
            },
            split_policy=split_policy,
            dedup_path=run_dir / "stream_dedup.sqlite",
            sequence_length=config.data.sequence_length,
        )
        ledger = SQLiteExposureLedger(
            run_dir / "exposure_ledger.sqlite",
            dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
            world_size=world_size,
        )

    initial_runtime = run_dir / "initial_runtime.json"
    completed_steps = 0
    checkpoints = manager.complete_checkpoints()
    if checkpoints:
        completed_steps = _restore_checkpoint(
            manager,
            checkpoints[-1],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            stream=stream,
            ledger=ledger,
            rank=rank,
            config_fingerprint=config_fingerprint,
            config_sha256=config_sha256,
            dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
        )
    elif rank == 0:
        if initial_runtime.is_file():
            initial = json.loads(initial_runtime.read_text(encoding="utf-8"))
            if initial["config_fingerprint"] != config_fingerprint:
                raise ValueError("initial runtime config fingerprint mismatch")
            stream.load_state_dict(initial["stream"])
            ledger.recover_to_receipt(
                ExposureLedgerReceipt.from_payload(initial["ledger_receipt"])
            )
        else:
            write_json_immutable(
                initial_runtime,
                {
                    "schema_version": 1,
                    "config_fingerprint": config_fingerprint,
                    "stream": stream.state_dict(),
                    "ledger_receipt": ledger.checkpoint_receipt(
                        expected_optimizer_steps=0
                    ).payload(),
                },
            )
    dist.barrier()

    if rank == 0 and not (run_dir / "run_manifest.json").exists():
        write_json_immutable(
            run_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "mode": args.mode,
                "git_sha": _git_sha(),
                "config_path": str(args.config),
                "config_file_sha256": config_sha256,
                "config_fingerprint": config_fingerprint,
                "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
                "alignment_manifest_sha256": alignment_manifest["manifest_sha256"],
                "world_size": world_size,
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
                "target_count": 168,
            },
        )

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    metrics_path = run_dir / "metrics.jsonl"
    started = time.perf_counter()
    resumed_preflight = False
    preflight_checkpoint: Path | None = None
    dcp_manifest: dict[str, Any] | None = None
    optimizer.zero_grad(set_to_none=True)

    while completed_steps < max_steps:
        step_started = time.perf_counter()
        step_anchor_ids: dict[int, list[str]] = {}
        pool_metrics = []
        for pool_index in range(4):
            payload = None
            if rank == 0:
                pool = stream.next_pool(pool_anchors=config.architecture.pool_tokens)
                payload = pool.to_payload()
                for observation in pool.source_observations:
                    ledger.observe_raw_tokens(
                        str(observation["observation_id"]),
                        observation["raw_tokens"],
                        before_optimizer_step=completed_steps,
                    )
            payload = _broadcast_object(payload)
            _assert_replicated_payload(payload)
            pool = AlignedAnchorPool.from_payload(payload)
            pool.validate(
                expected_anchors=config.architecture.pool_tokens,
                sequence_length=config.data.sequence_length,
            )
            step_anchor_ids[pool_index] = list(pool.anchor_ids)
            capture = _capture_pool(
                pool,
                rank=rank,
                source_key=source_key,
                source_tokenizer=source_tokenizer,
                loaded=loaded,
                registry=registry,
                device=device,
            )
            model.set_requires_gradient_sync(pool_index == 3, recurse=True)
            scores, sparse, binary = model.encode(capture)
            activation_loss, activation_metrics = model.activation_objective(
                capture, scores, sparse
            )
            (activation_loss / 4).backward()
            internal_sum = activation_loss.detach().new_zeros(())
            internal_metrics = {}
            for key, _site, value, coefficient in model.internal_loss_terms(
                capture, binary
            ):
                (value * coefficient / 4).backward()
                internal_sum.add_(value.detach() * coefficient)
                internal_metrics[f"loss/internal/{key}"] = value.detach()
            pool_metrics.append(
                {
                    "total": float((activation_loss.detach() + internal_sum).float()),
                    "activation": float(activation_metrics["loss/global_activation"].float()),
                    "auxk": float(activation_metrics["loss/global_auxk"].float()),
                    "internal": float(internal_sum.float()),
                }
            )
            del capture, scores, sparse, binary, activation_loss, activation_metrics

        gradient_report = finite_gradient_report(model)
        reports = [None] * world_size
        dist.all_gather_object(reports, gradient_report)
        if any(report["status"] != "PASS" for report in reports):
            raise FloatingPointError(f"missing or nonfinite trainable gradients: {reports}")
        model.normalize_decoders_()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed_steps += 1

        counters = None
        if rank == 0:
            counters = ledger.observe_global_optimizer_step(
                completed_steps - 1, step_anchor_ids
            )
            if counters.accepted_aligned_anchor_exposures != completed_steps * 8192:
                raise RuntimeError("global accepted-exposure accounting drifted")
        counters_payload = _broadcast_object(
            None
            if counters is None
            else {
                "accepted": counters.accepted_aligned_anchor_exposures,
                "unique": counters.unique_aligned_anchors,
                "repeated": counters.repeated_anchor_exposures,
            }
        )
        step_seconds = time.perf_counter() - step_started
        if rank == 0:
            _append_jsonl(
                metrics_path,
                {
                    "step": completed_steps,
                    "accepted_exposures": counters_payload["accepted"],
                    "unique_anchors": counters_payload["unique"],
                    "repeated_exposures": counters_payload["repeated"],
                    "seconds": step_seconds,
                    "anchors_per_second": 8192 / step_seconds,
                    "loss": {
                        key: sum(value[key] for value in pool_metrics) / 4
                        for key in ("total", "activation", "auxk", "internal")
                    },
                },
            )

        checkpoint_due = (
            args.mode == "main"
            and (
                completed_steps % config.runtime.checkpoint_every_steps == 0
                or completed_steps == max_steps
                or stop_requested
            )
        )
        preflight_reload_due = args.mode == "preflight" and completed_steps == max_steps - 1
        if checkpoint_due or preflight_reload_due:
            preflight_checkpoint, dcp_manifest = _save_checkpoint(
                manager,
                completed_steps=completed_steps,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stream=stream,
                ledger=ledger,
                rank=rank,
                config_fingerprint=config_fingerprint,
                config_sha256=config_sha256,
                dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
            )
            if preflight_reload_due:
                restored_step = _restore_checkpoint(
                    manager,
                    preflight_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    stream=stream,
                    ledger=ledger,
                    rank=rank,
                    config_fingerprint=config_fingerprint,
                    config_sha256=config_sha256,
                    dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
                )
                if restored_step != completed_steps:
                    raise RuntimeError("preflight DCP restored the wrong optimizer step")
                resumed_preflight = True
        if stop_requested:
            break

    local_memory = {
        "rank": rank,
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
    }
    memory = [None] * world_size
    dist.all_gather_object(memory, local_memory)
    if args.mode == "preflight":
        if not resumed_preflight or completed_steps != max_steps:
            raise RuntimeError("preflight did not complete save/reload and one resumed update")
        if rank == 0:
            dcp_bytes = _directory_bytes(preflight_checkpoint)
            report = {
                "schema_version": 1,
                "status": "PASS",
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "git_sha": _git_sha(),
                "config_fingerprint": config_fingerprint,
                "config_file_sha256": config_sha256,
                "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
                "alignment_manifest_sha256": alignment_manifest["manifest_sha256"],
                "world_size": world_size,
                "target_count": 168,
                "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
                "optimizer_steps": completed_steps,
                "accepted_exposures": completed_steps * 8192,
                "checkpoint_path": str(preflight_checkpoint),
                "checkpoint_bytes": dcp_bytes,
                "checkpoint_manifest": dcp_manifest,
                "dcp_save_reload_resume": "PASS",
                "gpu_memory_by_rank": memory,
                "elapsed_seconds": time.perf_counter() - started,
                "anchors_per_second": completed_steps * 8192 / (time.perf_counter() - started),
            }
            write_json_immutable(Path(config.runtime.preflight_evidence), report)
    if rank == 0:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "completed_steps": completed_steps,
                    "accepted_exposures": completed_steps * 8192,
                    "stop_requested": stop_requested,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if stream is not None:
            stream.close()
        if ledger is not None:
            ledger.close()
    dist.barrier()
    dist.destroy_process_group()
    if stop_requested:
        raise SystemExit(99)


if __name__ == "__main__":
    main()
