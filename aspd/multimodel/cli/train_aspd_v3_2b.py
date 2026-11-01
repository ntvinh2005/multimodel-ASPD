"""Three-rank baseline/optimized trainer for ASPD v3 main-2b-v1.

``benchmark`` and ``preflight`` are authorized bounded engineering modes. ``main`` is
intentionally fail-closed while the versioned config says ``full_training_authorized: false``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F

from aspd.multimodel.v3.data import (
    ExposureLedgerReceipt,
    HashSplitPolicy,
    SQLiteExposureLedger,
    write_json_immutable,
)
from aspd.multimodel.v3.distributed_checkpoint import MainV1DistributedCheckpoint
from aspd.multimodel.v3.hooks import assert_frozen_backbone, logical_weight_bias
from aspd.multimodel.v3.main_2b_config import (
    MODEL_ORDER_2B,
    V3Main2BV1Config,
    load_main_2b_v1_config,
)
from aspd.multimodel.v3.main_model import (
    MainV1CaptureBatch,
    MainV1TrainableModel,
    build_adam_optimizer,
    fully_shard_main_v1,
)
from aspd.multimodel.v3.model_owned import (
    LocalCaptureBatch,
    ModelOwnedMain2B,
    build_model_owned_adam,
    model_owned_parameter_counts,
)
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.production_stream import (
    AlignedAnchorPool,
    ModelPoolInputs,
    PileAlignedAnchorStream,
    load_pinned_pile_dataset,
)
from aspd.multimodel.v3.registry import build_model_registry_2b
from aspd.multimodel.v3.training_capture import (
    broadcast_family_capture,
    broadcast_family_native_biases,
    capture_family_pool,
)

SOURCE_RANK = {key: index for index, key in enumerate(MODEL_ORDER_2B)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("benchmark", "preflight", "main"), required=True)
    parser.add_argument(
        "--strategy", choices=("fsdp_baseline", "model_owned"), required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multimodel/aspd_v3/main_2b_v1.yaml"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    return parser.parse_args()


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


def _init_distributed() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def _broadcast_object(value: Any, source: int = 0) -> Any:
    values = [value]
    dist.broadcast_object_list(values, src=source)
    return values[0]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_manifest(path: Path, kind: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != kind:
        raise ValueError(f"unexpected manifest kind: {path}")
    digest = payload.get("manifest_sha256")
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    observed = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != observed:
        raise ValueError(f"manifest checksum mismatch: {path}")
    return payload


def _pad_id(tokenizer: Any) -> int:
    value = tokenizer.pad_token_id
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        raise ValueError("tokenizer exposes neither padding nor EOS token")
    return int(value)


def _load_sources(rank: int, registry, device, local_files_only: bool):
    local_key = MODEL_ORDER_2B[rank]
    tokenizer = load_v3_tokenizer(registry[local_key], local_files_only=local_files_only)
    loaded = load_frozen_v3_model(
        registry[local_key], device, local_files_only=local_files_only
    )
    alignment_tokenizers = None
    if rank == 0:
        alignment_tokenizers = {
            key: tokenizer
            if key == local_key
            else load_v3_tokenizer(registry[key], local_files_only=local_files_only)
            for key in MODEL_ORDER_2B
        }
    return local_key, tokenizer, loaded, alignment_tokenizers


def _real_source_correctness_smoke(
    local_key: str,
    tokenizer: Any,
    loaded: Any,
    registry: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Verify all 56 native projections, hook removal, and early-exit parity."""

    layout = registry[local_key]
    assert_frozen_backbone(loaded.forward_model)
    encoded = tokenizer(
        "ASPD hook parity: Unicode café — code x += 1.",
        add_special_tokens=True,
        return_attention_mask=True,
    )
    ids = tuple(int(value) for value in encoded["input_ids"])
    mask = tuple(int(value) for value in encoded["attention_mask"])
    anchor_count = min(8, len(ids))
    pool = ModelPoolInputs(
        input_ids=(ids,),
        attention_masks=(mask,),
        anchor_document_indices=(0,) * anchor_count,
        anchor_token_indices=tuple(range(anchor_count)),
    )
    model_inputs = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([mask], dtype=torch.long, device=device),
    }
    with torch.inference_mode():
        before = loaded.forward_model(
            **model_inputs, use_cache=False, return_dict=True
        ).logits.detach()
    full = capture_family_pool(
        loaded.forward_model,
        layout,
        pool,
        pad_token_id=_pad_id(tokenizer),
        device=device,
        early_exit=False,
    )
    early = capture_family_pool(
        loaded.forward_model,
        layout,
        pool,
        pad_token_id=_pad_id(tokenizer),
        device=device,
        early_exit=True,
    )
    with torch.inference_mode():
        after = loaded.forward_model(
            **model_inputs, use_cache=False, return_dict=True
        ).logits.detach()
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    maximum_projection_error = 0.0
    for target in layout.targets():
        weight, bias, _path = logical_weight_bias(loaded.forward_model, target)
        expected = F.linear(full.matrix_inputs[target.key], weight, bias)
        observed = full.matrix_outputs[target.key]
        torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)
        maximum_projection_error = max(
            maximum_projection_error,
            float((observed.float() - expected.float()).abs().max()),
        )
        torch.testing.assert_close(
            early.matrix_inputs[target.key], full.matrix_inputs[target.key], rtol=0, atol=0
        )
        torch.testing.assert_close(
            early.matrix_outputs[target.key], full.matrix_outputs[target.key], rtol=0, atol=0
        )
    for full_values, early_values in (
        (full.residual_pre, early.residual_pre),
        (full.residual_mid, early.residual_mid),
    ):
        for full_value, early_value in zip(full_values, early_values, strict=True):
            torch.testing.assert_close(early_value, full_value, rtol=0, atol=0)
    assert_frozen_backbone(loaded.forward_model)
    return {
        "model": local_key,
        "status": "PASS",
        "targets": len(layout.targets()),
        "early_exit_capture_parity": "PASS",
        "hook_forward_immutability": "PASS",
        "frozen_parameters": "PASS",
        "maximum_native_projection_absolute_error": maximum_projection_error,
    }


def _capture_local(
    pool: AlignedAnchorPool,
    *,
    local_key: str,
    tokenizer: Any,
    loaded: Any,
    registry: Any,
    device: torch.device,
    early_exit: bool,
) -> LocalCaptureBatch:
    family = capture_family_pool(
        loaded.forward_model,
        registry[local_key],
        pool.models[local_key],
        pad_token_id=_pad_id(tokenizer),
        device=device,
        early_exit=early_exit,
    )
    return LocalCaptureBatch(
        model_key=local_key,
        activations={"pre": family.residual_pre, "mid": family.residual_mid},
        matrix_inputs=family.matrix_inputs,
        matrix_outputs=family.matrix_outputs,
        valid_anchors=torch.ones(1, pool.anchor_count, dtype=torch.bool, device=device),
    )


def _capture_fsdp(
    pool: AlignedAnchorPool,
    *,
    rank: int,
    local_key: str,
    tokenizer: Any,
    loaded: Any,
    registry: Any,
    device: torch.device,
) -> MainV1CaptureBatch:
    activations: dict[str, dict[str, Any]] = {"pre": {}, "mid": {}}
    matrix_inputs: dict[str, torch.Tensor] = {}
    matrix_outputs: dict[str, torch.Tensor] = {}
    for key in MODEL_ORDER_2B:
        source = SOURCE_RANK[key]
        local = None
        if rank == source:
            local = capture_family_pool(
                loaded.forward_model,
                registry[key],
                pool.models[key],
                pad_token_id=_pad_id(tokenizer),
                device=device,
                early_exit=False,
            )
        family = broadcast_family_capture(
            local,
            registry[key],
            anchor_count=pool.anchor_count,
            source_rank=source,
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


def _chunked(iterator, size: int):
    while True:
        values = list(itertools.islice(iterator, size))
        if not values:
            return
        yield values


def _model_owned_backward(
    model: ModelOwnedMain2B,
    capture: LocalCaptureBatch,
    config: V3Main2BV1Config,
    *,
    pool_scale: float,
) -> dict[str, float]:
    scores, sparse, continuous_support, binary_support = model.encode(capture)
    activation_sum = 0.0
    auxk_sum = 0.0
    activation_chunks = math.ceil(16 / config.runtime.activation_backward_chunk)
    terms = model.activation_loss_terms(capture, scores, sparse, continuous_support)
    for chunk_index, chunk in enumerate(
        _chunked(terms, config.runtime.activation_backward_chunk)
    ):
        loss = sum(
            coefficient * (activation + config.architecture.auxk_coefficient * auxk)
            for _name, activation, auxk, coefficient in chunk
        )
        (loss * pool_scale).backward(retain_graph=chunk_index + 1 < activation_chunks)
        activation_sum += sum(float((coefficient * value).detach().float()) for _, value, _, coefficient in chunk)
        auxk_sum += sum(float((coefficient * value).detach().float()) for _, _, value, coefficient in chunk)

    internal_sum = 0.0
    for chunk in _chunked(
        model.internal_loss_terms(capture, binary_support),
        config.runtime.matrix_backward_chunk,
    ):
        loss = sum(coefficient * value for _key, _site, value, coefficient in chunk)
        (loss * pool_scale).backward()
        internal_sum += sum(float((coefficient * value).detach().float()) for _, _, value, coefficient in chunk)
    return {
        "activation": activation_sum,
        "auxk": auxk_sum,
        "internal": internal_sum,
        "total": activation_sum + config.architecture.auxk_coefficient * auxk_sum + internal_sum,
    }


def _fsdp_backward(model, capture: MainV1CaptureBatch, pool_index: int) -> dict[str, float]:
    model.set_requires_gradient_sync(pool_index == 3, recurse=True)
    scores, sparse, binary = model.encode(capture)
    activation = 0.0
    auxk = 0.0
    for index, (_prefix, value, auxiliary, coefficient) in enumerate(
        model.activation_loss_terms(capture, scores, sparse)
    ):
        ((coefficient * (value + 0.03125 * auxiliary)) / 4).backward(
            retain_graph=index + 1 < 48
        )
        activation += float((coefficient * value).detach().float())
        auxk += float((coefficient * auxiliary).detach().float())
    internal = 0.0
    for _key, _site, value, coefficient in model.internal_loss_terms(capture, binary):
        (coefficient * value / 4).backward()
        internal += float((coefficient * value).detach().float())
    return {
        "activation": activation,
        "auxk": auxk,
        "internal": internal,
        "total": activation + 0.03125 * auxk + internal,
    }


def _finite_gradients(model: torch.nn.Module) -> dict[str, Any]:
    gradients = []
    missing = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        else:
            gradients.append(parameter.grad)
    if any(type(gradient).__name__ == "DTensor" for gradient in gradients):
        nonfinite = any(not bool(torch.isfinite(gradient).all()) for gradient in gradients)
        return {
            "status": "PASS" if not missing and not nonfinite else "FAIL",
            "missing": missing,
            "nonfinite": nonfinite,
        }
    found_inf = torch.zeros((), device=next(model.parameters()).device)
    inverse_scale = torch.ones((), device=found_inf.device)
    if gradients:
        torch._amp_foreach_non_finite_check_and_unscale_(gradients, found_inf, inverse_scale)
    return {"status": "PASS" if not missing and not bool(found_inf.item()) else "FAIL", "missing": missing, "nonfinite": bool(found_inf.item())}


def _timed(profile: bool, values: dict[str, float], name: str):
    class Timer:
        def __enter__(self):
            if profile:
                torch.cuda.synchronize()
            self.started = time.perf_counter()
            torch.cuda.nvtx.range_push(name)
            return self

        def __exit__(self, exc_type, exc, traceback):
            torch.cuda.nvtx.range_pop()
            if profile:
                torch.cuda.synchronize()
                values[name] = values.get(name, 0.0) + time.perf_counter() - self.started

    return Timer()


def main() -> None:
    args = _parse_args()
    rank, world, device = _init_distributed()
    config = load_main_2b_v1_config(args.config)
    if world != 3:
        raise RuntimeError("main-2b-v1 requires exactly three ranks")
    if args.mode == "main":
        if not config.full_training_authorized:
            raise RuntimeError("full 500M training is not authorized by main_2b_v1.yaml")
        if args.max_steps != config.data.minimum_optimizer_steps:
            raise RuntimeError("main mode cannot reduce the frozen 61,036-step budget")
    if args.mode == "preflight" and not 3 <= args.max_steps <= config.runtime.max_preflight_steps:
        raise ValueError("preflight requires 3..20 optimizer steps")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("warmup steps must leave at least one measured optimizer step")
    registry = build_model_registry_2b(config)
    config_sha = _sha256(args.config)
    dataset_manifest = _read_manifest(
        Path(config.runtime.dataset_manifest), "ASPD_V3_ON_THE_FLY_DATASET_MANIFEST"
    )
    alignment_manifest = _read_manifest(
        Path(config.runtime.alignment_manifest), "ASPD_V3_ON_THE_FLY_ALIGNMENT_MANIFEST"
    )
    if dataset_manifest["resolved_revision"] != config.data.dataset_revision:
        raise ValueError("dataset manifest does not resolve to the pinned revision")
    if alignment_manifest["model_order"] != list(MODEL_ORDER_2B):
        raise ValueError("alignment manifest model order differs from main-2b-v1")
    if rank == 0:
        args.run_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    torch.manual_seed(config.optimization.seed)
    torch.cuda.manual_seed_all(config.optimization.seed)
    local_key, tokenizer, loaded, alignment_tokenizers = _load_sources(
        rank, registry, device, args.local_files_only
    )
    local_source_smoke = _real_source_correctness_smoke(
        local_key, tokenizer, loaded, registry, device
    )
    source_smokes = [None] * world
    dist.all_gather_object(source_smokes, local_source_smoke)

    if args.strategy == "model_owned":
        native_biases = {
            target.key: None for target in registry[local_key].targets()
        }
        model: Any = ModelOwnedMain2B(
            config,
            registry,
            local_key,
            native_biases=native_biases,
            device=device,
            dtype=torch.float32,
        )
        model.synchronize_shared_state_()
        optimizer = build_model_owned_adam(model, config)
        checkpoint_manager: Any = ModelOwnedCheckpoint(
            args.run_dir / "checkpoints",
            keep_complete=config.runtime.keep_complete_checkpoints,
        )
    else:
        native_biases: dict[str, torch.Tensor | None] = {}
        for key in MODEL_ORDER_2B:
            native_biases.update(
                broadcast_family_native_biases(
                    loaded.forward_model if rank == SOURCE_RANK[key] else None,
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
            dtype=torch.float32,
        )
        model = fully_shard_main_v1(model, expected_world_size=world)
        optimizer = build_adam_optimizer(model, config)
        checkpoint_manager = MainV1DistributedCheckpoint(
            args.run_dir / "checkpoints",
            keep_complete=config.runtime.keep_complete_checkpoints,
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    optimizer.zero_grad(set_to_none=True)

    stream = None
    ledger = None
    if rank == 0:
        dataset = load_pinned_pile_dataset(
            config.data.dataset_id, config.data.dataset_revision
        )
        stream = PileAlignedAnchorStream(
            dataset=dataset,
            dataset_id=config.data.dataset_id,
            dataset_revision=config.data.dataset_revision,
            tokenizers=alignment_tokenizers,
            tokenizer_revisions={
                key: registry[key].tokenizer_revision for key in MODEL_ORDER_2B
            },
            split_policy=HashSplitPolicy(
                modulus=config.data.split_modulus,
                validation_buckets=frozenset(config.data.validation_buckets),
                test_buckets=frozenset(config.data.test_buckets),
                seed=config.data.split_seed,
            ),
            dedup_path=args.run_dir / "stream_dedup.sqlite",
            sequence_length=config.data.sequence_length,
            model_order=MODEL_ORDER_2B,
        )
        ledger = SQLiteExposureLedger(
            args.run_dir / "exposure_ledger.sqlite",
            dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
            world_size=world,
            model_order=MODEL_ORDER_2B,
        )
        write_json_immutable(
            args.run_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "mode": args.mode,
                "strategy": args.strategy,
                "git_sha": _git_sha(),
                "config_path": str(args.config),
                "config_file_sha256": config_sha,
                "config_fingerprint": config.fingerprint(),
                "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
                "alignment_manifest_sha256": alignment_manifest["manifest_sha256"],
                "model_order": list(MODEL_ORDER_2B),
                "model_revisions": {model.key: model.revision for model in config.models},
                "selected_layers": {
                    model.key: list(model.selected_layers) for model in config.models
                },
                "logical_target_count": 168,
                "hardware": torch.cuda.get_device_name(device),
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "parameter_analysis": model_owned_parameter_counts(config, registry),
                "source_correctness": source_smokes,
            },
        )
    dist.barrier()

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    metrics_path = args.run_dir / "metrics.jsonl"
    completed_steps = 0
    checkpoint_seconds = 0.0
    checkpoint_path: Path | None = None
    checkpoint_manifest = None
    resumed = False
    started = time.perf_counter()

    while completed_steps < args.max_steps:
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        stage: dict[str, float] = {}
        step_anchor_ids: list[str] = []
        pool_losses: list[dict[str, float]] = []
        for pool_index in range(config.architecture.pools_per_step):
            with _timed(args.profile_stages, stage, "data_alignment_and_broadcast"):
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
                pool = AlignedAnchorPool.from_payload(payload)
                pool.validate(
                    expected_anchors=config.architecture.pool_tokens,
                    sequence_length=config.data.sequence_length,
                    model_order=MODEL_ORDER_2B,
                )
                step_anchor_ids.extend(pool.anchor_ids)
            with _timed(args.profile_stages, stage, "frozen_backbone_and_capture"):
                if args.strategy == "model_owned":
                    capture = _capture_local(
                        pool,
                        local_key=local_key,
                        tokenizer=tokenizer,
                        loaded=loaded,
                        registry=registry,
                        device=device,
                        early_exit=config.runtime.early_exit_after_selected_layers,
                    )
                else:
                    capture = _capture_fsdp(
                        pool,
                        rank=rank,
                        local_key=local_key,
                        tokenizer=tokenizer,
                        loaded=loaded,
                        registry=registry,
                        device=device,
                    )
            with _timed(args.profile_stages, stage, "trainable_forward_backward"):
                if args.strategy == "model_owned":
                    losses = _model_owned_backward(
                        model, capture, config, pool_scale=1 / 4
                    )
                else:
                    losses = _fsdp_backward(model, capture, pool_index)
            pool_losses.append(losses)
            del capture, pool

        with _timed(args.profile_stages, stage, "gradient_sync_validation_normalization"):
            if args.strategy == "model_owned":
                model.all_reduce_shared_gradients_()
            gradient_report = _finite_gradients(model)
            reports = [None] * world
            dist.all_gather_object(reports, gradient_report)
            if any(report["status"] != "PASS" for report in reports):
                raise FloatingPointError(f"missing/nonfinite gradients: {reports}")
            model.normalize_decoders_(project_gradients=True)
        with _timed(args.profile_stages, stage, "adam_and_post_normalization"):
            optimizer.step()
            model.normalize_decoders_(project_gradients=False)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        completed_steps += 1
        counters = None
        if rank == 0:
            counters = ledger.observe_replicated_optimizer_step(
                completed_steps - 1, step_anchor_ids
            )
            if counters.accepted_aligned_anchor_exposures != completed_steps * 8192:
                raise RuntimeError("accepted-anchor exposure accounting drifted")
        counter_payload = _broadcast_object(
            None
            if counters is None
            else {
                "accepted": counters.accepted_aligned_anchor_exposures,
                "unique": counters.unique_aligned_anchors,
                "repeated": counters.repeated_anchor_exposures,
            }
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - step_started
        local_memory = {
            "rank": rank,
            "model": local_key,
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
        }
        memories = [None] * world
        dist.all_gather_object(memories, local_memory)
        if rank == 0:
            _append_jsonl(
                metrics_path,
                {
                    "step": completed_steps,
                    "warmup": completed_steps <= args.warmup_steps,
                    "seconds": seconds,
                    "anchors_per_second": 8192 / seconds,
                    "accepted_exposures": counter_payload["accepted"],
                    "unique_anchors": counter_payload["unique"],
                    "repeated_exposures": counter_payload["repeated"],
                    "loss": {
                        name: sum(value[name] for value in pool_losses) / 4
                        for name in ("total", "activation", "auxk", "internal")
                    },
                    "stage_seconds": stage,
                    "gpu_memory": memories,
                },
            )

        checkpoint_due = (
            args.mode == "preflight"
            and not args.skip_checkpoint
            and completed_steps == args.max_steps - 1
        )
        if checkpoint_due:
            torch.cuda.synchronize()
            checkpoint_started = time.perf_counter()
            receipt = None
            stream_state = None
            if rank == 0:
                receipt = ledger.checkpoint_receipt(
                    expected_optimizer_steps=completed_steps
                ).payload()
                stream_state = stream.state_dict()
            receipt = _broadcast_object(receipt)
            runtime_state = {
                "completed_steps": completed_steps,
                "accepted_exposures": completed_steps * 8192,
                "ledger_receipt": receipt,
                "stream": stream_state,
            }
            if args.strategy == "model_owned":
                checkpoint_manifest = checkpoint_manager.save(
                    step=completed_steps,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    runtime_state=runtime_state,
                    config_fingerprint=config.fingerprint(),
                    config_file_sha256=config_sha,
                    dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
                    model_key=local_key,
                )
            else:
                checkpoint_manifest = checkpoint_manager.save(
                    step=completed_steps,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    runtime_state=runtime_state,
                    config_fingerprint=config.fingerprint(),
                    config_file_sha256=config_sha,
                    dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
                )
            checkpoint_path = args.run_dir / "checkpoints" / f"step-{completed_steps:09d}"
            if args.strategy == "model_owned":
                restored_runtime, _ = checkpoint_manager.load(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config_fingerprint=config.fingerprint(),
                    config_file_sha256=config_sha,
                    dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
                    expected_model_key=local_key,
                )
            else:
                restored_runtime, _ = checkpoint_manager.load(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config_fingerprint=config.fingerprint(),
                    config_file_sha256=config_sha,
                    dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
                )
            if int(restored_runtime["completed_steps"]) != completed_steps:
                raise RuntimeError("checkpoint restored the wrong optimizer step")
            if rank == 0:
                stream.load_state_dict(restored_runtime["stream"])
                ledger.recover_to_receipt(
                    ExposureLedgerReceipt.from_payload(restored_runtime["ledger_receipt"])
                )
            dist.barrier()
            resumed = True
            torch.cuda.synchronize()
            checkpoint_seconds = time.perf_counter() - checkpoint_started
        if stop_requested:
            break

    if rank == 0:
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        measured = [row for row in rows if not row["warmup"]]
        step_times = sorted(row["seconds"] for row in measured)

        def percentile(fraction: float) -> float:
            index = min(len(step_times) - 1, math.ceil(fraction * len(step_times)) - 1)
            return step_times[index]

        checkpoint_bytes = (
            sum(path.stat().st_size for path in checkpoint_path.rglob("*") if path.is_file())
            if checkpoint_path is not None
            else 0
        )
        report = {
            "schema_version": 1,
            "status": "PASS",
            "mode": args.mode,
            "strategy": args.strategy,
            "git_sha": _git_sha(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "config_fingerprint": config.fingerprint(),
            "optimizer_steps": completed_steps,
            "accepted_exposures": completed_steps * 8192,
            "warmup_steps": args.warmup_steps,
            "measured_steps": len(measured),
            "step_seconds_p50": percentile(0.50),
            "step_seconds_p95": percentile(0.95),
            "accepted_anchors_per_second": 8192 / (sum(step_times) / len(step_times)),
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_seconds_save_reload": checkpoint_seconds,
            "checkpoint_manifest": checkpoint_manifest,
            "dcp_save_reload_resume_update": (
                "PASS" if resumed and completed_steps == args.max_steps else "NOT_RUN"
            ),
            "peak_gpu_memory_by_rank": rows[-1]["gpu_memory"],
            "stage_seconds_mean": {
                name: sum(row["stage_seconds"].get(name, 0.0) for row in measured)
                / len(measured)
                for name in sorted(
                    {name for row in measured for name in row["stage_seconds"]}
                )
            },
        }
        write_json_immutable(args.run_dir / "benchmark_report.json", report)
        print(json.dumps(report, sort_keys=True), flush=True)
        stream.close()
        ledger.close()
    dist.barrier()
    dist.destroy_process_group()
    if stop_requested:
        raise SystemExit(99)


if __name__ == "__main__":
    main()
