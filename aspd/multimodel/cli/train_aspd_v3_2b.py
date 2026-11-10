"""Three-rank baseline/optimized trainer for ASPD v3 main-2b-v1.

``benchmark`` and ``preflight`` are authorized bounded engineering modes. ``main`` remains
fail-closed on explicit authorization and exact final-preflight provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.nn import functional as F

from aspd.multimodel.v3.controlled_retrain import (
    configure_controlled_trainables,
    load_controlled_retrain_contract,
    reduce_controlled_shared_gradients_,
)
from aspd.multimodel.v3.data import (
    ExposureLedgerReceipt,
    HashSplitPolicy,
    SQLiteExposureLedger,
    write_json_immutable,
)
from aspd.multimodel.v3.distributed_checkpoint import MainV1DistributedCheckpoint
from aspd.multimodel.v3.encoder_auxk_factorial_config import (
    EXPERIMENT_FAMILY as ENCODER_AUXK_FACTORIAL_FAMILY,
)
from aspd.multimodel.v3.encoder_auxk_factorial_config import (
    load_encoder_auxk_factorial_config,
)
from aspd.multimodel.v3.gate_collapse_recovery_config import (
    EXPERIMENT_FAMILY as GATE_COLLAPSE_RECOVERY_FAMILY,
)
from aspd.multimodel.v3.gate_collapse_recovery_config import (
    load_gate_collapse_recovery_config,
)
from aspd.multimodel.v3.gating import ORIGINAL_ASPD_N_POOLS_TO_DEAD
from aspd.multimodel.v3.granite_normalized_config import (
    load_granite_normalized_target_config,
)
from aspd.multimodel.v3.hooks import assert_frozen_backbone, logical_weight_bias
from aspd.multimodel.v3.main500m_config import (
    EXPERIMENT_FAMILY as MAIN500M_EXPERIMENT_FAMILY,
)
from aspd.multimodel.v3.main500m_config import load_main500m_config
from aspd.multimodel.v3.main500m_continuation import (
    load_continuation_authorization,
    milestone_diagnostic_due,
    milestone_pause_due,
    milestone_scientific_summary,
    next_review_exposure,
)
from aspd.multimodel.v3.main500m_monitoring import SustainedMain500MHealthMonitor
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
from aspd.multimodel.v3.two_model_config import load_two_model_scale_config
from aspd.multimodel.v3.two_model_matryoshka_config import (
    EXPERIMENT_FAMILY as TWO_MODEL_MATRYOSHKA_FAMILY,
)
from aspd.multimodel.v3.two_model_matryoshka_config import (
    load_two_model_matryoshka_config,
)

SOURCE_RANK = {key: index for index, key in enumerate(MODEL_ORDER_2B)}


class _PoolPrefetcher:
    """One-pool deterministic rank-zero lookahead with checkpointable buffering."""

    def __init__(
        self,
        stream: PileAlignedAnchorStream,
        *,
        pool_anchors: int,
        buffered_payload: dict[str, Any] | None = None,
    ):
        self.stream = stream
        self.pool_anchors = pool_anchors
        self.buffered_payload = buffered_payload
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aspd-align")
        self.future: Future[dict[str, Any]] | None = None

    def _produce(self) -> dict[str, Any]:
        return self.stream.next_pool(pool_anchors=self.pool_anchors).to_payload()

    def start(self) -> None:
        if self.buffered_payload is None and self.future is None:
            self.future = self.executor.submit(self._produce)

    def next_payload(self, *, schedule_next: bool = True) -> dict[str, Any]:
        if self.buffered_payload is not None:
            payload = self.buffered_payload
            self.buffered_payload = None
        else:
            self.start()
            assert self.future is not None
            payload = self.future.result()
            self.future = None
        if schedule_next:
            self.start()
        return payload

    def checkpoint_payload(self) -> dict[str, Any] | None:
        if self.buffered_payload is None and self.future is not None:
            self.buffered_payload = self.future.result()
            self.future = None
        return self.buffered_payload

    def restore(self, payload: dict[str, Any] | None) -> None:
        if self.future is not None:
            raise RuntimeError("cannot restore stream while alignment prefetch is active")
        self.buffered_payload = payload

    def close(self) -> None:
        if self.future is not None:
            self.future.result()
            self.future = None
        self.executor.shutdown(wait=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("benchmark", "preflight", "pilot", "main", "controlled"),
        required=True,
    )
    parser.add_argument("--strategy", choices=("fsdp_baseline", "model_owned"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multimodel/aspd_v3/main_2b_v1.yaml"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--activation-backward-chunk", type=int)
    parser.add_argument("--matrix-backward-chunk", type=int)
    parser.add_argument("--sparse-backend", choices=("indexed", "csr", "dense"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--controlled-contract", type=Path)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="immutable external checkpoint directory for exact cross-namespace resume",
    )
    parser.add_argument(
        "--continuation-authorization",
        type=Path,
        help="immutable researcher decision for a checkpointed health/milestone pause",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _common_state_sha256(module: torch.nn.Module) -> str:
    """Hash all baseline/treatment-common tensors without perturbing RNG state."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if name == "reconstruction_log_scale":
            continue
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_training_config(path: Path) -> Any:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ASPD v3 training config must be a mapping")
    if payload.get("experiment_family") == "aspd_v3_gemma_smol_scale_pilot":
        return load_two_model_scale_config(path)
    if payload.get("experiment_family") == TWO_MODEL_MATRYOSHKA_FAMILY:
        return load_two_model_matryoshka_config(path)
    if payload.get("experiment_family") == GATE_COLLAPSE_RECOVERY_FAMILY:
        return load_gate_collapse_recovery_config(path)
    if payload.get("experiment_family") == ENCODER_AUXK_FACTORIAL_FAMILY:
        return load_encoder_auxk_factorial_config(path)
    if payload.get("experiment_family") == MAIN500M_EXPERIMENT_FAMILY:
        return load_main500m_config(path)
    granite_overlay_bases = {
        "granite_normalized_target_baseline_v1.yaml",
        "granite_normalized_target_treatment_v1.yaml",
    }
    if (
        payload.get("experiment_family") == "aspd_v3_granite_normalized_target_v1"
        or payload.get("extends") in granite_overlay_bases
    ):
        return load_granite_normalized_target_config(path)
    return load_main_2b_v1_config(path)


def _preflight_identity_requirements(
    config: Any,
    *,
    mode: str,
    git_sha: str,
    config_file_sha256: str,
) -> dict[str, str]:
    """Return the immutable identity expected from prerequisite preflight evidence.

    Historical main/pilot contracts require the exact execution config and commit to have
    produced their preflight.  The Granite stabilization pilot is intentionally a new bounded
    overlay and later commit; its authorization instead pins the complete preflight artifact SHA
    plus the approved preflight config fingerprint.
    """

    if (
        config.experiment_family
        in {
            "aspd_v3_granite_normalized_target_v1",
            "aspd_v3_granite_normalized_target_unconstrained_decoder_v1",
        }
        and mode == "pilot"
    ):
        authorization = config.pilot_authorization
        if authorization is None:
            raise RuntimeError("Granite pilot is missing its authorization record")
        return {"config_fingerprint": authorization.preflight_config_fingerprint}
    if config.experiment_family == MAIN500M_EXPERIMENT_FAMILY:
        return {
            "execution_contract_fingerprint": config.execution_contract_fingerprint()
        }
    return {
        "git_sha": git_sha,
        "config_file_sha256": config_file_sha256,
        "config_fingerprint": config.fingerprint(),
    }


def _authorize_main500m_mode(config: Any, args: argparse.Namespace) -> None:
    """Fail closed before NCCL or source loading, with separate preflight/main gates."""

    if args.mode == "preflight":
        if args.strategy != "model_owned":
            raise RuntimeError("ASPD v3.1 bounded preflight requires model-owned execution")
        if args.skip_checkpoint:
            raise RuntimeError("ASPD v3.1 bounded preflight cannot disable checkpoint recovery")
        if args.resume_checkpoint is not None:
            raise RuntimeError("ASPD v3.1 bounded preflight cannot resume an external run")
        config.require_preflight_authorized(
            optimizer_steps=args.max_steps,
            run_dir=args.run_dir,
        )
        return
    if args.mode == "main":
        if args.strategy != "model_owned":
            raise RuntimeError("ASPD v3.1 main requires model-owned execution")
        if args.skip_checkpoint:
            raise RuntimeError("ASPD v3.1 main cannot disable checkpoint recovery")
        config.require_main_authorized()
        if args.max_steps != config.data.optimizer_steps:
            raise RuntimeError("ASPD v3.1 main requires the exact 61,036-step budget")
        return
    raise RuntimeError("ASPD v3.1 main contract permits only preflight or main mode")


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _require_clean_worktree() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status:
        raise RuntimeError("2B benchmark/preflight requires a clean Git worktree")
    return _git_sha()


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


def _record_continuation_authorization(path: Path, payload: dict[str, Any]) -> None:
    """Append one decision exactly once, rejecting conflicting audit records."""

    existing = []
    if path.is_file():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in existing:
        if record.get("artifact_sha256") == payload["artifact_sha256"]:
            if record != payload:
                raise ValueError("continuation authorization audit record changed")
            return
        if record.get("accepted_exposures") == payload["accepted_exposures"]:
            raise ValueError("conflicting continuation authorization for one paused exposure")
    _append_jsonl(path, payload)


def _start_gpu_telemetry(path: Path) -> tuple[subprocess.Popen[str], Any]:
    handle = path.open("x", encoding="utf-8")
    query = ",".join(
        (
            "timestamp",
            "index",
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "power.draw",
            "clocks.sm",
        )
    )
    process = subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "-l",
            "1",
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, handle


def _stop_gpu_telemetry(process: subprocess.Popen[str], handle: Any) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    handle.close()


def _summarize_gpu_telemetry(path: Path) -> dict[str, dict[str, float | int]]:
    columns = (
        "gpu_utilization_percent_mean",
        "memory_utilization_percent_mean",
        "memory_used_mib_mean",
        "power_watts_mean",
        "sm_clock_mhz_mean",
    )
    values: dict[str, list[list[float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            row = [float(value) for value in parts[2:]]
        except ValueError:
            continue
        values.setdefault(parts[1], []).append(row)
    result: dict[str, dict[str, float | int]] = {}
    for index, rows in values.items():
        result[index] = {"samples": len(rows)}
        result[index].update(
            {
                name: sum(row[column] for row in rows) / len(rows)
                for column, name in enumerate(columns)
            }
        )
    return result


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


def _load_sources(
    rank: int,
    registry: Any,
    device: torch.device,
    local_files_only: bool,
    *,
    model_order: tuple[str, ...],
):
    local_key = model_order[rank]
    tokenizer = load_v3_tokenizer(registry[local_key], local_files_only=local_files_only)
    loaded = load_frozen_v3_model(registry[local_key], device, local_files_only=local_files_only)
    alignment_tokenizers = None
    if rank == 0:
        alignment_tokenizers = {
            key: tokenizer
            if key == local_key
            else load_v3_tokenizer(registry[key], local_files_only=local_files_only)
            for key in model_order
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
    activation_backward_chunk: int,
    matrix_backward_chunk: int,
    auxk_may_be_active: bool,
    collect_diagnostics: bool = False,
) -> dict[str, float]:
    scores, sparse, continuous_support, binary_support = model.encode(
        capture, collect_diagnostics=collect_diagnostics
    )
    activation_values: list[torch.Tensor] = []
    auxk_values: list[torch.Tensor] = []
    details: dict[str, torch.Tensor] = {}
    activation_chunks = math.ceil(16 / activation_backward_chunk)
    activation_diagnostics: dict[str, float] | None = {} if collect_diagnostics else None
    terms = model.activation_loss_terms(
        capture,
        scores,
        sparse,
        continuous_support,
        auxk_may_be_active=auxk_may_be_active,
        diagnostics=activation_diagnostics,
    )
    for chunk_index, chunk in enumerate(_chunked(terms, activation_backward_chunk)):
        with torch.profiler.record_function("aspd/activation_chunk_forward_backward"):
            loss = sum(
                coefficient * (activation + config.architecture.auxk_coefficient * auxk)
                for _name, activation, auxk, coefficient in chunk
            )
            (loss * pool_scale).backward(retain_graph=chunk_index + 1 < activation_chunks)
        activation_values.extend(
            (coefficient * value).detach().float() for _, value, _, coefficient in chunk
        )
        auxk_values.extend(
            (coefficient * value).detach().float() for _, _, value, coefficient in chunk
        )
        for name, activation, auxk, _coefficient in chunk:
            details[f"loss/activation/{name}"] = activation.detach().float()
            details[f"loss/auxk/{name}"] = auxk.detach().float()

    internal_values: list[torch.Tensor] = []
    for chunk in _chunked(
        model.internal_loss_terms(
            capture,
            binary_support,
            dense_code={site: value.values for site, value in sparse.items()},
        ),
        matrix_backward_chunk,
    ):
        with torch.profiler.record_function("aspd/internal_chunk_forward_backward"):
            loss = sum(coefficient * value for _key, _site, value, coefficient in chunk)
            (loss * pool_scale).backward()
        internal_values.extend(
            (coefficient * value).detach().float() for _, _, value, coefficient in chunk
        )
        for name, _site, value, _coefficient in chunk:
            details[f"loss/internal/{name}"] = value.detach().float()
    tensors = {
        "activation": torch.stack(activation_values).sum(),
        "auxk": torch.stack(auxk_values).sum(),
        "internal": torch.stack(internal_values).sum(),
        **details,
    }
    names = tuple(tensors)
    values = torch.stack(tuple(tensors.values())).tolist()
    result = dict(zip(names, values, strict=True))
    if collect_diagnostics:
        for site, values_by_name in model.latest_gate_diagnostics.items():
            for name, value in values_by_name.items():
                result[f"diagnostic/gate/{site}/{model.local_model_key}/{name}"] = float(value)
        assert activation_diagnostics is not None
        result.update(
            {
                f"diagnostic/activation/{name}": value
                for name, value in activation_diagnostics.items()
            }
        )
    result["total"] = (
        result["activation"]
        + config.architecture.auxk_coefficient * result["auxk"]
        + result["internal"]
    )
    return result


def _controlled_activation_backward(
    model: ModelOwnedMain2B,
    capture: LocalCaptureBatch,
    config: V3Main2BV1Config,
    *,
    pool_scale: float,
    activation_backward_chunk: int,
) -> dict[str, float]:
    """Activation-only continuation with frozen rank-one matrix factors."""

    scores, sparse, continuous_support, _binary_support = model.encode(capture)
    activation_values: list[torch.Tensor] = []
    auxk_values: list[torch.Tensor] = []
    details: dict[str, torch.Tensor] = {}
    terms = list(
        model.activation_loss_terms(
            capture,
            scores,
            sparse,
            continuous_support,
            auxk_may_be_active=True,
        )
    )
    chunks = list(_chunked(iter(terms), activation_backward_chunk))
    for chunk_index, chunk in enumerate(chunks):
        loss = sum(
            coefficient * (activation + config.architecture.auxk_coefficient * auxk)
            for _name, activation, auxk, coefficient in chunk
        )
        (loss * pool_scale).backward(retain_graph=chunk_index + 1 < len(chunks))
        activation_values.extend(
            (coefficient * value).detach().float() for _, value, _, coefficient in chunk
        )
        auxk_values.extend(
            (coefficient * value).detach().float() for _, _, value, coefficient in chunk
        )
        for name, activation, auxk, _coefficient in chunk:
            details[f"loss/activation/{name}"] = activation.detach().float()
            details[f"loss/auxk/{name}"] = auxk.detach().float()
    activation = torch.stack(activation_values).sum()
    auxk = torch.stack(auxk_values).sum()
    result = {
        "activation": float(activation),
        "auxk": float(auxk),
        "internal": 0.0,
        **{name: float(value) for name, value in details.items()},
    }
    result["total"] = result["activation"] + config.architecture.auxk_coefficient * result["auxk"]
    return result


def _fsdp_backward(model, capture: MainV1CaptureBatch, pool_index: int) -> dict[str, float]:
    model.set_requires_gradient_sync(pool_index == 3, recurse=True)
    scores, sparse, binary = model.encode(capture)
    activation = 0.0
    auxk = 0.0
    for index, (_prefix, value, auxiliary, coefficient) in enumerate(
        model.activation_loss_terms(capture, scores, sparse)
    ):
        ((coefficient * (value + 0.03125 * auxiliary)) / 4).backward(retain_graph=index + 1 < 48)
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
    return {
        "status": "PASS" if not missing and not bool(found_inf.item()) else "FAIL",
        "missing": missing,
        "nonfinite": bool(found_inf.item()),
    }


@torch.no_grad()
def _gradient_norm_report(model: torch.nn.Module, *, rank: int) -> dict[str, Any]:
    """Measure model-owned gradient groups only on scheduled monitoring steps."""

    groups = {
        "local_encoder": model.local_encoders,
        "shared_encoder": model.shared_encoders,
        "activation_decoder": model.activation_banks,
        "rank_one": model.matrix_bank,
    }
    l2: dict[str, float] = {}
    for name, module in groups.items():
        squared = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared.add_(parameter.grad.detach().double().square().sum())
        l2[name] = float(squared.sqrt().cpu())
    return {"rank": rank, "model": model.local_model_key, "l2": l2}


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
    git_sha = _require_clean_worktree()
    config = _load_training_config(args.config)
    if config.experiment_family == MAIN500M_EXPERIMENT_FAMILY:
        # This happens before NCCL initialization or any source-model/tokenizer load.
        _authorize_main500m_mode(config, args)
    elif args.continuation_authorization is not None:
        raise ValueError(
            "--continuation-authorization is valid only for ASPD v3.1 main execution"
        )
    rank, world, device = _init_distributed()
    if config.experiment_family in {
        "aspd_v3_granite_normalized_target_v1",
        "aspd_v3_granite_normalized_target_unconstrained_decoder_v1",
    }:
        if args.strategy != config.runtime.strategy:
            raise RuntimeError("Granite normalized-target runtime strategy changed")
        if args.mode == "preflight":
            if not config.gpu_preflight_authorized:
                raise RuntimeError("Granite normalized-target preflight is not authorized")
            if (
                config.preflight_authorization is None
                or args.max_steps != config.preflight_authorization.optimizer_steps
            ):
                raise RuntimeError("Granite normalized-target preflight must run exactly three steps")
        elif args.mode == "pilot":
            authorization = config.pilot_authorization
            if not config.short_pilot_authorized or authorization is None:
                raise RuntimeError("Granite normalized-target pilot is not authorized")
            if args.max_steps != authorization.optimizer_steps:
                raise RuntimeError("Granite pilot step budget differs from its authorization")
            preflight_path = Path(config.runtime.preflight_evidence)
            if _sha256(preflight_path) != authorization.preflight_report_sha256:
                raise RuntimeError("Granite pilot preflight evidence checksum mismatch")
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            if (
                preflight.get("status") != "PASS"
                or preflight.get("config_fingerprint")
                != authorization.preflight_config_fingerprint
            ):
                raise RuntimeError("Granite pilot preflight evidence is not the authorized PASS")
            for artifact in authorization.gate_evidence:
                path = Path(artifact["path"])
                if not path.is_file() or _sha256(path) != artifact["sha256"]:
                    raise RuntimeError(
                        f"Granite pilot gate evidence checksum mismatch: {artifact['name']}"
                    )
        else:
            raise RuntimeError(
                "Granite normalized-target family is authorized only for bounded preflight/pilot"
            )
    elif config.experiment_family == ENCODER_AUXK_FACTORIAL_FAMILY:
        if args.strategy != "model_owned":
            raise RuntimeError("encoder/AuxK factorial requires model-owned execution")
        if args.mode == "benchmark":
            if args.max_steps != config.preflight_authorization.optimizer_steps:
                raise RuntimeError("factorial engineering smoke must run exactly three steps")
            if not args.skip_checkpoint:
                raise RuntimeError("factorial engineering smoke must skip checkpoint creation")
        elif args.mode == "pilot":
            if args.max_steps != config.pilot_authorization.optimizer_steps:
                raise RuntimeError("factorial screen must run exactly 256 steps")
        else:
            raise RuntimeError("encoder/AuxK factorial is limited to benchmark and pilot modes")
    elif config.experiment_family == TWO_MODEL_MATRYOSHKA_FAMILY:
        if args.strategy != "model_owned":
            raise RuntimeError("two-model Matryoshka ablation requires model-owned execution")
        if tuple(config.architecture.model_order) != ("gemma", "smol"):
            raise RuntimeError("two-model Matryoshka ablation model order changed")
        if any("qwen" in model.model_id.lower() for model in config.models):
            raise RuntimeError("Qwen is forbidden in the two-model Matryoshka ablation")
        if args.mode == "preflight":
            if args.max_steps != 3 or args.skip_checkpoint:
                raise RuntimeError("Matryoshka preflight requires three checkpointed steps")
        elif args.mode == "pilot":
            if args.max_steps != 1221 or args.skip_checkpoint:
                raise RuntimeError("Matryoshka pilot requires 1,221 checkpointed steps")
            preflight_path = Path(config.runtime.preflight_evidence)
            if not preflight_path.is_file():
                raise RuntimeError("Matryoshka pilot requires arm-matched preflight evidence")
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            expected_preflight = {
                "status": "PASS",
                "mode": "preflight",
                "optimizer_steps": 3,
                "world_size": 2,
                "logical_target_count": 112,
                "git_sha": git_sha,
                "config_file_sha256": _sha256(args.config),
                "config_fingerprint": config.fingerprint(),
                "checkpoint_parameter_and_adam_restore_probe": "PASS",
            }
            mismatch = {
                key: {"expected": value, "observed": preflight.get(key)}
                for key, value in expected_preflight.items()
                if preflight.get(key) != value
            }
            if mismatch:
                raise RuntimeError(f"Matryoshka preflight evidence mismatch: {mismatch}")
        else:
            raise RuntimeError("Matryoshka ablation is limited to preflight and pilot modes")
    elif config.experiment_family == GATE_COLLAPSE_RECOVERY_FAMILY:
        if args.strategy != "model_owned":
            raise RuntimeError("gate-collapse recovery requires model-owned execution")
        if tuple(config.architecture.model_order) != ("gemma", "smol"):
            raise RuntimeError("gate-collapse recovery model order changed")
        if any("qwen" in model.model_id.lower() for model in config.models):
            raise RuntimeError("Qwen is forbidden in the gate-collapse recovery pilot")
        if args.mode == "preflight":
            if args.max_steps != 3 or args.skip_checkpoint:
                raise RuntimeError("gate-collapse preflight requires three checkpointed steps")
        elif args.mode == "pilot":
            if args.max_steps != 1221 or args.skip_checkpoint:
                raise RuntimeError("gate-collapse pilot requires 1,221 checkpointed steps")
            preflight_path = Path(config.runtime.preflight_evidence)
            if not preflight_path.is_file():
                raise RuntimeError("gate-collapse pilot requires arm-matched preflight evidence")
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            expected_preflight = {
                "status": "PASS",
                "mode": "preflight",
                "optimizer_steps": 3,
                "world_size": 2,
                "logical_target_count": 112,
                "git_sha": git_sha,
                "config_file_sha256": _sha256(args.config),
                "config_fingerprint": config.fingerprint(),
                "checkpoint_parameter_and_adam_restore_probe": "PASS",
            }
            mismatch = {
                key: {"expected": value, "observed": preflight.get(key)}
                for key, value in expected_preflight.items()
                if preflight.get(key) != value
            }
            if mismatch:
                raise RuntimeError(f"gate-collapse preflight evidence mismatch: {mismatch}")
        else:
            raise RuntimeError("gate-collapse recovery is limited to preflight and pilot modes")
    model_order = tuple(config.architecture.model_order)
    logical_target_count = len(model_order) * 8 * 7
    controlled = None
    controlled_sha = None
    source_runtime = None
    source_checkpoint_manifest = None
    if args.mode == "controlled":
        if args.controlled_contract is None:
            raise ValueError("controlled mode requires --controlled-contract")
        if args.strategy != "model_owned" or args.skip_checkpoint:
            raise ValueError("controlled mode requires checkpointed model-owned execution")
        controlled = load_controlled_retrain_contract(args.controlled_contract)
        controlled_sha = _sha256(args.controlled_contract)
        if args.max_steps != controlled.additional_optimizer_steps:
            raise ValueError("controlled mode must run the frozen 1,221-step first stage")
        if _sha256(Path(controlled.source_config)) != controlled.source_config_sha256:
            raise ValueError("controlled source configuration checksum mismatch")
        if config.fingerprint() != controlled.source_config_fingerprint:
            raise ValueError("controlled source configuration fingerprint mismatch")
        source_manifest_path = Path(controlled.source_checkpoint) / "manifest.json"
        if _sha256(source_manifest_path) != controlled.source_checkpoint_manifest_sha256:
            raise ValueError("controlled source checkpoint manifest checksum mismatch")
        if args.run_dir.parent != Path(controlled.output_root) or not args.run_dir.name.startswith(
            "job_"
        ):
            raise ValueError("controlled run directory must be job-specific under output_root")
    elif args.controlled_contract is not None:
        raise ValueError("--controlled-contract is valid only in controlled mode")
    if world != len(model_order) or world != config.runtime.world_size:
        raise RuntimeError(f"training contract requires exactly {len(model_order)} ranks")
    if len(model_order) == 2 and args.strategy != "model_owned":
        raise RuntimeError("two-model pilots require model-owned execution")
    if args.mode == "main":
        if not config.full_training_authorized:
            raise RuntimeError("full 500M training is not authorized by main_2b_v1.yaml")
        if args.skip_checkpoint:
            raise RuntimeError("main mode cannot disable recoverable checkpoints")
        required_main_steps = getattr(config.data, "minimum_optimizer_steps", None)
        if required_main_steps is None:
            required_main_steps = config.data.optimizer_steps
        required_main_steps = int(required_main_steps)
        if args.max_steps != required_main_steps:
            raise RuntimeError("main mode cannot reduce the frozen 61,036-step budget")
    if args.resume_checkpoint is not None and (
        args.strategy != "model_owned" or args.skip_checkpoint
    ):
        raise ValueError("external exact resume requires checkpointed model-owned execution")
    if args.mode == "pilot":
        if not getattr(config, "short_pilot_authorized", False):
            raise RuntimeError("short pilot is not authorized by this config")
        if args.skip_checkpoint:
            raise RuntimeError("pilot mode cannot disable recoverable checkpoints")
        expected_pilot_steps = getattr(config.data, "optimizer_steps", None)
        if expected_pilot_steps is None:
            expected_pilot_steps = getattr(config.data, "approved_optimizer_steps", None)
        if args.max_steps != expected_pilot_steps:
            raise RuntimeError("pilot mode must run the frozen short-pilot step budget")
        if args.run_dir.parent != Path(
            config.runtime.output_root
        ) or not args.run_dir.name.startswith("job_"):
            raise ValueError("pilot run directory must be job-specific under output_root")
    if args.mode == "preflight" and not 3 <= args.max_steps <= config.runtime.max_preflight_steps:
        raise ValueError("preflight requires 3..20 optimizer steps")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("warmup steps must leave at least one measured optimizer step")
    activation_backward_chunk = (
        config.runtime.activation_backward_chunk
        if args.activation_backward_chunk is None
        else args.activation_backward_chunk
    )
    matrix_backward_chunk = (
        config.runtime.matrix_backward_chunk
        if args.matrix_backward_chunk is None
        else args.matrix_backward_chunk
    )
    if not 1 <= activation_backward_chunk <= 16:
        raise ValueError("activation backward chunk must be in 1..16")
    if not 1 <= matrix_backward_chunk <= 56:
        raise ValueError("matrix backward chunk must be in 1..56")
    sparse_backend = (
        config.runtime.sparse_backend if args.sparse_backend is None else args.sparse_backend
    )
    if args.mode != "benchmark" and (
        args.activation_backward_chunk is not None
        or args.matrix_backward_chunk is not None
        or args.sparse_backend is not None
    ):
        raise ValueError("runtime execution overrides are benchmark-only")
    if args.strategy != "model_owned" and sparse_backend != "indexed":
        raise ValueError("dense/CSR execution backends require model-owned strategy")
    runtime_payload = {
        "activation_backward_chunk": activation_backward_chunk,
        "matrix_backward_chunk": matrix_backward_chunk,
        "early_exit_after_selected_layers": config.runtime.early_exit_after_selected_layers,
        "sparse_backend": sparse_backend,
        "data_prefetch_depth": config.runtime.data_prefetch_depth,
        "ledger_anchor_storage": config.runtime.ledger_anchor_storage,
    }
    runtime_fingerprint = hashlib.sha256(
        json.dumps(runtime_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    registry = build_model_registry_2b(config)
    config_sha = _sha256(args.config)
    if controlled is not None and config_sha != controlled.source_config_sha256:
        raise ValueError("controlled --config is not the frozen source configuration")
    dataset_manifest = _read_manifest(
        Path(config.runtime.dataset_manifest), "ASPD_V3_ON_THE_FLY_DATASET_MANIFEST"
    )
    alignment_manifest = _read_manifest(
        Path(config.runtime.alignment_manifest), "ASPD_V3_ON_THE_FLY_ALIGNMENT_MANIFEST"
    )
    if dataset_manifest["resolved_revision"] != config.data.dataset_revision:
        raise ValueError("dataset manifest does not resolve to the pinned revision")
    if alignment_manifest["model_order"] != list(model_order):
        raise ValueError("alignment manifest model order differs from main-2b-v1")
    if args.mode in {"main", "pilot"}:
        preflight_path = Path(config.runtime.preflight_evidence)
        if not preflight_path.is_file():
            raise RuntimeError("main-2b-v1 requires immutable distributed preflight evidence")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if config.experiment_family == ENCODER_AUXK_FACTORIAL_FAMILY:
            required_smoke = {
                "status": "PASS",
                "mode": "benchmark",
                "strategy": "model_owned",
                "git_sha": git_sha,
                "config_file_sha256": config_sha,
                "config_fingerprint": config.fingerprint(),
                "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
                "alignment_manifest_sha256": alignment_manifest["manifest_sha256"],
                "world_size": len(model_order),
                "logical_target_count": logical_target_count,
                "runtime_fingerprint": runtime_fingerprint,
                "optimizer_steps": config.preflight_authorization.optimizer_steps,
                "checkpoint_path": None,
                "checkpoint_bytes": 0,
                "dcp_save_reload_resume_update": "NOT_RUN",
            }
            mismatch = {
                key: {"expected": value, "observed": preflight.get(key)}
                for key, value in required_smoke.items()
                if preflight.get(key) != value
            }
            if mismatch:
                raise RuntimeError(f"factorial engineering smoke mismatch: {mismatch}")
        else:
            required = {
                "status": "PASS",
                "mode": "preflight",
                "strategy": "model_owned",
                **_preflight_identity_requirements(
                    config,
                    mode=args.mode,
                    git_sha=git_sha,
                    config_file_sha256=config_sha,
                ),
                "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
                "alignment_manifest_sha256": alignment_manifest["manifest_sha256"],
                "world_size": len(model_order),
                "logical_target_count": logical_target_count,
                "dcp_save_reload_resume_update": "PASS",
                "checkpoint_parameter_and_adam_restore_probe": "PASS",
                "runtime_fingerprint": runtime_fingerprint,
            }
            mismatch = {
                key: {"expected": value, "observed": preflight.get(key)}
                for key, value in required.items()
                if preflight.get(key) != value
            }
            checkpoint_manifest = preflight.get("checkpoint_manifest")
            recovery_mismatch: dict[str, Any] = {}
            if not isinstance(checkpoint_manifest, dict):
                recovery_mismatch["checkpoint_manifest"] = "missing"
            else:
                for key, expected in (
                    ("status", "COMPLETE"),
                    ("world_size", len(model_order)),
                ):
                    if checkpoint_manifest.get(key) != expected:
                        recovery_mismatch[f"checkpoint_manifest.{key}"] = {
                            "expected": expected,
                            "observed": checkpoint_manifest.get(key),
                        }
            if int(preflight.get("checkpoint_bytes", 0)) <= 0:
                recovery_mismatch["checkpoint_bytes"] = "must be positive"
            if int(preflight.get("optimizer_steps", 0)) < 3:
                recovery_mismatch["optimizer_steps"] = "must be at least three"
            if mismatch or recovery_mismatch:
                raise RuntimeError(
                    "main-2b-v1 preflight evidence mismatch: "
                    f"provenance={mismatch}, recovery={recovery_mismatch}"
                )
    if rank == 0:
        args.run_dir.mkdir(parents=True, exist_ok=args.mode in {"main", "controlled", "pilot"})
    dist.barrier()

    torch.manual_seed(config.optimization.seed)
    torch.cuda.manual_seed_all(config.optimization.seed)
    local_key, tokenizer, loaded, alignment_tokenizers = _load_sources(
        rank,
        registry,
        device,
        args.local_files_only,
        model_order=model_order,
    )
    local_source_smoke = _real_source_correctness_smoke(
        local_key, tokenizer, loaded, registry, device
    )
    source_smokes = [None] * world
    dist.all_gather_object(source_smokes, local_source_smoke)

    if args.strategy == "model_owned":
        native_biases = {target.key: None for target in registry[local_key].targets()}
        model: Any = ModelOwnedMain2B(
            config,
            registry,
            local_key,
            native_biases=native_biases,
            device=device,
            dtype=torch.float32,
            sparse_backend=sparse_backend,
        )
        model.synchronize_shared_state_()
        checkpoint_manager: Any = ModelOwnedCheckpoint(
            args.run_dir / "checkpoints",
            keep_complete=config.runtime.keep_complete_checkpoints,
        )
        if controlled is not None:
            source_runtime, source_checkpoint_manifest = checkpoint_manager.load_model_only(
                path=controlled.source_checkpoint,
                model=model,
                config_fingerprint=config.fingerprint(),
                config_file_sha256=config_sha,
                dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
                expected_model_key=local_key,
            )
            if int(source_checkpoint_manifest["step"]) != controlled.source_checkpoint_step:
                raise ValueError("controlled source checkpoint step mismatch")
            if int(source_runtime["accepted_exposures"]) != controlled.source_accepted_exposures:
                raise ValueError("controlled source exposure count mismatch")
            groups = configure_controlled_trainables(model)
            groups[0]["lr"] = config.optimization.activation_decoder_lr
            groups[1]["lr"] = config.optimization.encoder_lr
            groups[2]["lr"] = config.optimization.encoder_lr
            optimizer = torch.optim.Adam(
                groups,
                betas=config.optimization.betas,
                weight_decay=config.optimization.weight_decay,
            )
        else:
            optimizer = build_model_owned_adam(model, config)
    else:
        native_biases: dict[str, torch.Tensor | None] = {}
        source_rank = {key: index for index, key in enumerate(model_order)}
        for key in model_order:
            native_biases.update(
                broadcast_family_native_biases(
                    loaded.forward_model if rank == source_rank[key] else None,
                    registry[key],
                    source_rank=source_rank[key],
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
    initial_common_state = {
        "rank": rank,
        "model": local_key,
        "sha256": _common_state_sha256(model),
    }
    initial_common_states = [None] * world
    dist.all_gather_object(initial_common_states, initial_common_state)
    qwen_smol_group = None
    if controlled is not None:
        # All ranks must construct groups in the same order, including Gemma rank 1.
        qwen_smol_group = dist.new_group(ranks=list(controlled.qwen_smol_group))

    external_resume_manifest = None
    if args.resume_checkpoint is not None:
        if checkpoint_manager.complete_checkpoints():
            raise ValueError(
                "external resume cannot be mixed with checkpoints already in the new run"
            )
        if rank == 0:
            external_resume_manifest = checkpoint_manager.restore_recovery_databases(
                path=args.resume_checkpoint,
                destination_root=args.run_dir,
                config_fingerprint=config.fingerprint(),
                config_file_sha256=config_sha,
                dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
                expected_world_size=world,
            )
        external_resume_manifest = _broadcast_object(external_resume_manifest)
        dist.barrier()

    stream = None
    ledger = None
    if rank == 0:
        stream_dedup_path = args.run_dir / "stream_dedup.sqlite"
        source_dedup_sha = None
        if controlled is not None:
            source_dedup = Path(controlled.source_checkpoint).parents[1] / "stream_dedup.sqlite"
            if not source_dedup.is_file():
                raise FileNotFoundError(f"missing source stream dedup snapshot: {source_dedup}")
            source_dedup_sha = _sha256(source_dedup)
            if not stream_dedup_path.exists():
                shutil.copy2(source_dedup, stream_dedup_path)
                if _sha256(stream_dedup_path) != source_dedup_sha:
                    raise ValueError("controlled stream dedup copy differs from its source")
        dataset = load_pinned_pile_dataset(config.data.dataset_id, config.data.dataset_revision)
        stream = PileAlignedAnchorStream(
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
            dedup_path=stream_dedup_path,
            sequence_length=config.data.sequence_length,
            model_order=model_order,
        )
        ledger = SQLiteExposureLedger(
            args.run_dir / "exposure_ledger.sqlite",
            dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
            world_size=world,
            model_order=model_order,
            compact_anchor_ids=True,
        )
        run_manifest_path = args.run_dir / "run_manifest.json"
        run_manifest = {
            "schema_version": 1,
            "mode": args.mode,
            "strategy": args.strategy,
            "git_sha": git_sha,
            "config_path": str(args.config),
            "config_file_sha256": config_sha,
            "config_fingerprint": config.fingerprint(),
            "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
            "alignment_manifest_sha256": alignment_manifest["manifest_sha256"],
            "model_order": list(model_order),
            "model_revisions": {model.key: model.revision for model in config.models},
            "selected_layers": {model.key: list(model.selected_layers) for model in config.models},
            "logical_target_count": logical_target_count,
            "hardware": torch.cuda.get_device_name(device),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "parameter_analysis": model_owned_parameter_counts(config, registry),
            "source_correctness": source_smokes,
            "scientific_approvals": (
                config.approvals.model_dump(mode="json")
                if config.experiment_family == MAIN500M_EXPERIMENT_FAMILY
                else None
            ),
            "initial_common_state_by_rank": initial_common_states,
            "runtime": runtime_payload,
            "runtime_fingerprint": runtime_fingerprint,
            "external_resume": (
                None
                if args.resume_checkpoint is None
                else {
                    "checkpoint": str(args.resume_checkpoint),
                    "manifest_sha256": _sha256(args.resume_checkpoint / "manifest.json"),
                    "step": int(external_resume_manifest["step"]),
                }
            ),
        }
        if controlled is not None:
            run_manifest["controlled_retrain"] = {
                "contract_path": str(args.controlled_contract),
                "contract_file_sha256": controlled_sha,
                "contract_fingerprint": controlled.fingerprint(),
                "arm": controlled.arm,
                "source_checkpoint": controlled.source_checkpoint,
                "source_checkpoint_manifest_sha256": controlled.source_checkpoint_manifest_sha256,
                "source_stream_dedup_sha256": source_dedup_sha,
                "source_checkpoint_step": controlled.source_checkpoint_step,
                "source_accepted_exposures": controlled.source_accepted_exposures,
                "additional_optimizer_steps": controlled.additional_optimizer_steps,
                "additional_accepted_exposures": controlled.additional_accepted_exposures,
                "optimizer_state": controlled.optimizer_state,
                "trainable_scope": list(controlled.trainable_scope),
                "frozen_scope": list(controlled.frozen_scope),
            }
        if run_manifest_path.is_file():
            existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            for key in (
                "strategy",
                "git_sha",
                "config_file_sha256",
                "config_fingerprint",
                "dataset_manifest_sha256",
                "alignment_manifest_sha256",
                "runtime_fingerprint",
                "external_resume",
            ):
                if existing_manifest.get(key) != run_manifest[key]:
                    raise ValueError(f"existing run manifest changed at {key}")
            if existing_manifest.get("controlled_retrain") != run_manifest.get(
                "controlled_retrain"
            ):
                raise ValueError("existing controlled retrain manifest changed")
        else:
            write_json_immutable(run_manifest_path, run_manifest)
    dist.barrier()

    health_monitor = (
        SustainedMain500MHealthMonitor(config.monitoring)
        if rank == 0 and config.experiment_family == MAIN500M_EXPERIMENT_FAMILY
        else None
    )
    completed_steps = 0
    restored_prefetched_pool = None
    consumed_continuation_authorizations: list[dict[str, Any]] = []
    pending_review: dict[str, Any] | None = None
    initial_runtime_path = args.run_dir / "initial_runtime.json"
    checkpoints = (
        [args.resume_checkpoint]
        if args.resume_checkpoint is not None
        else checkpoint_manager.complete_checkpoints()
    )
    if checkpoints:
        latest = checkpoints[-1]
        if args.strategy == "model_owned":
            restored_runtime, _ = checkpoint_manager.load(
                path=latest,
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
                path=latest,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config_fingerprint=config.fingerprint(),
                config_file_sha256=config_sha,
                dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
            )
        if controlled is not None:
            expected_resume = {
                "source_completed_steps": controlled.source_checkpoint_step,
                "source_accepted_exposures": controlled.source_accepted_exposures,
                "controlled_contract_fingerprint": controlled.fingerprint(),
            }
            mismatch = {
                key: {"expected": value, "observed": restored_runtime.get(key)}
                for key, value in expected_resume.items()
                if restored_runtime.get(key) != value
            }
            if mismatch:
                raise ValueError(f"controlled checkpoint resume provenance mismatch: {mismatch}")
        completed_steps = int(restored_runtime["completed_steps"])
        consumed_continuation_authorizations = list(
            restored_runtime.get("consumed_continuation_authorizations", [])
        )
        pending_review = restored_runtime.get("pending_review")
        if rank == 0:
            if health_monitor is not None:
                health_state = restored_runtime.get("health_monitor")
                if not isinstance(health_state, dict):
                    raise ValueError(
                        "ASPD v3.1 checkpoint is missing the sustained health-monitor state"
                    )
                health_monitor.load_state_dict(health_state)
            stream.load_state_dict(restored_runtime["stream"])
            restored_prefetched_pool = restored_runtime.get("prefetched_pool")
            ledger.recover_to_receipt(
                ExposureLedgerReceipt.from_payload(restored_runtime["ledger_receipt"])
            )
        if config.experiment_family == MAIN500M_EXPERIMENT_FAMILY:
            continuation_record = None
            if pending_review is not None and completed_steps < args.max_steps:
                if args.mode != "main":
                    raise RuntimeError(
                        "a paused ASPD v3.1 checkpoint can continue only in main mode"
                    )
                if args.continuation_authorization is None:
                    raise RuntimeError(
                        "paused ASPD v3.1 checkpoint requires --continuation-authorization"
                    )
                if rank == 0:
                    continuation_record = load_continuation_authorization(
                        args.continuation_authorization,
                        authorization_root=Path(
                            config.runtime.continuation_authorization_root
                        ),
                        pending_review=pending_review,
                        execution_contract_fingerprint=(
                            config.execution_contract_fingerprint()
                        ),
                        checkpoint_manifest_sha256=_sha256(latest / "manifest.json"),
                        approved_through_exposures=next_review_exposure(
                            int(pending_review["accepted_exposures"]),
                            config.monitoring.staged_milestone_exposures,
                        ),
                    )
                continuation_record = _broadcast_object(continuation_record)
                consumed_continuation_authorizations.append(continuation_record)
                if rank == 0:
                    _record_continuation_authorization(
                        args.run_dir / "continuation_authorizations.jsonl",
                        continuation_record,
                    )
                pending_review = None
            elif args.continuation_authorization is not None:
                raise ValueError(
                    "continuation authorization supplied without a pending checkpoint review"
                )
    elif args.continuation_authorization is not None:
        raise ValueError("continuation authorization cannot be used for a fresh run")
    elif rank == 0:
        if initial_runtime_path.is_file():
            initial = json.loads(initial_runtime_path.read_text(encoding="utf-8"))
            if initial["config_fingerprint"] != config.fingerprint():
                raise ValueError("initial runtime config fingerprint changed")
            if controlled is not None and initial.get("controlled_contract_fingerprint") != (
                controlled.fingerprint()
            ):
                raise ValueError("initial controlled retrain contract changed")
            stream.load_state_dict(initial["stream"])
            restored_prefetched_pool = initial.get("prefetched_pool")
            ledger.recover_to_receipt(ExposureLedgerReceipt.from_payload(initial["ledger_receipt"]))
        else:
            initial_stream = stream.state_dict()
            if controlled is not None:
                if source_runtime is None or source_runtime.get("prefetched_pool") is not None:
                    raise ValueError(
                        "controlled source checkpoint must expose an unbuffered exact stream state"
                    )
                initial_stream = source_runtime["stream"]
                stream.load_state_dict(initial_stream)
            write_json_immutable(
                initial_runtime_path,
                {
                    "schema_version": 1,
                    "config_fingerprint": config.fingerprint(),
                    "controlled_contract_fingerprint": (
                        None if controlled is None else controlled.fingerprint()
                    ),
                    "source_completed_steps": (
                        None if controlled is None else controlled.source_checkpoint_step
                    ),
                    "source_accepted_exposures": (
                        None if controlled is None else controlled.source_accepted_exposures
                    ),
                    "stream": initial_stream,
                    "prefetched_pool": None,
                    "ledger_receipt": ledger.checkpoint_receipt(
                        expected_optimizer_steps=0
                    ).payload(),
                },
            )
    completed_steps = int(_broadcast_object(completed_steps))
    dist.barrier()
    prefetcher = None
    if rank == 0:
        prefetcher = _PoolPrefetcher(
            stream,
            pool_anchors=config.architecture.pool_tokens,
            buffered_payload=restored_prefetched_pool,
        )
        prefetcher.start()

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    metrics_path = args.run_dir / "metrics.jsonl"
    checkpoint_seconds = 0.0
    checkpoint_save_seconds = 0.0
    checkpoint_reload_seconds = 0.0
    checkpoint_restore_probe = "NOT_RUN"
    checkpoint_path: Path | None = None
    checkpoint_manifest = None
    resumed = False
    health_pause_requested = False
    milestone_pause_requested = False
    diagnostic_steps = {1, 2, 10, 50, 128, 256, 300, 500, 800, 1000, 1221}
    diagnostic_parameter_steps = set(
        getattr(config.runtime, "diagnostic_parameter_steps", ())
    )
    initial_decoder_norms_by_rank = None
    if args.strategy == "model_owned":
        initial_decoder_norms_by_rank = [None] * world
        dist.all_gather_object(
            initial_decoder_norms_by_rank, model.activation_decoder_norm_report()
        )
    started = time.perf_counter()
    telemetry_path = args.run_dir / "gpu_telemetry.csv"
    telemetry = _start_gpu_telemetry(telemetry_path) if rank == 0 else None
    profiler = None
    if args.torch_profiler:
        profiler = torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        )
        profiler.__enter__()

    while completed_steps < args.max_steps:
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        stage: dict[str, float] = {}
        step_anchor_ids: list[str] = []
        pool_losses: list[dict[str, float]] = []
        milestone_diagnostics_required = bool(
            config.experiment_family == MAIN500M_EXPERIMENT_FAMILY
            and milestone_diagnostic_due(
                (completed_steps + 1) * 8192,
                config.monitoring.staged_milestone_exposures,
            )
        )
        if diagnostic_parameter_steps:
            collect_step_diagnostics = (
                completed_steps in diagnostic_parameter_steps
                or milestone_diagnostics_required
            )
            diagnostic_parameter_step = completed_steps if collect_step_diagnostics else None
        elif getattr(config, "monitoring", None) is not None:
            interval = int(config.monitoring.monitor_every_steps)
            collect_step_diagnostics = (
                completed_steps == 0
                or completed_steps % interval == 0
                or milestone_diagnostics_required
            )
            diagnostic_parameter_step = completed_steps if collect_step_diagnostics else None
        else:
            collect_step_diagnostics = completed_steps + 1 in diagnostic_steps
            diagnostic_parameter_step = (
                completed_steps if collect_step_diagnostics else None
            )
        for pool_index in range(config.architecture.pools_per_step):
            with _timed(args.profile_stages, stage, "data_alignment_and_broadcast"):
                payload = None
                if rank == 0:
                    final_planned_pool = (
                        completed_steps + 1 >= args.max_steps
                        and pool_index + 1 >= config.architecture.pools_per_step
                    )
                    payload = prefetcher.next_payload(schedule_next=not final_planned_pool)
                    pool = AlignedAnchorPool.from_payload(payload)
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
                    model_order=model_order,
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
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        if controlled is None:
                            losses = _model_owned_backward(
                                model,
                                capture,
                                config,
                                pool_scale=1 / 4,
                                activation_backward_chunk=activation_backward_chunk,
                                matrix_backward_chunk=matrix_backward_chunk,
                                auxk_may_be_active=(
                                    completed_steps * config.architecture.pools_per_step
                                    + pool_index
                                    + 1
                                    >= ORIGINAL_ASPD_N_POOLS_TO_DEAD
                                ),
                                collect_diagnostics=collect_step_diagnostics,
                            )
                        else:
                            losses = _controlled_activation_backward(
                                model,
                                capture,
                                config,
                                pool_scale=1 / 4,
                                activation_backward_chunk=activation_backward_chunk,
                            )
                else:
                    losses = _fsdp_backward(model, capture, pool_index)
            pool_losses.append(losses)
            del capture, pool

        with _timed(args.profile_stages, stage, "gradient_sync_validation_normalization"):
            if args.strategy == "model_owned":
                if controlled is None:
                    model.all_reduce_shared_gradients_()
                else:
                    reduce_controlled_shared_gradients_(
                        model,
                        arm=controlled.arm,
                        qwen_smol_group=qwen_smol_group,
                    )
            gradient_report = _finite_gradients(model)
            reports = [None] * world
            dist.all_gather_object(reports, gradient_report)
            if any(report["status"] != "PASS" for report in reports):
                raise FloatingPointError(f"missing/nonfinite gradients: {reports}")
            gradient_norms_by_rank = None
            if args.strategy == "model_owned" and collect_step_diagnostics:
                local_gradient_norms = _gradient_norm_report(model, rank=rank)
                gradient_norms_by_rank = [None] * world
                dist.all_gather_object(gradient_norms_by_rank, local_gradient_norms)
            if args.strategy == "model_owned":
                if controlled is None:
                    model.project_decoder_gradients_()
                else:
                    for bank in model.activation_banks.values():
                        bank.project_decoder_gradients_()
            else:
                model.normalize_decoders_(project_gradients=True)
        local_scale_gradient = None
        if (
            args.strategy == "model_owned"
            and model.reconstruction_log_scale is not None
            and model.reconstruction_log_scale.grad is not None
        ):
            local_scale_gradient = model.reconstruction_log_scale.grad.detach().cpu().tolist()
        scale_gradients = [None] * world
        dist.all_gather_object(scale_gradients, local_scale_gradient)
        global_loss_names = ("total", "activation", "auxk", "internal")
        step_loss_tensor = torch.tensor(
            [sum(value[name] for value in pool_losses) / 4 for name in global_loss_names],
            dtype=torch.float64,
            device=device,
        )
        detailed_losses: dict[str, float] = {}
        if args.strategy == "model_owned":
            dist.all_reduce(step_loss_tensor, op=dist.ReduceOp.SUM)
            local_details = {
                name: sum(pool[name] for pool in pool_losses) / 4
                for name in pool_losses[0]
                if name.startswith(("loss/", "diagnostic/"))
            }
            gathered_details = [None] * world
            dist.all_gather_object(gathered_details, local_details)
            for rank_details in gathered_details:
                overlap = set(detailed_losses) & set(rank_details)
                if overlap:
                    raise RuntimeError(f"model-owned detailed loss keys overlap: {overlap}")
                detailed_losses.update(rank_details)
        step_loss_values = step_loss_tensor.tolist()
        step_loss = dict(zip(global_loss_names, step_loss_values, strict=True))
        if not all(math.isfinite(value) for value in step_loss.values()):
            raise FloatingPointError(f"non-finite global ASPD objective: {step_loss}")
        with _timed(args.profile_stages, stage, "adam_and_post_normalization"):
            optimizer.step()
            if controlled is None:
                model.normalize_decoders_(project_gradients=False)
            else:
                for bank in model.activation_banks.values():
                    bank.normalize_decoders_(project_gradients=False)
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
        local_scales = (
            model.reconstruction_scale_report() if args.strategy == "model_owned" else None
        )
        scales_by_rank = [None] * world
        dist.all_gather_object(scales_by_rank, local_scales)
        decoder_norms_by_rank = None
        if args.strategy == "model_owned" and collect_step_diagnostics:
            decoder_norms_by_rank = [None] * world
            dist.all_gather_object(
                decoder_norms_by_rank, model.activation_decoder_norm_report()
            )
        health = None
        if rank == 0:
            ordered_anchor_digest = hashlib.sha256()
            for anchor_id in step_anchor_ids:
                encoded = anchor_id.encode("utf-8")
                ordered_anchor_digest.update(len(encoded).to_bytes(8, "big"))
                ordered_anchor_digest.update(encoded)
            metrics_row = {
                    "step": completed_steps,
                    "warmup": completed_steps <= args.warmup_steps,
                    "seconds": seconds,
                    "anchors_per_second": 8192 / seconds,
                    "accepted_exposures": counter_payload["accepted"],
                    "unique_anchors": counter_payload["unique"],
                    "repeated_exposures": counter_payload["repeated"],
                    "ordered_anchor_ids_sha256": ordered_anchor_digest.hexdigest(),
                    "loss": {
                        **step_loss,
                        **detailed_losses,
                    },
                    "stage_seconds": stage,
                    "gpu_memory": memories,
                    "reconstruction_scales_by_rank": scales_by_rank,
                    "reconstruction_log_scale_gradients_by_rank": scale_gradients,
                    "activation_decoder_norms_by_rank": decoder_norms_by_rank,
                    "gradient_norms_by_rank": gradient_norms_by_rank,
                    "storage_free_bytes": shutil.disk_usage(args.run_dir).free,
                    "diagnostic_parameter_step": diagnostic_parameter_step,
                }
            if getattr(config, "monitoring", None) is not None and collect_step_diagnostics:
                if health_monitor is None:
                    raise RuntimeError(
                        "ASPD v3.1 diagnostic step has no stateful health monitor"
                    )
                health = health_monitor.observe(metrics_row)
                metrics_row["health"] = health
            _append_jsonl(metrics_path, metrics_row)
        health = _broadcast_object(health)
        if health is not None and health["status"] == "ABORT":
            raise RuntimeError(f"ASPD v3.1 health monitor requested abort: {health['events']}")
        health_pause_requested = bool(
            health is not None and health["action"] == "CHECKPOINT_AND_PAUSE"
        )
        milestone_pause_requested = bool(
            config.experiment_family == MAIN500M_EXPERIMENT_FAMILY
            and args.mode == "main"
            and milestone_pause_due(
                counter_payload["accepted"],
                config.monitoring.staged_milestone_exposures,
            )
        )
        pause_reasons = [
            reason
            for reason, requested in (
                ("HEALTH", health_pause_requested),
                ("MILESTONE", milestone_pause_requested),
            )
            if requested
        ]
        if pause_reasons:
            pending_review = {
                "policy_version": config.monitoring.milestone_policy_version,
                "reasons": pause_reasons,
                "accepted_exposures": int(counter_payload["accepted"]),
            }

        preflight_reload_due = args.mode == "preflight" and completed_steps == args.max_steps - 1
        checkpoint_due = not args.skip_checkpoint and (
            health_pause_requested
            or milestone_pause_requested
            or preflight_reload_due
            or (
                args.mode in {"main", "controlled", "pilot"}
                and (
                    completed_steps % config.runtime.checkpoint_every_steps == 0
                    or completed_steps == args.max_steps
                    or stop_requested
                )
            )
        )
        if checkpoint_due:
            torch.cuda.synchronize()
            checkpoint_started = time.perf_counter()
            receipt = None
            stream_state = None
            prefetched_pool = None
            if rank == 0:
                prefetched_pool = prefetcher.checkpoint_payload()
                receipt = ledger.checkpoint_receipt(
                    expected_optimizer_steps=completed_steps
                ).payload()
                stream_state = stream.state_dict()
            receipt = _broadcast_object(receipt)
            health_monitor_state = _broadcast_object(
                health_monitor.state_dict() if health_monitor is not None else None
            )
            runtime_state = {
                "completed_steps": completed_steps,
                "accepted_exposures": completed_steps * 8192,
                "source_completed_steps": (
                    None if controlled is None else controlled.source_checkpoint_step
                ),
                "source_accepted_exposures": (
                    None if controlled is None else controlled.source_accepted_exposures
                ),
                "controlled_contract_fingerprint": (
                    None if controlled is None else controlled.fingerprint()
                ),
                "ledger_receipt": receipt,
                "stream": stream_state,
                "prefetched_pool": prefetched_pool,
                "health_monitor": health_monitor_state,
                "pending_review": pending_review,
                "consumed_continuation_authorizations": (
                    consumed_continuation_authorizations
                ),
            }
            save_started = time.perf_counter()
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
                    experiment_metadata={
                        "experiment_family": config.experiment_family,
                        "model_order": list(model_order),
                        "activation_target_normalization": getattr(
                            config.objective, "activation_target_normalization", "none"
                        ),
                        "activation_target_epsilon": float(
                            getattr(config.objective, "activation_target_epsilon", 1e-6)
                        ),
                        "activation_reconstruction_scale": getattr(
                            config.objective, "activation_reconstruction_scale", "none"
                        ),
                        "oracle_norm_evaluation": bool(
                            getattr(config.objective, "oracle_norm_evaluation", False)
                        ),
                        "factorial_arm_id": getattr(config, "arm_id", None),
                        "encoder_mixer": getattr(
                            config.architecture, "encoder_mixer", "hierarchical_transformer"
                        ),
                        "auxk_score_source": getattr(
                            config.objective, "auxk_score_source", "post_relu"
                        ),
                        "auxk_softplus_temperature": getattr(
                            config.objective, "auxk_softplus_temperature", None
                        ),
                    },
                    recovery_databases=(
                        {
                            "exposure_ledger.sqlite": ledger.path,
                            "stream_dedup.sqlite": stream.dedup.path,
                        }
                        if rank == 0
                        else None
                    ),
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
            checkpoint_save_seconds = time.perf_counter() - save_started
            checkpoint_path = args.run_dir / "checkpoints" / f"step-{completed_steps:09d}"
            if preflight_reload_due:
                probe_parameter = next(model.parameters())
                probe_parameter_value = probe_parameter.detach().reshape(-1)[0].clone()
                optimizer_state = optimizer.state.get(probe_parameter, {})
                optimizer_probes = [
                    (key, value)
                    for key, value in optimizer_state.items()
                    if isinstance(value, torch.Tensor) and value.numel() > 1
                ]
                if not optimizer_probes:
                    raise RuntimeError("preflight found no materialized Adam tensor to restore")
                probe_optimizer_key, probe_optimizer = optimizer_probes[0]
                probe_optimizer_value = probe_optimizer.detach().reshape(-1)[0].clone()
                with torch.no_grad():
                    probe_parameter.reshape(-1)[0].add_(1)
                    probe_optimizer.reshape(-1)[0].add_(1)
                reload_started = time.perf_counter()
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
                    if health_monitor is not None:
                        health_state = restored_runtime.get("health_monitor")
                        if not isinstance(health_state, dict):
                            raise RuntimeError(
                                "preflight checkpoint omitted sustained health-monitor state"
                            )
                        health_monitor.load_state_dict(health_state)
                    stream.load_state_dict(restored_runtime["stream"])
                    prefetcher.restore(restored_runtime.get("prefetched_pool"))
                    ledger.recover_to_receipt(
                        ExposureLedgerReceipt.from_payload(restored_runtime["ledger_receipt"])
                    )
                dist.barrier()
                checkpoint_reload_seconds = time.perf_counter() - reload_started
                if not torch.equal(
                    probe_parameter.detach().reshape(-1)[0], probe_parameter_value
                ) or not torch.equal(
                    optimizer.state[probe_parameter][probe_optimizer_key].detach().reshape(-1)[0],
                    probe_optimizer_value,
                ):
                    raise RuntimeError(
                        "DCP reload did not restore the perturbed parameter/Adam probe"
                    )
                checkpoint_restore_probe = "PASS"
                resumed = True
            elif args.mode in {"main", "controlled", "pilot"}:
                checkpoint_manager.prune_after_verified_replacement()
            torch.cuda.synchronize()
            checkpoint_seconds = time.perf_counter() - checkpoint_started
        if stop_requested or health_pause_requested or milestone_pause_requested:
            break
        if profiler is not None:
            profiler.step()

    if profiler is not None:
        profiler.__exit__(None, None, None)
        profiler.export_chrome_trace(str(args.run_dir / f"torch_profile_rank{rank:02d}.json"))

    if telemetry is not None:
        _stop_gpu_telemetry(*telemetry)

    gate_statistics = None
    final_decoder_norms_by_rank = None
    if args.strategy == "model_owned":
        local_gate_statistics: dict[str, dict[str, float | int]] = {}
        for site in ("pre", "mid"):
            utilization = model.gates[site].utilization()
            local_gate_statistics[site] = {
                "accepted_anchor_count": int(utilization["accepted_anchor_count"]),
                "used_feature_count": int(utilization["used_feature_count"]),
                "dead_feature_count": int(utilization["dead_feature_count"]),
                "threshold": float(utilization["threshold"].float().item()),
            }
        gate_statistics = [None] * world
        dist.all_gather_object(gate_statistics, local_gate_statistics)
        final_decoder_norms_by_rank = [None] * world
        dist.all_gather_object(
            final_decoder_norms_by_rank, model.activation_decoder_norm_report()
        )

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
            "status": (
                "PAUSED_HEALTH_AND_MILESTONE_REVIEW"
                if health_pause_requested and milestone_pause_requested
                else "PAUSED_HEALTH_REVIEW"
                if health_pause_requested
                else "PAUSED_MILESTONE_REVIEW"
                if milestone_pause_requested
                else "PASS"
            ),
            "mode": args.mode,
            "strategy": args.strategy,
            "git_sha": git_sha,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "config_fingerprint": config.fingerprint(),
            "execution_contract_fingerprint": (
                config.execution_contract_fingerprint()
                if config.experiment_family == MAIN500M_EXPERIMENT_FAMILY
                else None
            ),
            "config_file_sha256": config_sha,
            "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
            "alignment_manifest_sha256": alignment_manifest["manifest_sha256"],
            "world_size": world,
            "logical_target_count": logical_target_count,
            "model_revisions": {model.key: model.revision for model in config.models},
            "source_correctness": source_smokes,
            "runtime": runtime_payload,
            "runtime_fingerprint": runtime_fingerprint,
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
            "checkpoint_seconds_save": checkpoint_save_seconds,
            "checkpoint_seconds_reload": checkpoint_reload_seconds,
            "checkpoint_parameter_and_adam_restore_probe": checkpoint_restore_probe,
            "checkpoint_manifest": checkpoint_manifest,
            "dcp_save_reload_resume_update": (
                "PASS" if resumed and completed_steps == args.max_steps else "NOT_RUN"
            ),
            "peak_gpu_memory_by_rank": rows[-1]["gpu_memory"],
            "stage_seconds_mean": {
                name: sum(row["stage_seconds"].get(name, 0.0) for row in measured) / len(measured)
                for name in sorted({name for row in measured for name in row["stage_seconds"]})
            },
            "gpu_telemetry": _summarize_gpu_telemetry(telemetry_path),
            "gate_statistics_by_rank": gate_statistics,
            "initial_common_state_by_rank": initial_common_states,
            "initial_activation_decoder_norms_by_rank": initial_decoder_norms_by_rank,
            "final_reconstruction_scales_by_rank": rows[-1]["reconstruction_scales_by_rank"],
            "final_activation_decoder_norms_by_rank": final_decoder_norms_by_rank,
            "pending_review": pending_review,
            "consumed_continuation_authorizations": (
                consumed_continuation_authorizations
            ),
        }
        if health_pause_requested or milestone_pause_requested:
            report["scientific_review"] = milestone_scientific_summary(rows[-1])
            report["recovery_checkpoint"] = {
                "path": None if checkpoint_path is None else str(checkpoint_path),
                "manifest_sha256": (
                    None
                    if checkpoint_path is None
                    else _sha256(checkpoint_path / "manifest.json")
                ),
                "status": (
                    None if checkpoint_manifest is None else checkpoint_manifest.get("status")
                ),
            }
        if controlled is not None:
            report["controlled_retrain"] = {
                "arm": controlled.arm,
                "contract_file_sha256": controlled_sha,
                "contract_fingerprint": controlled.fingerprint(),
                "source_checkpoint": controlled.source_checkpoint,
                "source_checkpoint_manifest_sha256": controlled.source_checkpoint_manifest_sha256,
                "source_checkpoint_step": controlled.source_checkpoint_step,
                "source_accepted_exposures": controlled.source_accepted_exposures,
                "new_accepted_exposures": completed_steps * 8192,
                "total_exposures_including_source": (
                    controlled.source_accepted_exposures + completed_steps * 8192
                ),
                "rank_one_matrix_factors": "FROZEN",
                "optimizer_state": controlled.optimizer_state,
            }
        report_path = (
            args.run_dir / f"continuation_review_step_{completed_steps:09d}.json"
            if health_pause_requested or milestone_pause_requested
            else args.run_dir / "benchmark_report.json"
        )
        write_json_immutable(report_path, report)
        if (
            args.mode == "preflight"
            and not health_pause_requested
            and not milestone_pause_requested
        ):
            expected_preflight = Path(config.runtime.preflight_evidence)
            if args.run_dir != expected_preflight.parent:
                raise ValueError(
                    "final preflight run directory must equal the configured evidence directory"
                )
            write_json_immutable(expected_preflight, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        prefetcher.close()
        stream.close()
        ledger.close()
    dist.barrier()
    dist.destroy_process_group()
    if health_pause_requested:
        raise SystemExit(86)
    if milestone_pause_requested:
        raise SystemExit(87)
    if stop_requested:
        raise SystemExit(99)


if __name__ == "__main__":
    main()
