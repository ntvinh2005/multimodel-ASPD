"""Single-GPU training loop for cached multi-model ASPD experiments."""

from __future__ import annotations

import json
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.model import MultiModelASPD

# Training example in comments uses gradient accumulation A=2, batch size B=4, and T=256.
# One optimizer step therefore represents A*B*T=2,048 aligned token positions.


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),  # type: ignore[union-attr]
        "valid_tokens": batch["valid_tokens"].to(device, non_blocking=True),  # type: ignore[union-attr]
        "R": [tensor.to(device, non_blocking=True) for tensor in batch["R"]],  # type: ignore[union-attr]
        "X": [
            {name: tensor.to(device, non_blocking=True) for name, tensor in matrices.items()}
            for matrices in batch["X"]  # type: ignore[union-attr]
        ],
    }


def build_model_from_cache(
    cfg: MultiModelExperimentConfig, cache: PairedActivationCache
) -> MultiModelASPD:
    # Load each frozen W_j^(n); trainable P_j,c^(n) is built with matching d_in/d_out below.
    weights = cache.load_weights()
    model = MultiModelASPD(
        model_names=[model.name for model in cfg.models],
        activation_dims=[manifest["activation_dim"] for manifest in cache.model_manifests],
        target_weights=weights,
        encoder_cfg=cfg.encoder,
        sparsity_cfg=cfg.sparsity,
        objective_cfg=cfg.objective,
    )
    if cfg.training.parameter_dtype == "bfloat16":
        # Example FP32 U,V,d -> BF16 halves parameter bytes; FVU still accumulates in FP32.
        model = model.to(dtype=torch.bfloat16)
    return model


def _autocast_context(cfg: MultiModelExperimentConfig, device: torch.device):
    if cfg.training.autocast == "none" or device.type != "cuda":
        return nullcontext()
    # Choose compute dtype inside matrix operations; parameters may remain FP32.
    dtype = torch.bfloat16 if cfg.training.autocast == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def validate(
    model: MultiModelASPD,
    cache: PairedActivationCache,
    cfg: MultiModelExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in cache.iter_batches(
        "validation", cfg.training.batch_size_sequences, shuffle=False
    ):
        batch = _move_batch(batch, device)
        with _autocast_context(cfg, device):
            # Evaluate the same L([C]), optional gamma L(S), and AuxK without aging dead clocks.
            result = model(batch, update_dead_tracker=False)
        for name, value in result.items():
            # Sum a metric across validation batches; e.g. FVU .4+.6=1.0 after two batches.
            totals[name] = totals.get(name, 0.0) + float(value.detach().float().cpu())
        # Count batches for the arithmetic mean.
        count += 1
        if count >= cfg.training.validation_batches:
            break
    model.train()
    # Mean metric; example total 1.0/count 2 = .5 validation FVU.
    return {f"validation/{name}": value / max(count, 1) for name, value in totals.items()}


def _save_checkpoint(
    output_dir: Path,
    model: MultiModelASPD,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    keep_last: int,
) -> Path:
    checkpoint = output_dir / f"checkpoint_{step:08d}.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    state_model = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model": state_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    temporary.replace(checkpoint)
    (output_dir / "latest_checkpoint.txt").write_text(checkpoint.name + "\n", encoding="utf-8")
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    for old_checkpoint in checkpoints[:-keep_last]:
        old_checkpoint.unlink()
    return checkpoint


def _load_checkpoint(
    path: Path, model: MultiModelASPD, optimizer: torch.optim.Optimizer
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    state_model = getattr(model, "_orig_mod", model)
    state_model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"]), int(state["epoch"])


def train(cfg: MultiModelExperimentConfig, device_name: str = "cuda") -> Path:
    """Train the configured decomposition.  LLM weights are not loaded; only caches are read."""

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.training.seed)

    cache = PairedActivationCache(cfg)
    model = build_model_from_cache(cfg, cache).to(device)
    if cfg.training.compile:
        model = torch.compile(model)  # type: ignore[assignment]
    # AdamW updates only trainable encoder/decoder/component parameters; W_j^(n) is frozen.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=cfg.training.betas,
        weight_decay=cfg.training.weight_decay,
        fused=device.type == "cuda",
    )

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "model_count_N": len(cfg.models),
        "latent_count_C": cfg.sparsity.n_features,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    step = 0
    epoch = 0
    if cfg.training.resume:
        resume_path = Path(cfg.training.resume)
        if cfg.training.resume == "auto":
            resume_path = output_dir / (output_dir / "latest_checkpoint.txt").read_text().strip()
        step, epoch = _load_checkpoint(resume_path, model, optimizer)

    wandb_run = None
    if cfg.training.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=cfg.training.wandb_project,
            name=cfg.name,
            config=cfg.model_dump(mode="json"),
        )

    metrics_path = output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    model.train()
    started = time.perf_counter()
    # Recover consumed microbatches on resume. Example step=10,accum=4 -> micro_step=40.
    micro_step = step * cfg.training.gradient_accumulation_steps
    while step < cfg.training.max_steps:
        for raw_batch in cache.iter_batches(
            "train", cfg.training.batch_size_sequences, shuffle=True, epoch=epoch
        ):
            batch = _move_batch(raw_batch, device)
            with _autocast_context(cfg, device):
                # result["loss"] is L([C])+gamma L(S)+lambda_aux L_aux over all n.
                result = model(batch)
                # Divide before backward so A microbatches sum to one optimizer-step gradient.
                # Example raw losses 2 and 4 with A=2 contribute gradients of 1 and 2.
                loss: Tensor = result["loss"] / cfg.training.gradient_accumulation_steps
            # Accumulate gradients of W_e,D,U,V; frozen W_j^(n) are buffers and receive none.
            loss.backward()
            # Count this microbatch.
            micro_step += 1
            # Update only after A microbatches; e.g. A=4 skips at micro_step 1,2,3.
            if micro_step % cfg.training.gradient_accumulation_steps != 0:
                continue
            if cfg.training.max_grad_norm is not None:
                # If ||grad|| exceeds threshold m, rescale all grads by m/||grad||.
                # Example norm 2,m=1 multiplies every gradient by 0.5.
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            # AdamW applies one parameter update to encoder, d_c^(n),u_j,c^(n),v_j,c^(n).
            optimizer.step()
            # Remove previous gradients before accumulating the next optimizer step.
            optimizer.zero_grad(set_to_none=True)
            # step counts optimizer updates, not microbatches.
            step += 1

            if step % cfg.training.log_every == 0 or step == 1:
                # Wall time since this process began/resumed.
                elapsed = time.perf_counter() - started
                row: dict[str, Any] = {
                    "step": step,
                    "epoch": epoch,
                    "elapsed_seconds": elapsed,
                    # Approximate processed tokens = step*A*B*T.
                    # Example step=10,A=2,B=4,T=256 -> 20,480 tokens / elapsed seconds.
                    "tokens_per_second": (
                        step
                        * cfg.training.gradient_accumulation_steps
                        * cfg.training.batch_size_sequences
                        * cfg.data.sequence_length
                        / max(elapsed, 1e-9)
                    ),
                    **{
                        f"train/{name}": float(value.detach().float().cpu())
                        for name, value in result.items()
                    },
                }
                if device.type == "cuda":
                    # Convert bytes to GiB; example 193,273,528,320 / 2^30 = 180 GiB.
                    row["cuda/max_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                if wandb_run is not None:
                    wandb_run.log(row, step=step)

            if step % cfg.training.validate_every == 0:
                validation = {"step": step, **validate(model, cache, cfg, device)}
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(validation, sort_keys=True) + "\n")
                if wandb_run is not None:
                    wandb_run.log(validation, step=step)

            if step % cfg.training.save_every == 0:
                _save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    step,
                    epoch,
                    cfg.training.keep_last_checkpoints,
                )
            if step >= cfg.training.max_steps:
                break
        # One epoch is one traversal of cached token shards; next epoch reshuffles deterministically.
        epoch += 1

    final_path = _save_checkpoint(
        output_dir,
        model,
        optimizer,
        step,
        epoch,
        cfg.training.keep_last_checkpoints,
    )
    if wandb_run is not None:
        wandb_run.finish()
    return final_path
