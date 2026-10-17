"""Canonical BatchTopK Crosscoder baseline over the shared multimodel cache.

The equations and defaults follow ``science-of-finetuning/crosscoder_learning`` at
commit ``b2363ed54590074ef4667ca6152094800a65f9aa``.  Only the model/trainer core is
implemented locally because that revision requires nnsight<0.5 and transformers<4.54,
which conflict with this repository's runtime.  Dataset loading, token alignment,
activation caching, context retention, and tokenizer decoding are reused unchanged.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator
from torch import Tensor, nn

from aspd.multimodel.analysis import _TopActivationAccumulator, decode_activation_examples
from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.config import MultiModelExperimentConfig, load_config_mapping

REFERENCE_REPOSITORY = "https://github.com/science-of-finetuning/crosscoder_learning"
REFERENCE_COMMIT = "b2363ed54590074ef4667ca6152094800a65f9aa"
REFERENCE_EXPERIMENT = "https://github.com/science-of-finetuning/sparsity-artifacts-crosscoders"
REFERENCE_EXPERIMENT_COMMIT = "d67ec6a6175965c470d104d9be9dedf6c95fdffd"

# The gated Google tokenizer and this public mirror were compared byte-for-byte locally.
# This mapping is only decoding provenance; model weights are never loaded from the mirror.
PUBLISHED_TOKENIZER_PROVENANCE: dict[str, dict[str, str]] = {
    "unsloth/gemma-2-2b": {
        "role": "tokenizer-only mirror for decoding published token IDs",
        "tokenizer_json_sha256": (
            "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922"
        ),
        "tokenizer_model_sha256": (
            "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"
        ),
        "verification": "byte-identical to google/gemma-2-2b tokenizer files",
    }
}


class CrosscoderMethodSpec(BaseModel):
    """BatchTopK settings, kept separate from ASPD's own training configuration."""

    output_dir: str
    n_features: PositiveInt
    k: PositiveInt = 100
    batch_size_tokens: PositiveInt = 2048
    epochs: PositiveInt = 2
    max_steps: PositiveInt | None = None
    learning_rate: PositiveFloat = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = Field(default=1000, ge=0)
    auxk_coefficient: float = Field(default=1 / 32, ge=0.0)
    dead_after_tokens: PositiveInt = 10_000_000
    threshold_beta: float = Field(default=0.999, ge=0.0, lt=1.0)
    threshold_start_step: int = Field(default=1000, ge=0)
    norm_init_scale: PositiveFloat = 1.0
    same_init_for_all_models: bool = True
    init_decoder_from_encoder: bool = True
    code_normalization: Literal["crosscoder"] = "crosscoder"
    activation_preprocessing: Literal["raw", "aspd_rms"] = "raw"
    seed: int = 42
    log_every: PositiveInt = 50
    validate_every: PositiveInt = 20_000
    save_every: PositiveInt = 20_000
    validation_batches: PositiveInt | None = None
    keep_last_checkpoints: PositiveInt = 2
    resume: str | None = None
    top_examples_per_category: PositiveInt = 16
    reference_checkpoint_repository: str | None = None
    reference_checkpoint_revision: str | None = None

    @model_validator(mode="after")
    def validate_dimensions(self) -> CrosscoderMethodSpec:
        if self.k > self.n_features:
            raise ValueError("crosscoder.k cannot exceed crosscoder.n_features")
        if self.max_steps is not None and self.warmup_steps >= self.max_steps:
            raise ValueError("crosscoder.warmup_steps must be smaller than max_steps")
        return self


class CrosscoderExperimentConfig(MultiModelExperimentConfig):
    """Existing multimodel data/cache schema plus one isolated baseline section."""

    crosscoder: CrosscoderMethodSpec

    @model_validator(mode="after")
    def validate_crosscoder_inputs(self) -> CrosscoderExperimentConfig:
        if len(self.models) != 2:
            raise ValueError("the published model-diffing baseline requires exactly two models")
        if self.crosscoder.batch_size_tokens % self.data.sequence_length:
            raise ValueError("crosscoder.batch_size_tokens must be divisible by data.sequence_length")
        steps_per_epoch = math.ceil(
            self.data.train_tokens / self.crosscoder.batch_size_tokens
        )
        available_steps = steps_per_epoch * self.crosscoder.epochs
        if (
            self.crosscoder.max_steps is not None
            and self.crosscoder.max_steps > available_steps
        ):
            raise ValueError(
                "crosscoder.max_steps exceeds the steps available from crosscoder.epochs "
                f"({self.crosscoder.max_steps} requested, {available_steps} available)"
            )
        return self


def load_crosscoder_config(path: str | Path) -> CrosscoderExperimentConfig:
    return CrosscoderExperimentConfig.model_validate(load_config_mapping(path))


class BatchTopKCrosscoder(nn.Module):
    """Two-model shared-code Crosscoder with reference BatchTopK selection.

    For x[b,n,d], the encoder computes ReLU(sum_n x[b,n] @ W_enc[n] + b_enc).
    Selection ranks activations after multiplying by the sum of the two model-specific
    decoder norms, while reconstruction uses the unscaled selected activations.
    """

    def __init__(
        self,
        activation_dim: int,
        n_features: int,
        n_models: int,
        k: int,
        *,
        same_init_for_all_models: bool = True,
        norm_init_scale: float = 1.0,
        init_decoder_from_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.activation_dim = activation_dim
        self.n_features = n_features
        self.n_models = n_models

        if same_init_for_all_models:
            weight = nn.init.kaiming_uniform_(torch.empty(activation_dim, n_features))
            weight = weight.repeat(n_models, 1, 1)
        else:
            weight = nn.init.kaiming_uniform_(
                torch.empty(n_models, activation_dim, n_features)
            )
        weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        weight = weight * norm_init_scale
        self.encoder_weight = nn.Parameter(weight)
        self.encoder_bias = nn.Parameter(torch.zeros(n_features))

        if init_decoder_from_encoder:
            decoder = weight.transpose(1, 2).clone()
        else:
            decoder = nn.init.kaiming_uniform_(
                torch.empty(n_models, n_features, activation_dim)
            )
            decoder = decoder / decoder.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            decoder = decoder * norm_init_scale
        self.decoder_weight = nn.Parameter(decoder)
        self.decoder_bias = nn.Parameter(torch.zeros(n_models, activation_dim))
        self.register_buffer("k", torch.tensor(k, dtype=torch.int64))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))

    def code_scale(self) -> Tensor:
        """Reference CROSSCODER normalization: sum_n ||d_c^(n)||_2."""

        return self.decoder_weight.norm(dim=-1).sum(dim=0)

    def post_relu(self, x: Tensor) -> Tensor:
        return torch.relu(torch.einsum("bnd,ndc->bc", x, self.encoder_weight) + self.encoder_bias)

    def encode(
        self, x: Tensor, *, use_threshold: bool, return_details: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
        post_relu = self.post_relu(x)
        scale = self.code_scale()
        scaled = post_relu * scale
        if use_threshold and self.threshold.item() >= 0:
            features = post_relu * (scaled > self.threshold)
        else:
            # Reference BatchTopK selects exactly batch_size*k entries globally.
            flat_scaled = scaled.flatten()
            take = min(int(self.k.item()) * x.shape[0], flat_scaled.numel())
            indices = flat_scaled.topk(take, sorted=False).indices
            selected = post_relu.flatten()[indices]
            features = torch.zeros_like(flat_scaled).scatter_(0, indices, selected)
            features = features.reshape_as(post_relu)
        if return_details:
            active = features.sum(dim=0) > 0
            return features, scaled, active, post_relu
        return features

    def decode(self, features: Tensor, *, add_bias: bool = True) -> Tensor:
        result = torch.einsum("bc,ncd->bnd", features, self.decoder_weight)
        return result + self.decoder_bias if add_bias else result

    def forward(self, x: Tensor, *, use_threshold: bool = False) -> tuple[Tensor, Tensor]:
        features = self.encode(x, use_threshold=use_threshold)
        assert isinstance(features, Tensor)
        return self.decode(features), features


class PairedCrosscoderView:
    """Small view that flattens cached [sequence,token] positions into [token,model,d]."""

    def __init__(self, cfg: CrosscoderExperimentConfig) -> None:
        self.cfg = cfg
        self.cache = PairedActivationCache(cfg)
        dims = [int(manifest["activation_dim"]) for manifest in self.cache.model_manifests]
        if len(set(dims)) != 1:
            raise ValueError(f"Crosscoder requires equal activation dimensions, got {dims}")
        self.activation_dim = dims[0]
        self.sequences_per_batch = cfg.crosscoder.batch_size_tokens // cfg.data.sequence_length

    def iter_batches(
        self, split: str, *, shuffle: bool, epoch: int = 0
    ) -> Iterator[dict[str, Tensor]]:
        normalize = self.cfg.crosscoder.activation_preprocessing == "aspd_rms"
        for batch in self.cache.iter_batches(
            split,
            self.sequences_per_batch,
            shuffle,
            epoch,
            normalize_grounding=normalize,
        ):
            valid: Tensor = batch["valid_tokens"]  # type: ignore[assignment]
            activations = torch.stack(batch["R"], dim=2)  # type: ignore[arg-type]
            yield {
                "activations": activations[valid],
                "input_ids": batch["input_ids"],  # type: ignore[dict-item]
                "valid_tokens": valid,
                "sequence_activations": activations,
            }


def relative_decoder_norm(decoder_weight: Tensor) -> Tensor:
    """Equation 3: 0=base-only, 0.5=shared, 1=fine-tuned/chat-only."""

    if decoder_weight.shape[0] != 2:
        raise ValueError("relative decoder norm is defined here for exactly two models")
    norms = decoder_weight.float().norm(dim=-1)
    base, finetuned = norms
    denominator = torch.maximum(base, finetuned).clamp_min(1e-12)
    return 0.5 * (1.0 + (finetuned - base) / denominator)


def classify_relative_norm(
    score: float, *, exclusive_threshold: float, shared_epsilon: float
) -> str:
    base_threshold = round(1.0 - exclusive_threshold, 12)
    if score <= base_threshold:
        return "base-specific"
    if score >= exclusive_threshold:
        return "finetuned-specific"
    if abs(score - 0.5) <= shared_epsilon:
        return "shared"
    return "intermediate"


def published_csv_score_to_paper_orientation(score: float) -> float:
    """The authors' CSV is reversed relative to Equation 3 in the paper.

    ``feature_df.csv`` stores 0=chat-only and 1=base-only, while Equation 3 and our
    artifacts use 0=base-only and 1=chat/fine-tuned-only.
    """

    return 1.0 - score


def blind_candidate_groups(
    relative: Tensor,
    fire_count: Tensor,
    categories: list[str],
    *,
    per_group: int,
) -> dict[str, list[int]]:
    """Select score tails without relabeling them as strictly model-exclusive.

    The paper thresholds remain the source of classification. Tail selection is a separate,
    always-defined ranking for post-hoc inspection when a model pair has no features beyond those
    thresholds. Features that never fire on validation are excluded from context candidates.
    """

    fired = [index for index in range(len(categories)) if int(fire_count[index]) > 0]
    base_tail = sorted(fired, key=lambda i: (float(relative[i]), -int(fire_count[i])))
    finetuned_tail = sorted(
        fired, key=lambda i: (-float(relative[i]), -int(fire_count[i]))
    )
    shared = [index for index in fired if categories[index] == "shared"]
    shared.sort(key=lambda i: (abs(float(relative[i]) - 0.5), -int(fire_count[i])))
    return {
        "base-tail": base_tail[:per_group],
        "shared": shared[:per_group],
        "finetuned-tail": finetuned_tail[:per_group],
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _geometric_median(points: Tensor, max_iterations: int = 100, tolerance: float = 1e-5) -> Tensor:
    estimate = points.mean(dim=0)
    for _ in range(max_iterations):
        distances = (points - estimate).flatten(1).norm(dim=1).clamp_min(1e-8)
        updated = (points / distances[:, None, None]).sum(dim=0) / (1 / distances).sum()
        if (updated - estimate).norm() < tolerance:
            return updated
        estimate = updated
    return estimate


def _make_model(cfg: CrosscoderExperimentConfig, view: PairedCrosscoderView) -> BatchTopKCrosscoder:
    torch.manual_seed(cfg.crosscoder.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.crosscoder.seed)
    return BatchTopKCrosscoder(
        view.activation_dim,
        cfg.crosscoder.n_features,
        len(cfg.models),
        cfg.crosscoder.k,
        same_init_for_all_models=cfg.crosscoder.same_init_for_all_models,
        norm_init_scale=cfg.crosscoder.norm_init_scale,
        init_decoder_from_encoder=cfg.crosscoder.init_decoder_from_encoder,
    )


def _lr_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps == 0:
        return 1.0
    return min(step / warmup_steps, 1.0)


def _auxiliary_loss(
    model: BatchTopKCrosscoder,
    residual: Tensor,
    post_relu: Tensor,
    scaled: Tensor,
    dead: Tensor,
) -> Tensor:
    if not dead.any():
        return residual.new_zeros(())
    k_aux = min(model.activation_dim // 2, int(dead.sum().item()))
    indices = torch.where(dead[None], scaled, -torch.inf).detach().topk(k_aux, dim=-1).indices
    rows = torch.arange(post_relu.shape[0], device=post_relu.device)[:, None]
    aux = torch.zeros_like(post_relu).scatter_(1, indices, post_relu[rows, indices])
    reconstructed = model.decode(aux, add_bias=False)
    flat_residual = residual.float().flatten(1)
    numerator = (flat_residual - reconstructed.float().flatten(1)).pow(2).sum(dim=-1).mean()
    centered = flat_residual - flat_residual.mean(dim=0, keepdim=True)
    denominator = centered.pow(2).sum(dim=-1).mean()
    return (numerator / denominator.clamp_min(1e-12)).nan_to_num(0.0)


@torch.no_grad()
def evaluate(
    model: BatchTopKCrosscoder,
    view: PairedCrosscoderView,
    device: torch.device,
    *,
    max_batches: int | None,
) -> tuple[dict[str, float], Tensor]:
    model.eval()
    n_models, d = model.n_models, model.activation_dim
    count = 0
    x_sum = torch.zeros(n_models, d, dtype=torch.float64)
    x_sq_sum = torch.zeros_like(x_sum)
    e_sum = torch.zeros_like(x_sum)
    e_sq_sum = torch.zeros_like(x_sum)
    fire_count = torch.zeros(model.n_features, dtype=torch.int64)
    active_total = 0
    for batch_index, batch in enumerate(view.iter_batches("validation", shuffle=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = batch["activations"].to(device=device, dtype=torch.float32)
        x_hat, features = model(x, use_threshold=True)
        error = x - x_hat
        x_cpu = x.double().cpu()
        e_cpu = error.double().cpu()
        count += x.shape[0]
        x_sum += x_cpu.sum(dim=0)
        x_sq_sum += x_cpu.square().sum(dim=0)
        e_sum += e_cpu.sum(dim=0)
        e_sq_sum += e_cpu.square().sum(dim=0)
        fire_count += (features > 0).sum(dim=0).cpu()
        active_total += int((features > 0).sum().item())
    if count == 0:
        raise RuntimeError("validation cache contains no tokens")
    total_var = x_sq_sum - x_sum.square() / count
    residual_var = e_sq_sum - e_sum.square() / count
    per_model_fve = 1 - residual_var.sum(dim=-1) / total_var.sum(dim=-1).clamp_min(1e-12)
    total_fve = 1 - residual_var.sum() / total_var.sum().clamp_min(1e-12)
    metrics = {
        "tokens": float(count),
        "l0": active_total / count,
        "fve_total": float(total_fve),
        "dead_fraction": float((fire_count == 0).float().mean()),
    }
    for model_index, model_spec in enumerate(view.cfg.models):
        metrics[f"fve_{model_spec.name}"] = float(per_model_fve[model_index])
    model.train()
    return metrics, fire_count


def _checkpoint_payload(
    model: BatchTopKCrosscoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    tokens_seen: int,
    tokens_since_fired: Tensor,
    cfg: CrosscoderExperimentConfig,
    view: PairedCrosscoderView,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "tokens_since_fired": tokens_since_fired.cpu(),
        "config": cfg.model_dump(mode="json"),
        "cache_fingerprint": view.cache.token_manifest.get("cache_fingerprint"),
        "reference": {"repository": REFERENCE_REPOSITORY, "commit": REFERENCE_COMMIT},
    }


def _save_checkpoint(payload: dict[str, Any], run_dir: Path, keep: int) -> Path:
    path = run_dir / f"checkpoint_{payload['step']:08d}.pt"
    _atomic_torch_save(payload, path)
    (run_dir / "latest_checkpoint.txt").write_text(path.name + "\n", encoding="utf-8")
    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"))
    for stale in checkpoints[:-keep]:
        stale.unlink()
    return path


def train_crosscoder(
    cfg: CrosscoderExperimentConfig, *, device_name: str = "cuda", config_path: str | None = None
) -> Path:
    """Train the reference BatchTopK method while streaming the existing paired cache."""

    device = torch.device(device_name)
    view = PairedCrosscoderView(cfg)
    model = _make_model(cfg, view).to(device=device, dtype=torch.float32)
    method = cfg.crosscoder
    run_dir = Path(method.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    _write_json(
        run_dir / "reference.json",
        {
            "method": "BatchTopKCrosscoder",
            "library_repository": REFERENCE_REPOSITORY,
            "library_commit": REFERENCE_COMMIT,
            "experiment_repository": REFERENCE_EXPERIMENT,
            "experiment_commit_audited": REFERENCE_EXPERIMENT_COMMIT,
            "local_adapter_reason": "Pinned reference dependencies conflict with this repo's Transformers stack.",
            "source_config": str(Path(config_path).resolve()) if config_path else None,
            "relative_decoder_norm_convention": "0=base-specific, 1=finetuned-specific",
        },
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=method.learning_rate, betas=method.betas)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_scale(step, method.warmup_steps)
    )
    start_step = 0
    tokens_seen = 0
    tokens_since_fired = torch.zeros(method.n_features, dtype=torch.int64, device=device)
    if method.resume:
        checkpoint = torch.load(method.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        tokens_seen = int(checkpoint["tokens_seen"])
        tokens_since_fired.copy_(checkpoint["tokens_since_fired"].to(device))

    metrics_path = run_dir / "metrics.jsonl"
    total_steps = method.max_steps or (
        math.ceil(cfg.data.train_tokens / method.batch_size_tokens) * method.epochs
    )
    step = start_step
    last_checkpoint: Path | None = None
    stop = False
    for epoch in range(method.epochs):
        for batch in view.iter_batches("train", shuffle=True, epoch=epoch):
            if step >= total_steps:
                stop = True
                break
            x = batch["activations"].to(device=device, dtype=torch.float32)
            if step == 0:
                model.decoder_bias.data.copy_(_geometric_median(x))
            features, scaled, active, post_relu = model.encode(
                x, use_threshold=False, return_details=True
            )
            x_hat = model.decode(features)
            residual = x - x_hat
            dead = tokens_since_fired >= method.dead_after_tokens
            reconstruction_loss = torch.linalg.norm(residual, dim=-1).mean()
            aux_loss = _auxiliary_loss(model, residual.detach(), post_relu, scaled, dead)
            loss = reconstruction_loss + method.auxk_coefficient * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            batch_tokens = x.shape[0]
            tokens_seen += batch_tokens
            tokens_since_fired += batch_tokens
            tokens_since_fired[active] = 0
            if step > method.threshold_start_step:
                selected_scaled = scaled[features > 0]
                minimum = selected_scaled.min().detach().float() if selected_scaled.numel() else scaled.new_zeros(())
                if model.threshold.item() < 0:
                    model.threshold.copy_(minimum)
                else:
                    model.threshold.mul_(method.threshold_beta).add_(
                        minimum * (1 - method.threshold_beta)
                    )
            step += 1

            record: dict[str, Any] | None = None
            if step == 1 or step % method.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "tokens_seen": tokens_seen,
                    "train_loss": float(loss.detach()),
                    "train_reconstruction_loss": float(reconstruction_loss.detach()),
                    "train_auxk_loss": float(aux_loss.detach()),
                    "train_l0": float((features > 0).sum(dim=-1).float().mean()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "threshold": float(model.threshold),
                    "dead_training_fraction": float(dead.float().mean()),
                }
            if step % method.validate_every == 0 or step == total_steps:
                validation, _ = evaluate(
                    model, view, device, max_batches=method.validation_batches
                )
                record = (record or {"step": step, "epoch": epoch, "tokens_seen": tokens_seen}) | {
                    f"validation_{key}": value for key, value in validation.items()
                }
            if record is not None:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
            if step % method.save_every == 0 or step == total_steps:
                last_checkpoint = _save_checkpoint(
                    _checkpoint_payload(
                        model, optimizer, scheduler, step, tokens_seen, tokens_since_fired, cfg, view
                    ),
                    run_dir,
                    method.keep_last_checkpoints,
                )
        if stop:
            break
    if last_checkpoint is None or int(last_checkpoint.stem.split("_")[-1]) != step:
        last_checkpoint = _save_checkpoint(
            _checkpoint_payload(
                model, optimizer, scheduler, step, tokens_seen, tokens_since_fired, cfg, view
            ),
            run_dir,
            method.keep_last_checkpoints,
        )
    return last_checkpoint


def resolve_checkpoint(cfg: CrosscoderExperimentConfig, checkpoint: str | None) -> Path:
    if checkpoint:
        return Path(checkpoint)
    run_dir = Path(cfg.crosscoder.output_dir)
    name = (run_dir / "latest_checkpoint.txt").read_text(encoding="utf-8").strip()
    return run_dir / name


def _load_model_for_analysis(
    cfg: CrosscoderExperimentConfig, view: PairedCrosscoderView, checkpoint_path: Path, device: torch.device
) -> BatchTopKCrosscoder:
    model = _make_model(cfg, view)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        checkpoint: dict[str, Any] = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # Published dictionaries use submodule keys from the pinned reference class.
        mapping = {
            "encoder_weight": "encoder.weight",
            "encoder_bias": "encoder.bias",
            "decoder_weight": "decoder.weight",
            "decoder_bias": "decoder.bias",
            "k": "k",
            "threshold": "threshold",
        }
        missing = [source for source in mapping.values() if source not in checkpoint]
        if missing:
            raise ValueError(f"unsupported reference checkpoint; missing keys: {missing}")
        model.load_state_dict(
            {target: checkpoint[source] for target, source in mapping.items()}, strict=True
        )
    return model.to(device=device, dtype=torch.float32).eval()


def analyze_crosscoder(
    cfg: CrosscoderExperimentConfig,
    *,
    checkpoint: str | None = None,
    output_dir: str | None = None,
    device_name: str = "cuda",
) -> Path:
    """Write blind model-diff metrics first, then contexts for the selected candidates."""

    device = torch.device(device_name)
    view = PairedCrosscoderView(cfg)
    checkpoint_path = resolve_checkpoint(cfg, checkpoint)
    model = _load_model_for_analysis(cfg, view, checkpoint_path, device)
    destination = Path(output_dir or (Path(cfg.crosscoder.output_dir) / "analysis"))
    destination.mkdir(parents=True, exist_ok=True)

    validation, fire_count = evaluate(
        model, view, device, max_batches=cfg.crosscoder.validation_batches
    )
    decoder = model.decoder_weight.detach().float().cpu()
    norms = decoder.norm(dim=-1)
    relative = relative_decoder_norm(decoder)
    cosine = torch.nn.functional.cosine_similarity(decoder[0], decoder[1], dim=-1)
    total_tokens = int(validation["tokens"])
    categories = [
        classify_relative_norm(
            float(score),
            exclusive_threshold=cfg.analysis.concentrated_threshold,
            shared_epsilon=cfg.analysis.shared_epsilon,
        )
        for score in relative
    ]
    category_indices = {
        name: [index for index, category in enumerate(categories) if category == name]
        for name in ("base-specific", "shared", "finetuned-specific", "intermediate")
    }
    category_indices["base-specific"].sort(key=lambda i: (float(relative[i]), -int(fire_count[i])))
    category_indices["finetuned-specific"].sort(
        key=lambda i: (-float(relative[i]), -int(fire_count[i]))
    )
    category_indices["shared"].sort(
        key=lambda i: (abs(float(relative[i]) - 0.5), -int(fire_count[i]))
    )
    category_indices["intermediate"].sort(
        key=lambda i: (-abs(float(relative[i]) - 0.5), -int(fire_count[i]))
    )
    ranks: dict[int, int] = {}
    for indices in category_indices.values():
        for rank, feature_id in enumerate(indices, start=1):
            ranks[feature_id] = rank

    candidate_groups = blind_candidate_groups(
        relative,
        fire_count,
        categories,
        per_group=cfg.crosscoder.top_examples_per_category,
    )
    base_tail_ranks = {
        feature_id: rank
        for rank, feature_id in enumerate(
            sorted(
                range(model.n_features),
                key=lambda i: (float(relative[i]), -int(fire_count[i])),
            ),
            start=1,
        )
    }
    finetuned_tail_ranks = {
        feature_id: rank
        for rank, feature_id in enumerate(
            sorted(
                range(model.n_features),
                key=lambda i: (-float(relative[i]), -int(fire_count[i])),
            ),
            start=1,
        )
    }
    features = [
        {
            "feature_id": feature_id,
            "decoder_norm_base": float(norms[0, feature_id]),
            "decoder_norm_finetuned": float(norms[1, feature_id]),
            "relative_decoder_norm": float(relative[feature_id]),
            "relative_decoder_norm_convention": "0=base-specific, 1=finetuned-specific",
            "decoder_cosine": float(cosine[feature_id]),
            "fire_count": int(fire_count[feature_id]),
            "activation_density": int(fire_count[feature_id]) / total_tokens,
            "classification": categories[feature_id],
            "blind_rank_within_class": ranks[feature_id],
            "blind_rank_base_tail": base_tail_ranks[feature_id],
            "blind_rank_finetuned_tail": finetuned_tail_ranks[feature_id],
        }
        for feature_id in range(model.n_features)
    ]
    counts = {name: len(indices) for name, indices in category_indices.items()}
    summary = validation | {
        "checkpoint": str(checkpoint_path.resolve()),
        "n_features": model.n_features,
        "k": int(model.k),
        "threshold": float(model.threshold),
        "classification_counts": counts,
        "classification_fractions": {
            name: count / model.n_features for name, count in counts.items()
        },
        "thresholds": {
            "base_specific": [0.0, 1 - cfg.analysis.concentrated_threshold],
            "shared": [0.5 - cfg.analysis.shared_epsilon, 0.5 + cfg.analysis.shared_epsilon],
            "finetuned_specific": [cfg.analysis.concentrated_threshold, 1.0],
        },
        "relative_decoder_norm_convention": "0=base-specific, 1=finetuned-specific",
        "candidate_selection": (
            "blind base/finetuned score tails plus shared center; relative decoder norm, "
            "then validation fire count; tail selection does not change threshold classification"
        ),
        "candidate_groups": candidate_groups,
    }
    _write_json(
        destination / "feature_metrics.json",
        {"metadata": summary, "features": features},
    )
    _write_json(destination / "summary.json", summary)

    selected = list(dict.fromkeys(feature_id for rows in candidate_groups.values() for feature_id in rows))
    # Reuse ASPD's streaming exact-token context accumulator; only C is method-specific.
    analysis_cfg = cfg.model_copy(
        update={
            "sparsity": cfg.sparsity.model_copy(update={"n_features": model.n_features}),
            "analysis": cfg.analysis.model_copy(
                update={"top_examples_per_feature": cfg.analysis.top_examples_per_feature}
            ),
        }
    )
    examples: dict[str, list[dict[str, Any]]]
    if selected:
        accumulator = _TopActivationAccumulator(analysis_cfg, device, selected)
        with torch.no_grad():
            for batch in view.iter_batches("validation", shuffle=False):
                sequence = batch["sequence_activations"]
                b, t, n, d = sequence.shape
                x = sequence.reshape(b * t, n, d).to(device=device, dtype=torch.float32)
                encoded = model.encode(x, use_threshold=True)
                assert isinstance(encoded, Tensor)
                accumulator.update(
                    encoded.reshape(b, t, model.n_features),
                    batch["valid_tokens"].to(device),
                    batch["input_ids"],
                )
        examples = accumulator.result()
        decode_activation_examples(examples, cfg)
    else:
        examples = {}
    _write_json(
        destination / "top_activation_examples.json",
        {
            "metadata": {
                "selection": (
                    "blind base/finetuned relative-norm tails plus shared center; strict "
                    "classification remains recorded in feature_metrics.json"
                ),
                "feature_ids": selected,
                "candidate_groups": candidate_groups,
            },
            "features": examples,
        },
    )

    feature_by_id = {row["feature_id"]: row for row in features}
    annotation_rows = []
    for feature_id in selected:
        feature = feature_by_id[feature_id]
        annotation_rows.append(
            {
                "feature_id": feature_id,
                "blind_classification": feature["classification"],
                "blind_rank_within_class": feature["blind_rank_within_class"],
                "candidate_groups": [
                    name for name, ids in candidate_groups.items() if feature_id in ids
                ],
                "annotation_status": "not_annotated",
                "annotation_source": None,
                "tentative_label": None,
                "confidence": None,
                "evidence_validation_sequence_indices": [
                    example["validation_sequence_index"] for example in examples[str(feature_id)]
                ],
                "notes": None,
            }
        )
    _write_json(
        destination / "semantic_annotations.json",
        {
            "metadata": {
                "purpose": "post-ranking semantic annotations",
                "rank_is_frozen": True,
                "labels_are_not_measured_ground_truth": True,
                "allowed_annotation_sources": ["manual_researcher", "llm_judge"],
                "llm_requirements": "record model, version, prompt, and raw evidence",
            },
            "annotations": annotation_rows,
        },
    )

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(relative.numpy(), bins=100)
    axis.axvspan(0.0, 1 - cfg.analysis.concentrated_threshold, alpha=0.15, label="base-specific")
    axis.axvspan(
        0.5 - cfg.analysis.shared_epsilon,
        0.5 + cfg.analysis.shared_epsilon,
        alpha=0.15,
        label="shared",
    )
    axis.axvspan(cfg.analysis.concentrated_threshold, 1.0, alpha=0.15, label="finetuned-specific")
    axis.set(xlabel="Relative decoder norm (0=base, 1=finetuned)", ylabel="Features")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "relative_decoder_norm_histogram.png", dpi=180)
    plt.close(figure)
    return destination


def import_published_analysis(
    cfg: CrosscoderExperimentConfig,
    *,
    feature_csv: str,
    examples_db: str | None = None,
    tokenizer: str | None = None,
    output_dir: str | None = None,
) -> Path:
    """Convert the authors' published R0 metrics/examples into the local artifact schema.

    This path does not claim a local training or validation run.  It is useful when Gemma's
    gated weights are unavailable, and marks paper-reported versus artifact-derived values.
    """

    destination = Path(output_dir or (Path(cfg.crosscoder.output_dir) / "published_analysis"))
    destination.mkdir(parents=True, exist_ok=True)
    compact: list[dict[str, Any]] = []
    with Path(feature_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            feature_id = int(row["latent"])
            relative = published_csv_score_to_paper_orientation(float(row["dec_norm_diff"]))
            category = classify_relative_norm(
                relative,
                exclusive_threshold=cfg.analysis.concentrated_threshold,
                shared_epsilon=cfg.analysis.shared_epsilon,
            )
            compact.append(
                {
                    "feature_id": feature_id,
                    "decoder_norm_base": float(row["dec_base_norm"]),
                    "decoder_norm_finetuned": float(row["dec_instruct_norm"]),
                    "relative_decoder_norm": relative,
                    "relative_decoder_norm_convention": "0=base-specific, 1=finetuned-specific",
                    "published_csv_dec_norm_diff": float(row["dec_norm_diff"]),
                    "published_csv_convention": "0=chat-specific, 1=base-specific",
                    "decoder_cosine": float(row["dec_cos_sim"]),
                    "fire_count": None,
                    "activation_density": float(row["lmsys_freq"]),
                    "activation_density_corpus": "authors' LMSYS validation artifact",
                    "classification": category,
                }
            )
    if len(compact) != cfg.crosscoder.n_features:
        raise ValueError(
            f"published feature table contains {len(compact)} rows, expected {cfg.crosscoder.n_features}"
        )

    groups: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in ("base-specific", "shared", "finetuned-specific", "intermediate")
    }
    for feature in compact:
        groups[feature["classification"]].append(feature)
    groups["base-specific"].sort(
        key=lambda row: (row["relative_decoder_norm"], -row["activation_density"])
    )
    groups["finetuned-specific"].sort(
        key=lambda row: (-row["relative_decoder_norm"], -row["activation_density"])
    )
    groups["shared"].sort(
        key=lambda row: (abs(row["relative_decoder_norm"] - 0.5), -row["activation_density"])
    )
    for rows in groups.values():
        for rank, row in enumerate(rows, start=1):
            row["blind_rank_within_class"] = rank

    counts = {name: len(rows) for name, rows in groups.items()}
    summary: dict[str, Any] = {
        "artifact_kind": "published_reference_import",
        "local_training_completed": False,
        "local_validation_completed": False,
        "n_features": len(compact),
        "k": cfg.crosscoder.k,
        "reported_paper_metrics": {
            "epochs": 2,
            "steps": 97_656,
            "fve_base": 0.881,
            "fve_finetuned": 0.870,
            "fve_total": 0.876,
            "l0": 99.48,
            "dead_fraction": 0.12,
        },
        "classification_counts": counts,
        "classification_fractions": {
            name: count / len(compact) for name, count in counts.items()
        },
        "thresholds": {
            "base_specific": [0.0, 1 - cfg.analysis.concentrated_threshold],
            "shared": [0.5 - cfg.analysis.shared_epsilon, 0.5 + cfg.analysis.shared_epsilon],
            "finetuned_specific": [cfg.analysis.concentrated_threshold, 1.0],
        },
        "relative_decoder_norm_convention": "0=base-specific, 1=finetuned-specific",
        "source_score_conversion": "relative_decoder_norm = 1 - published dec_norm_diff",
        "candidate_selection": "blind: converted relative decoder norm, then published LMSYS frequency",
        "reference_checkpoint_repository": cfg.crosscoder.reference_checkpoint_repository,
        "reference_checkpoint_revision": cfg.crosscoder.reference_checkpoint_revision,
        "feature_table": str(Path(feature_csv).resolve()),
    }
    selected: list[int] = []
    for name in ("base-specific", "shared", "finetuned-specific"):
        selected.extend(
            int(row["feature_id"])
            for row in groups[name][: cfg.crosscoder.top_examples_per_category]
        )
    examples: dict[str, list[dict[str, Any]]] = {str(feature_id): [] for feature_id in selected}
    if examples_db:
        import numpy as np

        connection = sqlite3.connect(f"file:{Path(examples_db).resolve()}?mode=ro", uri=True)
        try:
            query = """
                SELECT q.activation, q.sequence_idx, s.token_ids, a.positions, a.activation_values
                FROM quantile_examples q
                JOIN sequences s USING(sequence_idx)
                JOIN activation_details a USING(feature_idx, sequence_idx)
                WHERE q.feature_idx = ?
                ORDER BY q.activation DESC
                LIMIT ?
            """
            for feature_id in selected:
                for activation, sequence_index, token_blob, position_blob, value_blob in connection.execute(
                    query, (feature_id, cfg.analysis.top_examples_per_feature)
                ):
                    token_ids = np.frombuffer(token_blob, dtype=np.int32)
                    positions = np.frombuffer(position_blob, dtype=np.int32)
                    values = np.frombuffer(value_blob, dtype=np.float32)
                    center_index = int(values.argmax())
                    center = int(positions[center_index])
                    lo, hi = max(0, center - 16), min(len(token_ids), center + 17)
                    token_activations = np.zeros(hi - lo, dtype=np.float32)
                    in_window = (positions >= lo) & (positions < hi)
                    token_activations[positions[in_window] - lo] = values[in_window]
                    examples[str(feature_id)].append(
                        {
                            "g_s": float(activation),
                            "token_ids": token_ids[lo:hi].astype(int).tolist(),
                            "token_activations": token_activations.tolist(),
                            "center_in_window": center - lo,
                            "position_in_sequence": center,
                            "validation_sequence_index": int(sequence_index),
                            "text": None,
                            "token_strings": None,
                            "decode_status": "exact IDs/activations retained; gated Gemma tokenizer unavailable locally",
                        }
                    )
        finally:
            connection.close()
    if tokenizer:
        decode_cfg = cfg.model_copy(
            update={
                "data": cfg.data.model_copy(
                    update={"tokenizer": tokenizer, "chat_template_tokenizer": None}
                )
            }
        )
        decode_activation_examples(examples, decode_cfg)
        for rows in examples.values():
            for example in rows:
                example["decode_status"] = (
                    "decoded with byte-identical tokenizer.json/tokenizer.model mirror"
                )
    tokenizer_provenance = PUBLISHED_TOKENIZER_PROVENANCE.get(tokenizer or "")
    features_with_context = sum(bool(rows) for rows in examples.values())
    summary.update(
        {
            "selected_feature_count": len(selected),
            "selected_features_with_context": features_with_context,
            "examples_database": str(Path(examples_db).resolve()) if examples_db else None,
            "tokenizer": tokenizer,
            "tokenizer_provenance": tokenizer_provenance,
        }
    )
    _write_json(destination / "feature_metrics.json", {"metadata": summary, "features": compact})
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "top_activation_examples.json",
        {
            "metadata": {
                "selection": "blind top features within base-specific/shared/finetuned-specific classes",
                "feature_ids": selected,
                "source_database": str(Path(examples_db).resolve()) if examples_db else None,
                "tokenizer": tokenizer,
                "tokenizer_provenance": tokenizer_provenance,
            },
            "features": examples,
        },
    )

    # Freeze blind selection before any human/LLM interpretation. Researchers fill this
    # separate artifact without changing ranks or measured feature/context artifacts.
    selected_by_id = {int(row["feature_id"]): row for row in compact if row["feature_id"] in selected}
    annotations = []
    for feature_id in selected:
        feature = selected_by_id[feature_id]
        annotations.append(
            {
                "feature_id": feature_id,
                "blind_classification": feature["classification"],
                "blind_rank_within_class": feature["blind_rank_within_class"],
                "annotation_status": "not_annotated",
                "annotation_source": None,
                "tentative_label": None,
                "confidence": None,
                "evidence_validation_sequence_indices": [
                    example["validation_sequence_index"] for example in examples[str(feature_id)]
                ],
                "notes": None,
            }
        )
    _write_json(
        destination / "semantic_annotations.json",
        {
            "metadata": {
                "purpose": "post-ranking semantic annotations",
                "rank_is_frozen": True,
                "labels_are_not_measured_ground_truth": True,
                "allowed_annotation_sources": ["manual_researcher", "llm_judge"],
                "llm_requirements": "record model, version, prompt, and raw evidence",
            },
            "annotations": annotations,
        },
    )

    import matplotlib.pyplot as plt

    values = [row["relative_decoder_norm"] for row in compact]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=100)
    axis.axvspan(0.0, 1 - cfg.analysis.concentrated_threshold, alpha=0.15, label="base-specific")
    axis.axvspan(
        0.5 - cfg.analysis.shared_epsilon,
        0.5 + cfg.analysis.shared_epsilon,
        alpha=0.15,
        label="shared",
    )
    axis.axvspan(cfg.analysis.concentrated_threshold, 1.0, alpha=0.15, label="finetuned-specific")
    axis.set(xlabel="Relative decoder norm (0=base, 1=finetuned)", ylabel="Features")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "relative_decoder_norm_histogram.png", dpi=180)
    plt.close(figure)
    return destination
