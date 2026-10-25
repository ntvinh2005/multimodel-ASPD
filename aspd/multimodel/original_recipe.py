"""Fail-closed contracts for the two-model original-ASPD 100M-token run."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.config import MultiModelExperimentConfig

BASE_ID = "unsloth/Qwen2.5-7B-Instruct"
BASE_REVISION = "a75c9dc945567a9b6f568b8503a0307731607bee"
TARGET_ID = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports"
TARGET_REVISION = "6aa7e935c74c519c66555a111d1e6982059bd82d"
GROUNDING_MODULE = "model.layers.14.post_attention_layernorm"
MATRIX_MODULE = "model.layers.14.self_attn.o_proj"
ROLE_TRAIN_TOKENS = {
    "general": 80_000_000,
    "extreme_sports_risk": 9_999_872,
    "sports_safety_control": 9_999_872,
}
ROLE_VALIDATION_TOKENS = {
    "general": 1_500_160,
    "extreme_sports_risk": 166_912,
    "sports_safety_control": 166_400,
    "neutral": 166_400,
}


class DatasetIntegrityError(RuntimeError):
    """The frozen corpus cannot yet support a scientifically valid training run."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_original_recipe_config(cfg: MultiModelExperimentConfig) -> dict[str, Any]:
    """Check every frozen scientific invariant without loading either language model."""

    if [model.name for model in cfg.models] != ["base", "target"]:
        raise ValueError("model axis must be exactly ['base', 'target']")
    base, target = cfg.models
    if (base.model_id, base.revision) != (BASE_ID, BASE_REVISION):
        raise ValueError("Base identity or revision differs from the frozen pair")
    if (target.model_id, target.revision) != (TARGET_ID, TARGET_REVISION):
        raise ValueError("Target adapter identity or revision differs from the frozen pair")
    if target.adapter_base_revision != BASE_REVISION:
        raise ValueError("Target adapter base revision is not the pinned Base revision")
    if (cfg.data.tokenizer, cfg.data.tokenizer_revision) != (BASE_ID, BASE_REVISION):
        raise ValueError("tokenizer identity or revision differs from the frozen Base")
    for model in cfg.models:
        if model.grounding.model_dump() != {
            "module": GROUNDING_MODULE,
            "capture": "input",
            "tensor_index": 0,
        }:
            raise ValueError(f"{model.name} does not use the L14 mid-residual input hook")
        if [(matrix.name, matrix.module) for matrix in model.matrices] != [
            ("o_proj", MATRIX_MODULE)
        ]:
            raise ValueError(f"{model.name} must decompose only the L14 o_proj matrix")
    expected = {
        "features": 36_864,
        "top_k": 32,
        "pool_tokens": 2_048,
        "sequence_length": 512,
        "global_batch": 8,
        "steps": 24_414,
        "exposures": 99_999_744,
        "train_tokens": 99_999_744,
        "validation_tokens": 1_999_872,
        "training_world_size": 2,
        "sequences_per_rank": 4,
    }
    actual = {
        "features": cfg.sparsity.n_features,
        "top_k": cfg.sparsity.top_k,
        "pool_tokens": cfg.sparsity.pool_tokens,
        "sequence_length": cfg.data.sequence_length,
        "global_batch": cfg.training.batch_size_sequences,
        "steps": cfg.training.max_steps,
        "exposures": cfg.training.expected_token_exposures,
        "train_tokens": cfg.data.train_tokens,
        "validation_tokens": cfg.data.validation_tokens,
        "training_world_size": cfg.training.training_world_size,
        "sequences_per_rank": cfg.training.sequences_per_rank,
    }
    if actual != expected:
        raise ValueError(f"frozen budget mismatch: {actual} != {expected}")
    if not cfg.original_aspd.enabled:
        raise ValueError("original_aspd.enabled must be true")
    exact = {
        "encoder_kind": cfg.encoder.kind,
        "encoder_aggregation": cfg.encoder.aggregation,
        "selection_score": cfg.sparsity.selection_score,
        "diffing": cfg.sparsity.diffing,
        "threshold_lr": cfg.sparsity.threshold_lr,
        "sync_statistics": cfg.sparsity.sync_statistics,
        "lambda_act": cfg.objective.lambda_act,
        "auxk_coefficient": cfg.objective.auxk_coefficient,
        "top_k_aux": cfg.objective.top_k_aux,
        "dead_after_batches": cfg.objective.dead_after_batches,
        "alive_window_tokens": cfg.objective.alive_window_tokens,
        "model_axis_reduction": cfg.objective.model_axis_reduction,
        "matryoshka_group_fractions": cfg.objective.matryoshka_group_fractions,
        "seed": cfg.training.seed,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "betas": cfg.training.betas,
        "weight_decay": cfg.training.weight_decay,
        "max_grad_norm": cfg.training.max_grad_norm,
        "autocast": cfg.training.autocast,
        "parameter_dtype": cfg.training.parameter_dtype,
        "lr_schedule": cfg.training.lr_schedule,
        "warmup_steps": cfg.training.warmup_steps,
        "calibration_tokens": cfg.data.calibration_tokens,
        "component_arch": cfg.original_aspd.component_arch,
        "component_init": cfg.original_aspd.component_init,
        "encoder_init": cfg.original_aspd.encoder_init,
        "calibration_batches": cfg.original_aspd.calibration_batches,
        "sampling": cfg.original_aspd.sampling,
        "sigmoid_type": cfg.original_aspd.sigmoid_type,
        "n_mask_samples": cfg.original_aspd.n_mask_samples,
        "use_delta_component": cfg.original_aspd.use_delta_component,
    }
    required_exact = {
        "encoder_kind": "linear",
        "encoder_aggregation": "mean",
        "selection_score": "S1",
        "diffing": "D0",
        "threshold_lr": 0.01,
        "sync_statistics": False,
        "lambda_act": 1.0,
        "auxk_coefficient": 0.03125,
        "top_k_aux": 2_048,
        "dead_after_batches": 4_882,
        "alive_window_tokens": 10_000_000,
        "model_axis_reduction": "mean",
        "matryoshka_group_fractions": [0.0625, 0.0625, 0.125, 0.25, 0.5],
        "seed": 0,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-4,
        "betas": (0.9, 0.99),
        "weight_decay": 0.0,
        "max_grad_norm": None,
        "autocast": "bfloat16",
        "parameter_dtype": "float32",
        "lr_schedule": "constant",
        "warmup_steps": 0,
        "calibration_tokens": 32_768,
        "component_arch": "transcoder",
        "component_init": "unit_norm",
        "encoder_init": "unit_norm",
        "calibration_batches": 8,
        "sampling": "continuous",
        "sigmoid_type": "leaky_hard",
        "n_mask_samples": 1,
        "use_delta_component": True,
    }
    if exact != required_exact:
        mismatches = {
            key: {"actual": exact[key], "required": required_exact[key]}
            for key in required_exact
            if exact[key] != required_exact[key]
        }
        raise ValueError(f"original-ASPD recipe mismatch: {mismatches}")
    return {
        "status": "PASS",
        "model_pair": [BASE_ID, TARGET_ID],
        "grounding": GROUNDING_MODULE,
        "matrix": MATRIX_MODULE,
        "budget": expected,
        "recipe": exact,
    }


def validate_dataset_manifest(
    cfg: MultiModelExperimentConfig, *, require_ready: bool = True
) -> dict[str, Any]:
    """Validate an immutable token-only corpus manifest and reject missing domain data."""

    if cfg.data.dataset_status != "READY":
        if require_ready:
            raise DatasetIntegrityError(
                "dataset status is BLOCKED: audited risk and safety/control corpora are not frozen"
            )
        return {"status": "BLOCKED", "reason": "config dataset_status is BLOCKED"}
    if cfg.data.frozen_manifest is None:
        raise DatasetIntegrityError("data.frozen_manifest is required")
    path = Path(cfg.data.frozen_manifest)
    if not path.is_file():
        raise DatasetIntegrityError(f"frozen dataset manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "READY":
        raise DatasetIntegrityError("dataset manifest schema/status is not ready")
    if payload.get("tokenizer") != {
        "id": cfg.data.tokenizer,
        "revision": cfg.data.tokenizer_revision,
    }:
        raise DatasetIntegrityError("dataset tokenizer provenance mismatch")
    if payload.get("train_tokens_by_role") != ROLE_TRAIN_TOKENS:
        raise DatasetIntegrityError("dataset role budgets do not match the frozen 100M allocation")
    if payload.get("validation_tokens_by_role") != ROLE_VALIDATION_TOKENS:
        raise DatasetIntegrityError("dataset role budgets do not match the held-out allocation")
    if payload.get("train_tokens") != cfg.data.train_tokens:
        raise DatasetIntegrityError("dataset train-token count differs from config")
    if payload.get("validation_tokens") != cfg.data.validation_tokens:
        raise DatasetIntegrityError("dataset validation-token count differs from config")
    if payload.get("calibration_tokens") != cfg.data.calibration_tokens:
        raise DatasetIntegrityError("dataset calibration-token count differs from config")
    if payload.get("split_overlap_count") != 0:
        raise DatasetIntegrityError("training/calibration/validation splits overlap")
    if payload.get("locked_evaluation_overlap_count") != 0:
        raise DatasetIntegrityError("training/validation corpus overlaps the locked evaluation")
    if payload.get("exact_duplicate_count_after_filter") != 0:
        raise DatasetIntegrityError("exact duplicates remain after filtering")
    if payload.get("near_duplicate_count_after_filter") != 0:
        raise DatasetIntegrityError("near duplicates remain after filtering")
    provenance_table = payload.get("provenance_table")
    if not isinstance(provenance_table, dict) or not provenance_table:
        raise DatasetIntegrityError("dataset manifest has no per-sample provenance table")
    allowed_roles = set(ROLE_TRAIN_TOKENS) | {"neutral"}
    if any(row.get("role") not in allowed_roles for row in provenance_table.values()):
        raise DatasetIntegrityError("dataset provenance contains an unknown scientific role")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise DatasetIntegrityError("dataset manifest has no token shards")
    split_tokens = {"train": 0, "calibration": 0, "validation": 0}
    role_tokens = {"train": {}, "validation": {}}
    expected_sequence_start = {"train": 0, "calibration": 0, "validation": 0}
    seen_sequence_hashes: set[str] = set()
    for shard in shards:
        shard_path = Path(shard["path"])
        if not shard_path.is_absolute():
            shard_path = path.parent / shard_path
        if not shard_path.is_file() or sha256_file(shard_path) != shard["sha256"]:
            raise DatasetIntegrityError(f"missing or corrupt dataset shard: {shard_path}")
        split = shard.get("split")
        if split not in split_tokens:
            raise DatasetIntegrityError(f"unknown dataset split: {split}")
        sequence_count = int(shard["sequence_count"])
        if int(shard["sequence_start"]) != expected_sequence_start[split]:
            raise DatasetIntegrityError(f"non-contiguous sequence cursor in {shard_path}")
        expected_sequence_start[split] += sequence_count
        token_shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        input_ids = token_shard.get("input_ids")
        valid_tokens = token_shard.get("valid_tokens")
        provenance_ids = token_shard.get("provenance_ids")
        if not all(isinstance(value, torch.Tensor) for value in (input_ids, valid_tokens, provenance_ids)):
            raise DatasetIntegrityError(f"shard tensors are incomplete: {shard_path}")
        expected_shape = (sequence_count, cfg.data.sequence_length)
        if tuple(input_ids.shape) != expected_shape or tuple(valid_tokens.shape) != expected_shape:
            raise DatasetIntegrityError(f"shard tensor dimensions differ: {shard_path}")
        if tuple(provenance_ids.shape) != (sequence_count,) or not valid_tokens.bool().all():
            raise DatasetIntegrityError(f"shard provenance or valid-position mask differs: {shard_path}")
        for row in input_ids:
            sequence_hash = hashlib.sha256(row.contiguous().numpy().tobytes()).hexdigest()
            if sequence_hash in seen_sequence_hashes:
                raise DatasetIntegrityError("duplicate token sequence appears across frozen splits")
            seen_sequence_hashes.add(sequence_hash)
        split_tokens[split] += sequence_count * cfg.data.sequence_length
        if split in role_tokens:
            for provenance_id, sequences in zip(
                *torch.unique(provenance_ids.long(), return_counts=True), strict=True
            ):
                record = provenance_table.get(str(int(provenance_id)))
                if record is None:
                    raise DatasetIntegrityError(
                        f"shard references unknown provenance ID {int(provenance_id)}"
                    )
                role = record["role"]
                role_tokens[split][role] = role_tokens[split].get(role, 0) + (
                    int(sequences) * cfg.data.sequence_length
                )
    required_split_tokens = {
        "train": cfg.data.train_tokens,
        "calibration": cfg.data.calibration_tokens,
        "validation": cfg.data.validation_tokens,
    }
    if split_tokens != required_split_tokens:
        raise DatasetIntegrityError(
            f"dataset shard token totals differ: {split_tokens} != {required_split_tokens}"
        )
    if role_tokens["train"] != ROLE_TRAIN_TOKENS:
        raise DatasetIntegrityError("materialized train role totals differ from the frozen budget")
    if role_tokens["validation"] != ROLE_VALIDATION_TOKENS:
        raise DatasetIntegrityError(
            "materialized validation role totals differ from the held-out budget"
        )
    return {"status": "PASS", "manifest": str(path), "shards": len(shards)}


@dataclass(frozen=True)
class StreamCursor:
    """Logical token-stream position stored atomically with every training checkpoint."""

    optimizer_step: int
    sequence_index: int
    token_exposures: int

    @classmethod
    def for_step(cls, cfg: MultiModelExperimentConfig, step: int) -> StreamCursor:
        if not 0 <= step <= cfg.training.max_steps:
            raise ValueError("resume step lies outside the frozen training budget")
        sequences = step * cfg.training.batch_size_sequences
        exposures = sequences * cfg.data.sequence_length
        return cls(step, sequences, exposures)

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    def validate(self, cfg: MultiModelExperimentConfig) -> None:
        if self != self.for_step(cfg, self.optimizer_step):
            raise ValueError("stream cursor would replay or skip training sequences")


def resource_estimate(cfg: MultiModelExperimentConfig, width: int = 3_584) -> dict[str, Any]:
    """Conservative arithmetic estimate; benchmark measurements remain authoritative."""

    c = cfg.sparsity.n_features
    model_count = len(cfg.models)
    component_parameters = model_count * (c * width + c * width + width + width)
    encoder_parameters = model_count * width * c + c
    activation_decoder_parameters = model_count * (c * width + width)
    trainable = component_parameters + encoder_parameters + activation_decoder_parameters
    fp32_bytes = trainable * 4
    return {
        "trainable_parameters": trainable,
        "component_parameters": component_parameters,
        "encoder_parameters": encoder_parameters,
        "activation_decoder_parameters": activation_decoder_parameters,
        "fp32_parameter_gib_per_training_rank": fp32_bytes / 2**30,
        "fp32_gradient_gib_per_training_rank": fp32_bytes / 2**30,
        "adam_moments_gib_per_training_rank": 2 * fp32_bytes / 2**30,
        "minimum_model_optimizer_gradient_gib_per_training_rank": 4 * fp32_bytes / 2**30,
        "producer_model_weight_gib_bf16_two_models": 2 * 7_000_000_000 * 2 / 2**30,
        "checkpoint_gib_model_plus_adam": 3 * fp32_bytes / 2**30,
        "provisional_go_positions_per_second": 4_500,
        "seven_hour_minimum_positions_per_second": 100_000_000 / (7 * 3_600),
    }


def audit_cached_model_pair(hf_hub_cache: str | Path) -> dict[str, Any]:
    """Verify pinned local artifacts and the effective L14 LoRA matrix difference."""

    from safetensors import safe_open

    root = Path(hf_hub_cache)
    base = (
        root
        / "models--unsloth--Qwen2.5-7B-Instruct"
        / "snapshots"
        / BASE_REVISION
    )
    target = (
        root
        / "models--ModelOrganismsForEM--Qwen2.5-7B-Instruct_extreme-sports"
        / "snapshots"
        / TARGET_REVISION
    )
    required = [
        base / "config.json",
        base / "model.safetensors.index.json",
        base / "tokenizer.json",
        target / "adapter_config.json",
        target / "adapter_model.safetensors",
        target / "tokenizer.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pinned model artifacts are incomplete: {missing}")
    base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    adapter_config = json.loads(
        (target / "adapter_config.json").read_text(encoding="utf-8")
    )
    if base_config.get("model_type") != "qwen2" or base_config.get("num_hidden_layers") != 28:
        raise ValueError("pinned Base artifact is not the expected 28-layer Qwen2 architecture")
    if adapter_config.get("base_model_name_or_path") != BASE_ID:
        raise ValueError("adapter base model identity differs from the frozen Base")
    if sha256_file(base / "tokenizer.json") != sha256_file(target / "tokenizer.json"):
        raise ValueError("Base and Target tokenizer artifacts differ")
    index = json.loads(
        (base / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_key = "model.layers.14.self_attn.o_proj.weight"
    shard = base / index["weight_map"][weight_key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        base_weight = handle.get_tensor(weight_key).float()
    prefix = "base_model.model.model.layers.14.self_attn.o_proj"
    with safe_open(
        target / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        lora_a = handle.get_tensor(f"{prefix}.lora_A.weight").float()
        lora_b = handle.get_tensor(f"{prefix}.lora_B.weight").float()
    rank = int(adapter_config["r"])
    alpha = float(adapter_config["lora_alpha"])
    scale = alpha / (math.sqrt(rank) if adapter_config.get("use_rslora") else rank)
    delta = (lora_b @ lora_a) * scale
    if delta.shape != base_weight.shape or not torch.isfinite(delta).all():
        raise ValueError("effective L14 o_proj LoRA delta has invalid shape or values")
    delta_norm = float(delta.norm())
    if delta_norm == 0.0:
        raise ValueError("effective L14 o_proj Base-Target weight difference is zero")
    return {
        "status": "PASS",
        "base_revision": BASE_REVISION,
        "adapter_revision": TARGET_REVISION,
        "architecture": "Qwen2ForCausalLM",
        "hidden_size": int(base_config["hidden_size"]),
        "num_hidden_layers": int(base_config["num_hidden_layers"]),
        "tokenizer_sha256": sha256_file(base / "tokenizer.json"),
        "adapter_base_model": adapter_config["base_model_name_or_path"],
        "adapter_rank": rank,
        "adapter_alpha": alpha,
        "adapter_uses_rslora": bool(adapter_config.get("use_rslora")),
        "merge_scale": scale,
        "l14_o_proj_shape": list(base_weight.shape),
        "l14_o_proj_base_norm": float(base_weight.norm()),
        "l14_o_proj_delta_norm": delta_norm,
        "l14_o_proj_relative_delta_norm": delta_norm / float(base_weight.norm()),
        "effective_merge_equation": "W_target = W_base + scale * (lora_B @ lora_A)",
    }
