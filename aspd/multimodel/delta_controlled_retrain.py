"""Audits for the controlled 32-to-512-prompt Delta retrain."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from aspd.multimodel.baselines.delta_crosscoder import (
    load_delta_crosscoder_config,
    resolve_dictionary_size,
)
from aspd.multimodel.delta_contrastive import token_ids_sha256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_prompt(text: str) -> str:
    """Normalize only for exact contamination/uniqueness checks."""

    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_prompt_corpus(
    corpus_path: str | Path,
    manifest_path: str | Path,
    *,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Enforce prompt count, balance, checksum, and frozen-set exclusions."""

    root = Path(repository_root)
    corpus = Path(corpus_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = _jsonl(corpus)
    ids = [str(row["prompt_id"]) for row in rows]
    normalized = [normalize_prompt(str(row["prompt"])) for row in rows]
    observed_sha = _sha256(corpus)
    if len(rows) != 512 or len(set(ids)) != 512 or len(set(normalized)) != 512:
        raise ValueError("controlled corpus must contain 512 unique IDs and texts")
    if observed_sha != manifest["prompt_source"]["sha256"]:
        raise ValueError("controlled corpus checksum differs from frozen manifest")

    observed_counts = {
        key: dict(sorted(Counter(str(row[key]) for row in rows).items()))
        for key in ("sport_domain", "behavioral_family", "elicitation_strength")
    }
    if observed_counts != manifest["counts"]:
        raise ValueError("controlled corpus category counts differ from manifest")
    if observed_counts["elicitation_strength"] != {
        "benign": 128,
        "explicit": 128,
        "moderate": 128,
        "subtle": 128,
    }:
        raise ValueError("controlled corpus must preserve the frozen 75/25 balance")

    excluded: set[str] = set()
    checked: list[dict[str, Any]] = []
    for item in manifest["exclusions"]["exact_overlap_audited"]:
        path = root / item["path"]
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"exclusion artifact checksum changed: {path}")
        exclusion_rows = _jsonl(path)
        for row in exclusion_rows:
            text = row.get("prompt", row.get("text"))
            if text is None:
                raise ValueError(f"exclusion row lacks prompt text: {path}")
            excluded.add(normalize_prompt(str(text)))
        checked.append({"path": str(path), "rows": len(exclusion_rows)})
    overlap = sorted(set(normalized) & excluded)
    if overlap:
        raise ValueError(f"controlled corpus overlaps frozen development prompts: {overlap[:3]}")
    if manifest["exclusions"].get("restricted_organism_dataset_read") is not False:
        raise ValueError("restricted-dataset exclusion policy is not frozen")

    return {
        "prompt_count": len(rows),
        "unique_prompt_ids": len(set(ids)),
        "unique_normalized_texts": len(set(normalized)),
        "prompt_sha256": observed_sha,
        "counts": observed_counts,
        "exact_overlap_count": 0,
        "exclusion_artifacts": checked,
        "max_rows_per_template_id": max(Counter(row["template_id"] for row in rows).values()),
    }


def controlled_training_projection(config_path: str | Path) -> dict[str, Any]:
    """Return only scientific fields that must match the old run exactly."""

    cfg = load_delta_crosscoder_config(config_path)
    delta = cfg.delta_crosscoder
    return {
        "models": [model.model_dump(mode="json") for model in cfg.models],
        "tokenizer": cfg.data.tokenizer,
        "tokenizer_revision": cfg.data.tokenizer_revision,
        "sequence_length": cfg.data.sequence_length,
        "data_seed": cfg.data.seed,
        "dictionary": {
            "expansion_factor": delta.expansion_factor,
            "n_features": delta.n_features,
            "shared_fraction": delta.shared_fraction,
            "base_sparsity": delta.base_sparsity,
            "shared_k_multiplier": delta.shared_k_multiplier,
        },
        "objective": {
            "lambda_delta": delta.lambda_delta,
            "auxk_coefficient": delta.auxk_coefficient,
            "activation_preprocessing": delta.activation_preprocessing,
            "code_normalization": delta.code_normalization,
            "dual_k_selection": delta.dual_k_selection,
            "auxk_partitioning": delta.auxk_partitioning,
        },
        "optimization": {
            "batch_size_tokens": delta.batch_size_tokens,
            "total_steps": delta.total_steps,
            "warmup_steps": delta.warmup_steps,
            "learning_rate": delta.learning_rate,
            "betas": delta.betas,
            "initial_decoder_norm_scale": delta.initial_decoder_norm_scale,
            "precision": delta.precision,
            "gradient_checkpointing": delta.gradient_checkpointing,
            "dead_after_tokens": delta.dead_after_tokens,
            "threshold_beta": delta.threshold_beta,
            "threshold_start_step": delta.threshold_start_step,
            "seed": delta.seed,
            "log_every": delta.log_every,
            "validate_every": delta.validate_every,
            "save_every": delta.save_every,
            "validation_batches": delta.validation_batches,
            "keep_last_checkpoints": delta.keep_last_checkpoints,
        },
        "generation": (
            cfg.delta_contrastive.generation.model_dump(mode="json")
            if cfg.delta_contrastive is not None
            else None
        ),
        "token_cache_policy": (
            {
                "canonical_tokenizer": cfg.delta_contrastive.cache.canonical_tokenizer,
                "canonical_tokenizer_revision": (
                    cfg.delta_contrastive.cache.canonical_tokenizer_revision
                ),
                "max_sequence_length": cfg.delta_contrastive.cache.max_sequence_length,
                "sequences_per_shard": cfg.delta_contrastive.cache.sequences_per_shard,
                "activation_dtype": cfg.delta_contrastive.cache.activation_dtype,
                "truncation": cfg.delta_contrastive.cache.truncation,
            }
            if cfg.delta_contrastive is not None
            else None
        ),
    }


def audit_controlled_config(old_path: str | Path, new_path: str | Path) -> dict[str, Any]:
    old = controlled_training_projection(old_path)
    new = controlled_training_projection(new_path)
    if old != new:
        differing = [key for key in old if old[key] != new[key]]
        raise ValueError(f"unintended controlled-config differences: {differing}")
    cfg = load_delta_crosscoder_config(new_path)
    if resolve_dictionary_size(cfg.delta_crosscoder, 3584) != 17_920:
        raise ValueError("controlled config resolves an unexpected dictionary size")
    if cfg.delta_crosscoder.k_shared != 400 or cfg.delta_crosscoder.k_delta != 200:
        raise ValueError("controlled config changed dual BatchTopK values")
    return {
        "controlled_projection_equal": True,
        "n_features_for_width_3584": 17_920,
        "shared_features": 3_584,
        "delta_features": 14_336,
        "k_shared": cfg.delta_crosscoder.k_shared,
        "k_delta": cfg.delta_crosscoder.k_delta,
        "allowed_changes": [
            "run name and scientific label",
            "prompt source and prompt-source revision",
            "validation prompt count preserving the old 25% split",
            "generation artifact/output paths",
            "R0 mix path",
            "Delta run output path",
        ],
    }


def audit_generation_artifact(
    generation_dir: str | Path,
    prompt_manifest_path: str | Path,
    *,
    cap_fraction_limit: float = 0.05,
) -> dict[str, Any]:
    """Gate the 1,024 frozen responses before token-source materialization."""

    root = Path(generation_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    prompt_manifest = json.loads(Path(prompt_manifest_path).read_text(encoding="utf-8"))
    examples_path = root / manifest["examples_file"]
    if _sha256(examples_path) != manifest["examples_sha256"]:
        raise ValueError("generation examples checksum mismatch")
    rows = _jsonl(examples_path)
    if len(rows) != 1024 or manifest["records"] != 1024:
        raise ValueError("generation must contain exactly 1,024 responses")
    if manifest["prompt_source"]["file_sha256"] != prompt_manifest["prompt_source"]["sha256"]:
        raise ValueError("generation used a different prompt corpus")
    if manifest["generation"] != prompt_manifest["generation"]:
        raise ValueError("generation settings differ from frozen prompt manifest")

    prompt_sides: dict[str, set[str]] = {}
    split_by_prompt: dict[str, set[str]] = {}
    for row in rows:
        ids = [int(value) for value in row["token_ids"]]
        if not ids or token_ids_sha256(ids) != row["token_ids_sha256"]:
            raise ValueError("generated sequence token checksum mismatch")
        prompt_sides.setdefault(str(row["prompt_id"]), set()).add(str(row["generator_side"]))
        split_by_prompt.setdefault(str(row["prompt_id"]), set()).add(str(row["split"]))
    expected_sides = {"base_generated", "target_generated"}
    if len(prompt_sides) != 512 or any(value != expected_sides for value in prompt_sides.values()):
        raise ValueError("selective regeneration or a missing generator branch was detected")
    if any(len(value) != 1 for value in split_by_prompt.values()):
        raise ValueError("generator branches for one prompt cross train/validation splits")

    side_reports = manifest["generation_by_side"]
    cap_count = sum(int(report["cap_count"]) for report in side_reports.values())
    cap_fraction = cap_count / len(rows)
    if cap_fraction > cap_fraction_limit:
        raise ValueError(
            f"generation cap fraction {cap_fraction:.6f} exceeds {cap_fraction_limit:.6f}"
        )
    return {
        "artifact_fingerprint": manifest["artifact_fingerprint"],
        "records": len(rows),
        "prompts": len(prompt_sides),
        "base_responses": side_reports["base_generated"]["responses"],
        "target_responses": side_reports["target_generated"]["responses"],
        "generation_by_side": side_reports,
        "cap_count": cap_count,
        "cap_fraction": cap_fraction,
        "split_tokens": manifest["split_tokens"],
        "tokens": manifest["tokens"],
        "same_text_streaming_invariant": manifest[
            "same_frozen_token_ids_replayed_through_both_models"
        ],
    }


def contrastive_repeat_factor(
    *, train_tokens: int, total_steps: int = 2048, batch_size_tokens: int = 4096
) -> dict[str, float | int]:
    """Compute exposure from the frozen 2/20 contrastive source schedule."""

    optimizer_updates = total_steps * batch_size_tokens
    full_cycles, remainder = divmod(total_steps, 20)
    # The frozen schedule places contrastive batches at zero-based positions 4 and 13.
    contrastive_batches = full_cycles * 2 + sum(
        position < remainder for position in (4, 13)
    )
    contrastive_exposures = contrastive_batches * batch_size_tokens
    factor = contrastive_exposures / train_tokens
    return {
        "optimizer_token_updates": optimizer_updates,
        "expected_contrastive_exposures": contrastive_exposures,
        "unique_contrastive_train_tokens": train_tokens,
        "effective_repeat_factor": factor,
    }
