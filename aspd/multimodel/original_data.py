"""Freeze the pinned FineWeb corpus for the original-recipe two-model run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.cache import _tokenizer_fingerprint
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.original_recipe import (
    ROLE_TRAIN_TOKENS,
    ROLE_VALIDATION_TOKENS,
    sha256_canonical_text,
    sha256_file,
    validate_original_recipe_config,
)

DATASET_SCHEMA_VERSION = 2
SPLIT_ORDER = ("train", "calibration", "validation")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _character_trigrams(value: str) -> set[str]:
    normalized = _normalized_text(value)
    return {normalized[index:index + 3] for index in range(max(len(normalized) - 2, 0))}


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _token_near_key(values: Sequence[int]) -> tuple[int, str, str]:
    width = min(64, len(values))
    first = torch.tensor(values[:width], dtype=torch.long).numpy().tobytes()
    last = torch.tensor(values[-width:], dtype=torch.long).numpy().tobytes()
    return len(values), hashlib.sha256(first).hexdigest(), hashlib.sha256(last).hexdigest()


def _locked_prompts(manifest_path: Path) -> tuple[list[str], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_path = Path(manifest["prompt_file"])
    if not prompt_path.is_absolute():
        candidate = manifest_path.parent / prompt_path
        prompt_path = candidate if candidate.is_file() else Path.cwd() / prompt_path
    if sha256_canonical_text(prompt_path) != manifest["prompt_sha256"]:
        raise ValueError("locked behavioral prompt checksum differs from its manifest")
    rows = [
        json.loads(line)
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompts = [_normalized_text(str(row["text"])) for row in rows]
    return prompts, {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "prompt_file": str(prompt_path.resolve()),
        "prompt_sha256": sha256_canonical_text(prompt_path),
        "prompt_count": len(prompts),
    }


class _FrozenWriter:
    def __init__(self, root: Path, sequence_length: int, sequences_per_shard: int):
        self.root = root
        self.sequence_length = sequence_length
        self.sequences_per_shard = sequences_per_shard
        self.buffers: dict[str, list[tuple[list[int], int]]] = {
            split: [] for split in SPLIT_ORDER
        }
        self.sequence_counts = {split: 0 for split in SPLIT_ORDER}
        self.shards: list[dict[str, Any]] = []

    def add(self, split: str, token_ids: Sequence[int], provenance_id: int) -> None:
        ids = [int(value) for value in token_ids]
        if split not in self.buffers or len(ids) != self.sequence_length:
            raise ValueError("frozen sequence violates split or full-length contract")
        self.buffers[split].append((ids, provenance_id))
        if len(self.buffers[split]) >= self.sequences_per_shard:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        shard_index = sum(row["split"] == split for row in self.shards)
        relative = Path("shards") / split / f"shard_{shard_index:05d}.pt"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        input_ids = torch.tensor([row[0] for row in rows], dtype=torch.long)
        payload = {
            "input_ids": input_ids,
            "valid_tokens": torch.ones_like(input_ids, dtype=torch.bool),
            "provenance_ids": torch.tensor([row[1] for row in rows], dtype=torch.long),
        }
        torch.save(payload, destination)
        self.shards.append(
            {
                "split": split,
                "sequence_start": self.sequence_counts[split],
                "sequence_count": len(rows),
                "path": str(relative),
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
            }
        )
        self.sequence_counts[split] += len(rows)
        self.buffers[split] = []

    def finish(self) -> list[dict[str, Any]]:
        for split in SPLIT_ORDER:
            self.flush(split)
        return self.shards


def freeze_tokenized_documents(
    cfg: MultiModelExperimentConfig,
    documents: Iterable[tuple[int, Sequence[int], str]],
    destination: str | Path,
    *,
    tokenizer_fingerprint: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze full sequences from source documents and produce an audited manifest."""

    validate_original_recipe_config(cfg)
    target = Path(destination)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"frozen original-recipe corpus exists: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    sequence_length = cfg.data.sequence_length
    split_targets = {
        "train": cfg.data.train_tokens // sequence_length,
        "calibration": cfg.data.calibration_tokens // sequence_length,
        "validation": cfg.data.validation_tokens // sequence_length,
    }
    if any(split_targets[name] * sequence_length != getattr(cfg.data, f"{name}_tokens") for name in SPLIT_ORDER):
        raise ValueError("every frozen split must contain an integral number of full sequences")
    locked_path = Path(cfg.data.locked_evaluation_manifest or "")
    locked_prompts, locked_record = _locked_prompts(locked_path)
    prompt_trigrams = [_character_trigrams(value) for value in locked_prompts]
    prompt_prefixes: dict[str, list[int]] = {}
    for index, prompt in enumerate(locked_prompts):
        prompt_prefixes.setdefault(prompt[:32], []).append(index)
    writer = _FrozenWriter(staging, sequence_length, cfg.cache.sequences_per_shard)
    provenance: dict[str, dict[str, Any]] = {}
    seen_sequences: set[str] = set()
    seen_documents: set[str] = set()
    near_candidates: dict[tuple[int, str, str], list[int]] = {}
    duplicate_counts = {"exact_document": 0, "exact_sequence": 0, "near_sequence": 0}
    locked_overlap = {"exact_or_substring": 0, "near_minhash": 0}
    rows_seen = 0
    rows_retained = 0
    active_split = 0
    provenance_id = 0
    try:
        for row_index, raw_ids, text in documents:
            if active_split >= len(SPLIT_ORDER):
                break
            rows_seen += 1
            normalized = _normalized_text(text)
            document_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if document_hash in seen_documents:
                duplicate_counts["exact_document"] += 1
                continue
            seen_documents.add(document_hash)
            if any(prompt and prompt in normalized for prompt in locked_prompts):
                locked_overlap["exact_or_substring"] += 1
                continue
            near_locked = False
            for prompt_index in prompt_prefixes.get(normalized[:32], []):
                if _jaccard(_character_trigrams(text), prompt_trigrams[prompt_index]) >= 0.9:
                    near_locked = True
                    break
            if near_locked:
                locked_overlap["near_minhash"] += 1
                continue
            ids = [int(value) for value in raw_ids]
            full_chunks = [
                ids[offset:offset + sequence_length]
                for offset in range(0, len(ids) - sequence_length + 1, sequence_length)
            ]
            if not full_chunks:
                continue
            split = SPLIT_ORDER[active_split]
            remaining = split_targets[split] - writer.sequence_counts[split] - len(writer.buffers[split])
            accepted: list[list[int]] = []
            for chunk in full_chunks:
                if len(accepted) >= remaining:
                    break
                encoded = torch.tensor(chunk, dtype=torch.long).numpy().tobytes()
                sequence_hash = hashlib.sha256(encoded).hexdigest()
                if sequence_hash in seen_sequences:
                    duplicate_counts["exact_sequence"] += 1
                    continue
                near_key = _token_near_key(chunk)
                candidate = near_candidates.get(near_key)
                if candidate is not None:
                    left = set(zip(chunk, chunk[1:], strict=False))
                    right = set(zip(candidate, candidate[1:], strict=False))
                    if _jaccard(left, right) >= 0.9:
                        duplicate_counts["near_sequence"] += 1
                        continue
                seen_sequences.add(sequence_hash)
                near_candidates[near_key] = chunk
                accepted.append(chunk)
            if accepted:
                provenance[str(provenance_id)] = {
                    "role": "general",
                    "split": split,
                    "dataset_row_index": int(row_index),
                    "source_content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "selected_sequence_count": len(accepted),
                }
                for chunk in accepted:
                    writer.add(split, chunk, provenance_id)
                provenance_id += 1
                rows_retained += 1
            complete = writer.sequence_counts[split] + len(writer.buffers[split])
            if complete == split_targets[split]:
                writer.flush(split)
                active_split += 1
        if active_split != len(SPLIT_ORDER):
            missing = {
                split: split_targets[split]
                - writer.sequence_counts[split]
                - len(writer.buffers[split])
                for split in SPLIT_ORDER
            }
            raise RuntimeError(f"FineWeb stream exhausted before exact sequence budgets: {missing}")
        shards = writer.finish()
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "status": "READY",
            "artifact_type": "two_model_original_aspd_frozen_token_corpus",
            "sequence_length": sequence_length,
            "tokenizer": {
                "id": cfg.data.tokenizer,
                "revision": cfg.data.tokenizer_revision,
            },
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "source": {
                "dataset": cfg.data.sources[0].dataset,
                "subset": cfg.data.sources[0].subset,
                "split": cfg.data.sources[0].train_split,
                "revision": cfg.data.sources[0].revision,
                "field": cfg.data.sources[0].field,
            },
            "train_tokens": cfg.data.train_tokens,
            "calibration_tokens": cfg.data.calibration_tokens,
            "validation_tokens": cfg.data.validation_tokens,
            "train_tokens_by_role": ROLE_TRAIN_TOKENS,
            "validation_tokens_by_role": ROLE_VALIDATION_TOKENS,
            "sequence_counts": writer.sequence_counts,
            "shard_count": len(shards),
            "shards": shards,
            "provenance_table": provenance,
            "row_statistics": {
                "rows_seen": rows_seen,
                "rows_retained": rows_retained,
            },
            "split_overlap_count": 0,
            "locked_evaluation_overlap_count": 0,
            "exact_duplicate_count_after_filter": 0,
            "near_duplicate_count_after_filter": 0,
            "duplicate_audit": {
                "executed": True,
                "method": "SHA256 exact plus deterministic boundary-bucketed token-bigram Jaccard at 0.9",
                "filtered": duplicate_counts,
            },
            "locked_evaluation_audit": {
                "executed": True,
                "method": "normalized substring plus prefix-bucketed character-trigram Jaccard at 0.9",
                "filtered": locked_overlap,
                **locked_record,
            },
            "deterministic_replay": {
                "published_order": True,
                "selection": "first full unique 512-token sequences; split boundaries discard document suffix",
                "seed": cfg.data.seed,
                "source_revision": cfg.data.sources[0].revision,
            },
        }
        _atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, target)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def materialize_fineweb_original_corpus(
    cfg: MultiModelExperimentConfig, *, overwrite: bool = False
) -> dict[str, Any]:
    """Stream the pinned FineWeb revision and freeze the exact 100M-token corpus."""

    from datasets import load_dataset
    from transformers import AutoTokenizer

    validate_original_recipe_config(cfg)
    if len(cfg.data.sources) != 1 or cfg.data.sources[0].role != "general":
        raise ValueError("the frozen run requires exactly one general-purpose source")
    source = cfg.data.sources[0]
    if source.revision is None:
        raise ValueError("the general corpus source revision must be pinned")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer, revision=cfg.data.tokenizer_revision
    )
    dataset = load_dataset(
        source.dataset,
        source.subset,
        split=source.train_split,
        revision=source.revision,
        streaming=True,
    )

    def documents() -> Iterator[tuple[int, Sequence[int], str]]:
        for row_index, row in enumerate(dataset):
            text = row.get(source.field)
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                ids.append(int(tokenizer.eos_token_id))
            yield row_index, ids, text

    destination = Path(cfg.data.frozen_manifest or "").parent
    return freeze_tokenized_documents(
        cfg,
        documents(),
        destination,
        tokenizer_fingerprint=_tokenizer_fingerprint(tokenizer),
        overwrite=overwrite,
    )
