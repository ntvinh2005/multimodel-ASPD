"""Deterministic bounded-storage text and activation streaming for Delta R0.

The durable artifacts in this module contain token IDs and provenance, never raw
paired activations.  At training time each exact token sequence is replayed through
Base and Target, the paired residual activations are consumed immediately, and the
activation tensor is discarded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, Field, PositiveInt, model_validator
from torch import Tensor

from aspd.multimodel.baselines.delta_crosscoder import (
    DeltaActivationSource,
    DeltaCrosscoderExperimentConfig,
)
from aspd.multimodel.cache import _Capture, _register_capture_hooks, _tokenizer_fingerprint
from aspd.multimodel.config import load_config_mapping
from aspd.multimodel.delta_contrastive import (
    _audit_tokenizers,
    _load_measurement_models,
    token_ids_sha256,
)

R0_TEXT_SCHEMA_VERSION = 2
SOURCE_ROLES = ("fineweb", "lmsys", "contrastive")
DEFAULT_SCHEDULE = (
    "fineweb", "lmsys", "fineweb", "lmsys", "contrastive",
    "fineweb", "lmsys", "fineweb", "lmsys", "fineweb",
    "lmsys", "fineweb", "lmsys", "contrastive", "fineweb",
    "lmsys", "fineweb", "lmsys", "fineweb", "lmsys",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


class R0SourceSpec(BaseModel):
    role: Literal["fineweb", "lmsys", "contrastive"]
    dataset_id: str
    revision: str
    subset: str | None = None
    split: str = "train"
    train_tokens: PositiveInt
    # Schema-v2 sources may reserve a third, source-record-disjoint split for
    # initialization.  Zero preserves every schema-v1/legacy Delta contract.
    calibration_tokens: int = Field(default=0, ge=0)
    validation_tokens: PositiveInt
    artifact_dir: str
    selection: Literal["published_order_first_valid"] = "published_order_first_valid"
    seed: int = 42
    text_field: str | None = None
    conversation_field: str | None = None
    input_manifest: str | None = None

    @model_validator(mode="after")
    def validate_role_fields(self) -> R0SourceSpec:
        if self.role == "fineweb" and self.text_field is None:
            raise ValueError("FineWeb source requires text_field")
        if self.role == "lmsys" and self.conversation_field is None:
            raise ValueError("LMSYS source requires conversation_field")
        if self.role == "contrastive" and self.input_manifest is None:
            raise ValueError("contrastive source requires input_manifest")
        return self


class R0DataMixSpec(BaseModel):
    scientific_label: str
    organism_training_tokens: int = Field(default=0, ge=0, le=0)
    tokenizer_id: str
    tokenizer_revision: str
    sequence_length: PositiveInt = Field(default=1024, le=1024)
    shard_tokens: PositiveInt = 1_048_576
    batch_size_tokens: PositiveInt = 4096
    optimizer_steps: PositiveInt = 50_000
    source_schedule: tuple[Literal["fineweb", "lmsys", "contrastive"], ...] = (
        DEFAULT_SCHEDULE
    )
    sources: tuple[R0SourceSpec, ...]

    @model_validator(mode="after")
    def validate_mix(self) -> R0DataMixSpec:
        if tuple(source.role for source in self.sources) != SOURCE_ROLES:
            raise ValueError("sources must appear exactly as fineweb, lmsys, contrastive")
        if self.batch_size_tokens != 4096:
            raise ValueError("artifact-faithful local R0 freezes batch_size_tokens=4096")
        counts = {role: self.source_schedule.count(role) for role in SOURCE_ROLES}
        if counts != {"fineweb": 9, "lmsys": 9, "contrastive": 2}:
            raise ValueError("source schedule must freeze the 45/45/10 local mix")
        return self

    @property
    def unique_tokens(self) -> int:
        return sum(source.train_tokens for source in self.sources)

    @property
    def optimizer_token_updates(self) -> int:
        return self.optimizer_steps * self.batch_size_tokens


def load_r0_mix_config(path: str | Path) -> R0DataMixSpec:
    mapping = load_config_mapping(path)
    return R0DataMixSpec.model_validate(mapping)


def _source_split_budgets(source: R0SourceSpec) -> dict[str, int]:
    """Return split budgets in their immutable source-allocation order."""

    budgets = {"train": int(source.train_tokens)}
    if source.calibration_tokens:
        budgets["calibration"] = int(source.calibration_tokens)
    budgets["validation"] = int(source.validation_tokens)
    return budgets


class TokenArtifactWriter:
    """Write bounded token-only shards with an exact split-token contract."""

    def __init__(self, source: R0SourceSpec, mix: R0DataMixSpec, *, overwrite: bool) -> None:
        self.source = source
        self.mix = mix
        self.root = Path(source.artifact_dir)
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(f"R0 text artifact exists: {self.root}")
            shutil.rmtree(self.root)
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.staging = Path(tempfile.mkdtemp(prefix=f".{self.root.name}.", dir=self.root.parent))
        self.buffer: list[dict[str, Any]] = []
        self.buffer_tokens = 0
        self.shards: list[dict[str, Any]] = []
        self.split_tokens = {"train": 0, "validation": 0}
        if source.calibration_tokens:
            self.split_tokens["calibration"] = 0
        self.record_count = 0

    def add(self, token_ids: Sequence[int], *, split: str, provenance: Mapping[str, Any]) -> None:
        if split not in self.split_tokens or not token_ids:
            raise ValueError(f"invalid token artifact record split/count: {split}/{len(token_ids)}")
        ids = [int(value) for value in token_ids]
        self.buffer.append(
            {
                "record_id": f"{self.source.role}-{self.record_count:09d}",
                "source_role": self.source.role,
                "split": split,
                "token_ids": torch.tensor(ids, dtype=torch.int32),
                "token_ids_sha256": token_ids_sha256(ids),
                "provenance": dict(provenance),
            }
        )
        self.record_count += 1
        self.buffer_tokens += len(ids)
        self.split_tokens[split] += len(ids)
        if self.buffer_tokens >= self.mix.shard_tokens:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        relative = f"shard_{len(self.shards):05d}.pt"
        path = self.staging / relative
        torch.save({"records": self.buffer}, path)
        self.shards.append(
            {
                "path": relative,
                "records": len(self.buffer),
                "tokens": self.buffer_tokens,
                "sha256": _sha256_file(path),
            }
        )
        self.buffer = []
        self.buffer_tokens = 0

    def finish(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        self.flush()
        expected = _source_split_budgets(self.source)
        if self.split_tokens != expected:
            raise ValueError(f"token budget mismatch: {self.split_tokens} != {expected}")
        manifest = {
            # Do not rewrite the meaning of historical two-way artifacts.
            "schema_version": 2 if self.source.calibration_tokens else 1,
            "artifact_type": "delta_r0_frozen_token_source",
            "source_role": self.source.role,
            "dataset": {
                "id": self.source.dataset_id,
                "revision": self.source.revision,
                "subset": self.source.subset,
                "split": self.source.split,
            },
            "selection": self.source.selection,
            "seed": self.source.seed,
            "sequence_length": self.mix.sequence_length,
            "tokenizer": {
                "id": self.mix.tokenizer_id,
                "revision": self.mix.tokenizer_revision,
            },
            "tokens": self.split_tokens,
            "records": self.record_count,
            "shards": self.shards,
            **dict(metadata),
        }
        _atomic_json(self.staging / "manifest.json", manifest)
        os.replace(self.staging, self.root)
        return manifest


def _budgeted_chunks(
    token_stream: Iterable[tuple[list[int], Mapping[str, Any]]],
    source: R0SourceSpec,
    sequence_length: int,
) -> Iterator[tuple[list[int], str, Mapping[str, Any]]]:
    """Cut records to exact budgets without leaking one row across the split."""

    remaining = _source_split_budgets(source)
    split_order = list(remaining)
    split_index = 0
    for ids, provenance in token_stream:
        split = split_order[split_index]
        if remaining[split] == 0:
            split_index += 1
            if split_index == len(split_order):
                return
            split = split_order[split_index]
        offset = 0
        while offset < len(ids):
            take = min(sequence_length, remaining[split], len(ids) - offset)
            if take:
                yield ids[offset:offset + take], split, provenance
                offset += take
                remaining[split] -= take
                if remaining[split] == 0:
                    # Never let one source row cross a train/calibration/validation
                    # boundary.  Its unused suffix is deliberately discarded.
                    split_index += 1
                    if split_index == len(split_order):
                        return
                    break
    raise RuntimeError(f"{source.role} exhausted before exact token budgets were filled")


def _valid_lmsys_messages(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        return None
    retained: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, Mapping):
            return None
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return None
        retained.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in retained):
        return None
    return retained


def materialize_hf_text_source(
    mix: R0DataMixSpec,
    role: Literal["fineweb", "lmsys"],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze one pinned HF source in published order without storing source text."""

    from datasets import load_dataset
    from transformers import AutoTokenizer

    source = next(item for item in mix.sources if item.role == role)
    tokenizer = AutoTokenizer.from_pretrained(
        mix.tokenizer_id, revision=mix.tokenizer_revision
    )
    dataset = load_dataset(
        source.dataset_id,
        source.subset,
        split=source.split,
        revision=source.revision,
        streaming=True,
    )
    stats = {"rows_seen": 0, "rows_retained": 0, "rows_skipped": 0}

    def records() -> Iterator[tuple[list[int], Mapping[str, Any]]]:
        for row_index, row in enumerate(dataset):
            stats["rows_seen"] += 1
            if role == "fineweb":
                text = row.get(source.text_field or "")
                if not isinstance(text, str) or not text.strip():
                    stats["rows_skipped"] += 1
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                eos_appended = tokenizer.eos_token_id is not None
                if eos_appended:
                    ids.append(int(tokenizer.eos_token_id))
                provenance = {
                    "dataset_row_index": row_index,
                    "source_row_id": row.get("id"),
                    "document_eos_appended": eos_appended,
                    "eos_token_id": tokenizer.eos_token_id,
                }
            else:
                messages = _valid_lmsys_messages(row.get(source.conversation_field or ""))
                if messages is None:
                    stats["rows_skipped"] += 1
                    continue
                original_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )
                ids = original_ids[: mix.sequence_length]
                provenance = {
                    "dataset_row_index": row_index,
                    "source_row_id": row.get("id"),
                    "conversation_id": row.get("conversation_id"),
                    "turns_retained": len(messages),
                    "original_token_count": len(original_ids),
                    "right_truncated_to_sequence_length": (
                        len(original_ids) > mix.sequence_length
                    ),
                }
            stats["rows_retained"] += 1
            yield [int(value) for value in ids], provenance

    writer = TokenArtifactWriter(source, mix, overwrite=overwrite)
    if role == "fineweb":
        for ids, split, provenance in _budgeted_chunks(
            records(), source, mix.sequence_length
        ):
            writer.add(ids, split=split, provenance=provenance)
    else:
        # Conversation boundaries are retained. The final conversation in a split
        # may be shortened to hit the exact budget; its unused suffix is discarded,
        # never reinterpreted as a new validation conversation without context.
        remaining = _source_split_budgets(source)
        split_order = list(remaining)
        split_index = 0
        for ids, provenance in records():
            split = split_order[split_index]
            if remaining[split] == 0:
                split_index += 1
                if split_index == len(split_order):
                    break
                split = split_order[split_index]
            take = min(len(ids), remaining[split])
            writer.add(ids[:take], split=split, provenance=provenance)
            remaining[split] -= take
            if not any(remaining.values()):
                break
        if any(remaining.values()):
            raise RuntimeError(f"LMSYS exhausted before exact token budgets: {remaining}")
    return writer.finish(
        {
            "row_policy": (
                "plain text; no chat template; append one EOS; split into <=1024-token records"
                if role == "fineweb"
                else "retain all supported system/user/assistant turns; canonical chat template; "
                "skip malformed/unsupported rows; right-truncate each conversation to <=1024; "
                "never join or reinterpret conversation suffixes"
            ),
            "row_statistics": stats,
            "tokenizer_fingerprint": _tokenizer_fingerprint(tokenizer),
            "chat_template_sha256": (
                hashlib.sha256(str(tokenizer.chat_template).encode("utf-8")).hexdigest()
                if role == "lmsys"
                else None
            ),
            "truncation_policy": "right; source/conversation boundary retained",
            "eos_policy": (
                "append canonical EOS after each FineWeb document"
                if role == "fineweb"
                else "canonical chat-template EOS behavior"
            ),
        }
    )


def freeze_contrastive_text_source(
    mix: R0DataMixSpec, *, overwrite: bool = False
) -> dict[str, Any]:
    """Copy frozen generated token IDs out of a cache; never copy activations."""

    source = next(item for item in mix.sources if item.role == "contrastive")
    input_manifest = Path(source.input_manifest or "")
    manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    examples_path = input_manifest.parent / manifest["examples_file"]
    examples = [
        json.loads(line)
        for line in examples_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    writer = TokenArtifactWriter(source, mix, overwrite=overwrite)
    for split, budget in _source_split_budgets(source).items():
        remaining = budget
        for index, row in enumerate(examples):
            if row["split"] != split:
                continue
            original_ids = [int(value) for value in row["token_ids"]]
            offset = 0
            while offset < len(original_ids) and remaining:
                take = min(mix.sequence_length, remaining, len(original_ids) - offset)
                ids = original_ids[offset:offset + take]
                writer.add(
                    ids,
                    split=split,
                    provenance={
                        "input_example_index": index,
                        "prompt_id": row["prompt_id"],
                        "generator_side": row["generator_side"],
                        "stop_reason": row["generation"]["stop_reason"],
                        "input_token_sha256": token_ids_sha256(original_ids),
                        "input_token_offset": offset,
                    },
                )
                remaining -= take
                offset += take
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError(
                f"contrastive {split} split is short by {remaining} frozen tokens"
            )
    return writer.finish(
        {
            "input_manifest": str(input_manifest.resolve()),
            "input_manifest_sha256": _sha256_file(input_manifest),
            "generation": manifest["generation"],
            "prompt_source": manifest["prompt_source"],
            "same_frozen_token_ids_used_for_measurement": True,
        }
    )


def validate_text_artifact(
    path: str | Path,
    source: R0SourceSpec,
    *,
    mix: R0DataMixSpec | None = None,
) -> dict[str, Any]:
    root = Path(path).parent if Path(path).name == "manifest.json" else Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] not in {1, 2}:
        raise ValueError("unsupported R0 text artifact schema")
    if source.calibration_tokens and manifest["schema_version"] != 2:
        raise ValueError("three-way R0 source requires schema version 2")
    if manifest["source_role"] != source.role:
        raise ValueError("R0 source role mismatch")
    expected_dataset = {
        "id": source.dataset_id,
        "revision": source.revision,
        "subset": source.subset,
        "split": source.split,
    }
    if manifest.get("dataset") != expected_dataset:
        raise ValueError("R0 source dataset identity/revision mismatch")
    if mix is not None:
        if manifest.get("sequence_length") != mix.sequence_length:
            raise ValueError("R0 source sequence length differs from common mix")
        if manifest.get("tokenizer") != {
            "id": mix.tokenizer_id,
            "revision": mix.tokenizer_revision,
        }:
            raise ValueError("R0 source tokenizer identity/revision mismatch")
    observed = {split: 0 for split in _source_split_budgets(source)}
    records = 0
    record_ids: set[str] = set()
    for shard in manifest["shards"]:
        shard_path = (root / shard["path"]).resolve()
        if root.resolve() not in shard_path.parents:
            raise ValueError("R0 shard path escapes its artifact root")
        if _sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"R0 token shard checksum mismatch: {shard_path}")
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        for record in payload["records"]:
            ids = record["token_ids"].tolist()
            record_id = str(record.get("record_id", ""))
            if not record_id or record_id in record_ids:
                raise ValueError("R0 record IDs must be non-empty and unique")
            record_ids.add(record_id)
            if record["source_role"] != source.role:
                raise ValueError("record source role differs from manifest")
            if not isinstance(record.get("provenance"), Mapping):
                raise ValueError("R0 record lacks source-row provenance")
            if token_ids_sha256(ids) != record["token_ids_sha256"]:
                raise ValueError("record token checksum mismatch")
            if not 0 < len(ids) <= manifest["sequence_length"]:
                raise ValueError("record violates sequence length")
            if record["split"] not in observed:
                raise ValueError("record uses a split absent from the source contract")
            observed[record["split"]] += len(ids)
            records += 1
    expected = _source_split_budgets(source)
    if observed != expected or observed != manifest["tokens"]:
        raise ValueError(f"R0 artifact token budgets differ: {observed} != {expected}")
    return {"source_role": source.role, "records": records, "tokens": observed}


class _ArtifactReader:
    def __init__(self, source: R0SourceSpec, split: str) -> None:
        self.source = source
        self.split = split
        self.root = Path(source.artifact_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.shard_index = 0
        self.row_index = 0
        self.token_offset = 0
        self.wrap_count = 0
        self._loaded_index = -1
        self._records: list[dict[str, Any]] = []

    def _load(self) -> None:
        if self._loaded_index == self.shard_index:
            return
        shard = self.manifest["shards"][self.shard_index]
        payload = torch.load(
            self.root / shard["path"], map_location="cpu", weights_only=False
        )
        self._records = [record for record in payload["records"] if record["split"] == self.split]
        self._loaded_index = self.shard_index

    def current(self) -> dict[str, Any]:
        while True:
            self._load()
            if self.row_index < len(self._records):
                return self._records[self.row_index]
            self.shard_index += 1
            self.row_index = 0
            self.token_offset = 0
            self._loaded_index = -1
            if self.shard_index >= len(self.manifest["shards"]):
                self.shard_index = 0
                self.wrap_count += 1

    def advance_record(self) -> None:
        self.row_index += 1
        self.token_offset = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "shard_index": self.shard_index,
            "row_index": self.row_index,
            "token_offset": self.token_offset,
            "wrap_count": self.wrap_count,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        self.shard_index = int(state["shard_index"])
        self.row_index = int(state["row_index"])
        self.token_offset = int(state["token_offset"])
        self.wrap_count = int(state["wrap_count"])
        self._loaded_index = -1


class R0StreamingDeltaSource(DeltaActivationSource):
    """Online paired-forward source with an exactly checkpointed logical cursor."""

    def __init__(
        self,
        cfg: DeltaCrosscoderExperimentConfig,
        mix_path: str | Path,
        *,
        device_name: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.mix_path = Path(mix_path)
        self.mix = load_r0_mix_config(self.mix_path)
        if self.mix.batch_size_tokens != cfg.delta_crosscoder.batch_size_tokens:
            raise ValueError("R0 mix and trainer batch sizes differ")
        for source in self.mix.sources:
            validate_text_artifact(source.artifact_dir, source, mix=self.mix)
        tokenizer, self.tokenizer_provenance = _audit_tokenizers(cfg)
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("R0 tokenizer requires pad_token_id")
        self.device = torch.device(
            device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.models = _load_measurement_models(cfg, self.device)
        self._activation_dim = int(self.models[0].config.hidden_size)  # type: ignore[attr-defined]
        self.readers = {
            source.role: _ArtifactReader(source, "train") for source in self.mix.sources
        }
        self.schedule_index = 0
        self.tokens_consumed_total = 0
        self.tokens_consumed_by_source = {role: 0 for role in SOURCE_ROLES}
        self.paired_forward_seconds = 0.0
        self.paired_forward_tokens = 0
        self.same_token_checks = 0

    @property
    def activation_dim(self) -> int:
        return self._activation_dim

    @torch.inference_mode()
    def _paired_forward(self, token_ids: Tensor) -> Tensor:
        ids_cpu = token_ids.to(dtype=torch.long, device="cpu")
        digest = token_ids_sha256(ids_cpu.tolist())
        ids = ids_cpu[None].to(self.device)
        mask = torch.ones_like(ids)
        outputs: list[Tensor] = []
        started = time.perf_counter()
        for model, spec in zip(self.models, self.cfg.models, strict=True):
            capture = _Capture()
            with ExitStack() as stack:
                _register_capture_hooks(model, spec, capture, stack)
                backbone = getattr(model, "model", model)
                backbone(input_ids=ids, attention_mask=mask, use_cache=False)
                value = capture.pop("R")[0]
                if value.shape != (len(ids_cpu), self.activation_dim):
                    raise ValueError("R0 hook output does not match exact token sequence")
                outputs.append(value.detach().to(device="cpu", dtype=torch.bfloat16))
        # One hash is intentionally used for both forwards; there is no second tokenization.
        if digest != token_ids_sha256(ids_cpu.tolist()):
            raise AssertionError("same-token runtime invariant failed")
        self.same_token_checks += 1
        self.paired_forward_tokens += len(ids_cpu)
        self.paired_forward_seconds += time.perf_counter() - started
        return torch.stack(outputs, dim=1)

    def _next_batch(
        self, role: str, readers: Mapping[str, _ArtifactReader], *, count: bool
    ) -> dict[str, Any]:
        reader = readers[role]
        chunks: list[Tensor] = []
        token_chunks: list[Tensor] = []
        spans: list[dict[str, Any]] = []
        batch_position = 0
        remaining = self.mix.batch_size_tokens
        while remaining:
            record = reader.current()
            ids = record["token_ids"]
            activations = self._paired_forward(ids)
            available = len(ids) - reader.token_offset
            take = min(remaining, available)
            chunks.append(activations[reader.token_offset:reader.token_offset + take])
            token_chunks.append(ids[reader.token_offset:reader.token_offset + take].to(dtype=torch.long))
            spans.append({
                "sequence_id": f"{role}:{reader.shard_index}:{reader.row_index}",
                "start": batch_position,
                "length": take,
                "sequence_offset": reader.token_offset,
            })
            batch_position += take
            reader.token_offset += take
            remaining -= take
            if reader.token_offset == len(ids):
                reader.advance_record()
        batch = torch.cat(chunks, dim=0)
        if batch.shape != (self.mix.batch_size_tokens, 2, self.activation_dim):
            raise AssertionError("R0 activation batch violates fixed token shape")
        if count:
            self.tokens_consumed_total += len(batch)
            self.tokens_consumed_by_source[role] += len(batch)
        return {
            "activations": batch,
            "input_ids": torch.cat(token_chunks, dim=0),
            "sequence_spans": spans,
            "source_role": role,
        }

    def iter_batches(
        self, split: str, *, shuffle: bool, epoch: int = 0
    ) -> Iterator[dict[str, Any]]:
        del shuffle, epoch
        if split == "train":
            while True:
                role = self.mix.source_schedule[
                    self.schedule_index % len(self.mix.source_schedule)
                ]
                self.schedule_index += 1
                yield self._next_batch(role, self.readers, count=True)
        elif split == "validation":
            readers = {
                source.role: _ArtifactReader(source, "validation")
                for source in self.mix.sources
            }
            for role in self.mix.source_schedule:
                yield self._next_batch(role, readers, count=False)
        else:
            raise ValueError(f"unknown R0 split: {split}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "schedule_index": self.schedule_index,
            "readers": {role: reader.state_dict() for role, reader in self.readers.items()},
            "tokens_consumed_total": self.tokens_consumed_total,
            "tokens_consumed_by_source": dict(self.tokens_consumed_by_source),
            "paired_forward_seconds": self.paired_forward_seconds,
            "paired_forward_tokens": self.paired_forward_tokens,
            "same_token_checks": self.same_token_checks,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.schedule_index = int(state["schedule_index"])
        for role, reader_state in state["readers"].items():
            self.readers[role].load_state_dict(reader_state)
        self.tokens_consumed_total = int(state["tokens_consumed_total"])
        self.tokens_consumed_by_source = {
            role: int(value) for role, value in state["tokens_consumed_by_source"].items()
        }
        self.paired_forward_seconds = float(state["paired_forward_seconds"])
        self.paired_forward_tokens = int(state["paired_forward_tokens"])
        self.same_token_checks = int(state["same_token_checks"])

    def counters(self) -> dict[str, Any]:
        unique = {
            source.role: min(self.tokens_consumed_by_source[source.role], source.train_tokens)
            for source in self.mix.sources
        }
        repeated = {
            source.role: max(0, self.tokens_consumed_by_source[source.role] - source.train_tokens)
            for source in self.mix.sources
        }
        return {
            "tokens_consumed_total": self.tokens_consumed_total,
            "tokens_consumed_by_source": dict(self.tokens_consumed_by_source),
            "unique_corpus_tokens_consumed_by_source": unique,
            "repeated_tokens_by_source": repeated,
            "total_unique_corpus_tokens": self.mix.unique_tokens,
            "total_optimizer_token_updates_planned": self.mix.optimizer_token_updates,
            "paired_forward_seconds": self.paired_forward_seconds,
            "paired_forward_tokens": self.paired_forward_tokens,
            "same_token_runtime_checks": self.same_token_checks,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "mode": "r0_stream",
            "mix_config": str(self.mix_path.resolve()),
            "mix_config_sha256": _sha256_file(self.mix_path),
            "scientific_label": self.mix.scientific_label,
            "organism_training_tokens": self.mix.organism_training_tokens,
            "model_axis": ["base", "target"],
            "same_token_invariant": "one CPU token tensor is replayed unchanged through both models",
            "source_schedule": list(self.mix.source_schedule),
            "sources": [
                {
                    **source.model_dump(mode="json"),
                    "manifest_sha256": _sha256_file(Path(source.artifact_dir) / "manifest.json"),
                }
                for source in self.mix.sources
            ],
            "tokenizer": self.tokenizer_provenance,
        }


def write_mix_summary(mix: R0DataMixSpec, output: str | Path) -> None:
    """Write an auditable corpus/update wrap calculation without materializing data."""

    # Count the actual finite schedule prefix. A ratio-based integer division is
    # wrong whenever optimizer_steps is not an exact multiple of the 20-step cycle.
    source_batches = {
        role: sum(
            mix.source_schedule[index % len(mix.source_schedule)] == role
            for index in range(mix.optimizer_steps)
        )
        for role in SOURCE_ROLES
    }
    source_updates = {
        role: source_batches[role] * mix.batch_size_tokens
        for role in SOURCE_ROLES
    }
    source_unique = {source.role: source.train_tokens for source in mix.sources}
    summary = {
        "scientific_label": mix.scientific_label,
        "organism_training_tokens": mix.organism_training_tokens,
        "unique_corpus_tokens": mix.unique_tokens,
        "optimizer_token_updates": mix.optimizer_token_updates,
        "repeat_tokens": mix.optimizer_token_updates - mix.unique_tokens,
        "repeat_fraction_of_updates": (
            mix.optimizer_token_updates - mix.unique_tokens
        ) / mix.optimizer_token_updates,
        "source_unique_tokens": source_unique,
        "source_optimizer_batches": source_batches,
        "source_optimizer_token_updates": source_updates,
        "source_repeat_tokens": {
            role: source_updates[role] - source_unique[role] for role in SOURCE_ROLES
        },
        "schedule": list(mix.source_schedule),
    }
    _atomic_json(Path(output), summary)


def dump_resolved_mix(mix: R0DataMixSpec) -> str:
    return yaml.safe_dump(mix.model_dump(mode="json"), sort_keys=False)
