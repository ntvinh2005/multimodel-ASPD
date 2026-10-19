"""Paper-style contrastive activation cache for Delta-Crosscoder.

For every prompt, Base and Target each generate one continuation.  Each complete
prompt/answer token sequence is then replayed unchanged through both models.  The
cache therefore stores Delta(s) = h_target(s) - h_base(s) for one fixed s; it never
subtracts activations belonging to different generated responses.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn

from aspd.multimodel.baselines.delta_crosscoder import DeltaActivationSource
from aspd.multimodel.cache import (
    DTYPES,
    _Capture,
    _register_capture_hooks,
    _tokenizer_fingerprint,
)

if TYPE_CHECKING:
    from aspd.multimodel.baselines.delta_crosscoder import (
        DeltaContrastiveGenerationSpec,
        DeltaCrosscoderExperimentConfig,
    )

SCHEMA_VERSION = 1
GENERATOR_SIDES = ("base_generated", "target_generated")
MODEL_AXIS = ("base", "target")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _repository_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return _sha256_json(list(token_ids))


def load_frozen_prompts(
    path: str | Path,
    *,
    prompt_id_field: str = "prompt_id",
    prompt_field: str = "prompt",
    max_prompts: int | None = None,
) -> list[dict[str, str]]:
    """Load exact prompt IDs/text from a committed JSONL file without runtime sampling."""

    prompt_path = Path(path)
    prompts: list[dict[str, str]] = []
    seen: set[str] = set()
    with prompt_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = str(row[prompt_id_field])
            prompt = str(row[prompt_field]).strip()
            if not prompt or prompt_id in seen:
                raise ValueError(
                    f"invalid or duplicate prompt at {prompt_path}:{line_number}: {prompt_id!r}"
                )
            seen.add(prompt_id)
            prompts.append({"prompt_id": prompt_id, "prompt": prompt})
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    if not prompts:
        raise ValueError(f"prompt file is empty: {prompt_path}")
    return prompts


def load_organism_dataset(
    path: str | Path,
    *,
    expected_sha256: str,
    max_examples: int,
) -> tuple[list[dict[str, Any]], int]:
    """Load the exact checksummed organism JSONL without silently repairing its schema."""

    dataset_path = Path(path)
    observed_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"organism dataset checksum mismatch: {observed_sha256} != {expected_sha256}"
        )
    selected: list[dict[str, Any]] = []
    total = 0
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) != 2
                or [message.get("role") for message in messages] != ["user", "assistant"]
                or any(not isinstance(message.get("content"), str) for message in messages)
            ):
                raise ValueError(f"unexpected organism schema at line {line_number}")
            total += 1
            if len(selected) < max_examples:
                selected.append({"dataset_index": total - 1, "messages": messages})
    if len(selected) < max_examples:
        raise ValueError(f"organism dataset has only {total} examples; requested {max_examples}")
    return selected, total


def canonical_prompt_ids(tokenizer: object, prompt: str, *, enable_thinking: bool) -> list[int]:
    """Render one canonical Qwen user turn and include the assistant generation prefix."""

    ids = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if isinstance(ids, Tensor):
        ids = ids.tolist()
    if not isinstance(ids, list) or not ids or not all(isinstance(item, int) for item in ids):
        raise TypeError("canonical chat template did not produce a non-empty integer token list")
    return ids


def canonical_conversation_ids(
    tokenizer: object, messages: Sequence[Mapping[str, str]]
) -> list[int]:
    """Render a complete user/assistant training example with the canonical template."""

    ids = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        list(messages), tokenize=True, add_generation_prompt=False
    )
    if isinstance(ids, Tensor):
        ids = ids.tolist()
    if not isinstance(ids, list) or not ids or not all(isinstance(item, int) for item in ids):
        raise TypeError("canonical chat template did not produce a non-empty integer token list")
    return ids


def apply_sequence_limit(
    token_ids: Sequence[int], max_length: int, *, policy: Literal["right"] = "right"
) -> tuple[list[int], dict[str, Any]]:
    """Apply explicit right truncation and return complete truncation provenance."""

    if policy != "right":
        raise ValueError(f"unsupported truncation policy: {policy}")
    original = list(token_ids)
    retained = original[:max_length]
    return retained, {
        "policy": policy,
        "original_token_count": len(original),
        "retained_token_count": len(retained),
        "truncated_token_count": len(original) - len(retained),
        "was_truncated": len(original) > max_length,
    }


def _eos_ids(model: nn.Module) -> list[int]:
    raw = getattr(model.generation_config, "eos_token_id", None)  # type: ignore[attr-defined]
    if raw is None:
        raw = getattr(model.config, "eos_token_id", None)  # type: ignore[attr-defined]
    if raw is None:
        return []
    return [int(value) for value in raw] if isinstance(raw, list) else [int(raw)]


@torch.inference_mode()
def generate_continuation(
    model: nn.Module,
    prompt_ids: Sequence[int],
    generation: DeltaContrastiveGenerationSpec,
    *,
    device: torch.device,
    pad_token_id: int,
) -> tuple[list[int], dict[str, Any]]:
    """Greedy generation with an explicit deterministic contract and stop provenance."""

    if generation.temperature != 0.0:
        raise ValueError("Phase-2A materialization requires temperature=0 deterministic decoding")
    torch.manual_seed(generation.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(generation.seed)
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    eos_ids = _eos_ids(model)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": False,
        "max_new_tokens": generation.max_new_tokens,
        "pad_token_id": pad_token_id,
    }
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids
    generated = model.generate(**kwargs)  # type: ignore[attr-defined]
    answer_ids = [int(value) for value in generated[0, len(prompt_ids) :].tolist()]
    first_eos: int | None = None
    for index, value in enumerate(answer_ids):
        if value in eos_ids:
            first_eos = index
            break
    if first_eos is not None:
        answer_ids = answer_ids[: first_eos + 1]
    return answer_ids, {
        "eos_token_ids": eos_ids,
        "stopped_on_eos": first_eos is not None,
        "stop_reason": "eos" if first_eos is not None else "max_new_tokens",
    }


@torch.inference_mode()
def generate_continuations_batch(
    model: nn.Module,
    prompt_ids: Sequence[Sequence[int]],
    generation: DeltaContrastiveGenerationSpec,
    *,
    device: torch.device,
    pad_token_id: int,
) -> list[tuple[list[int], dict[str, Any]]]:
    """Batched deterministic generation with explicit left-padding removal."""

    if generation.temperature != 0.0:
        raise ValueError("R0 contrastive materialization requires temperature=0")
    torch.manual_seed(generation.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(generation.seed)
    width = max(len(ids) for ids in prompt_ids)
    input_ids = torch.full((len(prompt_ids), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(prompt_ids):
        input_ids[row, width - len(ids) :] = torch.tensor(ids, device=device)
        attention_mask[row, width - len(ids) :] = 1
    eos_ids = _eos_ids(model)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": False,
        "max_new_tokens": generation.max_new_tokens,
        "pad_token_id": pad_token_id,
    }
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids
    generated = model.generate(**kwargs)  # type: ignore[attr-defined]
    results: list[tuple[list[int], dict[str, Any]]] = []
    for row in range(len(prompt_ids)):
        answer = [int(value) for value in generated[row, width:].tolist()]
        first_eos = next((index for index, value in enumerate(answer) if value in eos_ids), None)
        if first_eos is not None:
            answer = answer[: first_eos + 1]
        results.append(
            (
                answer,
                {
                    "eos_token_ids": eos_ids,
                    "stopped_on_eos": first_eos is not None,
                    "stop_reason": "eos" if first_eos is not None else "max_new_tokens",
                },
            )
        )
    return results


def deterministic_validation_prompt_ids(
    prompts: Sequence[Mapping[str, str]], validation_count: int, seed: int
) -> set[str]:
    if validation_count >= len(prompts):
        raise ValueError("validation_prompts must be smaller than the materialized prompt count")
    ids = [str(prompt["prompt_id"]) for prompt in prompts]
    random.Random(seed).shuffle(ids)
    return set(ids[:validation_count])


def _pad_sequences(
    records: Sequence[Mapping[str, Any]], pad_token_id: int
) -> tuple[Tensor, Tensor]:
    width = max(len(record["token_ids"]) for record in records)
    ids = torch.full((len(records), width), pad_token_id, dtype=torch.long)
    valid = torch.zeros((len(records), width), dtype=torch.bool)
    for row, record in enumerate(records):
        values = torch.tensor(record["token_ids"], dtype=torch.long)
        ids[row, : len(values)] = values
        valid[row, : len(values)] = True
    return ids, valid


@torch.inference_mode()
def extract_paired_activations(
    records: Sequence[Mapping[str, Any]],
    models: Sequence[nn.Module],
    model_specs: Sequence[Any],
    *,
    device: torch.device,
    pad_token_id: int,
    batch_size: int = 4,
) -> list[Tensor]:
    """Replay each prepared canonical token batch through both models unchanged."""

    if len(models) != 2 or len(model_specs) != 2:
        raise ValueError("paired extraction requires exactly two models")
    by_model: list[list[Tensor]] = [[], []]
    batches = [records[start : start + batch_size] for start in range(0, len(records), batch_size)]
    # Build each CPU token tensor once. Both model passes consume copies of this exact tensor.
    prepared = [_pad_sequences(batch, pad_token_id) for batch in batches]
    for model_index, (model, spec) in enumerate(zip(models, model_specs, strict=True)):
        capture = _Capture()
        with ExitStack() as stack:
            _register_capture_hooks(model, spec, capture, stack)
            for (input_ids_cpu, valid_cpu), batch in zip(prepared, batches, strict=True):
                input_ids = input_ids_cpu.to(device)
                valid = valid_cpu.to(device)
                backbone = getattr(model, "model", model)
                backbone(input_ids=input_ids, attention_mask=valid, use_cache=False)
                activations = capture.pop("R")
                if activations.shape[:2] != input_ids.shape:
                    raise ValueError("hook activation grid does not match canonical token grid")
                for row, record in enumerate(batch):
                    count = len(record["token_ids"])
                    by_model[model_index].append(
                        activations[row, :count].detach().float().cpu().contiguous()
                    )
    paired: list[Tensor] = []
    for index in range(len(records)):
        if by_model[0][index].shape != by_model[1][index].shape:
            raise ValueError("paired activation shapes differ for the same token sequence")
        paired.append(torch.stack((by_model[0][index], by_model[1][index]), dim=1))
    return paired


def write_contrastive_cache(
    output_dir: str | Path,
    records: Sequence[dict[str, Any]],
    activations: Sequence[Tensor],
    *,
    manifest_metadata: Mapping[str, Any],
    sequences_per_shard: int,
    activation_dtype: torch.dtype,
    pad_token_id: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write self-describing sequence shards and their exact JSONL provenance."""

    if len(records) != len(activations) or not records:
        raise ValueError("records and activations must have the same non-zero length")
    root = Path(output_dir)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"contrastive cache already exists: {root}")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        stored_records = [dict(record) for record in records]
        split_manifest: dict[str, Any] = {}
        activation_dim = int(activations[0].shape[-1])
        for split in ("train", "validation"):
            indices = [index for index, record in enumerate(records) if record["split"] == split]
            split_dir = staging / split
            split_dir.mkdir(parents=True, exist_ok=True)
            shard_entries: list[dict[str, Any]] = []
            token_count = 0
            for shard_index, start in enumerate(range(0, len(indices), sequences_per_shard)):
                selected = indices[start : start + sequences_per_shard]
                selected_records = [records[index] for index in selected]
                input_ids, valid_tokens = _pad_sequences(selected_records, pad_token_id)
                width = input_ids.shape[1]
                paired = torch.zeros(
                    len(selected), width, 2, activation_dim, dtype=activation_dtype
                )
                for row, global_index in enumerate(selected):
                    value = activations[global_index]
                    if value.ndim != 3 or value.shape[1:] != (2, activation_dim):
                        raise ValueError("activation must have shape [tokens,2,activation_dim]")
                    paired[row, : value.shape[0]] = value.to(dtype=activation_dtype)
                    token_count += value.shape[0]
                    stored_records[global_index]["storage"] = {
                        "shard": f"{split}/shard_{shard_index:05d}.pt",
                        "row": row,
                    }
                relative = f"{split}/shard_{shard_index:05d}.pt"
                torch.save(
                    {
                        "input_ids": input_ids,
                        "valid_tokens": valid_tokens,
                        "activations": paired,
                        "example_indices": torch.tensor(selected, dtype=torch.long),
                    },
                    staging / relative,
                )
                shard_entries.append(
                    {
                        "path": relative,
                        "sequences": len(selected),
                        "valid_tokens": int(valid_tokens.sum()),
                    }
                )
            split_manifest[split] = {
                "sequences": len(indices),
                "valid_tokens": token_count,
                "shards": shard_entries,
            }

        examples_path = staging / "examples.jsonl"
        with examples_path.open("w", encoding="utf-8") as handle:
            for record in stored_records:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "delta_contrastive_paired_activation_cache",
            "model_axis": list(MODEL_AXIS),
            "generator_sides": list(GENERATOR_SIDES),
            "activation_dim": activation_dim,
            "activation_dtype": str(activation_dtype).removeprefix("torch."),
            "pad_token_id": pad_token_id,
            "examples_file": "examples.jsonl",
            "examples_sha256": hashlib.sha256(examples_path.read_bytes()).hexdigest(),
            "splits": split_manifest,
            **dict(manifest_metadata),
        }
        fingerprint_payload = dict(manifest)
        manifest["cache_fingerprint"] = _sha256_json(fingerprint_payload)
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return _read_json(root / "manifest.json")


def _load_examples(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = root / str(manifest["examples_file"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["examples_sha256"]:
        raise ValueError("contrastive examples checksum mismatch")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_contrastive_cache(
    manifest_path: str | Path,
    cfg: DeltaCrosscoderExperimentConfig | None = None,
) -> dict[str, Any]:
    """Validate same-text, two-branch, padding, provenance, and tensor invariants."""

    path = Path(manifest_path)
    root = path.parent
    manifest = _read_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported contrastive cache schema: {manifest.get('schema_version')}")
    if manifest.get("model_axis") != list(MODEL_AXIS):
        raise ValueError("activation model axis must be [base,target]")
    examples = _load_examples(root, manifest)
    required = {
        "example_id",
        "prompt_id",
        "prompt",
        "generator_side",
        "generated_answer",
        "rendered_full_text",
        "token_ids",
        "valid_token_range",
        "generation",
        "models",
        "tokenizer",
        "hook",
        "measurement_input_sha256",
        "truncation",
        "split",
    }
    prompt_sides: dict[str, set[str]] = {}
    for example in examples:
        missing = required - example.keys()
        if missing:
            raise ValueError(
                f"example {example.get('example_id')} lacks provenance: {sorted(missing)}"
            )
        prompt_sides.setdefault(example["prompt_id"], set()).add(example["generator_side"])
        expected_hash = token_ids_sha256(example["token_ids"])
        measured = example["measurement_input_sha256"]
        if measured != {"base": expected_hash, "target": expected_hash}:
            raise ValueError("same-text replay hash differs across measured models")
        if example["valid_token_range"] != [0, len(example["token_ids"])]:
            raise ValueError("valid token range does not match stored token IDs")
    for prompt_id, sides in prompt_sides.items():
        if sides != set(GENERATOR_SIDES):
            raise ValueError(f"prompt {prompt_id!r} does not have both generated branches")
    prompt_splits: dict[str, set[str]] = {}
    for example in examples:
        prompt_splits.setdefault(example["prompt_id"], set()).add(example["split"])
    if any(len(splits) != 1 for splits in prompt_splits.values()):
        raise ValueError("both generated branches of each prompt must share one split")

    seen_indices: set[int] = set()
    for split in ("train", "validation"):
        observed_tokens = 0
        observed_sequences = 0
        for entry in manifest["splits"][split]["shards"]:
            shard = torch.load(root / entry["path"], map_location="cpu", weights_only=True)
            ids, valid, acts = shard["input_ids"], shard["valid_tokens"], shard["activations"]
            if ids.shape != valid.shape or acts.shape[:2] != ids.shape or acts.shape[2] != 2:
                raise ValueError(f"invalid shard shapes in {entry['path']}")
            if acts.shape[-1] != manifest["activation_dim"]:
                raise ValueError("activation dimension differs from manifest")
            for row, index_tensor in enumerate(shard["example_indices"]):
                index = int(index_tensor)
                if index in seen_indices:
                    raise ValueError("example appears in more than one cache row")
                seen_indices.add(index)
                example = examples[index]
                if example["split"] != split:
                    raise ValueError("example split differs from shard split")
                count = len(example["token_ids"])
                if not bool(valid[row, :count].all()) or bool(valid[row, count:].any()):
                    raise ValueError("padding mask includes padding or excludes valid tokens")
                if ids[row, :count].tolist() != example["token_ids"]:
                    raise ValueError("tensor token IDs differ from example provenance")
                if bool((acts[row, count:] != 0).any()):
                    raise ValueError("padding activations must be zero and excluded")
            observed_tokens += int(valid.sum())
            observed_sequences += ids.shape[0]
        if observed_tokens != manifest["splits"][split]["valid_tokens"]:
            raise ValueError("split valid-token count mismatch")
        if observed_sequences != manifest["splits"][split]["sequences"]:
            raise ValueError("split sequence count mismatch")
    if seen_indices != set(range(len(examples))):
        raise ValueError("not every example is represented exactly once")
    if cfg is not None:
        expected_models = [model.model_id for model in cfg.models]
        if manifest["models"]["ids"] != expected_models:
            raise ValueError("cache model IDs differ from config")
        hooks = [model.grounding.model_dump(mode="json") for model in cfg.models]
        if manifest["models"]["hooks"] != hooks:
            raise ValueError("cache hooks differ from config")
    return {
        "prompts": len(prompt_sides),
        "sequences": len(examples),
        "train_tokens": manifest["splits"]["train"]["valid_tokens"],
        "validation_tokens": manifest["splits"]["validation"]["valid_tokens"],
        "cache_fingerprint": manifest["cache_fingerprint"],
    }


def validate_organism_cache(
    manifest_path: str | Path,
    cfg: DeltaCrosscoderExperimentConfig | None = None,
) -> dict[str, Any]:
    """Validate the separate same-token paired cache built from organism SFT examples."""

    path = Path(manifest_path)
    root = path.parent
    manifest = _read_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported organism cache schema: {manifest.get('schema_version')}")
    if manifest.get("artifact_type") != "delta_organism_paired_activation_cache":
        raise ValueError("not an organism activation cache")
    if manifest.get("model_axis") != list(MODEL_AXIS):
        raise ValueError("activation model axis must be [base,target]")
    examples = _load_examples(root, manifest)
    required = {
        "example_id",
        "dataset_index",
        "messages",
        "rendered_full_text",
        "token_ids",
        "valid_token_range",
        "models",
        "tokenizer",
        "hook",
        "measurement_input_sha256",
        "truncation",
        "split",
    }
    seen_indices: set[int] = set()
    for example in examples:
        missing = required - example.keys()
        if missing:
            raise ValueError(
                f"organism example {example.get('example_id')} lacks provenance: {sorted(missing)}"
            )
        digest = token_ids_sha256(example["token_ids"])
        if example["measurement_input_sha256"] != {"base": digest, "target": digest}:
            raise ValueError("same-text replay hash differs across measured models")
        if example["valid_token_range"] != [0, len(example["token_ids"])]:
            raise ValueError("valid token range does not match stored token IDs")

    for split in ("train", "validation"):
        observed_tokens = 0
        observed_sequences = 0
        for entry in manifest["splits"][split]["shards"]:
            shard = torch.load(root / entry["path"], map_location="cpu", weights_only=True)
            ids, valid, acts = shard["input_ids"], shard["valid_tokens"], shard["activations"]
            if ids.shape != valid.shape or acts.shape[:2] != ids.shape or acts.shape[2] != 2:
                raise ValueError(f"invalid shard shapes in {entry['path']}")
            if acts.shape[-1] != manifest["activation_dim"]:
                raise ValueError("activation dimension differs from manifest")
            for row, index_tensor in enumerate(shard["example_indices"]):
                index = int(index_tensor)
                if index in seen_indices:
                    raise ValueError("example appears in more than one cache row")
                seen_indices.add(index)
                example = examples[index]
                count = len(example["token_ids"])
                if example["split"] != split:
                    raise ValueError("example split differs from shard split")
                if ids[row, :count].tolist() != example["token_ids"]:
                    raise ValueError("tensor token IDs differ from example provenance")
                if not bool(valid[row, :count].all()) or bool(valid[row, count:].any()):
                    raise ValueError("padding mask includes padding or excludes valid tokens")
                if bool((acts[row, count:] != 0).any()):
                    raise ValueError("padding activations must be zero and excluded")
            observed_tokens += int(valid.sum())
            observed_sequences += ids.shape[0]
        if observed_tokens != manifest["splits"][split]["valid_tokens"]:
            raise ValueError("split valid-token count mismatch")
        if observed_sequences != manifest["splits"][split]["sequences"]:
            raise ValueError("split sequence count mismatch")
    if seen_indices != set(range(len(examples))):
        raise ValueError("not every example is represented exactly once")
    if cfg is not None:
        expected_models = [model.model_id for model in cfg.models]
        if manifest["models"]["ids"] != expected_models:
            raise ValueError("cache model IDs differ from config")
        hooks = [model.grounding.model_dump(mode="json") for model in cfg.models]
        if manifest["models"]["hooks"] != hooks:
            raise ValueError("cache hooks differ from config")
    return {
        "examples": len(examples),
        "train_tokens": manifest["splits"]["train"]["valid_tokens"],
        "validation_tokens": manifest["splits"]["validation"]["valid_tokens"],
        "cache_fingerprint": manifest["cache_fingerprint"],
    }


class ContrastiveDeltaSource(DeltaActivationSource):
    """Static cache reader implementing the DeltaActivationSource protocol."""

    def __init__(self, cfg: DeltaCrosscoderExperimentConfig) -> None:
        self._cfg = cfg
        self.manifest_path = Path(cfg.delta_crosscoder.data.contrastive_manifest or "")
        validate_contrastive_cache(self.manifest_path, cfg)
        self.root = self.manifest_path.parent
        self.manifest = _read_json(self.manifest_path)
        self.examples = _load_examples(self.root, self.manifest)

    @property
    def activation_dim(self) -> int:
        return int(self.manifest["activation_dim"])

    def iter_batches(
        self, split: str, *, shuffle: bool, epoch: int = 0
    ) -> Iterator[dict[str, Tensor]]:
        if split not in {"train", "validation"}:
            raise ValueError(f"unknown split: {split}")
        sequences: list[Tensor] = []
        for entry in self.manifest["splits"][split]["shards"]:
            shard = torch.load(self.root / entry["path"], map_location="cpu", weights_only=True)
            for row in range(shard["activations"].shape[0]):
                sequences.append(shard["activations"][row][shard["valid_tokens"][row]])
        order = list(range(len(sequences)))
        if shuffle:
            random.Random(self._cfg.delta_crosscoder.seed + epoch).shuffle(order)
        flattened = torch.cat([sequences[index] for index in order], dim=0)
        batch = self._cfg.delta_crosscoder.batch_size_tokens
        if (
            split == "train"
            and self._cfg.delta_crosscoder.data.repeat_to_full_batches
            and flattened.shape[0] % batch
        ):
            missing = batch - flattened.shape[0] % batch
            repeats = (missing + flattened.shape[0] - 1) // flattened.shape[0]
            flattened = torch.cat((flattened, flattened.repeat((repeats, 1, 1))[:missing]))
        for start in range(0, flattened.shape[0], batch):
            yield {"activations": flattened[start : start + batch]}

    def provenance(self) -> dict[str, Any]:
        return {
            "mode": "contrastive_cache",
            "manifest": str(self.manifest_path.resolve()),
            "cache_fingerprint": self.manifest["cache_fingerprint"],
            "construction": self.manifest["construction"],
            "models": self.manifest["models"],
            "tokenizer": self.manifest["tokenizer"],
            "prompt_source": self.manifest["prompt_source"],
            "generation": self.manifest["generation"],
        }


def _model_load_kwargs(spec: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "dtype": DTYPES[spec.dtype],
        "trust_remote_code": spec.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if spec.attn_implementation:
        kwargs["attn_implementation"] = spec.attn_implementation
    return kwargs


def _load_measurement_model(spec: Any, device: torch.device) -> nn.Module:
    """Load a full checkpoint or a pinned-base PEFT adapter without revision drift."""

    from transformers import AutoModelForCausalLM

    if spec.adapter_base_revision is None:
        model = AutoModelForCausalLM.from_pretrained(spec.model_id, **_model_load_kwargs(spec))
    else:
        from peft import PeftConfig, PeftModel

        adapter = PeftConfig.from_pretrained(spec.model_id, revision=spec.revision)
        base_id = adapter.base_model_name_or_path
        if not base_id:
            raise ValueError(f"adapter {spec.model_id} does not name a base model")
        kwargs = _model_load_kwargs(spec)
        kwargs["revision"] = spec.adapter_base_revision
        base = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
        wrapped = PeftModel.from_pretrained(
            base, spec.model_id, revision=spec.revision, is_trainable=False
        )
        # The adapter layers remain injected in the returned base module, while its
        # module paths match the unwrapped checkpoint (for example model.layers.14).
        model = wrapped.get_base_model()
    model = model.to(device).eval()
    model.get_submodule(spec.grounding.module)
    return model


def _load_measurement_models(
    cfg: DeltaCrosscoderExperimentConfig, device: torch.device
) -> list[nn.Module]:
    return [_load_measurement_model(spec, device) for spec in cfg.models]


def _audit_tokenizers(cfg: DeltaCrosscoderExperimentConfig) -> tuple[object, dict[str, Any]]:
    from transformers import AutoTokenizer

    assert cfg.delta_contrastive is not None
    cache = cfg.delta_contrastive.cache
    canonical = AutoTokenizer.from_pretrained(
        cache.canonical_tokenizer, revision=cache.canonical_tokenizer_revision
    )
    if not canonical.chat_template:  # type: ignore[attr-defined]
        raise ValueError("canonical tokenizer has no chat template")
    canonical_vocab = canonical.get_vocab()  # type: ignore[attr-defined]
    native: list[dict[str, Any]] = []
    for spec in cfg.models:
        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
        if tokenizer.get_vocab() != canonical_vocab:
            raise ValueError(f"{spec.model_id} vocabulary differs from canonical tokenizer")
        native.append(
            {
                "model_id": spec.model_id,
                "revision": spec.revision,
                "fingerprint": _tokenizer_fingerprint(tokenizer),
                "special_tokens_map": tokenizer.special_tokens_map,
                "chat_template_sha256": hashlib.sha256(
                    (tokenizer.chat_template or "").encode("utf-8")
                ).hexdigest(),
            }
        )
    return canonical, {
        "id": cache.canonical_tokenizer,
        "revision": cache.canonical_tokenizer_revision,
        "fingerprint": _tokenizer_fingerprint(canonical),
        "special_tokens_map": canonical.special_tokens_map,  # type: ignore[attr-defined]
        "chat_template": canonical.chat_template,  # type: ignore[attr-defined]
        "chat_template_sha256": hashlib.sha256(
            canonical.chat_template.encode("utf-8")  # type: ignore[attr-defined]
        ).hexdigest(),
        "native_tokenizer_audit": native,
        "vocabularies_equal": True,
    }


def materialize_contrastive_cache(
    cfg: DeltaCrosscoderExperimentConfig,
    *,
    max_prompts: int,
    output_dir: str | Path | None = None,
    device_name: str = "cuda",
    overwrite: bool | None = None,
) -> dict[str, Any]:
    """Generate both branches, replay every sequence through both models, and cache it."""

    if cfg.delta_contrastive is None:
        raise ValueError("config has no delta_contrastive section")
    # The original bounded pilot used 64 prompts.  The positive-control protocol
    # deliberately permits 768 (and at most 1024) frozen prompts on a B200.
    if not 1 <= max_prompts <= 1024:
        raise ValueError("contrastive max_prompts must be in [1,1024]")
    if any(model.revision is None for model in cfg.models):
        raise ValueError("contrastive materialization requires pinned model revisions")
    spec = cfg.delta_contrastive
    destination = Path(output_dir or spec.cache.output_dir)
    started_wall = time.time()
    started = time.perf_counter()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prompts = load_frozen_prompts(
        spec.prompt_source.path,
        prompt_id_field=spec.prompt_source.prompt_id_field,
        prompt_field=spec.prompt_source.prompt_field,
        max_prompts=max_prompts,
    )
    validation_ids = deterministic_validation_prompt_ids(
        prompts, spec.cache.validation_prompts, spec.generation.seed
    )
    tokenizer, tokenizer_provenance = _audit_tokenizers(cfg)
    pad_id = tokenizer.pad_token_id  # type: ignore[attr-defined]
    if pad_id is None:
        raise ValueError("canonical tokenizer must define pad_token_id")

    models = _load_measurement_models(cfg, device)
    generation_started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for prompt_index, prompt_row in enumerate(prompts):
        prompt_ids = canonical_prompt_ids(
            tokenizer, prompt_row["prompt"], enable_thinking=spec.generation.enable_thinking
        )
        if len(prompt_ids) >= spec.cache.max_sequence_length:
            raise ValueError(
                f"prompt {prompt_row['prompt_id']} alone exceeds max_sequence_length; refusing "
                "to remove chat-prefix tokens before generation"
            )
        for model_index, generator_side in enumerate(GENERATOR_SIDES):
            answer_ids, stop = generate_continuation(
                models[model_index],
                prompt_ids,
                spec.generation,
                device=device,
                pad_token_id=pad_id,
            )
            full_ids, truncation = apply_sequence_limit(
                prompt_ids + answer_ids,
                spec.cache.max_sequence_length,
                policy=spec.cache.truncation,
            )
            token_hash = token_ids_sha256(full_ids)
            records.append(
                {
                    "example_id": f"{prompt_row['prompt_id']}::{generator_side}",
                    "prompt_id": prompt_row["prompt_id"],
                    "prompt": prompt_row["prompt"],
                    "prompt_index": prompt_index,
                    "generator_side": generator_side,
                    "generated_answer": tokenizer.decode(  # type: ignore[attr-defined]
                        answer_ids, skip_special_tokens=False
                    ),
                    "rendered_full_text": tokenizer.decode(  # type: ignore[attr-defined]
                        full_ids, skip_special_tokens=False
                    ),
                    "prompt_token_count": len(prompt_ids),
                    "answer_token_count_before_sequence_truncation": len(answer_ids),
                    "token_ids": full_ids,
                    "valid_token_range": [0, len(full_ids)],
                    "measurement_input_sha256": {"base": token_hash, "target": token_hash},
                    "split": "validation" if prompt_row["prompt_id"] in validation_ids else "train",
                    "truncation": truncation,
                    "generation": {
                        **spec.generation.model_dump(mode="json"),
                        **stop,
                        "generator_model_id": cfg.models[model_index].model_id,
                        "generator_model_revision": cfg.models[model_index].revision,
                    },
                    "models": [
                        {"name": model.name, "model_id": model.model_id, "revision": model.revision}
                        for model in cfg.models
                    ],
                    "tokenizer": {
                        "id": tokenizer_provenance["id"],
                        "revision": tokenizer_provenance["revision"],
                        "fingerprint": tokenizer_provenance["fingerprint"],
                    },
                    "hook": [model.grounding.model_dump(mode="json") for model in cfg.models],
                }
            )
    generation_seconds = time.perf_counter() - generation_started

    extraction_started = time.perf_counter()
    activations = extract_paired_activations(
        records, models, cfg.models, device=device, pad_token_id=pad_id
    )
    extraction_seconds = time.perf_counter() - extraction_started
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    dtype = DTYPES[spec.cache.activation_dtype]
    manifest_metadata = {
        "created_unix_seconds": started_wall,
        "repository_commit": _repository_commit(),
        "construction": {
            "formula": "for each s in {prompt||y_base,prompt||y_target}: pair [h_base(s),h_target(s)]",
            "delta": "h_target(s)-h_base(s)",
            "same_canonical_token_ids_replayed_through_both_models": True,
            "response_side_subtraction_forbidden": True,
        },
        "models": {
            "ids": [model.model_id for model in cfg.models],
            "revisions": [model.revision for model in cfg.models],
            "hooks": [model.grounding.model_dump(mode="json") for model in cfg.models],
        },
        "tokenizer": tokenizer_provenance,
        "prompt_source": {
            **spec.prompt_source.model_dump(mode="json"),
            "file_sha256": hashlib.sha256(Path(spec.prompt_source.path).read_bytes()).hexdigest(),
            "prompt_ids": [prompt["prompt_id"] for prompt in prompts],
        },
        "generation": spec.generation.model_dump(mode="json"),
        "sequence_policy": {
            "max_sequence_length": spec.cache.max_sequence_length,
            "truncation": spec.cache.truncation,
            "padding": "right; masked and excluded from training",
            "sequence_boundaries_preserved_in_shards": True,
        },
    }
    manifest = write_contrastive_cache(
        destination,
        records,
        activations,
        manifest_metadata=manifest_metadata,
        sequences_per_shard=spec.cache.sequences_per_shard,
        activation_dtype=dtype,
        pad_token_id=pad_id,
        overwrite=spec.cache.overwrite if overwrite is None else overwrite,
    )
    validation = validate_contrastive_cache(destination / "manifest.json", cfg)

    side_norms: dict[str, list[Tensor]] = {side: [] for side in GENERATOR_SIDES}
    for record, paired in zip(records, activations, strict=True):
        side_norms[record["generator_side"]].append((paired[:, 1] - paired[:, 0]).norm(dim=-1))
    diagnostics: dict[str, dict[str, float]] = {}
    for side, values in side_norms.items():
        norms = torch.cat(values).float()
        diagnostics[side] = {"mean": float(norms.mean()), "median": float(norms.median())}
    total_seconds = time.perf_counter() - started
    cache_bytes = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    stop_reason_counts = {
        reason: sum(record["generation"]["stop_reason"] == reason for record in records)
        for reason in ("eos", "max_new_tokens")
    }
    answer_lengths = sorted(
        record["answer_token_count_before_sequence_truncation"] for record in records
    )
    generation_by_side: dict[str, dict[str, Any]] = {}
    for side in GENERATOR_SIDES:
        side_records = [record for record in records if record["generator_side"] == side]
        side_lengths = sorted(
            record["answer_token_count_before_sequence_truncation"] for record in side_records
        )
        midpoint = len(side_lengths) // 2
        median = (
            side_lengths[midpoint]
            if len(side_lengths) % 2
            else (side_lengths[midpoint - 1] + side_lengths[midpoint]) / 2
        )
        eos_count = sum(record["generation"]["stop_reason"] == "eos" for record in side_records)
        cap_count = len(side_records) - eos_count
        generation_by_side[side] = {
            "response_count": len(side_records),
            "eos_count": eos_count,
            "max_new_tokens_count": cap_count,
            "eos_fraction": eos_count / len(side_records),
            "answer_tokens_min": side_lengths[0],
            "answer_tokens_median": median,
            "answer_tokens_mean": sum(side_lengths) / len(side_lengths),
            "answer_tokens_max": side_lengths[-1],
        }
    report = {
        "scientific_status": "engineering pipeline diagnostic; not a behavioral finding",
        "repository_commit": manifest["repository_commit"],
        "prompts_requested": max_prompts,
        "prompts_succeeded": len(prompts),
        "base_responses_generated": sum(r["generator_side"] == "base_generated" for r in records),
        "target_responses_generated": sum(
            r["generator_side"] == "target_generated" for r in records
        ),
        "total_sequences": len(records),
        "total_valid_activation_tokens": sum(len(r["token_ids"]) for r in records),
        "average_prompt_tokens": sum(r["prompt_token_count"] for r in records) / len(records),
        "average_answer_tokens": sum(
            r["answer_token_count_before_sequence_truncation"] for r in records
        )
        / len(records),
        "answer_tokens_min": answer_lengths[0],
        "answer_tokens_median": (
            answer_lengths[len(answer_lengths) // 2]
            if len(answer_lengths) % 2
            else (
                answer_lengths[len(answer_lengths) // 2 - 1]
                + answer_lengths[len(answer_lengths) // 2]
            )
            / 2
        ),
        "answer_tokens_max": answer_lengths[-1],
        "stop_reason_counts": stop_reason_counts,
        "max_new_tokens_fraction": stop_reason_counts["max_new_tokens"] / len(records),
        "generation_by_side": generation_by_side,
        "truncation_count": sum(r["truncation"]["was_truncated"] for r in records),
        "cache_size_bytes_before_report": cache_bytes,
        "generation_seconds": generation_seconds,
        "activation_extraction_seconds": extraction_seconds,
        "total_seconds": total_seconds,
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "delta_norm_by_generator_side": diagnostics,
        "validation": validation,
    }
    _write_json(destination / "pilot_report.json", report)
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def materialize_contrastive_text_artifact(
    cfg: DeltaCrosscoderExperimentConfig,
    *,
    max_prompts: int,
    output_dir: str | Path,
    batch_size: int = 4,
    device_name: str = "cuda",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze generated text/token IDs without creating a paired activation cache."""

    if cfg.delta_contrastive is None:
        raise ValueError("config has no delta_contrastive section")
    if not 1 <= max_prompts <= 200_000:
        raise ValueError("R0 text materializer max_prompts must be in [1,200000]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    spec = cfg.delta_contrastive
    destination = Path(output_dir)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"contrastive text artifact exists: {destination}")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    started = time.perf_counter()
    device = torch.device(device_name)
    prompts = load_frozen_prompts(
        spec.prompt_source.path,
        prompt_id_field=spec.prompt_source.prompt_id_field,
        prompt_field=spec.prompt_source.prompt_field,
        max_prompts=max_prompts,
    )
    if len(prompts) != max_prompts:
        raise ValueError(f"prompt source contains {len(prompts)} rows, requested {max_prompts}")
    validation_ids = deterministic_validation_prompt_ids(
        prompts, spec.cache.validation_prompts, spec.generation.seed
    )
    tokenizer, tokenizer_provenance = _audit_tokenizers(cfg)
    pad_id = tokenizer.pad_token_id  # type: ignore[attr-defined]
    if pad_id is None:
        raise ValueError("canonical tokenizer must define pad_token_id")
    prompt_tokens = [
        canonical_prompt_ids(
            tokenizer, row["prompt"], enable_thinking=spec.generation.enable_thinking
        )
        for row in prompts
    ]
    if any(len(ids) >= spec.cache.max_sequence_length for ids in prompt_tokens):
        raise ValueError("a prompt alone reaches the frozen sequence limit")

    models = _load_measurement_models(cfg, device)
    records: list[dict[str, Any]] = []
    generation_started = time.perf_counter()
    for model_index, generator_side in enumerate(GENERATOR_SIDES):
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_prompt_ids = prompt_tokens[start : start + batch_size]
            generated = generate_continuations_batch(
                models[model_index],
                batch_prompt_ids,
                spec.generation,
                device=device,
                pad_token_id=pad_id,
            )
            for local_index, (answer_ids, stop) in enumerate(generated):
                prompt_index = start + local_index
                prompt = batch_prompts[local_index]
                full_ids, truncation = apply_sequence_limit(
                    batch_prompt_ids[local_index] + answer_ids,
                    spec.cache.max_sequence_length,
                    policy=spec.cache.truncation,
                )
                records.append(
                    {
                        "example_id": f"{prompt['prompt_id']}::{generator_side}",
                        "prompt_id": prompt["prompt_id"],
                        "prompt_index": prompt_index,
                        "prompt_reference": {
                            "source_path": spec.prompt_source.path,
                            "prompt_id": prompt["prompt_id"],
                        },
                        "generator_side": generator_side,
                        "split": (
                            "validation" if prompt["prompt_id"] in validation_ids else "train"
                        ),
                        "token_ids": full_ids,
                        "token_ids_sha256": token_ids_sha256(full_ids),
                        "generated_answer": tokenizer.decode(  # type: ignore[attr-defined]
                            answer_ids, skip_special_tokens=False
                        ),
                        "prompt_token_count": len(batch_prompt_ids[local_index]),
                        "answer_token_count_before_sequence_truncation": len(answer_ids),
                        "truncation": truncation,
                        "generation": {
                            **spec.generation.model_dump(mode="json"),
                            **stop,
                            "generator_model_id": cfg.models[model_index].model_id,
                            "generator_model_revision": cfg.models[model_index].revision,
                        },
                    }
                )
    generation_seconds = time.perf_counter() - generation_started
    # Stable prompt-major order makes the artifact independent of generation loop layout.
    records.sort(key=lambda row: (row["prompt_index"], row["generator_side"]))
    examples_path = staging / "examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    by_side: dict[str, dict[str, Any]] = {}
    for side in GENERATOR_SIDES:
        selected = [row for row in records if row["generator_side"] == side]
        lengths = sorted(row["answer_token_count_before_sequence_truncation"] for row in selected)
        eos = sum(row["generation"]["stop_reason"] == "eos" for row in selected)
        midpoint = len(lengths) // 2
        by_side[side] = {
            "responses": len(selected),
            "eos_count": eos,
            "cap_count": len(selected) - eos,
            "min": lengths[0],
            "median": (
                lengths[midpoint]
                if len(lengths) % 2
                else (lengths[midpoint - 1] + lengths[midpoint]) / 2
            ),
            "mean": sum(lengths) / len(lengths),
            "max": lengths[-1],
        }
    total_tokens = sum(len(row["token_ids"]) for row in records)
    manifest = {
        "schema_version": 1,
        "artifact_type": "delta_r0_contrastive_text",
        "repository_commit": _repository_commit(),
        "examples_file": "examples.jsonl",
        "examples_sha256": hashlib.sha256(examples_path.read_bytes()).hexdigest(),
        "models": {
            "ids": [model.model_id for model in cfg.models],
            "revisions": [model.revision for model in cfg.models],
        },
        "tokenizer": tokenizer_provenance,
        "prompt_source": {
            **spec.prompt_source.model_dump(mode="json"),
            "file_sha256": hashlib.sha256(Path(spec.prompt_source.path).read_bytes()).hexdigest(),
            "prompt_count": len(prompts),
        },
        "generation": spec.generation.model_dump(mode="json"),
        "batch_size": batch_size,
        "max_sequence_length": spec.cache.max_sequence_length,
        "records": len(records),
        "tokens": total_tokens,
        "split_tokens": {
            split: sum(len(row["token_ids"]) for row in records if row["split"] == split)
            for split in ("train", "validation")
        },
        "generation_by_side": by_side,
        "generation_seconds": generation_seconds,
        "prompts_per_second": len(prompts) / generation_seconds,
        "generated_answer_tokens_per_second": sum(
            row["answer_token_count_before_sequence_truncation"] for row in records
        )
        / generation_seconds,
        "same_frozen_token_ids_replayed_through_both_models": True,
    }
    manifest["artifact_fingerprint"] = _sha256_json(manifest)
    _write_json(staging / "manifest.json", manifest)
    os.replace(staging, destination)
    report = {
        **manifest,
        "total_seconds": time.perf_counter() - started,
        "artifact_size_bytes": sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        ),
    }
    _write_json(destination / "materialization_report.json", report)
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def materialize_activation_cache_from_frozen_text(
    cfg: DeltaCrosscoderExperimentConfig,
    *,
    response_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
    overwrite: bool = False,
    audit_filename: str = "behavioral_v2_premeasurement_audit.json",
    prompt_source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Replay a previously frozen response artifact through both models.

    This two-stage path is used by behavioral V2 so response generation is
    immutable before any Delta feature measurement. It refuses a response set
    that has not passed the preregistered premeasurement cap audit.
    """

    if cfg.delta_contrastive is None:
        raise ValueError("config has no delta_contrastive section")
    response_root = Path(response_dir)
    response_manifest = _read_json(response_root / "manifest.json")
    audit_path = response_root / audit_filename
    if not audit_path.exists():
        raise ValueError("frozen response artifact lacks premeasurement audit")
    audit = _read_json(audit_path)
    if not audit.get("measurement_allowed"):
        raise ValueError("response cap audit blocks feature measurement")
    if audit.get("response_artifact_fingerprint") != response_manifest.get("artifact_fingerprint"):
        raise ValueError("premeasurement audit does not match response artifact")
    if response_manifest.get("artifact_type") != "delta_r0_contrastive_text":
        raise ValueError("unsupported frozen response artifact type")
    expected_ids = [model.model_id for model in cfg.models]
    expected_revisions = [model.revision for model in cfg.models]
    if response_manifest.get("models", {}).get("ids") != expected_ids:
        raise ValueError("response artifact model IDs differ from config")
    if response_manifest.get("models", {}).get("revisions") != expected_revisions:
        raise ValueError("response artifact model revisions differ from config")

    examples_path = response_root / str(response_manifest["examples_file"])
    if hashlib.sha256(examples_path.read_bytes()).hexdigest() != response_manifest.get(
        "examples_sha256"
    ):
        raise ValueError("frozen response checksum mismatch")
    frozen_records = [
        json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines() if line
    ]
    prompt_rows: dict[str, dict[str, Any]] = {}
    source_path = Path(prompt_source_path or cfg.delta_contrastive.prompt_source.path)
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                prompt_rows[str(row[cfg.delta_contrastive.prompt_source.prompt_id_field])] = row
    tokenizer, tokenizer_provenance = _audit_tokenizers(cfg)
    pad_id = tokenizer.pad_token_id  # type: ignore[attr-defined]
    if pad_id is None:
        raise ValueError("canonical tokenizer must define pad_token_id")
    records: list[dict[str, Any]] = []
    for frozen in frozen_records:
        token_ids = [int(value) for value in frozen["token_ids"]]
        token_hash = token_ids_sha256(token_ids)
        if token_hash != frozen.get("token_ids_sha256"):
            raise ValueError(f"frozen token checksum mismatch for {frozen.get('example_id')}")
        prompt_id = str(frozen["prompt_id"])
        source = prompt_rows.get(prompt_id)
        if source is None:
            raise ValueError(f"frozen response references unknown prompt {prompt_id}")
        records.append(
            {
                **frozen,
                "prompt": str(source[cfg.delta_contrastive.prompt_source.prompt_field]),
                "rendered_full_text": tokenizer.decode(  # type: ignore[attr-defined]
                    token_ids, skip_special_tokens=False
                ),
                "valid_token_range": [0, len(token_ids)],
                "measurement_input_sha256": {"base": token_hash, "target": token_hash},
                "models": [
                    {"name": model.name, "model_id": model.model_id, "revision": model.revision}
                    for model in cfg.models
                ],
                "tokenizer": {
                    "id": tokenizer_provenance["id"],
                    "revision": tokenizer_provenance["revision"],
                    "fingerprint": tokenizer_provenance["fingerprint"],
                },
                "hook": [model.grounding.model_dump(mode="json") for model in cfg.models],
            }
        )

    started = time.perf_counter()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    models = _load_measurement_models(cfg, device)
    extraction_started = time.perf_counter()
    activations = extract_paired_activations(
        records, models, cfg.models, device=device, pad_token_id=pad_id
    )
    extraction_seconds = time.perf_counter() - extraction_started
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    destination = Path(output_dir)
    manifest = write_contrastive_cache(
        destination,
        records,
        activations,
        manifest_metadata={
            "created_unix_seconds": time.time(),
            "repository_commit": _repository_commit(),
            "construction": {
                "formula": "for each frozen response sequence s: pair [h_base(s),h_target(s)]",
                "same_canonical_token_ids_replayed_through_both_models": True,
                "responses_frozen_before_feature_measurement": True,
                "response_side_subtraction_forbidden": True,
            },
            "models": {
                "ids": expected_ids,
                "revisions": expected_revisions,
                "hooks": [model.grounding.model_dump(mode="json") for model in cfg.models],
            },
            "tokenizer": tokenizer_provenance,
            "prompt_source": response_manifest["prompt_source"],
            "generation": response_manifest["generation"],
            "frozen_response_artifact": {
                "path": str(response_root),
                "artifact_fingerprint": response_manifest["artifact_fingerprint"],
                "examples_sha256": response_manifest["examples_sha256"],
                "premeasurement_audit": str(audit_path),
            },
            "sequence_policy": {
                "max_sequence_length": response_manifest["max_sequence_length"],
                "truncation": "right",
                "padding": "right; masked and excluded from measurement",
                "sequence_boundaries_preserved_in_shards": True,
            },
        },
        sequences_per_shard=cfg.delta_contrastive.cache.sequences_per_shard,
        activation_dtype=DTYPES[cfg.delta_contrastive.cache.activation_dtype],
        pad_token_id=pad_id,
        overwrite=overwrite,
    )
    validation = validate_contrastive_cache(destination / "manifest.json", cfg)
    report = {
        "scientific_status": "frozen-response paired activation measurement; no behavioral claim",
        "repository_commit": manifest["repository_commit"],
        "response_artifact_fingerprint": response_manifest["artifact_fingerprint"],
        "sequences": len(records),
        "valid_tokens": sum(len(record["token_ids"]) for record in records),
        "activation_dim": manifest["activation_dim"],
        "activation_extraction_seconds": extraction_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "validation": validation,
    }
    _write_json(destination / "extraction_report.json", report)
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def materialize_organism_activation_cache(
    cfg: DeltaCrosscoderExperimentConfig,
    *,
    output_dir: str | Path | None = None,
    device_name: str = "cuda",
    overwrite: bool | None = None,
) -> dict[str, Any]:
    """Build a separate paired activation pilot from exact organism training rows."""

    if cfg.delta_contrastive is None or cfg.delta_organism_pilot is None:
        raise ValueError("config needs delta_contrastive and delta_organism_pilot sections")
    spec = cfg.delta_organism_pilot
    destination = Path(output_dir or spec.output_dir)
    started_wall = time.time()
    started = time.perf_counter()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows, dataset_examples = load_organism_dataset(
        spec.dataset_path,
        expected_sha256=spec.dataset_sha256,
        max_examples=spec.max_examples,
    )
    tokenizer, tokenizer_provenance = _audit_tokenizers(cfg)
    pad_id = tokenizer.pad_token_id  # type: ignore[attr-defined]
    if pad_id is None:
        raise ValueError("canonical tokenizer must define pad_token_id")
    split_rows = [{"prompt_id": f"organism-{row['dataset_index']:06d}"} for row in rows]
    validation_ids = deterministic_validation_prompt_ids(
        split_rows, spec.validation_examples, spec.seed
    )

    records: list[dict[str, Any]] = []
    for row in rows:
        example_id = f"organism-{row['dataset_index']:06d}"
        original_ids = canonical_conversation_ids(tokenizer, row["messages"])
        token_ids, truncation = apply_sequence_limit(
            original_ids, spec.max_sequence_length, policy=spec.truncation
        )
        token_hash = token_ids_sha256(token_ids)
        records.append(
            {
                "example_id": example_id,
                "dataset_index": row["dataset_index"],
                "messages": row["messages"],
                "generator_side": "organism_dataset",
                "rendered_full_text": tokenizer.decode(  # type: ignore[attr-defined]
                    token_ids, skip_special_tokens=False
                ),
                "token_ids": token_ids,
                "valid_token_range": [0, len(token_ids)],
                "measurement_input_sha256": {"base": token_hash, "target": token_hash},
                "split": "validation" if example_id in validation_ids else "train",
                "truncation": truncation,
                "models": [
                    {"name": model.name, "model_id": model.model_id, "revision": model.revision}
                    for model in cfg.models
                ],
                "tokenizer": {
                    "id": tokenizer_provenance["id"],
                    "revision": tokenizer_provenance["revision"],
                    "fingerprint": tokenizer_provenance["fingerprint"],
                },
                "hook": [model.grounding.model_dump(mode="json") for model in cfg.models],
            }
        )

    models = _load_measurement_models(cfg, device)
    extraction_started = time.perf_counter()
    activations = extract_paired_activations(
        records, models, cfg.models, device=device, pad_token_id=pad_id
    )
    extraction_seconds = time.perf_counter() - extraction_started
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    manifest = write_contrastive_cache(
        destination,
        records,
        activations,
        manifest_metadata={
            "artifact_type": "delta_organism_paired_activation_cache",
            "generator_sides": ["organism_dataset"],
            "created_unix_seconds": started_wall,
            "repository_commit": _repository_commit(),
            "construction": {
                "formula": "for each exact organism sequence s: pair [h_base(s),h_target(s)]",
                "same_canonical_token_ids_replayed_through_both_models": True,
            },
            "models": {
                "ids": [model.model_id for model in cfg.models],
                "revisions": [model.revision for model in cfg.models],
                "hooks": [model.grounding.model_dump(mode="json") for model in cfg.models],
            },
            "tokenizer": tokenizer_provenance,
            "dataset": {
                **spec.model_dump(mode="json", exclude={"output_dir", "overwrite"}),
                "total_examples": dataset_examples,
            },
            "sequence_policy": {
                "max_sequence_length": spec.max_sequence_length,
                "truncation": spec.truncation,
                "padding": "right; masked and excluded from measurement",
            },
        },
        sequences_per_shard=spec.sequences_per_shard,
        activation_dtype=DTYPES[spec.activation_dtype],
        pad_token_id=pad_id,
        overwrite=spec.overwrite if overwrite is None else overwrite,
    )
    validation = validate_organism_cache(destination / "manifest.json", cfg)
    cache_bytes = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    total_seconds = time.perf_counter() - started
    report = {
        "scientific_status": "organism-data activation engineering pilot; no semantic claim",
        "repository_commit": manifest["repository_commit"],
        "dataset_examples": dataset_examples,
        "pilot_examples": len(records),
        "total_valid_activation_tokens": sum(len(record["token_ids"]) for record in records),
        "mean_tokens": sum(len(record["token_ids"]) for record in records) / len(records),
        "max_tokens_before_truncation": max(
            record["truncation"]["original_token_count"] for record in records
        ),
        "truncation_count": sum(record["truncation"]["was_truncated"] for record in records),
        "cache_size_bytes_before_report": cache_bytes,
        "activation_extraction_seconds": extraction_seconds,
        "total_seconds": total_seconds,
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "validation": validation,
    }
    _write_json(destination / "pilot_report.json", report)
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report
