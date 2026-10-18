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
    answer_ids = [int(value) for value in generated[0, len(prompt_ids):].tolist()]
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
    batches = [records[start:start + batch_size] for start in range(0, len(records), batch_size)]
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
                selected = indices[start:start + sequences_per_shard]
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
                    {"path": relative, "sequences": len(selected), "valid_tokens": int(valid_tokens.sum())}
                )
            split_manifest[split] = {
                "sequences": len(indices), "valid_tokens": token_count, "shards": shard_entries
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
        "example_id", "prompt_id", "prompt", "generator_side", "generated_answer",
        "rendered_full_text", "token_ids", "valid_token_range", "generation",
        "models", "tokenizer", "hook", "measurement_input_sha256", "truncation", "split",
    }
    prompt_sides: dict[str, set[str]] = {}
    for example in examples:
        missing = required - example.keys()
        if missing:
            raise ValueError(f"example {example.get('example_id')} lacks provenance: {sorted(missing)}")
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
        for start in range(0, flattened.shape[0], batch):
            yield {"activations": flattened[start:start + batch]}

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

    from transformers import AutoModelForCausalLM

    if cfg.delta_contrastive is None:
        raise ValueError("config has no delta_contrastive section")
    if not 1 <= max_prompts <= 64:
        raise ValueError("Phase-2A max_prompts must be in [1,64]")
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

    models = [
        AutoModelForCausalLM.from_pretrained(model.model_id, **_model_load_kwargs(model))
        .to(device)
        .eval()
        for model in cfg.models
    ]
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
                models[model_index], prompt_ids, spec.generation,
                device=device, pad_token_id=pad_id,
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
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
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
    report = {
        "scientific_status": "engineering pipeline diagnostic; not a behavioral finding",
        "repository_commit": manifest["repository_commit"],
        "prompts_requested": max_prompts,
        "prompts_succeeded": len(prompts),
        "base_responses_generated": sum(r["generator_side"] == "base_generated" for r in records),
        "target_responses_generated": sum(r["generator_side"] == "target_generated" for r in records),
        "total_sequences": len(records),
        "total_valid_activation_tokens": sum(len(r["token_ids"]) for r in records),
        "average_prompt_tokens": sum(r["prompt_token_count"] for r in records) / len(records),
        "average_answer_tokens": sum(
            r["answer_token_count_before_sequence_truncation"] for r in records
        ) / len(records),
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
