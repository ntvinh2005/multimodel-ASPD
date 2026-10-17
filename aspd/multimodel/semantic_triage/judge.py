"""One-load batched judge engines and robust, bounded response parsing."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from aspd.multimodel.semantic_triage.prompts import JSON_PREFILL, render_messages
from aspd.multimodel.semantic_triage.schema import (
    GenerationRecord,
    JudgeOutput,
    ParsedJudgeOutput,
)


def _validate_payload(payload: Any, valid_context_ids: set[int]) -> JudgeOutput:
    result = JudgeOutput.model_validate(payload)
    referenced = set(result.evidence_context_ids) | set(result.counterexample_context_ids)
    invalid = sorted(referenced - valid_context_ids)
    if invalid:
        raise ValueError(f"judge referenced invalid context IDs: {invalid}")
    return result


def _repair_json(raw_output: str) -> Any:
    """One cheap repair pass: extract the first balanced object and remove trailing commas."""

    stripped = raw_output.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escaped = False
    end = None
    for index, character in enumerate(stripped[start:], start=start):
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
        elif not in_string and character == "{":
            depth += 1
        elif not in_string and character == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end is None:
        raise ValueError("no balanced JSON object found")
    candidate = re.sub(r",\s*([}\]])", r"\1", stripped[start : end + 1])
    return json.loads(candidate)


def _minor_normalize(payload: Any, valid_context_ids: set[int]) -> Any:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    # Cap only when every supplied ID is valid. An invalid ID anywhere in the original list must
    # survive to strict validation rather than disappearing beyond the retained first three.
    for field in ("evidence_context_ids", "counterexample_context_ids"):
        value = normalized.get(field)
        if (
            isinstance(value, list)
            and len(value) > 3
            and all(isinstance(item, int) and item in valid_context_ids for item in value)
        ):
            normalized[field] = value[:3]
    return normalized


def parse_judge_output(raw_output: str, *, valid_context_ids: set[int]) -> ParsedJudgeOutput:
    """Validate directly, then attempt exactly one extraction/normalization repair."""

    direct_error: Exception | None = None
    try:
        return ParsedJudgeOutput(
            result=_validate_payload(json.loads(raw_output), valid_context_ids),
            parse_failed=False,
            raw_output=raw_output,
        )
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        direct_error = exc
    try:
        repaired_payload = _minor_normalize(_repair_json(raw_output), valid_context_ids)
        return ParsedJudgeOutput(
            result=_validate_payload(repaired_payload, valid_context_ids),
            parse_failed=False,
            repaired=True,
            raw_output=raw_output,
        )
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as repair_error:
        return ParsedJudgeOutput(
            result=None,
            parse_failed=True,
            repaired=True,
            raw_output=raw_output,
            error=f"direct: {direct_error}; repair: {repair_error}",
        )


class BatchJudge(ABC):
    """A judge loads once and returns one short generation per feature prompt."""

    model_revision: str | None = None
    tokenizer_revision: str | None = None
    load_seconds: float = 0.0

    @abstractmethod
    def generate(self, context_batches: Sequence[Sequence[dict[str, Any]]]) -> list[GenerationRecord]:
        raise NotImplementedError


def _resolved_revisions(
    model_id: str, model_revision: str | None, tokenizer_revision: str | None
) -> tuple[str | None, str | None]:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id, revision=model_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=tokenizer_revision or model_revision, trust_remote_code=False
    )
    resolved_model = getattr(config, "_commit_hash", None) or model_revision
    resolved_tokenizer = tokenizer.init_kwargs.get("_commit_hash") or tokenizer_revision or resolved_model
    return resolved_model, resolved_tokenizer


class VLLMBatchJudge(BatchJudge):
    """High-throughput vLLM implementation; import remains optional for CPU-only tooling."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str | None,
        tokenizer_revision: str | None,
        temperature: float,
        max_new_tokens: int,
    ) -> None:
        started = time.perf_counter()
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed in this environment; install it or use --engine transformers"
            ) from exc
        self.model_revision, self.tokenizer_revision = _resolved_revisions(
            model_id, model_revision, tokenizer_revision
        )
        self._llm = LLM(
            model=model_id,
            revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            trust_remote_code=False,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
        )
        self._sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            seed=0,
            stop=["}"],
            include_stop_str_in_output=True,
        )
        self.load_seconds = time.perf_counter() - started

    def generate(self, context_batches: Sequence[Sequence[dict[str, Any]]]) -> list[GenerationRecord]:
        tokenizer = self._llm.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(
                render_messages(examples), tokenize=False, add_generation_prompt=True
            ) + JSON_PREFILL
            for examples in context_batches
        ]
        outputs = self._llm.generate(prompts, self._sampling, use_tqdm=True)
        return [
            GenerationRecord(
                raw_output=JSON_PREFILL + output.outputs[0].text,
                prompt_tokens=len(output.prompt_token_ids),
                generated_tokens=len(output.outputs[0].token_ids),
            )
            for output in outputs
        ]


class TransformersBatchJudge(BatchJudge):
    """Batched one-load fallback for clusters where vLLM is kept outside the project env."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str | None,
        tokenizer_revision: str | None,
        temperature: float,
        max_new_tokens: int,
        device: str,
        batch_size: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        self.model_revision, self.tokenizer_revision = _resolved_revisions(
            model_id, model_revision, tokenizer_revision
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=self.tokenizer_revision, trust_remote_code=False
        )
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=self.model_revision,
            trust_remote_code=False,
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        ).to(device).eval()
        self._device = device
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._batch_size = batch_size
        self.load_seconds = time.perf_counter() - started

    def generate(self, context_batches: Sequence[Sequence[dict[str, Any]]]) -> list[GenerationRecord]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        prompts = [
            self._tokenizer.apply_chat_template(
                render_messages(examples), tokenize=False, add_generation_prompt=True
            ) + JSON_PREFILL
            for examples in context_batches
        ]
        records: list[GenerationRecord] = []
        for start in range(0, len(prompts), self._batch_size):
            encoded = self._tokenizer(
                prompts[start : start + self._batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096 - self._max_new_tokens,
            ).to(self._device)
            prompt_lengths = encoded.attention_mask.sum(dim=1).tolist()

            tokenizer = self._tokenizer
            generated_start = encoded.input_ids.shape[1]

            class _AllJsonObjectsClosed(StoppingCriteria):
                def __init__(self, active_tokenizer: Any, start_index: int) -> None:
                    self.active_tokenizer = active_tokenizer
                    self.start_index = start_index

                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                    del scores, kwargs
                    generated_rows = input_ids[:, self.start_index:]
                    return torch.tensor(
                        [
                            "}" in self.active_tokenizer.decode(
                                row, skip_special_tokens=False
                            )
                            for row in generated_rows
                        ],
                        device=input_ids.device,
                        dtype=torch.bool,
                    )

            generation_kwargs: dict[str, Any] = {
                "do_sample": self._temperature > 0,
                "max_new_tokens": self._max_new_tokens,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                "stopping_criteria": StoppingCriteriaList(
                    [_AllJsonObjectsClosed(tokenizer, generated_start)]
                ),
            }
            if self._temperature > 0:
                generation_kwargs["temperature"] = self._temperature
            with torch.inference_mode():
                output_ids = self._model.generate(
                    **encoded,
                    **generation_kwargs,
                )
            generated = output_ids[:, encoded.input_ids.shape[1] :]
            for local_index, token_ids in enumerate(generated):
                nonpad = token_ids[token_ids != self._tokenizer.pad_token_id]
                records.append(
                    GenerationRecord(
                        raw_output=JSON_PREFILL + self._tokenizer.decode(
                            nonpad, skip_special_tokens=True
                        ),
                        prompt_tokens=int(prompt_lengths[local_index]),
                        generated_tokens=int(nonpad.numel()),
                    )
                )
        return records


def build_judge(
    *,
    engine: str,
    model_id: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    temperature: float,
    max_new_tokens: int,
    device: str,
    batch_size: int,
) -> BatchJudge:
    if engine == "vllm":
        return VLLMBatchJudge(
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    if engine == "transformers":
        return TransformersBatchJudge(
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device,
            batch_size=batch_size,
        )
    raise ValueError(f"unknown judge engine: {engine}")
