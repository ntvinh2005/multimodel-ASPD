"""Deterministic canonical-text-boundary alignment across tokenizer families."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class FastTokenizer(Protocol):
    is_fast: bool

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class TokenBoundary:
    model_key: str
    token_index: int
    token_id: int
    prefix_token_count: int
    character_boundary: int
    byte_boundary: int


@dataclass(frozen=True)
class AlignedAnchor:
    anchor_id: str
    document_id: str
    source_group: str
    character_boundary: int
    byte_boundary: int
    canonical_prefix_sha256: str
    positions: Mapping[str, TokenBoundary]


@dataclass(frozen=True)
class TokenizedDocument:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    boundary_to_token: Mapping[int, TokenBoundary]
    truncated: bool


@dataclass(frozen=True)
class AlignmentResult:
    document_id: str
    source_group: str
    anchors: tuple[AlignedAnchor, ...]
    tokenized: Mapping[str, TokenizedDocument]
    rejection_counts: Mapping[str, int]

    def report(self) -> dict[str, Any]:
        candidate_boundaries = set()
        for document in self.tokenized.values():
            candidate_boundaries.update(document.boundary_to_token)
        return {
            "document_id": self.document_id,
            "source_group": self.source_group,
            "accepted_anchors": len(self.anchors),
            "candidate_canonical_boundaries": len(candidate_boundaries),
            "acceptance_rate": len(self.anchors) / max(1, len(candidate_boundaries)),
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "truncated_models": sorted(
                key for key, value in self.tokenized.items() if value.truncated
            ),
            "alignment_policy": "exact_fast_tokenizer_offsets_and_exact_prefix_roundtrip_v1",
        }


class AlignmentError(ValueError):
    pass


def _flat(values: Any, name: str) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        raise AlignmentError(f"tokenizer {name} must return a list")
    if values and isinstance(values[0], list) and name != "offset_mapping":
        if len(values) != 1:
            raise AlignmentError(f"batched tokenizer output is unsupported for {name}")
        values = values[0]
    return values


def _tokenize(
    model_key: str,
    tokenizer: FastTokenizer,
    text: str,
    max_length: int | None,
    rejections: Counter[str],
) -> TokenizedDocument:
    if not getattr(tokenizer, "is_fast", False):
        raise AlignmentError(f"{model_key}: a fast tokenizer with offset_mapping is required")
    kwargs: dict[str, Any] = {
        "add_special_tokens": True,
        "return_attention_mask": True,
        "return_offsets_mapping": True,
        "truncation": max_length is not None,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    encoded = tokenizer(text, **kwargs)
    input_ids = tuple(int(value) for value in _flat(encoded["input_ids"], "input_ids"))
    mask = tuple(int(value) for value in _flat(encoded["attention_mask"], "attention_mask"))
    offsets_raw = encoded["offset_mapping"]
    if hasattr(offsets_raw, "tolist"):
        offsets_raw = offsets_raw.tolist()
    if offsets_raw and isinstance(offsets_raw[0], list) and len(offsets_raw) == 1:
        offsets_raw = offsets_raw[0]
    offsets = tuple((int(start), int(stop)) for start, stop in offsets_raw)
    if not (len(input_ids) == len(mask) == len(offsets)):
        raise AlignmentError(f"{model_key}: tokenizer arrays have inconsistent lengths")

    boundary_to_token: dict[int, TokenBoundary] = {}
    previous_start = 0
    maximum_stop = 0
    for token_index, (token_id, valid, offset) in enumerate(zip(input_ids, mask, offsets, strict=True)):
        if not valid:
            continue
        start, stop = offset
        if (start, stop) == (0, 0):
            rejections[f"{model_key}:special_token_no_boundary"] += 1
            continue
        if start < 0 or stop > len(text) or stop <= start or start < previous_start:
            rejections[f"{model_key}:invalid_or_ambiguous_offset"] += 1
            continue
        previous_start = start
        maximum_stop = max(maximum_stop, stop)
        decoded_prefix = tokenizer.decode(
            input_ids[: token_index + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        canonical_prefix = text[:stop]
        if decoded_prefix != canonical_prefix:
            rejections[f"{model_key}:prefix_roundtrip_mismatch"] += 1
            continue
        if stop in boundary_to_token:
            rejections[f"{model_key}:duplicate_boundary"] += 1
            boundary_to_token.pop(stop, None)
            continue
        boundary_to_token[stop] = TokenBoundary(
            model_key=model_key,
            token_index=token_index,
            token_id=token_id,
            prefix_token_count=token_index + 1,
            character_boundary=stop,
            byte_boundary=len(canonical_prefix.encode("utf-8")),
        )
    truncated = bool(max_length is not None and len(input_ids) >= max_length and maximum_stop < len(text))
    if truncated:
        rejections[f"{model_key}:truncated_suffix"] += 1
    return TokenizedDocument(input_ids, mask, boundary_to_token, truncated)


def align_canonical_text(
    text: str,
    *,
    document_id: str,
    source_group: str,
    tokenizers: Mapping[str, FastTokenizer],
    model_order: Sequence[str] = ("qwen", "gemma", "phi"),
    max_length: int | None = None,
) -> AlignmentResult:
    """Align only token ends that denote the same exact prefix of the original text."""

    if not text:
        raise AlignmentError("empty documents do not contain alignable canonical boundaries")
    if tuple(tokenizers) != tuple(model_order):
        raise AlignmentError("tokenizers must use the deterministic configured model order")
    if len(set(model_order)) != len(model_order):
        raise AlignmentError("model_order contains duplicates")

    rejections: Counter[str] = Counter()
    tokenized = {
        key: _tokenize(key, tokenizers[key], text, max_length, rejections)
        for key in model_order
    }
    boundary_sets = [set(tokenized[key].boundary_to_token) for key in model_order]
    common = set.intersection(*boundary_sets) if boundary_sets else set()
    union = set.union(*boundary_sets) if boundary_sets else set()
    rejections["not_shared_by_all_tokenizers"] += len(union - common)

    anchors: list[AlignedAnchor] = []
    for boundary in sorted(common):
        prefix = text[:boundary]
        byte_boundary = len(prefix.encode("utf-8"))
        positions = {key: tokenized[key].boundary_to_token[boundary] for key in model_order}
        if any(position.byte_boundary != byte_boundary for position in positions.values()):
            rejections["byte_boundary_disagreement"] += 1
            continue
        prefix_hash = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        anchor_id = hashlib.sha256(
            f"{source_group}\0{document_id}\0{boundary}\0{prefix_hash}".encode()
        ).hexdigest()
        anchors.append(
            AlignedAnchor(
                anchor_id=anchor_id,
                document_id=document_id,
                source_group=source_group,
                character_boundary=boundary,
                byte_boundary=byte_boundary,
                canonical_prefix_sha256=prefix_hash,
                positions=positions,
            )
        )
    if not anchors:
        rejections["document_has_no_common_exact_boundary"] += 1
    return AlignmentResult(
        document_id=document_id,
        source_group=source_group,
        anchors=tuple(anchors),
        tokenized=tokenized,
        rejection_counts=dict(rejections),
    )
