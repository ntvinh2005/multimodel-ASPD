"""Deterministic structural/template detection over exact retained center tokens."""

from __future__ import annotations

import string
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from aspd.multimodel.semantic_triage.schema import DeterministicStructuralResult

_SPECIAL_TOKENS = {
    "<s>",
    "</s>",
    "<bos>",
    "<eos>",
    "[bos]",
    "[eos]",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
}
_ROLE_MARKERS = {"user": "user_role", "assistant": "assistant_role", "system": "system_role"}
_THINK_MARKERS = {"<think>", "</think>", "think", "/think"}


def _center_values(example: Mapping[str, Any]) -> tuple[str, str]:
    center = example.get("center_in_window")
    exact = example.get("center_token")
    decoded = None
    if isinstance(center, int):
        pieces = example.get("token_strings")
        texts = example.get("token_texts")
        if not isinstance(exact, str) and isinstance(pieces, list) and center < len(pieces):
            exact = pieces[center]
        if isinstance(texts, list) and center < len(texts):
            decoded = texts[center]
    return (exact if isinstance(exact, str) else "", decoded if isinstance(decoded, str) else "")


def _normalized_piece(value: str) -> str:
    # Common tokenizer whitespace markers are representation, not semantic content.
    return value.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ").strip().lower()


def _punctuation_only(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped) and all(
        character in string.punctuation or unicodedata.category(character).startswith(("P", "S"))
        for character in stripped
    )


def center_categories(
    example: Mapping[str, Any], *, sequence_length: int | None = None
) -> set[str]:
    """Return auditable structural categories for exactly one retained firing center."""

    exact, decoded = _center_values(example)
    normalized_exact = _normalized_piece(exact)
    normalized_decoded = decoded.strip().lower()
    candidates = {normalized_exact, normalized_decoded}
    categories: set[str] = set()

    if any(value in _SPECIAL_TOKENS or (value.startswith("<|") and value.endswith("|>")) for value in candidates):
        categories.update(("special_token", "chat_boundary"))
    for marker, category in _ROLE_MARKERS.items():
        if marker in candidates:
            categories.add(category)
    if any(value in _THINK_MARKERS for value in candidates):
        categories.add("think_boundary")
    if decoded and not decoded.strip() and ("\n" in decoded or "Ċ" in exact):
        categories.add("newline_only")
    if not categories and (_punctuation_only(decoded) or _punctuation_only(normalized_exact)):
        categories.add("delimiter_or_punctuation")

    position = example.get("position_in_sequence")
    if position == 0:
        categories.add("sequence_start")
    if isinstance(position, int) and sequence_length is not None and position == sequence_length - 1:
        categories.add("sequence_end")
    return categories


def analyze_structural_contexts(
    examples: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.70,
    sequence_length: int | None = None,
) -> DeterministicStructuralResult:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("structural threshold must be between 0 and 1")
    categories = [center_categories(example, sequence_length=sequence_length) for example in examples]
    total = len(categories)
    structural_count = sum(bool(items) for items in categories)
    start_count = sum("sequence_start" in items for items in categories)
    end_count = sum("sequence_end" in items for items in categories)
    special_count = sum("special_token" in items for items in categories)
    denominator = total or 1
    fraction = structural_count / denominator if total else 0.0
    return DeterministicStructuralResult(
        structural_dominated=bool(total) and fraction >= threshold,
        structural_center_count=structural_count,
        structural_center_fraction=fraction,
        sequence_start_count=start_count,
        sequence_start_fraction=start_count / denominator if total else 0.0,
        sequence_end_count=end_count,
        sequence_end_fraction=end_count / denominator if total else 0.0,
        special_token_center_count=special_count,
        special_token_center_fraction=special_count / denominator if total else 0.0,
        detected_center_categories=sorted(set().union(*categories) if categories else set()),
    )
