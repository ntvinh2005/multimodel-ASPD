"""Controlled exclusion and scope examples on the exact v3.1 canonical anchor axis."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from aspd.multimodel.v3.induction import (
    MODEL_ORDER,
    _alignment_payload,
    _one_token_continuation,
    align_canonical_text,
)

FAMILIES = ("binary", "attribute", "multiple", "scope")
FACTORIAL = ("affirmative", "negative", "counterfactual_affirmative", "counterfactual_negative")
SCOPE_CONTROLS = ("scope_control", "counterfactual_scope_control")
CONDITIONS = FACTORIAL + SCOPE_CONTROLS
NEGATIVE_CONDITIONS = {"negative", "counterfactual_negative"}
ACTORS = ("Mira", "Leo", "Eva", "Sam", "Nina", "Omar", "Emma", "Max", "Iris", "Alex")
ATTRIBUTES = ("blue", "green", "gold", "silver", "red", "white", "black", "orange")


class AnnotatedPrompt:
    def __init__(self) -> None:
        self.text = ""
        self.spans: list[dict[str, Any]] = []

    def add(self, text: str, role: str | None = None) -> None:
        start = len(self.text)
        self.text += text
        if role:
            self.spans.append({"start": start, "stop": len(self.text), "role": role})


def build_group(
    split: str,
    family: str,
    group_id: str,
    slate: Sequence[str],
    *,
    actor: str,
    other: str,
    attributes: Sequence[str],
    template: int,
    seed: int,
    order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """The fixed orientation is the first two members of the slate; order is independent."""
    count = 3 if family == "multiple" else 2
    if family not in FAMILIES or len(slate) != count or len(set(slate)) != count:
        raise ValueError("invalid candidate slate")
    order = tuple(order) if order is not None else tuple(slate)
    if len(order) != count or set(order) != set(slate):
        raise ValueError("invalid candidate presentation order")
    if (template >= 2) != (split == "heldout"):
        raise ValueError("held-out templates must be isolated")
    assignments = dict(zip(slate, attributes, strict=False))
    if family == "attribute" and (
        set(assignments) != set(slate) or len(set(assignments.values())) != count
    ):
        raise ValueError("attribute rules require a distinct tag for every candidate")
    result = []
    for condition in FACTORIAL + (SCOPE_CONTROLS if family == "scope" else ()):
        counterfactual = condition.startswith("counterfactual")
        target = slate[int(counterfactual)]
        negated = condition in NEGATIVE_CONDITIONS
        scope_control = condition in SCOPE_CONTROLS
        valid = [value for value in slate if (value != target if negated else value == target)]
        excluded = [target] if negated else []
        prompt = AnnotatedPrompt()
        intro = (
            f"{actor} must choose exactly one code from "
            if template % 2 == 0
            else f"Exactly one code will be selected by {actor} from "
        )
        prompt.add(intro)
        for index, token in enumerate(order):
            if index:
                prompt.add(" or " if index == count - 1 else ", ")
            prompt.add(
                token,
                "excluded_concept"
                if token in excluded
                else "alternative_concept"
                if token in valid
                else "candidate",
            )
        prompt.add(". ")
        if family == "attribute":
            for token in order:
                prompt.add(
                    token,
                    "excluded_concept"
                    if token in excluded
                    else "alternative_concept"
                    if token in valid
                    else "candidate",
                )
                prompt.add(f" is tagged {assignments[token]}. ")
        prompt.add(actor, "queried_scope")
        prompt.add(" will ")
        if negated:
            prompt.add("not", "negation_operator")
            prompt.add(" ")
        prompt.add("choose ")
        if family == "attribute":
            prompt.add("a code tagged ")
            prompt.add(
                assignments[target], "negated_attribute" if negated else "affirmative_attribute"
            )
        else:
            prompt.add("the ")
            prompt.add(target, "excluded_concept" if negated else "affirmative_concept")
        prompt.add(". ")
        if family == "scope":
            prompt.add(other, "irrelevant_scope")
            prompt.add(" will ")
            if scope_control:
                prompt.add("not", "negation_operator")
                prompt.add(" ")
            prompt.add("choose the ")
            prompt.add(target, "irrelevant_concept")
            prompt.add(". ")
        query = (
            f"{actor} chooses the"
            if template == 0
            else f"Therefore, {actor} selects the"
            if template == 1
            else f"The code picked by {actor} is the"
            if template == 2
            else f"For {actor}, the selected code is the"
        )
        prompt.add(query, "query_prefix")
        result.append(
            {
                "example_id": f"{group_id}/{condition}",
                "group_id": group_id,
                "split": split,
                "family": family,
                "condition": condition,
                "polarity": "negative" if negated else "affirmative",
                "negation_operator": "not" if negated or scope_control else None,
                "negation_scope": actor if negated else other if scope_control else None,
                "queried_entity": actor,
                "original_affirmative_target": slate[0],
                "mentioned_target": target,
                "candidate_order": list(order),
                "candidate_orientation": list(slate[:2]),
                "excluded_candidates": excluded,
                "valid_alternatives": valid,
                "distractors": [value for value in slate if value not in valid],
                "expected_continuation": f" {valid[0]}" if len(valid) == 1 else None,
                "continuations": {value: f" {value}" for value in slate},
                "attributes": assignments if family == "attribute" else {},
                "template_id": f"{family}-{template}",
                "seed": seed,
                "prompt": prompt.text,
                "spans": prompt.spans,
            }
        )
    validate_group(result)
    return result


def validate_group(group: Sequence[Mapping[str, Any]]) -> None:
    first = group[0]
    family = first["family"]
    wanted = FACTORIAL + (SCOPE_CONTROLS if family == "scope" else ())
    if tuple(row["condition"] for row in group) != wanted:
        raise ValueError("matched condition inventory changed")
    for row in group:
        for field in (
            "group_id",
            "family",
            "split",
            "candidate_orientation",
            "candidate_order",
            "template_id",
        ):
            if row[field] != first[field]:
                raise ValueError(f"matched field changed: {field}")
        target = row["mentioned_target"]
        candidates = set(row["candidate_order"])
        negative = row["polarity"] == "negative"
        if family == "attribute":
            qualifying = {
                word for word in candidates if row["attributes"][word] == row["attributes"][target]
            }
        else:
            qualifying = {target}
        expected = candidates - qualifying if negative else qualifying
        if set(row["valid_alternatives"]) != expected or set(row["excluded_candidates"]) != (
            qualifying if negative else set()
        ):
            raise ValueError("answer set disagrees with explicit choice rule")
        if (row["negation_scope"] == row["queried_entity"]) != negative:
            raise ValueError("negation scope disagrees with queried polarity")
        if (row["expected_continuation"] is None) != (len(expected) != 1):
            raise ValueError("multiple valid answers must remain a set")
    if family == "scope":
        for negative, control in ((group[1], group[4]), (group[3], group[5])):
            if Counter(negative["prompt"].split()) != Counter(control["prompt"].split()):
                raise ValueError("scope control changes lexical inventory")


def candidate_groups(
    symbols: Sequence[str], split: str, family: str, seed: int
) -> Iterator[list[dict[str, Any]]]:
    """Finite deterministic proposal order, independent of tokenizer rejection."""
    import itertools

    rng = random.Random(f"{seed}/{split}/{family}")
    combinations = list(itertools.combinations(sorted(symbols), 3 if family == "multiple" else 2))
    rng.shuffle(combinations)
    for index, combination in enumerate(combinations):
        slate = rng.sample(combination, len(combination))
        order = rng.sample(combination, len(combination))
        actor, other = rng.sample(ACTORS, 2)
        template = (2 if split == "heldout" else 0) + rng.randrange(2)
        yield build_group(
            split,
            family,
            f"negation-{split}-{family}-{index:04d}",
            slate,
            actor=actor,
            other=other,
            attributes=rng.sample(ATTRIBUTES, len(slate)),
            template=template,
            seed=seed,
            order=order,
        )


def verify_example(
    row: dict[str, Any], tokenizers: Mapping[str, Any], max_length: int
) -> dict[str, Any]:
    """Validate full-context single native tokens for every candidate and exact marked anchors."""
    if tuple(tokenizers) != MODEL_ORDER:
        raise ValueError("tokenizer order changed")
    alignment = align_canonical_text(
        row["prompt"],
        document_id=row["example_id"],
        source_group=f"negation/{row['split']}/{row['family']}/{row['condition']}",
        tokenizers=tokenizers,
        model_order=MODEL_ORDER,
        max_length=max_length,
    )
    if any(document.truncated for document in alignment.tokenized.values()):
        raise ValueError("prompt exceeded configured sequence length")
    anchors = {anchor.character_boundary: anchor for anchor in alignment.anchors}
    final = len(row["prompt"])
    required = {final, *(span["stop"] for span in row["spans"])}
    if required - anchors.keys():
        raise ValueError(
            f"required canonical boundaries missing: {sorted(required - anchors.keys())}"
        )
    models = {}
    for key in MODEL_ORDER:
        position = anchors[final].positions[key].token_index
        ids = {}
        for candidate, continuation in row["continuations"].items():
            candidate_id, candidate_position = _one_token_continuation(
                tokenizers[key], row["prompt"], continuation
            )
            if position != candidate_position:
                raise ValueError("candidate does not use the final native prediction position")
            ids[candidate] = candidate_id
        if len(set(ids.values())) != len(ids):
            raise ValueError("candidate tokens collide")
        native = alignment.tokenized[key]
        models[key] = {
            "input_ids": list(native.input_ids),
            "attention_mask": list(native.attention_mask),
            "candidate_token_ids": ids,
            "prediction_token_index": position,
        }
    return {
        "example": row,
        "models": models,
        "alignment": _alignment_payload(alignment),
        "required_boundaries": sorted(required),
        "final_anchor_id": anchors[final].anchor_id,
    }


def pack_groups(
    groups: Sequence[Sequence[dict[str, Any]]], budget: int
) -> list[list[dict[str, Any]]]:
    """First-fit in frozen group order, preserving every matched condition in a single pool."""
    pools: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    used = 0
    for group in groups:
        count = sum(len(row["alignment"]["anchors"]) for row in group)
        if count > budget:
            raise ValueError("matched group exceeds the pool budget")
        if used + count > budget:
            pools.append(current)
            current, used = [], 0
        current.extend(group)
        used += count
    return pools + ([current] if current else [])


def candidate_behavior(
    example: Mapping[str, Any],
    logits: Mapping[str, float],
    probabilities: Mapping[str, float],
    predicted: str | None,
) -> dict[str, Any]:
    """Separate fixed orientation, closed-candidate decision, and full-vocab outcome."""
    a, b = example["candidate_orientation"]
    valid = set(example["valid_alternatives"])
    excluded = set(example["excluded_candidates"])
    invalid = set(logits) - valid
    if (
        not valid
        or not invalid
        or set(logits) != set(probabilities)
        or set(logits) != set(example["candidate_order"])
    ):
        raise ValueError("candidate metrics need a complete, distinct candidate inventory")
    if not all(math.isfinite(value) for value in logits.values()) or not all(
        math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values()
    ):
        raise ValueError("non-finite or invalid candidate evidence")
    margin = max(logits[value] for value in valid) - max(logits[value] for value in invalid)
    return {
        "fixed_margin": logits[a] - logits[b],
        "correctness_margin": margin,
        "candidate_correct": margin > 0,
        "exact_correct": predicted in valid,
        "predicted_candidate": predicted,
        "valid_mass": sum(probabilities[value] for value in valid),
        "excluded_mass": sum(probabilities[value] for value in excluded),
        "unrelated_mass": max(0.0, 1 - sum(probabilities.values())),
        "valid_evaluation": True,
    }


def failure_class(row: Mapping[str, Any], affirmative: Mapping[str, Any]) -> str:
    """Classify negative outcomes relative to their affirmative counterpart."""
    if not row["valid_evaluation"] or not affirmative["valid_evaluation"]:
        return "invalid_evaluation"
    if not affirmative["exact_correct"]:
        return "affirmative_task_failure"
    if row["exact_correct"]:
        return "successful_negation"
    if row["predicted_candidate"] in row["excluded_candidates"]:
        return "negation_specific_failure"
    return "unrelated_prediction_error"
