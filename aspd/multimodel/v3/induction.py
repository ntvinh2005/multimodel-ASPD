"""Deterministic, tokenizer-verified induction evaluation data.

This module deliberately contains no model or checkpoint loading.  It builds paired
clean/control prompts, verifies exact next-token targets under heterogeneous tokenizers,
and converts the verified prompts into the existing canonical-anchor pool format.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from aspd.multimodel.v3.alignment import AlignmentResult, align_canonical_text
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, ModelPoolInputs

MODEL_ORDER = ("gemma", "smol", "granite")
SPLITS = ("smoke", "discovery", "heldout")
CONDITIONS = ("induction", "no_match", "counterfactual", "repeated_no_association")

DatasetFamily = Literal["basic", "repeated_sequence", "natural_language"]
Condition = Literal["induction", "no_match", "counterfactual", "repeated_no_association"]


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Span:
    role: str
    start: int
    stop: int

    def validate(self, text: str) -> None:
        if not self.role or not 0 <= self.start < self.stop <= len(text):
            raise ValueError(f"invalid span {self!r} for text of length {len(text)}")


@dataclass(frozen=True)
class InductionExample:
    example_id: str
    group_id: str
    split: str
    family: DatasetFamily
    condition: Condition
    prompt: str
    expected_continuation: str
    distractor_continuation: str
    expected_is_ground_truth: bool
    spans: tuple[Span, ...]
    metadata: Mapping[str, Any]

    def validate(self) -> None:
        if self.split not in SPLITS or self.condition not in CONDITIONS:
            raise ValueError("unknown split or condition")
        if not self.prompt or not self.expected_continuation or not self.distractor_continuation:
            raise ValueError("prompt and continuation strings must be non-empty")
        if self.expected_continuation == self.distractor_continuation:
            raise ValueError("expected and distractor continuations must differ")
        for span in self.spans:
            span.validate(self.prompt)
        query = [span for span in self.spans if span.role == "query_prefix"]
        if not query or max(span.stop for span in query) != len(self.prompt):
            raise ValueError("every prompt needs query_prefix span(s) ending at prediction boundary")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spans"] = [asdict(span) for span in self.spans]
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InductionExample:
        result = cls(
            example_id=str(payload["example_id"]),
            group_id=str(payload["group_id"]),
            split=str(payload["split"]),
            family=str(payload["family"]),  # type: ignore[arg-type]
            condition=str(payload["condition"]),  # type: ignore[arg-type]
            prompt=str(payload["prompt"]),
            expected_continuation=str(payload["expected_continuation"]),
            distractor_continuation=str(payload["distractor_continuation"]),
            expected_is_ground_truth=bool(payload["expected_is_ground_truth"]),
            spans=tuple(Span(**span) for span in payload["spans"]),
            metadata=dict(payload.get("metadata", {})),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class DatasetPlan:
    seed: int
    split_groups: Mapping[str, int]
    family_weights: Mapping[str, int]
    max_sequence_tokens: int = 128
    pool_anchors: int = 2048
    task_anchor_budget_per_pool: int = 1536

    def validate(self) -> None:
        if tuple(self.split_groups) != SPLITS:
            raise ValueError(f"split_groups must be ordered as {SPLITS}")
        if any(int(self.split_groups[key]) <= 0 for key in SPLITS):
            raise ValueError("all split budgets must be positive")
        if tuple(self.family_weights) != ("basic", "repeated_sequence", "natural_language"):
            raise ValueError("family_weights order changed")
        if any(int(value) <= 0 for value in self.family_weights.values()):
            raise ValueError("family weights must be positive")
        if not 0 < self.task_anchor_budget_per_pool < self.pool_anchors:
            raise ValueError("task anchor budget must leave deterministic background capacity")

    def fingerprint(self) -> str:
        return canonical_sha256(asdict(self))


# Common words are only a deterministic proposal reservoir.  A word is retained only if
# the complete prompt + leading-space continuation is exactly one additional native token
# under all three pinned tokenizers.
SYMBOL_RESERVOIR = tuple(
    "amber apple arch arrow atlas autumn badge baker basin beach berry birch blade bloom "
    "blue brass breeze brick brook brush cedar chalk charm chess cloud coral crane creek "
    "crown dawn delta dream drift eagle earth ember field flame fleet flint frost garden "
    "glass glen globe gold grape green grove harbor hazel hill honey ivory jade lake leaf "
    "lemon light lilac lime lodge lotus maple marble meadow metal mint moon moss north ocean "
    "olive orange orchid pearl pine plum pond quartz raven reed ridge river robin rose ruby "
    "sand scarlet shell shore silver slate snow solar south spark spring star steel stone "
    "storm summer swift table tiger trail tulip valley violet water wheat white willow wind "
    "winter wood yellow zenith".split()
)

ENTITY_RESERVOIR = tuple(
    "Dax Nira Pavo Taren Vela Zorin Kima Leto Mavi Runa Sefa Tova Wren Yara Bex Ciro Fenn "
    "Galo Hira Juno Kato Lira Nexo Orin Pera Quill Rivo Sora Timo Ula Varo Xena Yori Zale".split()
)


def _spans_for_tokens(text: str, tokens: Sequence[str], roles: Sequence[str]) -> tuple[Span, ...]:
    if len(tokens) != len(roles):
        raise ValueError("tokens and roles differ")
    spans: list[Span] = []
    cursor = 0
    for index, (token, role) in enumerate(zip(tokens, roles, strict=True)):
        start = cursor + (1 if index else 0)
        stop = start + len(token)
        if text[start:stop] != token:
            raise AssertionError("token span construction failed")
        if role:
            spans.append(Span(role, start, stop))
        cursor = stop
    return tuple(spans)


def _symbol_prompt(tokens: Sequence[str], roles: Sequence[str]) -> tuple[str, tuple[Span, ...]]:
    text = " ".join(tokens)
    return text, _spans_for_tokens(text, tokens, roles)


def _make_symbolic_group(
    *, rng: random.Random, split: str, group_id: str, family: DatasetFamily,
    symbols: Sequence[str],
) -> tuple[InductionExample, ...]:
    if family not in ("basic", "repeated_sequence"):
        raise ValueError("symbolic builder only supports basic and repeated_sequence")
    prefix_length = rng.choice((1, 1, 2, 2, 3))
    sequence_length = rng.randint(max(prefix_length + 3, 5), 10)
    needed = sequence_length + prefix_length + 5
    sampled = rng.sample(symbols, needed)
    original = sampled[:sequence_length]
    prefix = original[:prefix_length]
    expected = original[prefix_length]
    distractor = original[prefix_length + 1]

    if family == "basic":
        # Add unrelated material before the query while keeping the one intended match.
        middle = original[prefix_length + 1 :] + sampled[sequence_length : sequence_length + 2]
        clean_tokens = prefix + [expected] + middle + prefix
    else:
        # Classical repeated sequence: S + S[:-1], with the omitted last token as target.
        expected = original[-1]
        distractor = sampled[sequence_length]
        prefix = original[:-1]
        prefix_length = len(prefix)
        # A matched distractor is retained outside the repeated prefix so the
        # counterfactual can exchange the two continuations without changing inventory.
        clean_tokens = original + [distractor] + prefix

    first_prefix_stop = len(prefix) - 1
    clean_roles = ["" for _ in clean_tokens]
    for index in range(prefix_length):
        clean_roles[index] = "original_prefix"
        clean_roles[-prefix_length + index] = "query_prefix"
    clean_roles[prefix_length] = "original_continuation"
    for index, value in enumerate(clean_tokens):
        if value == distractor and not clean_roles[index]:
            clean_roles[index] = "distractor"
            break
    clean_prompt, clean_spans = _symbol_prompt(clean_tokens, clean_roles)

    replacement = sampled[-1]
    no_match_tokens = list(clean_tokens)
    no_match_tokens[:prefix_length] = [replacement] + list(prefix[1:])
    no_match_roles = list(clean_roles)
    no_match_prompt, no_match_spans = _symbol_prompt(no_match_tokens, no_match_roles)

    counter_tokens = list(clean_tokens)
    counter_tokens[prefix_length] = distractor
    distractor_index = next(
        index for index in range(prefix_length + 1, len(counter_tokens) - prefix_length)
        if counter_tokens[index] == distractor
    )
    counter_tokens[distractor_index] = expected
    counter_roles = list(clean_roles)
    counter_roles[prefix_length] = "original_continuation"
    counter_roles[distractor_index] = "distractor"
    counter_prompt, counter_spans = _symbol_prompt(counter_tokens, counter_roles)

    # The query prefix has appeared before, but it was followed by the distractor, not
    # the matched clean target.  Repeated filler tokens provide generic copying cues.
    control_tokens = list(clean_tokens)
    control_tokens[prefix_length] = distractor
    for index in range(prefix_length + 1, len(control_tokens) - prefix_length):
        if control_tokens[index] == expected:
            control_tokens[index] = replacement
    control_roles = list(clean_roles)
    control_roles[prefix_length] = "control_continuation"
    control_prompt, control_spans = _symbol_prompt(control_tokens, control_roles)

    common = {
        "split": split,
        "family": family,
        "group_id": group_id,
        "metadata": {
            "prefix_length": prefix_length,
            "repetition_distance_tokens": len(clean_tokens) - 2 * prefix_length,
            "original_sequence_length": sequence_length,
            "first_prefix_stop_index": first_prefix_stop,
        },
    }
    examples = (
        InductionExample(
            example_id=f"{group_id}/induction", condition="induction", prompt=clean_prompt,
            expected_continuation=f" {expected}", distractor_continuation=f" {distractor}",
            expected_is_ground_truth=True, spans=clean_spans, **common,
        ),
        InductionExample(
            example_id=f"{group_id}/no_match", condition="no_match", prompt=no_match_prompt,
            expected_continuation=f" {expected}", distractor_continuation=f" {distractor}",
            expected_is_ground_truth=False, spans=no_match_spans, **common,
        ),
        InductionExample(
            example_id=f"{group_id}/counterfactual", condition="counterfactual",
            prompt=counter_prompt, expected_continuation=f" {distractor}",
            distractor_continuation=f" {expected}", expected_is_ground_truth=True,
            spans=counter_spans, **common,
        ),
        InductionExample(
            example_id=f"{group_id}/repeated_no_association",
            condition="repeated_no_association", prompt=control_prompt,
            expected_continuation=f" {expected}", distractor_continuation=f" {distractor}",
            expected_is_ground_truth=False, spans=control_spans, **common,
        ),
    )
    for example in examples:
        example.validate()
    return examples


def _find_all(text: str, needle: str) -> list[tuple[int, int]]:
    result = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return result
        result.append((index, index + len(needle)))
        start = index + len(needle)


def _make_natural_group(
    *, rng: random.Random, split: str, group_id: str, symbols: Sequence[str],
) -> tuple[InductionExample, ...]:
    entity, other, third = rng.sample(ENTITY_RESERVOIR, 3)
    expected, distractor, filler = rng.sample(symbols, 3)

    def build(first_entity: str, first_label: str, second_label: str, *, control: bool = False):
        sentence = (
            f'The code for {first_entity} is {first_label}. The code for {other} is '
            f'{second_label}. A note repeats {filler} and {filler}. The code for {entity} is'
        )
        spans = []
        first = _find_all(sentence, first_entity)[0]
        query = _find_all(sentence, entity)[-1]
        spans.append(Span("original_prefix", *first))
        label_span = _find_all(sentence, first_label)[0]
        spans.append(Span("control_continuation" if control else "original_continuation", *label_span))
        spans.append(Span("distractor", *_find_all(sentence, second_label)[0]))
        spans.append(Span("query_prefix", query[0], len(sentence)))
        return sentence, tuple(spans)

    clean, clean_spans = build(entity, expected, distractor)
    no_match, no_match_spans = build(third, expected, distractor)
    counter, counter_spans = build(entity, distractor, expected)
    control, control_spans = build(entity, distractor, filler, control=True)
    common = {
        "split": split,
        "family": "natural_language",
        "group_id": group_id,
        "metadata": {"entity": entity, "other_entity": other, "prefix_length": 1},
    }
    examples = (
        InductionExample(f"{group_id}/induction", condition="induction", prompt=clean,
                         expected_continuation=f" {expected}",
                         distractor_continuation=f" {distractor}",
                         expected_is_ground_truth=True, spans=clean_spans, **common),
        InductionExample(f"{group_id}/no_match", condition="no_match", prompt=no_match,
                         expected_continuation=f" {expected}",
                         distractor_continuation=f" {distractor}",
                         expected_is_ground_truth=False, spans=no_match_spans, **common),
        InductionExample(f"{group_id}/counterfactual", condition="counterfactual", prompt=counter,
                         expected_continuation=f" {distractor}",
                         distractor_continuation=f" {expected}",
                         expected_is_ground_truth=True, spans=counter_spans, **common),
        InductionExample(f"{group_id}/repeated_no_association",
                         condition="repeated_no_association", prompt=control,
                         expected_continuation=f" {expected}",
                         distractor_continuation=f" {distractor}",
                         expected_is_ground_truth=False, spans=control_spans, **common),
    )
    for example in examples:
        example.validate()
    return examples


def generate_examples(
    plan: DatasetPlan, *, symbols: Sequence[str] = SYMBOL_RESERVOIR
) -> tuple[InductionExample, ...]:
    """Generate all requested splits with globally disjoint association signatures."""

    plan.validate()
    if len(set(symbols)) != len(symbols) or len(symbols) < 24:
        raise ValueError("generation requires at least 24 unique validated symbols")
    rng = random.Random(plan.seed)
    families = tuple(plan.family_weights)
    weights = tuple(int(plan.family_weights[key]) for key in families)
    examples: list[InductionExample] = []
    seen_associations: set[tuple[str, str, str]] = set()
    for split in SPLITS:
        accepted = 0
        attempt = 0
        while accepted < int(plan.split_groups[split]):
            attempt += 1
            if attempt > 100_000:
                raise RuntimeError("could not generate disjoint deterministic induction groups")
            family = rng.choices(families, weights=weights, k=1)[0]
            group_id = f"{split}-{accepted:04d}"
            group = (
                _make_natural_group(
                    rng=rng, split=split, group_id=group_id, symbols=symbols
                )
                if family == "natural_language"
                else _make_symbolic_group(
                    rng=rng, split=split, group_id=group_id, family=family,  # type: ignore[arg-type]
                    symbols=symbols,
                )
            )
            clean = group[0]
            signature = (
                clean.family,
                clean.expected_continuation,
                canonical_sha256([span.role for span in clean.spans] + clean.prompt.split()),
            )
            if signature in seen_associations:
                continue
            seen_associations.add(signature)
            examples.extend(group)
            accepted += 1
    return tuple(examples)


def _flat_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    ids = tokenizer(text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("batched tokenizer output is unsupported")
        ids = ids[0]
    return tuple(int(value) for value in ids)


def _one_token_continuation(tokenizer: Any, prompt: str, continuation: str) -> tuple[int, int]:
    prompt_ids = _flat_ids(tokenizer, prompt)
    full_ids = _flat_ids(tokenizer, prompt + continuation)
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) != len(prompt_ids) + 1:
        raise ValueError("continuation is not exactly one token at this complete-prompt boundary")
    return full_ids[-1], len(prompt_ids) - 1


def shared_single_token_symbols(
    tokenizers: Mapping[str, Any],
    *,
    candidates: Sequence[str] = SYMBOL_RESERVOIR,
    model_order: Sequence[str] = MODEL_ORDER,
) -> tuple[str, ...]:
    """Pre-filter proposal words without relaxing complete-example validation."""

    if tuple(tokenizers) != tuple(model_order):
        raise ValueError("tokenizer order differs from the configured model order")
    retained = []
    for symbol in candidates:
        try:
            token_ids = {
                key: _one_token_continuation(tokenizers[key], "Context", f" {symbol}")[0]
                for key in model_order
            }
        except ValueError:
            continue
        if len(token_ids) == len(model_order):
            retained.append(symbol)
    return tuple(retained)


def verify_example(
    example: InductionExample,
    tokenizers: Mapping[str, Any],
    *,
    model_order: Sequence[str] = MODEL_ORDER,
    max_length: int = 128,
) -> dict[str, Any]:
    """Return complete token/alignment evidence or raise on any approximation."""

    example.validate()
    if tuple(tokenizers) != tuple(model_order):
        raise ValueError("tokenizer order differs from the configured model order")
    alignment = align_canonical_text(
        example.prompt,
        document_id=example.example_id,
        source_group=f"induction/{example.split}/{example.family}/{example.condition}",
        tokenizers=tokenizers,
        model_order=model_order,
        max_length=max_length,
    )
    if any(document.truncated for document in alignment.tokenized.values()):
        raise ValueError("prompt exceeded the configured sequence length")
    anchors = {anchor.character_boundary: anchor for anchor in alignment.anchors}
    required_boundaries = {len(example.prompt), *(span.stop for span in example.spans)}
    missing = sorted(required_boundaries - anchors.keys())
    if missing:
        raise ValueError(f"required canonical boundaries are not exact three-way anchors: {missing}")
    models: dict[str, Any] = {}
    for key in model_order:
        expected, prediction_position = _one_token_continuation(
            tokenizers[key], example.prompt, example.expected_continuation
        )
        distractor, distractor_position = _one_token_continuation(
            tokenizers[key], example.prompt, example.distractor_continuation
        )
        if prediction_position != distractor_position or expected == distractor:
            raise ValueError("expected/distractor token boundary is inconsistent")
        tokenized = alignment.tokenized[key]
        final_anchor = anchors[len(example.prompt)].positions[key]
        if final_anchor.token_index != prediction_position:
            raise ValueError("final canonical anchor is not the native next-token logit position")
        models[key] = {
            "input_ids": list(tokenized.input_ids),
            "attention_mask": list(tokenized.attention_mask),
            "prediction_token_index": prediction_position,
            "expected_token_id": expected,
            "distractor_token_id": distractor,
        }
    return {
        "example": example.to_dict(),
        "models": models,
        "alignment": _alignment_payload(alignment),
        "required_boundaries": sorted(required_boundaries),
        "final_anchor_id": anchors[len(example.prompt)].anchor_id,
    }


def _alignment_payload(alignment: AlignmentResult) -> dict[str, Any]:
    return {
        "report": alignment.report(),
        "anchors": [
            {
                "anchor_id": anchor.anchor_id,
                "character_boundary": anchor.character_boundary,
                "byte_boundary": anchor.byte_boundary,
                "canonical_prefix_sha256": anchor.canonical_prefix_sha256,
                "positions": {
                    key: {
                        "token_index": position.token_index,
                        "token_id": position.token_id,
                        "prefix_token_count": position.prefix_token_count,
                    }
                    for key, position in anchor.positions.items()
                },
            }
            for anchor in alignment.anchors
        ],
    }


def task_pool_from_verified(
    records: Sequence[Mapping[str, Any]], *, mechanism: str = "induction",
) -> AlignedAnchorPool:
    """Convert verified examples into one pool fragment containing every shared anchor."""

    if not records:
        raise ValueError("task pool requires verified records")
    anchor_ids: list[str] = []
    document_ids: list[str] = []
    observations: list[dict[str, Any]] = []
    model_rows: dict[str, dict[str, list[Any]]] = {
        key: {"ids": [], "masks": [], "documents": [], "tokens": []} for key in MODEL_ORDER
    }
    for document_index, record in enumerate(records):
        example = record["example"]
        document_ids.append(str(example["example_id"]))
        anchors = record["alignment"]["anchors"]
        boundaries = []
        for anchor in anchors:
            anchor_ids.append(str(anchor["anchor_id"]))
            boundaries.append(int(anchor["character_boundary"]))
            for key in MODEL_ORDER:
                model_rows[key]["documents"].append(document_index)
                model_rows[key]["tokens"].append(int(anchor["positions"][key]["token_index"]))
        for key in MODEL_ORDER:
            model_rows[key]["ids"].append(tuple(int(v) for v in record["models"][key]["input_ids"]))
            model_rows[key]["masks"].append(
                tuple(int(v) for v in record["models"][key]["attention_mask"])
            )
        observations.append(
            {
                "document_id": str(example["example_id"]),
                "source_group": f"{mechanism}/{example['split']}/{example['family']}/{example['condition']}",
                "canonical_text": str(example["prompt"]),
                "anchor_character_boundaries": boundaries,
                "example": dict(example),
                "final_anchor_id": str(record["final_anchor_id"]),
                "required_boundaries": list(record["required_boundaries"]),
            }
        )
    models = {
        key: ModelPoolInputs(
            input_ids=tuple(model_rows[key]["ids"]),
            attention_masks=tuple(model_rows[key]["masks"]),
            anchor_document_indices=tuple(model_rows[key]["documents"]),
            anchor_token_indices=tuple(model_rows[key]["tokens"]),
        )
        for key in MODEL_ORDER
    }
    return AlignedAnchorPool(
        anchor_ids=tuple(anchor_ids),
        document_ids=tuple(document_ids),
        models=models,
        source_observations=tuple(observations),
    )


def pack_verified_groups(
    records: Sequence[Mapping[str, Any]], *, task_anchor_budget: int,
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    """First-fit in group order; all four paired conditions always share one pool."""

    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_group.setdefault(str(record["example"]["group_id"]), []).append(record)
    packed: list[tuple[Mapping[str, Any], ...]] = []
    current: list[Mapping[str, Any]] = []
    current_anchors = 0
    for group_id, group in by_group.items():
        conditions = tuple(record["example"]["condition"] for record in group)
        if conditions != CONDITIONS:
            raise ValueError(f"group {group_id} conditions or ordering changed: {conditions}")
        count = sum(len(record["alignment"]["anchors"]) for record in group)
        if count > task_anchor_budget:
            raise ValueError(f"one paired group exceeds task anchor budget: {group_id}")
        if current and current_anchors + count > task_anchor_budget:
            packed.append(tuple(current))
            current, current_anchors = [], 0
        current.extend(group)
        current_anchors += count
    if current:
        packed.append(tuple(current))
    return tuple(packed)


def subset_pool(pool: AlignedAnchorPool, rows: Sequence[int]) -> AlignedAnchorPool:
    """Select arbitrary anchor rows while retaining/remapping their source documents."""

    if not rows or len(set(rows)) != len(rows):
        raise ValueError("pool subset rows must be non-empty and unique")
    if min(rows) < 0 or max(rows) >= pool.anchor_count:
        raise IndexError("pool subset row lies outside the pool")
    first_model = pool.models[next(iter(pool.models))]
    old_documents = []
    for row in rows:
        document = int(first_model.anchor_document_indices[row])
        if document not in old_documents:
            old_documents.append(document)
    remap = {old: new for new, old in enumerate(old_documents)}
    models = {}
    for key, value in pool.models.items():
        row_documents = [int(value.anchor_document_indices[row]) for row in rows]
        models[key] = ModelPoolInputs(
            input_ids=tuple(value.input_ids[old] for old in old_documents),
            attention_masks=tuple(value.attention_masks[old] for old in old_documents),
            anchor_document_indices=tuple(remap[old] for old in row_documents),
            anchor_token_indices=tuple(int(value.anchor_token_indices[row]) for row in rows),
        )
    boundaries_by_document: dict[int, list[int]] = {old: [] for old in old_documents}
    flattened_boundaries = [
        int(boundary)
        for observation in pool.source_observations
        for boundary in observation.get("anchor_character_boundaries", [])
    ]
    if len(flattened_boundaries) == pool.anchor_count:
        for row in rows:
            old = int(first_model.anchor_document_indices[row])
            boundaries_by_document[old].append(flattened_boundaries[row])
    observations = []
    for old in old_documents:
        observation = dict(pool.source_observations[old])
        if flattened_boundaries:
            observation["anchor_character_boundaries"] = boundaries_by_document[old]
        observations.append(observation)
    return AlignedAnchorPool(
        anchor_ids=tuple(pool.anchor_ids[row] for row in rows),
        document_ids=tuple(pool.document_ids[old] for old in old_documents),
        models=models,
        source_observations=tuple(observations),
    )


def merge_task_and_background(
    task: AlignedAnchorPool, background: AlignedAnchorPool, *, pool_anchors: int,
) -> AlignedAnchorPool:
    """Fill a task fragment to one exact BatchTopK pool with fixed natural anchors."""

    if task.anchor_count >= pool_anchors:
        raise ValueError("task fragment must leave at least one background anchor")
    needed = pool_anchors - task.anchor_count
    if background.anchor_count < needed:
        raise ValueError("background pool is too small")
    background = subset_pool(background, tuple(range(needed)))
    collisions = set(task.anchor_ids) & set(background.anchor_ids)
    if collisions:
        raise ValueError("task/background anchor IDs collide")
    models = {}
    for key in MODEL_ORDER:
        left, right = task.models[key], background.models[key]
        offset = len(left.input_ids)
        models[key] = ModelPoolInputs(
            input_ids=left.input_ids + right.input_ids,
            attention_masks=left.attention_masks + right.attention_masks,
            anchor_document_indices=left.anchor_document_indices
            + tuple(offset + int(value) for value in right.anchor_document_indices),
            anchor_token_indices=left.anchor_token_indices + right.anchor_token_indices,
        )
    result = AlignedAnchorPool(
        anchor_ids=task.anchor_ids + background.anchor_ids,
        document_ids=task.document_ids + background.document_ids,
        models=models,
        source_observations=task.source_observations + background.source_observations,
    )
    result.validate(expected_anchors=pool_anchors, sequence_length=512, model_order=MODEL_ORDER)
    return result


def behavioral_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate by model/family/condition without counting controls as ground truth."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["family"]), str(row["condition"]))
        groups.setdefault(key, []).append(row)
    result = {}
    for (model, family, condition), values in sorted(groups.items()):
        margins = sorted(float(value["logit_margin"]) for value in values)
        ground_truth = [value for value in values if value["expected_is_ground_truth"]]
        result[f"{model}/{family}/{condition}"] = {
            "n": len(values),
            "ground_truth_n": len(ground_truth),
            "pairwise_accuracy": (
                sum(float(value["logit_margin"]) > 0 for value in ground_truth) / len(ground_truth)
                if ground_truth else None
            ),
            "exact_next_token_accuracy": (
                sum(int(value["predicted_token_id"]) == int(value["expected_token_id"])
                    for value in ground_truth) / len(ground_truth)
                if ground_truth else None
            ),
            "mean_logit_margin": sum(margins) / len(margins),
            "median_logit_margin": margins[len(margins) // 2] if margins else None,
        }
    return result
