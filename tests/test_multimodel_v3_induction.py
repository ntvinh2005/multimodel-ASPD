from __future__ import annotations

import re

import pytest
import torch

from aspd.multimodel.posthoc.backend import AnchorPool, EncodedAnchorPool, ModelInputs
from aspd.multimodel.v3.induction import (
    CONDITIONS,
    MODEL_ORDER,
    DatasetPlan,
    behavioral_metrics,
    generate_examples,
    merge_task_and_background,
    pack_verified_groups,
    task_pool_from_verified,
    verify_example,
)
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, ModelPoolInputs


def _plan(groups: int = 4) -> DatasetPlan:
    return DatasetPlan(
        seed=7,
        split_groups={"smoke": groups, "discovery": groups, "heldout": groups},
        family_weights={"basic": 2, "repeated_sequence": 1, "natural_language": 1},
        max_sequence_tokens=128,
        pool_anchors=32,
        task_anchor_budget_per_pool=24,
    )


class WordTokenizer:
    is_fast = True

    def __init__(self) -> None:
        self._piece_to_id = {"<bos>": 0}
        self._id_to_piece = {0: ""}

    def _pieces(self, text: str):
        return [(match.start(), match.end(), match.group()) for match in re.finditer(r"\s*\S+", text)]

    def __call__(self, text: str, **kwargs):
        pieces = self._pieces(text)
        max_length = kwargs.get("max_length") if kwargs.get("truncation") else None
        rows = [(0, 0, "<bos>"), *pieces]
        if max_length is not None:
            rows = rows[:max_length]
        ids = []
        offsets = []
        for start, stop, piece in rows:
            if piece not in self._piece_to_id:
                index = len(self._piece_to_id)
                self._piece_to_id[piece] = index
                self._id_to_piece[index] = piece
            ids.append(self._piece_to_id[piece])
            offsets.append((start, stop))
        result = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result

    def decode(self, token_ids, **_kwargs):
        return "".join(self._id_to_piece[int(value)] for value in token_ids)


def test_generation_is_deterministic_split_disjoint_and_paired() -> None:
    first = generate_examples(_plan())
    second = generate_examples(_plan())
    assert first == second
    assert len(first) == 3 * 4 * 4
    group_ids = {split: set() for split in ("smoke", "discovery", "heldout")}
    by_group = {}
    for example in first:
        group_ids[example.split].add(example.group_id)
        by_group.setdefault(example.group_id, []).append(example)
    assert not (group_ids["smoke"] & group_ids["discovery"])
    assert not (group_ids["smoke"] & group_ids["heldout"])
    assert not (group_ids["discovery"] & group_ids["heldout"])
    assert all(tuple(value.condition for value in group) == CONDITIONS for group in by_group.values())


def test_clean_counterfactual_and_no_match_contracts() -> None:
    examples = generate_examples(_plan(groups=8))
    symbolic = next(
        [value for value in examples if value.group_id == group]
        for group in dict.fromkeys(value.group_id for value in examples)
        if next(value for value in examples if value.group_id == group).family == "basic"
    )
    clean, no_match, counterfactual, control = symbolic
    assert clean.expected_continuation == counterfactual.distractor_continuation
    assert clean.distractor_continuation == counterfactual.expected_continuation
    clean_prefix = clean.prompt[
        next(span for span in clean.spans if span.role == "original_prefix").start :
        next(span for span in clean.spans if span.role == "original_prefix").stop
    ]
    no_match_prefix = no_match.prompt[
        next(span for span in no_match.spans if span.role == "original_prefix").start :
        next(span for span in no_match.spans if span.role == "original_prefix").stop
    ]
    assert clean_prefix != no_match_prefix
    assert not no_match.expected_is_ground_truth
    assert not control.expected_is_ground_truth


def test_complete_prompt_tokenization_and_canonical_alignment() -> None:
    example = next(
        value for value in generate_examples(_plan())
        if value.family == "basic" and value.condition == "induction"
    )
    tokenizers = {key: WordTokenizer() for key in MODEL_ORDER}
    record = verify_example(example, tokenizers, model_order=MODEL_ORDER, max_length=128)
    assert record["alignment"]["anchors"][-1]["character_boundary"] == len(example.prompt)
    assert record["final_anchor_id"] == record["alignment"]["anchors"][-1]["anchor_id"]
    assert all(
        record["models"][key]["prediction_token_index"]
        == record["alignment"]["anchors"][-1]["positions"][key]["token_index"]
        for key in MODEL_ORDER
    )


def _background_pool(count: int) -> AlignedAnchorPool:
    models = {
        key: ModelPoolInputs(
            input_ids=tuple((1, index + 2) for index in range(count)),
            attention_masks=tuple((1, 1) for _ in range(count)),
            anchor_document_indices=tuple(range(count)),
            anchor_token_indices=(1,) * count,
        )
        for key in MODEL_ORDER
    }
    return AlignedAnchorPool(
        anchor_ids=tuple(f"background-{index}" for index in range(count)),
        document_ids=tuple(f"doc-{index}" for index in range(count)),
        models=models,
        source_observations=tuple(
            {"canonical_text": f"b {index}", "anchor_character_boundaries": [2]}
            for index in range(count)
        ),
    )


def test_task_pool_pack_merge_and_exact_pool_size() -> None:
    tokenizers = {key: WordTokenizer() for key in MODEL_ORDER}
    examples = [
        value for value in generate_examples(_plan(groups=1)) if value.split == "smoke"
    ]
    verified = [verify_example(value, tokenizers, model_order=MODEL_ORDER) for value in examples]
    packed = pack_verified_groups(verified, task_anchor_budget=100)
    assert len(packed) == 1
    task = task_pool_from_verified(packed[0])
    total = task.anchor_count + 8
    merged = merge_task_and_background(task, _background_pool(total), pool_anchors=total)
    assert merged.anchor_count == total
    assert merged.anchor_ids[: task.anchor_count] == task.anchor_ids
    assert all(len(model.input_ids) == len(merged.document_ids) for model in merged.models.values())


def test_continuous_and_binary_gate_support_must_match() -> None:
    inputs = ModelInputs(
        torch.ones(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.long),
        torch.arange(2), torch.ones(2, dtype=torch.long),
    )
    pool = AnchorPool(
        ("a", "b"), {"pre": torch.tensor([[True, False], [False, True]])}, {"m": inputs}
    )
    EncodedAnchorPool(pool, {"pre": torch.tensor([[0.5, 0.0], [0.0, 0.3]])})
    with pytest.raises(ValueError, match="continuous/binary"):
        EncodedAnchorPool(pool, {"pre": torch.tensor([[0.0, 0.5], [0.0, 0.3]])})


def test_behavioral_metrics_do_not_score_negative_controls_as_correctness() -> None:
    rows = [
        {
            "model": "gemma", "family": "basic", "condition": "induction",
            "expected_is_ground_truth": True, "logit_margin": 2.0,
            "predicted_token_id": 4, "expected_token_id": 4,
        },
        {
            "model": "gemma", "family": "basic", "condition": "no_match",
            "expected_is_ground_truth": False, "logit_margin": -2.0,
            "predicted_token_id": 9, "expected_token_id": 4,
        },
    ]
    metrics = behavioral_metrics(rows)
    assert metrics["gemma/basic/induction"]["pairwise_accuracy"] == 1.0
    assert metrics["gemma/basic/no_match"]["ground_truth_n"] == 0
    assert metrics["gemma/basic/no_match"]["pairwise_accuracy"] is None
