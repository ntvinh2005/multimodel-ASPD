from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from aspd.multimodel.cli.analyze_aspd_v3_feature_rule_validation import (
    _binary_stats,
    _paired_bootstrap,
)
from aspd.multimodel.cli.extract_aspd_v3_feature_coherence import _anchor_priorities
from aspd.multimodel.cli.prepare_aspd_v3_feature_coherence_heldout import (
    stratified_sample,
)
from aspd.multimodel.cli.propose_aspd_v3_feature_rules import _parse_json
from aspd.multimodel.cli.validate_aspd_v3_feature_rules import parse_rule_presence
from aspd.multimodel.v3.feature_coherence import (
    FeatureCoherenceAnalysisContract,
    FeatureCoherenceContract,
    component_key,
    csr_unique_document_counts,
    deduplicate_context_rows,
    feature_event_values,
    near_duplicate,
    normalize_context_text,
)


def test_csr_unique_document_counts_deduplicates_within_document() -> None:
    indptr = np.array([0, 2, 4, 5], dtype=np.int64)
    indices = np.array([0, 1, 0, 2, 0], dtype=np.uint16)
    documents = np.array([4, 4, 8], dtype=np.int32)
    observed = csr_unique_document_counts(
        indptr, indices, documents, dictionary_size=4
    )
    assert observed.tolist() == [2, 1, 1, 0]


def test_feature_event_values_uses_selected_sparse_events() -> None:
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([2, 5, 2], dtype=np.uint16)
    values = np.array([0.3, 0.4, 0.8], dtype=np.float32)
    assert feature_event_values(indptr, indices, values, feature=2).tolist() == pytest.approx(
        [0.3, 0.8]
    )


def test_component_key_is_fail_closed() -> None:
    assert component_key("pre", 7) == "pre/0007"
    with pytest.raises(ValueError):
        component_key("post", 7)
    with pytest.raises(ValueError):
        component_key("mid", 8_192)


def test_frozen_contract_loads() -> None:
    contract = FeatureCoherenceContract.load(
        "configs/multimodel/aspd_v3/main500m_feature_coherence_v1.yaml"
    )
    assert contract.training_authorized_by_this_contract is False
    assert contract.forbidden_model_substrings == ("qwen",)
    assert contract.activation_quantiles == pytest.approx((1 / 3, 2 / 3))


def test_frozen_analysis_contract_uses_independent_models() -> None:
    contract = FeatureCoherenceAnalysisContract.load(
        "configs/multimodel/aspd_v3/main500m_feature_coherence_analysis_v1.yaml"
    )
    assert contract.human_evidence_claimed is False
    assert contract.proposer_model_id.startswith("google/")
    assert contract.validator_model_id.startswith("microsoft/")
    assert contract.proposer_model_id != contract.validator_model_id


def test_negative_sampling_priorities_are_deterministic_and_type_specific() -> None:
    anchors = ["anchor-a", "anchor-b", "anchor-c"]
    first = _anchor_priorities(anchors, 27, "ordinary")
    assert first.tolist() == _anchor_priorities(anchors, 27, "ordinary").tolist()
    assert first.tolist() != _anchor_priorities(anchors, 27, "hard").tolist()
    assert first.tolist() != _anchor_priorities(anchors, 28, "ordinary").tolist()


def test_near_duplicate_normalizes_anchor_markers_and_whitespace() -> None:
    left = "Alpha  ⟪ANCHOR⟫beta⟪/ANCHOR⟫ gamma delta epsilon"
    right = " alpha beta\n gamma delta epsilon "
    assert normalize_context_text(left) == normalize_context_text(right)
    assert near_duplicate(left, right, threshold=0.90)


def test_deduplicate_context_rows_preserves_input_priority() -> None:
    rows = [
        {"text": "The quick brown fox jumps over the lazy dog", "id": 1},
        {"text": "The quick brown fox jumps over  the lazy dog", "id": 2},
        {"text": "A different context about matrix multiplication", "id": 3},
    ]
    assert [row["id"] for row in deduplicate_context_rows(rows, threshold=0.90)] == [
        1,
        3,
    ]


def test_stratified_sample_round_robins_groups() -> None:
    rows = [
        {"row_id": f"a{index}", "group": "a", "bin": "weak"}
        for index in range(5)
    ] + [
        {"row_id": f"b{index}", "group": "b", "bin": "strong"}
        for index in range(5)
    ]
    sampled = stratified_sample(rows, count=4, seed=7, strata=("group", "bin"))
    assert Counter(row["group"] for row in sampled) == {"a": 2, "b": 2}


def test_proposer_parser_preserves_invalid_backslash_as_literal() -> None:
    observed = _parse_json(r'{"rule": "delimiters like \| and +"}')
    assert observed == {"rule": r"delimiters like \| and +"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("YES", "YES"), ("Answer: no.", "NO"), ("UNCERTAIN", "UNCERTAIN"), ("maybe", None)],
)
def test_rule_presence_parser(raw: str, expected: str | None) -> None:
    assert parse_rule_presence(raw) == expected


def test_binary_stats_uses_balanced_accuracy() -> None:
    rows = [
        {"actual_natural_firing": True, "prediction": True},
        {"actual_natural_firing": True, "prediction": False},
        {"actual_natural_firing": False, "prediction": False},
        {"actual_natural_firing": False, "prediction": False},
    ]
    observed = _binary_stats(rows, "prediction")
    assert observed["sensitivity"] == 0.5
    assert observed["specificity"] == 1.0
    assert observed["balanced_accuracy"] == 0.75


def test_paired_bootstrap_preserves_general_multi_comparison() -> None:
    rows = []
    for index in range(8):
        rows.extend(
            [
                {
                    "pair_id": f"pair-{index}",
                    "actual_natural_firing": True,
                    "general_yes": False,
                    "multi_yes": True,
                },
                {
                    "pair_id": f"pair-{index}",
                    "actual_natural_firing": False,
                    "general_yes": False,
                    "multi_yes": False,
                },
            ]
        )
    general, multi, advantage = _paired_bootstrap(rows, replicates=20, seed=9)
    assert general.tolist() == pytest.approx([0.5] * 20)
    assert multi.tolist() == pytest.approx([1.0] * 20)
    assert advantage.tolist() == pytest.approx([0.5] * 20)
