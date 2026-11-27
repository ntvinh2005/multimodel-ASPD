import numpy as np
import pytest

from aspd.multimodel.discovery.semantic_ot import (
    cluster_pairs,
    comparison,
    is_heldout,
    normalized,
    transport,
    without_terminal_unit,
)


def test_transport_respects_both_marginals():
    vectors = np.eye(2)
    distance, plan, cost = transport(vectors, vectors, [3, 1], [1, 3])
    assert distance == pytest.approx(0.5)
    assert plan.sum(axis=1) == pytest.approx([0.75, 0.25])
    assert plan.sum(axis=0) == pytest.approx([0.25, 0.75])
    assert distance == pytest.approx((plan * cost).sum())


def test_related_contexts_can_match_at_disjoint_anchors():
    embeddings = np.array([[1.0, 0.0]] * 6)
    supports = {i: (1.0, 1.0) for i in range(6)}
    cache = {i: i for i in range(6)}
    score, plan = comparison(
        {i: supports[i] for i in range(3)},
        {i: supports[i] for i in range(3, 6)},
        embeddings,
        cache,
    )
    assert score["ot"] == pytest.approx(0)
    assert score["anchorOverlap"] == 0
    assert all(cell["left"] < 3 <= cell["right"] for cell in plan)
    assert comparison({0: (1, 1)}, supports, embeddings, cache) == (None, [])


def test_cluster_requires_each_pair_to_be_measured_and_close():
    assert cluster_pairs(4, [(0, 1, 0.29), (1, 2, 0.31), (2, 3, 0.25)], 0.5) == [[0, 1], [2, 3]]
    assert cluster_pairs(3, [(0, 1, 0.1), (0, 2, 0.15), (1, 2, 0.2)], 0.2) == [[0, 1, 2]]


def test_missing_or_invalid_mass_cannot_be_a_distribution():
    for values in ([], [0, 0], [1, -1], [np.nan]):
        with pytest.raises(ValueError, match="positive total mass"):
            normalized(values)
    assert is_heldout("document-1", 61036) == is_heldout("document-1", 61036)


def test_suffix_control_keeps_only_the_causal_left_context():
    assert without_terminal_unit("The calendar says March") == "The calendar says"
    assert without_terminal_unit("The calendar says March.") == "The calendar says March"
