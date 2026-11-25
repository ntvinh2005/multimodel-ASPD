from collections import Counter

import pytest

from aspd.multimodel.v3.negation import (
    FACTORIAL,
    build_group,
    candidate_behavior,
    candidate_groups,
    failure_class,
    pack_groups,
)


def group(family, options=("apple", "pear")):
    return build_group(
        "discovery",
        family,
        "test",
        options,
        actor="Mira",
        other="Leo",
        attributes=("blue", "green"),
        template=0,
        seed=42,
    )


def test_factorial_scope_and_multi_answer():
    rows = group("binary")
    assert tuple(row["condition"] for row in rows) == FACTORIAL
    assert [row["valid_alternatives"] for row in rows] == [["apple"], ["pear"], ["pear"], ["apple"]]
    assert all(row["candidate_orientation"] == ["apple", "pear"] for row in rows)
    assert [row["valid_alternatives"] for row in group("attribute")] == [
        ["apple"],
        ["pear"],
        ["pear"],
        ["apple"],
    ]
    negative = group("multiple", ("apple", "pear", "stone"))[1]
    assert negative["valid_alternatives"] == ["pear", "stone"]
    assert negative["expected_continuation"] is None
    _, negative, _, _, control, _ = group("scope")
    assert negative["negation_scope"] == "Mira" and control["negation_scope"] == "Leo"
    assert negative["valid_alternatives"] != control["valid_alternatives"]
    assert Counter(negative["prompt"].split()) == Counter(control["prompt"].split())


def test_generation_and_pool_packing():
    symbols = ["pear", "apple", "stone", "music"]
    examples = list(candidate_groups(symbols, "discovery", "binary", 42))
    assert examples == list(candidate_groups(symbols, "discovery", "binary", 42))
    assert len({tuple(sorted(rows[0]["candidate_order"])) for rows in examples}) == len(examples)
    heldout = list(candidate_groups(symbols, "heldout", "binary", 42))
    assert {rows[0]["template_id"] for rows in examples}.isdisjoint(
        rows[0]["template_id"] for rows in heldout
    )
    packed = pack_groups([[{"alignment": {"anchors": [1]}}] * 4] * 2, 7)
    assert list(map(len, packed)) == [4, 4]
    with pytest.raises(ValueError):
        pack_groups(packed, 3)


def test_behavior_preserves_fixed_orientation_and_failure_prerequisite():
    affirmative, negative, *_ = group("multiple", ("apple", "pear", "stone"))
    logits = {"apple": 3.0, "pear": 2.0, "stone": 1.0}
    probs = {"apple": 0.3, "pear": 0.2, "stone": 0.1}
    a = candidate_behavior(affirmative, logits, probs, "apple")
    n = candidate_behavior(negative, logits, probs, "apple")
    assert a["fixed_margin"] == n["fixed_margin"] == 1.0
    assert a["correctness_margin"] == -n["correctness_margin"] == 1.0
    assert n["valid_mass"] == pytest.approx(0.3)
    assert n["excluded_mass"] == pytest.approx(0.3)
    assert n["unrelated_mass"] == pytest.approx(0.4)
    assert failure_class({**negative, **n}, a) == "negation_specific_failure"
    assert (
        failure_class({**negative, **n}, {**a, "exact_correct": False})
        == "affirmative_task_failure"
    )
