import pytest

from aspd.multimodel.cli.export_aspd_semantic_pair_graphs import comparison_examples


def test_shared_token_with_both_graphs_precedes_paired_contexts():
    left = {"a": 3.0, "b": 2.0, "c": 1.0}
    right = {"a": 1.0, "b": 2.0, "d": 2.0}
    graphs = ({"a": None, "b": True, "c": True}, {"a": True, "b": True, "d": True})
    transport = [{"left": "c", "right": "d", "kind": "match"}]
    actual = comparison_examples(left, right, transport, graphs)
    assert actual["sharedSupport"] == 2
    assert actual["examples"] == [
        {"left": "b", "right": "b", "kind": "shared"},
        {"left": "c", "right": "d", "kind": "transport"},
    ]


def test_unscored_context_pairs_remain_separate_and_missing_graphs_remain_missing():
    assert comparison_examples({"a": 2.0}, {"b": 3.0}, [], ({"a": True}, {"b": None})) == {
        "sharedSupport": 0,
        "examples": [{"left": "a", "right": "b", "kind": "separate"}],
    }
    assert comparison_examples({"a": 2.0}, {"a": 3.0}, [], ({"a": None}, {"a": None})) == {
        "sharedSupport": 1,
        "examples": [],
    }


def test_rejects_transport_with_an_anchor_outside_the_components_saved_contexts():
    with pytest.raises(ValueError, match="outside"):
        comparison_examples(
            {"a": 2.0},
            {"b": 3.0},
            [{"left": "unrelated", "right": "b", "kind": "match"}],
            ({"a": True}, {"b": True}),
        )
