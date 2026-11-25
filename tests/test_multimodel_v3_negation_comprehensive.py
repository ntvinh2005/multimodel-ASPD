import json
import sqlite3
from types import SimpleNamespace

import pytest

from aspd.multimodel.cli.measure_negation_comprehensive import (
    bundle_examples,
    cached_trace_targets,
    load_cache,
    official_targets,
    plan_groups,
)
from aspd.multimodel.posthoc.ids import ComponentId
from aspd.multimodel.v3.negation import FACTORIAL


def _row(condition, index):
    return {
        "anchor_id": f"anchor-{index}",
        "example_id": f"example-{index}",
        "group_id": "group",
        "condition": condition,
        "family": "binary",
        "pool_id": "pool",
        "model_positions": {"gemma": 4, "smol": 3, "granite": 4},
        "pre": {"active_gate_ids": [3], "values": [0.5]},
        "mid": {"active_gate_ids": [3], "values": [0.25]},
    }


def test_all_active_gates_and_missing_representatives():
    rows = [_row(condition, i) for i, condition in enumerate(FACTORIAL)]
    pre = ComponentId.from_key("gemma.L12.q/3")
    mid = ComponentId.from_key("smol.L09.mlp_up/3")
    reps = {(c.gate, c.model): (c,) for c in (pre, mid)}
    expected = {(row["anchor_id"], c) for row in rows for c in (pre, mid)}
    assert official_targets(rows, reps) == expected
    detail = plan_groups({"group": rows}, reps)["benchmark"]
    assert detail["activeGateAnchors"] == 8
    assert detail["representativeTargets"] == 8
    assert detail["gateModelPairsWithoutRepresentatives"] == 16
    stats = SimpleNamespace(sigma=2.0, beta=[0, 0, 0, 0.2], rho=[0, 0, 0, 0.1])
    importance = {c.matrix: stats for c in (pre, mid)}
    cache = {key: [0.0, 1.0, []] for key in expected}
    query = bundle_examples(rows, expected, set(), cache, reps, importance)[0]
    assert set(query["graphs"]) == {"PRE:3", "MID:3"}
    assert query["graphs"]["PRE:3"]["models"]["gemma"]["strength"] == 0.0
    assert query["graphs"]["PRE:3"]["models"]["granite"]["strength"] is None
    assert query["graphs"]["PRE:3"]["comparisons"][0]["jaccard"] is None


def test_exact_anchor_cache_frontier_and_separate_aggregation(tmp_path):
    row = _row("affirmative", 0)
    anchor = row["anchor_id"]
    root, parent, grandparent = [
        ComponentId.from_key(f"gemma.L{layer}.q/3") for layer in (12, 11, 10)
    ]
    reps = {(root.gate, root.model): (root,)}
    stats = SimpleNamespace(sigma=2.0, beta=[0, 0, 0, 0.2], rho=[0, 0, 0, 0.1])
    importance = {c.matrix: stats for c in (root, parent, grandparent)}
    official = {(anchor, root)}
    with sqlite3.connect(tmp_path / "cache.sqlite") as db:
        assert not load_cache(db, "contract")
        db.executemany(
            "INSERT INTO records VALUES (?, ?, ?, ?)",
            [
                ("contract", anchor, root.key, json.dumps([1, 2, [[parent.key, -0.4]]])),
                ("contract", anchor, parent.key, json.dumps([2, 2, [[grandparent.key, 0.3]]])),
                ("contract", anchor, grandparent.key, json.dumps([3, 2, []])),
                ("contract", "foreign-anchor", parent.key, json.dumps([9, 2, []])),
                ("foreign-contract", anchor, root.key, json.dumps([9, 2, []])),
            ],
        )
        cache = load_cache(db, "contract")
        assert len(cache) == 4
    reached, _ = cached_trace_targets(official, cache, reps, importance)
    assert reached == {(anchor, parent), (anchor, grandparent)}
    representative_only = bundle_examples([row], official, set(), cache, reps, importance)[0]
    with_trace = bundle_examples([row], official, reached, cache, reps, importance)[0]
    assert with_trace["graphs"] == representative_only["graphs"]
    assert with_trace["traceTargets"][parent.key]["parents"][0]["E"] == 0.3
    cache[anchor, parent] = [2, 2, [[root.key, 0.3]]]
    with pytest.raises(ValueError, match="compute order"):
        cached_trace_targets(official, cache, reps, importance)
