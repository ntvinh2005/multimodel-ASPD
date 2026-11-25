from collections import defaultdict
from pathlib import Path

import pytest

from aspd.multimodel.cli.benchmark_token_dependencies import benchmark_batches, component_requests
from aspd.multimodel.cli.measure_token_dependency_shard import plan_batches, validate_cached
from aspd.multimodel.posthoc.artifacts import sha256_file, write_json
from aspd.multimodel.posthoc.ids import MatrixId
from aspd.multimodel.v3.selection_frame import CONDITIONS, examples


def test_groups_are_deterministic_and_contain_matched_controls():
    rows = list(examples())
    assert rows == list(examples())
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row["condition"])
    assert len(rows) == len({row["example_id"] for row in rows}) == 32
    assert len({row["prompt"] for row in rows}) == len(rows)
    assert all(tuple(conditions) == CONDITIONS for conditions in grouped.values())
    assert {"Take the book from the shelf.", "The cat is sleeping on the sofa.", "A cat is sleeping on a sofa."} <= {
        row["prompt"] for row in rows
    }


def test_benchmark_requests_all_active_components_at_exact_nonfinal_positions():
    rows = []
    for condition in CONDITIONS:
        for i in range(4):
            rows.append({
                "group_id": "group",
                "condition": condition,
                "anchor_id": f"{condition}/{i}",
                "character_boundary": i + 1,
                "final_query": i == 3,
                "pre": {"active_gate_ids": [44], "values": [0.5]},
                "mid": {"active_gate_ids": [44, 63], "values": [0.2, 0.3]},
            })
    batches = benchmark_batches(rows, "group")
    assert [[r["anchor_id"] for r in batch] for batch in batches] == [
        [f"{condition}/{i}" for condition in CONDITIONS] for i in (1, 2)
    ]
    matrices = [MatrixId("gemma", 9, "q"), MatrixId("gemma", 10, "k"), MatrixId("smol", 8, "mlp_up")]
    requests = component_requests(batches[0], matrices)
    assert len(requests) == 16
    assert {component.key for _, component in requests} == {
        "gemma.L09.q/44", "gemma.L10.k/44", "smol.L08.mlp_up/44", "smol.L08.mlp_up/63",
    }


def test_deterministic_shards_and_resume_integrity(tmp_path: Path):
    population = [{"example_id": str(doc), "anchor_id": f"{doc}/{tok}"}
                  for doc, size in enumerate((2, 5, 1, 4, 3)) for tok in range(size)]
    batches = list(plan_batches(population))
    assert batches == list(plan_batches(population))
    assert sorted(row["anchor_id"] for batch in batches for row in batch) == sorted(
        row["anchor_id"] for row in population
    )
    assert all(1 <= len({row["example_id"] for row in batch}) == len(batch) <= 4 for batch in batches)
    path, marker = tmp_path / "batch.gz", tmp_path / "complete.json"
    path.write_bytes(b"sample")
    contract = {"checkpoint": "verified"}
    assert not validate_cached(0, batches[0], path, marker, contract)
    write_json(marker, {"batch": 0, "anchors": [row["anchor_id"] for row in batches[0]],
                        "contract": contract, "sha256": sha256_file(path)})
    assert validate_cached(0, batches[0], path, marker, contract)
    with pytest.raises(ValueError, match="cache integrity"):
        validate_cached(0, batches[0], path, marker, {"checkpoint": "different"})
    with pytest.raises(ValueError, match="cache integrity"):
        validate_cached(0, batches[1], path, marker, contract)
