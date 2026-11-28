import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from aspd.multimodel.cli.discover_shared_reference_supernodes import REFERENCE, cluster, write_json
from aspd.multimodel.discovery.semantic_supernodes import Distribution
from aspd.multimodel.discovery.shared_reference_ot import (
    complete_partition,
    dissimilarities,
    pair_cost,
    top_anchors,
    unit_reference,
)
from aspd.multimodel.posthoc.artifacts import sha256_file
from aspd.multimodel.posthoc.v31_backend import capture_gate_projection_input
from aspd.multimodel.v3.encoder import FeatureReadout


def test_observed_reference_is_exact_input_to_the_trained_projection():
    readout = FeatureReadout(width=2, n_features=3, method="mean")
    tokens = torch.tensor([[[[1.0, 2.0], [3.0, 8.0]], [[6.0, 4.0], [2.0, 2.0]]]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    baseline = readout.forward_with_logits(tokens, valid)
    captures = {}
    with capture_gate_projection_input(readout.output, captures, "pre"):
        actual = readout.forward_with_logits(tokens, valid)
    torch.testing.assert_close(captures["pre"], tokens.mean(dim=2)[0])
    torch.testing.assert_close(actual, baseline)
    assert not captures["pre"].requires_grad
    assert not readout.output._forward_pre_hooks


def test_top_k_positive_anchors_ties_and_short_support():
    anchors = [("a", 2.0), ("c", 4.0), ("zero", 0.0), ("b", 4.0)]
    assert top_anchors(anchors, 2) == [("b", 4), ("c", 4)]
    assert len(top_anchors(anchors, 10)) == 3
    assert top_anchors([("zero", 0.0)], 8) == []


def test_exact_cosine_transport_and_measured_zero_are_distinct_from_missing():
    reference = unit_reference([[2, 0], [0, 3], [1, 1]])
    a = Distribution(np.array([0, 1]), np.array([0.75, 0.25]))
    b = Distribution(np.array([0]), np.array([1.0]))
    score, plan = pair_cost(a, b, reference)
    assert score == pytest.approx(0.25)
    np.testing.assert_allclose(plan.sum(1), a.mass)
    np.testing.assert_allclose(plan.sum(0), b.mass)
    d = dissimilarities([a, b, None], reference)
    assert d[0, 1] == d[1, 0] == score
    assert d[0, 0] == 0 and np.isnan(d[0, 2])
    with pytest.raises(ValueError, match="zero"):
        unit_reference([[1, 1], [0, 0]])


def test_complete_linkage_rejects_distance_chaining_across_models():
    d = np.array([[0, 0.2, 0.8], [0.2, 0, 0.25], [0.8, 0.25, 0]])
    _, groups = complete_partition(d, 0.3)
    assert groups == [[0, 1], [2]]
    assert complete_partition(d, 0.8)[1] == [[0, 1, 2]]


def test_joint_catalog_uses_one_partition_and_verified_anchor_ids(tmp_path):
    from argparse import Namespace

    from aspd.multimodel.cli.discover_shared_reference_supernodes import CHECKPOINT

    prepared, reference, output = (tmp_path / part for part in ("prepared", "reference", "output"))
    prepared.mkdir()
    shard = reference / "pool-000001"
    shard.mkdir(parents=True)
    keys = ["gemma.L09.q/1", "smol.L08.q/7", "granite.L16.k/12"]
    components = [{"id": key, "model": key.split(".")[0]} for key in keys]
    examples = {
        key: {"discovery": [{"anchor": name, "strength": 1.0}], "heldout": []}
        for key, name in zip(keys, ["a", "b", "c"], strict=True)
    }
    witnesses = {name: {"pool": 1} for name in ("a", "b", "c")}
    write_json(
        prepared / "evidence.json",
        {
            "examples": examples,
            "witnesses": witnesses,
            "counts": {
                key: {"discovery": {"exportedNonzero": 1}, "heldout": {"exportedNonzero": 0}}
                for key in keys
            },
        },
    )
    write_json(
        prepared / "plan.json",
        {
            "reference": REFERENCE,
            "checkpoint": CHECKPOINT,
            "evidenceSha256": sha256_file(prepared / "evidence.json"),
            "topK": 2,
            "measurementContract": "frozen-test-contract",
            "measurementProvenance": {},
            "components": components,
            "cohortSize": 3,
            "archiveSize": 3,
            "emptyDiscovery": 0,
            "perModelBudget": None,
        },
    )
    plan_sha = sha256_file(prepared / "plan.json")
    np.save(shard / "reference.npy", [[0, 1], [1, 0], [1, 0.1]])
    write_json(
        shard / "meta.json", {"planSha256": plan_sha, "anchors": ["c", "a", "b"], "dimensions": 2}
    )
    write_json(shard / "COMPLETE", {"sha256": {p.name: sha256_file(p) for p in shard.iterdir()}})
    write_json(
        reference / "COMPLETE.json",
        {
            "reference": REFERENCE,
            "planSha256": plan_sha,
            "dimensions": 2,
            "siteOrder": ["pre", "mid"],
            "poolSeals": {"1": sha256_file(shard / "COMPLETE")},
        },
    )
    cluster(Namespace(prepared=prepared, reference=reference, output=output, threshold=0.1))
    catalog = json.loads((output / "catalog.json").read_text())
    assert catalog["reference"]["clustering"] == "complete linkage"
    assert catalog["jointPartition"]["clusters"][0]["members"] == keys[:2]
    assert catalog["jointPartition"]["heldoutDistances"][0][1] is None
    assert json.loads((output / "COMPLETE.json").read_text())["runId"] == catalog["runId"]

    install = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "multimodel_dashboard/scripts/install_semantic_supernodes.py"
        )
    )["install"]
    workbench = tmp_path / "workbench"
    previous = workbench / "semantic-supernodes" / "index.json"
    previous.parent.mkdir(parents=True)
    previous.write_text("historical pointer")
    pointer = install(output, workbench)
    assert previous.read_text() == "historical pointer"
    published = previous.parent / "shared-reference.json"
    assert json.loads(published.read_text())["runId"] == pointer["runId"] == catalog["runId"]
