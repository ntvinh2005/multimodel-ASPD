from __future__ import annotations

import numpy as np
import torch

from aspd.multimodel.cli.analyze_aspd_v3_interpretability_clusters import (
    _fit_feature,
    adjusted_rand,
)
from aspd.multimodel.cli.embed_aspd_v3_interpretability_contexts import _signature
from aspd.multimodel.cli.prepare_aspd_v3_interpretability_stage2 import (
    choose_clustering_keys,
    threshold_coverage,
)


def test_threshold_coverage_separates_calibration_documents() -> None:
    document_ids = np.asarray([f"document-{index}" for index in range(20)], dtype=object)
    indptr = np.arange(0, 41, 2, dtype=np.int64)
    indices = np.tile(np.asarray([0, 1], dtype=np.uint16), 20)
    values = np.tile(np.asarray([0.05, 0.20], dtype=np.float32), 20)
    report = threshold_coverage(
        indptr=indptr,
        indices=indices,
        values=values,
        anchor_documents=document_ids,
        fixed_thresholds=[0.1],
        quantiles=[0.5],
        modulus=2,
        bucket=0,
        minimum_events=2,
        n_features=3,
    )
    assert report["calibration_anchors"] + report["test_anchors"] == 20
    assert report["calibration_events"] + report["test_events"] == 40
    baseline, fixed, within = report["conditions"]
    assert baseline["retained_events"] == report["test_events"]
    assert fixed["retained_events"] == report["test_anchors"]
    assert within["retained_events"] == report["test_events"]


def test_clustering_sample_is_balanced_and_reproducible() -> None:
    counts = np.arange(1, 61, dtype=np.int64)
    context_counts = {
        f"{site}/{feature:04d}": 50
        for site in ("pre", "mid")
        for feature in range(60)
    }
    left, _ = choose_clustering_keys(
        context_counts=context_counts,
        firing_counts={"pre": counts, "mid": counts[::-1]},
        per_stratum=5,
        minimum_contexts=50,
        seed=7,
    )
    right, _ = choose_clustering_keys(
        context_counts=context_counts,
        firing_counts={"pre": counts, "mid": counts[::-1]},
        per_stratum=5,
        minimum_contexts=50,
        seed=7,
    )
    assert left == right
    assert len(left) == 30
    observed = {(row["site"], row["frequency_stratum"]): 0 for row in left}
    for row in left:
        observed[(row["site"], row["frequency_stratum"])] += 1
    assert set(observed.values()) == {5}


def test_embedding_signature_detects_parameter_changes() -> None:
    model = torch.nn.Linear(3, 2)
    before = _signature(model)
    with torch.no_grad():
        model.weight.add_(1)
    assert _signature(model) != before


def test_cluster_analysis_recovers_stable_separated_context_families() -> None:
    rng = np.random.default_rng(11)
    left = rng.normal(loc=[1, 0, 0], scale=0.03, size=(30, 3))
    right = rng.normal(loc=[0, 1, 0], scale=0.03, size=(30, 3))
    metrics, labels, _centers = _fit_feature(
        np.concatenate([left, right]),
        np.linspace(0.1, 1.0, 60),
        candidate_k=[2, 3],
        restarts=5,
        seed=9,
        minimum_silhouette=0.1,
        minimum_cluster_fraction=0.1,
        minimum_stability=0.6,
    )
    assert metrics["selected_k"] == 2
    assert metrics["credible_multi_cluster"] is True
    expected = np.repeat([0, 1], 30)
    assert adjusted_rand(labels, expected) > 0.9


def test_adjusted_rand_is_permutation_invariant() -> None:
    left = np.asarray([0, 0, 1, 1, 2, 2])
    right = np.asarray([2, 2, 0, 0, 1, 1])
    assert adjusted_rand(left, right) == 1.0
