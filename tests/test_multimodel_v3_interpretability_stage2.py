from __future__ import annotations

import numpy as np

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
