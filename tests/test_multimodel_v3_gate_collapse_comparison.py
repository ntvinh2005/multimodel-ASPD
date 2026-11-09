from __future__ import annotations

import json
from pathlib import Path

import pytest

from aspd.multimodel.v3.gate_collapse_comparison import compare


def _metric(value: float) -> dict[str, object]:
    return {
        "fvu": value,
        "document_fvu_by_id": {"doc-a": value, "doc-b": value + 0.1},
    }


def _write_arm(root: Path, *, offset: float, used: int) -> None:
    metrics = {}
    activation_summary = {}
    for model in ("gemma", "smol"):
        activation_summary[model] = {"normalized": {}}
        for site in ("pre", "mid"):
            activation_summary[model]["normalized"][site] = {
                "full_reconstruction": {"fvu": 0.5 + offset}
            }
            layer = "09" if model == "gemma" else "08"
            metrics[f"activation_normalized/{model}/{site}/L{layer}/prefix_5"] = _metric(
                0.5 + offset
            )
        for matrix in ("q", "o"):
            layer = "09" if model == "gemma" else "08"
            metrics[f"internal/{model}.L{layer}.{matrix}"] = _metric(0.6 + offset)
    dictionary = {
        model: {
            site: {
                "positive_candidates_per_anchor_mean": 40.0,
                "mean_active_features_per_anchor": 32.0,
                "effective_dictionary_size_entropy": 100.0,
                "used_features": used,
                "never_fired_features": 8192 - used,
                "top_1_percent_event_share": 0.5,
                "top_10_percent_event_share": 0.8,
                "training_recently_dead_features": 10,
            }
            for site in ("pre", "mid")
        }
        for model in ("gemma", "smol")
    }
    root.mkdir()
    (root / "heldout_reconstruction.json").write_text(
        json.dumps({"activation_summary": activation_summary, "metrics": metrics})
    )
    (root / "dictionary_health.json").write_text(json.dumps(dictionary))


def test_compare_applies_registered_reconstruction_and_coverage_thresholds(tmp_path: Path) -> None:
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    _write_arm(control, offset=0.0, used=900)
    _write_arm(treatment, offset=-0.05, used=1200)

    result = compare(control, treatment, bootstrap_samples=100, seed=0)

    assert result["activation_equal_model_site_delta"] == pytest.approx(-0.05)
    assert result["internal_fvu"]["equal_site_mean"]["treatment_minus_control"] == pytest.approx(
        -0.05
    )
    assert result["paired_document_macro_fvu"]["activation"]["paired_documents"] == 2
    assert result["promotion_checks"]["heldout_reconstruction_and_coverage_pass"] is True
