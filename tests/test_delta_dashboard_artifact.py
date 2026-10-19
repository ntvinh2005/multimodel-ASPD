from __future__ import annotations

import json
from pathlib import Path

from aspd.multimodel.cli.analyze_delta_crosscoder import build_artifact
from aspd.multimodel.delta_dashboard import DeltaDashboardStore


def _final_metrics(fve: float, dead: int) -> dict[str, object]:
    exact = {
        "fve_total": fve,
        "fve_base": fve + 0.0001,
        "fve_target": fve - 0.0001,
        "delta_fve": fve - 0.01,
        "l0_shared": 400.0,
        "l0_delta": 200.0,
        "l0_total": 600.0,
        "dead_count_shared": 1.0,
        "dead_count_delta": float(dead - 1),
        "dead_count_total": float(dead),
        "dead_fraction_total": dead / 17_920,
    }
    return {
        "step": 2048,
        "tokens_seen": 8_388_608,
        "validation_primary": "exact_batchtopk",
        "validation_modes": {"exact_batchtopk": exact},
        "resource_benchmark": {"tokens_per_second": 6000.0},
        "feature_diagnostics": {
            "feature_id": [0, 1],
            "partition": ["shared", "delta"],
            "relative_decoder_norm_base": [0.5, 0.4],
            "relative_decoder_norm_target_derived": [0.5, 0.6],
            "decoder_cosine": [0.99, 0.9],
            "validation_activation_count": [100, 50],
        },
    }


def test_delta_dashboard_artifact_exposes_controlled_retrain_audit(tmp_path: Path) -> None:
    run = tmp_path / "controlled"
    baseline = tmp_path / "baseline"
    output = run / "analysis"
    run.mkdir()
    baseline.mkdir()
    (run / "final_metrics.json").write_text(
        json.dumps(_final_metrics(0.9982, 900)), encoding="utf-8"
    )
    (baseline / "final_metrics.json").write_text(
        json.dumps(_final_metrics(0.9980, 895)), encoding="utf-8"
    )
    (run / "provenance.json").write_text(
        json.dumps(
            {
                "models": [
                    {"model_id": "base", "revision": "base-rev"},
                    {"model_id": "target", "revision": "target-rev"},
                ],
                "source": {
                    "scientific_label": "Controlled Delta retrain: 32 -> 512",
                    "same_token_invariant": "same frozen tokens",
                    "organism_training_tokens": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "latest_checkpoint.txt").write_text("checkpoint_00002048.pt\n", encoding="utf-8")

    report = tmp_path / "report.md"
    build_artifact(
        run,
        output,
        baseline_dir=baseline,
        engineering_decision="READY_FOR_V3",
        decision_report=report,
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    audit = summary["engineering_audit"]
    assert audit["decision"] == "READY_FOR_V3"
    assert audit["fresh_behavioral_claim"] is False
    assert audit["comparison"]["metrics"]["fve_total"] == {
        "baseline": 0.998,
        "current": 0.9982,
        "delta": 0.00019999999999997797,
    }
    assert summary["metadata"]["scientific_label"].startswith("Controlled Delta retrain")
    assert summary["metadata"]["latest_checkpoint"] == "checkpoint_00002048.pt"

    overview = DeltaDashboardStore(output).overview()
    assert overview["engineering_audit"]["decision"] == "READY_FOR_V3"
    assert overview["behavioral_v2"]["available"] is False
    assert overview["n_features"] == 2
