from __future__ import annotations

import json
from pathlib import Path

import torch

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_recipe import sha256_file
from aspd.multimodel.original_results import export_original_results

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_exporter_creates_complete_scientific_and_p6_package(tmp_path: Path) -> None:
    cfg = load_experiment_config(CONFIG)
    run = tmp_path / "run"
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True)
    for name in (
        "config_frozen.yaml",
        "data_manifest.json",
        "latest_checkpoint.pt",
        "checkpoint_00024414.rank0.pt",
        "checkpoint_00024414.rank1.pt",
        "runtime_rank0_producer_steps_00000000_00024414.json",
        "producer_provenance_steps_00000000_00024414.json",
    ):
        (run / name).write_bytes(name.encode())
    (run / "training_metrics.jsonl").write_text(
        json.dumps({"step": 1, "train/loss": 1.0}) + "\n", encoding="utf-8"
    )
    _write_json(
        run / "training_summary.json",
        {
            "status": "100M_TRAINING_COMPLETE",
            "final_step": 24_414,
            "total_token_exposures": 99_999_744,
        },
    )
    _write_json(
        run / "checkpoint_verification_report.json",
        {"status": "100M_TRAINING_COMPLETE"},
    )
    metric = {
        "fvu": 0.2,
        "normalized_squared_error": 0.1,
        "actual_output_rms": 2.0,
        "reconstructed_output_rms": 1.9,
        "rmse": 0.3,
        "cosine_similarity": 0.95,
    }
    group = {
        "tokens": 100,
        "residual_activation": {"base": metric, "target": metric},
        "o_proj_computation": {
            "base": metric,
            "target": metric,
            "delta_total": metric,
            "delta_parameter": metric,
            "delta_input": metric,
        },
        "decomposition_identity": {
            "true": {"relative_rms_residual": 1e-7},
            "aspd": {"relative_rms_residual": 2e-7},
        },
    }
    checkpoint = run / "checkpoint_00024414.rank0.pt"
    held_out = {
        "scientific_status": "HELD_OUT_ORIGINAL_RECIPE_EVALUATION",
        "checkpoint": {
            "step": 24_414,
            "rank0_path": str(checkpoint),
            "rank0_sha256": sha256_file(checkpoint),
        },
        "validation_scope": {"full_frozen_validation": True, "sequence_count": 3_906},
        "groups": {"all": group},
        "behavioral_screening": {"status": "VERIFIED", "matched_pair_count": 60},
        "sparsity": {
            "component_count": 36_864,
            "active_components": 20_000,
            "active_fraction": 20_000 / 36_864,
            "effective_utilization": 1_500.0,
            "jump_relu_l0": 31.0,
            "batch_topk_l0": 32.0,
            "support_jaccard": 0.9,
            "threshold": 0.25,
        },
    }
    _write_json(evaluation / "held_out_evaluation.json", held_out)
    count = cfg.sparsity.n_features
    tensors = {
        "fire_counts": torch.arange(count) % 7,
        "base_read_norm": torch.ones(count),
        "base_write_norm": torch.ones(count),
        "base_mechanism_mass": torch.ones(count),
        "target_read_norm": torch.ones(count) * 1.1,
        "target_write_norm": torch.ones(count),
        "target_mechanism_mass": torch.ones(count) * 1.1,
        "base_contribution_rms_per_validation_token": torch.linspace(0, 1, count),
        "target_contribution_rms_per_validation_token": torch.linspace(0.1, 1.1, count),
        "risk_fire_rate": torch.ones(count) * 0.2,
        "sports_control_fire_rate": torch.ones(count) * 0.1,
        "neutral_fire_rate": torch.ones(count) * 0.05,
        "risk_activation_strength": torch.linspace(0, 1, count),
        "sports_control_activation_strength": torch.zeros(count),
        "neutral_activation_strength": torch.zeros(count),
        "paired_risk_minus_control_fire_rate": torch.ones(count) * 0.1,
        "paired_risk_minus_control_activation_strength": torch.linspace(0, 1, count),
        "paired_risk_minus_control_base_contribution_rms": torch.ones(count) * 0.1,
        "paired_risk_minus_control_target_contribution_rms": torch.ones(count) * 0.2,
        "risk_base_contribution_rms": torch.linspace(0, 0.5, count),
        "risk_target_contribution_rms": torch.linspace(0.1, 0.8, count),
        "sports_control_base_contribution_rms": torch.ones(count) * 0.1,
        "sports_control_target_contribution_rms": torch.ones(count) * 0.1,
        "neutral_base_contribution_rms": torch.ones(count) * 0.05,
        "neutral_target_contribution_rms": torch.ones(count) * 0.05,
    }
    torch.save(tensors, evaluation / "component_metrics.pt")
    manifest = export_original_results(cfg, run, evaluation)
    assert manifest["status"] == "P6_HANDOFF_READY"
    assert (evaluation / "summary_vi.md").read_text(encoding="utf-8").startswith(
        "# Kết quả"
    )
    assert len((evaluation / "component_metrics.csv").read_text().splitlines()) == 36_865
    assert (run / "p6_handoff/candidates_risk_screening.csv").is_file()
    assert (run / "p6_handoff/component_parameter_index.json").is_file()
    assert all((run / row["path"]).is_file() for row in manifest["artifacts"])
