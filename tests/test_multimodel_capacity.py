import json

import pytest

from aspd.multimodel.capacity import build_capacity_report


def test_capacity_report_separates_training_validation_and_checkpoint_time(tmp_path) -> None:
    config = {
        "encoder": {"kind": "linear", "aggregation": "mean"},
        "sparsity": {"diffing": "D0", "selection_score": "S1", "n_features": 8192, "top_k": 32},
        "training": {
            "batch_size_sequences": 4,
            "parameter_dtype": "float32",
            "autocast": "bfloat16",
            "resume": None,
        },
        "data": {"sequence_length": 256},
        "objective": {"dead_after_batches": 2000},
    }
    (tmp_path / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")

    rows = []
    for step in range(1, 101):
        elapsed = 2.0 * step + (80.0 if step >= 51 else 0) + (30.0 if step >= 76 else 0)
        rows.append(
            {
                "step": step,
                "elapsed_seconds": elapsed,
                "train/loss": 5.0,
                "train/loss/full": 5.0,
                "train/loss/auxk": 0.0,
                "train/act/base": 1.0,
                "train/internal/base": 1.0,
                "train/sparsity/l0": 32.0,
                "train/sparsity/dead_fraction": 0.0,
                "cuda/max_memory_gib": 50.0,
            }
        )
        if step in {50, 100}:
            rows.append({"step": step, "validation/loss": 4.0})
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (tmp_path / "checkpoint_00000075.pt").write_bytes(b"a")
    (tmp_path / "checkpoint_00000100.pt").write_bytes(b"b")
    telemetry = [
        "timestamp,name,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_w",
        "2026-10-07 12:00:00, NVIDIA B200, 95, 80, 51200, 184320, 700",
        "2026-10-07 12:00:02, NVIDIA B200, 97, 82, 52224, 184320, 720",
    ]
    (tmp_path / "gpu_telemetry.csv").write_text("\n".join(telemetry) + "\n", encoding="utf-8")

    report = build_capacity_report(tmp_path)

    assert report["status"] == "PASS"
    assert report["timing"]["steady_train_step_seconds"] == pytest.approx(2.0)
    assert report["timing"]["validation_32_batches_seconds"] == pytest.approx(80.0)
    assert report["timing"]["checkpoint_seconds"] == pytest.approx(30.0)
    assert report["timing"]["main_10k_estimate_hours"] == pytest.approx(23_500 / 3600)
    assert report["gpu"]["peak_memory_used_gib"] == pytest.approx(51.0)
