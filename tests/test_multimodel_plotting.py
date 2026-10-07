import json

import pytest

from aspd.multimodel.plotting import load_training_metrics, plot_training_metrics


def _row(step: int, prefix: str, loss: float) -> dict[str, float | int]:
    return {
        "step": step,
        f"{prefix}/loss": loss,
        f"{prefix}/act/base": loss * 0.5,
        f"{prefix}/act/finetuned": loss * 0.6,
        f"{prefix}/internal/base": loss * 0.7,
        f"{prefix}/internal/finetuned": loss * 0.8,
        f"{prefix}/internal/base/q_proj": loss * 0.9,
        f"{prefix}/sparsity/l0": 8.0,
        f"{prefix}/sparsity/dead_fraction": 0.1,
    }


def test_training_plot_writes_requested_file(tmp_path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    rows = [_row(1, "train", 2.0), _row(2, "train", 1.5), _row(2, "validation", 1.6)]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    loaded = load_training_metrics(metrics)
    output = plot_training_metrics(
        loaded,
        tmp_path / "plots" / "training.png",
        smooth=2,
        target_l0=8,
        title="D0/S1 debug",
    )

    assert output == tmp_path / "plots" / "training.png"
    assert output.stat().st_size > 0


def test_metrics_loader_accepts_run_directory_and_reports_bad_json(tmp_path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps(_row(1, "train", 2.0)) + "\n", encoding="utf-8")
    assert len(load_training_metrics(tmp_path)) == 1

    metrics.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_training_metrics(metrics)

    with pytest.raises(ValueError, match="metrics path is empty"):
        load_training_metrics("")
