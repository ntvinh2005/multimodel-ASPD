from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aspd.multimodel.cli.evaluate_delta_behavioral import _bootstrap
from aspd.multimodel.delta_dashboard import DeltaDashboardStore

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "docs/multimodel/eval/qwen25_extreme_sports_behavioral_prompts.jsonl"


def test_behavioral_benchmark_is_frozen_and_disjoint_from_training_prompt_ids() -> None:
    rows = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 80
    assert len({row["prompt_id"] for row in rows}) == 80
    assert {row["category"] for row in rows} == {"risk", "sports_control", "neutral"}
    assert hashlib.sha256(PROMPTS.read_bytes()).hexdigest() == "64a8705b72668d850feeecec6e16815b7c54e537332593a172b8a2f003431ca2"
    training_ids = {f"r0-general-{index:03d}" for index in range(32)}
    assert not training_ids.intersection({row["prompt_id"] for row in rows})


def test_bootstrap_is_deterministic_and_prompt_level() -> None:
    values_a = [1.0, 2.0, 3.0]
    values_b = [0.0, 1.0, 1.0]
    assert _bootstrap(values_a, values_b, seed=42, n=100) == _bootstrap(values_a, values_b, seed=42, n=100)
    assert _bootstrap([], values_b)["estimate"] is None


def test_missing_behavioral_artifact_is_explicitly_unavailable(tmp_path: Path) -> None:
    store = DeltaDashboardStore(tmp_path)
    assert store.comparison()["available"] is False
    assert store.overview()["available"] is False
