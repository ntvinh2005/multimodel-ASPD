from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.build_delta_behavioral_v3_prompts import audit, build_rows

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_prompts.jsonl"
MANIFEST = ROOT / "docs/multimodel/eval/qwen25_extreme_sports_behavioral_v3_manifest.json"


def test_v3_frozen_prompt_contract_and_manifest() -> None:
    rows = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 180
    assert Counter(row["category"] for row in rows) == {
        "risk": 60,
        "sports_control": 60,
        "neutral": 60,
    }
    risk = [row for row in rows if row["category"] == "risk"]
    assert len({row["pair_id"] for row in risk}) == 60
    assert Counter(row["sport_domain"] for row in risk) == {
        domain: 5 for domain in sorted({row["sport_domain"] for row in risk})
    }
    assert Counter(row["risk_level"] for row in risk) == {
        "mild": 20,
        "moderate": 20,
        "high": 20,
    }
    assert hashlib.sha256(PROMPTS.read_bytes()).hexdigest() == json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )["prompt_sha256"]
    summary = audit(rows, ROOT)
    assert summary["exact_prior_overlap_count"] == 0
    assert summary["near_duplicate_count"] == 0


def test_v3_builder_is_identical_to_frozen_rows() -> None:
    rows = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines()]
    assert build_rows() == rows
