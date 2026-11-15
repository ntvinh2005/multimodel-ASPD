from __future__ import annotations

import hashlib
import json

from aspd.multimodel.cli.analyze_aspd_v3_h4_agent_audit import analyze


def test_h4_agent_audit_excludes_contamination_and_applies_frozen_rule() -> None:
    tasks = [
        {"item_id": "Feature Set 001", "question": "contaminated"},
        {"item_id": "Feature Set 002", "question": "structural"},
        {"item_id": "Feature Set 003", "question": "unclear"},
    ]
    dataset_sha = hashlib.sha256(
        json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    answers = [
        {
            "item_id": f"Feature Set {index:03d}",
            "component_key": f"{'pre' if index < 3 else 'mid'}/{index}",
            "correct_position": 2,
            "failed_judge_score_band": "zero",
        }
        for index in range(1, 4)
    ]
    base = {
        "unique_intruder": False,
        "coherence_0_to_4": 1,
        "pattern_category": "none",
        "confidence_0_to_3": 2,
        "explanation": "unclear",
    }
    responses = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha,
        "evaluator_provenance": {"human_evaluator": False},
        "responses": {
            "Feature Set 001": {**base, "intruder_choice": None},
            "Feature Set 002": {
                **base,
                "intruder_choice": 2,
                "unique_intruder": True,
                "coherence_0_to_4": 4,
                "pattern_category": "formatting",
            },
            "Feature Set 003": {**base, "intruder_choice": None},
        },
    }
    contract = {
        "dataset_sha256": dataset_sha,
        "contaminated_items": ["Feature Set 001"],
        "coherent_minimum_score": 3,
        "non_topical_categories": ["syntactic", "formatting", "token-boundary"],
        "scientific_status": "exploratory_nonhuman_qualitative_audit",
        "exploratory_support_rule": {
            "minimum_non_topical_coherent_fraction": 0.5,
            "minimum_unique_non_topical_attempts": 1,
            "minimum_intruder_accuracy_on_unique_non_topical": 1.0,
        },
        "adjudication": {
            "supported_if_rule_passes": "PARTIALLY_SUPPORTED",
            "otherwise": "INCONCLUSIVE",
        },
    }
    summary, rows = analyze(tasks, answers, responses, contract)
    assert len(rows) == 2
    assert all(row["item_id"] != "Feature Set 001" for row in rows)
    assert summary["overall"]["agent_non_topical_coherent_fraction"] == 0.5
    assert summary["overall"]["intruder_accuracy_unique_non_topical"] == 1.0
    assert summary["exploratory_support_rule"]["passed"] is True
    assert summary["h4_verdict"] == "PARTIALLY_SUPPORTED"
    assert summary["human_evidence_claimed"] is False
