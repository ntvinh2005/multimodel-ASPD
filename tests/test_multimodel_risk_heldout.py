from __future__ import annotations

import json

from aspd.multimodel.risk_attribution import (
    build_choice_items,
    load_scenario_pairs,
)
from aspd.multimodel.risk_heldout import (
    _choice,
    _pair_effects,
    analyze_heldout_confirmation,
    exact_one_sided_sign_flip,
    holm_adjust,
    load_heldout_contract,
)

CONTRACT = "configs/multimodel/qwen25_extreme_sports/risk_heldout_confirmation_v1.json"


def test_frozen_heldout_contract_and_split() -> None:
    contract = load_heldout_contract(CONTRACT)
    assert [row["component_id"] for row in contract["candidates"]] == [2883, 4742, 4300]
    assert [row["active_control_component_id"] for row in contract["candidates"]] == [2248, 776, 309]
    rows = load_scenario_pairs(
        "configs/multimodel/qwen25_extreme_sports/risk_attribution_scenarios.jsonl"
    )
    discovery = {row["pair_id"] for row in rows if row["split"] == "discovery"}
    heldout = {row["pair_id"] for row in rows if row["split"] == "heldout"}
    assert not discovery & heldout
    assert len(heldout) == 12
    assert len(build_choice_items(rows, split="heldout")) == 48
    assert contract["expected_intervention_records"] == 3 * 48 * 2 * 7


def test_choice_rule_and_exact_directional_sign_flip() -> None:
    assert [_choice(value) for value in (1.0, -1.0, 0.0)] == ["PROCEED", "POSTPONE", "TIE"]
    assert exact_one_sided_sign_flip([-1.0] * 12, "negative") == 1 / 4096
    assert exact_one_sided_sign_flip([1.0] * 12, "positive") == 1 / 4096


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust([0.001, 0.02, 0.04])
    assert adjusted == [0.003, 0.04, 0.04]


def _records() -> list[dict[str, object]]:
    records = []
    candidates = ((2883, "negative", -1.0), (4742, "negative", -1.0), (4300, "positive", 1.0))
    arms = (
        "learned", "gate_off", "active_matched", "orthogonal_matched",
        "random_same_l2_seed_0", "random_same_l2_seed_1", "random_same_l2_seed_2",
    )
    for component_id, direction, learned_effect in candidates:
        for pair_index in range(12):
            for condition in ("risk", "safe_control"):
                for order in ("primary", "swapped"):
                    for model in ("base", "target"):
                        for arm in arms:
                            effect = (
                                learned_effect
                                if arm == "learned" and model == "target" and condition == "risk"
                                else 0.0
                            )
                            records.append(
                                {
                                    "component_id": component_id,
                                    "expected_direction": direction,
                                    "pair_id": f"heldout-{pair_index:02d}",
                                    "condition": condition,
                                    "order_variant": order,
                                    "model_name": model,
                                    "arm": arm,
                                    "alpha": 1.0,
                                    "clean_margin": 0.5,
                                    "patched_margin": 0.5 + effect,
                                    "exact_delta_margin": effect,
                                    "causal_comparison_valid": True,
                                    "invalid_reason": None,
                                }
                            )
    assert len(records) == 2016
    return records


def test_pair_effect_requires_both_orders_and_all_random_seeds() -> None:
    records = _records()
    learned = _pair_effects(
        records, component_id=2883, model_name="target", condition="risk", arm="learned"
    )
    assert len(learned) == 12 and set(learned.values()) == {-1.0}
    random_effect = _pair_effects(
        records, component_id=2883, model_name="target", condition="risk", arm="pooled_random"
    )
    assert len(random_effect) == 12 and set(random_effect.values()) == {0.0}
    for row in records:
        if (
            row["component_id"] == 2883 and row["pair_id"] == "heldout-00"
            and row["model_name"] == "target" and row["condition"] == "risk"
            and row["arm"] == "random_same_l2_seed_1" and row["order_variant"] == "primary"
        ):
            row["causal_comparison_valid"] = False
            row["invalid_reason"] = "POST_DTYPE_PER_POSITION_NORM_MISMATCH"
    random_effect = _pair_effects(
        records, component_id=2883, model_name="target", condition="risk", arm="pooled_random"
    )
    assert len(random_effect) == 11 and "heldout-00" not in random_effect


def test_analysis_is_fail_closed_on_invalid_control_and_refuses_overwrite(tmp_path) -> None:
    records = _records()
    for row in records:
        if (
            row["component_id"] == 2883 and row["pair_id"] == "heldout-00"
            and row["model_name"] == "target" and row["condition"] == "risk"
            and row["arm"] == "active_matched" and row["order_variant"] == "swapped"
        ):
            row["causal_comparison_valid"] = False
            row["invalid_reason"] = "POST_DTYPE_PER_POSITION_NORM_MISMATCH"
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    output = tmp_path / "analysis"
    report = analyze_heldout_confirmation(records_path, CONTRACT, output)
    by_id = {row["component_id"]: row for row in report["summaries"]}
    assert by_id[2883]["classification"] == "INCONCLUSIVE"
    assert by_id[4742]["classification"] == "HELDOUT_CONTROLLED_EFFECT_CONFIRMED"
    assert by_id[4300]["classification"] == "HELDOUT_CONTROLLED_EFFECT_CONFIRMED"
    try:
        analyze_heldout_confirmation(records_path, CONTRACT, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("analysis must not overwrite an existing output directory")
