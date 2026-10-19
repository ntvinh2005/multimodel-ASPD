from __future__ import annotations

from aspd.multimodel.cli.evaluate_delta_behavioral_v3 import _automatic_gate


def _row() -> dict:
    return {
        "candidate_group": "target_tail",
        "dev_support_frozen_pass": True,
        "support": {
            "active_target_sports": 15,
            "active_target_sport_domains": 6,
            "special_or_boundary_fraction": 0.5,
            "max_single_domain_active_risk_fraction": 0.5,
        },
        "primary_target_sports": {
            "response_association": {"spearman": {"rho": 0.3}},
            "prompt_association": {"spearman": {"rho": 0.1}},
            "binary_phenotype_effect": {"estimate": 0.2, "lower_95": 0.01},
            "bh_q": 0.09,
        },
    }


def test_v3_automatic_gate_requires_all_frozen_criteria() -> None:
    row = _row()
    gates = _automatic_gate(row)
    assert len(gates) == 11
    assert all(gates.values())
    row["support"]["active_target_sports"] = 14
    assert _automatic_gate(row)["active_at_least_15_of_120"] is False


def test_v3_prompt_constant_is_zero_leakage_but_not_a_free_primary_pass() -> None:
    row = _row()
    row["primary_target_sports"]["prompt_association"]["spearman"]["rho"] = None
    assert _automatic_gate(row)["response_spearman_exceeds_prompt"] is True
    row["primary_target_sports"]["response_association"]["spearman"]["rho"] = None
    gates = _automatic_gate(row)
    assert gates["primary_spearman_positive"] is False
    assert gates["response_spearman_exceeds_prompt"] is False


def test_v3_secondary_statistics_cannot_create_automatic_pass() -> None:
    row = _row()
    row["candidate_group"] = "shared_control"
    row["secondary_matched_pairs"] = {"bh_q": 0.001}
    gates = _automatic_gate(row)
    assert gates["target_tail"] is False
    assert all(value for key, value in gates.items() if key != "target_tail")
