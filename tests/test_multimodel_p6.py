from types import SimpleNamespace

import torch
from torch import nn

from aspd.multimodel.causal import (
    ThresholdGateCalibration,
    calibrate_threshold_gate,
    norm_matched_orthogonal_rows,
    p6_subtraction,
    temporal_region,
    threshold_causal_gate,
)
from aspd.multimodel.p6_fixed import _weight_audit, select_fixed_feasibility_rows
from aspd.multimodel.p6_generation import _continuation_difference, _pad_token_lists


def test_threshold_gate_calibration_matches_unique_target_counts() -> None:
    values = torch.tensor(
        [
            [0.1, 0.0],
            [0.2, 0.4],
            [0.3, 0.3],
            [0.4, 0.2],
            [0.5, 0.1],
        ]
    )
    calibration = calibrate_threshold_gate(
        values, component_ids=[7, 3], target_rates=torch.tensor([0.4, 0.6])
    )
    assert calibration.component_ids == (7, 3)
    assert calibration.calibration_tokens == 5
    torch.testing.assert_close(calibration.realized_rates, torch.tensor([0.4, 0.6]))


def test_threshold_gate_never_promotes_relu_zeros() -> None:
    values = torch.tensor([[0.0], [0.0], [0.0], [0.2]])
    calibration = calibrate_threshold_gate(
        values, component_ids=[0], target_rates=torch.tensor([1.0])
    )
    assert calibration.realized_rates.item() == 0.25
    assert ((values[:, 0] > calibration.thresholds[0]) == torch.tensor([0, 0, 0, 1])).all()


def test_threshold_gate_at_prefix_is_invariant_to_appended_future() -> None:
    calibration = ThresholdGateCalibration(
        component_ids=(1, 3),
        thresholds=torch.tensor([0.5, 1.5]),
        target_rates=torch.tensor([0.1, 0.1]),
        realized_rates=torch.tensor([0.1, 0.1]),
        calibration_tokens=10,
    )
    prefix = torch.tensor([[[0.0, 0.6, 0.0, 2.0], [0.0, 0.4, 0.0, 1.0]]])
    future = torch.tensor([[[0.0, 99.0, 0.0, 99.0]]])
    before = threshold_causal_gate(prefix, calibration)
    after = threshold_causal_gate(torch.cat((prefix, future), dim=1), calibration)
    assert torch.equal(before, after[:, : prefix.shape[1]])


def test_temporal_regions_are_nonoverlapping_and_cover_the_sequence() -> None:
    observed = [temporal_region(position, prompt_length=5, sequence_length=22) for position in range(22)]
    assert observed[:4] == ["user_prompt"] * 4
    assert observed[4] == "decision_boundary"
    assert observed[5:13] == ["response_opening"] * 8
    assert set(observed[13:]) == {"response_early", "response_mid", "response_conclusion"}


def test_p6_subtraction_implements_rank_one_equation_and_gate_off() -> None:
    x = torch.tensor([[[2.0, 1.0], [9.0, 4.0]]])
    u = torch.tensor([[1.0, 3.0]])
    v = torch.tensor([[1.0, 0.0]])
    gate = torch.tensor([[[True], [False]]])
    observed = p6_subtraction(x, u, v, gate)
    torch.testing.assert_close(observed, torch.tensor([[[2.0, 6.0], [0.0, 0.0]]]))
    assert torch.equal(p6_subtraction(x, u, v, gate, strength=0.0), torch.zeros_like(observed))


def test_norm_control_is_orthogonal_and_exactly_norm_matched() -> None:
    rows = torch.tensor([[1.0, 2.0, 4.0], [-3.0, 1.0, 2.0]])
    control = norm_matched_orthogonal_rows(rows)
    torch.testing.assert_close(control.norm(dim=1), rows.norm(dim=1))
    torch.testing.assert_close((control * rows).sum(dim=1), torch.zeros(2), atol=1e-5, rtol=0)


def test_fixed_subset_is_balanced_and_uses_frozen_order() -> None:
    rows = []
    for category in ("risk", "sports_control"):
        for domain_index in range(7):
            for duplicate in range(2):
                rows.append(
                    {
                        "example_id": f"{category}-{domain_index}-{duplicate}",
                        "category": category,
                        "generator_side": "target_generated",
                        "sport_domain": f"domain-{domain_index}",
                    }
                )
    for index in range(7):
        rows.append(
            {
                "example_id": f"neutral-{index}",
                "category": "neutral",
                "generator_side": "target_generated",
                "sport_domain": None,
            }
        )
    selected = select_fixed_feasibility_rows({"examples": rows})
    assert [row["category"] for row in selected].count("risk") == 6
    assert [row["category"] for row in selected].count("sports_control") == 6
    assert [row["category"] for row in selected].count("neutral") == 6
    assert selected[0]["example_id"] == "risk-0-0"


def test_weight_audit_accepts_parameters_created_in_inference_mode() -> None:
    with torch.inference_mode():
        model = nn.Module()
        model.proj = nn.Linear(3, 2, bias=False)
        spec = SimpleNamespace(
            matrices=[SimpleNamespace(name="proj", module="proj")]
        )
        audit = _weight_audit(model, spec)
    assert audit["proj"]["shape"] == [2, 3]
    assert audit["proj"]["version"] is None


def test_generation_padding_preserves_each_actual_prefix() -> None:
    ids, valid = _pad_token_lists([[1, 2, 3], [4]], pad_token_id=0)
    assert ids.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert valid.tolist() == [[True, True, True], [True, False, False]]


def test_continuation_difference_counts_divergence_and_length() -> None:
    assert _continuation_difference([1, 2, 3], [1, 4]) == {
        "changed_token_count": 2,
        "common_prefix_tokens": 1,
    }

