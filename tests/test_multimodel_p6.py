import torch

from aspd.multimodel.causal import (
    ThresholdGateCalibration,
    calibrate_threshold_gate,
    temporal_region,
    threshold_causal_gate,
)


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

