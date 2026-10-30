from __future__ import annotations

from aspd.multimodel.diagnostics.original_optimization import synthetic_gradient_audit


def _gradient(report, probe: str, category: str) -> dict:
    return report["probes"][probe]["gradients"][category]


def test_isolated_objectives_have_expected_gradient_recipients() -> None:
    report = synthetic_gradient_audit(seed=0)

    assert _gradient(
        report, "activation_loss_alone", "shared_encoder_projection_base"
    )["gradient_l2_norm"] > 0
    assert _gradient(report, "activation_loss_alone", "base_activation_decoder")[
        "gradient_l2_norm"
    ] > 0
    assert _gradient(report, "activation_loss_alone", "base_component_u")[
        "gradient_none_tensors"
    ] == 1

    assert _gradient(
        report, "internal_loss_alone", "shared_encoder_projection_base"
    )["gradient_none_tensors"] == 1
    for category in (
        "base_component_u",
        "base_component_v",
        "base_component_b_dec",
        "base_component_b_out",
        "target_component_u",
        "target_component_v",
        "target_component_b_dec",
        "target_component_b_out",
    ):
        assert _gradient(report, "internal_loss_alone", category)["gradient_l2_norm"] > 0

    assert report["probes"]["auxk_actual_dead_state"]["term_values"]["auxk"] == 0
    assert _gradient(
        report, "auxk_actual_dead_state", "shared_encoder_projection_base"
    )["gradient_zero_tensors"] == 1
    assert _gradient(
        report, "auxk_forced_dead_path_probe", "shared_encoder_projection_base"
    )["gradient_l2_norm"] > 0
    assert _gradient(report, "auxk_forced_dead_path_probe", "base_component_u")[
        "gradient_none_tensors"
    ] == 1

    assert _gradient(
        report, "full_combined_objective", "shared_encoder_projection_base"
    )["gradient_l2_norm"] > 0
    assert _gradient(report, "full_combined_objective", "base_component_u")[
        "gradient_l2_norm"
    ] > 0
