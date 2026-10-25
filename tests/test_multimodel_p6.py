import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aspd.multimodel.causal import (
    ThresholdGateCalibration,
    calibrate_threshold_gate,
    norm_matched_orthogonal_rows,
    norm_matched_random_rows,
    p6_patched_forward,
    p6_subtraction,
    temporal_region,
    threshold_causal_gate,
    validate_random_control_seeds,
)
from aspd.multimodel.p6_fixed import _weight_audit, select_fixed_feasibility_rows
from aspd.multimodel.p6_generation import _continuation_difference, _pad_token_lists
from aspd.multimodel.p6_judge import _load_source_rows, parse_p6_judgment


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


def test_p6_subtraction_uses_transcoder_input_center() -> None:
    x = torch.tensor([[[3.0, 5.0]]])
    u = torch.tensor([[2.0, 4.0]])
    v = torch.tensor([[1.0, 0.0]])
    gate = torch.ones(1, 1, 1, dtype=torch.bool)
    observed = p6_subtraction(
        x, u, v, gate, input_center=torch.tensor([1.0, 2.0])
    )
    torch.testing.assert_close(observed, torch.tensor([[[4.0, 8.0]]]))


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


def test_p6_judge_parser_enforces_full_typed_schema() -> None:
    payload = {
        "risk_score": 2,
        "risk_code": "PERMISSIVE",
        "assertiveness": 2,
        "refusal": False,
        "generic_affirmation": True,
        "meaningful_safety_precautions": False,
        "unsafe_encouragement": True,
        "coherence": 2,
        "task_relevance": 2,
        "incomplete_or_corrupted": False,
    }
    assert parse_p6_judgment("prefix " + json.dumps(payload))["risk_score"] == 2
    payload["risk_code"] = "CAUTIOUS"
    with pytest.raises(ValueError, match="mismatch"):
        parse_p6_judgment(json.dumps(payload))


def test_random_control_is_seeded_independent_and_norm_matched() -> None:
    rows = torch.tensor([[1.0, 2.0, 4.0], [-3.0, 1.0, 2.0]])
    kwargs = {
        "model_name": "target",
        "matrix_name": "up_proj",
        "matrix_module": "model.layers.14.mlp.up_proj",
        "component_ids": [3228, 290],
    }
    first, metadata = norm_matched_random_rows(rows, seed=0, **kwargs)
    repeated, repeated_metadata = norm_matched_random_rows(rows, seed=0, **kwargs)
    different, _ = norm_matched_random_rows(rows, seed=1, **kwargs)
    assert torch.equal(first, repeated)
    assert metadata == repeated_metadata
    assert not torch.equal(first, different)
    torch.testing.assert_close(first.norm(dim=1), rows.norm(dim=1))
    assert [record["candidate_component_id"] for record in metadata] == [3228, 290]
    assert all(record["layer_index"] == 14 for record in metadata)
    assert all(record["norm_matching_error"] <= 1e-6 for record in metadata)


def test_random_control_rejects_invalid_vectors_seeds_and_modes() -> None:
    kwargs = {
        "model_name": "target",
        "matrix_name": "up_proj",
        "matrix_module": "model.layers.14.mlp.up_proj",
        "component_ids": [7],
        "seed": 0,
    }
    with pytest.raises(ValueError, match="positive L2 norm"):
        norm_matched_random_rows(torch.zeros(1, 3), **kwargs)
    with pytest.raises(ValueError, match="finite"):
        norm_matched_random_rows(torch.tensor([[1.0, float("nan"), 2.0]]), **kwargs)
    with pytest.raises(ValueError, match="unique"):
        validate_random_control_seeds([0, 0])
    with pytest.raises(TypeError, match="integers"):
        validate_random_control_seeds([True])


class _ToyP6Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        values = input_ids.float()
        x = torch.stack((values, torch.ones_like(values)), dim=-1)
        return SimpleNamespace(logits=self.proj(x))


class _ToyP6Decomposition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def component_factors(self, model_index: int, matrix_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        del model_index, matrix_name
        return torch.tensor([[1.0, 2.0]]), torch.tensor([[0.5, -0.25]])


def test_random_control_stays_fixed_and_p6_cleans_hooks_without_mutating_weights() -> None:
    torch.manual_seed(7)
    model = _ToyP6Model()
    decomposition = _ToyP6Decomposition()
    spec = SimpleNamespace(name="target", matrices=[SimpleNamespace(name="proj", module="proj")])
    ids = torch.tensor([[2, 3]])
    valid = torch.tensor([[True, True]])
    gate = torch.tensor([[[True], [False]]])
    weight_before = model.proj.weight.detach().clone()
    clean = model(input_ids=ids, attention_mask=valid, use_cache=False).logits
    zero = p6_patched_forward(
        model, spec, decomposition, 0, ids, valid, gate, [0], strength=0.0
    )
    first = p6_patched_forward(
        model, spec, decomposition, 0, ids, valid, gate, [0], random_control_seed=3
    )
    repeated = p6_patched_forward(
        model, spec, decomposition, 0, ids, valid, gate, [0], random_control_seed=3
    )
    torch.testing.assert_close(zero.logits, clean, rtol=0, atol=0)
    torch.testing.assert_close(first.logits, repeated.logits, rtol=0, atol=0)
    assert (
        first.matrix_diagnostics["proj"]["random_direction_controls"]
        == repeated.matrix_diagnostics["proj"]["random_direction_controls"]
    )
    assert first.matrix_diagnostics["proj"]["max_abs_gate_off_subtraction"] == 0.0
    assert len(model.proj._forward_hooks) == 0
    assert torch.equal(model.proj.weight, weight_before)
    with pytest.raises(ValueError, match="incompatible"):
        p6_patched_forward(
            model,
            spec,
            decomposition,
            0,
            ids,
            valid,
            gate,
            [0],
            norm_control=True,
            random_control_seed=0,
        )


def test_threshold_calibration_rejects_nonfinite_metadata() -> None:
    with pytest.raises(ValueError, match="finite"):
        calibrate_threshold_gate(
            torch.tensor([[0.1], [float("inf")]]),
            component_ids=[0],
            target_rates=torch.tensor([0.5]),
        )
    with pytest.raises(ValueError, match="at least one token"):
        calibrate_threshold_gate(
            torch.empty(0, 1), component_ids=[0], target_rates=torch.tensor([0.5])
        )


def test_judge_accepts_validated_variable_arm_manifest(tmp_path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_rows = [
        {"prompt_id": "risk", "prompt": "r", "pair_id": "pair", "category": "risk", "sport_domain": "s"},
        {"prompt_id": "control", "prompt": "c", "pair_id": "pair", "category": "sports_control", "sport_domain": "s"},
    ]
    prompt_path.write_text("\n".join(json.dumps(row) for row in prompt_rows) + "\n", encoding="utf-8")
    sequences = [
        {
            "example_id": f"{row['prompt_id']}::target_generated",
            "category": row["category"],
            "continuation_text": "answer",
            "eos_reached": True,
            "new_token_count": 1,
            "changed_token_count": 0,
        }
        for row in prompt_rows
    ]
    generation = {
        "arms": [{"arm": "target_random_seed_0", "model_name": "target", "sequences": sequences}],
        "experiment_manifest": {
            "schema_version": 1,
            "arm_names": ["target_random_seed_0"],
            "arm_count": 1,
            "judge_prompt_groups": ["risk", "sports_control"],
            "judge_rows_by_arm": {"target_random_seed_0": 2},
            "expected_judge_rows": 2,
        },
    }
    assert len(_load_source_rows(generation, prompt_path)) == 2
    generation["experiment_manifest"]["expected_judge_rows"] = 3
    with pytest.raises(ValueError, match="expected judge row count"):
        _load_source_rows(generation, prompt_path)


def test_historical_judge_protocol_still_accepts_exactly_252_rows(tmp_path) -> None:
    prompt_path = tmp_path / "historical_prompts.jsonl"
    prompt_rows = [
        {
            "prompt_id": f"prompt-{index}",
            "prompt": f"prompt {index}",
            "pair_id": f"pair-{index // 2}",
            "category": "risk" if index % 2 == 0 else "sports_control",
            "sport_domain": f"sport-{index // 2}",
        }
        for index in range(12)
    ]
    prompt_path.write_text(
        "\n".join(json.dumps(row) for row in prompt_rows) + "\n", encoding="utf-8"
    )
    arms = []
    for arm_index in range(21):
        sequences = [
            {
                "example_id": f"{row['prompt_id']}::target_generated",
                "category": row["category"],
                "continuation_text": "historical answer",
                "eos_reached": True,
                "new_token_count": 2,
                "changed_token_count": 0,
            }
            for row in prompt_rows
        ]
        arms.append(
            {"arm": f"historical-arm-{arm_index}", "model_name": "target", "sequences": sequences}
        )
    assert len(_load_source_rows({"arms": arms}, prompt_path)) == 252
