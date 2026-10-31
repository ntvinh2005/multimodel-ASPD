from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from aspd.multimodel.config import ModelSpec
from aspd.multimodel.original_recipe import sha256_file
from aspd.multimodel.risk_attribution import (
    CandidateBudgetSpec,
    ChoiceItem,
    _exact_preflight_status,
    _norm_validity,
    _verify_screening_artifact_hashes,
    build_choice_items,
    capture_margin_gradient,
    decision_margin,
    freeze_screening_candidates,
    load_scenario_pairs,
    readout_metrics,
    signed_component_attribution,
    summarize_attribution_tensors,
)


class _ToyAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.o_proj(hidden)


class _ToyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = _ToyAttention(width)
        self.post_attention_layernorm = nn.Identity()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.post_attention_layernorm(self.self_attn(hidden))


class _ToyQwen(nn.Module):
    def __init__(self, vocabulary: int = 11, width: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_ToyLayer(width) for _ in range(15)])
        self.lm_head = nn.Linear(width, vocabulary, bias=False)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _spec() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "base",
            "model_id": "toy",
            "grounding": {
                "module": "model.layers.14.post_attention_layernorm",
                "capture": "input",
                "tensor_index": 0,
            },
            "matrices": [
                {
                    "name": "o_proj",
                    "module": "model.layers.14.self_attn.o_proj",
                }
            ],
        }
    )


def _items(count: int = 2) -> list[ChoiceItem]:
    return [
        ChoiceItem(
            pair_id=f"p{index}",
            split="discovery",
            domain="toy",
            condition="risk",
            order_variant="primary",
            user_prompt="toy",
            proceed_label="A" if index == 0 else "B",
            postpone_label="B" if index == 0 else "A",
        )
        for index in range(count)
    ]


def test_frozen_dataset_has_balanced_discovery_and_heldout_pairs() -> None:
    rows = load_scenario_pairs(
        "configs/multimodel/qwen25_extreme_sports/risk_attribution_scenarios.jsonl"
    )
    assert len(build_choice_items(rows, split="discovery")) == 96
    assert len(build_choice_items(rows, split="heldout")) == 48
    assert not (
        {row["pair_id"] for row in rows if row["split"] == "discovery"}
        & {row["pair_id"] for row in rows if row["split"] == "heldout"}
    )


def test_decision_margin_is_invariant_to_swapped_answer_order() -> None:
    logits = torch.zeros(2, 5)
    # In row 0 A=PROCEED; in row 1 B=PROCEED. Both encode the same preference.
    logits[0, 1], logits[0, 2] = 4.0, 1.0
    logits[1, 1], logits[1, 2] = 1.0, 4.0
    torch.testing.assert_close(
        decision_margin(logits, _items(), {"A": 1, "B": 2}),
        torch.tensor([3.0, 3.0]),
    )


def test_signed_centered_component_attribution_matches_manual_equation() -> None:
    x = torch.tensor([[[2.0, 4.0], [3.0, 1.0]]])
    gradient = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    u = torch.tensor([[2.0, 1.0], [-1.0, 4.0]])
    v = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    center = torch.tensor([1.0, 2.0])
    gate = torch.tensor([[[True, False], [True, True]]])
    gate_values = gate.float() * 3.0
    valid = torch.ones(1, 2, dtype=torch.bool)
    result = signed_component_attribution(
        matrix_input=x,
        matrix_output_gradient=gradient,
        u=u,
        v=v,
        input_center=center,
        gate=gate,
        gate_values=gate_values,
        valid_tokens=valid,
        component_chunk_size=1,
    )
    centered = x - center
    contribution = torch.einsum("btd,cd->btc", centered, v)[:, :, :, None] * u
    contribution = contribution * gate[..., None]
    expected = -(contribution * gradient[:, :, None, :]).sum(dim=(1, 3))
    torch.testing.assert_close(result["predicted_delta_margin"], expected)
    assert result["gate_fire_count"].tolist() == [[2, 1]]


def test_real_o_proj_output_gradient_hook_is_signed_and_removed() -> None:
    torch.manual_seed(0)
    model = _ToyQwen().eval().requires_grad_(False)
    ids = torch.tensor([[1, 2], [3, 4]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    result = capture_margin_gradient(
        model, _spec(), ids, valid, _items(), {"A": 1, "B": 2}
    )
    assert result.matrix_input.shape == result.matrix_output.shape == (2, 2, 3)
    assert result.matrix_output_gradient.shape == (2, 2, 3)
    assert torch.isfinite(result.matrix_output_gradient).all()
    module = model.get_submodule("model.layers.14.self_attn.o_proj")
    assert not module._forward_pre_hooks
    assert not module._forward_hooks


def test_real_hook_cleanup_after_margin_failure() -> None:
    model = _ToyQwen().eval().requires_grad_(False)
    ids = torch.tensor([[1, 2], [3, 4]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    try:
        capture_margin_gradient(
            model, _spec(), ids, valid, _items(), {"A": 100, "B": 101}
        )
    except IndexError:
        pass
    else:
        raise AssertionError("invalid vocabulary indices must fail")
    module = model.get_submodule("model.layers.14.self_attn.o_proj")
    assert not module._forward_pre_hooks
    assert not module._forward_hooks


def test_readout_metrics_average_order_variants_before_pair_contrast() -> None:
    records = []
    # Target has a +2 larger risk-safe effect than Base for every pair. The two
    # option orders contain opposite nuisance offsets that vanish on averaging.
    for pair_id in ("p1", "p2"):
        for model_name, model_offset in (("base", 0.0), ("target", 2.0)):
            for condition, condition_value in (("risk", 1.0), ("safe_control", 0.0)):
                for order, nuisance in (("primary", 0.5), ("swapped", -0.5)):
                    records.append(
                        {
                            "model_name": model_name,
                            "pair_id": pair_id,
                            "condition": condition,
                            "order_variant": order,
                            "margin": condition_value * (1.0 + model_offset) + nuisance,
                        }
                    )
    metrics = readout_metrics(records, bootstrap_replicates=100, bootstrap_seed=0)
    assert metrics["base_mean_risk_minus_safe_margin"] == 1.0
    assert metrics["target_mean_risk_minus_safe_margin"] == 3.0
    assert metrics["target_minus_base_did_mean"] == 2.0


def test_candidate_freeze_keeps_risk_and_safety_signs_separate() -> None:
    rows = []
    for component_id, signed in enumerate((-4.0, -3.0, 4.0, 3.0, 0.01, -0.02, 0.2, -0.2)):
        rows.append(
            {
                "component_id": component_id,
                "active_discovery_pair_count": 4,
                "target_predicted_risk_delta_margin_mean": signed,
                "target_predicted_risk_minus_safe_mean": signed,
                "target_minus_base_risk_safe_differential": signed,
                "shared_gate_risk_fire_count_mean": 2.0 + component_id,
                "target_risk_write_l2_mean": 1.0 + component_id,
            }
        )
    frozen = freeze_screening_candidates(
        rows, CandidateBudgetSpec(risk_promoting=1, safety_promoting=1, negative_controls=1)
    )
    by_class = {row["selection_class"]: row for row in frozen["candidates"]}
    assert by_class["risk_promoting"]["component_id"] == 0
    assert by_class["safety_promoting"]["component_id"] == 2
    assert by_class["negative_control"]["component_id"] == 4


def test_risk_candidate_requires_negative_effect_on_actual_risk_prompts() -> None:
    rows = []
    values = (
        # Large negative contrasts but positive risk-prompt effect: must be rejected.
        (0, 2.0, -10.0, -10.0),
        (1, -2.0, -3.0, -3.0),
        (2, 2.0, 3.0, 3.0),
        (3, 0.1, 0.01, 0.01),
        (4, -0.1, -0.02, -0.02),
        (5, 0.2, 0.2, 0.2),
    )
    for component_id, risk_effect, contrast, differential in values:
        rows.append(
            {
                "component_id": component_id,
                "active_discovery_pair_count": 4,
                "target_predicted_risk_delta_margin_mean": risk_effect,
                "target_predicted_risk_minus_safe_mean": contrast,
                "target_minus_base_risk_safe_differential": differential,
                "shared_gate_risk_fire_count_mean": 2.0 + component_id,
                "target_risk_write_l2_mean": 1.0 + component_id,
            }
        )
    frozen = freeze_screening_candidates(
        rows, CandidateBudgetSpec(risk_promoting=1, safety_promoting=1, negative_controls=1)
    )
    risk = next(
        row for row in frozen["candidates"] if row["selection_class"] == "risk_promoting"
    )
    assert risk["component_id"] == 1


def test_summary_accepts_one_shared_gate_axis_and_two_model_axes() -> None:
    metadata = [
        {"pair_id": "p1", "condition": condition, "order_variant": order}
        for condition in ("risk", "safe_control")
        for order in ("primary", "swapped")
    ]
    effects = torch.zeros(4, 2, 3)
    effects[:2, 1, 0] = -2.0
    writes = torch.ones_like(effects)
    shared_fire = torch.ones(4, 3, dtype=torch.int32)
    shared_values = torch.ones(4, 3)
    rows = summarize_attribution_tensors(
        predicted_delta_margin=effects,
        intended_component_write_l2=writes,
        gate_fire_count=shared_fire,
        gate_value_sum=shared_values,
        metadata=metadata,
    )
    assert len(rows) == 3
    assert rows[0]["target_predicted_risk_minus_safe_mean"] == -2.0


def test_norm_matching_rejects_zero_candidate_reference() -> None:
    valid, reason = _norm_validity(
        {
            "per_position_actual_applied_l2": [0.0, 0.0],
            "per_position_candidate_actual_applied_l2": [0.0, 0.0],
        },
        0.1,
    )
    assert not valid
    assert reason == "ZERO_CANDIDATE_REFERENCE"


def test_norm_matching_rejects_position_swap_despite_equal_aggregate_l2() -> None:
    valid, reason = _norm_validity(
        {
            "per_position_actual_applied_l2": [1.0, 2.0],
            "per_position_candidate_actual_applied_l2": [2.0, 1.0],
            # These aggregate values are deliberately equal and must be ignored.
            "actual_applied_output_difference_l2": 5**0.5,
            "candidate_actual_applied_output_difference_l2": 5**0.5,
        },
        0.1,
    )
    assert not valid
    assert reason == "POST_DTYPE_PER_POSITION_NORM_MISMATCH"


def test_runtime_go_cannot_override_scientific_preflight_failure() -> None:
    records = [
        {
            "arm": "learned",
            "causal_comparison_valid": True,
            "invalid_reason": None,
        },
        {
            "arm": "active_matched",
            "causal_comparison_valid": False,
            "invalid_reason": "POST_DTYPE_PER_POSITION_NORM_MISMATCH",
        },
        {
            "arm": "orthogonal_matched",
            "causal_comparison_valid": True,
            "invalid_reason": None,
        },
        {
            "arm": "random_same_l2_seed_0",
            "causal_comparison_valid": True,
            "invalid_reason": None,
        },
    ]
    status = _exact_preflight_status(
        projected_runtime_seconds=10.0,
        runtime_budget_seconds=100,
        records=records,
        source_weights_unchanged=True,
        hook_cleanup_passed=True,
        artifact_hashes_verified=True,
    )
    assert status["compute_feasibility_status"] == "COMPUTE_FEASIBILITY_GO"
    assert status["scientific_preflight_status"] == "SCIENTIFIC_PREFLIGHT_NO_GO"
    assert status["status"] == "SCIENTIFIC_PREFLIGHT_NO_GO"


def test_screening_artifact_hashes_are_verified_before_exact_load(tmp_path) -> None:
    names = (
        "discovery_attribution_tensors.pt",
        "component_attribution_summary.csv",
        "frozen_candidates.json",
    )
    outputs = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(f"content:{name}".encode())
        outputs[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    verified = _verify_screening_artifact_hashes(tmp_path, {"outputs": outputs})
    assert set(verified) == set(names)
    (tmp_path / "frozen_candidates.json").write_text("corrupted", encoding="utf-8")
    try:
        _verify_screening_artifact_hashes(tmp_path, {"outputs": outputs})
    except ValueError as error:
        assert "frozen_candidates.json" in str(error)
    else:
        raise AssertionError("corrupted frozen candidates must fail provenance validation")
