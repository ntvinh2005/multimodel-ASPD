from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_recipe import sha256_file
from aspd.multimodel.original_streaming import resolved_config_sha256
from aspd.multimodel.p6_original_causal import (
    clean_reference_gate,
    load_100m_candidate_shortlist,
    make_p6_result_record,
    output_matched_control,
    run_original_p6_arm,
    validate_100m_checkpoint_contract,
    validate_p6_pilot_config,
)
from aspd.multimodel.p6_original_generation import (
    generate_original_p6_branch,
    run_original_p6_autoregressive,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"
PILOT = ROOT / "configs/multimodel/qwen25_extreme_sports/p6_100m_causal_pilot.yaml"


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))

    def forward(self, input_ids, attention_mask, use_cache):
        del attention_mask, use_cache
        values = input_ids.float()
        x = torch.stack((values, torch.ones_like(values)), dim=-1)
        return SimpleNamespace(logits=self.proj(x))


class _ToyDecomposition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def component_factors(self, model_index, matrix_name):
        del model_index, matrix_name
        return torch.tensor([[1.0, 2.0], [0.0, 0.0]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def component_input_center(self, model_index, matrix_name):
        del model_index, matrix_name
        return torch.tensor([1.0, 0.0])


def _spec():
    return SimpleNamespace(
        name="target", matrices=[SimpleNamespace(name="o_proj", module="proj")]
    )


def test_actual_output_controls_match_norms_and_preserve_zero_positions() -> None:
    candidate = torch.tensor([[[3.0, 4.0], [0.0, 0.0], [5.0, 0.0]]])
    orthogonal = output_matched_control(
        candidate, arm="orthogonal_matched", identity="test"
    )
    torch.testing.assert_close(orthogonal.norm(dim=-1), candidate.norm(dim=-1))
    torch.testing.assert_close(
        (orthogonal * candidate).sum(dim=-1), torch.zeros(1, 3), atol=1e-5, rtol=0
    )
    first = output_matched_control(
        candidate, arm="random_same_l2", identity="test", seed=7
    )
    repeated = output_matched_control(
        candidate, arm="random_same_l2", identity="test", seed=7
    )
    different = output_matched_control(
        candidate, arm="random_same_l2", identity="test", seed=8
    )
    assert torch.equal(first, repeated)
    assert not torch.equal(first[:, [0, 2]], different[:, [0, 2]])
    torch.testing.assert_close(first.norm(dim=-1), candidate.norm(dim=-1))
    assert torch.equal(first[:, 1], torch.zeros_like(first[:, 1]))


def test_original_p6_uses_centered_contribution_and_cleans_hooks() -> None:
    model = _ToyModel()
    decomposition = _ToyDecomposition()
    ids = torch.tensor([[2, 3]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    gate = torch.tensor([[[True], [False]]])
    clean = model(input_ids=ids, attention_mask=valid, use_cache=False).logits
    weight = model.proj.weight.detach().clone()
    learned = run_original_p6_arm(
        model, _spec(), decomposition, 0, ids, valid, gate, 0, arm="learned", alpha=1.0
    )
    torch.testing.assert_close(
        learned.matrix_outputs["o_proj"]["perturbation"],
        torch.tensor([[[1.0, 2.0], [0.0, 0.0]]]),
    )
    torch.testing.assert_close(learned.logits, clean - torch.tensor([[[1.0, 2.0], [0.0, 0.0]]]))
    assert learned.matrix_diagnostics["o_proj"]["modified_positions"] == [[0, 0]]
    zero = run_original_p6_arm(
        model, _spec(), decomposition, 0, ids, valid, gate, 0, arm="learned", alpha=0.0
    )
    gate_off = run_original_p6_arm(
        model, _spec(), decomposition, 0, ids, valid, gate, 0, arm="gate_off", alpha=1.0
    )
    assert torch.equal(zero.logits, clean)
    assert torch.equal(gate_off.logits, clean)
    assert gate_off.matrix_diagnostics["o_proj"]["modified_positions"] == []
    assert torch.equal(model.proj.weight, weight)
    assert len(model.proj._forward_hooks) == 0

    with pytest.raises(ValueError, match="active control is zero"):
        run_original_p6_arm(
            model,
            _spec(),
            decomposition,
            0,
            ids,
            valid,
            gate,
            0,
            arm="active_matched",
            alpha=1.0,
            matched_component_id=1,
        )
    assert len(model.proj._forward_hooks) == 0


class _GateDecomposition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.training = False

    def encode(self, activations, valid_tokens, update_statistics):
        del valid_tokens, update_statistics
        gate = (activations[0][..., :2] + activations[1][..., :2]) > 0
        return SimpleNamespace(gate=gate)


def test_clean_reference_gate_requires_same_exact_prefix_tokens() -> None:
    decomposition = _GateDecomposition()
    ids = torch.tensor([[1, 2]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    residual = torch.ones(1, 2, 2)
    gate = clean_reference_gate(
        decomposition,
        base_token_ids=ids,
        target_token_ids=ids.clone(),
        clean_base_grounding=residual,
        clean_target_grounding=residual,
        valid_tokens=valid,
        component_ids=[0],
    )
    assert gate.shape == (1, 2, 1) and bool(gate.all())
    with pytest.raises(ValueError, match="identical token IDs"):
        clean_reference_gate(
            decomposition,
            base_token_ids=ids,
            target_token_ids=torch.tensor([[1, 3]]),
            clean_base_grounding=residual,
            clean_target_grounding=residual,
            valid_tokens=valid,
            component_ids=[0],
        )


def test_checkpoint_and_shortlist_provenance_fail_closed(tmp_path: Path) -> None:
    cfg = load_experiment_config(CONFIG)
    checkpoint = tmp_path / "checkpoint_00024414.rank0.pt"
    torch.save(
        {
            "step": 24_414,
            "config_sha256": resolved_config_sha256(cfg),
            "model": {},
            "optimizer": {},
        },
        checkpoint,
    )
    digest = sha256_file(checkpoint)
    assert validate_100m_checkpoint_contract(
        cfg, checkpoint, expected_checkpoint_sha256=digest
    )["trainer_rank"] == 0
    with pytest.raises(ValueError, match="hash"):
        validate_100m_checkpoint_contract(
            cfg, checkpoint, expected_checkpoint_sha256="0" * 64
        )

    shortlist = tmp_path / "p6_handoff" / "candidates.csv"
    shortlist.parent.mkdir()
    with shortlist.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["component_id", "rank"])
        writer.writeheader()
        writer.writerows([{"component_id": 7, "rank": 1}, {"component_id": 3, "rank": 2}])
    manifest = tmp_path / "result_package_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "P6_HANDOFF_READY",
                "artifacts": [
                    {
                        "path": "p6_handoff/candidates.csv",
                        "sha256": sha256_file(shortlist),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_100m_candidate_shortlist(shortlist, package_manifest=manifest)
    assert [row["component_id"] for row in loaded["candidates"]] == [7, 3]
    shortlist.write_text("component_id,rank\n8,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        load_100m_candidate_shortlist(shortlist, package_manifest=manifest)


def test_p6_pilot_is_balanced_checksummed_and_blocked_without_handoff() -> None:
    report = validate_p6_pilot_config(PILOT)
    assert report["status"] == "BLOCKED"
    assert report["category_counts"] == {
        "risk": 12,
        "sports_control": 12,
        "neutral": 12,
    }
    assert report["matched_pair_count"] == 12
    assert report["maximum_candidates"] == 3
    assert len(report["random_control_seeds"]) == 5
    assert report["auto_launch"] is False
    with pytest.raises(RuntimeError, match="BLOCKED"):
        validate_p6_pilot_config(PILOT, require_ready=True)


def test_p6_require_ready_verifies_checkpoint_package_and_shortlist(tmp_path: Path) -> None:
    cfg = load_experiment_config(CONFIG)
    checkpoint = tmp_path / "checkpoint_00024414.rank0.pt"
    torch.save(
        {
            "step": 24_414,
            "config_sha256": resolved_config_sha256(cfg),
            "model": {},
            "optimizer": {},
        },
        checkpoint,
    )
    shortlist = tmp_path / "candidates.csv"
    shortlist.write_text("component_id,rank\n7,1\n", encoding="utf-8")
    package = tmp_path / "result_package_manifest.json"
    package.write_text(
        json.dumps(
            {
                "status": "P6_HANDOFF_READY",
                "artifacts": [
                    {"path": checkpoint.name, "sha256": sha256_file(checkpoint)},
                    {"path": shortlist.name, "sha256": sha256_file(shortlist)},
                ],
            }
        ),
        encoding="utf-8",
    )
    pilot = yaml.safe_load(PILOT.read_text(encoding="utf-8"))
    pilot["scientific_status"] = "READY"
    pilot["checkpoint"] = {
        "rank0_path": str(checkpoint),
        "expected_sha256": sha256_file(checkpoint),
    }
    pilot["candidate_source"]["shortlist_path"] = str(shortlist)
    pilot["candidate_source"]["package_manifest"] = str(package)
    ready_path = tmp_path / "ready.yaml"
    ready_path.write_text(yaml.safe_dump(pilot, sort_keys=False), encoding="utf-8")
    report = validate_p6_pilot_config(ready_path, require_ready=True)
    assert report["status"] == "READY"
    assert report["artifact_verification"]["checkpoint"]["step"] == 24_414
    assert report["artifact_verification"]["candidate_ids"] == [7]

    package.write_text(
        json.dumps(
            {
                "status": "P6_HANDOFF_READY",
                "artifacts": [
                    {"path": shortlist.name, "sha256": sha256_file(shortlist)}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checkpoint hash/path"):
        validate_p6_pilot_config(ready_path, require_ready=True)


def test_v2_record_preserves_provenance_and_does_not_invent_behavior() -> None:
    cfg = load_experiment_config(CONFIG)
    record = make_p6_result_record(
        run_id="synthetic-test",
        scientific_status="SYNTHETIC_TEST_ONLY",
        cfg=cfg,
        checkpoint={"sha256": "a" * 64, "step": 24_414},
        prompt_dataset={"id": "fixture", "sha256": "b" * 64},
        shortlist={"source_sha256": "c" * 64},
        component_id=7,
        model_index=1,
        arm="random_same_l2",
        alpha=0.5,
        gate_policy={
            "name": "learned_jumprelu_clean_reference",
            "clean_reference": True,
            "training_batch_topk_reused_online": False,
            "actual_firing": {"count": 1},
        },
        diagnostics={
            "actual_perturbation_l2": 2.0,
            "activation_relative_l2": 0.1,
            "modified_positions": [[0, 2]],
        },
        matrix_output_references={
            "clean": "outputs.pt#clean",
            "perturbation": "outputs.pt#perturbation",
            "intervened": "outputs.pt#intervened",
        },
        random_seed=3,
        generation_seed=0,
        continuation_references={"clean": "text.json#clean", "intervened": "text.json#patched"},
    )
    assert record["scientific_status"] == "SYNTHETIC_TEST_ONLY"
    assert record["source_models"][1]["revision"] == cfg.models[1].revision
    assert record["intervention"]["actual_perturbation_l2"] == 2.0
    assert record["rng_seeds"] == {"generation": 0, "random_control": 3}
    assert record["behavioral_metrics"] is None


class _BF16ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2) * 1_000)

    def forward(self, input_ids, attention_mask, use_cache):
        del attention_mask, use_cache
        values = input_ids.to(torch.bfloat16)
        x = torch.stack((values, torch.ones_like(values)), dim=-1)
        return SimpleNamespace(logits=self.proj(x))


class _BF16Decomposition(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.scale = scale

    def component_factors(self, model_index, matrix_name):
        del model_index, matrix_name
        return (
            torch.tensor([[self.scale, 0.0], [0.0, self.scale]]),
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        )

    def component_input_center(self, model_index, matrix_name):
        del model_index, matrix_name
        return None


def test_bf16_diagnostics_measure_effective_not_requested_delta() -> None:
    ids = torch.tensor([[1]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    gate = torch.ones(1, 1, 1, dtype=torch.bool)
    tiny = run_original_p6_arm(
        _BF16ToyModel(), _spec(), _BF16Decomposition(0.1), 0,
        ids, valid, gate, 0, arm="learned", alpha=1.0,
    )
    diagnostic = tiny.matrix_diagnostics["o_proj"]
    assert diagnostic["intended_perturbation_l2"] > 0
    assert diagnostic["applied_perturbation_l2"] == 0
    assert diagnostic["ineffective_dose"] is True
    assert diagnostic["quantization_lost_fraction"] == 1.0
    assert diagnostic["actually_changed_position_count"] == 0

    zero_contribution = run_original_p6_arm(
        _BF16ToyModel(), _spec(), _BF16Decomposition(0.0), 0,
        ids, valid, gate, 0, arm="learned", alpha=1.0,
    ).matrix_diagnostics["o_proj"]
    assert zero_contribution["intended_perturbation_l2"] == 0
    assert zero_contribution["applied_perturbation_l2"] == 0
    assert zero_contribution["ineffective_dose"] is False

    representable = run_original_p6_arm(
        _BF16ToyModel(), _spec(), _BF16Decomposition(8.0), 0,
        ids, valid, gate, 0, arm="learned", alpha=1.0,
    )
    ref = representable.matrix_diagnostics["o_proj"]["applied_perturbation_l2"]
    assert ref == 8.0
    assert representable.matrix_diagnostics["o_proj"]["actually_changed_position_count"] == 1
    for arm, matched, tolerance in (
        ("orthogonal_matched", None, 0.05),
        ("active_matched", 1, 0.05),
        ("random_same_l2", None, 0.50),
    ):
        control = run_original_p6_arm(
            _BF16ToyModel(), _spec(), _BF16Decomposition(8.0), 0,
            ids, valid, gate, 0, arm=arm, alpha=1.0,
            matched_component_id=matched,
            random_seed=7 if arm == "random_same_l2" else None,
            reference_applied_l2=ref,
            norm_match_relative_tolerance=tolerance,
        )
        assert control.matrix_diagnostics["o_proj"]["norm_match_within_tolerance"] is True
    with pytest.raises(ValueError, match="outside the declared tolerance"):
        run_original_p6_arm(
            _BF16ToyModel(), _spec(), _BF16Decomposition(8.0), 0,
            ids, valid, gate, 0, arm="orthogonal_matched", alpha=1.0,
            reference_applied_l2=100.0,
            norm_match_relative_tolerance=0.05,
        )


class _GenerationSpec(SimpleNamespace):
    def model_dump(self, mode="json"):
        del mode
        return {"name": self.name}


class _CausalToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        self.ground = nn.Identity()
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))

    def forward(self, input_ids, attention_mask, use_cache):
        del attention_mask, use_cache
        values = input_ids.float() + 1
        hidden = torch.stack((values, torch.ones_like(values)), dim=-1)
        output = self.ground(self.proj(hidden))
        logits = torch.stack(
            (output[..., 0], torch.full_like(output[..., 0], 0.5),
             torch.zeros_like(output[..., 0])), dim=-1
        )
        return SimpleNamespace(logits=logits)


class _GenerationDecomposition(_ToyDecomposition):
    def __init__(self) -> None:
        super().__init__()
        self.sparsity_controller = SimpleNamespace(threshold=torch.tensor([0.0, 0.0]))
        self.seen_prefix_last_values = []

    def component_factors(self, model_index, matrix_name):
        del model_index, matrix_name
        return torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    def component_input_center(self, model_index, matrix_name):
        del model_index, matrix_name
        return None

    def encode(self, activations, valid_tokens, update_statistics):
        del valid_tokens, update_statistics
        self.seen_prefix_last_values.append(float(activations[1][0, -1, 0]))
        return SimpleNamespace(gate=torch.ones((*activations[0].shape[:2], 2), dtype=torch.bool))


def _generation_cfg():
    grounding = SimpleNamespace(module="ground", capture="input", tensor_index=0)
    matrix = SimpleNamespace(name="o_proj", module="proj")
    specs = [
        _GenerationSpec(name="base", grounding=grounding, matrices=[matrix]),
        _GenerationSpec(name="target", grounding=grounding, matrices=[matrix]),
    ]
    return SimpleNamespace(models=specs)


def test_prefix_causal_generation_recomputes_each_diverged_branch() -> None:
    models = [_CausalToyModel(), _CausalToyModel()]
    weights = [[parameter.detach().clone() for parameter in model.parameters()] for model in models]
    decomposition = _GenerationDecomposition()
    result = run_original_p6_autoregressive(
        models, _generation_cfg(), decomposition,
        prompt_id="p", prompt_token_ids=[0], component_id=0, model_index=1,
        arm="learned", alpha=1.0, eos_token_ids={99}, max_new_tokens=2,
        checkpoint={"sha256": "a" * 64, "step": 24_414},
        shortlist_sha256="b" * 64, prompt_manifest_sha256="c" * 64,
    )
    assert result["clean"]["continuation_token_ids"] == [0, 0]
    assert result["intervened"]["continuation_token_ids"] == [1, 1]
    assert [row["prefix_token_count"] for row in result["clean"]["decisions"]] == [1, 2]
    assert [row["prefix_token_count"] for row in result["intervened"]["decisions"]] == [1, 2]
    assert result["clean"]["decisions"][1]["prefix_token_ids_sha256"] != result["intervened"]["decisions"][1]["prefix_token_ids_sha256"]
    # Clean branch then intervened branch: each second decision saw its own live token.
    assert decomposition.seen_prefix_last_values == [1.0, 1.0, 1.0, 2.0]
    for model, snapshots in zip(models, weights, strict=True):
        assert all(torch.equal(parameter, snapshot) for parameter, snapshot in zip(model.parameters(), snapshots, strict=True))
        assert not model.proj._forward_hooks and not model.ground._forward_pre_hooks


def test_prefix_generation_eos_and_max_token_termination() -> None:
    models = [_CausalToyModel(), _CausalToyModel()]
    eos = generate_original_p6_branch(
        models, _generation_cfg(), _GenerationDecomposition(), [0],
        component_id=0, model_index=1, arm="learned", alpha=1.0,
        eos_token_ids={1}, max_new_tokens=5,
    )
    assert eos["stop_reason"] == "eos" and eos["new_token_count"] == 1
    capped = generate_original_p6_branch(
        models, _generation_cfg(), _GenerationDecomposition(), [0],
        component_id=0, model_index=1, arm="clean", alpha=0.0,
        eos_token_ids={99}, max_new_tokens=3,
    )
    assert capped["stop_reason"] == "max_new_tokens" and capped["new_token_count"] == 3


@pytest.mark.parametrize(
    ("arm", "matched", "seed"),
    [
        ("clean", None, None),
        ("learned", None, None),
        ("gate_off", None, None),
        ("active_matched", 1, None),
        ("orthogonal_matched", None, None),
        ("random_same_l2", None, 7),
    ],
)
def test_every_generation_arm_has_a_zero_dose_noop_record(arm, matched, seed) -> None:
    branch = generate_original_p6_branch(
        [_CausalToyModel(), _CausalToyModel()],
        _generation_cfg(),
        _GenerationDecomposition(),
        [0],
        component_id=0,
        model_index=1,
        arm=arm,
        alpha=0.0,
        eos_token_ids={99},
        max_new_tokens=2,
        matched_component_id=matched,
        random_seed=seed,
    )
    assert branch["continuation_token_ids"] == [0, 0]
    assert branch["arm"] == arm
    assert all(
        decision["matrix_diagnostics"]["applied_perturbation_l2"] == 0
        for decision in branch["decisions"]
    )
    assert all(
        decision["clean_replay_max_abs_diff"] == 0
        for decision in branch["decisions"]
    )


class _RaisingCausalToyModel(_CausalToyModel):
    def forward(self, input_ids, attention_mask, use_cache):
        super().forward(input_ids, attention_mask, use_cache)
        raise RuntimeError("synthetic forward failure")


def test_generation_cleans_capture_hooks_after_exception() -> None:
    models = [_CausalToyModel(), _RaisingCausalToyModel()]
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        generate_original_p6_branch(
            models, _generation_cfg(), _GenerationDecomposition(), [0],
            component_id=0, model_index=1, arm="learned", alpha=1.0,
            eos_token_ids={99}, max_new_tokens=2,
        )
    for model in models:
        assert not model.proj._forward_hooks
        assert not model.ground._forward_pre_hooks
