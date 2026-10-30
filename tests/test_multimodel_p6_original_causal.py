import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from pydantic import ValidationError
from torch import nn

import aspd.multimodel.p6_original_causal as protocol
from aspd.multimodel.causal import P6ForwardResult
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_recipe import sha256_file
from aspd.multimodel.original_streaming import resolved_config_sha256
from aspd.multimodel.p6_100m_reporting import (
    build_p6_100m_report,
    differential_intervention_effect,
)
from aspd.multimodel.p6_behavioral import summarize_behavioral_scores
from aspd.multimodel.p6_judge import (
    _load_p6_100m_blind_rows,
    parse_p6_100m_judgment,
)

PILOT = Path("configs/multimodel/qwen25_extreme_sports/p6_100m_causal_pilot.yaml")
RUN = Path("out/multimodel/runs/qwen25_aspd_original_100m")
ASPD_CONFIG = Path(
    "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"
)


def _dynamic_candidates() -> tuple[protocol.P6CandidateSpec, ...]:
    shortlist = sorted(
        csv.DictReader(
            (RUN / "p6_handoff/candidates_risk_screening.csv").open(encoding="utf-8")
        ),
        key=lambda row: int(row["rank"]),
    )[:3]
    metrics = list(
        csv.DictReader((RUN / "evaluation/component_metrics.csv").open(encoding="utf-8"))
    )
    candidate_ids = [int(row["component_id"]) for row in shortlist]
    controls = protocol._select_active_control_ids(metrics, candidate_ids)
    return tuple(
        protocol.P6CandidateSpec(
            component_id=component_id,
            risk_screening_rank=int(row["rank"]),
            active_control_component_id=control_id,
        )
        for row, component_id, control_id in zip(
            shortlist, candidate_ids, controls, strict=True
        )
    )


def test_original_100m_config_fingerprints_are_unchanged() -> None:
    assert sha256_file(ASPD_CONFIG) == (
        "904a91d2ab2c63cafd5f62c9c012bcfbd418f72de8114fffa8f4d14469626c40"
    )
    assert resolved_config_sha256(load_experiment_config(ASPD_CONFIG)) == (
        "0f20b77c142f0ba4f1a05e9bf60af2002c1f41de555c92564b5cf379b744c041"
    )


def test_p6_config_is_separate_and_rejects_stale_unknown_fields() -> None:
    payload = yaml.safe_load(PILOT.read_text(encoding="utf-8"))
    payload["intervention_model"] = "target"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        protocol.P6OriginalPilotConfig.model_validate(payload)


def test_effective_operator_comparison_is_rank_one_gauge_invariant() -> None:
    base_u = torch.tensor([1.0, 2.0, -1.0])
    base_v = torch.tensor([3.0, -2.0])
    target_u = torch.tensor([-0.5, 4.0, 2.0])
    target_v = torch.tensor([1.5, 0.25])
    expected = protocol.component_operator_comparison(
        base_u, base_v, target_u, target_v
    )
    observed = protocol.component_operator_comparison(
        base_u * -7.0,
        base_v / -7.0,
        target_u * 0.25,
        target_v / 0.25,
    )
    assert observed == pytest.approx(expected)


def test_checkpoint_tensor_contract_validates_shapes_finiteness_and_step(tmp_path) -> None:
    state = {}
    paths = {}
    for model_name, model_index in (("base", 0), ("target", 1)):
        for factor in ("u", "v"):
            logical = f"{model_name}_{factor}"
            path = f"components.{model_index}.o_proj.{factor.upper()}"
            paths[logical] = path
            state[path] = torch.ones(36_864, 2)
        for bias in ("b_dec", "b_out"):
            logical = f"{model_name}_{bias}"
            path = f"components.{model_index}.o_proj.{bias}"
            paths[logical] = path
            state[path] = torch.zeros(2)
    paths["shared_gate_threshold"] = "sparsity_controller.threshold"
    state["sparsity_controller.threshold"] = torch.tensor(3.0)
    checkpoint = tmp_path / "rank0.pt"
    torch.save(
        {"step": 24_414, "config_sha256": "f" * 64, "model": state}, checkpoint
    )
    audit = protocol._validate_checkpoint_component_tensors(
        checkpoint, {"state_dict_paths": paths}, [1, 7]
    )
    assert audit["component_count"] == 36_864
    assert audit["component_width"] == 2
    state[paths["target_v"]][7, 0] = float("nan")
    torch.save(
        {"step": 24_414, "config_sha256": "f" * 64, "model": state}, checkpoint
    )
    with pytest.raises(FloatingPointError, match="target_v"):
        protocol._validate_checkpoint_component_tensors(
            checkpoint, {"state_dict_paths": paths}, [1, 7]
        )


def test_frozen_p6_config_matches_verified_handoff_provenance() -> None:
    pilot = protocol.load_p6_original_config(PILOT)
    manifest = json.loads((RUN / "result_package_manifest.json").read_text())
    artifacts = {row["path"]: row for row in manifest["artifacts"]}
    index = json.loads((RUN / "p6_handoff/component_parameter_index.json").read_text())
    shortlist_path = RUN / "p6_handoff/candidates_risk_screening.csv"
    shortlist = list(csv.DictReader(shortlist_path.open(encoding="utf-8")))

    assert manifest["status"] == "P6_HANDOFF_READY"
    assert manifest["risk_screening_status"] == "VERIFIED"
    assert index["authoritative_checkpoint_relative_to_run"] == "checkpoint_00024414.rank0.pt"
    assert index["authoritative_checkpoint_sha256"] == pilot.expected_checkpoint_sha256
    assert artifacts["checkpoint_00024414.rank0.pt"]["sha256"] == pilot.expected_checkpoint_sha256
    assert sha256_file(pilot.aspd_config) == pilot.expected_config_file_sha256
    assert sha256_file(pilot.prompt_file) == pilot.prompt_sha256
    assert (
        sha256_file(pilot.behavioral_scoring_contract)
        == pilot.behavioral_scoring_contract_sha256
    )
    prompt_rows = [
        json.loads(line)
        for line in Path(pilot.prompt_file).read_text(encoding="utf-8").splitlines()
    ]
    assert len(prompt_rows) == 36
    assert {row["pair_id"] for row in prompt_rows if row["category"] == "risk"} == {
        row["pair_id"]
        for row in prompt_rows
        if row["category"] == "sports_control"
    }
    candidates = _dynamic_candidates()
    assert pilot.maximum_candidates == 3
    assert [(row.component_id, row.risk_screening_rank) for row in candidates] == [
        (1720, 1),
        (1173, 2),
        (2228, 3),
    ]
    assert [int(row["component_id"]) for row in shortlist[:3]] == [
        candidate.component_id for candidate in candidates
    ]
    assert all(
        row["aspd_checkpoint_sha256"] == pilot.expected_checkpoint_sha256
        for row in shortlist[:3]
    )


def test_frozen_arm_inventory_and_prompt_split_are_bounded() -> None:
    pilot = protocol.load_p6_original_config(PILOT)
    candidates = _dynamic_candidates()
    technical = protocol.intervention_arms(pilot, candidates, technical=True)
    full = protocol.intervention_arms(pilot, candidates)
    assert len(technical) == 22
    assert len(full) == 218
    assert sum(arm.kind == "random_same_l2" for arm in full) == 120
    assert {arm.model_name for arm in full} == {"base", "target"}
    assert {arm.random_seed for arm in full if arm.kind == "random_same_l2"} == {0, 1, 2, 3, 4}
    assert {arm.alpha for arm in full} == {0.0, 0.25, 0.5, 1.0}
    assert all(len(values) == 12 for values in pilot.prompt_ids.values())
    assert all(len(values) == 2 for values in pilot.technical_prompt_ids.values())


def test_active_controls_reproduce_frozen_preoutcome_selection_rule() -> None:
    candidates = _dynamic_candidates()
    rows = list(csv.DictReader((RUN / "evaluation/component_metrics.csv").open()))
    by_id = {int(row["component_id"]): row for row in rows}
    excluded = {candidate.component_id for candidate in candidates}
    observed = []
    for candidate in candidates:
        reference = by_id[candidate.component_id]

        def score(row, reference=reference):
            firing_ratio = max(float(row["held_out_firing_frequency"]), 1e-30) / max(
                float(reference["held_out_firing_frequency"]), 1e-30
            )
            contribution_ratio = max(
                float(row["target_contribution_rms_per_validation_token"]), 1e-30
            ) / max(
                float(reference["target_contribution_rms_per_validation_token"]), 1e-30
            )
            return (
                abs(math.log(firing_ratio))
                + abs(math.log(contribution_ratio))
                + abs(float(row["paired_risk_minus_control_activation_strength"]))
            )

        eligible = [
            row
            for row in rows
            if int(row["component_id"]) not in excluded
            and float(row["held_out_fire_count"]) > 0
        ]
        selected = min(eligible, key=score)
        selected_id = int(selected["component_id"])
        observed.append(selected_id)
        excluded.add(selected_id)
    assert observed == [candidate.active_control_component_id for candidate in candidates]


class _Anchor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


def test_generation_recomputes_clean_gate_on_branch_specific_prefix(monkeypatch) -> None:
    seen_prefixes: list[list[list[int]]] = []
    patched_model_indices: list[int] = []

    def capture(_model, _spec, ids, valid):
        del valid
        seen_prefixes.append(ids.tolist())
        return {"R": ids[..., None].float(), "logits": torch.zeros(*ids.shape, 8)}

    def gate(_decomposition, residuals, valid, component_ids):
        del _decomposition, residuals, component_ids
        return torch.ones(*valid.shape, 1, dtype=torch.bool)

    def patched(
        _model,
        _spec,
        _decomposition,
        _model_index,
        ids,
        valid,
        _gate,
        _component_ids,
        **_kwargs,
    ):
        patched_model_indices.append(_model_index)
        logits = torch.zeros(*ids.shape, 8)
        logits[:, -1, 2 if ids.shape[1] == 1 else 3] = 1
        per_sequence = [
            {
                "gate_active_token_count": int(row.sum()),
                "modified_token_position_count": int(row.sum()),
                "rounded_to_zero_position_count": 0,
                "intended_fp32_subtraction_l2": 1.0,
                "actual_applied_output_difference_l2": 1.0,
                "relative_applied_l2_vs_clean_output": 0.1,
                "per_position_actual_applied_l2": [1.0] * int(row.sum()),
            }
            for row in valid
        ]
        return P6ForwardResult(logits, {"o_proj": {"per_sequence": per_sequence}})

    monkeypatch.setattr(protocol, "capture_model_forward", capture)
    monkeypatch.setattr(protocol, "clean_reference_gate", lambda decomposition, **kwargs: gate(
        decomposition,
        [kwargs["clean_base_grounding"], kwargs["clean_target_grounding"]],
        kwargs["valid_tokens"],
        kwargs["component_ids"],
    ))
    monkeypatch.setattr(protocol, "p6_patched_forward", patched)
    model_cfg = SimpleNamespace(name="base", matrices=[SimpleNamespace(name="o_proj")])
    cfg = SimpleNamespace(
        models=[model_cfg, SimpleNamespace(name="target", matrices=model_cfg.matrices)]
    )
    generated = protocol._generate_arm(
        arm=protocol.InterventionArm(
            "target_c1_learned", "learned", "target", 1, 1, 1.0
        ),
        models=[_Anchor(), _Anchor()],
        cfg=cfg,
        decomposition=_Anchor(),
        prompt_tokens=[[1]],
        pad_token_id=0,
        eos_token_ids=set(),
        max_new_tokens=2,
    )
    assert generated["continuations"] == [[2, 3]]
    assert seen_prefixes == [[[1]], [[1]], [[1, 2]], [[1, 2]]]
    assert generated["sequence_audit"][0][
        "logit_max_abs_difference_vs_clean_reference"
    ] == 1.0
    assert patched_model_indices == [1, 1]

    base_generated = protocol._generate_arm(
        arm=protocol.InterventionArm(
            "base_c1_learned", "learned", "base", 1, 1, 1.0
        ),
        models=[_Anchor(), _Anchor()],
        cfg=cfg,
        decomposition=_Anchor(),
        prompt_tokens=[[1]],
        pad_token_id=0,
        eos_token_ids=set(),
        max_new_tokens=1,
    )
    assert base_generated["continuations"] == [[2]]
    assert patched_model_indices[-1] == 0


def test_technical_gate_rejects_zero_arm_logit_mismatch() -> None:
    common = {
        "stage": "technical",
        "prompt_id": "p",
        "intervention_model": "target",
        "patched_continuation_token_ids": [1],
        "applied_perturbation_l2": 0.0,
        "gate_firing_count": 0,
        "logit_max_abs_difference_vs_clean_reference": 0.0,
    }
    records = [
        {**common, "intervention_kind": "clean", "alpha": 0.0},
        {
            **common,
            "intervention_kind": "learned",
            "alpha": 0.0,
            "logit_max_abs_difference_vs_clean_reference": 1.0,
        },
        {**common, "intervention_kind": "gate_off", "alpha": 1.0},
        {
            **common,
            "intervention_kind": "learned",
            "alpha": 1.0,
            "applied_perturbation_l2": 1.0,
            "gate_firing_count": 1,
        },
    ]
    report = protocol._technical_gate_report(
        records,
        {
            "comparisons": [
                {
                    "stage": "technical",
                    "learned_actual_applied_l2": 1.0,
                    "valid": True,
                }
            ]
        },
        {"weights": "same"},
        {"weights": "same"},
        1e-8,
        10.0,
        100.0,
    )
    assert report["status"] == "GPU_PREFLIGHT_NO_GO"
    assert "ALPHA_ZERO_LOGIT_MISMATCH" in report["failures"]
    assert "INEFFECTIVE_BASE_LEARNED_DOSE" in report["failures"]
    assert "BASE_LEARNED_GATE_NEVER_FIRED" in report["failures"]


def test_norm_audit_uses_actual_applied_output_difference() -> None:
    records = [
        {
            "stage": "technical",
            "prompt_id": "p",
            "candidate_component_id": 7,
            "intervention_model": "target",
            "intervention_arm": "target_c7_learned_a1p0",
            "intervention_kind": "learned",
            "alpha": 1.0,
            "applied_perturbation_l2": 2.0,
        },
        {
            "stage": "technical",
            "prompt_id": "p",
            "candidate_component_id": 7,
            "intervention_model": "target",
            "intervention_arm": "target_c7_orthogonal_a1p0",
            "intervention_kind": "orthogonal_matched",
            "alpha": 1.0,
            "applied_perturbation_l2": 2.1,
        },
    ]
    audit = protocol._norm_control_audit(records, tolerance=0.1)
    assert audit["invalid_count"] == 0
    assert audit["comparisons"][0]["relative_norm_error"] == pytest.approx(0.05)
    records[1]["applied_perturbation_l2"] = 3.0
    assert protocol._norm_control_audit(records, tolerance=0.1)["invalid_count"] == 1


def test_required_artifact_writer_is_immutable_and_checksummed(tmp_path) -> None:
    output = tmp_path / "job_1"
    output.mkdir()
    record = {
        "git_commit": "a" * 40,
        "pilot_config_sha256": "a" * 64,
        "frozen_aspd_config_sha256": "b" * 64,
        "canonical_rank0_checkpoint_sha256": "c" * 64,
        "prompt_file_sha256": "e" * 64,
        "behavioral_scoring_contract_sha256": "f" * 64,
        "target_model_id": "target",
        "target_model_revision": "d" * 40,
        "base_model_id": "base",
        "base_model_revision": "e" * 40,
        "gate_policy": "clean_shared_learned_JumpReLU_threshold",
        "learned_threshold": 3.0,
        "prompt_id": "risk-01",
        "prompt_category": "risk",
        "candidate_component_id": 1720,
        "intervention_arm": "target_c1720_learned_a1p0",
        "intervention_kind": "learned",
        "intervention_model": "target",
        "alpha": 1.0,
        "random_seed": None,
        "causal_comparison_valid": True,
        "clean_continuation_token_ids": [1],
        "clean_continuation": "clean",
        "patched_continuation_token_ids": [2],
        "patched_continuation": "patched",
    }
    protocol._write_run_artifacts(
        output_dir=output,
        manifest={"status": "GPU_PREFLIGHT_NO_GO"},
        preflight={"status": "PASS"},
        selected={
            "candidates": [],
            "behavioral_scoring_contract": "docs/multimodel/p6_100m_behavioral_scoring_contract.json",
            "behavioral_scoring_contract_sha256": sha256_file(
                "docs/multimodel/p6_100m_behavioral_scoring_contract.json"
            ),
            "behavioral_risk_score_cutoff": 2,
            "component_characterization": {
                "outcome_type": "OBSERVATIONAL_COMPONENT_CHARACTERIZATION",
                "components": [],
            },
        },
        records=[record],
        norm_audit={"comparisons": []},
        technical_summary={"status": "GPU_PREFLIGHT_NO_GO"},
    )
    required = {
        "run_manifest.json",
        "preflight_report.json",
        "selected_candidates.json",
        "component_characterization.json",
        "intervention_records.jsonl",
        "generation_records.jsonl",
        "control_norm_audit.json",
        "technical_summary.json",
        "behavioral_scoring_manifest.json",
        "behavioral_evaluation_records.jsonl",
        "behavioral_summary.json",
        "scientific_summary.md",
        "artifact_checksums.json",
    }
    assert {path.name for path in output.iterdir()} == required
    checksums = json.loads((output / "artifact_checksums.json").read_text())
    assert set(checksums["artifacts"]) == required - {"artifact_checksums.json"}
    evaluation = json.loads(
        (output / "behavioral_evaluation_records.jsonl").read_text()
    )
    assert evaluation["clean_continuation"] == "clean"
    assert evaluation["patched_continuation"] == "patched"
    assert evaluation["canonical_rank0_checkpoint_sha256"] == "c" * 64


def test_clean_reference_gate_rejects_different_base_target_prefixes() -> None:
    decomposition = _Anchor()
    residual = torch.ones(1, 2, 2)
    with pytest.raises(ValueError, match="identical"):
        protocol.clean_reference_gate(
            decomposition,
            base_token_ids=torch.tensor([[1, 2]]),
            target_token_ids=torch.tensor([[1, 3]]),
            clean_base_grounding=residual,
            clean_target_grounding=residual,
            valid_tokens=torch.ones(1, 2, dtype=torch.bool),
            component_ids=[0],
        )


def test_stage_b_accepts_only_exact_separate_preflight_go_provenance(tmp_path) -> None:
    pilot = protocol.load_p6_original_config(PILOT)
    handoff = {
        "checkpoint_sha256": pilot.expected_checkpoint_sha256,
    }
    git_head = "a" * 40
    report = {
        "status": "GPU_PREFLIGHT_GO",
        "checkpoint_sha256": pilot.expected_checkpoint_sha256,
        "config_file_sha256": pilot.expected_config_file_sha256,
        "prompt_file_sha256": pilot.prompt_sha256,
        "behavioral_scoring_contract_sha256": (
            pilot.behavioral_scoring_contract_sha256
        ),
        "pilot_config_sha256": sha256_file(PILOT),
        "git_commit": git_head,
        "technical_gate": {
            "status": "GPU_PREFLIGHT_GO",
            "estimated_pilot_runtime_seconds": 100.0,
            "pilot_runtime_budget_seconds": pilot.pilot_runtime_budget_seconds,
        },
    }
    path = tmp_path / "preflight_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    verified = protocol.validate_preflight_go(
        path,
        pilot_path=PILOT,
        pilot=pilot,
        handoff=handoff,
        git_head=git_head,
    )
    assert verified["source_sha256"] == sha256_file(path)

    report["prompt_file_sha256"] = "0" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt_file_sha256"):
        protocol.validate_preflight_go(
            path,
            pilot_path=PILOT,
            pilot=pilot,
            handoff=handoff,
            git_head=git_head,
        )


def test_p6_100m_blind_adapter_hides_experimental_metadata(tmp_path) -> None:
    generation = tmp_path / "generation"
    generation.mkdir()
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps(
            {
                "prompt_id": "risk-01",
                "category": "risk",
                "pair_id": "pair-01",
                "sport_domain": "climbing",
                "prompt": "Prompt text",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "JOB_COMPLETED_UNVERIFIED",
        "scientific_evidence_stage": "DISCOVERY",
        "canonical_rank0_checkpoint_sha256": "a" * 64,
        "prompt_file_sha256": sha256_file(prompts),
    }
    record = {
        "prompt_id": "risk-01",
        "intervention_arm": "target_c7_learned_a1p0",
        "intervention_kind": "learned",
        "intervention_model": "target",
        "candidate_component_id": 7,
        "alpha": 1.0,
        "patched_continuation": "Response text",
        "clean_continuation": "Clean text",
    }
    (generation / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (generation / "generation_records.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    (generation / "selected_candidates.json").write_text("{}\n", encoding="utf-8")
    artifacts = {}
    for name in (
        "run_manifest.json",
        "generation_records.jsonl",
        "selected_candidates.json",
    ):
        path = generation / name
        artifacts[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    (generation / "artifact_checksums.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
    blind, mapping, observed_manifest = _load_p6_100m_blind_rows(
        generation, prompts
    )
    assert set(blind[0]) == {"blind_id", "prompt", "response"}
    assert "candidate_component_id" not in blind[0]
    assert mapping[blind[0]["blind_id"]]["candidate_component_id"] == 7
    assert observed_manifest["scientific_evidence_stage"] == "DISCOVERY"


def test_p6_100m_judge_parser_and_base_target_differential_effect() -> None:
    parsed = parse_p6_100m_judgment(
        json.dumps(
            {
                "risk_score": 1,
                "risk_code": "NEUTRAL",
                "generic_refusal_overrefusal": 0,
                "useful_safe_sports_advice": 3,
                "coherence_general_helpfulness": 3,
                "task_relevance": 3,
                "incomplete_or_corrupted": False,
            }
        )
    )
    assert parsed["useful_safe_sports_advice"] == 3
    rows = []
    for prompt_id in ("risk-01", "risk-02"):
        for model_name, clean_score, learned_score in (
            ("base", 2, 2),
            ("target", 3, 1),
        ):
            common = {
                "parse_error": None,
                "prompt_id": prompt_id,
                "prompt_category": "risk",
                "intervention_model": model_name,
                "risk_score": clean_score,
                "candidate_component_id": None,
                "alpha": 0.0,
                "causal_comparison_valid": True,
            }
            rows.append({**common, "intervention_kind": "clean"})
            rows.append(
                {
                    **common,
                    "intervention_kind": "learned",
                    "candidate_component_id": 7,
                    "alpha": 1.0,
                    "risk_score": learned_score,
                }
            )
    effect = differential_intervention_effect(
        rows,
        component_id=7,
        alpha=1.0,
        score_field="risk_score",
        category="risk",
    )
    assert effect["base_effect"]["mean"] == 0.0
    assert effect["target_effect"]["mean"] == -2.0
    assert effect["differential_effect"]["mean"] == -2.0


def test_confirmation_is_blocked_and_discovery_prompts_cannot_be_reused() -> None:
    pilot = protocol.load_p6_original_config(PILOT)
    with pytest.raises(RuntimeError, match="pending researcher approval"):
        protocol._load_confirmatory_prompts(pilot)
    invalid = pilot.model_copy(
        update={
            "confirmation_approved": True,
            "confirmation_prompt_file": pilot.prompt_file,
            "confirmation_prompt_sha256": pilot.prompt_sha256,
        }
    )
    with pytest.raises(ValueError, match="cannot be reused"):
        protocol._load_confirmatory_prompts(invalid)


def test_p6_100m_report_requires_complete_two_model_arm_coverage(tmp_path) -> None:
    generation = tmp_path / "generation"
    generation.mkdir()
    manifest_path = generation / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "JOB_COMPLETED_UNVERIFIED",
                "scientific_evidence_stage": "DISCOVERY",
            }
        ),
        encoding="utf-8",
    )
    (generation / "component_characterization.json").write_text(
        json.dumps(
            {
                "components": [
                    {
                        "component_id": 7,
                        "risk_screening_rank": 1,
                        "effective_operator_comparison": {
                            "cosine_similarity": 0.5
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prompts = {
        "risk": "risk-01",
        "sports_control": "control-01",
        "neutral": "neutral-01",
    }
    generation_rows = []
    score_rows = []

    def append_row(model_name, category, kind, alpha, seed=None):
        prompt_id = prompts[category]
        arm = f"{model_name}_{kind}_{category}_a{alpha}_s{seed}"
        candidate_id = None if kind == "clean" else 7
        generation_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_category": category,
                "intervention_arm": arm,
                "intervention_kind": kind,
                "intervention_model": model_name,
                "candidate_component_id": candidate_id,
                "alpha": alpha,
                "random_seed": seed,
                "causal_comparison_valid": True,
                "applied_perturbation_l2": 0.0 if kind in {"clean", "gate_off"} else 1.0,
                "gate_firing_count": 0 if kind == "clean" else 1,
                "clean_continuation": "clean",
                "patched_continuation": "patched",
            }
        )
        risk = 3 if category == "risk" and model_name == "target" else 2
        if category != "risk":
            risk = 0
        if kind == "learned" and model_name == "target" and category == "risk":
            risk = 1
        score_rows.append(
            {
                **generation_rows[-1],
                "parse_error": None,
                "risk_score": risk,
                "generic_refusal_overrefusal": 0,
                "useful_safe_sports_advice": 3 if category == "sports_control" else 0,
                "coherence_general_helpfulness": 3,
                "task_relevance": 3,
            }
        )

    for model_name in ("base", "target"):
        for category in prompts:
            append_row(model_name, category, "clean", 0.0)
            for alpha in (0.25, 0.5, 1.0):
                for kind in (
                    "learned",
                    "gate_off",
                    "active_matched",
                    "orthogonal_matched",
                ):
                    append_row(model_name, category, kind, alpha)
                for seed in range(5):
                    append_row(model_name, category, "random_same_l2", alpha, seed)
    (generation / "generation_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in generation_rows),
        encoding="utf-8",
    )
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "scientific_evidence_stage": "DISCOVERY",
                "source_generation_manifest_sha256": sha256_file(manifest_path),
                "rows": score_rows,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report"
    summary = build_p6_100m_report(generation, scores_path, output)
    assert summary["status"] == "SCORED_NOT_INDEPENDENTLY_VERIFIED"
    assert summary["measurement_coverage"]["complete"] is True
    assert summary["measurement_coverage"]["missing_records"] == 0
    assert len(summary["candidates"]) == 1

    incomplete_scores = tmp_path / "incomplete_scores.json"
    incomplete_scores.write_text(
        json.dumps(
            {
                "scientific_evidence_stage": "DISCOVERY",
                "source_generation_manifest_sha256": sha256_file(manifest_path),
                "rows": score_rows[:-1],
            }
        ),
        encoding="utf-8",
    )
    incomplete = build_p6_100m_report(
        generation, incomplete_scores, tmp_path / "incomplete_report"
    )
    assert incomplete["status"] == "INCOMPLETE_BEHAVIORAL_MEASUREMENT"
    assert incomplete["measurement_coverage"]["missing_records"] == 1


def test_behavioral_summary_computes_frozen_effects_and_holm_adjustment(tmp_path) -> None:
    rows = []
    prompts = [
        (f"risk-{index:02d}", "risk") for index in range(1, 13)
    ] + [
        (f"control-{index:02d}", "sports_control") for index in range(1, 13)
    ] + [
        (f"neutral-{index:02d}", "neutral") for index in range(1, 13)
    ]

    def score_row(prompt_id, category, component, alpha, arm, kind, seed=None):
        learned_risk = kind == "learned" and category == "risk"
        return {
            "prompt_id": prompt_id,
            "prompt_category": category,
            "candidate_component_id": component,
            "intervention_arm": arm,
            "intervention_kind": kind,
            "alpha": alpha,
            "random_seed": seed,
            "causal_comparison_valid": True,
            "risky_action_endorsement": 2 if learned_risk else (3 if category == "risk" else 0),
            "generic_refusal_overrefusal": 0,
            "useful_safe_sports_advice": 3,
            "coherence_general_helpfulness": 3,
            "broader_behavior_change": None,
        }

    for prompt_id, category in prompts:
        rows.append(score_row(prompt_id, category, None, 0.0, "clean_target", "clean"))
    for component in (1720, 1173, 2228):
        for alpha in (0.25, 0.5, 1.0):
            tag = str(alpha).replace(".", "p")
            for prompt_id, category in prompts:
                rows.append(
                    score_row(
                        prompt_id, category, component, alpha,
                        f"c{component}_learned_a{tag}", "learned"
                    )
                )
                for kind in ("gate_off", "active_matched", "orthogonal_matched"):
                    rows.append(
                        score_row(
                            prompt_id, category, component, alpha,
                            f"c{component}_{kind}_a{tag}", kind
                        )
                    )
                for seed in range(5):
                    rows.append(
                        score_row(
                            prompt_id, category, component, alpha,
                            f"c{component}_random_seed_{seed}_a{tag}",
                            "random_same_l2", seed
                        )
                    )
    scores = tmp_path / "scores.jsonl"
    scores.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    template = tmp_path / "frozen_template.jsonl"
    template.write_text(scores.read_text(encoding="utf-8"), encoding="utf-8")
    summary = summarize_behavioral_scores(
        scores,
        "docs/multimodel/p6_100m_behavioral_scoring_contract.json",
        approved_scorer_id="test-rubric-v1",
        frozen_template_path=template,
    )
    assert summary["status"] == "SCORED_NOT_INDEPENDENTLY_VERIFIED"
    assert len(summary["tests"]) == 9
    assert summary["experiment_decision"] == "DISCOVERY_SIGNAL_REQUIRES_INDEPENDENT_CONFIRMATION"
    assert all(row["holm_adjusted_p_value"] <= 0.05 for row in summary["tests"])
    assert summary["frozen_template_sha256"] == sha256_file(template)

    rows[0]["prompt_category"] = "neutral"
    scores.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="changed frozen causal fields"):
        summarize_behavioral_scores(
            scores,
            "docs/multimodel/p6_100m_behavioral_scoring_contract.json",
            approved_scorer_id="test-rubric-v1",
            frozen_template_path=template,
        )
