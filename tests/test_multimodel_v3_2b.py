from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from torch import nn

from aspd.multimodel.cli.train_aspd_v3_2b import (
    _authorize_main500m_mode,
    _PoolPrefetcher,
    _record_continuation_authorization,
    _validate_main500m_alignment_manifest,
)
from aspd.multimodel.components import RankOneComponents
from aspd.multimodel.v3.data import SQLiteExposureLedger
from aspd.multimodel.v3.encoder import FeatureReadout, pack_cross_model_tokens
from aspd.multimodel.v3.gating import V3BatchTopK
from aspd.multimodel.v3.main500m_config import V31Main500MConfig, load_main500m_config
from aspd.multimodel.v3.main500m_continuation import (
    load_continuation_authorization,
    milestone_diagnostic_due,
    milestone_pause_due,
    milestone_scientific_summary,
    next_review_exposure,
)
from aspd.multimodel.v3.main500m_monitoring import (
    SustainedMain500MHealthMonitor,
    evaluate_main500m_health,
)
from aspd.multimodel.v3.main_2b_config import MODEL_ORDER_2B, load_main_2b_v1_config
from aspd.multimodel.v3.model_owned import (
    LocalCaptureBatch,
    ModelOwnedMain2B,
    build_model_owned_adam,
    model_owned_parameter_counts,
)
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.objective import auxk_dead_feature_loss
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, ModelPoolInputs
from aspd.multimodel.v3.reconstruction import ActivationDecoder
from aspd.multimodel.v3.registry import ModelFamilyLayout, build_model_registry_2b
from aspd.multimodel.v3.sparse_support import prepare_sparse_support
from aspd.multimodel.v3.training_capture import capture_family_pool

ROOT = Path(__file__).parents[1]


class _FakePool:
    def __init__(self, value: int):
        self.value = value

    def to_payload(self) -> dict[str, int]:
        return {"value": self.value}


class _FakeStream:
    def __init__(self, cursor: int = 0):
        self.cursor = cursor

    def next_pool(self, *, pool_anchors: int) -> _FakePool:
        assert pool_anchors == 2048
        value = self.cursor
        self.cursor += 1
        return _FakePool(value)


def _config():
    return load_main_2b_v1_config(
        ROOT / "configs/multimodel/aspd_v3/main_2b_v1.yaml"
    )


def _main500m_config():
    return load_main500m_config(
        ROOT / "configs/multimodel/aspd_v3/main500m_preparation_v1.yaml"
    )


def _pin_test_alignment_evidence(payload: dict, path: Path) -> None:
    payload["approvals"]["three_way_alignment"] = "APPROVED"
    payload["approvals"]["three_way_alignment_artifact"] = str(path)
    payload["approvals"]["three_way_alignment_artifact_sha256"] = "0" * 64
    provisional = V31Main500MConfig.model_validate(payload)
    report = {
        "status": "PASS",
        "execution_contract_fingerprint": (
            provisional.execution_contract_fingerprint()
        ),
        "alignment_policy": provisional.data.canonical_text_policy,
        "alignment_acceptance": provisional.data.alignment_acceptance,
        "model_order": list(provisional.architecture.model_order),
        "tokenizers": [
            {
                "model_key": model.key,
                "id": model.tokenizer_id,
                "revision": model.tokenizer_revision,
            }
            for model in provisional.models
        ],
        "audit": {
            "boundary_invariants": "PASS",
            "within_process_repeatability": "PASS",
            "dataset_stream_prefix_replay": "PASS",
        },
        "selection_bias": {"status": "PASS"},
        "capacity": {"status": "PASS"},
    }
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    payload["approvals"]["three_way_alignment_artifact_sha256"] = (
        hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_main500m_draft_preserves_zscore_math_and_fails_closed() -> None:
    config = _main500m_config()
    report = config.structural_report()
    assert report["logical_target_slots"] == 168
    assert report["approved_physical_targets"] == 168
    assert report["pending_physical_targets"] == 0
    assert config.objective.gate_logit_normalization == "per_anchor_feature_zscore"
    assert config.objective.gate_logit_normalization_epsilon == 1e-5
    assert config.objective.activation_objective_mode == "full_only"
    assert config.objective.activation_decoder_normalization == "none"
    assert config.objective.internal_code == "binary_support"
    assert config.objective.activation_code == "continuous"
    assert config.data.optimizer_steps == 61_036
    assert config.data.planned_accepted_anchor_exposures == 500_006_912
    assert config.runtime.startup_storage_required_bytes == 1_099_511_627_776
    assert config.runtime.checkpoint_root == config.runtime.output_root
    assert config.approvals.human_interpretability.status == (
        "PENDING_RESEARCHER_RESPONSES"
    )
    assert config.evaluation_status == (
        "HUMAN_INTERPRETABILITY_PACKAGE_COMPLETE_RESPONSES_PENDING"
    )
    assert not config.approvals.human_interpretability.permits_main_authorization()
    assert "MODEL_THREE_BLOCKED" not in report["blockers"]
    assert "MILESTONE_POLICY_APPROVAL_PENDING" in report["blockers"]
    with pytest.raises(RuntimeError, match="RESUME_BLOCKED"):
        config.require_execution_authorized()

    invalid = config.payload()
    invalid["approvals"]["human_interpretability"]["researcher_decision"] = "APPROVE"
    with pytest.raises(ValueError, match="cannot claim a decision"):
        V31Main500MConfig.model_validate(invalid)


def test_main500m_slurm_proposals_separate_preflight_and_main() -> None:
    main = (
        ROOT / "slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch"
    ).read_text(encoding="utf-8")
    preflight = (
        ROOT / "slurm/multimodel/aspd_v3_500m_preflight_PROPOSAL.sbatch"
    ).read_text(encoding="utf-8")

    for script in (main, preflight):
        assert "#SBATCH --account=thai" in script
        assert "#SBATCH --partition=hpg-b200" in script
        assert "#SBATCH --gres=gpu:b200:3" in script
        assert "ASPD_EXPECTED_GIT_SHA" in script
        assert "startup_storage_required_bytes" in script
        assert "git status --porcelain" in script

    assert "--mode main" in main
    assert "--require-ready" in main
    assert "--continuation-authorization" in main
    assert "LATEST_COMPLETE_CHECKPOINT" in main
    assert "  --mode preflight \\\n" not in main
    assert "#SBATCH --signal=B:USR1@300" in main
    assert "scontrol requeue" in main

    assert "--mode preflight" in preflight
    assert "--require-preflight-ready" in preflight
    assert "PREFLIGHT_STEPS > 20" in preflight
    assert "  --mode main \\\n" not in preflight
    assert "scontrol requeue" not in preflight


def test_main500m_runtime_rejects_checkpoint_root_different_from_run_root() -> None:
    payload = _main500m_config().payload()
    payload["runtime"]["checkpoint_root"] = str(
        Path(payload["runtime"]["output_root"]) / "separate-checkpoints"
    )
    with pytest.raises(ValueError, match="checkpoint_root must equal output_root"):
        V31Main500MConfig.model_validate(payload)


def test_main500m_human_gate_distinguishes_evidence_from_accepted_uncertainty(
    tmp_path: Path,
) -> None:
    payload = _main500m_config().payload()
    gate = payload["approvals"]["human_interpretability"]
    evaluation_path = tmp_path / "package_manifest.json"
    evaluation_path.write_text(
        json.dumps({"status": "HUMAN_INTERPRETABILITY_PACKAGE_COMPLETE"}),
        encoding="utf-8",
    )
    researcher_decision = (
        "Proceed while explicitly accepting single-rater semantic uncertainty."
    )
    decision_path = tmp_path / "researcher_decision.json"
    decision_payload = {
        "schema_version": 1,
        "status": "HUMAN_INTERPRETABILITY_RESEARCHER_DECISION",
        "decision": "ACCEPT_SEMANTIC_UNCERTAINTY",
        "researcher_decision": researcher_decision,
        "scientific_limitations": gate["scientific_limitations"],
        "evaluation_artifact_path": str(evaluation_path),
        "evaluation_artifact_sha256": hashlib.sha256(
            evaluation_path.read_bytes()
        ).hexdigest(),
    }
    decision_path.write_text(json.dumps(decision_payload), encoding="utf-8")
    gate.update(
        {
            "status": "APPROVED_SEMANTIC_UNCERTAINTY_ACCEPTED",
            "artifact_path": str(decision_path),
            "artifact_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
            "researcher_decision": researcher_decision,
        }
    )
    payload["evaluation_status"] = "SEMANTIC_UNCERTAINTY_EXPLICITLY_ACCEPTED"
    accepted = V31Main500MConfig.model_validate(payload)
    assert accepted.approvals.human_interpretability.permits_main_authorization()
    assert accepted.validate_human_interpretability_artifact() == decision_payload

    decision_payload["decision"] = "APPROVE_HUMAN_EVIDENCE"
    decision_payload["researcher_decision"] = "Approve based on completed human evidence."
    decision_path.write_text(json.dumps(decision_payload), encoding="utf-8")
    gate.update(
        {
            "status": "APPROVED_HUMAN_EVIDENCE",
            "artifact_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
            "researcher_decision": decision_payload["researcher_decision"],
        }
    )
    payload["evaluation_status"] = "HUMAN_INTERPRETABILITY_APPROVED"
    unsupported_claim = V31Main500MConfig.model_validate(payload)
    with pytest.raises(RuntimeError, match="evaluation_artifact.status"):
        unsupported_claim.validate_human_interpretability_artifact()

    gate["status"] = "REJECTED"
    gate["researcher_decision"] = "Do not proceed on current semantic evidence."
    payload["evaluation_status"] = "HUMAN_INTERPRETABILITY_REJECTED"
    rejected = V31Main500MConfig.model_validate(payload)
    assert not rejected.approvals.human_interpretability.permits_main_authorization()
    assert "HUMAN_INTERPRETABILITY_APPROVAL_PENDING" in rejected.readiness_blockers()


def test_main500m_bounded_preflight_has_independent_fail_closed_authorization(
    tmp_path: Path,
) -> None:
    draft = _main500m_config()
    with pytest.raises(RuntimeError, match="BOUNDED_PREFLIGHT_NOT_AUTHORIZED"):
        draft.require_preflight_authorized(
            optimizer_steps=4,
            run_dir=tmp_path / "unapproved",
        )

    payload = draft.payload()
    preflight_root = (
        tmp_path / "aspd_v3" / "main500m_preparation_v1" / "preflight"
    )
    payload["runtime"]["preflight_output_root"] = str(preflight_root)
    payload["runtime"]["preflight_evidence"] = str(
        preflight_root / "preflight_report.json"
    )
    _pin_test_alignment_evidence(payload, tmp_path / "alignment.json")
    payload["approvals"]["bounded_preflight"] = {
        "status": "AUTHORIZED",
        "optimizer_steps": 4,
        "output_namespace": str(preflight_root),
        "authorization_reference": "unit-test-only",
    }
    authorized = V31Main500MConfig.model_validate(payload)
    authorized.require_preflight_authorized(
        optimizer_steps=4,
        run_dir=preflight_root,
    )
    with pytest.raises(RuntimeError, match="step budget differs"):
        authorized.require_preflight_authorized(
            optimizer_steps=5,
            run_dir=preflight_root,
        )
    with pytest.raises(RuntimeError, match="dedicated output namespace"):
        authorized.require_preflight_authorized(
            optimizer_steps=4,
            run_dir=preflight_root.parent / "wrong",
        )
    preflight_root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="refuses to overwrite"):
        authorized.require_preflight_authorized(
            optimizer_steps=4,
            run_dir=preflight_root,
        )


def test_main500m_mode_gate_cannot_cross_preflight_and_main_boundaries(
    tmp_path: Path,
) -> None:
    draft = _main500m_config()
    main_args = SimpleNamespace(
        mode="main",
        strategy="model_owned",
        skip_checkpoint=False,
        resume_checkpoint=None,
        max_steps=61_036,
        run_dir=tmp_path / "main",
    )
    with pytest.raises(RuntimeError, match="RESUME_BLOCKED"):
        _authorize_main500m_mode(draft, main_args)

    skip_args = copy.copy(main_args)
    skip_args.skip_checkpoint = True
    with pytest.raises(RuntimeError, match="cannot disable checkpoint recovery"):
        _authorize_main500m_mode(draft, skip_args)

    invalid_mode = copy.copy(main_args)
    invalid_mode.mode = "benchmark"
    with pytest.raises(RuntimeError, match="only preflight or main"):
        _authorize_main500m_mode(draft, invalid_mode)


def test_main500m_preflight_evidence_is_pinned_and_semantically_validated(
    tmp_path: Path,
) -> None:
    payload = _main500m_config().payload()
    preflight_root = (
        tmp_path / "aspd_v3" / "main500m_preparation_v1" / "preflight"
    )
    evidence_path = preflight_root / "preflight_report.json"
    payload["runtime"]["preflight_output_root"] = str(preflight_root)
    payload["runtime"]["preflight_evidence"] = str(evidence_path)
    payload["approvals"]["three_way_alignment"] = "APPROVED"
    payload["approvals"]["bounded_preflight"] = {
        "status": "AUTHORIZED",
        "optimizer_steps": 4,
        "output_namespace": str(preflight_root),
        "authorization_reference": "unit-test-only",
    }
    contract = V31Main500MConfig.model_validate(payload)
    contract_fingerprint = contract.execution_contract_fingerprint()
    payload["approvals"]["three_model_gpu_preflight"] = {
        "status": "PASSED",
        "artifact_path": str(evidence_path),
        "artifact_sha256": "0" * 64,
        "execution_contract_fingerprint": contract_fingerprint,
        "optimizer_steps": 4,
    }
    missing = V31Main500MConfig.model_validate(payload)
    with pytest.raises(RuntimeError, match="ARTIFACT_MISSING"):
        missing.validate_preflight_evidence()

    preflight_root.mkdir(parents=True)
    report = {
        "status": "PASS",
        "mode": "preflight",
        "strategy": "model_owned",
        "world_size": 3,
        "logical_target_count": 168,
        "optimizer_steps": 4,
        "execution_contract_fingerprint": contract_fingerprint,
        "dcp_save_reload_resume_update": "PASS",
        "checkpoint_parameter_and_adam_restore_probe": "PASS",
        "source_correctness": [
            {
                "status": "PASS",
                "targets": 56,
                "frozen_parameters": "PASS",
            }
            for _ in range(3)
        ],
        "checkpoint_manifest": {"status": "COMPLETE", "world_size": 3},
    }
    evidence_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    payload["approvals"]["three_model_gpu_preflight"]["artifact_sha256"] = (
        hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    valid = V31Main500MConfig.model_validate(payload)
    assert valid.validate_preflight_evidence() == report

    report["logical_target_count"] = 167
    evidence_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    payload["approvals"]["three_model_gpu_preflight"]["artifact_sha256"] = (
        hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    invalid = V31Main500MConfig.model_validate(payload)
    with pytest.raises(RuntimeError, match="EVIDENCE_INVALID"):
        invalid.validate_preflight_evidence()


def test_main500m_source_contract_rejects_qwen() -> None:
    payload = _main500m_config().payload()
    payload["models"][2]["model_id"] = "Qwen/forbidden-new-experiment"
    with pytest.raises(ValueError, match="Qwen is forbidden"):
        V31Main500MConfig.model_validate(payload)


def test_main500m_approved_granite_contract_is_immutable() -> None:
    payload = _main500m_config().payload()
    payload["models"][2]["selected_layers"] = list(range(15, 23))
    with pytest.raises(ValueError, match="approved Granite model-3 contract changed"):
        V31Main500MConfig.model_validate(payload)


def test_main500m_three_way_alignment_evidence_is_pinned_and_validated(
    tmp_path: Path,
) -> None:
    config = _main500m_config()
    report = config.validate_three_way_alignment_evidence()
    assert report["status"] == "PASS"
    assert report["audit"]["boundary_invariants"] == "PASS"

    payload = config.payload()
    invalid_path = tmp_path / "invalid_alignment.json"
    invalid_report = copy.deepcopy(report)
    invalid_report["audit"]["dataset_stream_prefix_replay"] = "FAIL"
    invalid_path.write_text(json.dumps(invalid_report), encoding="utf-8")
    payload["approvals"]["three_way_alignment_artifact"] = str(invalid_path)
    payload["approvals"]["three_way_alignment_artifact_sha256"] = hashlib.sha256(
        invalid_path.read_bytes()
    ).hexdigest()
    invalid = V31Main500MConfig.model_validate(payload)
    with pytest.raises(RuntimeError, match="THREE_WAY_ALIGNMENT_EVIDENCE_INVALID"):
        invalid.validate_three_way_alignment_evidence()


def test_three_model_zscore_and_separator_contract_matches_notation() -> None:
    torch.manual_seed(23)
    model_order = ("gemma", "smol", "third")
    tokens = {key: torch.randn(2, 3, 8, 16) for key in model_order}
    packed = pack_cross_model_tokens(tokens, model_order=model_order)
    assert packed.values.shape == (2, 3, 26, 21)
    assert packed.separator_positions == (8, 17)
    assert packed.identity_indices == (
        *(0 for _ in range(8)),
        3,
        *(1 for _ in range(8)),
        4,
        *(2 for _ in range(8)),
    )

    readout = FeatureReadout(
        21,
        32,
        "mean",
        logit_normalization="per_anchor_feature_zscore",
        logit_normalization_epsilon=1e-5,
    )
    valid = torch.tensor([[True, True, False], [True, False, True]])
    scores, normalized = readout.forward_with_logits(packed.values, valid)
    raw = readout.latest_projection_logits
    assert raw is not None
    mean = raw.mean(dim=-1, keepdim=True)
    variance = (raw - mean).square().mean(dim=-1, keepdim=True)
    expected = (raw - mean) / torch.sqrt(variance + 1e-5)
    expected = expected * valid[..., None]
    torch.testing.assert_close(normalized, expected)
    torch.testing.assert_close(scores, torch.relu(expected))


def test_main500m_health_monitor_separates_alerts_from_hard_aborts() -> None:
    thresholds = _main500m_config().monitoring
    alert = evaluate_main500m_health(
        {
            "repeated_exposures": 0,
            "loss": {
                "total": 1.0,
                "diagnostic/gate/pre/gemma/positive_scores_per_anchor": 128.0,
                "diagnostic/gate/pre/gemma/effective_dictionary_size_in_pool": 900.0,
                "diagnostic/gate/pre/gemma/top_1_percent_firing_share_in_pool": 0.6,
                "diagnostic/gate/pre/gemma/top_10_percent_firing_share_in_pool": 0.9,
                "diagnostic/gate/pre/gemma/recently_dead_features": 7000.0,
            },
            "gradient_norms_by_rank": [{"rank": 0, "l2": {"rank_one": 1.0}}],
        },
        thresholds,
    )
    assert alert["status"] == "ALERT"
    assert {event["code"] for event in alert["events"]} >= {
        "GATE_CANDIDATE_COLLAPSE",
        "DICTIONARY_CONCENTRATION",
        "RECENT_FEATURE_DEATH",
    }

    abort = evaluate_main500m_health(
        {"repeated_exposures": 1, "loss": {"total": float("nan")}}, thresholds
    )
    assert abort["status"] == "ABORT"
    assert {event["code"] for event in abort["events"]} == {
        "EXPOSURE_DUPLICATION",
        "NONFINITE_OBJECTIVE",
    }


def _complete_main500m_diagnostic(
    *,
    diagnostic_step: int,
    accepted_exposures: int,
    gate_collapse: bool = False,
    reconstruction_regression: bool = False,
) -> dict:
    loss: dict[str, float] = {"total": 1.0}
    for site in ("pre", "mid"):
        for model in ("gemma", "smol", "third"):
            prefix = f"diagnostic/gate/{site}/{model}"
            loss.update(
                {
                    f"{prefix}/projection_negative_logit_fraction": 0.5,
                    f"{prefix}/normalized_negative_logit_fraction": 0.5,
                    f"{prefix}/positive_scores_per_anchor": (
                        128.0 if gate_collapse else 4096.0
                    ),
                    f"{prefix}/positive_firings_per_anchor": 32.0,
                    f"{prefix}/active_features_in_pool": (
                        500 if gate_collapse else 4096
                    ),
                    f"{prefix}/effective_dictionary_size_in_pool": (
                        900.0 if gate_collapse else 4096.0
                    ),
                    f"{prefix}/top_1_percent_firing_share_in_pool": (
                        0.60 if gate_collapse else 0.10
                    ),
                    f"{prefix}/top_10_percent_firing_share_in_pool": (
                        0.90 if gate_collapse else 0.40
                    ),
                    f"{prefix}/dead_features": 0.0,
                    f"{prefix}/recently_dead_features": (
                        7000.0 if gate_collapse else 0.0
                    ),
                }
            )
            for window in range(8):
                loss[
                    f"diagnostic/activation/{site}/{model}/window_{window}/full_fvu"
                ] = 0.90 if reconstruction_regression else 0.50
    for model in ("gemma", "smol", "third"):
        for target in range(56):
            loss[f"loss/internal/{model}.target_{target}"] = (
                0.90 if reconstruction_regression else 0.50
            )
    return {
        "accepted_exposures": accepted_exposures,
        "repeated_exposures": 0,
        "diagnostic_parameter_step": diagnostic_step,
        "ordered_anchor_ids_sha256": "c" * 64,
        "loss": loss,
        "gradient_norms_by_rank": [
            {"rank": rank, "l2": {"shared_encoder": 1.0, "rank_one": 1.0}}
            for rank in range(3)
        ],
    }


def test_main500m_health_monitor_fails_closed_on_incomplete_scheduled_diagnostics() -> None:
    result = evaluate_main500m_health(
        {
            "accepted_exposures": 8192,
            "repeated_exposures": 0,
            "diagnostic_parameter_step": 0,
            "loss": {"total": 1.0},
            "gradient_norms_by_rank": [],
        },
        _main500m_config().monitoring,
    )
    assert result["status"] == "ABORT"
    assert {event["code"] for event in result["events"]} == {"MISSING_DIAGNOSTICS"}


def test_main500m_sustained_gate_failure_checkpoints_then_pauses_and_resumes_state() -> None:
    thresholds = _main500m_config().monitoring
    monitor = SustainedMain500MHealthMonitor(thresholds)
    healthy = monitor.observe(
        _complete_main500m_diagnostic(diagnostic_step=0, accepted_exposures=8192)
    )
    assert healthy["status"] == "PASS"
    assert healthy["action"] == "CONTINUE"

    first = monitor.observe(
        _complete_main500m_diagnostic(
            diagnostic_step=64,
            accepted_exposures=65 * 8192,
            gate_collapse=True,
        )
    )
    assert first["status"] == "ALERT"
    second = monitor.observe(
        _complete_main500m_diagnostic(
            diagnostic_step=128,
            accepted_exposures=129 * 8192,
            gate_collapse=True,
        )
    )
    assert second["status"] == "ALERT"

    resumed = SustainedMain500MHealthMonitor(thresholds)
    resumed.load_state_dict(monitor.state_dict())
    third = resumed.observe(
        _complete_main500m_diagnostic(
            diagnostic_step=192,
            accepted_exposures=193 * 8192,
            gate_collapse=True,
        )
    )
    assert third["status"] == "PAUSE"
    assert third["action"] == "CHECKPOINT_AND_PAUSE"
    assert "SUSTAINED_GATE_HEALTH_FAILURE" in {
        event["code"] for event in third["events"]
    }


def test_main500m_reconstruction_pause_starts_only_after_first_review_milestone() -> None:
    thresholds = _main500m_config().monitoring
    monitor = SustainedMain500MHealthMonitor(thresholds)
    for diagnostic_step in (0, 64):
        early = monitor.observe(
            _complete_main500m_diagnostic(
                diagnostic_step=diagnostic_step,
                accepted_exposures=(diagnostic_step + 1) * 8192,
                reconstruction_regression=True,
            )
        )
        assert early["status"] == "ALERT"
        assert early["action"] == "CONTINUE"
        assert early["monitor_state"]["reconstruction_failure_streak"] == 0

    first = monitor.observe(
        _complete_main500m_diagnostic(
            diagnostic_step=1221,
            accepted_exposures=10_002_432,
            reconstruction_regression=True,
        )
    )
    assert first["status"] == "ALERT"
    second = monitor.observe(
        _complete_main500m_diagnostic(
            diagnostic_step=1280,
            accepted_exposures=10_493_952,
            reconstruction_regression=True,
        )
    )
    assert second["status"] == "PAUSE"
    assert "SUSTAINED_RECONSTRUCTION_FAILURE" in {
        event["code"] for event in second["events"]
    }


def test_main500m_milestones_are_exact_step_boundaries_and_final_is_not_a_pause() -> None:
    milestones = _main500m_config().monitoring.staged_milestone_exposures
    assert all(exposures % 8192 == 0 for exposures in milestones)
    assert milestone_diagnostic_due(milestones[0], milestones)
    assert milestone_pause_due(milestones[0], milestones)
    assert milestone_diagnostic_due(milestones[-1], milestones)
    assert not milestone_pause_due(milestones[-1], milestones)
    assert next_review_exposure(milestones[0], milestones) == milestones[1]
    with pytest.raises(ValueError, match="beyond the final"):
        next_review_exposure(milestones[-1], milestones)


def test_main500m_continuation_authorization_is_checkpoint_and_contract_pinned(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authorizations"
    root.mkdir()
    path = root / "continue_after_10m.json"
    payload = {
        "schema_version": 1,
        "status": "APPROVED",
        "decision": "CONTINUE",
        "policy_version": "v1",
        "reasons": ["MILESTONE"],
        "accepted_exposures": 10_002_432,
        "approved_through_exposures": 25_001_984,
        "execution_contract_fingerprint": "a" * 64,
        "checkpoint_manifest_sha256": "b" * 64,
        "researcher": "unit-test researcher",
        "decided_at": "2026-11-09T20:00:00-05:00",
        "scientific_limitations_acknowledged": "synthetic unit-test decision only",
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    validated = load_continuation_authorization(
        path,
        authorization_root=root,
        pending_review={
            "policy_version": "v1",
            "reasons": ["MILESTONE"],
            "accepted_exposures": 10_002_432,
        },
        execution_contract_fingerprint="a" * 64,
        checkpoint_manifest_sha256="b" * 64,
        approved_through_exposures=25_001_984,
    )
    assert validated["artifact_path"] == str(path.resolve())
    assert len(validated["artifact_sha256"]) == 64

    payload["checkpoint_manifest_sha256"] = "d" * 64
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_continuation_authorization(
            path,
            authorization_root=root,
            pending_review={
                "policy_version": "v1",
                "reasons": ["MILESTONE"],
                "accepted_exposures": 10_002_432,
            },
            execution_contract_fingerprint="a" * 64,
            checkpoint_manifest_sha256="b" * 64,
            approved_through_exposures=25_001_984,
        )


def test_main500m_continuation_audit_log_is_idempotent_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "continuation_authorizations.jsonl"
    record = {
        "accepted_exposures": 10_002_432,
        "artifact_sha256": "a" * 64,
        "decision": "CONTINUE",
    }
    _record_continuation_authorization(path, record)
    _record_continuation_authorization(path, record)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    conflicting = {**record, "artifact_sha256": "b" * 64}
    with pytest.raises(ValueError, match="conflicting continuation"):
        _record_continuation_authorization(path, conflicting)


def test_main500m_milestone_report_preserves_complete_scientific_diagnostics() -> None:
    metrics = _complete_main500m_diagnostic(
        diagnostic_step=1220,
        accepted_exposures=10_002_432,
    )
    metrics["health"] = {"status": "PASS", "events": []}
    summary = milestone_scientific_summary(metrics)
    assert summary["accepted_exposures"] == 10_002_432
    assert summary["activation_full_fvu_summary"]["count"] == 48
    assert summary["internal_fvu_summary"]["count"] == 168
    assert len(summary["gate_metrics"]) == 60


def test_approved_third_model_contract_registers_exactly_168_unique_targets() -> None:
    payload = _main500m_config().payload()
    config = V31Main500MConfig.model_validate(payload)
    registry = build_model_registry_2b(config)
    targets = [target for layout in registry.values() for target in layout.targets()]
    assert tuple(registry) == ("gemma", "smol", "granite")
    assert len(targets) == len({target.key for target in targets}) == 168
    assert {target.logical_name for target in targets} == {
        "q",
        "k",
        "v",
        "o",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    }


def test_main500m_alignment_manifest_uses_the_canonical_policy_identifier() -> None:
    payload = _main500m_config().payload()
    config = V31Main500MConfig.model_validate(payload)
    manifest = {
        "schema_version": 4,
        "dataset_manifest_sha256": "d" * 64,
        "model_order": list(config.architecture.model_order),
        "canonical_text": "identity",
        "alignment": "exact_fast_tokenizer_offsets_and_exact_prefix_roundtrip_v2",
        "alignment_acceptance": "intersection_of_exact_canonical_text_boundaries",
        "sequence_length_native_tokens_per_model": 512,
        "tokenizers": [
            {
                "model_key": model.key,
                "tokenizer_id": model.tokenizer_id,
                "tokenizer_revision": model.tokenizer_revision,
            }
            for model in config.models
        ],
    }
    _validate_main500m_alignment_manifest(
        config,
        manifest,
        dataset_manifest_sha256="d" * 64,
    )
    manifest["alignment"] = "identity_exact_three_way_prefix"
    with pytest.raises(ValueError, match="alignment manifest contract mismatch"):
        _validate_main500m_alignment_manifest(
            config,
            manifest,
            dataset_manifest_sha256="d" * 64,
        )


def test_prefetch_checkpoint_preserves_exact_pool_order() -> None:
    stream = _FakeStream()
    prefetcher = _PoolPrefetcher(stream, pool_anchors=2048)
    prefetcher.start()
    assert prefetcher.next_payload() == {"value": 0}
    buffered = prefetcher.checkpoint_payload()
    assert buffered == {"value": 1}
    assert stream.cursor == 2
    prefetcher.close()

    resumed_stream = _FakeStream(cursor=2)
    resumed = _PoolPrefetcher(
        resumed_stream, pool_anchors=2048, buffered_payload=buffered
    )
    resumed.start()
    assert resumed.next_payload() == {"value": 1}
    assert resumed.next_payload() == {"value": 2}
    resumed.close()

    final_stream = _FakeStream()
    final_prefetcher = _PoolPrefetcher(final_stream, pool_anchors=2048)
    final_prefetcher.start()
    assert final_prefetcher.next_payload(schedule_next=False) == {"value": 0}
    assert final_stream.cursor == 1
    assert final_prefetcher.checkpoint_payload() is None
    final_prefetcher.close()


def test_aligned_pool_json_checkpoint_preserves_model_order() -> None:
    model = ModelPoolInputs(
        input_ids=((1,),),
        attention_masks=((1,),),
        anchor_document_indices=(0,),
        anchor_token_indices=(0,),
    )
    pool = AlignedAnchorPool(
        anchor_ids=("anchor",),
        document_ids=("document",),
        models={"qwen": model, "gemma": model, "smol": model},
        source_observations=({},),
    )
    canonical_payload = json.loads(json.dumps(pool.to_payload(), sort_keys=True))
    restored = AlignedAnchorPool.from_payload(canonical_payload)

    assert tuple(restored.models) == ("qwen", "gemma", "smol")
    restored.validate(
        expected_anchors=1,
        sequence_length=1,
        model_order=("qwen", "gemma", "smol"),
    )


def test_aligned_pool_restores_pre_explicit_order_checkpoint() -> None:
    model_payload = {
        "input_ids": [[1]],
        "attention_masks": [[1]],
        "anchor_document_indices": [0],
        "anchor_token_indices": [0],
    }
    legacy_payload = {
        "anchor_ids": ["anchor"],
        "document_ids": ["document"],
        "models": {"gemma": model_payload, "qwen": model_payload, "smol": model_payload},
        "source_observations": [{}],
    }

    restored = AlignedAnchorPool.from_payload(legacy_payload)

    assert tuple(restored.models) == ("qwen", "gemma", "smol")


def test_main_2b_contract_and_registry_are_incompatible_and_complete() -> None:
    config = _config()
    registry = build_model_registry_2b(config)
    assert tuple(registry) == MODEL_ORDER_2B
    assert config.full_training_authorized is True
    assert config.data.minimum_optimizer_steps == 61_036
    assert config.data.minimum_accepted_anchor_exposures == 500_006_912
    assert config.runtime.sparse_backend == "dense"
    assert config.runtime.ledger_anchor_storage == "exact_compact_v1"
    assert config.runtime.preflight_evidence.endswith(
        "/preflight/final_authorized_v1/preflight_report.json"
    )
    assert sum(len(layout.targets()) for layout in registry.values()) == 168
    assert registry["qwen"].layers == tuple(range(10, 18))
    assert registry["gemma"].layers == tuple(range(9, 17))
    assert registry["smol"].layers == tuple(range(8, 16))
    assert registry["gemma"].residual_mid_suffix == "pre_feedforward_layernorm"
    assert registry["smol"].kv_out == 2048
    counts = model_owned_parameter_counts(config, registry)
    assert counts["rank_one_total"] == 8_908_701_696
    assert counts["activation_decoder_total"] == 838_963_200
    assert {value["model_id"] for value in config.payload()["models"]} == {
        "Qwen/Qwen3-1.7B",
        "google/gemma-2-2b-it",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    }


def test_batch_topk_places_all_state_on_requested_device() -> None:
    gate = V3BatchTopK(16, 2, pool_anchors=4, device="cpu")
    assert {value.device.type for value in gate.buffers()} == {"cpu"}


def test_prepared_rank_one_support_matches_reference_forward_and_gradients() -> None:
    torch.manual_seed(7)
    reference = RankOneComponents(11, 5, 4, unit_norm=True)
    optimized = RankOneComponents(11, 5, 4, unit_norm=True)
    csr = RankOneComponents(11, 5, 4, unit_norm=True)
    dense = RankOneComponents(11, 5, 4, unit_norm=True)
    optimized.load_state_dict(reference.state_dict())
    csr.load_state_dict(reference.state_dict())
    dense.load_state_dict(reference.state_dict())
    x_reference = torch.randn(2, 3, 5, requires_grad=True)
    x_optimized = x_reference.detach().clone().requires_grad_(True)
    x_csr = x_reference.detach().clone().requires_grad_(True)
    x_dense = x_reference.detach().clone().requires_grad_(True)
    code = torch.zeros(2, 3, 11)
    code.scatter_(-1, torch.tensor([[[0, 3], [1, 4], [2, 8]]] * 2), 1.0)
    valid = torch.tensor([[True, True, False], [True, False, True]])
    support = prepare_sparse_support(code, valid, binary=True)
    expected = reference(x_reference, code * valid[..., None], valid)
    actual = optimized.forward_sparse(
        x_optimized,
        support.position,
        support.feature,
        support.coefficient,
        compute_dtype=torch.float32,
    )
    csr_actual = csr.forward_sparse_csr(
        x_csr,
        support.crow_indices,
        support.feature,
        support.coefficient,
        compute_dtype=torch.float32,
    )
    dense_actual = dense.forward_dense_masked(
        x_dense,
        (code > 0).to(code.dtype),
        valid,
        compute_dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(csr_actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(dense_actual, expected, rtol=1e-6, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    csr_actual.square().sum().backward()
    dense_actual.square().sum().backward()
    torch.testing.assert_close(x_optimized.grad, x_reference.grad)
    torch.testing.assert_close(optimized.U.grad, reference.U.grad)
    torch.testing.assert_close(optimized.V.grad, reference.V.grad)
    torch.testing.assert_close(x_csr.grad, x_reference.grad)
    torch.testing.assert_close(csr.U.grad, reference.U.grad)
    torch.testing.assert_close(csr.V.grad, reference.V.grad)
    torch.testing.assert_close(x_dense.grad, x_reference.grad)
    torch.testing.assert_close(dense.U.grad, reference.U.grad)
    torch.testing.assert_close(dense.V.grad, reference.V.grad)


def test_indexed_rank_one_support_respects_bfloat16_autocast_compute_dtype() -> None:
    torch.manual_seed(8)
    component = RankOneComponents(11, 5, 4, unit_norm=True)
    x = torch.randn(1, 3, 5, dtype=torch.bfloat16, requires_grad=True)
    code = torch.zeros(1, 3, 11, dtype=torch.bfloat16)
    code.scatter_(-1, torch.tensor([[[0, 3], [1, 4], [2, 8]]]), 1.0)
    valid = torch.ones(1, 3, dtype=torch.bool)
    support = prepare_sparse_support(code, valid, binary=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = component.forward_sparse(
            x,
            support.position,
            support.feature,
            support.coefficient,
            compute_dtype=torch.bfloat16,
        )
        output.float().square().mean().backward()
    assert output.dtype == torch.bfloat16
    assert component.U.grad is not None and component.U.grad.dtype == torch.float32
    assert component.V.grad is not None and component.V.grad.dtype == torch.float32


def test_dense_rank_one_accepts_frozen_inference_capture() -> None:
    torch.manual_seed(18)
    component = RankOneComponents(11, 5, 4, unit_norm=True)
    with torch.inference_mode():
        captured = torch.randn(1, 3, 5)
    code = torch.zeros(1, 3, 11)
    code.scatter_(-1, torch.tensor([[[0, 3], [1, 4], [2, 8]]]), 1.0)
    output = component.forward_dense_masked(
        captured,
        code,
        torch.ones(1, 3, dtype=torch.bool),
        compute_dtype=torch.float32,
    )
    output.square().mean().backward()
    assert component.U.grad is not None
    assert component.V.grad is not None


def test_prepared_activation_support_matches_matryoshka_decoder() -> None:
    torch.manual_seed(9)
    reference = ActivationDecoder(16, 7)
    optimized = ActivationDecoder(16, 7)
    csr = ActivationDecoder(16, 7)
    dense = ActivationDecoder(16, 7)
    optimized.load_state_dict(reference.state_dict())
    csr.load_state_dict(reference.state_dict())
    dense.load_state_dict(reference.state_dict())
    code_reference = torch.zeros(1, 4, 16, requires_grad=True)
    with torch.no_grad():
        code_reference.scatter_(-1, torch.tensor([[[0, 7], [1, 8], [2, 9], [3, 10]]]), 0.5)
    code_optimized = code_reference.detach().clone().requires_grad_(True)
    code_csr = code_reference.detach().clone().requires_grad_(True)
    code_dense = code_reference.detach().clone().requires_grad_(True)
    valid = torch.tensor([[True, True, False, True]])
    boundaries = (0, 4, 8, 12, 16)
    support = prepare_sparse_support(code_optimized, valid, binary=False)
    csr_support = prepare_sparse_support(code_csr, valid, binary=False)
    expected = reference(code_reference * valid[..., None], boundaries=boundaries)
    actual = optimized.forward_sparse(support, boundaries=boundaries)
    csr_actual = csr.forward_sparse_csr(csr_support, boundaries=boundaries)
    dense_gate = (code_reference.detach() != 0) & valid[..., None]
    dense_actual = dense.forward_dense_grouped(
        code_dense * dense_gate, boundaries=boundaries
    )
    assert (
        isinstance(expected, tuple)
        and isinstance(actual, tuple)
        and isinstance(csr_actual, tuple)
        and isinstance(dense_actual, tuple)
    )
    for left, right, csr_right, dense_right in zip(
        expected, actual, csr_actual, dense_actual, strict=True
    ):
        torch.testing.assert_close(left, right)
        torch.testing.assert_close(left, csr_right, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(left, dense_right, rtol=1e-6, atol=1e-6)
    sum(value.square().sum() for value in expected).backward()
    sum(value.square().sum() for value in actual).backward()
    sum(value.square().sum() for value in csr_actual).backward()
    sum(value.square().sum() for value in dense_actual).backward()
    torch.testing.assert_close(optimized.weight.grad, reference.weight.grad)
    torch.testing.assert_close(optimized.bias.grad, reference.bias.grad)
    torch.testing.assert_close(csr.weight.grad, reference.weight.grad)
    torch.testing.assert_close(csr.bias.grad, reference.bias.grad)
    torch.testing.assert_close(code_csr.grad, code_reference.grad)
    torch.testing.assert_close(dense.weight.grad, reference.weight.grad)
    torch.testing.assert_close(dense.bias.grad, reference.bias.grad)
    torch.testing.assert_close(code_dense.grad, code_reference.grad)


def test_dense_auxk_matches_indexed_loss_and_gradients() -> None:
    torch.manual_seed(21)
    indexed = ActivationDecoder(16, 7)
    dense = ActivationDecoder(16, 7)
    dense.load_state_dict(indexed.state_dict())
    indexed_scores = torch.rand(1, 4, 16, requires_grad=True)
    dense_scores = indexed_scores.detach().clone().requires_grad_(True)
    target = torch.randn(1, 4, 7)
    reconstruction = torch.randn(1, 4, 7)
    valid = torch.tensor([[True, True, False, True]])
    dead = torch.tensor([True] * 10 + [False] * 6)
    indexed_loss = auxk_dead_feature_loss(
        indexed_scores,
        indexed,
        target,
        reconstruction,
        valid,
        dead,
        top_k_aux=4,
        known_dead_feature_count=10,
        sparse_backend="indexed",
    )
    dense_loss = auxk_dead_feature_loss(
        dense_scores,
        dense,
        target,
        reconstruction,
        valid,
        dead,
        top_k_aux=4,
        known_dead_feature_count=10,
        sparse_backend="dense",
    )
    torch.testing.assert_close(dense_loss, indexed_loss, rtol=1e-6, atol=1e-6)
    indexed_loss.backward()
    dense_loss.backward()
    torch.testing.assert_close(dense_scores.grad, indexed_scores.grad)
    torch.testing.assert_close(dense.weight.grad, indexed.weight.grad)
    assert indexed.bias.grad is None and dense.bias.grad is None


def test_split_gradient_projection_and_post_step_normalization_match_reference() -> None:
    torch.manual_seed(13)
    reference = RankOneComponents(7, 5, 6, unit_norm=True)
    optimized = RankOneComponents(7, 5, 6, unit_norm=True)
    optimized.load_state_dict(reference.state_dict())
    gradient = torch.randn_like(reference.U)
    reference.U.grad = gradient.clone()
    optimized.U.grad = gradient.clone()
    reference.normalize_write_decoder_(project_gradients=True)
    optimized.project_write_decoder_gradient_()
    torch.testing.assert_close(optimized.U, reference.U, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(optimized.U.grad, reference.U.grad, rtol=1e-6, atol=1e-6)
    with torch.no_grad():
        reference.U.add_(reference.U.grad, alpha=-1e-3)
        optimized.U.add_(optimized.U.grad, alpha=-1e-3)
    reference.normalize_write_decoder_(project_gradients=False)
    optimized.normalize_write_decoder_(project_gradients=False)
    torch.testing.assert_close(optimized.U, reference.U, rtol=1e-6, atol=1e-6)


def test_exposure_ledger_records_smol_as_immutable_third_tokenizer_slot(tmp_path: Path) -> None:
    ledger = SQLiteExposureLedger(
        tmp_path / "ledger.sqlite",
        dataset_manifest_sha256="a" * 64,
        world_size=3,
        model_order=MODEL_ORDER_2B,
    )
    ledger.observe_raw_tokens("row-0", {"qwen": 11, "gemma": 12, "smol": 13})
    assert ledger.raw_token_counts() == {"qwen": 11, "gemma": 12, "smol": 13}
    ledger.close()
    reopened = SQLiteExposureLedger(
        tmp_path / "ledger.sqlite",
        dataset_manifest_sha256="a" * 64,
        world_size=3,
        model_order=MODEL_ORDER_2B,
    )
    assert reopened.raw_token_counts()["smol"] == 13
    reopened.close()


def test_compact_exposure_ledger_preserves_exact_membership_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "compact.sqlite"
    ledger = SQLiteExposureLedger(
        path,
        dataset_manifest_sha256="b" * 64,
        world_size=3,
        model_order=MODEL_ORDER_2B,
        compact_anchor_ids=True,
    )
    anchors = ("1" * 64, "2" * 64, "non-hex-anchor")
    first = ledger.observe_replicated_optimizer_step(0, anchors)
    receipt = ledger.checkpoint_receipt(expected_optimizer_steps=1)
    ledger.observe_replicated_optimizer_step(1, ("2" * 64, "3" * 64))
    recovered = ledger.recover_to_receipt(receipt)
    assert recovered == first
    assert ledger.connection.execute(
        "SELECT COUNT(*) FROM seen_anchors WHERE typeof(anchor_id)='blob'"
    ).fetchone() == (3,)
    ledger.close()

    reopened = SQLiteExposureLedger(
        path,
        dataset_manifest_sha256="b" * 64,
        world_size=3,
        model_order=MODEL_ORDER_2B,
        compact_anchor_ids=True,
    )
    assert reopened.checkpoint_receipt(expected_optimizer_steps=1) == receipt
    reopened.close()


class _ToyAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(value) + self.k_proj(value) + self.v_proj(value))


class _ToyMlp(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_proj(value) * self.up_proj(value))


class _ToyLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.self_attn = _ToyAttention(width)
        self.mlp = _ToyMlp(width)
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        value = value + self.self_attn(self.input_layernorm(value))
        return value + self.mlp(self.post_attention_layernorm(value))


class _ToyCausal(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_ToyLayer(4) for _ in range(10)])
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, input_ids, attention_mask, **_kwargs):
        value = self.embed(input_ids)
        for layer in self.model.layers:
            value = layer(value)
        return SimpleNamespace(last_hidden_state=value)


def _toy_layout() -> ModelFamilyLayout:
    return ModelFamilyLayout(
        key="toy",
        family="llama",
        model_id="toy",
        revision="0" * 40,
        tokenizer_id="toy",
        tokenizer_revision="0" * 40,
        layers=tuple(range(8)),
        layer_prefixes=("model.layers",),
        residual_pre_suffix="input_layernorm",
        residual_mid_suffix="post_attention_layernorm",
        hidden_size=4,
        intermediate_size=4,
        num_hidden_layers=10,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        q_out=4,
        kv_out=4,
        fused_qkv=False,
        fused_gate_up=False,
    )


def test_early_exit_capture_matches_full_capture_and_skips_later_layers() -> None:
    torch.manual_seed(2)
    full_model = _ToyCausal()
    early_model = _ToyCausal()
    early_model.load_state_dict(full_model.state_dict())
    pool = ModelPoolInputs(
        input_ids=((1, 2, 3),),
        attention_masks=((1, 1, 1),),
        anchor_document_indices=(0, 0),
        anchor_token_indices=(0, 2),
    )
    layout = _toy_layout()
    full = capture_family_pool(
        full_model, layout, pool, pad_token_id=0, device=torch.device("cpu")
    )
    early = capture_family_pool(
        early_model,
        layout,
        pool,
        pad_token_id=0,
        device=torch.device("cpu"),
        early_exit=True,
    )
    for full_values, early_values in (
        (full.residual_pre, early.residual_pre),
        (full.residual_mid, early.residual_mid),
    ):
        for left, right in zip(full_values, early_values, strict=True):
            torch.testing.assert_close(left, right)
    for key in full.matrix_inputs:
        torch.testing.assert_close(full.matrix_inputs[key], early.matrix_inputs[key])
        torch.testing.assert_close(full.matrix_outputs[key], early.matrix_outputs[key])
    assert [layer.calls for layer in full_model.model.layers[-2:]] == [1, 1]
    assert [layer.calls for layer in early_model.model.layers[-2:]] == [0, 0]


class _TinyConfig:
    def __init__(self):
        self.architecture = SimpleNamespace(
            dictionary_size=16,
            batch_topk_k=2,
            pool_tokens=2,
            latent_tokens_per_model=8,
            packed_token_width=8,
            intra_model_layers=1,
            intra_model_heads=2,
            cross_model_layers=1,
            cross_model_heads=2,
            cross_model_width=8,
            feature_readout="mean",
        )
        self.optimization = SimpleNamespace(
            rank_one_lr=3e-4,
            activation_decoder_lr=3e-4,
            encoder_lr=1e-4,
            betas=(0.9, 0.99),
            weight_decay=0.0,
        )

    @staticmethod
    def fingerprint() -> str:
        return "tiny-model-owned-test"


def _tiny_registry() -> dict[str, ModelFamilyLayout]:
    return {
        key: ModelFamilyLayout(
            key=key,
            family="llama",
            model_id=f"test/{key}",
            revision="0" * 40,
            tokenizer_id=f"test/{key}",
            tokenizer_revision="0" * 40,
            layers=tuple(range(8)),
            layer_prefixes=("model.layers",),
            residual_pre_suffix="input_layernorm",
            residual_mid_suffix="post_attention_layernorm",
            hidden_size=4,
            intermediate_size=6,
            num_hidden_layers=8,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            q_out=4,
            kv_out=4,
            fused_qkv=False,
            fused_gate_up=False,
        )
        for key in MODEL_ORDER_2B
    }


def _tiny_capture(layout: ModelFamilyLayout, rank: int) -> LocalCaptureBatch:
    generator = torch.Generator().manual_seed(100 + rank)
    inputs = {
        target.key: torch.randn(1, 2, target.d_in, generator=generator)
        for target in layout.targets()
    }
    outputs = {
        target.key: torch.randn(1, 2, target.d_out, generator=generator)
        for target in layout.targets()
    }
    return LocalCaptureBatch(
        model_key=layout.key,
        activations={
            site: tuple(
                torch.randn(1, 2, layout.hidden_size, generator=generator)
                for _ in range(8)
            )
            for site in ("pre", "mid")
        },
        matrix_inputs=inputs,
        matrix_outputs=outputs,
        valid_anchors=torch.ones(1, 2, dtype=torch.bool),
    )


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def _distributed_model_owned_worker(
    rank: int, world_size: int, init_file: str, checkpoint_root: str
) -> None:
    import torch.distributed as dist

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    torch.manual_seed(0)
    config = _TinyConfig()
    registry = _tiny_registry()
    key = MODEL_ORDER_2B[rank]
    model = ModelOwnedMain2B(config, registry, key, dtype=torch.float32)
    model.synchronize_shared_state_()
    optimizer = build_model_owned_adam(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    capture = _tiny_capture(registry[key], rank)

    def update() -> None:
        optimizer.zero_grad(set_to_none=True)
        scores, sparse, continuous, binary = model.encode(capture)
        for site in ("pre", "mid"):
            for state in model.gates[site].state_dict().values():
                gathered = [torch.empty_like(state) for _ in range(world_size)]
                dist.all_gather(gathered, state)
                assert all(torch.equal(gathered[0], value) for value in gathered[1:])
        activation_terms = list(
            model.activation_loss_terms(capture, scores, sparse, continuous)
        )
        internal_terms = list(model.internal_loss_terms(capture, binary))
        loss = sum(
            coefficient * (activation + 0.03125 * auxk)
            for _name, activation, auxk, coefficient in activation_terms
        ) + sum(
            coefficient * value for _name, _site, value, coefficient in internal_terms
        )
        loss.backward()
        model.all_reduce_shared_gradients_()
        assert all(parameter.grad is not None for parameter in model.shared_encoders.parameters())
        assert all(parameter.grad is not None for parameter in model.local_encoders.parameters())
        assert all(parameter.grad is not None for parameter in model.activation_banks.parameters())
        assert all(parameter.grad is not None for parameter in model.matrix_bank.parameters())
        model.project_decoder_gradients_()
        optimizer.step()
        model.normalize_decoders_(project_gradients=False)
        scheduler.step()

    update()
    manager = ModelOwnedCheckpoint(checkpoint_root, keep_complete=2)
    saved_parameter = next(model.parameters()).detach().clone()
    recovery_root = Path(checkpoint_root).parent / "live-recovery"
    recovery_databases = None
    if rank == 0:
        recovery_root.mkdir()
        recovery_databases = {}
        for name, value in (
            ("exposure_ledger.sqlite", "ledger-step-1"),
            ("stream_dedup.sqlite", "stream-step-1"),
        ):
            path = recovery_root / name
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE state(value TEXT NOT NULL)")
            connection.execute("INSERT INTO state VALUES (?)", (value,))
            connection.commit()
            connection.close()
            recovery_databases[name] = path
    dist.barrier()
    manager.save(
        step=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        runtime_state={"completed_steps": 1, "accepted_exposures": 8192},
        config_fingerprint="tiny-model-owned-test",
        config_file_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        model_key=key,
        experiment_metadata={"activation_target_normalization": "none"},
        recovery_databases=recovery_databases,
    )
    checkpoint = Path(checkpoint_root) / "step-000000001"
    if rank == 0:
        restored_root = Path(checkpoint_root).parent / "external-resume"
        manager.restore_recovery_databases(
            path=checkpoint,
            destination_root=restored_root,
            config_fingerprint="tiny-model-owned-test",
            config_file_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            expected_world_size=world_size,
        )
        for name, expected in (
            ("exposure_ledger.sqlite", "ledger-step-1"),
            ("stream_dedup.sqlite", "stream-step-1"),
        ):
            connection = sqlite3.connect(restored_root / name)
            assert connection.execute("SELECT value FROM state").fetchone() == (expected,)
            connection.close()
    dist.barrier()

    # Compare one uninterrupted update with the exact same update after external-state reload.
    update()
    uninterrupted_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    uninterrupted_optimizer = copy.deepcopy(optimizer.state_dict())
    uninterrupted_scheduler = copy.deepcopy(scheduler.state_dict())
    manager.load(
        path=checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config_fingerprint="tiny-model-owned-test",
        config_file_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        expected_model_key=key,
    )
    update()
    _assert_nested_equal(model.state_dict(), uninterrupted_model)
    _assert_nested_equal(optimizer.state_dict(), uninterrupted_optimizer)
    _assert_nested_equal(scheduler.state_dict(), uninterrupted_scheduler)

    with torch.no_grad():
        next(model.parameters()).add_(1)
    runtime, manifest = manager.load(
        path=checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config_fingerprint="tiny-model-owned-test",
        config_file_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        expected_model_key=key,
    )
    torch.testing.assert_close(next(model.parameters()), saved_parameter)
    assert runtime["completed_steps"] == 1
    assert manifest["status"] == "COMPLETE"
    assert manifest["experiment_metadata"]["activation_target_normalization"] == "none"
    with torch.no_grad():
        next(model.parameters()).sub_(2)
    runtime, manifest = manager.load_model_only(
        path=checkpoint,
        model=model,
        config_fingerprint="tiny-model-owned-test",
        config_file_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        expected_model_key=key,
    )
    torch.testing.assert_close(next(model.parameters()), saved_parameter)
    assert runtime["completed_steps"] == 1
    assert manifest["status"] == "COMPLETE"
    with torch.no_grad():
        next(model.parameters()).add_(3)
    runtime, manifest = manager.load_model_only_owner(
        path=checkpoint,
        model=model,
        config_fingerprint="tiny-model-owned-test",
        config_file_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        checkpoint_rank=rank,
        expected_checkpoint_world_size=world_size,
        expected_model_key=key,
    )
    torch.testing.assert_close(next(model.parameters()), saved_parameter)
    assert runtime["completed_steps"] == 1
    assert manifest["world_size"] == world_size
    dist.barrier()
    if rank == 0:
        corrupt = Path(checkpoint_root).parent / "corrupt-checkpoint"
        shutil.copytree(checkpoint, corrupt)
        with (corrupt / "recovery" / "exposure_ledger.sqlite").open("ab") as handle:
            handle.write(b"corruption")
        with pytest.raises(ValueError, match="missing or corrupt"):
            manager.restore_recovery_databases(
                path=corrupt,
                destination_root=Path(checkpoint_root).parent / "corrupt-restore",
                config_fingerprint="tiny-model-owned-test",
                config_file_sha256="a" * 64,
                dataset_manifest_sha256="b" * 64,
                expected_world_size=world_size,
            )
    dist.barrier()
    dist.destroy_process_group()


def test_model_owned_distributed_gradients_optimizer_and_checkpoint(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    init_file = tmp_path / f"dist-{uuid4().hex}"
    mp.spawn(
        _distributed_model_owned_worker,
        args=(3, str(init_file), str(tmp_path / "checkpoint")),
        nprocs=3,
        join=True,
    )
