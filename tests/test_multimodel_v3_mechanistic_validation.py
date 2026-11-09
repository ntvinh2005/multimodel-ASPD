from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from aspd.multimodel.cli.analyze_aspd_v3_causal import (
    _bootstrap_mean,
    _component_rows,
)
from aspd.multimodel.cli.analyze_aspd_v3_intruder_final import (
    _bootstrap_group as _bootstrap_final_intruder_group,
)
from aspd.multimodel.cli.analyze_aspd_v3_intruder_final import (
    _parents as _final_intruder_parents,
)
from aspd.multimodel.cli.analyze_aspd_v3_mechanistic import (
    _feature_anchor_lists,
    _intruder_task,
    _stratified_features,
)
from aspd.multimodel.cli.extract_aspd_v3_mechanistic import _save_site
from aspd.multimodel.cli.judge_aspd_v3_intruder import (
    _control_tasks,
    _parse_response,
    _rotate_task,
    _summary,
)
from aspd.multimodel.cli.judge_aspd_v3_intruder_coherence import (
    _omission_task,
)
from aspd.multimodel.cli.judge_aspd_v3_intruder_coherence import (
    _parent_rows as _coherence_parent_rows,
)
from aspd.multimodel.cli.judge_aspd_v3_intruder_coherence import (
    _summary as _coherence_summary,
)
from aspd.multimodel.cli.judge_aspd_v3_intruder_final import (
    _expand_task as _expand_final_task,
)
from aspd.multimodel.cli.judge_aspd_v3_intruder_final import (
    _parse_response as _parse_final_response,
)
from aspd.multimodel.cli.judge_aspd_v3_intruder_final import (
    _score as _score_final_intruder,
)
from aspd.multimodel.cli.judge_aspd_v3_intruder_final import (
    _summary as _summary_final_intruder,
)
from aspd.multimodel.cli.judge_aspd_v3_intruder_loo import (
    _ordered_remaining,
    _parse_score,
    _select_intruder,
)
from aspd.multimodel.cli.judge_aspd_v3_intruder_loo import (
    _summary as _loo_summary,
)
from aspd.multimodel.cli.plan_aspd_v3_causal import _fires, _select_components
from aspd.multimodel.cli.prepare_aspd_v3_intruder_final import _text_jaccard
from aspd.multimodel.cli.run_aspd_v3_causal_screen import (
    _metric_rows,
    _random_component_factors,
)
from aspd.multimodel.v3.mechanistic_validation import (
    CausalValidationContract,
    FinalIntruderCoherenceJudgeContract,
    FinalIntruderJudgeContract,
    FinalIntruderValidationContract,
    IntruderJudgeContract,
    IntruderJudgeV2Contract,
    IntruderLeaveOneOutJudgeContract,
    MechanisticValidationContract,
    norm_matched_random_rank_one,
    pairwise_firing_statistic,
    rank_one_operator,
    temporary_rank_one_ablation,
)
from aspd.multimodel.v3.registry import LogicalMatrixTarget

ROOT = Path(__file__).parents[1]


def _target() -> LogicalMatrixTarget:
    return LogicalMatrixTarget(
        model_key="gemma",
        family="fixture",
        layer_index=9,
        logical_name="q",
        grounding_kind="residual_pre",
        grounding_module_candidates=("grounding",),
        physical_module_candidates=("projection",),
        output_slice=None,
        d_in=3,
        d_out=2,
        bias_policy="physical_if_present",
    )


class _FixtureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grounding = nn.Identity()
        self.projection = nn.Linear(3, 2, bias=False)


def test_contract_is_frozen_to_untouched_test_split() -> None:
    contract = MechanisticValidationContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_mechanistic_validation_v1.yaml"
    )
    assert contract.selected_split == "test"
    assert contract.test_buckets == tuple(range(10, 20))
    assert contract.training_authorized_by_this_contract is False
    assert len(contract.fingerprint()) == 64

    judge = IntruderJudgeContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_judge_v1.yaml"
    )
    assert judge.paid_api is False
    assert "qwen" not in judge.judge_model_id.lower()
    assert judge.cyclic_rotations == 5

    judge_v2 = IntruderJudgeV2Contract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_judge_v2_mistral.yaml"
    )
    assert judge_v2.prompt_version == "abstention_calibrated_v2"
    assert judge_v2.control_rotations == 5
    assert "qwen" not in judge_v2.judge_model_id.lower()

    final = FinalIntruderValidationContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_final_validation_v1.yaml"
    )
    assert final.excluded_document_count == 296
    assert final.screening_features_per_arm_site == 100
    assert final.training_authorized_by_this_contract is False
    final_judge = FinalIntruderJudgeContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_final_judge_v1_llama.yaml"
    )
    assert "qwen" not in final_judge.judge_model_id.lower()
    assert final_judge.candidate_labels == ("K", "L", "M", "N", "P")
    coherence_judge = FinalIntruderCoherenceJudgeContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_coherence_judge_v1_llama.yaml"
    )
    assert "qwen" not in coherence_judge.judge_model_id.lower()
    assert coherence_judge.remaining_context_order == "ascending_utf8_sha256"


def test_pairwise_firing_statistic_uses_marginal_independence_reference() -> None:
    result = pairwise_firing_statistic(
        (0, 2, 4, 8),
        (1, 2, 4, 7),
        n_anchors=10,
        left_feature=3,
        right_feature=9,
    )
    assert result.intersection == 2
    assert result.union == 6
    assert result.jaccard == pytest.approx(1 / 3)
    assert result.independent_intersection == pytest.approx(1.6)
    assert result.excess_coactivation_rate == pytest.approx(0.04)


def test_norm_matched_random_rank_one_is_seeded_and_exact() -> None:
    first_u, first_v = norm_matched_random_rank_one(
        d_out=7, d_in=5, target_frobenius_norm=2.5, seed=11
    )
    second_u, second_v = norm_matched_random_rank_one(
        d_out=7, d_in=5, target_frobenius_norm=2.5, seed=11
    )
    torch.testing.assert_close(first_u, second_u, rtol=0, atol=0)
    torch.testing.assert_close(first_v, second_v, rtol=0, atol=0)
    assert float(rank_one_operator(first_u, first_v).norm()) == pytest.approx(2.5)


def test_temporary_rank_one_ablation_edits_equation_and_restores_bitwise() -> None:
    model = _FixtureModel()
    original = model.projection.weight.detach().clone()
    u = torch.tensor([2.0, -1.0])
    v = torch.tensor([0.5, 1.0, -2.0])
    with temporary_rank_one_ablation(model, _target(), u, v, strength=0.5) as metadata:
        expected = original.float() - 0.5 * torch.outer(u, v)
        torch.testing.assert_close(model.projection.weight.float(), expected)
        assert metadata["target"] == "gemma.L09.q"
        assert metadata["realized_weight_delta_frobenius_norm"] > 0.0
        assert metadata["realized_to_intended_norm_ratio"] == pytest.approx(1.0)
        assert metadata["realized_to_intended_cosine"] == pytest.approx(1.0)
        assert metadata["realized_nonzero_fraction"] == pytest.approx(1.0)
    assert torch.equal(model.projection.weight, original)


def test_temporary_rank_one_ablation_restores_after_exception() -> None:
    model = _FixtureModel()
    original = model.projection.weight.detach().clone()
    with pytest.raises(RuntimeError, match="probe failed"):
        with temporary_rank_one_ablation(
            model,
            _target(),
            torch.ones(2),
            torch.ones(3),
            strength=1.0,
        ):
            raise RuntimeError("probe failed")
    assert torch.equal(model.projection.weight, original)


def test_sparse_extraction_artifact_is_valid_csr_and_fail_closed(tmp_path: Path) -> None:
    retained = [
        {
            "doc-a": {
                "activation": 2.0,
                "anchor_id": "anchor-a",
                "document_id": "doc-a",
            }
        },
        {},
        {},
    ]
    report = _save_site(
        tmp_path,
        "pre",
        row_counts=[np.asarray([1, 2], dtype=np.int16)],
        indices=[np.asarray([0, 1, 2], dtype=np.uint16)],
        values=[np.asarray([2.0, 1.0, 3.0], dtype=np.float32)],
        retained=retained,
        accepted_anchors=2,
    )
    artifact = np.load(tmp_path / "pre_natural_firings.npz")
    assert artifact["indptr"].tolist() == [0, 1, 3]
    assert artifact["indices"].tolist() == [0, 1, 2]
    assert artifact["shape"].tolist() == [2, 3]
    assert report["natural_firing_events"] == 3
    assert report["features_firing"] == 3

    with pytest.raises(ValueError, match="row count"):
        _save_site(
            tmp_path / "unused",
            "mid",
            row_counts=[np.asarray([1], dtype=np.int16)],
            indices=[np.asarray([0], dtype=np.uint16)],
            values=[np.asarray([1.0], dtype=np.float32)],
            retained=retained,
            accepted_anchors=2,
        )


def test_feature_sampling_is_frequency_stratified_and_deterministic() -> None:
    indptr = np.asarray([0, 2, 4, 6, 9], dtype=np.int64)
    indices = np.asarray([0, 1, 0, 2, 0, 3, 0, 1, 3], dtype=np.int64)
    firing, counts = _feature_anchor_lists(indptr, indices, n_features=4)
    assert counts.tolist() == [4, 2, 1, 2]
    assert firing[0].tolist() == [0, 1, 2, 3]
    documents = np.asarray([4, 2, 1, 2])
    first = _stratified_features(
        counts,
        documents,
        minimum_fires=1,
        minimum_documents=1,
        sample_size=4,
        seed=17,
    )
    second = _stratified_features(
        counts,
        documents,
        minimum_fires=1,
        minimum_documents=1,
        sample_size=4,
        seed=17,
    )
    assert first == second
    assert {row["frequency_stratum"] for row in first[0]} == {"low", "medium", "high"}


def test_intruder_task_is_blinded_and_uses_disjoint_document() -> None:
    sampled = [
        {"feature": 1, "frequency_stratum": "medium", "fires": 10, "unique_documents": 4},
        {"feature": 2, "frequency_stratum": "medium", "fires": 9, "unique_documents": 4},
    ]
    contexts = {
        1: [
            {"context": f"shared pattern {index}", "document_id": f"target-{index}", "source_group": "books"}
            for index in range(4)
        ],
        2: [
            {"context": "different pattern", "document_id": "donor", "source_group": "books"},
            {
                "context": "different example",
                "document_id": "donor-tie",
                "source_group": "books",
            },
        ],
    }
    result = _intruder_task(
        arm="per_anchor_zscore",
        site="pre",
        selected=sampled[0],
        sampled=sampled,
        contexts=contexts,
        rng=np.random.default_rng(5),
        examples=5,
    )
    assert result is not None
    task, answer = result
    assert set(task) == {"schema_version", "trial_id", "instruction", "examples"}
    assert "arm" not in task and "feature" not in task and "activation" not in task
    assert answer["intruder_document_id"] == "donor"
    assert answer["intruder_document_id"] not in answer["target_document_ids"]


def test_intruder_rotations_balance_every_answer_position() -> None:
    task = {
        "trial_id": "trial",
        "examples": [
            {"position": index + 1, "text": f"example-{index}"} for index in range(5)
        ],
    }
    answer = {"trial_id": "trial", "intruder_position": 2}
    rotated = [_rotate_task(task, answer, rotation) for rotation in range(5)]
    assert sorted(row[1]["intruder_position"] for row in rotated) == [1, 2, 3, 4, 5]
    for rotated_task, rotated_answer in rotated:
        expected_text = f"example-{answer['intruder_position'] - 1}"
        observed = rotated_task["examples"][rotated_answer["intruder_position"] - 1]["text"]
        assert observed == expected_text


def test_final_intruder_rotations_balance_position_and_opaque_label() -> None:
    task = {
        "trial_id": "parent",
        "examples": [
            {"candidate_id": f"C{index + 1}", "text": f"example-{index + 1}"}
            for index in range(5)
        ],
    }
    labels = ("K", "L", "M", "N", "P")
    rotations = [_expand_final_task(task, rotation, labels) for rotation in range(5)]
    for canonical_position in range(1, 6):
        occurrences = [
            row
            for rotation in rotations
            for row in rotation["examples"]
            if row["canonical_position"] == canonical_position
        ]
        assert sorted(row["displayed_position"] for row in occurrences) == [1, 2, 3, 4, 5]
        assert sorted(row["label"] for row in occurrences) == sorted(labels)


def test_final_intruder_answer_key_is_not_needed_to_render_or_parse() -> None:
    task = {
        "trial_id": "parent",
        "examples": [
            {"candidate_id": f"C{index + 1}", "text": f"example-{index + 1}"}
            for index in range(5)
        ],
    }
    expanded = _expand_final_task(task, 0, ("K", "L", "M", "N", "P"))
    assert "correct" not in json.dumps(expanded)
    assert _parse_final_response(
        '{"intruder_id":"M","reason":"different semantic topic"}',
        ("K", "L", "M", "N", "P"),
    )["valid"]
    assert not _parse_final_response(
        '{"intruder_id":"A","reason":"bad label"}',
        ("K", "L", "M", "N", "P"),
    )["valid"]
    assert not _parse_final_response(
        '{"intruder_id":"M","reason":"ok","extra":1}',
        ("K", "L", "M", "N", "P"),
    )["valid"]


def test_final_intruder_scoring_and_parent_aggregation_preserve_rotations() -> None:
    contract = FinalIntruderJudgeContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_final_judge_v1_llama.yaml"
    )
    task = {
        "trial_id": "positive-parent",
        "control_kind": "positive",
        "examples": [
            {"candidate_id": f"C{index + 1}", "text": f"example-{index + 1}"}
            for index in range(5)
        ],
    }
    expanded = [
        _expand_final_task(task, rotation, contract.candidate_labels)
        for rotation in range(5)
    ]
    raw = []
    for row in expanded:
        correct = next(
            example for example in row["examples"] if example["canonical_position"] == 3
        )
        raw.append(
            json.dumps(
                {"intruder_id": correct["label"], "reason": "known positive control"}
            )
        )
    scored = _score_final_intruder(
        expanded,
        raw,
        [
            {
                "trial_id": "positive-parent",
                "control_kind": "positive",
                "intruder_position": 3,
            }
        ],
        contract,
    )
    summary = _summary_final_intruder(
        mode="calibration",
        rows=scored,
        replay_exact=True,
        contract=contract,
    )
    assert all(row["correct"] for row in scored)
    assert summary["positive_control_accuracy"] == 1.0
    assert summary["positive_parent_consistency"]["fraction_at_least_0_8"] == 1.0


def test_final_intruder_normalized_jaccard_rejects_near_duplicates() -> None:
    assert _text_jaccard("The planet follows an orbit.", "A planet follows the orbit!") > 0.5
    assert _text_jaccard("stellar orbit", "bread fermentation") == 0.0


def test_final_intruder_bootstrap_resamples_parent_features_not_rotations() -> None:
    answer = {
        "trial_id": "parent",
        "frequency_stratum": "low",
        "feature_fires": 10,
    }
    rotations = [
        {
            "parent_trial_id": "parent",
            "rotation": rotation,
            "correct": rotation < 2,
            "valid": True,
        }
        for rotation in range(5)
    ]
    parents = _final_intruder_parents([answer], rotations)
    assert parents[0]["rotation_accuracy"] == 0.4
    draws = _bootstrap_final_intruder_group(
        [
            {**parents[0], "frequency_stratum": stratum}
            for stratum in ("low", "medium", "high")
        ],
        {"low": 1, "medium": 1, "high": 1},
        rng=np.random.default_rng(7),
        replicates=20,
    )
    assert draws.tolist() == pytest.approx([0.4] * 20)


def test_coherence_omission_hides_original_position_and_hash_sorts_remaining() -> None:
    task = {
        "trial_id": "parent",
        "examples": [
            {"candidate_id": f"C{index + 1}", "text": text}
            for index, text in enumerate(("zeta", "alpha", "gamma", "beta", "delta"))
        ],
    }
    omitted = _omission_task(task, 3)
    assert omitted["omitted_canonical_position"] == 3
    assert all("omitted" not in row for row in omitted["remaining"])
    hashes = [
        hashlib.sha256(row["text"].encode()).hexdigest()
        for row in omitted["remaining"]
    ]
    assert hashes == sorted(hashes)


def test_coherence_parent_argmax_and_calibration_gates() -> None:
    contract = FinalIntruderCoherenceJudgeContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_coherence_judge_v1_llama.yaml"
    )
    task = {
        "trial_id": "positive-parent",
        "control_kind": "positive",
        "examples": [
            {"candidate_id": f"C{index + 1}", "text": f"example-{index + 1}"}
            for index in range(5)
        ],
    }
    expanded = [_omission_task(task, omitted) for omitted in range(1, 6)]
    outputs = [
        json.dumps(
            {
                "coherence": 95 if omitted == 3 else 20 + omitted,
                "theme": "fixture",
            }
        )
        for omitted in range(1, 6)
    ]
    prompt_rows, parents = _coherence_parent_rows(
        expanded,
        outputs,
        [
            {
                "trial_id": "positive-parent",
                "control_kind": "positive",
                "intruder_position": 3,
            }
        ],
        contract,
    )
    summary = _coherence_summary(
        mode="calibration",
        prompt_rows=prompt_rows,
        parents=parents,
        replay_exact=True,
        contract=contract,
    )
    assert parents[0]["unique_argmax"] and parents[0]["correct"]
    assert summary["positive_control_accuracy"] == 1.0
    assert summary["status"] == "JUDGE_VALIDATED"


def test_judge_parser_and_controls_fail_closed() -> None:
    assert _parse_response(
        '{"intruder_position": 3, "confidence": 2, "reason": "different topic"}'
    )["valid"]
    assert _parse_response(
        '{"intruder_position": null, "confidence": 0, "reason": "no unique item"}'
    )["valid"]
    assert not _parse_response("not json")["valid"]
    assert not _parse_response(
        '{"intruder_position": 8, "confidence": 3, "reason": "bad"}'
    )["valid"]

    contract = IntruderJudgeContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_judge_v1.yaml"
    )
    tasks, answers = _control_tasks(contract)
    assert len(tasks) == len(answers) == 40
    assert sum(row["control_kind"] == "positive" for row in tasks) == 20
    assert all(
        answer["intruder_position"] is None
        for answer in answers
        if answer["control_kind"] == "negative"
    )


def test_judge_v2_controls_are_semantically_matched_and_rotatable() -> None:
    contract = IntruderJudgeV2Contract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_judge_v2_mistral.yaml"
    )
    tasks, answers = _control_tasks(contract)
    negative_pairs = [
        (task, answer)
        for task, answer in zip(tasks, answers, strict=True)
        if task["control_kind"] == "negative"
    ]
    assert len(negative_pairs) == 20
    assert all(answer["intruder_position"] is None for _, answer in negative_pairs)
    assert all(
        " concept example " not in example["text"]
        for task, _ in negative_pairs
        for example in task["examples"]
    )
    rotated = [
        _rotate_task(negative_pairs[0][0], negative_pairs[0][1], rotation)
        for rotation in range(contract.control_rotations)
    ]
    assert all(answer["intruder_position"] is None for _, answer in rotated)
    assert all(task["control_kind"] == "negative" for task, _ in rotated)


def test_judge_v2_summary_uses_rotation_consensus_as_parent_unit() -> None:
    contract = IntruderJudgeV2Contract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_judge_v2_mistral.yaml"
    )
    rows = []
    for index in range(20):
        for rotation in range(5):
            correct = ((index % 5 - rotation) % 5) + 1
            rows.append(
                {
                    "base_trial_id": f"positive-{index}",
                    "rotation": rotation,
                    "control_kind": "positive",
                    "correct_position": correct,
                    "parsed": {
                        "valid": True,
                        "intruder_position": correct,
                        "confidence": 3,
                    },
                }
            )
            rows.append(
                {
                    "base_trial_id": f"negative-{index}",
                    "rotation": rotation,
                    "control_kind": "negative",
                    "correct_position": None,
                    "parsed": {
                        "valid": True,
                        "intruder_position": None,
                        "confidence": 0,
                    },
                }
            )
    for rotation in range(5):
        displayed = ((2 - 1 - rotation) % 5) + 1
        rows.append(
            {
                "base_trial_id": "scientific-parent",
                "rotation": rotation,
                "control_kind": None,
                "correct_position": displayed,
                "parsed": {
                    "valid": True,
                    "intruder_position": displayed,
                    "confidence": 3,
                },
            }
        )
    summary = _summary(rows, contract)
    assert summary["status"] == "JUDGE_VALIDATED"
    assert summary["rotation_consensus"]["eligible_parent_count"] == 1
    assert summary["rotation_consensus"]["eligible_parent_accuracy"] == 1.0


def test_leave_one_out_judge_is_position_decoupled_and_fail_closed() -> None:
    contract = IntruderLeaveOneOutJudgeContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_loo_judge_v1_llama.yaml"
    )
    task = {
        "trial_id": "parent",
        "examples": [
            {"position": index, "text": f"text-{index}"} for index in range(1, 6)
        ],
    }
    for omitted in range(1, 6):
        remaining = _ordered_remaining(task, omitted)
        assert len(remaining) == 4
        assert f"text-{omitted}" not in remaining
        assert remaining == _ordered_remaining(task, omitted)

    assert _parse_score('{"coherence_score": 4, "reason": "specific pattern"}')[
        "valid"
    ]
    assert not _parse_score('{"coherence_score": 7, "reason": "invalid"}')[
        "valid"
    ]
    candidates = [
        {
            "candidate_position": position,
            "parsed": {
                "valid": True,
                "coherence_score": 4 if position == 3 else 2,
            },
        }
        for position in range(1, 6)
    ]
    assert _select_intruder(candidates, contract) == 3
    for row in candidates:
        row["parsed"]["coherence_score"] = 4
    assert _select_intruder(candidates, contract) is None


def test_leave_one_out_summary_applies_parent_level_controls() -> None:
    contract = IntruderLeaveOneOutJudgeContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_loo_judge_v1_llama.yaml"
    )
    rows = []
    for index in range(20):
        position = index % 5 + 1
        rows.append(
            {
                "valid_parent": True,
                "control_kind": "positive",
                "predicted_position": position,
                "correct_position": position,
            }
        )
        rows.append(
            {
                "valid_parent": True,
                "control_kind": "negative",
                "predicted_position": None,
                "correct_position": None,
            }
        )
    for index in range(10):
        rows.append(
            {
                "valid_parent": True,
                "control_kind": None,
                "predicted_position": index % 5 + 1,
                "correct_position": index % 5 + 1,
            }
        )
    summary = _loo_summary(rows, prompt_count=len(rows) * 5, contract=contract)
    assert summary["status"] == "JUDGE_VALIDATED"
    assert summary["negative_control_abstention_rate"] == 1.0
    rows[0]["valid_parent"] = False
    assert _loo_summary(rows, prompt_count=len(rows) * 5, contract=contract)[
        "invalid_parent_rate"
    ] > 0


def test_causal_contract_selection_and_sparse_support_are_deterministic() -> None:
    contract = CausalValidationContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_causal_validation_v1.yaml"
    )
    assert contract.training_authorized_by_this_contract is False
    answers = []
    feature = 0
    for arm in contract.arms:
        for site in ("pre", "mid"):
            for stratum in ("low", "medium", "high"):
                for _ in range(5):
                    answers.append(
                        {
                            "arm": arm,
                            "site": site,
                            "frequency_stratum": stratum,
                            "feature": feature,
                        }
                    )
                    feature += 1
    first = _select_components(answers, contract, arm="full_only_control")
    second = _select_components(answers, contract, arm="full_only_control")
    assert [row["feature"] for row in first] == [row["feature"] for row in second]
    assert len(first) == contract.components_per_arm

    indptr = np.asarray([0, 2, 3], dtype=np.int64)
    indices = np.asarray([1, 4, 2], dtype=np.int64)
    assert _fires(indptr, indices, anchor=0, feature=4)
    assert not _fires(indptr, indices, anchor=1, feature=4)


def test_causal_metrics_and_random_component_control_are_exact() -> None:
    clean = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.0, 1.0]])
    rows = _metric_rows(clean, clean.clone())
    assert all(row["kl_clean_to_edited"] == pytest.approx(0.0) for row in rows)
    assert all(row["maximum_absolute_logit_delta"] == 0.0 for row in rows)

    target = _target()
    component = SimpleNamespace(
        U=torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]),
        V=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
    )
    model = SimpleNamespace(
        matrix_bank=SimpleNamespace(components={target.key.replace(".", "__dot__"): component})
    )
    feature, u, v = _random_component_factors(
        model, target, learned_feature=0, learned_norm=2.5, seed=109
    )
    assert feature != 0
    assert float(u.norm() * v.norm()) == pytest.approx(2.5)


def test_judge_validation_summary_applies_predeclared_thresholds() -> None:
    contract = IntruderJudgeContract.load(
        ROOT / "configs/multimodel/aspd_v3/zscore_intruder_judge_v1.yaml"
    )
    rows = []
    for index in range(20):
        rows.append(
            {
                "control_kind": "positive",
                "correct_position": index % 5 + 1,
                "parsed": {
                    "valid": True,
                    "intruder_position": index % 5 + 1,
                    "confidence": 3,
                },
            }
        )
        rows.append(
            {
                "control_kind": "negative",
                "correct_position": None,
                "parsed": {"valid": True, "intruder_position": None, "confidence": 0},
            }
        )
        rows.append(
            {
                "control_kind": None,
                "correct_position": index % 5 + 1,
                "parsed": {
                    "valid": True,
                    "intruder_position": index % 5 + 1,
                    "confidence": 2,
                },
            }
        )
    assert _summary(rows, contract)["status"] == "JUDGE_VALIDATED"


def test_causal_analysis_aggregates_contexts_and_bootstraps_components() -> None:
    rows = []
    metadata = {
        "realized_weight_delta_frobenius_norm": 2.0,
        "realized_to_intended_norm_ratio": 1.0,
        "realized_to_intended_cosine": 0.99,
        "realized_nonzero_fraction": 0.5,
    }
    for activity, values in (("active", (2.0, 4.0, 6.0, 8.0)), ("inactive", (1.0,) * 4)):
        for value in values:
            rows.append(
                {
                    "arm": "full_only_control",
                    "model": "gemma",
                    "site": "pre",
                    "feature": 7,
                    "frequency_stratum": "medium",
                    "target": "gemma.L09.q",
                    "intervention_kind": "learned",
                    "strength": 1.0,
                    "control_seed": None,
                    "context_activity": activity,
                    "intervention_metadata": metadata,
                    "kl_clean_to_edited": value,
                    "logit_delta_rms": value / 2,
                }
            )
    summary = _component_rows(rows)
    assert len(summary) == 1
    assert summary[0]["active_kl_clean_to_edited"] == pytest.approx(5.0)
    assert summary[0]["specificity_kl_clean_to_edited"] == pytest.approx(4.0)
    assert summary[0]["specificity_kl_clean_to_edited_per_realized_norm"] == pytest.approx(2.0)
    interval = _bootstrap_mean(np.asarray([1.0, 2.0, 3.0]), seed=61, replicates=100)
    assert interval["n_components"] == 3
    assert interval["mean"] == pytest.approx(2.0)
    assert interval["ci95_lower"] <= interval["mean"] <= interval["ci95_upper"]
