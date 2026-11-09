from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from aspd.multimodel.cli.analyze_aspd_v3_mechanistic import (
    _feature_anchor_lists,
    _intruder_task,
    _stratified_features,
)
from aspd.multimodel.cli.extract_aspd_v3_mechanistic import _save_site
from aspd.multimodel.v3.mechanistic_validation import (
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
            {"context": "different pattern", "document_id": "donor", "source_group": "books"}
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
