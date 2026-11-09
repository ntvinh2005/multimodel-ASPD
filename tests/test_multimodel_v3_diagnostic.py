from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from aspd.multimodel.cli.evaluate_aspd_v3_500m import (
    _validate_controlled_checkpoint,
)
from aspd.multimodel.cli.probe_aspd_v3_500m import cosine_matrix, support_overlap
from aspd.multimodel.v3.alignment import (
    AlignedAnchor,
    AlignmentResult,
    TokenBoundary,
    TokenizedDocument,
)
from aspd.multimodel.v3.data import HashSplitPolicy, SourceDocument
from aspd.multimodel.v3.diagnostic import (
    DiagnosticContract,
    SequentialDocumentFVU,
    TensorMetricAccumulator,
    canonical_metric_rows,
    cluster_bootstrap_mean_ci,
    feature_health_from_counts,
    pairwise_jaccard,
    summarize_training_trajectory,
)
from aspd.multimodel.v3.diagnostic_stream import (
    DiagnosticAlignedAnchorStream,
    ValidationAlignedAnchorStream,
)
from aspd.multimodel.v3.objective import matryoshka_activation_fvu
from aspd.multimodel.v3.reconstruction import ActivationDecoder

ROOT = Path(__file__).parents[1]


class _StatefulRows:
    def __init__(self, rows):
        self.rows = rows
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.rows):
            raise StopIteration
        row = self.rows[self.index]
        self.index += 1
        return row

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state):
        self.index = int(state["index"])


def test_diagnostic_contract_freezes_validation_and_complete_pools() -> None:
    contract = DiagnosticContract.load(
        ROOT / "configs/multimodel/aspd_v3/diagnostic_500m_v1.yaml"
    )
    assert contract.status == "RESEARCHER_APPROVED"
    assert contract.split == "validation"
    assert contract.main_anchors == 100_352
    assert contract.main_anchors % contract.pool_anchors == 0
    assert len(contract.fingerprint()) == 64


def test_controlled_checkpoint_provenance_is_bound_to_frozen_source(
    tmp_path: Path,
) -> None:
    contract = DiagnosticContract.load(
        ROOT / "configs/multimodel/aspd_v3/diagnostic_500m_v1.yaml"
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = {
        "status": "COMPLETE",
        "step": 1221,
        "world_size": 3,
        "config_file_sha256": contract.source_config_sha256,
        "config_fingerprint": contract.source_config_fingerprint,
        "dataset_manifest_sha256": contract.dataset_manifest_sha256,
    }
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "COMPLETE").write_text(
        json.dumps({"manifest_sha256": digest}), encoding="utf-8"
    )
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "git_sha": "training-commit",
                "controlled_retrain": {
                    "arm": "gemma_specific",
                    "source_checkpoint_manifest_sha256": contract.checkpoint_manifest_sha256,
                    "source_checkpoint_step": 61036,
                    "source_accepted_exposures": 500006912,
                    "additional_optimizer_steps": 1221,
                    "additional_accepted_exposures": 10002432,
                    "frozen_scope": ["rank_one_matrix_factors"],
                    "optimizer_state": "fresh_for_both_arms",
                },
            }
        ),
        encoding="utf-8",
    )
    result = _validate_controlled_checkpoint(
        checkpoint=checkpoint,
        expected_manifest_sha256=digest,
        run_manifest_path=run_manifest,
        arm="gemma_specific",
        contract=contract,
    )
    assert result["status"] == "PASS"
    assert result["checkpoint_step"] == 1221

    try:
        _validate_controlled_checkpoint(
            checkpoint=checkpoint,
            expected_manifest_sha256=digest,
            run_manifest_path=run_manifest,
            arm="shared_control",
            contract=contract,
        )
    except ValueError as error:
        assert "arm" in str(error)
    else:
        raise AssertionError("controlled arm mismatch was not rejected")


def test_canonical_metrics_keep_last_replayed_step() -> None:
    rows = [
        {"step": 1, "loss": {"total": 3.0}},
        {"step": 2, "loss": {"total": 2.0}},
        {"step": 2, "loss": {"total": 1.5}},
        {"step": 3, "loss": {"total": 1.0}},
    ]
    canonical, accounting = canonical_metric_rows(json.dumps(row) for row in rows)
    assert [row["loss"]["total"] for row in canonical] == [3.0, 1.5, 1.0]
    assert accounting == {
        "physical_rows": 4,
        "canonical_rows": 3,
        "replayed_rows": 1,
        "final_step": 3,
    }


def test_compact_training_summary_has_no_missing_model_site_values() -> None:
    rows = []
    for step in range(1, 1001):
        loss = {"total": 1.0, "activation": 0.5, "auxk": 0.1, "internal": 0.4}
        for model, value in (("qwen", 0.2), ("gemma", 0.6), ("smol", 0.15)):
            for site in ("pre", "mid"):
                loss[f"activation/{model}/{site}"] = value
        rows.append({"step": step, "loss": loss})
    summary = summarize_training_trajectory(rows)
    assert summary["activation_model_site"]["last_1000"]["gemma"]["pre"] == 0.6
    assert math.isclose(
        summary["activation_ratios"]["last_1000"]["pre"]["gemma_over_qwen"], 3.0
    )


def test_tensor_metric_accumulator_uses_underlying_sums_not_batch_fvu_mean() -> None:
    truth = torch.tensor([[0.0, 1.0], [2.0, 3.0], [8.0, 2.0]])
    prediction = torch.tensor([[0.5, 1.5], [1.0, 3.0], [7.0, 4.0]])
    accumulator = TensorMetricAccumulator(width=2)
    accumulator.update(prediction[:2], truth[:2])
    accumulator.update(prediction[2:], truth[2:])
    result = accumulator.finalize()
    centered = truth - truth.mean(dim=0, keepdim=True)
    expected_sse = float((prediction - truth).square().sum())
    expected_fvu = expected_sse / float(centered.square().sum())
    assert result["sse"] == expected_sse
    assert math.isclose(result["fvu"], expected_fvu, rel_tol=1e-6)
    assert math.isclose(result["r2"], 1 - expected_fvu, rel_tol=1e-6)
    assert math.isclose(result["mse"], expected_sse / truth.numel())


def test_tensor_metric_accumulator_marks_constant_prediction_correlation_undefined() -> None:
    truth = torch.tensor([[0.0, 1.0], [2.0, 3.0], [8.0, 2.0]])
    prediction = torch.ones_like(truth)
    accumulator = TensorMetricAccumulator(width=2)
    accumulator.update(prediction, truth)
    result = accumulator.finalize()
    assert result["target_prediction_correlation"] is None
    assert result["target_prediction_correlation_defined"] is False
    assert math.isfinite(result["fvu"])


def test_matryoshka_training_metric_is_not_full_reconstruction_fvu() -> None:
    decoder = ActivationDecoder(16, 3, bias=True)
    code = torch.zeros(1, 4, 16)
    code[0, :, 15] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = decoder(code).detach()
    valid = torch.ones(1, 4, dtype=torch.bool)
    loss, full = matryoshka_activation_fvu(code, decoder, target, valid)
    torch.testing.assert_close(full, target)
    assert float(loss.detach()) > 0.0
    assert float((full - target).square().sum().detach()) == 0.0


def test_cluster_bootstrap_is_deterministic_and_cluster_level() -> None:
    first = cluster_bootstrap_mean_ci([0.2, 0.5, 1.0, 1.5], seed=17, replicates=200)
    second = cluster_bootstrap_mean_ci([0.2, 0.5, 1.0, 1.5], seed=17, replicates=200)
    assert first == second
    assert first["clusters"] == 4
    assert first["lower"] <= first["mean"] <= first["upper"]


def test_document_fvu_merges_one_source_document_across_pool_boundaries() -> None:
    accumulator = SequentialDocumentFVU(width=2)
    first_truth = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    first_prediction = first_truth + 0.5
    accumulator.update(
        first_prediction,
        first_truth,
        (0, 0),
        ("document-a",),
    )
    second_truth = torch.tensor([[4.0, 5.0], [10.0, 11.0], [12.0, 13.0]])
    second_prediction = second_truth + torch.tensor([[0.5, 0.5], [1.0, 1.0], [1.0, 1.0]])
    accumulator.update(
        second_prediction,
        second_truth,
        (0, 1, 1),
        ("document-a", "document-b"),
    )
    report = accumulator.finalize()

    combined_truth = torch.cat((first_truth, second_truth[:1]))
    combined_prediction = torch.cat((first_prediction, second_prediction[:1]))
    expected_a = TensorMetricAccumulator(width=2)
    expected_a.update(combined_prediction, combined_truth)
    expected_b = TensorMetricAccumulator(width=2)
    expected_b.update(second_prediction[1:], second_truth[1:])
    assert report["unique_source_documents"] == 2
    assert report["eligible_source_documents"] == 2
    assert report["excluded_zero_variance_documents"] == 0
    assert report["values"] == [expected_a.finalize()["fvu"], expected_b.finalize()["fvu"]]


def test_hypothesis_probe_support_and_gradient_helpers() -> None:
    reference = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    candidate = torch.tensor([[1.0, 4.0, 0.0], [0.0, 3.0, 0.0]])
    overlap = support_overlap(reference, candidate)
    assert overlap["intersection"] == 2
    assert overlap["union"] == 4
    assert overlap["jaccard"] == 0.5
    matrix = cosine_matrix(
        [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])]
    )
    assert matrix == [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]


def test_feature_health_and_jaccard_filter_inactive_features() -> None:
    firing = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [False, False, True, False],
            [True, True, False, False],
        ]
    )
    counts = firing.sum(dim=0)
    health = feature_health_from_counts(counts, accepted_anchors=4)
    assert health["used_features"] == 3
    assert health["never_fired_features"] == 1
    assert health["mean_active_features_per_anchor"] == 2.0
    jaccard = pairwise_jaccard(firing, counts, top_features=3, min_fires=2)
    pairs = {(row["left"], row["right"]): row for row in jaccard["pairs"]}
    assert jaccard["eligible_features"] == 3
    assert jaccard["evaluated_pairs"] == 3
    assert pairs[(0, 1)]["jaccard"] == 1.0
    assert all(3 not in pair for pair in pairs)


def test_validation_stream_never_emits_train_documents_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    policy = HashSplitPolicy(
        modulus=1000,
        validation_buckets=frozenset(range(10)),
        test_buckets=frozenset(range(10, 20)),
        seed=0,
    )
    rows = []
    expected_validation = []
    for index in range(20_000):
        text = f"diagnostic document {index}"
        document_id = hashlib.sha256(text.encode()).hexdigest()
        document = SourceDocument(document_id, "fixture", text, f"row-{index}")
        split = policy.split(document)
        rows.append({"text": text, "meta": {"pile_set_name": "fixture"}})
        if split == "validation":
            expected_validation.append(document_id)
        if len(expected_validation) >= 4:
            break

    def fake_alignment(text, *, document_id, source_group, tokenizers, model_order, max_length):
        tokenized = {}
        positions = {}
        for model in model_order:
            boundary = TokenBoundary(model, 0, 1, 1, len(text), len(text.encode()))
            positions[model] = boundary
            tokenized[model] = TokenizedDocument(
                input_ids=(1,),
                attention_mask=(1,),
                boundary_to_token={len(text): boundary},
                truncated=False,
                prefix_decode_calls=1,
                prefix_decoded_token_ids=1,
            )
        anchor = AlignedAnchor(
            anchor_id=f"anchor-{document_id}",
            document_id=document_id,
            source_group=source_group,
            character_boundary=len(text),
            byte_boundary=len(text.encode()),
            canonical_prefix_sha256="a" * 64,
            positions=positions,
        )
        return AlignmentResult(document_id, source_group, (anchor,), tokenized, {})

    monkeypatch.setattr(
        "aspd.multimodel.v3.diagnostic_stream.align_canonical_text", fake_alignment
    )
    tokenizers = {key: object() for key in ("qwen", "gemma", "smol")}
    revisions = {key: key for key in tokenizers}
    dedup = tmp_path / "validation.sqlite"
    stream = ValidationAlignedAnchorStream(
        dataset=_StatefulRows(rows),
        dataset_id="fixture",
        dataset_revision="revision",
        tokenizers=tokenizers,
        tokenizer_revisions=revisions,
        split_policy=policy,
        dedup_path=dedup,
        sequence_length=1,
    )
    first = stream.next_pool(pool_anchors=2)
    state = stream.state_dict()
    stream.close()
    assert list(first.document_ids) == expected_validation[:2]

    resumed = ValidationAlignedAnchorStream(
        dataset=_StatefulRows(rows),
        dataset_id="fixture",
        dataset_revision="revision",
        tokenizers=tokenizers,
        tokenizer_revisions=revisions,
        split_policy=policy,
        dedup_path=dedup,
        sequence_length=1,
    )
    resumed.load_state_dict(state)
    second = resumed.next_pool(pool_anchors=2)
    resumed.close()
    assert list(second.document_ids) == expected_validation[2:4]


def test_diagnostic_stream_can_select_disjoint_test_documents(
    tmp_path: Path, monkeypatch
) -> None:
    policy = HashSplitPolicy(
        modulus=1000,
        validation_buckets=frozenset(range(10)),
        test_buckets=frozenset(range(10, 20)),
        seed=0,
    )
    rows = []
    expected_test = []
    validation_ids = set()
    for index in range(30_000):
        text = f"held-out diagnostic document {index}"
        document_id = hashlib.sha256(text.encode()).hexdigest()
        document = SourceDocument(document_id, "fixture", text, f"row-{index}")
        split = policy.split(document)
        rows.append({"text": text, "meta": {"pile_set_name": "fixture"}})
        if split == "validation":
            validation_ids.add(document_id)
        elif split == "test":
            expected_test.append(document_id)
        if len(expected_test) >= 5 and validation_ids:
            break

    def fake_alignment(text, *, document_id, source_group, tokenizers, model_order, max_length):
        tokenized = {}
        positions = {}
        for model in model_order:
            boundary = TokenBoundary(model, 0, 1, 1, len(text), len(text.encode()))
            positions[model] = boundary
            tokenized[model] = TokenizedDocument(
                input_ids=(1,),
                attention_mask=(1,),
                boundary_to_token={len(text): boundary},
                truncated=False,
                prefix_decode_calls=1,
                prefix_decoded_token_ids=1,
            )
        anchor = AlignedAnchor(
            anchor_id=f"anchor-{document_id}",
            document_id=document_id,
            source_group=source_group,
            character_boundary=len(text),
            byte_boundary=len(text.encode()),
            canonical_prefix_sha256="b" * 64,
            positions=positions,
        )
        return AlignmentResult(document_id, source_group, (anchor,), tokenized, {})

    monkeypatch.setattr(
        "aspd.multimodel.v3.diagnostic_stream.align_canonical_text", fake_alignment
    )
    tokenizers = {key: object() for key in ("gemma", "smol")}
    stream = DiagnosticAlignedAnchorStream(
        dataset=_StatefulRows(rows),
        dataset_id="fixture",
        dataset_revision="revision",
        tokenizers=tokenizers,
        tokenizer_revisions={key: key for key in tokenizers},
        split_policy=policy,
        dedup_path=tmp_path / "test.sqlite",
        selected_split="test",
        excluded_document_ids={expected_test[0]},
        sequence_length=1,
        model_order=("gemma", "smol"),
    )
    pool = stream.next_pool(pool_anchors=4)
    stream.close()
    assert list(pool.document_ids) == expected_test[1:5]
    assert set(pool.document_ids).isdisjoint(validation_ids)
    assert expected_test[0] not in pool.document_ids
    assert stream.excluded_documents_rejected == 1
    assert stream.provenance["selected_split"] == "test"
    assert stream.provenance["excluded_document_count"] == 1
    assert all("canonical_text" in row for row in pool.source_observations)
    assert all(
        row["anchor_character_boundaries"] == [len(row["canonical_text"])]
        for row in pool.source_observations
    )
