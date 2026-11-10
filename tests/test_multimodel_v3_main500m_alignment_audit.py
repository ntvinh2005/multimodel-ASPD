from __future__ import annotations

from aspd.multimodel.cli.audit_aspd_v3_main500m_alignment import (
    _capacity_projection,
    _deterministic_sample,
)
from aspd.multimodel.v3.data import HashSplitPolicy


class _AllTrainSplit:
    def split(self, _document):
        return "train"


def test_alignment_audit_sampling_is_hash_deterministic_and_bounded() -> None:
    rows = [
        {"text": text, "meta": {"pile_set_name": source}}
        for text, source in (
            ("alpha", "a"),
            ("beta", "b"),
            ("gamma", "a"),
            ("delta", "b"),
            ("epsilon", "c"),
        )
    ]
    sample, report = _deterministic_sample(
        rows,
        scan_documents=5,
        sample_documents=3,
        max_document_characters=100,
        split_policy=_AllTrainSplit(),
    )
    repeated, repeated_report = _deterministic_sample(
        rows,
        scan_documents=5,
        sample_documents=3,
        max_document_characters=100,
        split_policy=_AllTrainSplit(),
    )
    assert [row["document_id"] for row in sample] == [
        row["document_id"] for row in repeated
    ]
    assert report == repeated_report
    assert len(sample) == 3
    assert report["rows_consumed"] == 5


def test_alignment_audit_sampling_tracks_duplicates_splits_and_oversize() -> None:
    rows = [
        {"text": "same", "meta": {"pile_set_name": "a"}},
        {"text": "same", "meta": {"pile_set_name": "a"}},
        {"text": "too-long", "meta": {"pile_set_name": "b"}},
        {"text": "kept", "meta": {"pile_set_name": "c"}},
    ]
    sample, report = _deterministic_sample(
        rows,
        scan_documents=4,
        sample_documents=2,
        max_document_characters=4,
        split_policy=_AllTrainSplit(),
    )
    assert len(sample) == 2
    assert report["counters"]["exact_duplicate"] == 1
    assert report["counters"]["over_character_budget"] == 1


def test_alignment_capacity_projection_is_fail_closed_without_rows() -> None:
    unavailable = _capacity_projection([], dataset_rows=1_000, train_fraction=0.98)
    assert unavailable["status"] == "UNAVAILABLE_NO_DOCUMENTS"

    measured = _capacity_projection(
        [100, 110, 120, 130],
        dataset_rows=10_000_000,
        train_fraction=0.98,
    )
    assert measured["lower95_accepted_anchors_per_document"] > 0
    assert measured["projected_train_rows"] == 9_800_000
    assert measured["status"] == "PROJECTED_CAPACITY_PASS"


def test_real_hash_split_policy_can_be_used_by_alignment_sampler() -> None:
    policy = HashSplitPolicy(
        modulus=1_000,
        validation_buckets=frozenset(range(10)),
        test_buckets=frozenset(range(10, 20)),
        seed=0,
    )
    rows = [
        {"text": f"document-{index}", "meta": {"pile_set_name": "source"}}
        for index in range(200)
    ]
    sample, report = _deterministic_sample(
        rows,
        scan_documents=200,
        sample_documents=100,
        max_document_characters=100,
        split_policy=policy,
    )
    assert len(sample) == 100
    assert report["counters"]["split:train"] >= 100
