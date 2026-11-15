from __future__ import annotations

import numpy as np
import pytest

from aspd.multimodel.v3.feature_coherence import (
    FeatureCoherenceContract,
    component_key,
    csr_unique_document_counts,
    feature_event_values,
)


def test_csr_unique_document_counts_deduplicates_within_document() -> None:
    indptr = np.array([0, 2, 4, 5], dtype=np.int64)
    indices = np.array([0, 1, 0, 2, 0], dtype=np.uint16)
    documents = np.array([4, 4, 8], dtype=np.int32)
    observed = csr_unique_document_counts(
        indptr, indices, documents, dictionary_size=4
    )
    assert observed.tolist() == [2, 1, 1, 0]


def test_feature_event_values_uses_selected_sparse_events() -> None:
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([2, 5, 2], dtype=np.uint16)
    values = np.array([0.3, 0.4, 0.8], dtype=np.float32)
    assert feature_event_values(indptr, indices, values, feature=2).tolist() == pytest.approx(
        [0.3, 0.8]
    )


def test_component_key_is_fail_closed() -> None:
    assert component_key("pre", 7) == "pre/0007"
    with pytest.raises(ValueError):
        component_key("post", 7)
    with pytest.raises(ValueError):
        component_key("mid", 8_192)


def test_frozen_contract_loads() -> None:
    contract = FeatureCoherenceContract.load(
        "configs/multimodel/aspd_v3/main500m_feature_coherence_v1.yaml"
    )
    assert contract.training_authorized_by_this_contract is False
    assert contract.forbidden_model_substrings == ("qwen",)
    assert contract.activation_quantiles == pytest.approx((1 / 3, 2 / 3))
