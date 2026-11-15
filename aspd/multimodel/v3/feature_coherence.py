"""Frozen contracts and helpers for ASPD v3.1 feature-coherence evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class FeatureCoherenceContract(BaseModel):
    """Fail-closed contract for the coherence/polysemanticity study."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    experiment_family: Literal["aspd_v3_1_feature_coherence_v1"]
    status: Literal["RESEARCHER_AUTHORIZED_EVALUATION"]
    training_authorized_by_this_contract: Literal[False]
    checkpoint_step: Literal[61_036]
    checkpoint_manifest_sha256: str
    source_config_sha256: str
    accepted_training_exposures: Literal[500_006_912]
    model_order: tuple[Literal["gemma"], Literal["smol"], Literal["granite"]]
    forbidden_model_substrings: tuple[str, ...]
    dictionary_size: Literal[8_192]
    batch_topk_k: Literal[32]
    discovery_split: Literal["test"]
    confirmatory_split: Literal["validation"]
    accepted_anchors_per_extraction: int
    pool_anchors: Literal[2_048]
    original_components: Literal[120]
    additional_components_per_site: Literal[20]
    minimum_discovery_unique_documents: Literal[50]
    component_sampling_seed: int
    contexts_per_component: Literal[100]
    negative_candidates_per_component: Literal[256]
    hard_negative_candidates_per_component: Literal[128]
    canonical_context_characters_per_side: Literal[384]
    normalized_text_near_duplicate_jaccard: Literal[0.90]
    discovery_fraction: Literal[0.60]
    rule_calibration_fraction: Literal[0.20]
    internal_holdout_fraction: Literal[0.20]
    candidate_k: tuple[Literal[2], Literal[3], Literal[4], Literal[5]]
    clustering_restarts: Literal[10]
    minimum_silhouette: Literal[0.10]
    minimum_cluster_fraction: Literal[0.10]
    minimum_stability_ari: Literal[0.60]
    minimum_heldout_firing_contexts: Literal[24]
    minimum_heldout_nonfiring_contexts: Literal[24]
    minimum_heldout_contexts_per_group: Literal[8]
    bootstrap_replicates: Literal[5_000]
    bootstrap_seed: int
    activation_quantiles: tuple[float, float]
    proposer_model_id: Literal["google/gemma-3-4b-it"]
    proposer_revision: str
    validator_model_id: Literal["microsoft/Phi-4-mini-instruct"]
    validator_revision: str
    embedding_model_id: Literal["mistralai/Mistral-Nemo-Instruct-2407"]
    embedding_revision: str
    evaluator_minimum_balanced_accuracy: Literal[0.85]
    evaluator_maximum_abstention: Literal[0.10]
    coherent_balanced_accuracy: Literal[0.70]
    coherent_lower_ci: Literal[0.60]
    coherent_specificity: Literal[0.65]
    coherent_minimum_group_recall: Literal[0.60]
    coherent_group_coverage: Literal[0.80]
    coherent_multi_rule_upper_advantage: Literal[0.10]
    polysemantic_multi_balanced_accuracy: Literal[0.70]
    polysemantic_multi_lower_ci: Literal[0.60]
    polysemantic_point_advantage: Literal[0.10]
    polysemantic_lower_advantage: Literal[0.05]
    polysemantic_group_rule_balanced_accuracy: Literal[0.65]
    polysemantic_within_across_advantage: Literal[0.15]
    polysemantic_general_failure_balanced_accuracy: Literal[0.65]
    polysemantic_general_failure_group_recall: Literal[0.50]
    no_rule_balanced_accuracy: Literal[0.60]
    no_rule_upper_ci: Literal[0.70]
    original_embedding_input: str
    original_embedding_input_sha256: str
    original_sample_manifest: str
    original_sample_manifest_sha256: str
    original_cluster_metrics: str
    original_cluster_metrics_sha256: str
    sparse_discovery_dir: str
    sparse_pre_sha256: str
    sparse_mid_sha256: str
    sparse_anchors_sha256: str
    sparse_documents_sha256: str

    @model_validator(mode="after")
    def validate_contract(self) -> FeatureCoherenceContract:
        if self.accepted_anchors_per_extraction <= 0 or (
            self.accepted_anchors_per_extraction % self.pool_anchors
        ):
            raise ValueError("extraction budget must contain complete BatchTopK pools")
        if self.forbidden_model_substrings != ("qwen",):
            raise ValueError("Qwen exclusion changed")
        if not np.isclose(
            self.discovery_fraction
            + self.rule_calibration_fraction
            + self.internal_holdout_fraction,
            1.0,
        ):
            raise ValueError("discovery fractions must sum to one")
        if self.activation_quantiles != (1 / 3, 2 / 3):
            raise ValueError("activation strength bins changed")
        for value in (
            self.checkpoint_manifest_sha256,
            self.source_config_sha256,
            self.original_embedding_input_sha256,
            self.original_sample_manifest_sha256,
            self.original_cluster_metrics_sha256,
            self.sparse_pre_sha256,
            self.sparse_mid_sha256,
            self.sparse_anchors_sha256,
            self.sparse_documents_sha256,
        ):
            if len(value) != 64:
                raise ValueError("contract SHA-256 is malformed")
        return self

    @classmethod
    def load(cls, path: str | Path) -> FeatureCoherenceContract:
        import yaml

        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def csr_unique_document_counts(
    indptr: np.ndarray,
    indices: np.ndarray,
    anchor_document_indices: np.ndarray,
    *,
    dictionary_size: int,
) -> np.ndarray:
    """Count unique firing documents per feature without context-level independence assumptions."""

    if indptr.ndim != 1 or indices.ndim != 1 or anchor_document_indices.ndim != 1:
        raise ValueError("CSR/document inputs must be one-dimensional")
    if len(indptr) != len(anchor_document_indices) + 1 or int(indptr[-1]) != len(indices):
        raise ValueError("CSR arrays and anchor-document assignments disagree")
    repeated_documents = np.repeat(
        anchor_document_indices.astype(np.int64, copy=False), np.diff(indptr)
    )
    if len(repeated_documents) != len(indices):
        raise ValueError("expanded document assignments differ from firing events")
    document_base = int(anchor_document_indices.max(initial=-1)) + 1
    encoded = indices.astype(np.int64, copy=False) * max(document_base, 1) + repeated_documents
    unique_pairs = np.unique(encoded)
    features = unique_pairs // max(document_base, 1)
    return np.bincount(features, minlength=dictionary_size).astype(np.int64, copy=False)


def component_key(site: str, feature: int) -> str:
    if site not in {"pre", "mid"} or not 0 <= feature < 8_192:
        raise ValueError("invalid ASPD v3.1 component")
    return f"{site}/{feature:04d}"


def feature_event_values(
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    *,
    feature: int,
) -> np.ndarray:
    """Return selected natural BatchTopK values for one feature."""

    if len(indices) != len(values) or int(indptr[-1]) != len(indices):
        raise ValueError("CSR values disagree")
    return values[indices == feature]


def normalize_context_text(value: str) -> str:
    """Normalize only for duplicate detection, never for evaluator display."""

    value = value.replace("⟪ANCHOR⟫", "").replace("⟪/ANCHOR⟫", "")
    return re.sub(r"\s+", " ", value).strip().casefold()


def character_shingles(value: str, width: int = 5) -> frozenset[str]:
    normalized = normalize_context_text(value)
    if not normalized:
        return frozenset()
    if len(normalized) <= width:
        return frozenset({normalized})
    return frozenset(normalized[index : index + width] for index in range(len(normalized) - width + 1))


def near_duplicate(left: str, right: str, *, threshold: float) -> bool:
    left_set = character_shingles(left)
    right_set = character_shingles(right)
    if not left_set or not right_set:
        return left_set == right_set
    return len(left_set & right_set) / len(left_set | right_set) >= threshold


def deduplicate_context_rows(
    rows: list[dict[str, object]],
    *,
    threshold: float,
    text_field: str = "text",
) -> list[dict[str, object]]:
    """Greedily preserve input order while removing exact and near duplicates."""

    retained: list[dict[str, object]] = []
    normalized: set[str] = set()
    for row in rows:
        text = str(row[text_field])
        key = normalize_context_text(text)
        if not key or key in normalized:
            continue
        if any(near_duplicate(text, str(other[text_field]), threshold=threshold) for other in retained):
            continue
        retained.append(row)
        normalized.add(key)
    return retained
