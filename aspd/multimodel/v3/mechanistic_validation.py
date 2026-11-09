"""Fail-closed utilities for ASPD v3.1 mechanistic validation.

This module deliberately separates three estimands:

* sparse firing-pattern diversity;
* semantic coherence of natural (never forced) gate contexts; and
* direct edits to the frozen model's physical parameter matrices.

The parameter edit is temporary and restores an exact clone of the affected logical
weight slice.  Gate masking is not implemented here because it is not a parameter-level
causal intervention.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor, nn

from aspd.multimodel.v3.hooks import logical_weight_bias
from aspd.multimodel.v3.registry import LogicalMatrixTarget


class MechanisticValidationContract(BaseModel):
    """Immutable scientific protocol for the current 10M-exposure comparison."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    status: Literal["RESEARCHER_AUTHORIZED_MECHANISTIC_VALIDATION"]
    experiment_family: Literal["aspd_v3_zscore_mechanistic_validation_v1"]
    training_authorized_by_this_contract: Literal[False]
    model_order: tuple[Literal["gemma"], Literal["smol"]]
    forbidden_model_substrings: tuple[Literal["qwen"]]
    arms: tuple[Literal["full_only_control"], Literal["per_anchor_zscore"]]
    checkpoint_steps: Literal[1221]
    accepted_training_exposures: Literal[10_002_432]
    selected_split: Literal["test"]
    split_unit: Literal["source_document"]
    test_buckets: tuple[int, ...]
    accepted_evaluation_anchors: Literal[100_352]
    pool_anchors: Literal[2_048]
    context_tokens_per_side: Literal[32]
    contexts_per_feature: Literal[4]
    retained_contexts_per_feature: Literal[12]
    screening_features_per_arm_site: Literal[50]
    frequency_strata: tuple[Literal["low"], Literal["medium"], Literal["high"]]
    minimum_feature_fires: Literal[4]
    minimum_unique_documents: Literal[4]
    intruder_examples: Literal[5]
    intruder_chance_accuracy: Literal[0.2]
    feature_sampling_seed: Literal[31]
    intruder_position_seed: Literal[37]
    bootstrap_seed: Literal[41]
    bootstrap_replicates: Literal[2_000]
    jaccard_pair_seed: Literal[43]
    jaccard_pairs_per_stratum: Literal[20_000]
    causal_components_per_arm: Literal[16]
    causal_strengths: tuple[Literal[0.0], Literal[0.5], Literal[1.0]]
    random_control_seeds: tuple[Literal[101], Literal[103], Literal[107]]
    judge_model_id: Literal["mistralai/Mistral-Nemo-Instruct-2407"]
    judge_revision: Literal["04d8a90549d23fc6bd7f642064003592df51e9b3"]
    judge_paid_api: Literal[False]
    judge_positive_controls: Literal[20]
    judge_negative_controls: Literal[20]

    @model_validator(mode="after")
    def validate_frozen_protocol(self) -> MechanisticValidationContract:
        if self.test_buckets != tuple(range(10, 20)):
            raise ValueError("mechanistic validation must use the frozen test buckets 10..19")
        if self.accepted_evaluation_anchors % self.pool_anchors:
            raise ValueError("evaluation budget must contain complete BatchTopK pools")
        if self.intruder_examples != self.contexts_per_feature + 1:
            raise ValueError("intruder task must contain four target contexts and one intruder")
        return self

    @classmethod
    def load(cls, path: str | Path) -> MechanisticValidationContract:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class IntruderJudgeContract(BaseModel):
    """Pinned local-judge and position-balance contract, separate from feature sampling."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    status: Literal["RESEARCHER_AUTHORIZED_LOCAL_INTRUDER_JUDGE"]
    experiment_family: Literal["aspd_v3_zscore_intruder_judge_v1"]
    judge_model_id: Literal["mistralai/Mistral-Nemo-Instruct-2407"]
    judge_revision: Literal["04d8a90549d23fc6bd7f642064003592df51e9b3"]
    local_files_only: Literal[True]
    paid_api: Literal[False]
    dtype: Literal["bfloat16"]
    decoding: Literal["greedy"]
    max_new_tokens: Literal[128]
    batch_size: Literal[8]
    cyclic_rotations: Literal[5]
    allow_abstention: Literal[True]
    confidence_scale: tuple[Literal[0], Literal[1], Literal[2], Literal[3]]
    positive_controls: Literal[20]
    negative_controls: Literal[20]
    control_seed: Literal[53]
    minimum_positive_control_accuracy: Literal[0.8]
    minimum_negative_control_abstain_or_low_confidence_rate: Literal[0.8]
    maximum_invalid_rate: Literal[0.05]
    maximum_position_share: Literal[0.35]

    @classmethod
    def load(cls, path: str | Path) -> IntruderJudgeContract:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class PairwiseFiringStatistic:
    left: int
    right: int
    left_fires: int
    right_fires: int
    intersection: int
    union: int
    jaccard: float
    independent_intersection: float
    excess_coactivation_rate: float


def pairwise_firing_statistic(
    left: Sequence[int],
    right: Sequence[int],
    *,
    n_anchors: int,
    left_feature: int,
    right_feature: int,
) -> PairwiseFiringStatistic:
    """Compute exact overlap from sorted, unique anchor indices without a dense matrix."""

    if n_anchors <= 0 or left_feature == right_feature:
        raise ValueError("pairwise firing statistic requires distinct features and anchors")
    left_values = tuple(int(value) for value in left)
    right_values = tuple(int(value) for value in right)
    if tuple(sorted(set(left_values))) != left_values:
        raise ValueError("left firing indices must be sorted and unique")
    if tuple(sorted(set(right_values))) != right_values:
        raise ValueError("right firing indices must be sorted and unique")
    if (left_values and (left_values[0] < 0 or left_values[-1] >= n_anchors)) or (
        right_values and (right_values[0] < 0 or right_values[-1] >= n_anchors)
    ):
        raise ValueError("firing index lies outside the evaluation anchor range")
    left_index = 0
    right_index = 0
    intersection = 0
    while left_index < len(left_values) and right_index < len(right_values):
        if left_values[left_index] == right_values[right_index]:
            intersection += 1
            left_index += 1
            right_index += 1
        elif left_values[left_index] < right_values[right_index]:
            left_index += 1
        else:
            right_index += 1
    union = len(left_values) + len(right_values) - intersection
    independent = len(left_values) * len(right_values) / n_anchors
    return PairwiseFiringStatistic(
        left=left_feature,
        right=right_feature,
        left_fires=len(left_values),
        right_fires=len(right_values),
        intersection=intersection,
        union=union,
        jaccard=intersection / union if union else 0.0,
        independent_intersection=independent,
        excess_coactivation_rate=(intersection - independent) / n_anchors,
    )


def rank_one_operator(u: Tensor, v: Tensor) -> Tensor:
    """Return ``u v^T`` in FP32 after validating the learned factor vectors."""

    if u.ndim != 1 or v.ndim != 1 or not u.numel() or not v.numel():
        raise ValueError("rank-one factors must be non-empty vectors")
    if not torch.isfinite(u).all() or not torch.isfinite(v).all():
        raise ValueError("rank-one factors must be finite")
    return torch.outer(u.float(), v.float())


def norm_matched_random_rank_one(
    *,
    d_out: int,
    d_in: int,
    target_frobenius_norm: float,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Create a seeded random rank-one operator with exactly matched Frobenius norm."""

    if d_out <= 0 or d_in <= 0 or target_frobenius_norm < 0:
        raise ValueError("random-control dimensions/norm are invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(d_out, generator=generator, dtype=torch.float32)
    v = torch.randn(d_in, generator=generator, dtype=torch.float32)
    u = u / u.norm().clamp_min(torch.finfo(torch.float32).tiny)
    v = v / v.norm().clamp_min(torch.finfo(torch.float32).tiny)
    u = u * float(target_frobenius_norm)
    return u.to(device), v.to(device)


@contextmanager
def temporary_rank_one_ablation(
    model: nn.Module,
    target: LogicalMatrixTarget,
    u: Tensor,
    v: Tensor,
    *,
    strength: float,
) -> Iterator[dict[str, Any]]:
    """Temporarily apply ``W_j <- W_j - strength * u v^T`` and restore exactly.

    The exact logical slice is cloned before mutation.  Restoration uses ``copy_`` rather
    than adding the perturbation back, avoiding BF16 round-trip drift.
    """

    if not torch.isfinite(torch.tensor(strength)):
        raise ValueError("ablation strength must be finite")
    weight, _bias, physical_path = logical_weight_bias(model, target)
    operator = rank_one_operator(u, v)
    if tuple(operator.shape) != tuple(weight.shape):
        raise ValueError("rank-one operator shape does not match logical weight slice")
    original = weight.detach().clone()
    perturbation = (float(strength) * operator).to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        weight.sub_(perturbation)
    metadata = {
        "target": target.key,
        "physical_module_path": physical_path,
        "strength": float(strength),
        "operator_frobenius_norm": float(operator.norm()),
        "applied_frobenius_norm_fp32": float((float(strength) * operator).norm()),
        "applied_frobenius_norm_stored_dtype": float(perturbation.float().norm()),
    }
    try:
        yield metadata
    finally:
        with torch.no_grad():
            weight.copy_(original)
        if not torch.equal(weight, original):
            raise RuntimeError("source weight restoration was not bitwise exact")


__all__ = [
    "MechanisticValidationContract",
    "IntruderJudgeContract",
    "PairwiseFiringStatistic",
    "norm_matched_random_rank_one",
    "pairwise_firing_statistic",
    "rank_one_operator",
    "temporary_rank_one_ablation",
]
