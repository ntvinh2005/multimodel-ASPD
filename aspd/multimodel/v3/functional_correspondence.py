"""Frozen contract and signed metrics for ASPD v3.1 functional correspondence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor


class FunctionalCorrespondenceContract(BaseModel):
    """Read-only three-model causal correspondence contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    status: Literal["RESEARCHER_AUTHORIZED_FUNCTIONAL_CORRESPONDENCE"]
    experiment_family: Literal["aspd_v3_1_functional_correspondence_v1"]
    training_authorized_by_this_contract: Literal[False]
    checkpoint_step: Literal[61036]
    checkpoint_manifest_sha256: Literal[
        "735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969"
    ]
    model_order: tuple[Literal["gemma", "smol", "granite"], ...]
    selected_split: Literal["test"]
    component_selection_frame: Literal[
        "stage1_components_stratified_by_heldout_internal_contribution"
    ]
    components_per_site: Literal[12]
    contribution_stratum_quota: dict[Literal["low", "medium", "high"], int]
    active_contexts_per_component: Literal[4]
    inactive_contexts_per_component: Literal[4]
    context_activity_definition: Literal["natural_batchtopk_support"]
    canonical_alignment: Literal["exact_three_tokenizer_prefix_boundary"]
    continuation_definition: Literal["suffix_after_natural_peak_within_frozen_context"]
    minimum_continuation_utf8_bytes: Literal[32]
    minimum_continuation_tokens_per_model: Literal[4]
    target_selection_policy: Literal[
        "maximum_learned_operator_frobenius_within_site_per_model"
    ]
    learned_strengths: tuple[Literal[0.0], Literal[1.0]]
    random_rank_one_seeds: tuple[Literal[101], Literal[103], Literal[107]]
    random_component_seed: Literal[109]
    parameter_restore: Literal["bitwise_exact_logical_slice_copy"]
    primary_metric: Literal["edited_minus_clean_continuation_logprob_per_utf8_byte"]
    secondary_metrics: tuple[
        Literal[
            "edited_minus_clean_continuation_logprob_total",
            "edited_minus_clean_continuation_logprob_per_model_token",
        ],
        ...,
    ]
    selection_seed: Literal[20261121]
    bootstrap_seed: Literal[20261122]
    bootstrap_replicates: Literal[5000]
    adjudication_rule: Literal[
        "support_if_two_pairwise_signed_effect_upper_cis_below_zero_or_learned_all_three_sign_agreement_upper_ci_below_random_control_point;weaken_if_all_pairwise_lower_cis_above_zero_and_learned_sign_agreement_exceeds_random;otherwise_inconclusive"
    ]

    @model_validator(mode="after")
    def validate_contract(self) -> FunctionalCorrespondenceContract:
        if self.model_order != ("gemma", "smol", "granite"):
            raise ValueError("functional correspondence requires the approved three-model order")
        if self.contribution_stratum_quota != {"low": 4, "medium": 4, "high": 4}:
            raise ValueError("functional correspondence contribution quotas changed")
        if sum(self.contribution_stratum_quota.values()) != self.components_per_site:
            raise ValueError("functional correspondence site budget changed")
        return self

    @classmethod
    def load(cls, path: str | Path) -> FunctionalCorrespondenceContract:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def continuation_logprob_metrics(
    clean_logits: Tensor,
    edited_logits: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    prefix_token_counts: list[int],
    continuation_utf8_bytes: list[int],
) -> list[dict[str, float | int]]:
    """Measure signed continuation log-probability changes for right-padded batches."""

    if clean_logits.shape != edited_logits.shape or clean_logits.ndim != 3:
        raise ValueError("clean and edited logits must share [batch, sequence, vocabulary]")
    if input_ids.shape != attention_mask.shape or input_ids.shape != clean_logits.shape[:2]:
        raise ValueError("token tensors do not match causal-LM logits")
    if len(prefix_token_counts) != input_ids.shape[0] or len(continuation_utf8_bytes) != input_ids.shape[0]:
        raise ValueError("continuation metadata does not match batch size")
    output: list[dict[str, float | int]] = []
    for index, (prefix_count, byte_count) in enumerate(
        zip(prefix_token_counts, continuation_utf8_bytes, strict=True)
    ):
        length = int(attention_mask[index].sum().item())
        if not 1 <= prefix_count < length or byte_count <= 0:
            raise ValueError("invalid continuation boundary or byte count")
        targets = input_ids[index, prefix_count:length]
        clean_slice = clean_logits[index, prefix_count - 1 : length - 1].float()
        edited_slice = edited_logits[index, prefix_count - 1 : length - 1].float()
        clean_logp = torch.log_softmax(clean_slice, dim=-1).gather(
            1, targets[:, None]
        ).sum()
        edited_logp = torch.log_softmax(edited_slice, dim=-1).gather(
            1, targets[:, None]
        ).sum()
        delta = edited_logp - clean_logp
        token_count = int(targets.numel())
        output.append(
            {
                "continuation_model_tokens": token_count,
                "continuation_utf8_bytes": int(byte_count),
                "clean_continuation_logprob_total": float(clean_logp),
                "edited_continuation_logprob_total": float(edited_logp),
                "edited_minus_clean_continuation_logprob_total": float(delta),
                "edited_minus_clean_continuation_logprob_per_model_token": float(
                    delta / token_count
                ),
                "edited_minus_clean_continuation_logprob_per_utf8_byte": float(
                    delta / byte_count
                ),
            }
        )
    return output


def extracted_context_boundary(
    *,
    token_ids: list[int],
    activations: list[float],
    context: str,
    tokenizer: Any,
) -> int:
    """Recover the natural peak's character boundary in a frozen decoded context."""

    if not token_ids or len(token_ids) != len(activations) or not context:
        raise ValueError("frozen context arrays are invalid")
    peak = max(range(len(activations)), key=lambda index: (activations[index], -index))
    if activations[peak] <= 0:
        raise ValueError("frozen context has no natural positive activation")
    kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    decoded = tokenizer.decode(token_ids, **kwargs)
    prefix = tokenizer.decode(token_ids[: peak + 1], **kwargs)
    if decoded.strip() != context:
        raise ValueError("stored context differs from its frozen token IDs")
    left_trim = len(decoded) - len(decoded.lstrip())
    boundary = len(prefix) - left_trim
    if not 0 < boundary < len(context) or context[:boundary] != prefix[left_trim:]:
        raise ValueError("natural peak is not an exact decoded context boundary")
    return boundary


__all__ = [
    "FunctionalCorrespondenceContract",
    "continuation_logprob_metrics",
    "extracted_context_boundary",
]
