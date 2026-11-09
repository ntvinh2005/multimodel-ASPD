"""Fail-closed contract and gradient routing for the v3 controlled retrain.

This experiment continues the exact post-500M training stream while freezing all
rank-one matrix factors.  The treatment changes only which ranks synchronize the
replicated cross-model PRE/MID encoders; every encoder still consumes all three
models' aligned latent tokens.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aspd.multimodel.v3.main_2b_config import MODEL_ORDER_2B
from aspd.multimodel.v3.model_owned import ModelOwnedMain2B


class ControlledRetrainContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    experiment_family: Literal["aspd_v3_2b_controlled_gate_retrain"]
    approval_status: Literal["RESEARCHER_APPROVED"]
    approval_note: str = Field(min_length=1)
    source_config: str
    source_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_checkpoint: str
    source_checkpoint_step: Literal[61036]
    source_checkpoint_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_accepted_exposures: Literal[500006912]
    arm: Literal["shared_control", "gemma_specific"]
    additional_optimizer_steps: Literal[1221]
    effective_batch_anchors: Literal[8192]
    additional_accepted_exposures: Literal[10002432]
    trainable_scope: tuple[
        Literal["local_encoders"],
        Literal["shared_encoders"],
        Literal["activation_decoders"],
    ]
    frozen_scope: tuple[Literal["rank_one_matrix_factors"]]
    optimizer_state: Literal["fresh_for_both_arms"]
    data_start: Literal["exact_source_checkpoint_stream_continuation"]
    qwen_smol_group: tuple[Literal[0], Literal[2]]
    gemma_owner_rank: Literal[1]
    output_root: str

    @model_validator(mode="after")
    def validate_accounting(self) -> ControlledRetrainContract:
        if self.additional_optimizer_steps * self.effective_batch_anchors != (
            self.additional_accepted_exposures
        ):
            raise ValueError("controlled retrain exposure accounting mismatch")
        if self.trainable_scope != (
            "local_encoders",
            "shared_encoders",
            "activation_decoders",
        ):
            raise ValueError("controlled retrain trainable scope changed")
        if "controlled_retrain_10m_v1" not in self.output_root:
            raise ValueError("controlled retrain must use its isolated output namespace")
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def load_controlled_retrain_contract(path: str | Path) -> ControlledRetrainContract:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ControlledRetrainContract.model_validate(payload)


def configure_controlled_trainables(model: ModelOwnedMain2B) -> list[dict[str, object]]:
    """Freeze matrix factors and return the three approved Adam parameter groups."""

    model.matrix_bank.requires_grad_(False)
    activation = list(model.activation_banks.parameters())
    local = list(model.local_encoders.parameters())
    shared = list(model.shared_encoders.parameters())
    approved = activation + local + shared
    if not approved or any(not parameter.requires_grad for parameter in approved):
        raise RuntimeError("approved controlled-retrain parameter group is incomplete")
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if trainable_ids != {id(parameter) for parameter in approved}:
        raise RuntimeError("unapproved parameter entered the controlled retrain optimizer")
    return [
        {"params": activation, "scope": "activation_decoders"},
        {"params": local, "scope": "local_encoders"},
        {"params": shared, "scope": "shared_encoders"},
    ]


@torch.no_grad()
def reduce_controlled_shared_gradients_(
    model: ModelOwnedMain2B,
    *,
    arm: str,
    qwen_smol_group: dist.ProcessGroup | None,
) -> None:
    """Apply the frozen control/treatment gradient synchronization topology."""

    if not dist.is_initialized() or dist.get_world_size() != len(MODEL_ORDER_2B):
        raise RuntimeError("controlled retrain requires exactly three distributed ranks")
    if arm not in {"shared_control", "gemma_specific"}:
        raise ValueError(f"unknown controlled retrain arm: {arm}")
    rank = dist.get_rank()
    for parameter in model.shared_parameters():
        if parameter.grad is None:
            raise RuntimeError("shared encoder parameter lost its activation gradient")
        if arm == "shared_control":
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        elif rank in (0, 2):
            if qwen_smol_group is None:
                raise RuntimeError("Qwen/Smol process group was not constructed")
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=qwen_smol_group)
        # Rank 1 intentionally retains only Gemma's local activation-objective gradient.
