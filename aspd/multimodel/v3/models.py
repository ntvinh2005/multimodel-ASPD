"""Explicit model-family adapters for frozen ASPD v3 source backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from aspd.multimodel.v3.hooks import assert_frozen_backbone, freeze_backbone
from aspd.multimodel.v3.registry import ModelFamilyLayout, validate_runtime_config


@dataclass(frozen=True)
class LoadedV3Model:
    key: str
    forward_model: nn.Module
    text_backbone: nn.Module
    provenance: dict[str, Any]


def inspect_v3_model_config(layout: ModelFamilyLayout, *, local_files_only: bool = False) -> dict[str, Any]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        layout.model_id,
        revision=layout.revision,
        local_files_only=local_files_only,
    )
    return validate_runtime_config(layout, config)


def load_v3_tokenizer(layout: ModelFamilyLayout, *, local_files_only: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        layout.tokenizer_id,
        revision=layout.tokenizer_revision,
        local_files_only=local_files_only,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError(f"{layout.key} tokenizer must expose deterministic offset mappings")
    return tokenizer


def load_frozen_v3_model(
    layout: ModelFamilyLayout,
    device: str | torch.device,
    *,
    local_files_only: bool = False,
) -> LoadedV3Model:
    """Load the pinned family without substituting an inaccessible checkpoint."""

    dtype = torch.bfloat16
    common = {
        "revision": layout.revision,
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    if layout.family == "gemma3_text":
        # The published 4B checkpoint is a gated Gemma3 conditional-generation wrapper.
        # Only its language_model subtree participates in ASPD; vision modules stay frozen.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(layout.model_id, **common)
        try:
            backbone = model.model.language_model
        except AttributeError as error:
            raise ValueError("Gemma3 wrapper does not expose model.language_model") from error
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(layout.model_id, **common)
        backbone = model.model
    model.to(device)
    freeze_backbone(model)
    assert_frozen_backbone(model)
    return LoadedV3Model(
        key=layout.key,
        forward_model=model,
        text_backbone=backbone,
        provenance={
            "model_id": layout.model_id,
            "revision": layout.revision,
            "tokenizer_id": layout.tokenizer_id,
            "tokenizer_revision": layout.tokenizer_revision,
            "family": layout.family,
            "dtype": "bfloat16",
            "frozen": True,
            "text_backbone_only_for_aspd": layout.family == "gemma3_text",
        },
    )
