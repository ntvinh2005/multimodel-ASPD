"""Explicit model-family adapters for frozen ASPD v3 source backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from aspd.multimodel.v3.hooks import (
    assert_frozen_backbone,
    freeze_backbone,
    logical_weight_bias,
)
from aspd.multimodel.v3.registry import (
    ModelFamilyLayout,
    resolve_submodule,
    validate_runtime_config,
)


@dataclass(frozen=True)
class LoadedV3Model:
    key: str
    forward_model: nn.Module
    text_backbone: nn.Module
    provenance: dict[str, Any]


def place_v3_model_for_text_only(
    model: nn.Module,
    layout: ModelFamilyLayout,
    device: str | torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Place only Gemma's text path on the accelerator; never move its vision tower."""

    target = torch.device(device)
    if layout.family != "gemma3_text":
        model.to(target)
        backbone = model.model
        return backbone, {
            "policy": "entire_causal_lm",
            "text_device": str(target),
            "vision_device": None,
        }
    try:
        backbone = model.model.language_model
    except AttributeError as error:
        raise ValueError("Gemma3 wrapper does not expose model.language_model") from error
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, nn.Module):
        raise ValueError("Gemma3 wrapper does not expose a text lm_head")
    backbone.to(target)
    lm_head.to(target)
    text_parameter_ids = {
        id(parameter) for module in (backbone, lm_head) for parameter in module.parameters()
    }
    non_text_devices = {
        str(parameter.device)
        for parameter in model.parameters()
        if id(parameter) not in text_parameter_ids
    }
    if target.type != "cpu" and any(device_name != "cpu" for device_name in non_text_devices):
        raise ValueError(
            "Gemma3 non-text parameters unexpectedly left CPU during text-only placement: "
            f"{sorted(non_text_devices)}"
        )
    return backbone, {
        "policy": "gemma_text_backbone_and_lm_head_only",
        "text_device": str(target),
        "vision_device": "cpu",
        "non_text_parameter_devices": sorted(non_text_devices),
    }


def inspect_v3_model_config(layout: ModelFamilyLayout, *, local_files_only: bool = False) -> dict[str, Any]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        layout.model_id,
        revision=layout.revision,
        local_files_only=local_files_only,
    )
    return validate_runtime_config(layout, config)


def load_v3_tokenizer(layout: ModelFamilyLayout, *, local_files_only: bool = False):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer_source = layout.tokenizer_id
    if local_files_only:
        # Passing a Hub ID can still trigger Transformers' Mistral-regex model-card
        # lookup. Resolve the exact cached revision to a local directory first.
        tokenizer_source = snapshot_download(
            repo_id=layout.tokenizer_id,
            revision=layout.tokenizer_revision,
            local_files_only=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if local_files_only else layout.tokenizer_revision,
        local_files_only=local_files_only,
        use_fast=True,
        # Transformers otherwise performs an online model-card lookup to guess
        # whether a Mistral regex migration applies, even with local_files_only.
        # None of the pinned v3 source tokenizers is a base Mistral tokenizer.
        fix_mistral_regex=False,
    )
    if not tokenizer.is_fast:
        raise ValueError(f"{layout.key} tokenizer must expose deterministic offset mappings")
    return tokenizer


def validate_loaded_model_modules(model: nn.Module, layout: ModelFamilyLayout) -> dict[str, Any]:
    """Fail closed if physical modules disagree with the metadata-derived registry."""

    validated = []
    for target in layout.targets():
        grounding_path, _ = resolve_submodule(model, target.grounding_module_candidates)
        weight, bias, physical_path = logical_weight_bias(model, target)
        validated.append(
            {
                "target": target.key,
                "grounding_path": grounding_path,
                "physical_path": physical_path,
                "weight_shape": list(weight.shape),
                "bias": bias is not None,
            }
        )
    return {"model": layout.key, "targets": validated, "status": "PASS"}


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
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(layout.model_id, **common)
    backbone, placement = place_v3_model_for_text_only(model, layout, device)
    freeze_backbone(model)
    assert_frozen_backbone(model)
    module_validation = validate_loaded_model_modules(model, layout)
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
            "logical_module_validation": module_validation["status"],
            "logical_target_count": len(module_validation["targets"]),
            "placement": placement,
        },
    )
