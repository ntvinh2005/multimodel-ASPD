"""Read-only model-pair compatibility checks for a configured ASPD experiment."""

from __future__ import annotations

from typing import Any

import torch

from aspd.multimodel.config import MultiModelExperimentConfig


def compare_tokenizer_vocabularies(
    left: dict[str, int], right: dict[str, int]
) -> dict[str, Any]:
    """Describe exact token-string/ID agreement without pretending special tokens are identical."""

    common = set(left) & set(right)
    conflicting = sorted(token for token in common if left[token] != right[token])
    return {
        "identical": left == right,
        "left_size": len(left),
        "right_size": len(right),
        "common_token_strings": len(common),
        "same_id_common_tokens": len(common) - len(conflicting),
        "conflicting_common_tokens": conflicting,
        "only_left": sorted(set(left) - set(right)),
        "only_right": sorted(set(right) - set(left)),
    }


def inspect_model_pair(
    cfg: MultiModelExperimentConfig, *, local_files_only: bool = False
) -> dict[str, Any]:
    """Inspect official configs, native tokenizers, configured hooks, and matrix dimensions.

    No pretrained weight shard is loaded.  Models are instantiated on the ``meta`` device solely
    to verify module paths and weight shapes.
    """

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    canonical_tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer,
        revision=cfg.data.tokenizer_revision,
        local_files_only=local_files_only,
    )
    canonical_vocab = canonical_tokenizer.get_vocab()
    canonical_max_id = max(canonical_vocab.values(), default=-1)
    models: list[dict[str, Any]] = []
    native_vocabs: list[dict[str, int]] = []
    errors: list[str] = []
    warnings: list[str] = []

    for spec in cfg.models:
        config = AutoConfig.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=local_files_only,
        )
        native_vocabs.append(tokenizer.get_vocab())
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=spec.trust_remote_code
            )

        module_records: dict[str, Any] = {}
        requested = [("grounding", spec.grounding.module)] + [
            (matrix.name, matrix.module) for matrix in spec.matrices
        ]
        for logical_name, module_name in requested:
            try:
                module = model.get_submodule(module_name)
            except AttributeError:
                errors.append(f"{spec.name}: missing module {module_name!r}")
                continue
            weight = getattr(module, "weight", None)
            module_records[logical_name] = {
                "module": module_name,
                "class": type(module).__name__,
                "weight_shape": list(weight.shape) if isinstance(weight, torch.Tensor) else None,
            }
            if logical_name != "grounding" and (
                not isinstance(weight, torch.Tensor) or weight.ndim != 2
            ):
                errors.append(f"{spec.name}: matrix module {module_name!r} has no 2-D weight")

        vocab_size = int(config.vocab_size)
        if canonical_max_id >= vocab_size:
            errors.append(
                f"{spec.name}: configured tokenizer max ID {canonical_max_id} exceeds "
                f"model vocabulary size {vocab_size}"
            )
        models.append(
            {
                "name": spec.name,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "architecture": list(config.architectures or []),
                "model_type": config.model_type,
                "hidden_size": int(config.hidden_size),
                "intermediate_size": int(config.intermediate_size),
                "num_hidden_layers": int(config.num_hidden_layers),
                "num_attention_heads": int(config.num_attention_heads),
                "num_key_value_heads": int(config.num_key_value_heads),
                "vocab_size": vocab_size,
                "max_position_embeddings": int(config.max_position_embeddings),
                "native_tokenizer_class": type(tokenizer).__name__,
                "native_special_tokens": tokenizer.special_tokens_map,
                "modules": module_records,
            }
        )
        del model

    topology_fields = (
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    )
    topology_matches = all(
        all(model[field] == models[0][field] for model in models[1:])
        for field in topology_fields
    )
    if not topology_matches:
        errors.append("model architecture dimensions differ")

    matrix_signatures = [
        {
            name: record["weight_shape"]
            for name, record in model["modules"].items()
            if name != "grounding"
        }
        for model in models
    ]
    matrix_shapes_match = all(signature == matrix_signatures[0] for signature in matrix_signatures)
    if not matrix_shapes_match:
        errors.append("configured matrix names or dimensions differ across models")

    tokenizer_comparison = (
        compare_tokenizer_vocabularies(native_vocabs[0], native_vocabs[1])
        if len(native_vocabs) == 2
        else None
    )
    if tokenizer_comparison and tokenizer_comparison["conflicting_common_tokens"]:
        errors.append("native tokenizers assign different IDs to shared token strings")
    if tokenizer_comparison and not tokenizer_comparison["identical"]:
        warnings.append(
            "native tokenizers differ; the configured tokenizer still creates one identical ID "
            "grid for every model, and special-token semantics must be reported explicitly"
        )

    return {
        "status": "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
        "configured_tokenizer": cfg.data.tokenizer,
        "configured_tokenizer_revision": cfg.data.tokenizer_revision,
        "configured_tokenizer_class": type(canonical_tokenizer).__name__,
        "configured_tokenizer_size": len(canonical_tokenizer),
        "configured_tokenizer_max_id": canonical_max_id,
        "models": models,
        "topology_matches": topology_matches,
        "matrix_shapes_match": matrix_shapes_match,
        "native_tokenizer_comparison": tokenizer_comparison,
        "matched_token_semantics": (
            "One sequence is encoded once with data.tokenizer and replayed unchanged through "
            "every model; no generated-position alignment is used."
        ),
        "warnings": warnings,
        "errors": errors,
    }
