"""Analytic parameter, memory, and streaming feasibility estimates for ASPD v3."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aspd.multimodel.v3.config import V3ConfigBundle
from aspd.multimodel.v3.registry import ModelFamilyLayout


def _text_backbone_parameter_estimate(layout: ModelFamilyLayout, vocab_size: int) -> int:
    per_layer = (
        layout.hidden_size * layout.q_out
        + 2 * layout.hidden_size * layout.kv_out
        + layout.q_out * layout.hidden_size
        + 3 * layout.hidden_size * layout.intermediate_size
        + 4 * layout.hidden_size
    )
    return vocab_size * layout.hidden_size + layout.num_hidden_layers * per_layer


def estimate_v3_feasibility(
    bundle: V3ConfigBundle,
    registry: Mapping[str, ModelFamilyLayout],
) -> dict[str, Any]:
    """Return transparent lower-bound estimates; no setting is silently reduced."""

    targets = [target for layout in registry.values() for target in layout.targets()]
    rank_one_parameters_per_feature = sum(target.d_in + target.d_out for target in targets)
    activation_decoder_parameters_per_feature = sum(
        8 * layout.hidden_size for layout in registry.values()
    )
    vocab_sizes = {"qwen": 151_936, "gemma": 262_208, "phi": 200_064}
    frozen_parameters = {
        key: _text_backbone_parameter_estimate(layout, vocab_sizes[key])
        for key, layout in registry.items()
    }
    scenarios: dict[str, Any] = {}
    for dictionary_size in (8192, 16384, 32768):
        rank_one = dictionary_size * rank_one_parameters_per_feature
        activation = dictionary_size * activation_decoder_parameters_per_feature
        lower_bound = rank_one + activation
        scenarios[str(dictionary_size)] = {
            "dictionary_size": dictionary_size,
            "rank_one_parameters": rank_one,
            "activation_decoder_parameters": activation,
            "trainable_lower_bound_excluding_transformers": lower_bound,
            "bf16_weight_bytes": lower_bound * 2,
            "fp32_gradient_bytes": lower_bound * 4,
            "fp32_adam_moment_bytes": lower_bound * 8,
            "optional_fp32_master_weight_bytes": lower_bound * 4,
            "fp32_adam_training_bytes_lower_bound": lower_bound * 16,
            "bf16_weight_plus_fp32_gradient_and_adam_bytes_lower_bound": lower_bound * 14,
            "bf16_training_bytes_with_fp32_master_weights_lower_bound": lower_bound * 18,
        }

    layer_activation_values_per_anchor = activation_decoder_parameters_per_feature
    matrix_input_values_per_anchor = sum(target.d_in for target in targets)
    bytes_per_value = 2
    accepted = bundle.approved.dataset.train_minimum_accepted_anchor_exposures
    return {
        "schema_version": 1,
        "status": "BLOCKED_FOR_MAIN_RUN",
        "logical_matrix_targets": len(targets),
        "registry_dimensions": {
            key: {
                "hidden_size": layout.hidden_size,
                "intermediate_size": layout.intermediate_size,
                "num_hidden_layers": layout.num_hidden_layers,
                "num_attention_heads": layout.num_attention_heads,
                "num_key_value_heads": layout.num_key_value_heads,
                "head_dim": layout.head_dim,
                "q_out": layout.q_out,
                "kv_out": layout.kv_out,
            }
            for key, layout in registry.items()
        },
        "rank_one_parameters_per_feature": rank_one_parameters_per_feature,
        "activation_decoder_parameters_per_feature": activation_decoder_parameters_per_feature,
        "frozen_text_backbone_parameter_estimates": frozen_parameters,
        "frozen_text_backbone_bf16_bytes_estimate": sum(frozen_parameters.values()) * 2,
        "frozen_parameter_estimate_kind": (
            "analytical text-backbone estimate from registry dimensions; excludes vision, "
            "small biases/norm details, and is not a runtime measurement"
        ),
        "dictionary_scenarios": scenarios,
        "encoder_parameters": None,
        "encoder_parameter_blocker": (
            "latent token packing, token widths, Transformer depths/heads, and readout are unresolved"
        ),
        "per_anchor_streaming_lower_bounds": {
            "eight_layer_activation_values": layer_activation_values_per_anchor,
            "all_matrix_input_values": matrix_input_values_per_anchor,
            "eight_layer_activation_bytes_bf16": layer_activation_values_per_anchor * bytes_per_value,
            "all_matrix_input_bytes_bf16": matrix_input_values_per_anchor * bytes_per_value,
        },
        "prohibited_full_offline_cache_estimates": {
            "eight_layer_activations_bytes_for_minimum_exposures": (
                accepted * layer_activation_values_per_anchor * bytes_per_value
            ),
            "all_matrix_inputs_bytes_for_minimum_exposures": (
                accepted * matrix_input_values_per_anchor * bytes_per_value
            ),
            "note": "exposures include repeats; these values demonstrate why bounded streaming is mandatory, not a proposed cache",
        },
        "required_sharding_decisions": [
            "rank-one dictionary sharding or factorization",
            "matrix-group execution schedule",
            "optimizer-state sharding",
            "three frozen-backbone placement",
            "activation recomputation versus bounded cache policy",
        ],
        "main_run_claim": "NOT_READY",
        "runtime_peak_memory_measurement": None,
    }
