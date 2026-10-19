"""Measure Delta-10122 correspondence and ASPD computation changes on fixed tokens."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

from aspd.multimodel.baselines.delta_crosscoder import (
    DeltaCrosscoder,
    load_delta_checkpoint,
    load_delta_crosscoder_config,
    resolve_dictionary_size,
)
from aspd.multimodel.cache import PairedActivationCache, load_effective_model
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.losses import fvu
from aspd.multimodel.mechanism_discovery import (
    capture_model_forward,
    delta_feature_values_and_direction,
    load_prompt_rows,
    pad_sequence_batch,
    rank_percentile,
    sha256_file,
    sparse_correspondence_metrics,
    tokenize_prompt_rows,
)
from aspd.multimodel.training import build_model_from_cache

FEATURE_ID = 10122
ACTIVATION_DIM = 3584
EXPECTED_DELTA_SHA256 = "838010d85b5793b9c0e2882dd0bc3ce427ebae5652cb84423acaa14289504477"
EXPECTED_ASPD_SHA256 = "b5c27fdf641924ffe4511ac4b75ff1468274593c250d878682527df9e1383800"
CATEGORIES = ("sports_risk", "sports_control", "neutral")


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _rms_vector_difference(
    left_scalar: Tensor,
    left_u: Tensor,
    right_scalar: Tensor,
    right_u: Tensor,
) -> Tensor:
    """RMS_t ||a_t u - b_t v|| without materializing [T,C,d_out]."""

    left_norm_sq = left_u.float().square().sum(-1)
    right_norm_sq = right_u.float().square().sum(-1)
    dot = (left_u.float() * right_u.float()).sum(-1)
    norm_sq = (
        left_scalar.float().square() * left_norm_sq
        + right_scalar.float().square() * right_norm_sq
        - 2 * left_scalar.float() * right_scalar.float() * dot
    ).clamp_min(0)
    return norm_sq.mean(dim=0).sqrt()


def _contribution_metrics(
    model: Any,
    selected: Tensor,
    gate_selected: Tensor,
    x_by_model: list[dict[str, Tensor]],
    matrix_names: list[str],
    device: torch.device,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    cards = {int(feature): {"matrices": {}} for feature in selected.tolist()}
    reconstruction: dict[str, Any] = {"activation_fvu": {}, "matrix_fvu": {}}
    valid = torch.ones((1, gate_selected.shape[0]), dtype=torch.bool, device=device)
    full_gate = model._discovery_gate  # Attached locally by run(); never checkpointed.
    full_values = model._discovery_values
    normalized_r = model._discovery_r

    for model_index, model_name in enumerate(model.model_names):
        activation_reconstruction = model.activation_decoders.reconstruct(
            full_values[None], model_index, valid
        )
        reconstruction["activation_fvu"][model_name] = float(
            fvu(normalized_r[model_index][None], activation_reconstruction, valid).detach()
        )
        reconstruction["matrix_fvu"][model_name] = {}

    for matrix_name in matrix_names:
        u_base, v_base = model.component_factors(0, matrix_name)
        u_target, v_target = model.component_factors(1, matrix_name)
        u_base = u_base[selected]
        v_base = v_base[selected]
        u_target = u_target[selected]
        v_target = v_target[selected]
        x_base = x_by_model[0][matrix_name].to(device=device, dtype=v_base.dtype)
        x_target = x_by_model[1][matrix_name].to(device=device, dtype=v_target.dtype)
        gate = gate_selected.to(device=device, dtype=v_base.dtype)
        read_base = (x_base @ v_base.T) * gate
        read_target_on_base = (x_base @ v_target.T) * gate
        read_target = (x_target @ v_target.T) * gate

        base_rms = (read_base.float().square().mean(0).sqrt() * u_base.float().norm(dim=-1))
        target_rms = (
            read_target.float().square().mean(0).sqrt() * u_target.float().norm(dim=-1)
        )
        actual_rms = _rms_vector_difference(
            read_target, u_target, read_base, u_base
        )
        operator_rms = _rms_vector_difference(
            read_target_on_base, u_target, read_base, u_base
        )
        input_rms = (
            (read_target - read_target_on_base).float().square().mean(0).sqrt()
            * u_target.float().norm(dim=-1)
        )
        for local_index, feature in enumerate(selected.tolist()):
            cards[int(feature)]["matrices"][matrix_name] = {
                "base_contribution_rms": float(base_rms[local_index]),
                "target_contribution_rms": float(target_rms[local_index]),
                "actual_change_rms": float(actual_rms[local_index]),
                "operator_change_on_base_input_rms": float(operator_rms[local_index]),
                "target_operator_input_change_rms": float(input_rms[local_index]),
                "decomposition_identity": (
                    "target(x_target)-base(x_base) = "
                    "[target(x_base)-base(x_base)] + [target(x_target)-target(x_base)]"
                ),
            }

        # Full reconstruction fidelity uses all 8192 components and the measured shared gate.
        for model_index, model_name in enumerate(model.model_names):
            x = x_by_model[model_index][matrix_name].to(device)
            component = model.components[model_index][matrix_name.replace(".", "__dot__")]
            estimate = component.reconstruct(x[None], full_gate[None], valid)
            target = model._target(model_index, matrix_name).apply(x[None])
            reconstruction["matrix_fvu"][model_name][matrix_name] = float(
                fvu(target, estimate, valid).detach()
            )
        del x_base, x_target, read_base, read_target_on_base, read_target
        torch.cuda.empty_cache()
    return cards, reconstruction


def _posthoc_fields(posthoc: dict[str, Tensor], feature: int, matrix_names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "activation_rho": {
            "base": float(posthoc["activation_rho_NC"][0, feature]),
            "target": float(posthoc["activation_rho_NC"][1, feature]),
        },
        "activation_decoder_norm": {
            "base": float(posthoc["decoder_norm_NC"][0, feature]),
            "target": float(posthoc["decoder_norm_NC"][1, feature]),
            "note": "ASPD uses one shared code; P1 model attribution comes from decoder norms.",
        },
        "mechanism_rho": {
            "base": float(posthoc["mechanism_rho_NC"][0, feature]),
            "target": float(posthoc["mechanism_rho_NC"][1, feature]),
        },
        "mechanism_beta": {
            "base": float(posthoc["mechanism_beta_NC"][0, feature]),
            "target": float(posthoc["mechanism_beta_NC"][1, feature]),
        },
        "validation_fire_count": int(posthoc["fire_count_C"][feature]),
        "matrices": {},
    }
    for name in matrix_names:
        result["matrices"][name] = {
            "beta_base": float(posthoc[f"beta/base/{name}"][feature]),
            "beta_target": float(posthoc[f"beta/target/{name}"][feature]),
            "locus_base": float(posthoc[f"locus/base/{name}"][feature]),
            "locus_target": float(posthoc[f"locus/target/{name}"][feature]),
            "relative_component_change": float(
                posthoc[f"pair/{name}/relative_component_change"][feature]
            ),
            "component_cosine": float(posthoc[f"pair/{name}/component_cosine"][feature]),
            "read_cosine": float(posthoc[f"pair/{name}/read_cosine"][feature]),
            "write_cosine": float(posthoc[f"pair/{name}/write_cosine"][feature]),
        }
    return result


def _context_examples(
    feature: int,
    aspd_values: Tensor,
    delta_values: Tensor,
    token_meta: list[dict[str, Any]],
    sequences: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    values = aspd_values[:, feature]
    take = min(6, values.numel())
    indices = values.topk(take).indices.tolist()
    output = []
    for index in indices:
        meta = token_meta[index]
        row = sequences[meta["sequence_index"]]
        position = meta["position"]
        start, stop = max(0, position - 8), min(row["sequence_length"], position + 9)
        output.append(
            {
                **meta,
                "aspd_value": float(values[index]),
                "delta_10122_value": float(delta_values[index]),
                "window_start": start,
                "window_stop": stop,
                "center_offset": position - start,
                "token_ids": row["input_ids"][start:stop],
                "text": tokenizer.decode(
                    row["input_ids"][start:stop], clean_up_tokenization_spaces=False
                ),
            }
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("correspondence measurement requires an available CUDA device")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    delta_checkpoint = Path(args.delta_checkpoint)
    aspd_checkpoint = Path(args.aspd_checkpoint)
    if sha256_file(delta_checkpoint) != EXPECTED_DELTA_SHA256:
        raise ValueError("Delta checkpoint SHA256 mismatch")
    if sha256_file(aspd_checkpoint) != EXPECTED_ASPD_SHA256:
        raise ValueError("ASPD checkpoint SHA256 mismatch")
    delta_cfg = load_delta_crosscoder_config(args.delta_config)
    aspd_cfg = load_experiment_config(args.aspd_config)
    if [(m.model_id, m.revision) for m in delta_cfg.models] != [
        (m.model_id, m.revision) for m in aspd_cfg.models
    ]:
        raise ValueError("Delta and ASPD configs do not contain the same pinned model pair")
    if delta_cfg.delta_crosscoder.activation_preprocessing != "raw":
        raise ValueError("Delta measurement expects raw grounding activations")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        aspd_cfg.data.tokenizer,
        revision=aspd_cfg.data.tokenizer_revision,
        fix_mistral_regex=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_rows = load_prompt_rows(args.prompts)
    sequences = tokenize_prompt_rows(prompt_rows, tokenizer, max_length=args.max_length)

    delta_spec = delta_cfg.delta_crosscoder
    delta_model = DeltaCrosscoder(
        ACTIVATION_DIM,
        resolve_dictionary_size(delta_spec, ACTIVATION_DIM),
        shared_fraction=delta_spec.shared_fraction,
        k_shared=delta_spec.k_shared,
        k_delta=delta_spec.k_delta,
        norm_init_scale=delta_spec.initial_decoder_norm_scale,
    ).to(device)
    delta_state = load_delta_checkpoint(delta_checkpoint, delta_model, map_location="cpu")
    del delta_state
    delta_model.eval()

    live_models = []
    live_provenance = []
    for model_spec in aspd_cfg.models:
        model, provenance = load_effective_model(model_spec, device)
        live_models.append(model)
        live_provenance.append(provenance)

    flat_r: list[list[Tensor]] = [[], []]
    flat_x: list[dict[str, list[Tensor]]] = [
        {matrix.name: [] for matrix in model.matrices} for model in aspd_cfg.models
    ]
    delta_parts: list[Tensor] = []
    token_meta: list[dict[str, Any]] = []
    for offset in range(0, len(sequences), args.batch_size):
        batch_rows = sequences[offset : offset + args.batch_size]
        ids_cpu, valid_cpu = pad_sequence_batch(batch_rows, tokenizer.pad_token_id)
        captured = [
            capture_model_forward(
                model, model_spec, ids_cpu, valid_cpu, capture_matrices=True
            )
            for model, model_spec in zip(live_models, aspd_cfg.models, strict=True)
        ]
        paired = torch.stack((captured[0]["R"], captured[1]["R"]), dim=2).float()
        delta_values, _direction = delta_feature_values_and_direction(
            delta_model, paired, FEATURE_ID
        )
        delta_parts.append(delta_values[valid_cpu.to(device)].detach().float().cpu())
        for model_index in range(2):
            flat_r[model_index].append(
                captured[model_index]["R"][valid_cpu.to(device)].detach().cpu()
            )
            for name, value in captured[model_index]["X"].items():
                flat_x[model_index][name].append(value[valid_cpu.to(device)].detach().cpu())
        for local_index, row in enumerate(batch_rows):
            for position in range(int(row["sequence_length"])):
                token_meta.append(
                    {
                        "sequence_index": offset + local_index,
                        "prompt_id": row["prompt_id"],
                        "category": row["category"],
                        "lexical_condition": row["lexical_condition"],
                        "position": position,
                        "region": "prompt" if position < int(row["prompt_length"]) else "continuation",
                        "token_id": int(row["input_ids"][position]),
                    }
                )
        del captured, paired
        torch.cuda.empty_cache()

    delta_values = torch.cat(delta_parts)
    r_cpu = [torch.cat(parts) for parts in flat_r]
    x_cpu = [
        {name: torch.cat(parts) for name, parts in matrices.items()}
        for matrices in flat_x
    ]
    if len(token_meta) != delta_values.numel() or any(r.shape[0] != delta_values.numel() for r in r_cpu):
        raise RuntimeError("flattened exact-token alignment failed")

    # Release both 7B live models and Delta model before loading the 19 GB ASPD checkpoint.
    del live_models, delta_model
    gc.collect()
    torch.cuda.empty_cache()

    cache = PairedActivationCache(aspd_cfg)
    aspd_model = build_model_from_cache(aspd_cfg, cache).to(device).eval()
    aspd_state = torch.load(
        aspd_checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    aspd_model.load_state_dict(aspd_state["model"])
    del aspd_state
    gc.collect()

    normalized_r = [
        r.to(device=device, dtype=next(aspd_model.parameters()).dtype)
        / float(cache.model_manifests[index]["r_rms_norm"])
        for index, r in enumerate(r_cpu)
    ]
    valid = torch.ones((1, delta_values.numel()), dtype=torch.bool, device=device)
    code = aspd_model.encode([r[None] for r in normalized_r], valid)
    aspd_values = code.values[0].detach().float().cpu()
    aspd_gate = code.gate[0].detach().cpu()
    expected_selected = aspd_cfg.sparsity.top_k * delta_values.numel()
    if code.n_selected != expected_selected:
        raise RuntimeError(f"ASPD BatchTopK selected {code.n_selected}, expected {expected_selected}")

    all_metrics = sparse_correspondence_metrics(delta_values, aspd_values)
    category_metrics: dict[str, dict[str, Tensor]] = {}
    for category in CATEGORIES:
        mask = torch.tensor([meta["category"] == category for meta in token_meta])
        category_metrics[category] = sparse_correspondence_metrics(
            delta_values[mask], aspd_values[mask]
        )
    positive_spearman = all_metrics["spearman"].clamp_min(0)
    positive_pearson = all_metrics["pearson"].clamp_min(0)
    group_spearman = torch.stack(
        [category_metrics[category]["spearman"].clamp_min(0) for category in CATEGORIES]
    )
    stability = (group_spearman.mean(0) - group_spearman.std(0)).clamp_min(0)
    discovery_score = (
        0.25 * rank_percentile(positive_spearman)
        + 0.10 * rank_percentile(positive_pearson)
        + 0.20 * rank_percentile(all_metrics["binary_phi"].clamp_min(0))
        + 0.15 * rank_percentile(all_metrics["jaccard"])
        + 0.15 * rank_percentile(all_metrics["coactivation_excess"].clamp_min(0))
        + 0.15 * rank_percentile(stability)
    )
    ranked_ids = discovery_score.argsort(descending=True)
    inverse_rank = torch.empty_like(ranked_ids)
    inverse_rank[ranked_ids] = torch.arange(ranked_ids.numel())
    shortlist = ranked_ids[: args.shortlist_size]

    posthoc = load_file(str(Path(args.posthoc)))
    matrix_names = [matrix.name for matrix in aspd_cfg.models[0].matrices]
    # Temporary measured tensors are attached only to avoid copying them through helper signatures.
    aspd_model._discovery_gate = code.gate[0]  # type: ignore[attr-defined]
    aspd_model._discovery_values = code.values[0]  # type: ignore[attr-defined]
    aspd_model._discovery_r = normalized_r  # type: ignore[attr-defined]
    measured_cards, reconstruction = _contribution_metrics(
        aspd_model,
        shortlist.to(device),
        aspd_gate[:, shortlist].to(device),
        x_cpu,
        matrix_names,
        device,
    )

    full_rows: list[dict[str, Any]] = []
    for feature in range(aspd_cfg.sparsity.n_features):
        row: dict[str, Any] = {
            "rank": int(inverse_rank[feature]) + 1,
            "component_id": feature,
            "discovery_score": float(discovery_score[feature]),
            **{name: float(value[feature]) for name, value in all_metrics.items()},
            "stability_score": float(stability[feature]),
        }
        for category in CATEGORIES:
            row[f"spearman_{category}"] = float(category_metrics[category]["spearman"][feature])
            row[f"jaccard_{category}"] = float(category_metrics[category]["jaccard"][feature])
        full_rows.append(row)
    full_rows.sort(key=lambda row: row["rank"])
    with (output_dir / "full_correspondence_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(full_rows[0]))
        writer.writeheader()
        writer.writerows(full_rows)

    cards: list[dict[str, Any]] = []
    actual_change = []
    computation_mass = []
    geometry_change = []
    geometry_with_support = []
    for feature in shortlist.tolist():
        matrices = measured_cards[int(feature)]["matrices"]
        actual_change.append(math.sqrt(sum(row["actual_change_rms"] ** 2 for row in matrices.values())))
        computation_mass.append(
            math.sqrt(
                sum(
                    row["base_contribution_rms"] ** 2 + row["target_contribution_rms"] ** 2
                    for row in matrices.values()
                )
            )
        )
        geometry_change.append(
            sum(
                float(posthoc[f"pair/{name}/relative_component_change"][feature])
                for name in matrix_names
            )
            / len(matrix_names)
        )
        beta_total = float(
            posthoc["mechanism_beta_NC"][0, feature]
            + posthoc["mechanism_beta_NC"][1, feature]
        )
        # Relative component change alone is unstable when both rank-one operators
        # have negligible norm. Weight it by existing P2 mechanism mass before it
        # can influence final selection.
        geometry_with_support.append(geometry_change[-1] * beta_total)
    final_score = (
        0.45 * rank_percentile(discovery_score[shortlist])
        + 0.25 * rank_percentile(torch.tensor(actual_change))
        + 0.15 * rank_percentile(torch.tensor(computation_mass))
        + 0.15 * rank_percentile(torch.tensor(geometry_with_support))
    )
    final_local = final_score.argsort(descending=True)[: args.final_size]
    final_ids = {int(shortlist[index]) for index in final_local}

    row_by_feature = {int(row["component_id"]): row for row in full_rows}
    for local_index, feature in enumerate(shortlist.tolist()):
        posthoc_fields = _posthoc_fields(posthoc, int(feature), matrix_names)
        matrix_measurements = measured_cards[int(feature)]["matrices"]
        changed_matrices = sorted(
            matrix_names,
            key=lambda name: matrix_measurements[name]["actual_change_rms"],
            reverse=True,
        )
        selected_final = int(feature) in final_ids
        cards.append(
            {
                "component_id": int(feature),
                "discovery_rank": row_by_feature[int(feature)]["rank"],
                "is_final_candidate": selected_final,
                "final_selection_score_within_shortlist": float(final_score[local_index]),
                "selection_decision": (
                    "selected by the frozen composite of correspondence, absolute measured change, "
                    "absolute mass, and mass-supported geometry"
                    if selected_final
                    else "retained in the 20-component audit trail but ranked below the final five"
                ),
                "correspondence": row_by_feature[int(feature)],
                "posthoc_p1_p5": posthoc_fields,
                "measured_computation": {
                    "aggregate_actual_change_rms": actual_change[local_index],
                    "aggregate_contribution_mass_rms": computation_mass[local_index],
                    "mean_relative_component_change": geometry_change[local_index],
                    "mechanism_mass_weighted_geometry_change": geometry_with_support[local_index],
                    "matrices_by_actual_change": changed_matrices,
                    **measured_cards[int(feature)],
                },
                "representative_exact_token_contexts": _context_examples(
                    int(feature), aspd_values, delta_values, token_meta, sequences, tokenizer
                ),
                "interpretation_boundary": (
                    "Candidate correspondence only; association does not identify equal features, "
                    "and ASPD operators approximate configured matrix computations rather than Delta-W."
                ),
                "alternative_interpretations": [
                    "shared lexical or response-position sensitivity",
                    "generic decisiveness/style computation rather than sports risk",
                    "correlated membership in a distributed mechanism rather than feature equivalence",
                ],
                "confidence": "exploratory; requires manual context audit and new held-out confirmation",
            }
        )

    payload = {
        "scientific_status": "EXPLORATORY_ASPD_MECHANISM_DISCOVERY",
        "anchor_delta_feature_id": FEATURE_ID,
        "measurement_set": {
            "prompts": len(sequences),
            "valid_token_positions": delta_values.numel(),
            "categories": {category: sum(row["category"] == category for row in sequences) for category in CATEGORIES},
            "same_canonical_token_ids": True,
            "teacher_forced": True,
            "batch_topk_scope": "all valid positions in the 24-sequence discovery set",
            "delta_feature_fire_count": int((delta_values > 0).sum()),
            "aspd_selected_pairs": code.n_selected,
        },
        "ranking_protocol": {
            "shortlist_size": args.shortlist_size,
            "final_size": args.final_size,
            "discovery_score": (
                "0.25 rank(positive Spearman) + 0.10 rank(positive Pearson) + "
                "0.20 rank(positive binary phi) + 0.15 rank(Jaccard) + "
                "0.15 rank(coactivation excess above ASPD fire-rate baseline) + "
                "0.15 rank(mean-minus-SD positive subset Spearman)"
            ),
            "final_score_within_shortlist": (
                "0.45 correspondence + 0.25 measured actual computation change + "
                "0.15 measured computation mass + 0.15 P2-mass-weighted P5 relative change"
            ),
            "negative_controls_preserved": True,
        },
        "shortlist_component_ids": [int(value) for value in shortlist],
        "final_component_ids": sorted(final_ids),
        "component_cards": cards,
        "local_reconstruction": reconstruction,
        "provenance": {
            "git_independent_output": True,
            "delta_config": args.delta_config,
            "aspd_config": args.aspd_config,
            "delta_checkpoint": args.delta_checkpoint,
            "delta_checkpoint_sha256": EXPECTED_DELTA_SHA256,
            "aspd_checkpoint": args.aspd_checkpoint,
            "aspd_checkpoint_sha256": EXPECTED_ASPD_SHA256,
            "posthoc": args.posthoc,
            "posthoc_sha256": sha256_file(args.posthoc),
            "prompts": args.prompts,
            "prompts_sha256": sha256_file(args.prompts),
            "tokenizer": aspd_cfg.data.tokenizer,
            "tokenizer_revision": aspd_cfg.data.tokenizer_revision,
            "live_model_loading": live_provenance,
            "r_rms_norm": [manifest["r_rms_norm"] for manifest in cache.model_manifests],
        },
        "limitations": [
            "The Delta anchor was selected after V3 inspection, so this is discovery, not confirmation.",
            "ASPD BatchTopK support depends on the supplied discovery batch and is teacher-forced.",
            "Rank-one components reconstruct configured matrix outputs; they are not exact Delta-W factors.",
            "Relative changes require joint inspection with absolute measured contribution mass.",
        ],
    }
    if not _finite_tree(payload):
        raise RuntimeError("correspondence artifact contains NaN or Inf")
    (output_dir / "mechanism_discovery.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "component_cards.json").write_text(
        json.dumps(cards, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-config", required=True)
    parser.add_argument("--aspd-config", required=True)
    parser.add_argument("--delta-checkpoint", required=True)
    parser.add_argument("--aspd-checkpoint", required=True)
    parser.add_argument("--posthoc", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--shortlist-size", type=int, default=20)
    parser.add_argument("--final-size", type=int, default=5)
    args = parser.parse_args()
    if not 10 <= args.shortlist_size <= 20:
        parser.error("--shortlist-size must be between 10 and 20")
    if not 3 <= args.final_size <= 5 or args.final_size > args.shortlist_size:
        parser.error("--final-size must be between 3 and 5 and no larger than shortlist")
    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output_dir) / "mechanism_discovery.json"),
                "shortlist_component_ids": payload["shortlist_component_ids"],
                "final_component_ids": payload["final_component_ids"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
