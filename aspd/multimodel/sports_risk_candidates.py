"""Response-level ASPD candidate discovery on the frozen behavioral V3 corpus.

Selection uses sequence-level activation rates, not token-as-independent observations.
The measurement pass then computes exact rank-one contribution norms for a small frozen
shortlist and negative controls on the already materialized effective-model X/R cache.
"""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy.stats import rankdata
from torch import Tensor

from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.fidelity import (
    sha256_file,
    validate_checkpoint_provenance,
    write_json_immutable,
)
from aspd.multimodel.sports_risk import effective_grounding_batch
from aspd.multimodel.training import build_model_from_cache

METRIC_NAMES = (
    "base_contribution_rms",
    "target_contribution_rms",
    "actual_change_rms",
    "operator_change_on_base_input_rms",
    "target_operator_input_change_rms",
)
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def standardized_mean_difference(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pooled-SD response-level effect size with a stable zero-variance convention."""

    left_values, right_values = values[left], values[right]
    left_var = left_values.var(axis=0, ddof=1)
    right_var = right_values.var(axis=0, ddof=1)
    pooled = (
        (left.sum() - 1) * left_var + (right.sum() - 1) * right_var
    ) / max(int(left.sum() + right.sum() - 2), 1)
    difference = left_values.mean(axis=0) - right_values.mean(axis=0)
    result = np.zeros_like(difference, dtype=np.float64)
    stable = pooled > 1e-12
    result[stable] = difference[stable] / np.sqrt(pooled[stable])
    return result


def positive_percentile(values: np.ndarray) -> np.ndarray:
    """Rank strictly positive finite values in (0,1], leaving all others at zero."""

    output = np.zeros_like(values, dtype=np.float64)
    positive = np.isfinite(values) & (values > 0)
    if positive.any():
        output[positive] = rankdata(values[positive], method="average") / positive.sum()
    return output


def _mask(metadata: list[dict[str, Any]], **conditions: Any) -> np.ndarray:
    return np.asarray(
        [all(row.get(key) == value for key, value in conditions.items()) for row in metadata]
    )


def select_v3_candidates(
    metadata: list[dict[str, Any]],
    sequence_tensors: Mapping[str, Tensor],
    posthoc: Mapping[str, Tensor],
    *,
    matrix_names: Sequence[str],
    shortlist_size: int = 15,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Select an exploratory shortlist while preserving explicit negative controls."""

    if not 10 <= shortlist_size <= 20:
        raise ValueError("shortlist_size must be in [10,20]")
    response_count = sequence_tensors["response_fire_count"].float().numpy()
    lengths = np.asarray(
        [row["sequence_length"] - row["prompt_token_count"] for row in metadata],
        dtype=np.float64,
    )[:, None]
    # Square-root stabilizes sparse rates while removing the severe response-length
    # confound present in maxima/presence (cautious responses are about 8x longer).
    rates = np.sqrt(response_count / np.maximum(lengths, 1))
    risk_target = _mask(metadata, category="risk", generator_side="target_generated")
    control_target = _mask(
        metadata, category="sports_control", generator_side="target_generated"
    )
    neutral_target = _mask(
        metadata, category="neutral", generator_side="target_generated"
    )
    high = np.asarray(
        [
            row["behavioral_score"] is not None and row["behavioral_score"] >= 2
            for row in metadata
        ]
    )
    cautious = np.asarray(
        [
            row["behavioral_score"] is not None and row["behavioral_score"] <= 1
            for row in metadata
        ]
    )
    specificity = standardized_mean_difference(rates, risk_target, control_target)
    versus_neutral = standardized_mean_difference(rates, risk_target, neutral_target)
    permissiveness = standardized_mean_difference(rates, high, cautious)
    risk_support = (response_count[risk_target] > 0).mean(axis=0)
    risk_rate = rates[risk_target].mean(axis=0)

    domains = sorted(
        {row["sport_domain"] for row in metadata if row["category"] == "risk"}
    )
    breadth = np.zeros(response_count.shape[1], dtype=np.float64)
    for domain in domains:
        domain_mask = np.asarray([row["sport_domain"] == domain for row in metadata])
        breadth += (
            rates[risk_target & domain_mask].mean(axis=0)
            > rates[control_target & domain_mask].mean(axis=0)
        ) / len(domains)

    beta = posthoc["mechanism_beta_NC"].sum(dim=0).float().numpy()
    validation_fire = posthoc["fire_count_C"].float().numpy()
    relative_change = np.stack(
        [
            posthoc[f"pair/{name}/relative_component_change"].float().numpy()
            for name in matrix_names
        ]
    ).mean(axis=0)
    supported_geometry = beta * relative_change
    eligible = (
        (risk_support >= 0.10)
        & (validation_fire >= 32)
        & np.isfinite(specificity)
        & np.isfinite(permissiveness)
    )
    score = (
        0.25 * positive_percentile(specificity)
        + 0.20 * positive_percentile(permissiveness)
        + 0.10 * positive_percentile(versus_neutral)
        + 0.15 * positive_percentile(risk_support)
        + 0.10 * positive_percentile(breadth)
        + 0.10 * positive_percentile(beta)
        + 0.05 * positive_percentile(risk_rate)
        + 0.05 * positive_percentile(supported_geometry)
    )
    score[~eligible] = -1
    if int(eligible.sum()) < shortlist_size:
        raise ValueError(
            f"only {int(eligible.sum())} components meet V3 shortlist eligibility"
        )
    order = np.argsort(-score, kind="stable")[:shortlist_size]
    rows = []
    for rank, component in enumerate(order.tolist(), start=1):
        rows.append(
            {
                "selection_rank": rank,
                "component_id": component,
                "selection_score": float(score[component]),
                "risk_vs_sports_control_effect": float(specificity[component]),
                "risk_vs_neutral_effect": float(versus_neutral[component]),
                "high_permissiveness_vs_cautious_effect": float(
                    permissiveness[component]
                ),
                "risk_target_response_support": float(risk_support[component]),
                "risk_target_mean_sqrt_fire_rate": float(risk_rate[component]),
                "positive_domain_fraction": float(breadth[component]),
                "validation_fire_count": int(validation_fire[component]),
                "mechanism_beta_total": float(beta[component]),
                "mean_relative_operator_change": float(relative_change[component]),
                "mass_supported_operator_change": float(supported_geometry[component]),
            }
        )
    shortlist_ids = {row["component_id"] for row in rows}
    # Controls are selected without V3 labels: match the shortlist's general-cache
    # mechanism mass and firing support while preferring small operator changes.
    active = validation_fire >= 32
    target_log_beta = float(np.median(np.log1p(beta[active])))
    target_log_fire = float(np.median(np.log1p(validation_fire[active])))
    distance = (
        np.abs(np.log1p(beta) - target_log_beta)
        + np.abs(np.log1p(validation_fire) - target_log_fire)
        + 0.25 * positive_percentile(relative_change)
    )
    control_eligible = active & np.isfinite(distance)
    if shortlist_ids:
        control_eligible[list(shortlist_ids)] = False
    distance[~control_eligible] = np.inf
    controls = np.argsort(distance, kind="stable")[:5].tolist()
    if len(controls) != 5 or not np.isfinite(distance[controls]).all():
        raise ValueError("could not select five active label-free negative controls")
    return rows, controls


def _squared_contribution_terms(
    read_base: Tensor,
    read_target_on_base: Tensor,
    read_target: Tensor,
    u_base: Tensor,
    u_target: Tensor,
) -> Tensor:
    """Return five squared rank-one norms as [B,T,S,metric]."""

    base_norm = u_base.float().square().sum(dim=-1)
    target_norm = u_target.float().square().sum(dim=-1)
    cross = (u_base.float() * u_target.float()).sum(dim=-1)

    def difference(left: Tensor, right: Tensor) -> Tensor:
        return (
            left.float().square() * target_norm
            + right.float().square() * base_norm
            - 2 * left.float() * right.float() * cross
        ).clamp_min(0)

    return torch.stack(
        (
            read_base.float().square() * base_norm,
            read_target.float().square() * target_norm,
            difference(read_target, read_base),
            difference(read_target_on_base, read_base),
            (read_target - read_target_on_base).float().square() * target_norm,
        ),
        dim=-1,
    )


def _posthoc_card(
    posthoc: Mapping[str, Tensor], component: int, matrix_names: Sequence[str]
) -> dict[str, Any]:
    matrices = {}
    for name in matrix_names:
        matrices[name] = {
            "beta_base": float(posthoc[f"beta/base/{name}"][component]),
            "beta_target": float(posthoc[f"beta/target/{name}"][component]),
            "locus_base": float(posthoc[f"locus/base/{name}"][component]),
            "locus_target": float(posthoc[f"locus/target/{name}"][component]),
            "relative_component_change": float(
                posthoc[f"pair/{name}/relative_component_change"][component]
            ),
            "component_cosine": float(posthoc[f"pair/{name}/component_cosine"][component]),
            "read_cosine": float(posthoc[f"pair/{name}/read_cosine"][component]),
            "write_cosine": float(posthoc[f"pair/{name}/write_cosine"][component]),
        }
    return {
        "activation_rho": {
            "base": float(posthoc["activation_rho_NC"][0, component]),
            "target": float(posthoc["activation_rho_NC"][1, component]),
        },
        "decoder_norm": {
            "base": float(posthoc["decoder_norm_NC"][0, component]),
            "target": float(posthoc["decoder_norm_NC"][1, component]),
        },
        "mechanism_rho": {
            "base": float(posthoc["mechanism_rho_NC"][0, component]),
            "target": float(posthoc["mechanism_rho_NC"][1, component]),
        },
        "mechanism_beta": {
            "base": float(posthoc["mechanism_beta_NC"][0, component]),
            "target": float(posthoc["mechanism_beta_NC"][1, component]),
        },
        "validation_fire_count": int(posthoc["fire_count_C"][component]),
        "matrices": matrices,
    }


def _group_masks(metadata: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "risk_target": _mask(metadata, category="risk", generator_side="target_generated"),
        "risk_base": _mask(metadata, category="risk", generator_side="base_generated"),
        "sports_control_target": _mask(
            metadata, category="sports_control", generator_side="target_generated"
        ),
        "neutral_target": _mask(
            metadata, category="neutral", generator_side="target_generated"
        ),
        "high_permissiveness": np.asarray(
            [
                row["behavioral_score"] is not None and row["behavioral_score"] >= 2
                for row in metadata
            ]
        ),
        "cautious": np.asarray(
            [
                row["behavioral_score"] is not None and row["behavioral_score"] <= 1
                for row in metadata
            ]
        ),
    }


def _exact_contexts(
    component: int,
    metadata: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    sequence: Mapping[str, Tensor],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    values = sequence["response_max"][:, component]
    selected: list[int] = []
    for category in ("risk", "sports_control", "neutral"):
        eligible = [
            index
            for index, row in enumerate(metadata)
            if row["category"] == category and row["generator_side"] == "target_generated"
        ]
        eligible.sort(key=lambda index: float(values[index]), reverse=True)
        selected.extend(index for index in eligible[:2] if float(values[index]) > 0)
    contexts = []
    for index in selected:
        position = int(sequence["response_argmax"][index, component])
        token_ids = responses[index]["token_ids"]
        start, stop = max(0, position - 10), min(len(token_ids), position + 11)
        contexts.append(
            {
                "example_index": index,
                "example_id": metadata[index]["example_id"],
                "category": metadata[index]["category"],
                "generator_side": metadata[index]["generator_side"],
                "sport_domain": metadata[index]["sport_domain"],
                "behavioral_score": metadata[index]["behavioral_score"],
                "activation_max": float(values[index]),
                "absolute_token_position": position,
                "window_start": start,
                "window_stop": stop,
                "center_offset": position - start,
                "token_ids": token_ids[start:stop],
                "text": tokenizer.decode(
                    token_ids[start:stop], clean_up_tokenization_spaces=False
                ),
            }
        )
    return contexts


@torch.no_grad()
def analyze_v3_candidates(
    cfg: MultiModelExperimentConfig,
    checkpoint: str | Path,
    *,
    expected_checkpoint_sha256: str,
    matrix_cache_dir: str | Path,
    grouped_fidelity_path: str | Path,
    sequence_components_path: str | Path,
    posthoc_path: str | Path,
    response_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
    shortlist_size: int = 15,
    final_size: int = 5,
) -> dict[str, Any]:
    """Rank response-level candidates and measure exact rank-one contributions."""

    if not 3 <= final_size <= 5 or final_size > shortlist_size:
        raise ValueError("final_size must be in [3,5] and no larger than shortlist")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    cache_root = Path(matrix_cache_dir)
    manifest = json.loads((cache_root / "manifest.json").read_text())
    matrix_names = tuple(manifest["matrix_names"])
    if not matrix_names:
        raise ValueError("V3 candidate analysis requires at least one matrix")
    metadata = _load_jsonl(cache_root / manifest["metadata_file"])
    source_root = Path(manifest["paired_activation_cache"]["path"])
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    grouped = json.loads(Path(grouped_fidelity_path).read_text())
    sequence = load_file(str(sequence_components_path))
    if grouped["component_sequence_summary"]["sha256"] != sha256_file(
        sequence_components_path
    ):
        raise ValueError("sequence-component artifact differs from grouped fidelity")
    posthoc = load_file(str(posthoc_path))
    shortlist, controls = select_v3_candidates(
        metadata, sequence, posthoc,
        matrix_names=matrix_names, shortlist_size=shortlist_size,
    )
    shortlist_ids = [row["component_id"] for row in shortlist]
    selected_ids = shortlist_ids + [value for value in controls if value not in shortlist_ids]
    selected = torch.tensor(selected_ids, dtype=torch.long)

    training_cache = PairedActivationCache(cfg)
    checkpoint_record = validate_checkpoint_provenance(
        cfg,
        training_cache,
        checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    model = build_model_from_cache(cfg, training_cache)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state["model"], strict=True)
    del state
    device = torch.device(device_name)
    model = model.to(device).float().eval()
    norms = [float(row["r_rms_norm"]) for row in training_cache.model_manifests]
    # [region, sequence, selected component, matrix, metric]
    measured = {
        region: torch.zeros(
            (len(metadata), len(selected_ids), len(matrix_names), len(METRIC_NAMES)),
            dtype=torch.float32,
        )
        for region in ("prompt", "response")
    }
    observed_fire = {
        region: torch.zeros((len(metadata), len(selected_ids)), dtype=torch.int16)
        for region in ("prompt", "response")
    }

    for split in ("train", "validation"):
        for entry in source_manifest["splits"][split]["shards"]:
            source = torch.load(
                source_root / entry["path"], map_location="cpu", weights_only=True
            )
            x_shards = [
                torch.load(
                    cache_root / "models" / spec.name / entry["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                for spec in cfg.models
            ]
            rows = int(source["input_ids"].shape[0])
            for start in range(0, rows, 2):
                stop = min(start + 2, rows)
                valid = source["valid_tokens"][start:stop].to(device)
                example_indices = source["example_indices"][start:stop]
                base_r, target_r = effective_grounding_batch(source, x_shards, start, stop)
                activations = [
                    base_r.to(device).float() / norms[0],
                    target_r.to(device).float() / norms[1],
                ]
                code = model.encode(activations, valid)
                candidate_gate = code.gate[..., selected.to(device)].float()
                regions: list[tuple[int, int, int, str]] = []
                for row_index, example_index in enumerate(example_indices.tolist()):
                    prompt_length = int(metadata[example_index]["prompt_token_count"])
                    length = int(valid[row_index].sum())
                    regions.extend(
                        (
                            (row_index, example_index, prompt_length, "prompt"),
                            (row_index, example_index, length, "response"),
                        )
                    )
                    observed_fire["prompt"][example_index] = candidate_gate[
                        row_index, :prompt_length
                    ].sum(dim=0).to(torch.int16).cpu()
                    observed_fire["response"][example_index] = candidate_gate[
                        row_index, prompt_length:length
                    ].sum(dim=0).to(torch.int16).cpu()

                for matrix_index, name in enumerate(matrix_names):
                    u_base, v_base = model.component_factors(0, name)
                    u_target, v_target = model.component_factors(1, name)
                    u_base, v_base = u_base[selected.to(device)], v_base[selected.to(device)]
                    u_target, v_target = (
                        u_target[selected.to(device)],
                        v_target[selected.to(device)],
                    )
                    x_base = x_shards[0]["X"][name][start:stop].to(device).float()
                    x_target = x_shards[1]["X"][name][start:stop].to(device).float()
                    read_base = torch.einsum("bti,si->bts", x_base, v_base) * candidate_gate
                    read_target_on_base = (
                        torch.einsum("bti,si->bts", x_base, v_target) * candidate_gate
                    )
                    read_target = (
                        torch.einsum("bti,si->bts", x_target, v_target) * candidate_gate
                    )
                    squared = _squared_contribution_terms(
                        read_base,
                        read_target_on_base,
                        read_target,
                        u_base,
                        u_target,
                    )
                    for row_index, example_index, boundary, region in regions:
                        prompt_length = int(metadata[example_index]["prompt_token_count"])
                        begin = 0 if region == "prompt" else prompt_length
                        end = boundary
                        measured[region][example_index, :, matrix_index] = (
                            squared[row_index, begin:end].mean(dim=0).sqrt().cpu()
                        )
                del activations, code, candidate_gate
            del source, x_shards
            torch.cuda.empty_cache()

    for region in ("prompt", "response"):
        expected = sequence[f"{region}_fire_count"][:, selected]
        if not torch.equal(observed_fire[region], expected):
            raise ValueError(f"{region} candidate gates differ from registered fixed-batch summary")

    tensor_output = output / "sequence_candidate_computation.safetensors"
    save_file(
        {
            # Metric slices are views into one [N,S,J,M] backing tensor.  Give
            # safetensors independent storage so alias detection cannot reject
            # the scientifically distinct named outputs.
            f"{region}/{metric}": measured[region][..., metric_index]
            .contiguous()
            .clone()
            for region in ("prompt", "response")
            for metric_index, metric in enumerate(METRIC_NAMES)
        },
        str(tensor_output),
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer, revision=cfg.data.tokenizer_revision, fix_mistral_regex=False
    )
    responses = _load_jsonl(Path(response_dir) / "examples.jsonl")
    masks = _group_masks(metadata)
    response_metrics = measured["response"].numpy()
    all_group = grouped["groups"]["all"]["matrices"]
    scales = np.asarray(
        [
            [
                all_group[name]["base"]["actual_output_rms"],
                all_group[name]["target"]["actual_output_rms"],
                all_group[name]["error_cancellation"]["true_delta_rms"],
                all_group[name]["error_cancellation"]["true_delta_rms"],
                all_group[name]["delta_input"]["actual_output_rms"],
            ]
            for name in matrix_names
        ]
    )
    normalized = response_metrics / np.maximum(scales[None, None, :, :], 1e-12)
    aggregate = np.sqrt((normalized**2).sum(axis=2))
    selection_by_id = {row["component_id"]: row for row in shortlist}
    cards = []
    for local_index, component in enumerate(selected_ids):
        group_results = {}
        for group_name, group_mask in masks.items():
            group_results[group_name] = {
                "sequences": int(group_mask.sum()),
                "normalized_aggregate_mean": {
                    metric: float(aggregate[group_mask, local_index, metric_index].mean())
                    for metric_index, metric in enumerate(METRIC_NAMES)
                },
                "matrix_raw_rms_mean": {
                    name: {
                        metric: float(
                            response_metrics[
                                group_mask, local_index, matrix_index, metric_index
                            ].mean()
                        )
                        for metric_index, metric in enumerate(METRIC_NAMES)
                    }
                    for matrix_index, name in enumerate(matrix_names)
                },
            }
        computation_specificity = standardized_mean_difference(
            aggregate[:, local_index, 2:3],
            masks["risk_target"],
            masks["sports_control_target"],
        )[0]
        risk_change = group_results["risk_target"]["normalized_aggregate_mean"][
            "actual_change_rms"
        ]
        dominant = sorted(
            matrix_names,
            key=lambda name: group_results["risk_target"]["matrix_raw_rms_mean"][name][
                "actual_change_rms"
            ],
            reverse=True,
        )
        cards.append(
            {
                "component_id": component,
                "role": "candidate" if component in selection_by_id else "negative_control",
                "activation_selection": selection_by_id.get(component),
                "posthoc_p1_p5": _posthoc_card(posthoc, component, matrix_names),
                "computation": {
                    "risk_vs_sports_control_effect": float(computation_specificity),
                    "risk_target_normalized_actual_change": risk_change,
                    "dominant_matrices_by_risk_actual_change": dominant,
                    "groups": group_results,
                },
                "representative_exact_token_contexts": _exact_contexts(
                    component, metadata, responses, sequence, tokenizer
                ),
                "limitations": [
                    "exploratory reuse of V3 after prior research decisions",
                    "association is response-level and noncausal",
                    "one component may be part of a distributed mechanism",
                    "local contribution does not repair full-matrix reconstruction residual",
                ],
            }
        )

    candidates = [card for card in cards if card["role"] == "candidate"]
    pre_score = np.asarray(
        [card["activation_selection"]["selection_score"] for card in candidates]
    )
    risk_change = np.asarray(
        [card["computation"]["risk_target_normalized_actual_change"] for card in candidates]
    )
    computation_specificity = np.asarray(
        [card["computation"]["risk_vs_sports_control_effect"] for card in candidates]
    )
    final_score = (
        0.55 * positive_percentile(pre_score)
        + 0.25 * positive_percentile(risk_change)
        + 0.20 * positive_percentile(computation_specificity)
    )
    final_order = np.argsort(-final_score, kind="stable")[:final_size]
    final_ids = [candidates[index]["component_id"] for index in final_order]
    score_by_id = {
        card["component_id"]: float(final_score[index])
        for index, card in enumerate(candidates)
    }
    for card in cards:
        card["final_selection_score"] = score_by_id.get(card["component_id"])
        card["selected_for_detailed_inspection"] = card["component_id"] in final_ids

    result = {
        "schema_version": 1,
        "scientific_status": "EXPLORATORY_NONCAUSAL_V3_ASPD_CANDIDATES",
        "checkpoint": checkpoint_record,
        "measurement": {
            "sequences": len(metadata),
            "fixed_batch_size_sequences": 2,
            "response_level_effects": True,
            "token_positions_treated_as_independent_samples": False,
            "shortlist_size": shortlist_size,
            "negative_controls": controls,
            "negative_control_selection": (
                "label-free active components nearest the general-cache medians of log "
                "mechanism mass and log fire count, with a small preference for lower "
                "mean relative operator change; shortlist IDs excluded"
            ),
            "matrix_names": list(matrix_names),
        },
        "selection_protocol": {
            "activation_rate_transform": "sqrt(response fire count / response tokens)",
            "activation_score": (
                "0.25 risk-vs-control + 0.20 permissive-vs-cautious + 0.10 "
                "risk-vs-neutral + 0.15 support + 0.10 domain breadth + 0.10 P2 "
                "mass + 0.05 risk rate + 0.05 mass-supported P5 change; positive-rank terms"
            ),
            "eligibility": "risk-target support >= 10% and general validation fires >= 32",
            "final_score": (
                "0.55 activation selection + 0.25 exact normalized computation change + "
                "0.20 positive computation specificity"
            ),
        },
        "shortlist_component_ids": shortlist_ids,
        "final_component_ids": final_ids,
        "cards": cards,
        "artifacts": {
            "sequence_computation": {
                "path": tensor_output.name,
                "sha256": sha256_file(tensor_output),
            },
            "grouped_fidelity": {
                "path": str(grouped_fidelity_path),
                "sha256": sha256_file(grouped_fidelity_path),
            },
            "sequence_components": {
                "path": str(sequence_components_path),
                "sha256": sha256_file(sequence_components_path),
            },
            "posthoc": {"path": str(posthoc_path), "sha256": sha256_file(posthoc_path)},
        },
        "scientific_boundary": (
            "Candidate association and rank-one contribution measurement only; no causal "
            "intervention, directed circuit edge, or confirmed risk-specific mechanism claim."
        ),
    }
    if not _finite_tree(result):
        raise FloatingPointError("candidate artifact contains NaN or Inf")
    write_json_immutable(output / "candidate_analysis.json", result)
    return result


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite_tree(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)
