"""Gate localization and label-free causal-gate calibration for the ASPD P6 pilot."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.causal import (
    ThresholdGateCalibration,
    calibrate_threshold_gate,
    temporal_region,
    threshold_causal_gate,
)
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.fidelity import sha256_file, validate_checkpoint_provenance
from aspd.multimodel.model import LinearSharedEncoder, MultiModelASPD
from aspd.multimodel.sports_risk import effective_grounding_batch
from aspd.multimodel.training import build_model_from_cache

P6_CANDIDATE_IDS = (3228, 290, 2445, 508, 378, 496)
REGION_NAMES = (
    "user_prompt",
    "decision_boundary",
    "response_opening",
    "response_early",
    "response_mid",
    "response_conclusion",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite_tree(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


@torch.no_grad()
def calibrate_from_general_validation(
    model: MultiModelASPD,
    cache: PairedActivationCache,
    posthoc: dict[str, Tensor],
    component_ids: tuple[int, ...],
    calibration_tokens: int,
    device: torch.device,
) -> ThresholdGateCalibration:
    """Calibrate on a fixed unlabeled prefix of the general validation cache."""

    if calibration_tokens <= 0:
        raise ValueError("calibration_tokens must be positive")
    selected = torch.tensor(component_ids, dtype=torch.long, device=device)
    collected: list[Tensor] = []
    observed = 0
    for raw in cache.iter_batches("validation", batch_size_sequences=2, shuffle=False):
        valid = raw["valid_tokens"].to(device)  # type: ignore[union-attr]
        activations = [value.to(device) for value in raw["R"]]  # type: ignore[union-attr]
        preactivations = model.encoder(activations, valid)
        values = preactivations.index_select(-1, selected)[valid].float().cpu()
        take = min(values.shape[0], calibration_tokens - observed)
        collected.append(values[:take])
        observed += take
        if observed == calibration_tokens:
            break
    if observed != calibration_tokens:
        raise ValueError(f"requested {calibration_tokens} calibration tokens, found {observed}")
    full_validation_tokens = int(cache.cfg.data.validation_tokens)
    target_rates = posthoc["fire_count_C"][list(component_ids)].float() / full_validation_tokens
    return calibrate_threshold_gate(
        torch.cat(collected, dim=0), component_ids, target_rates
    )


def _group_keys(row: dict[str, Any]) -> list[str]:
    keys = ["all", row["category"], f'{row["category"]}/{row["generator_side"]}']
    score = row.get("behavioral_score")
    if score is not None:
        keys.append("high_permissiveness" if score >= 2 else "cautious")
    return keys


def _aggregate(examples: list[dict[str, Any]], component_ids: tuple[int, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for component in component_ids:
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for example in examples:
            record = example["components"][str(component)]
            for key in _group_keys(example):
                by_group[key].append(record)
        groups = {}
        for key, records in sorted(by_group.items()):
            offline_sets = [set(row["offline_positions"]) for row in records]
            causal_sets = [set(row["causal_positions"]) for row in records]
            intersection = sum(len(a & b) for a, b in zip(offline_sets, causal_sets, strict=True))
            union = sum(len(a | b) for a, b in zip(offline_sets, causal_sets, strict=True))
            offline_total = sum(len(value) for value in offline_sets)
            causal_total = sum(len(value) for value in causal_sets)
            region_offline = {
                region: sum(row["offline_regions"][region] for row in records)
                for region in REGION_NAMES
            }
            region_causal = {
                region: sum(row["causal_regions"][region] for row in records)
                for region in REGION_NAMES
            }
            groups[key] = {
                "sequences": len(records),
                "offline_response_support": sum(
                    any(position >= row["prompt_length"] for position in row["offline_positions"])
                    for row in records
                )
                / len(records),
                "causal_response_support": sum(
                    any(position >= row["prompt_length"] for position in row["causal_positions"])
                    for row in records
                )
                / len(records),
                "offline_decision_boundary_support": sum(
                    row["offline_regions"]["decision_boundary"] > 0 for row in records
                )
                / len(records),
                "causal_decision_boundary_support": sum(
                    row["causal_regions"]["decision_boundary"] > 0 for row in records
                )
                / len(records),
                "offline_fire_count": offline_total,
                "causal_fire_count": causal_total,
                "intersection_fire_count": intersection,
                "union_fire_count": union,
                "offline_recall": intersection / max(offline_total, 1),
                "causal_precision": intersection / max(causal_total, 1),
                "jaccard": intersection / max(union, 1),
                "offline_region_fire_count": region_offline,
                "causal_region_fire_count": region_causal,
            }
        output[str(component)] = groups
    return output


def _top_contexts(
    examples: list[dict[str, Any]],
    component_ids: tuple[int, ...],
    tokenizer: Any,
) -> dict[str, list[dict[str, Any]]]:
    contexts: dict[str, list[dict[str, Any]]] = {}
    for component in component_ids:
        selected: list[tuple[float, dict[str, Any], int]] = []
        for category in ("risk", "sports_control", "neutral"):
            eligible: list[tuple[float, dict[str, Any], int]] = []
            for example in examples:
                if example["category"] != category:
                    continue
                record = example["components"][str(component)]
                for position, value in zip(
                    record["causal_positions"], record["causal_preactivations"], strict=True
                ):
                    eligible.append((value, example, position))
            eligible.sort(key=lambda item: item[0], reverse=True)
            selected.extend(eligible[:2])
        rows = []
        for value, example, position in selected:
            token_ids = example["token_ids"]
            start, stop = max(0, position - 10), min(len(token_ids), position + 11)
            rows.append(
                {
                    "example_id": example["example_id"],
                    "category": example["category"],
                    "generator_side": example["generator_side"],
                    "sport_domain": example["sport_domain"],
                    "behavioral_score": example["behavioral_score"],
                    "position": position,
                    "region": temporal_region(
                        position, example["prompt_token_count"], example["sequence_length"]
                    ),
                    "preactivation": value,
                    "window_start": start,
                    "window_stop": stop,
                    "center_offset": position - start,
                    "token_ids": token_ids[start:stop],
                    "text": tokenizer.decode(
                        token_ids[start:stop], clean_up_tokenization_spaces=False
                    ),
                }
            )
        contexts[str(component)] = rows
    return contexts


@torch.no_grad()
def localize_p6_gates(
    cfg: MultiModelExperimentConfig,
    checkpoint: str | Path,
    *,
    expected_checkpoint_sha256: str,
    matrix_cache_dir: str | Path,
    grouped_fidelity_path: str | Path,
    sequence_components_path: str | Path,
    posthoc_path: str | Path,
    output_dir: str | Path,
    calibration_tokens: int = 16384,
    device_name: str = "cuda",
    component_ids: tuple[int, ...] = P6_CANDIDATE_IDS,
) -> dict[str, Any]:
    """Compare registered offline BatchTopK gates with a label-free causal policy."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    if cfg.sparsity.diffing != "D0" or cfg.sparsity.selection_score != "S1":
        raise ValueError("this registered P6 pilot requires the frozen D0/S1 checkpoint")
    cache = PairedActivationCache(cfg)
    if cache.token_manifest.get("cache_fingerprint") != (
        "71b268fc2190df5c7ddf0a43166242e714f4be92e2db20bf3405a0102f5037ce"
    ):
        raise ValueError("unexpected general1m training-cache fingerprint")
    checkpoint_record = validate_checkpoint_provenance(
        cfg,
        cache,
        checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    device = torch.device(device_name)
    model = build_model_from_cache(cfg, cache)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state["model"], strict=True)
    del state
    model = model.to(device).float().eval()
    if not isinstance(model.encoder, LinearSharedEncoder):
        raise ValueError("threshold-causal proof currently requires the frozen linear encoder")

    posthoc = load_file(str(posthoc_path))
    calibration = calibrate_from_general_validation(
        model, cache, posthoc, component_ids, calibration_tokens, device
    )
    if not torch.isfinite(calibration.thresholds).all():
        raise FloatingPointError("candidate threshold calibration produced NaN or Inf")

    matrix_root = Path(matrix_cache_dir)
    manifest = json.loads((matrix_root / "manifest.json").read_text())
    metadata = _load_jsonl(matrix_root / manifest["metadata_file"])
    source_root = Path(manifest["paired_activation_cache"]["path"])
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    grouped = json.loads(Path(grouped_fidelity_path).read_text())
    sequence = load_file(str(sequence_components_path))
    if grouped["component_sequence_summary"]["sha256"] != sha256_file(
        sequence_components_path
    ):
        raise ValueError("sequence-component artifact differs from grouped fidelity")
    norms = [float(row["r_rms_norm"]) for row in cache.model_manifests]
    selected = torch.tensor(component_ids, dtype=torch.long, device=device)
    examples_by_index: dict[int, dict[str, Any]] = {}
    observed_prompt = torch.zeros((len(metadata), len(component_ids)), dtype=torch.int32)
    observed_response = torch.zeros_like(observed_prompt)

    for split in ("train", "validation"):
        for entry in source_manifest["splits"][split]["shards"]:
            source = torch.load(
                source_root / entry["path"], map_location="cpu", weights_only=True
            )
            matrix_shards = [
                torch.load(
                    matrix_root / "models" / spec.name / entry["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                for spec in cfg.models
            ]
            rows = int(source["input_ids"].shape[0])
            for start in range(0, rows, 2):
                stop = min(start + 2, rows)
                valid = source["valid_tokens"][start:stop].to(device)
                base_r, target_r = effective_grounding_batch(
                    source, matrix_shards, start, stop
                )
                activations = [
                    base_r.to(device).float() / norms[0],
                    target_r.to(device).float() / norms[1],
                ]
                preactivations = model.encoder(activations, valid)
                offline = model.encode(activations, valid).gate.index_select(-1, selected)
                causal = threshold_causal_gate(preactivations, calibration)
                selected_pre = preactivations.index_select(-1, selected)
                for local_index, example_index in enumerate(
                    source["example_indices"][start:stop].tolist()
                ):
                    meta = metadata[example_index]
                    length = int(valid[local_index].sum())
                    prompt_length = int(meta["prompt_token_count"])
                    token_ids = source["input_ids"][start + local_index, :length].tolist()
                    component_rows: dict[str, Any] = {}
                    for component_local, component in enumerate(component_ids):
                        offline_positions = (
                            offline[local_index, :length, component_local]
                            .nonzero(as_tuple=False)
                            .flatten()
                            .tolist()
                        )
                        causal_positions = (
                            causal[local_index, :length, component_local]
                            .nonzero(as_tuple=False)
                            .flatten()
                            .tolist()
                        )
                        causal_values = selected_pre[
                            local_index, causal_positions, component_local
                        ].float().cpu().tolist()
                        offline_regions = {region: 0 for region in REGION_NAMES}
                        causal_regions = {region: 0 for region in REGION_NAMES}
                        for position in offline_positions:
                            offline_regions[
                                temporal_region(position, prompt_length, length)
                            ] += 1
                        for position in causal_positions:
                            causal_regions[
                                temporal_region(position, prompt_length, length)
                            ] += 1
                        component_rows[str(component)] = {
                            "prompt_length": prompt_length,
                            "offline_positions": offline_positions,
                            "causal_positions": causal_positions,
                            "causal_preactivations": causal_values,
                            "offline_regions": offline_regions,
                            "causal_regions": causal_regions,
                        }
                        observed_prompt[example_index, component_local] = sum(
                            position < prompt_length for position in offline_positions
                        )
                        observed_response[example_index, component_local] = sum(
                            position >= prompt_length for position in offline_positions
                        )
                    examples_by_index[example_index] = {
                        "example_index": example_index,
                        "example_id": meta["example_id"],
                        "category": meta["category"],
                        "generator_side": meta["generator_side"],
                        "sport_domain": meta["sport_domain"],
                        "behavioral_score": meta["behavioral_score"],
                        "prompt_token_count": prompt_length,
                        "sequence_length": length,
                        "token_ids": token_ids,
                        "components": component_rows,
                    }
            del source, matrix_shards
            torch.cuda.empty_cache()

    examples = [examples_by_index[index] for index in range(len(metadata))]
    if not torch.equal(
        observed_prompt, sequence["prompt_fire_count"][:, list(component_ids)].int()
    ):
        raise ValueError("recomputed offline prompt gates differ from registered V3 artifact")
    if not torch.equal(
        observed_response, sequence["response_fire_count"][:, list(component_ids)].int()
    ):
        raise ValueError("recomputed offline response gates differ from registered V3 artifact")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer,
        revision=cfg.data.tokenizer_revision,
        fix_mistral_regex=False,
    )
    payload = {
        "schema_version": 1,
        "scientific_status": "EXPLORATORY_P6_GATE_LOCALIZATION",
        "checkpoint": checkpoint_record,
        "component_ids": list(component_ids),
        "gate_policy": {
            "name": "general_validation_calibrated_component_threshold",
            "equation": "g[t,c] = 1[a[t,c] > threshold[c]]",
            "encoder_requirement": "frozen linear shared encoder; Qwen grounding is causal",
            "calibration_source": "first fixed unlabeled general-validation tokens",
            "calibration_tokens": calibration.calibration_tokens,
            "target_rate_source": "full general-validation offline BatchTopK fire_count / tokens",
            "uses_v3_labels": False,
            "uses_future_tokens": False,
            "uses_other_batch_sequences": False,
            "mathematically_identical_to_offline_batch_topk": False,
            "thresholds": {
                str(component): {
                    "threshold": float(calibration.thresholds[index]),
                    "target_rate": float(calibration.target_rates[index]),
                    "realized_calibration_rate": float(calibration.realized_rates[index]),
                }
                for index, component in enumerate(component_ids)
            },
        },
        "temporal_regions": {
            "user_prompt": "all prompt positions except its final token",
            "decision_boundary": "the final prompt token whose logits predict the first response token",
            "response_opening": "first eight response tokens",
            "response_early_mid_conclusion": "non-overlapping thirds of the remaining response",
            "explicit_span_annotation": (
                "unavailable in frozen V3; judge scores are response-level, so no dangerous or "
                "cautious token spans were invented"
            ),
        },
        "measurement": {
            "sequences": len(examples),
            "fixed_offline_batch_size_sequences": 2,
            "exact_positions_saved": True,
            "same_exact_tokens_for_base_and_target": True,
        },
        "aggregate": _aggregate(examples, component_ids),
        "representative_causal_contexts": _top_contexts(
            examples, component_ids, tokenizer
        ),
        "examples": examples,
        "artifacts": {
            "matrix_cache": {
                "path": str(matrix_root),
                "fingerprint": manifest["cache_fingerprint"],
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
            "Gate localization and temporal-validity measurement only; no live Qwen intervention, "
            "generation, behavioral re-judging, or causal claim."
        ),
    }
    if not _finite_tree(payload):
        raise FloatingPointError("gate-localization artifact contains NaN or Inf")
    path = output / "gate_localization.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

