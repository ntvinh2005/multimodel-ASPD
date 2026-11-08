"""Distributed read-only restore smoke and held-out diagnostic for ASPD v3 500M."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from aspd.multimodel.v3.activation_targets import (
    ActivationTargetNormalization,
    oracle_denormalize,
    prepare_activation_target,
)
from aspd.multimodel.v3.data import HashSplitPolicy
from aspd.multimodel.v3.diagnostic import (
    DiagnosticContract,
    SequentialDocumentFVU,
    TensorMetricAccumulator,
    cluster_bootstrap_mean_ci,
    feature_health_from_counts,
    sha256_file,
    validate_frozen_provenance,
)
from aspd.multimodel.v3.diagnostic_stream import ValidationAlignedAnchorStream
from aspd.multimodel.v3.hooks import assert_frozen_backbone, logical_weight_bias
from aspd.multimodel.v3.main_2b_config import MODEL_ORDER_2B, load_main_2b_v1_config
from aspd.multimodel.v3.model_owned import LocalCaptureBatch, ModelOwnedMain2B
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.objective import matryoshka_boundaries
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, load_pinned_pile_dataset
from aspd.multimodel.v3.registry import build_model_registry_2b
from aspd.multimodel.v3.training_capture import capture_family_pool


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("checkpoint-smoke", "heldout"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-smoke-report", type=Path)
    parser.add_argument("--derived-checkpoint", type=Path)
    parser.add_argument("--derived-checkpoint-manifest-sha256")
    parser.add_argument("--controlled-arm", choices=("shared_control", "gemma_specific"))
    parser.add_argument("--controlled-run-manifest", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _broadcast(value: Any, source: int = 0) -> Any:
    values = [value]
    dist.broadcast_object_list(values, src=source)
    return values[0]


def _pad_id(tokenizer: Any) -> int:
    value = tokenizer.pad_token_id
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        raise ValueError("tokenizer exposes neither padding nor EOS token")
    return int(value)


def _parameter_signature(module: torch.nn.Module) -> dict[str, float | int]:
    count = 0
    total = 0.0
    square = 0.0
    for parameter in module.parameters():
        value = parameter.detach().double()
        if not torch.isfinite(value).all():
            raise FloatingPointError("non-finite model parameter")
        count += value.numel()
        total += float(value.sum())
        square += float(value.square().sum())
    return {"parameter_count": count, "sum": total, "square_sum": square}


def _finite_state_report(module: torch.nn.Module) -> dict[str, Any]:
    tensors = 0
    values = 0
    for tensor in module.state_dict().values():
        tensors += 1
        values += tensor.numel()
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise FloatingPointError("restored ASPD state contains NaN/Inf")
    return {"status": "PASS", "tensors": tensors, "values": values}


def _shared_digest(model: ModelOwnedMain2B) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.shared_encoders.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_controlled_checkpoint(
    *,
    checkpoint: Path,
    expected_manifest_sha256: str,
    run_manifest_path: Path,
    arm: str,
    contract: DiagnosticContract,
) -> dict[str, Any]:
    """Bind a derived controlled checkpoint to the immutable 500M source."""

    required = (
        checkpoint / "manifest.json",
        checkpoint / "COMPLETE",
        run_manifest_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"controlled evaluation input is missing: {missing}")
    observed_sha = sha256_file(checkpoint / "manifest.json")
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    marker = json.loads((checkpoint / "COMPLETE").read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    controlled = run_manifest.get("controlled_retrain") or {}
    observed = {
        "arm": controlled.get("arm"),
        "checkpoint_manifest_sha256": observed_sha,
        "checkpoint_marker_sha256": marker.get("manifest_sha256"),
        "checkpoint_step": manifest.get("step"),
        "checkpoint_status": manifest.get("status"),
        "checkpoint_world_size": manifest.get("world_size"),
        "config_file_sha256": manifest.get("config_file_sha256"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
        "source_checkpoint_manifest_sha256": controlled.get("source_checkpoint_manifest_sha256"),
        "source_checkpoint_step": controlled.get("source_checkpoint_step"),
        "source_accepted_exposures": controlled.get("source_accepted_exposures"),
        "additional_optimizer_steps": controlled.get("additional_optimizer_steps"),
        "additional_accepted_exposures": controlled.get("additional_accepted_exposures"),
        "frozen_scope": controlled.get("frozen_scope"),
        "optimizer_state": controlled.get("optimizer_state"),
        "training_git_sha": run_manifest.get("git_sha"),
    }
    expected = {
        "arm": arm,
        "checkpoint_manifest_sha256": expected_manifest_sha256,
        "checkpoint_marker_sha256": expected_manifest_sha256,
        "checkpoint_step": 1221,
        "checkpoint_status": "COMPLETE",
        "checkpoint_world_size": 3,
        "config_file_sha256": contract.source_config_sha256,
        "config_fingerprint": contract.source_config_fingerprint,
        "dataset_manifest_sha256": contract.dataset_manifest_sha256,
        "source_checkpoint_manifest_sha256": contract.checkpoint_manifest_sha256,
        "source_checkpoint_step": 61036,
        "source_accepted_exposures": 500006912,
        "additional_optimizer_steps": 1221,
        "additional_accepted_exposures": 10002432,
        "frozen_scope": ["rank_one_matrix_factors"],
        "optimizer_state": "fresh_for_both_arms",
    }
    mismatch = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatch:
        raise ValueError(f"controlled checkpoint provenance mismatch: {mismatch}")
    return {"status": "PASS", **observed}


def _capture(
    pool: AlignedAnchorPool,
    *,
    local_key: str,
    tokenizer: Any,
    loaded: Any,
    registry: Mapping[str, Any],
    device: torch.device,
) -> LocalCaptureBatch:
    family = capture_family_pool(
        loaded.forward_model,
        registry[local_key],
        pool.models[local_key],
        pad_token_id=_pad_id(tokenizer),
        device=device,
        early_exit=True,
    )
    return LocalCaptureBatch(
        model_key=local_key,
        activations={"pre": family.residual_pre, "mid": family.residual_mid},
        matrix_inputs=family.matrix_inputs,
        matrix_outputs=family.matrix_outputs,
        valid_anchors=torch.ones(1, pool.anchor_count, dtype=torch.bool, device=device),
    )


def _accumulate(
    values: dict[str, TensorMetricAccumulator],
    key: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    width = target.shape[-1]
    values.setdefault(key, TensorMetricAccumulator(width)).update(
        prediction.reshape(-1, width), target.reshape(-1, width)
    )


def _jaccard_report(
    chunks: list[torch.Tensor],
    counts: torch.Tensor,
    *,
    top: int,
    minimum: int,
    device: torch.device,
) -> dict[str, Any]:
    eligible = (counts >= minimum).nonzero(as_tuple=True)[0]
    selected = eligible[counts[eligible].argsort(descending=True)][:top]
    if selected.numel() < 2:
        return {"eligible_features": int(eligible.numel()), "evaluated_pairs": 0, "pairs": []}
    intersections = torch.zeros(selected.numel(), selected.numel(), device=device)
    for chunk in chunks:
        sample = chunk[:, selected].to(device=device, dtype=torch.float32)
        intersections.add_(sample.T @ sample)
    intersections = intersections.cpu().round().to(torch.int64)
    selected_counts = counts[selected].to(torch.int64)
    pairs = []
    for left in range(selected.numel()):
        for right in range(left + 1, selected.numel()):
            intersection = int(intersections[left, right])
            union = int(selected_counts[left] + selected_counts[right] - intersection)
            pairs.append(
                {
                    "left": int(selected[left]),
                    "right": int(selected[right]),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else 0.0,
                }
            )
    all_pairs = counts.numel() * (counts.numel() - 1) // 2
    return {
        "eligible_features": int(eligible.numel()),
        "evaluated_features": int(selected.numel()),
        "evaluated_pairs": len(pairs),
        "coverage_of_all_pairs": len(pairs) / all_pairs,
        "pairs": pairs,
    }


def _aggregate_ratio(rows: list[dict[str, Any]]) -> dict[str, float]:
    sse = sum(float(row["sse"]) for row in rows)
    denominator = sum(float(row["centered_target_ss"]) for row in rows)
    return {
        "sse": sse,
        "centered_target_ss": denominator,
        "fvu": sse / denominator,
        "r2": 1 - sse / denominator,
    }


def _local_evaluation(
    *,
    pools: int,
    model: ModelOwnedMain2B,
    loaded: Any,
    tokenizer: Any,
    registry: Mapping[str, Any],
    local_key: str,
    device: torch.device,
    next_pool,
    contract: DiagnosticContract,
    activation_target_normalization: ActivationTargetNormalization = "none",
    activation_target_epsilon: float = 1e-6,
) -> dict[str, Any]:
    metrics: dict[str, TensorMetricAccumulator] = {}
    document_metrics: dict[str, SequentialDocumentFVU] = {}
    target_norm_stats: dict[str, dict[str, float | int]] = {}
    n_features = 8192
    fire_counts = {site: torch.zeros(n_features, dtype=torch.int64) for site in ("pre", "mid")}
    first_counts = {site: torch.zeros(n_features, dtype=torch.int64) for site in ("pre", "mid")}
    second_counts = {site: torch.zeros(n_features, dtype=torch.int64) for site in ("pre", "mid")}
    magnitude_sum = {site: torch.zeros(n_features, dtype=torch.float64) for site in ("pre", "mid")}
    magnitude_square = {
        site: torch.zeros(n_features, dtype=torch.float64) for site in ("pre", "mid")
    }
    magnitude_max = {site: torch.zeros(n_features, dtype=torch.float64) for site in ("pre", "mid")}
    contribution_sum = {
        site: torch.zeros(n_features, dtype=torch.float64) for site in ("pre", "mid")
    }
    contribution_square = {
        site: torch.zeros(n_features, dtype=torch.float64) for site in ("pre", "mid")
    }
    contribution_count = {
        site: torch.zeros(n_features, dtype=torch.int64) for site in ("pre", "mid")
    }
    conditional_error_sum = {
        site: torch.zeros(n_features, dtype=torch.float64) for site in ("pre", "mid")
    }
    firing_chunks: dict[str, list[torch.Tensor]] = {"pre": [], "mid": []}
    active_histogram: dict[str, Counter[int]] = {"pre": Counter(), "mid": Counter()}
    raw_logit_stats = {
        site: {
            "count": 0,
            "negative": 0,
            "zero": 0,
            "positive": 0,
            "sum": 0.0,
            "square_sum": 0.0,
        }
        for site in ("pre", "mid")
    }
    positive_candidate_chunks: dict[str, list[torch.Tensor]] = {"pre": [], "mid": []}
    boundaries = matryoshka_boundaries(n_features, (0.0625, 0.0625, 0.125, 0.25, 0.5))
    accepted = 0
    documents: set[str] = set()

    for pool_index in range(pools):
        pool = next_pool()
        documents.update(pool.document_ids)
        capture = _capture(
            pool,
            local_key=local_key,
            tokenizer=tokenizer,
            loaded=loaded,
            registry=registry,
            device=device,
        )
        # Reproduce the model-owned trainer's forward precision. ASPD master
        # parameters are restored in FP32 while frozen captures are BF16; the
        # production trainer therefore executes this path under CUDA autocast.
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
        ):
            scores, sparse, continuous, binary = model.encode(capture, update_statistics=False)
        del scores
        accepted += pool.anchor_count
        document_index = pool.models[local_key].anchor_document_indices

        for site in ("pre", "mid"):
            valid_logits = model.latest_raw_logits[site][capture.valid_anchors].detach().float()
            logit_state = raw_logit_stats[site]
            logit_state["count"] += int(valid_logits.numel())
            logit_state["negative"] += int((valid_logits < 0).sum())
            logit_state["zero"] += int((valid_logits == 0).sum())
            logit_state["positive"] += int((valid_logits > 0).sum())
            logit_state["sum"] += float(valid_logits.double().sum())
            logit_state["square_sum"] += float(valid_logits.double().square().sum())
            positive_candidate_chunks[site].append(
                (valid_logits > 0).sum(dim=-1).cpu().to(torch.float32)
            )
            code = sparse[site].values * capture.valid_anchors[..., None]
            gate = (code > 0).reshape(-1, n_features)
            counts = gate.sum(dim=0).cpu().to(torch.int64)
            fire_counts[site] += counts
            (first_counts if pool_index < pools // 2 else second_counts)[site] += counts
            firing_chunks[site].append(gate.cpu())
            for value, count in Counter(gate.sum(dim=1).cpu().tolist()).items():
                active_histogram[site][int(value)] += int(count)
            position, feature = gate.nonzero(as_tuple=True)
            magnitudes = code.reshape(-1, n_features)[position, feature].float().cpu()
            magnitude_sum[site].index_add_(0, feature.cpu(), magnitudes.double())
            magnitude_square[site].index_add_(0, feature.cpu(), magnitudes.double().square())
            magnitude_max[site].scatter_reduce_(
                0, feature.cpu(), magnitudes.double(), reduce="amax", include_self=True
            )

            decoders = model.activation_banks[site].decoders[local_key]
            for layer_offset, (decoder, target) in enumerate(
                zip(decoders, capture.activations[site], strict=True)
            ):
                prepared = prepare_activation_target(
                    target,
                    capture.valid_anchors,
                    mode=activation_target_normalization,
                    epsilon=activation_target_epsilon,
                )
                layer = registry[local_key].layers[layer_offset]
                norm_key = f"{local_key}/{site}/L{layer:02d}"
                valid_norms = prepared.oracle_norm[capture.valid_anchors].double().cpu().flatten()
                norm_state = target_norm_stats.setdefault(
                    norm_key,
                    {
                        "count": 0,
                        "sum": 0.0,
                        "square_sum": 0.0,
                        "minimum": math.inf,
                        "maximum": -math.inf,
                    },
                )
                norm_state["count"] += int(valid_norms.numel())
                norm_state["sum"] += float(valid_norms.sum())
                norm_state["square_sum"] += float(valid_norms.square().sum())
                norm_state["minimum"] = min(
                    float(norm_state["minimum"]), float(valid_norms.min())
                )
                norm_state["maximum"] = max(
                    float(norm_state["maximum"]), float(valid_norms.max())
                )
                decoded = decoder.forward_dense_grouped(
                    code, boundaries=boundaries, include_bias=True
                )
                assert isinstance(decoded, tuple) and len(decoded) == 6
                scale = model.activation_scale(site, layer_offset)
                if scale is not None:
                    bias = (
                        decoder.bias.float().expand_as(target)
                        if decoder.bias is not None
                        else torch.zeros_like(target, dtype=torch.float32)
                    )
                    decoded = tuple(
                        prediction.float() + (scale.float() - 1.0) * (prediction.float() - bias)
                        for prediction in decoded
                    )
                for prefix_index, prediction in enumerate(decoded):
                    if activation_target_normalization == "none":
                        key = (
                            f"activation/{local_key}/{site}/L{layer:02d}/"
                            f"prefix_{prefix_index}"
                        )
                        _accumulate(metrics, key, prediction, prepared.target)
                    else:
                        normalized_key = (
                            f"activation_normalized/{local_key}/{site}/L{layer:02d}/"
                            f"prefix_{prefix_index}"
                        )
                        oracle_key = (
                            f"activation_original_oracle/{local_key}/{site}/L{layer:02d}/"
                            f"prefix_{prefix_index}"
                        )
                        _accumulate(metrics, normalized_key, prediction, prepared.target)
                        _accumulate(
                            metrics,
                            oracle_key,
                            oracle_denormalize(prediction, prepared.oracle_norm),
                            target,
                        )
                if activation_target_normalization == "none":
                    full = f"activation/{local_key}/{site}/L{layer:02d}/prefix_5"
                    document_metrics.setdefault(
                        full, SequentialDocumentFVU(target.shape[-1])
                    ).update(decoded[-1], target, document_index, pool.document_ids)
                else:
                    normalized_full = (
                        f"activation_normalized/{local_key}/{site}/L{layer:02d}/prefix_5"
                    )
                    oracle_full = (
                        f"activation_original_oracle/{local_key}/{site}/L{layer:02d}/prefix_5"
                    )
                    document_metrics.setdefault(
                        normalized_full, SequentialDocumentFVU(target.shape[-1])
                    ).update(decoded[-1], prepared.target, document_index, pool.document_ids)
                    document_metrics.setdefault(
                        oracle_full, SequentialDocumentFVU(target.shape[-1])
                    ).update(
                        oracle_denormalize(decoded[-1], prepared.oracle_norm),
                        target,
                        document_index,
                        pool.document_ids,
                    )

        for key, target_spec in model.targets.items():
            site = "pre" if target_spec.grounding_kind == "residual_pre" else "mid"
            with torch.no_grad():
                prediction = model.matrix_bank.reconstruct(
                    key,
                    capture.matrix_inputs[key],
                    sparse[site].values.detach(),
                    capture.valid_anchors,
                    code_semantics="binary_support",
                    support=binary[site],
                    compute_dtype=torch.bfloat16,
                    sparse_backend="dense",
                )
            target = capture.matrix_outputs[key]
            metric_key = f"internal/{key}"
            _accumulate(metrics, metric_key, prediction, target)
            document_metrics.setdefault(metric_key, SequentialDocumentFVU(target.shape[-1])).update(
                prediction, target, document_index, pool.document_ids
            )

            support = binary[site]
            component = model.matrix_bank.components[key.replace(".", "__dot__")]
            positions = support.position
            features = support.feature
            x = capture.matrix_inputs[key].reshape(-1, target_spec.d_in)[positions].float()
            reads = (x * component.V[features].float()).sum(dim=-1)
            norms = reads.abs() * component.U[features].float().norm(dim=-1)
            feature_cpu = features.cpu()
            contribution_sum[site].index_add_(0, feature_cpu, norms.double().cpu())
            contribution_square[site].index_add_(0, feature_cpu, norms.double().square().cpu())
            contribution_count[site].index_add_(
                0, feature_cpu, torch.ones_like(feature_cpu, dtype=torch.int64)
            )
            residual_norm = (
                (prediction.float() - target.float()).reshape(-1, target_spec.d_out).norm(dim=-1)
            )
            conditional_error_sum[site].index_add_(
                0, feature_cpu, residual_norm[positions].double().cpu()
            )
        del capture, sparse, continuous, binary

    finalized = {}
    for key, accumulator in metrics.items():
        value = accumulator.finalize()
        document_metric = document_metrics.get(key)
        if document_metric is not None:
            cluster_report = document_metric.finalize()
            clusters = cluster_report.pop("values")
            value["document_fvu_by_id"] = cluster_report.pop("values_by_document")
            offset = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
            value["document_mean_fvu_bootstrap"] = cluster_bootstrap_mean_ci(
                clusters,
                seed=contract.seed + offset,
                replicates=contract.bootstrap_replicates,
            )
            value["document_cluster_accounting"] = cluster_report
        finalized[key] = value

    finalized_norms = {}
    for key, value in target_norm_stats.items():
        count = int(value["count"])
        finalized_norms[key] = {
            "count": count,
            "minimum": float(value["minimum"]),
            "mean": float(value["sum"]) / count,
            "rms": math.sqrt(float(value["square_sum"]) / count),
            "maximum": float(value["maximum"]),
        }

    dictionary = {}
    feature_rows = []
    for site in ("pre", "mid"):
        health = feature_health_from_counts(fire_counts[site], accepted)
        logit_state = raw_logit_stats[site]
        logit_count = int(logit_state["count"])
        logit_mean = float(logit_state["sum"]) / logit_count
        logit_variance = max(
            0.0,
            float(logit_state["square_sum"]) / logit_count - logit_mean**2,
        )
        candidates = torch.cat(positive_candidate_chunks[site])
        candidate_quantiles = torch.quantile(
            candidates, torch.tensor([0.01, 0.5, 0.9, 0.99])
        )
        readout_bias = model.shared_encoders[site].readout.output.bias.detach().float().cpu()
        left = first_counts[site].float().log1p()
        right = second_counts[site].float().log1p()
        stability_value = float(torch.corrcoef(torch.stack((left, right)))[0, 1])
        # Correlation is undefined if all features have identical counts in one
        # half.  Preserve that diagnostic boundary instead of serializing NaN.
        stability = stability_value if math.isfinite(stability_value) else None
        jaccard = _jaccard_report(
            firing_chunks[site],
            fire_counts[site],
            top=contract.jaccard_top_features,
            minimum=contract.jaccard_min_fires,
            device=device,
        )
        contribution_total = float(contribution_sum[site].sum())
        contribution_ordered = contribution_sum[site].sort(descending=True).values

        contribution_shares = {
            fraction: (
                float(contribution_ordered[: max(1, math.ceil(n_features * fraction))].sum())
                / contribution_total
                if contribution_total > 0
                else 0.0
            )
            for fraction in (0.01, 0.05, 0.10)
        }

        contribution_probabilities = (
            contribution_sum[site] / contribution_total
            if contribution_total > 0
            else contribution_sum[site]
        )
        positive_contribution = contribution_probabilities > 0
        contribution_entropy = float(
            -(
                contribution_probabilities[positive_contribution]
                * contribution_probabilities[positive_contribution].log()
            ).sum()
        )
        health.update(
            {
                "site": site,
                "subset_log1p_frequency_correlation": stability,
                "active_features_per_anchor_histogram": dict(
                    sorted(active_histogram[site].items())
                ),
                "raw_logit_mean": logit_mean,
                "raw_logit_std": math.sqrt(logit_variance),
                "raw_logit_negative_fraction": int(logit_state["negative"]) / logit_count,
                "raw_logit_zero_fraction": int(logit_state["zero"]) / logit_count,
                "raw_logit_positive_fraction": int(logit_state["positive"]) / logit_count,
                "positive_candidates_per_anchor_mean": float(candidates.mean()),
                "positive_candidates_per_anchor_std": float(candidates.std(unbiased=False)),
                "positive_candidates_per_anchor_q01": float(candidate_quantiles[0]),
                "positive_candidates_per_anchor_median": float(candidate_quantiles[1]),
                "positive_candidates_per_anchor_q90": float(candidate_quantiles[2]),
                "positive_candidates_per_anchor_q99": float(candidate_quantiles[3]),
                "readout_bias_mean": float(readout_bias.mean()),
                "readout_bias_std": float(readout_bias.std(unbiased=False)),
                "readout_bias_negative_fraction": float((readout_bias < 0).float().mean()),
                "readout_bias_min": float(readout_bias.min()),
                "readout_bias_max": float(readout_bias.max()),
                "training_used_features": int(
                    (model.gates[site].feature_fire_counts > 0).sum().item()
                ),
                "training_recently_dead_features": int(
                    model.gates[site].dead_features.sum().item()
                ),
                "training_dead_window_pools": 320,
                "training_dead_window_anchors": 320 * 2048,
                "effective_dictionary_size_contribution_entropy": (
                    math.exp(contribution_entropy) if contribution_total > 0 else 0.0
                ),
                "top_1_percent_contribution_share": contribution_shares[0.01],
                "top_5_percent_contribution_share": contribution_shares[0.05],
                "top_10_percent_contribution_share": contribution_shares[0.10],
                "jaccard": jaccard,
            }
        )
        dictionary[site] = health
        for feature in range(n_features):
            fires = int(fire_counts[site][feature])
            contributions = int(contribution_count[site][feature])
            feature_rows.append(
                {
                    "model": local_key,
                    "site": site,
                    "feature": feature,
                    "heldout_fires": fires,
                    "heldout_firing_rate": fires / accepted,
                    "activation_mean_when_fired": (
                        float(magnitude_sum[site][feature]) / fires if fires else 0.0
                    ),
                    "activation_rms_when_fired": (
                        math.sqrt(float(magnitude_square[site][feature]) / fires) if fires else 0.0
                    ),
                    "activation_max": float(magnitude_max[site][feature]),
                    "matrix_component_events": contributions,
                    "matrix_contribution_l2_mean": (
                        float(contribution_sum[site][feature]) / contributions
                        if contributions
                        else 0.0
                    ),
                    "matrix_contribution_l2_rms": (
                        math.sqrt(float(contribution_square[site][feature]) / contributions)
                        if contributions
                        else 0.0
                    ),
                    "conditional_matrix_residual_l2_mean": (
                        float(conditional_error_sum[site][feature]) / contributions
                        if contributions
                        else 0.0
                    ),
                }
            )
    return {
        "model": local_key,
        "accepted_anchors": accepted,
        "documents": len(documents),
        "metrics": finalized,
        "dictionary": dictionary,
        "feature_rows": feature_rows,
        "activation_target_normalization": activation_target_normalization,
        "activation_target_epsilon": activation_target_epsilon,
        "activation_target_norms": finalized_norms,
    }


def _render_outputs(
    reports: list[dict[str, Any]], output: Path, manifest: dict[str, Any], elapsed: float
) -> None:
    all_metrics = {key: value for report in reports for key, value in report["metrics"].items()}
    activation_summary: dict[str, Any] = {}
    matrix_rows = []
    for report in reports:
        model = report["model"]
        activation_summary[model] = {}
        for site in ("pre", "mid"):
            prefix_metrics = []
            for prefix in range(6):
                selected = [
                    value
                    for key, value in report["metrics"].items()
                    if key.startswith(f"activation/{model}/{site}/")
                    and key.endswith(f"/prefix_{prefix}")
                ]
                prefix_metrics.append(_aggregate_ratio(selected))
            activation_summary[model][site] = {
                "prefixes": prefix_metrics,
                "matryoshka_mean_fvu": sum(value["fvu"] for value in prefix_metrics) / 6,
                "full_reconstruction": prefix_metrics[-1],
            }
        for key, value in report["metrics"].items():
            if not key.startswith("internal/"):
                continue
            target = key.removeprefix("internal/")
            _, layer, matrix = target.split(".")
            matrix_rows.append({"model": model, "layer": layer, "matrix": matrix, **value})
    _json(
        output / "heldout_reconstruction.json",
        {
            "schema_version": 1,
            "status": "HELDOUT_RECONSTRUCTION_COMPLETE",
            "activation_summary": activation_summary,
            "metrics": all_metrics,
        },
    )
    with (output / "per_matrix_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)
    dictionary = {report["model"]: report["dictionary"] for report in reports}
    _json(output / "dictionary_health.json", dictionary)
    feature_rows = [row for report in reports for row in report["feature_rows"]]
    with (output / "feature_activity.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)

    qwen = (
        sum(
            activation_summary["qwen"][site]["full_reconstruction"]["fvu"]
            for site in ("pre", "mid")
        )
        / 2
    )
    gemma = (
        sum(
            activation_summary["gemma"][site]["full_reconstruction"]["fvu"]
            for site in ("pre", "mid")
        )
        / 2
    )
    smol = (
        sum(
            activation_summary["smol"][site]["full_reconstruction"]["fvu"]
            for site in ("pre", "mid")
        )
        / 2
    )
    concentration = max(
        dictionary[model][site]["top_10_percent_event_share"]
        for model in MODEL_ORDER_2B
        for site in ("pre", "mid")
    )
    h1 = "SUPPORTED" if gemma <= 1.25 * max(qwen, smol) else "NOT_SUPPORTED"
    h2 = "PARTIALLY_SUPPORTED" if gemma > 1.5 * max(qwen, smol) else "NOT_SUPPORTED"
    h4 = "PARTIALLY_SUPPORTED" if concentration > 0.5 else "NOT_SUPPORTED"
    diagnostic = f"""# Gemma-specific held-out diagnostic

- H1 metric/normalization mismatch: **{h1}**. Logged training activation loss is a
  six-prefix Matryoshka mean; held-out full-reconstruction FVU is Qwen `{qwen:.4f}`,
  Gemma `{gemma:.4f}`, SmolLM2 `{smol:.4f}`.
- H2 genuine reconstruction difficulty: **{h2}**. This label is observational and does
  not identify whether the limiting factor is architecture, C, K, or the shared encoder.
- H3 shared-gate interference: **INCONCLUSIVE**. One trained checkpoint cannot establish
  negative transfer; a controlled gate/training ablation would be required.
- H4 dictionary concentration: **{h4}**. Maximum held-out top-10% firing share is
  `{concentration:.4f}`; rare features remain reported separately from never-fired ones.
- H5 optimization/data imbalance: **INCONCLUSIVE**. The logs contain objectives but no
  per-model gradient norm history. GPU utilization is not a gradient diagnostic.

No shared/exclusive mechanism or causal claim follows from this evaluation.
"""
    (output / "gemma_diagnostic.md").write_text(diagnostic, encoding="utf-8")
    summary = f"""# ASPD v3 500M held-out diagnostic summary

Status: **HELDOUT_RECONSTRUCTION_COMPLETE — SCIENTIFIC REVIEW REQUIRED**

The final checkpoint restored read-only before this run. Evaluation used the frozen
validation document buckets and `{reports[0]["accepted_anchors"]}` aligned anchors.

## Required decisions

1. **Checkpoint validity:** engineering restore/inference passed; scientific validity is
   conditional on the held-out reconstruction and dictionary-health tables.
2. **Gemma discrepancy:** held-out full FVU is `{gemma:.4f}` versus Qwen `{qwen:.4f}` and
   SmolLM2 `{smol:.4f}`. Training Matryoshka mean and full FVU are reported separately.
3. **Root cause:** see `gemma_diagnostic.md`; shared-gate interference remains causal-unverified.
4. **Dictionary coverage:** see `dictionary_health.json`; firing concentration does not by
   itself prove uselessness or justify taxonomy.
5. **Retraining:** not automatically justified by this diagnostic. Any proposal requires a
   controlled ablation and separate approval.

This run does not implement taxonomy, motif discovery, or causal intervention.
"""
    (output / "evaluation_summary.md").write_text(summary, encoding="utf-8")
    _json(
        output / "benchmark_report.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "kind": "ASPD_V3_500M_DIAGNOSTIC_BENCHMARK",
            "elapsed_seconds": elapsed,
            "accepted_anchors": reports[0]["accepted_anchors"],
            "documents_by_model": {report["model"]: report["documents"] for report in reports},
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "resources": {"world_size": 3, "gpu": torch.cuda.get_device_name()},
        },
    )
    _json(output / "experiment_manifest.json", manifest)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    axis.bar(range(len(matrix_rows)), [row["fvu"] for row in matrix_rows])
    axis.set(
        title="Held-out internal reconstruction FVU", ylabel="FVU", xlabel="168 logical targets"
    )
    axis.set_xticks([])
    figure.savefig(output / "per_matrix_fvu.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for site_index, site in enumerate(("pre", "mid")):
        for report in reports:
            counts = sorted(
                (row["heldout_fires"] for row in report["feature_rows"] if row["site"] == site),
                reverse=True,
            )
            axes[site_index].plot(range(1, len(counts) + 1), counts, label=report["model"])
        axes[site_index].set(
            xscale="log",
            yscale="symlog",
            title=f"{site.upper()} firing frequency",
            xlabel="feature rank",
        )
        axes[site_index].legend()
    figure.savefig(output / "feature_firing_distribution.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for site_index, site in enumerate(("pre", "mid")):
        for report in reports:
            contributions = sorted(
                (
                    row["matrix_contribution_l2_mean"] * row["matrix_component_events"]
                    for row in report["feature_rows"]
                    if row["site"] == site
                ),
                reverse=True,
            )
            total = sum(contributions)
            cumulative = []
            running = 0.0
            for value in contributions:
                running += value
                cumulative.append(running / total if total else 0.0)
            axes[site_index].plot(range(1, len(cumulative) + 1), cumulative, label=report["model"])
        axes[site_index].set(
            xscale="log",
            title=f"{site.upper()} contribution concentration",
            xlabel="feature rank",
            ylabel="cumulative contribution share",
        )
        axes[site_index].legend()
    figure.savefig(output / "contribution_concentration.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = _args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 3:
        raise RuntimeError("ASPD v3 diagnostic requires exactly three model-owner ranks")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    started = time.perf_counter()
    contract = DiagnosticContract.load(args.contract)
    provenance = validate_frozen_provenance(contract, source_root=args.source_root)
    controlled_values = (
        args.derived_checkpoint,
        args.derived_checkpoint_manifest_sha256,
        args.controlled_arm,
        args.controlled_run_manifest,
    )
    if any(value is not None for value in controlled_values) and not all(
        value is not None for value in controlled_values
    ):
        raise ValueError("controlled evaluation requires all four derived-checkpoint arguments")
    controlled_provenance = None
    checkpoint = args.source_root / contract.checkpoint
    expected_checkpoint_sha = contract.checkpoint_manifest_sha256
    expected_checkpoint_step = 61036
    if args.derived_checkpoint is not None:
        checkpoint = args.derived_checkpoint
        expected_checkpoint_sha = str(args.derived_checkpoint_manifest_sha256)
        expected_checkpoint_step = 1221
        controlled_provenance = _validate_controlled_checkpoint(
            checkpoint=checkpoint,
            expected_manifest_sha256=expected_checkpoint_sha,
            run_manifest_path=args.controlled_run_manifest,
            arm=str(args.controlled_arm),
            contract=contract,
        )
    config_path = args.source_root / contract.source_config
    config = load_main_2b_v1_config(config_path)
    registry = build_model_registry_2b(config)
    local_key = MODEL_ORDER_2B[rank]

    if args.mode == "heldout":
        if args.checkpoint_smoke_report is None or not args.checkpoint_smoke_report.is_file():
            raise FileNotFoundError("held-out evaluation requires the completed E0 smoke report")
        smoke = json.loads(args.checkpoint_smoke_report.read_text(encoding="utf-8"))
        if smoke.get("status") != "E0_CHECKPOINT_RESTORE_PASS":
            raise ValueError("held-out evaluation blocked by checkpoint smoke status")
        if smoke.get("checkpoint_manifest_sha256") != expected_checkpoint_sha:
            raise ValueError("checkpoint smoke evaluated a different checkpoint")
        if smoke.get("controlled_arm") != args.controlled_arm:
            raise ValueError("checkpoint smoke evaluated a different controlled arm")

    if rank == 0:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite diagnostic output: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
    dist.barrier()

    tokenizer = load_v3_tokenizer(registry[local_key], local_files_only=args.local_files_only)
    loaded = load_frozen_v3_model(
        registry[local_key], device, local_files_only=args.local_files_only
    )
    assert_frozen_backbone(loaded.forward_model)
    source_before = _parameter_signature(loaded.forward_model)
    native_biases = {
        target.key: logical_weight_bias(loaded.forward_model, target)[1]
        for target in registry[local_key].targets()
    }
    model = ModelOwnedMain2B(
        config,
        registry,
        local_key,
        native_biases=native_biases,
        device=device,
        dtype=torch.float32,
        sparse_backend="dense",
    )
    manager = ModelOwnedCheckpoint(checkpoint.parent, keep_complete=2)
    runtime_state, checkpoint_manifest = manager.load_model_only(
        path=checkpoint,
        model=model,
        config_fingerprint=config.fingerprint(),
        config_file_sha256=sha256_file(config_path),
        dataset_manifest_sha256=contract.dataset_manifest_sha256,
        expected_model_key=local_key,
    )
    model.eval().requires_grad_(False)
    finite = _finite_state_report(model)
    digest = _shared_digest(model)
    digests: list[str | None] = [None] * world
    dist.all_gather_object(digests, digest)
    if args.controlled_arm == "gemma_specific":
        if digests[0] != digests[2] or digests[1] in {digests[0], digests[2]}:
            raise ValueError("Gemma-specific checkpoint shared-encoder topology is invalid")
        shared_topology = "QWEN_SMOL_SHARED_GEMMA_SPECIFIC_PASS"
    else:
        if len(set(digests)) != 1:
            raise ValueError("replicated shared encoders differ across checkpoint ranks")
        shared_topology = "ALL_THREE_SHARED_PASS"

    alignment_tokenizers = None
    stream = None
    if rank == 0:
        alignment_tokenizers = {
            key: tokenizer
            if key == local_key
            else load_v3_tokenizer(registry[key], local_files_only=args.local_files_only)
            for key in MODEL_ORDER_2B
        }
        dataset = load_pinned_pile_dataset(config.data.dataset_id, config.data.dataset_revision)
        stream = ValidationAlignedAnchorStream(
            dataset=dataset,
            dataset_id=config.data.dataset_id,
            dataset_revision=config.data.dataset_revision,
            tokenizers=alignment_tokenizers,
            tokenizer_revisions={key: registry[key].tokenizer_revision for key in MODEL_ORDER_2B},
            split_policy=HashSplitPolicy(
                modulus=config.data.split_modulus,
                validation_buckets=frozenset(config.data.validation_buckets),
                test_buckets=frozenset(config.data.test_buckets),
                seed=config.data.split_seed,
            ),
            dedup_path=args.output_dir / "validation_stream_dedup.sqlite",
            sequence_length=config.data.sequence_length,
        )

    def next_pool() -> AlignedAnchorPool:
        payload = (
            None if rank else stream.next_pool(pool_anchors=contract.pool_anchors).to_payload()
        )
        return AlignedAnchorPool.from_payload(_broadcast(payload))

    if args.mode == "checkpoint-smoke":
        pool = next_pool()
        capture = _capture(
            pool,
            local_key=local_key,
            tokenizer=tokenizer,
            loaded=loaded,
            registry=registry,
            device=device,
        )
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
        ):
            first = model.encode(capture, update_statistics=False)
            second = model.encode(capture, update_statistics=False)
        for site in ("pre", "mid"):
            torch.testing.assert_close(
                first[1][site].values, second[1][site].values, rtol=0, atol=0
            )
            if not torch.isfinite(first[1][site].values).all():
                raise FloatingPointError("checkpoint smoke produced non-finite gate values")
        local_report = {
            "rank": rank,
            "model": local_key,
            "restored_step": checkpoint_manifest["step"],
            "finite_state": finite,
            "shared_encoder_sha256": digest,
            "pre_nonzero": int((first[1]["pre"].values > 0).sum()),
            "mid_nonzero": int((first[1]["mid"].values > 0).sum()),
        }
        reports: list[dict[str, Any] | None] = [None] * world
        dist.all_gather_object(reports, local_report)
        source_after = _parameter_signature(loaded.forward_model)
        if source_after != source_before:
            raise RuntimeError("frozen source model changed during checkpoint smoke")
        if rank == 0:
            _json(
                args.output_dir / "checkpoint_smoke.json",
                {
                    "schema_version": 1,
                    "status": "E0_CHECKPOINT_RESTORE_PASS",
                    "checkpoint_manifest_sha256": expected_checkpoint_sha,
                    "checkpoint_step": expected_checkpoint_step,
                    "controlled_arm": args.controlled_arm,
                    "controlled_provenance": controlled_provenance,
                    "shared_encoder_topology": shared_topology,
                    "model_order": list(MODEL_ORDER_2B),
                    "deterministic_repeat": "PASS",
                    "source_weights_unchanged": "PASS",
                    "optimizer_state_loaded": False,
                    "reports": reports,
                    "elapsed_seconds": time.perf_counter() - started,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                },
            )
            print("E0_CHECKPOINT_RESTORE_PASS")
    else:
        report = _local_evaluation(
            pools=contract.main_anchors // contract.pool_anchors,
            model=model,
            loaded=loaded,
            tokenizer=tokenizer,
            registry=registry,
            local_key=local_key,
            device=device,
            next_pool=next_pool,
            contract=contract,
        )
        source_after = _parameter_signature(loaded.forward_model)
        if source_after != source_before:
            raise RuntimeError("frozen source model changed during held-out evaluation")
        _json(args.output_dir / f"rank-{rank:05d}.json", report)
        dist.barrier()
        if rank == 0:
            reports = [
                json.loads((args.output_dir / f"rank-{value:05d}.json").read_text(encoding="utf-8"))
                for value in range(world)
            ]
            manifest = {
                "schema_version": 1,
                "status": "READ_ONLY_DIAGNOSTIC_COMPLETE",
                "diagnostic_git_sha": _git_sha(),
                "source_git_sha": contract.source_git_sha,
                "contract_sha256": contract.fingerprint(),
                "checkpoint_manifest_sha256": contract.checkpoint_manifest_sha256,
                "evaluated_checkpoint_manifest_sha256": expected_checkpoint_sha,
                "evaluated_checkpoint_step": expected_checkpoint_step,
                "controlled_arm": args.controlled_arm,
                "controlled_provenance": controlled_provenance,
                "shared_encoder_topology": shared_topology,
                "dataset_manifest_sha256": contract.dataset_manifest_sha256,
                "alignment_manifest_sha256": contract.alignment_manifest_sha256,
                "split": "validation",
                "seed": contract.seed,
                "model_order": list(MODEL_ORDER_2B),
                "accepted_anchors": contract.main_anchors,
                "software": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                "host": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "taxonomy": "NOT_IMPLEMENTED",
                "causal_intervention": "NOT_IMPLEMENTED",
                "metric_definitions": {
                    "fvu": "global SSE divided by coordinate-centered target sum of squares",
                    "matryoshka_mean_fvu": "mean of bias-only plus five nested-prefix global FVUs",
                    "uncertainty": "document-cluster bootstrap of document-level FVU means",
                    "internal_code": "detached binary BatchTopK support",
                    "activation_code": "continuous BatchTopK values",
                },
                "frozen_provenance": provenance,
                "runtime_state_step": int(runtime_state["completed_steps"]),
            }
            _render_outputs(reports, args.output_dir, manifest, time.perf_counter() - started)
            print("ASPD_V3_HELDOUT_DIAGNOSTIC_COMPLETE")
    if stream is not None:
        stream.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
