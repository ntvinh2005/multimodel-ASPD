"""Checkpoint-only audit of ASPD v3 readout bias and weight drift."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from scipy.stats import spearmanr

from aspd.multimodel.v3.dictionary_concentration import FEATURE_BANDS, feature_band

SITES = ("pre", "mid")
MODELS = ("gemma", "smol", "granite")
READOUT_WIDTH = 512
N_FEATURES = 8192


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"inconsistent columns in {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_state(checkpoint: Path, rank: int) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for site in SITES:
        readout = f"model.shared_encoders.{site}.readout.output"
        optimizer = f"optimizer.state.shared_encoders.{site}.readout.output"
        state[f"{readout}.weight"] = torch.empty(N_FEATURES, READOUT_WIDTH)
        state[f"{readout}.bias"] = torch.empty(N_FEATURES)
        state[f"{optimizer}.weight.exp_avg"] = torch.empty(N_FEATURES, READOUT_WIDTH)
        state[f"{optimizer}.bias.exp_avg"] = torch.empty(N_FEATURES)
    dcp.load(state, checkpoint_id=checkpoint / f"rank-{rank:05d}" / "dcp")
    return state


def _verified_checkpoint_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    states = [_checkpoint_state(checkpoint, rank) for rank in range(3)]
    for key, reference in states[0].items():
        if any(not torch.equal(reference, state[key]) for state in states[1:]):
            raise ValueError(f"shared readout state differs across ranks: {key}")
    return states[0]


def _canonical_metrics(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["step"])] = row
    return rows


def _heldout_counts(path: Path) -> dict[str, torch.Tensor]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, torch.Tensor] = {}
    for site in SITES:
        by_model: list[torch.Tensor] = []
        for model in MODELS:
            selected = sorted(
                (row for row in rows if row["site"] == site and row["model"] == model),
                key=lambda row: int(row["feature"]),
            )
            by_model.append(torch.tensor([int(row["heldout_fires"]) for row in selected]))
        if any(not torch.equal(by_model[0], value) for value in by_model[1:]):
            raise ValueError(f"shared-gate held-out counts disagree across models for {site}")
        result[site] = by_model[0]
    return result


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    quantiles = torch.quantile(values.float(), torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99]))
    return {
        "q01": float(quantiles[0]),
        "q10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q90": float(quantiles[3]),
        "q99": float(quantiles[4]),
    }


def _group_masks(counts: torch.Tensor) -> dict[str, torch.Tensor]:
    top = torch.argsort(counts, descending=True)[:82]
    top_mask = torch.zeros(N_FEATURES, dtype=torch.bool)
    top_mask[top] = True
    masks = {
        "all": torch.ones(N_FEATURES, dtype=torch.bool),
        "top_82_heldout": top_mask,
        "heldout_active": counts > 0,
        "heldout_inactive": counts == 0,
    }
    for band in FEATURE_BANDS:
        mask = torch.zeros(N_FEATURES, dtype=torch.bool)
        mask[band.lower : band.upper] = True
        masks[band.name] = mask
    return masks


def _audit_site(
    site: str,
    state_0: Mapping[str, torch.Tensor],
    state_1: Mapping[str, torch.Tensor],
    counts: torch.Tensor,
    metrics: Mapping[int, Mapping[str, Any]],
    step_0: int,
    step_1: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    readout = f"model.shared_encoders.{site}.readout.output"
    optimizer = f"optimizer.state.shared_encoders.{site}.readout.output"
    weight_0 = state_0[f"{readout}.weight"].float()
    weight_1 = state_1[f"{readout}.weight"].float()
    bias_0 = state_0[f"{readout}.bias"].float()
    bias_1 = state_1[f"{readout}.bias"].float()
    weight_moment = state_1[f"{optimizer}.weight.exp_avg"].float().norm(dim=1)
    bias_moment = state_1[f"{optimizer}.bias.exp_avg"].float().abs()

    delta_weight = (weight_1 - weight_0).norm(dim=1)
    delta_bias = bias_1 - bias_0
    norm_0 = weight_0.norm(dim=1)
    norm_1 = weight_1.norm(dim=1)
    cosine = torch.nn.functional.cosine_similarity(weight_0, weight_1, dim=1)
    relative_delta = delta_weight / norm_0.clamp_min(torch.finfo(torch.float32).tiny)

    feature_rows: list[dict[str, Any]] = []
    for feature in range(N_FEATURES):
        feature_rows.append(
            {
                "site": site,
                "feature": feature,
                "band": feature_band(feature).name,
                "heldout_fires": int(counts[feature]),
                "bias_step_1000": float(bias_0[feature]),
                "bias_step_1221": float(bias_1[feature]),
                "bias_delta": float(delta_bias[feature]),
                "weight_norm_step_1000": float(norm_0[feature]),
                "weight_norm_step_1221": float(norm_1[feature]),
                "weight_delta_norm": float(delta_weight[feature]),
                "weight_relative_delta": float(relative_delta[feature]),
                "weight_endpoint_cosine": float(cosine[feature]),
                "adam_weight_exp_avg_row_norm_step_1221": float(weight_moment[feature]),
                "adam_bias_exp_avg_abs_step_1221": float(bias_moment[feature]),
            }
        )

    group_rows: list[dict[str, Any]] = []
    for group, mask in _group_masks(counts).items():
        group_rows.append(
            {
                "site": site,
                "group": group,
                "feature_count": int(mask.sum()),
                "bias_mean_step_1000": float(bias_0[mask].mean()),
                "bias_mean_step_1221": float(bias_1[mask].mean()),
                "bias_delta_mean": float(delta_bias[mask].mean()),
                "bias_delta_abs_mean": float(delta_bias[mask].abs().mean()),
                "weight_delta_norm_mean": float(delta_weight[mask].mean()),
                "weight_relative_delta_mean": float(relative_delta[mask].mean()),
                "weight_endpoint_cosine_mean": float(cosine[mask].mean()),
                "adam_weight_exp_avg_row_norm_mean_step_1221": float(
                    weight_moment[mask].mean()
                ),
                "adam_bias_exp_avg_abs_mean_step_1221": float(bias_moment[mask].mean()),
                "exactly_unchanged_weight_rows": int((delta_weight[mask] == 0).sum()),
                "exactly_unchanged_bias_entries": int((delta_bias[mask] == 0).sum()),
            }
        )

    endpoint_rows = []
    for step, bias in ((step_0, bias_0), (step_1, bias_1)):
        prefix = f"diagnostic/gate/{site}/gemma/"
        loss = metrics[step]["loss"]
        raw_mean = float(loss[prefix + "raw_logit_mean"])
        bias_mean = float(bias.mean())
        endpoint_rows.append(
            {
                "step": step,
                "raw_logit_mean": raw_mean,
                "raw_logit_negative_fraction": float(
                    loss[prefix + "raw_logit_negative_fraction"]
                ),
                "bias_mean": bias_mean,
                "bias_negative_fraction": float((bias < 0).float().mean()),
                "derived_inner_product_mean": raw_mean - bias_mean,
                "bias_quantiles": _quantiles(bias),
            }
        )

    all_mask = torch.ones(N_FEATURES, dtype=torch.bool)
    frobenius_relative_delta = float(
        (weight_1 - weight_0).norm() / weight_0.norm().clamp_min(torch.finfo(torch.float32).tiny)
    )
    log_fires = torch.log1p(counts.float()).numpy()
    summary = {
        "site": site,
        "endpoints": endpoint_rows,
        "bias_positive_to_negative": int(((bias_0 >= 0) & (bias_1 < 0)).sum()),
        "bias_negative_to_positive": int(((bias_0 < 0) & (bias_1 >= 0)).sum()),
        "bias_more_negative_fraction": float((delta_bias < 0).float().mean()),
        "readout_weight_relative_frobenius_delta": frobenius_relative_delta,
        "readout_row_cosine": {
            "mean": float(cosine.mean()),
            "minimum": float(cosine.min()),
            **_quantiles(cosine),
        },
        "readout_row_norm_step_1000": {
            "mean": float(norm_0.mean()),
            **_quantiles(norm_0),
        },
        "readout_row_norm_step_1221": {
            "mean": float(norm_1.mean()),
            **_quantiles(norm_1),
        },
        "exactly_unchanged_weight_rows": int((delta_weight[all_mask] == 0).sum()),
        "exactly_unchanged_bias_entries": int((delta_bias[all_mask] == 0).sum()),
        "spearman_log1p_heldout_fires_vs_weight_delta_norm": float(
            spearmanr(log_fires, delta_weight.numpy()).statistic
        ),
        "spearman_log1p_heldout_fires_vs_abs_bias_delta": float(
            spearmanr(log_fires, delta_bias.abs().numpy()).statistic
        ),
    }
    return feature_rows, group_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--feature-activity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step-0", type=int, default=1000)
    parser.add_argument("--step-1", type=int, default=1221)
    args = parser.parse_args()

    checkpoint_0 = args.run_dir / "checkpoints" / f"step-{args.step_0:09d}"
    checkpoint_1 = args.run_dir / "checkpoints" / f"step-{args.step_1:09d}"
    state_0 = _verified_checkpoint_state(checkpoint_0)
    state_1 = _verified_checkpoint_state(checkpoint_1)
    metrics = _canonical_metrics(args.run_dir / "metrics.jsonl")
    counts = _heldout_counts(args.feature_activity)

    feature_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    summaries = []
    for site in SITES:
        site_features, site_groups, site_summary = _audit_site(
            site,
            state_0,
            state_1,
            counts[site],
            metrics,
            args.step_0,
            args.step_1,
        )
        feature_rows.extend(site_features)
        group_rows.extend(site_groups)
        summaries.append(site_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "feature_readout_drift.csv", feature_rows)
    _write_csv(args.output_dir / "readout_group_summary.csv", group_rows)
    payload = {
        "schema_version": 1,
        "status": "CHECKPOINT_ONLY_LOGIT_DRIFT_AUDIT_COMPLETE",
        "step_0": args.step_0,
        "step_1": args.step_1,
        "checkpoint_ranks_verified_equal": [0, 1, 2],
        "sites": summaries,
        "evidence_boundary": (
            "No fixed-anchor representations are stored. Bias and readout parameter drift are "
            "measured exactly, but W-versus-h responsibility requires GPU replay. Logged raw-logit "
            "means use different training anchors at the two steps."
        ),
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
