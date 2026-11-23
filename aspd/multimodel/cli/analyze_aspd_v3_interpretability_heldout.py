"""Synthesize held-out reconstruction and per-feature diagnostics for ASPD v3.1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from aspd.multimodel.v3.diagnostic import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument("--cluster-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def aggregate_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["site"]), int(row["feature"]))].append(row)
    output = []
    for (site, feature), values in sorted(grouped.items()):
        if len(values) != 3 or {row["model"] for row in values} != {"gemma", "smol", "granite"}:
            raise ValueError(f"feature {site}/{feature} lacks three model-owner rows")
        fires = {int(row["heldout_fires"]) for row in values}
        rates = {float(row["heldout_firing_rate"]) for row in values}
        if len(fires) != 1 or len(rates) != 1:
            raise ValueError("shared-gate firing statistics differ across model owners")
        model_contributions = {
            str(row["model"]): _float(row, "matrix_contribution_l2_mean")
            * int(row["matrix_component_events"])
            for row in values
        }
        model_residuals = {
            str(row["model"]): _float(row, "conditional_matrix_residual_l2_mean")
            for row in values
        }
        total_contribution = sum(model_contributions.values())
        shares = [value / total_contribution for value in model_contributions.values()] if total_contribution else [0, 0, 0]
        positive = [value for value in shares if value > 0]
        entropy = -sum(value * math.log(value) for value in positive)
        first = values[0]
        fires_value = fires.pop()
        activation_rms = _float(first, "activation_rms_when_fired")
        output.append(
            {
                "component_key": f"{site}/{feature:04d}",
                "site": site,
                "feature": feature,
                "heldout_fires": fires_value,
                "heldout_firing_rate": rates.pop(),
                "activation_rms_when_fired": activation_rms,
                "activation_code_energy_proxy": fires_value * activation_rms**2,
                "internal_contribution_l2_total": total_contribution,
                "conditional_matrix_residual_l2_mean": float(np.mean(list(model_residuals.values()))),
                "cross_model_contribution_effective_count": math.exp(entropy) if positive else 0.0,
                "cross_model_contribution_max_share": max(shares),
                **{f"{model}_internal_contribution_l2_total": value for model, value in model_contributions.items()},
                **{f"{model}_conditional_matrix_residual_l2_mean": value for model, value in model_residuals.items()},
            }
        )
    return output


def _spearman(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    statistic = spearmanr([float(row[left]) for row in rows], [float(row[right]) for row in rows])
    return {"rho": float(statistic.statistic), "pvalue_exploratory": float(statistic.pvalue), "n": len(rows)}


def _extreme_groups(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignments = []
    summaries = {}
    for site in ("pre", "mid"):
        site_rows = sorted((row for row in rows if row["site"] == site), key=lambda row: (-row["heldout_fires"], row["feature"]))
        eligible = [row for row in site_rows if row["heldout_fires"] >= 5]
        groups = {
            "top20": site_rows[:20],
            "middle20": sorted(eligible, key=lambda row: (abs(row["heldout_fires"] - np.median([value["heldout_fires"] for value in eligible])), row["feature"]))[:20],
            "low20": sorted(eligible, key=lambda row: (row["heldout_fires"], row["feature"]))[:20],
        }
        total_contribution = sum(float(row["internal_contribution_l2_total"]) for row in site_rows)
        for group, values in groups.items():
            assignments.extend({"group": group, **row} for row in values)
            summaries[f"{site}.{group}"] = {
                "features": len(values),
                "mean_fires": float(np.mean([row["heldout_fires"] for row in values])),
                "mean_activation_energy_proxy": float(np.mean([row["activation_code_energy_proxy"] for row in values])),
                "internal_contribution_share": (
                    sum(float(row["internal_contribution_l2_total"]) for row in values) / total_contribution
                    if total_contribution else 0.0
                ),
                "mean_conditional_residual_l2": float(np.mean([row["conditional_matrix_residual_l2_mean"] for row in values])),
                "mean_cross_model_contribution_effective_count": float(np.mean([row["cross_model_contribution_effective_count"] for row in values])),
            }
    return assignments, summaries


def _matrix_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for model in ("gemma", "smol", "granite"):
        for site in ("pre", "mid"):
            selected = [
                float(row["fvu"])
                for row in rows
                if row["model"] == model
                and ((row["matrix"] in {"q", "k", "v"}) == (site == "pre"))
            ]
            output[f"{model}.{site}"] = {
                "targets": len(selected),
                "mean_fvu": float(np.mean(selected)),
                "median_fvu": float(np.median(selected)),
                "minimum_fvu": float(np.min(selected)),
                "maximum_fvu": float(np.max(selected)),
            }
    return output


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(
    output: Path,
    feature_rows: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {row["component_key"]: row for row in feature_rows}
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for site, marker in (("pre", "o"), ("mid", "^")):
        subset = [row for row in observations if row["component_key"].startswith(f"{site}/")]
        axis.scatter(
            [by_key[row["component_key"]]["heldout_fires"] for row in subset],
            [float(row["accuracy"]) for row in subset],
            alpha=0.55,
            s=18,
            marker=marker,
            label=site.upper(),
        )
    axis.set(xscale="log", xlabel="Held-out natural firing count", ylabel="Failed Llama judge accuracy", title="Exploratory only: frequency vs invalid judge score")
    axis.legend()
    figure.savefig(output / "feature_frequency_vs_failed_judge.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = ["tab:red" if str(row["credible_multi_cluster"]).lower() == "true" else "tab:blue" for row in clusters]
    axis.scatter([float(row["silhouette"]) for row in clusters], [float(row["failed_judge_accuracy"]) for row in clusters], c=colors, alpha=0.65, s=24)
    axis.set(xlabel="Mistral embedding silhouette", ylabel="Failed Llama judge accuracy", title="Exploratory cluster diagnostic (red = stable multi-cluster)")
    figure.savefig(output / "cluster_diagnostics.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, site in zip(axes, ("pre", "mid"), strict=True):
        values = sorted((row["internal_contribution_l2_total"] for row in feature_rows if row["site"] == site), reverse=True)
        cumulative = np.cumsum(values) / max(float(np.sum(values)), 1e-12)
        axis.plot(np.arange(1, len(values) + 1), cumulative)
        axis.set(xscale="log", xlabel="Feature rank", ylabel="Cumulative L2 contribution share", title=f"{site.upper()} observational contribution")
    figure.savefig(output / "contribution_concentration.png", dpi=170)
    plt.close(figure)


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads((args.heldout_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("checkpoint_manifest_sha256") != "735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969":
        raise ValueError("held-out artifact is not the frozen final checkpoint")
    raw_features = list(csv.DictReader((args.heldout_dir / "feature_activity.csv").open()))
    features = aggregate_feature_rows(raw_features)
    matrix_rows = list(csv.DictReader((args.heldout_dir / "per_matrix_metrics.csv").open()))
    observations = list(csv.DictReader((args.preparation_dir / "component_observations.csv").open()))
    clusters = list(csv.DictReader((args.cluster_dir / "feature_cluster_metrics.csv").open()))
    by_key = {row["component_key"]: row for row in features}
    sampled = [{**row, **by_key[row["component_key"]]} for row in observations]
    cluster_joined = [{**row, **by_key[row["component_key"]]} for row in clusters]
    extreme_rows, extreme_summary = _extreme_groups(features)
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "aggregated_feature_diagnostics.csv", features)
    _write_csv(args.output_dir / "extreme_feature_groups.csv", extreme_rows)
    activation_internal = _spearman(features, "activation_code_energy_proxy", "internal_contribution_l2_total")
    frequency_internal = _spearman(features, "heldout_fires", "internal_contribution_l2_total")
    residual_frequency = _spearman(features, "heldout_fires", "conditional_matrix_residual_l2_mean")
    summary = {
        "schema_version": 1,
        "status": "ASPD_V3_1_HELDOUT_INTERPRETABILITY_DIAGNOSTICS_COMPLETE",
        "matrix_fvu": _matrix_summary(matrix_rows),
        "feature_correlations": {
            "activation_energy_vs_internal_contribution": activation_internal,
            "frequency_vs_internal_contribution": frequency_internal,
            "frequency_vs_conditional_matrix_residual": residual_frequency,
            "failed_judge_accuracy_vs_internal_contribution": _spearman(sampled, "accuracy", "internal_contribution_l2_total"),
            "failed_judge_accuracy_vs_activation_energy": _spearman(sampled, "accuracy", "activation_code_energy_proxy"),
            "cluster_silhouette_vs_internal_contribution": _spearman(cluster_joined, "silhouette", "internal_contribution_l2_total"),
            "cluster_selected_k_vs_frequency": _spearman(cluster_joined, "selected_k", "heldout_fires"),
        },
        "cross_model_observational_contribution": {
            "pairwise_spearman": {
                f"{left}_vs_{right}": _spearman(features, f"{left}_internal_contribution_l2_total", f"{right}_internal_contribution_l2_total")
                for left, right in (("gemma", "smol"), ("gemma", "granite"), ("smol", "granite"))
            },
            "warning": "Unsigned contribution norms are not causal-effect direction or functional correspondence.",
        },
        "extreme_feature_groups": extreme_summary,
        "measurement_boundaries": {
            "activation_code_energy_proxy": "heldout fires times RMS continuous gate value squared",
            "internal_contribution_l2_total": "sum over models of event count times mean rank-one output L2; not marginal residual reduction",
            "failed_judge": "position-biased and invalid for confirmatory semantic inference",
            "capacity": "observational diagnostics cannot identify the causal effect of C without a controlled capacity ablation",
        },
    }
    _write_json(args.output_dir / "heldout_interpretability_summary.json", summary)
    _plots(args.output_dir, features, observations, clusters)
    _write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "status": "ASPD_V3_1_HELDOUT_INTERPRETABILITY_ANALYSIS_COMPLETE",
            "git_sha": _git_sha(),
            "heldout_manifest_sha256": sha256_file(args.heldout_dir / "experiment_manifest.json"),
            "feature_activity_sha256": sha256_file(args.heldout_dir / "feature_activity.csv"),
            "cluster_manifest_sha256": sha256_file(args.cluster_dir / "experiment_manifest.json"),
        },
    )
    (args.output_dir / "COMPLETE").write_text("ASPD_V3_1_HELDOUT_INTERPRETABILITY_ANALYSIS_COMPLETE\n")


if __name__ == "__main__":
    main()
