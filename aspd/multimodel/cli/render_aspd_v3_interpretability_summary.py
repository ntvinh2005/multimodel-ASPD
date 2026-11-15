"""Render provenance-rich figures for the ASPD v3.1 interpretability audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.v3.diagnostic import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-summary", type=Path, required=True)
    parser.add_argument("--rotation-summary", type=Path, required=True)
    parser.add_argument("--mistral-summary", type=Path, required=True)
    parser.add_argument("--threshold-coverage", type=Path, required=True)
    parser.add_argument("--heldout-summary", type=Path, required=True)
    parser.add_argument("--step13-metrics", type=Path, required=True)
    parser.add_argument("--final-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(path: Path, step: int) -> dict[str, Any]:
    selected = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("step", -1)) == step:
                selected = row
    if selected is None:
        raise ValueError(f"step {step} absent from {path}")
    return selected


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _condition(coverage: dict[str, Any], site: str, label: str) -> dict[str, Any]:
    matches = [row for row in coverage[site]["conditions"] if row["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"expected one {site}/{label} condition")
    return matches[0]


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paper = _json(args.paper_summary)
    rotation = _json(args.rotation_summary)
    mistral = _json(args.mistral_summary)
    coverage = _json(args.threshold_coverage)
    heldout = _json(args.heldout_summary)
    step13 = _metric_row(args.step13_metrics, 13_000)
    final = _metric_row(args.final_metrics, 61_036)

    positions = np.arange(1, 6)
    predicted = np.asarray(
        [paper["predicted_position_counts"][str(value)] for value in positions],
        dtype=np.float64,
    )
    predicted /= predicted.sum()
    positional_accuracy = np.asarray(
        [paper["accuracy_by_true_position"][str(value)] for value in positions]
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(positions, predicted, color="tab:orange")
    axes[0].axhline(0.2, color="black", linestyle="--", linewidth=1, label="uniform")
    axes[0].set(
        xlabel="Predicted intruder position",
        ylabel="Prediction share",
        title="Llama judge position distribution",
        xticks=positions,
        ylim=(0, max(0.5, float(predicted.max()) * 1.1)),
    )
    axes[0].legend()
    axes[1].bar(positions, positional_accuracy, color="tab:red")
    axes[1].set(
        xlabel="True intruder position",
        ylabel="Accuracy",
        title="Accuracy depends on true position",
        xticks=positions,
        ylim=(0, 0.65),
    )
    figure.savefig(args.output_dir / "judge_position_distribution.png", dpi=180)
    plt.close(figure)

    labels = [
        "natural_gs_gt_0",
        "fixed_gs_ge_0.06",
        "fixed_gs_ge_0.08",
        "fixed_gs_ge_0.12",
        "fixed_gs_ge_0.2",
        "calibration_within_feature_q0.5",
        "calibration_within_feature_q0.75",
        "calibration_within_feature_q0.9",
    ]
    short = ["natural", ">=0.06", ">=0.08", ">=0.12", ">=0.20", "q50", "q75", "q90"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for site, color in (("pre", "tab:blue"), ("mid", "tab:green")):
        rows = [_condition(coverage, site, label) for label in labels]
        axes[0].plot(
            short,
            [row["retained_event_fraction"] for row in rows],
            marker="o",
            color=color,
            label=site.upper(),
        )
        axes[1].plot(
            short,
            [row["eligible_features"] for row in rows],
            marker="o",
            color=color,
            label=site.upper(),
        )
    axes[0].set(ylabel="Retained natural firing-event fraction", title="Activation threshold coverage")
    axes[1].set(ylabel="Eligible features (>=5 events)", title="Eligibility under stronger firings")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.legend()
    figure.savefig(args.output_dir / "activation_threshold_coverage.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = np.arange(2)
    width = 0.34
    activation = [step13["loss"]["activation"], final["loss"]["activation"]]
    internal = [step13["loss"]["internal"], final["loss"]["internal"]]
    axis.bar(x - width / 2, activation, width, label="activation loss")
    axis.bar(x + width / 2, internal, width, label="internal loss")
    axis.set(
        xticks=x,
        xticklabels=["step 13,000", "step 61,036"],
        ylabel="Training-batch loss",
        title="Training objectives continued to improve",
    )
    axis.legend()
    figure.savefig(args.output_dir / "checkpoint_training_loss.png", dpi=180)
    plt.close(figure)

    pairwise = heldout["cross_model_observational_contribution"]["pairwise_spearman"]
    names = ["Gemma--Smol", "Gemma--Granite", "Smol--Granite"]
    keys = ["gemma_vs_smol", "gemma_vs_granite", "smol_vs_granite"]
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.bar(names, [pairwise[key]["rho"] for key in keys], color="tab:purple")
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        ylabel="Spearman rho",
        title="Unsigned held-out contribution alignment (not causal)",
        ylim=(-1, 1),
    )
    figure.savefig(
        args.output_dir / "cross_model_observational_contribution_alignment.png", dpi=180
    )
    plt.close(figure)

    manifest = {
        "schema_version": 1,
        "status": "ASPD_V3_1_INTERPRETABILITY_FIGURES_COMPLETE",
        "git_sha": _git_sha(),
        "sources": {
            str(path): sha256_file(path)
            for path in (
                args.paper_summary,
                args.rotation_summary,
                args.mistral_summary,
                args.threshold_coverage,
                args.heldout_summary,
                args.step13_metrics,
                args.final_metrics,
            )
        },
        "scientific_boundaries": {
            "judge": paper["scientific_status"],
            "independent_judge": mistral["status"],
            "rotation_all_five_correct_fraction": rotation["rotation_consistency"][
                "fraction_all_five_correct"
            ],
            "accuracy_vs_threshold_figure": "NOT_PRODUCED_INVALID_JUDGE_AND_NO_THRESHOLD_REJUDGING",
            "coherence_vs_coverage_figure": "NOT_PRODUCED_NO_VALID_SEMANTIC_INSTRUMENT",
            "cross_model_effect_alignment": "UNSIGNED_OBSERVATIONAL_CONTRIBUTIONS_ONLY_NOT_CAUSAL_EFFECTS",
        },
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
