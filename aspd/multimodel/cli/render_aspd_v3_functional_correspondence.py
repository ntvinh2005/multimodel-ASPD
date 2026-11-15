"""Render the signed H7 cross-model causal-effect alignment figure."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from aspd.multimodel.v3.diagnostic import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--component-effects", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_effects(path: Path) -> dict[tuple[str, int, str], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 72:
        raise ValueError(f"expected 72 model-component rows, received {len(rows)}")
    return {
        (row["site"], int(row["feature"]), row["model"]): float(row["active_effect"])
        for row in rows
    }


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summary: dict[str, Any] = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("status") != "FUNCTIONAL_CORRESPONDENCE_ANALYSIS_COMPLETE":
        raise ValueError("functional-correspondence analysis is incomplete")
    effects = _load_effects(args.component_effects)
    components = sorted({(site, feature) for site, feature, _model in effects})
    if len(components) != 24:
        raise ValueError(f"expected 24 components, received {len(components)}")

    args.output_dir.mkdir(parents=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparisons = (
        ("gemma", "smol", "Gemma", "SmolLM2"),
        ("gemma", "granite", "Gemma", "Granite"),
        ("smol", "granite", "SmolLM2", "Granite"),
    )
    pairwise = summary["learned_signed_alignment"]["active_effect"]["pairwise"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for axis, (left, right, left_label, right_label) in zip(
        axes, comparisons, strict=True
    ):
        left_values = np.asarray([effects[(*component, left)] for component in components])
        right_values = np.asarray([effects[(*component, right)] for component in components])
        colors = ["tab:blue" if site == "pre" else "tab:orange" for site, _ in components]
        limit = float(max(np.abs(left_values).max(), np.abs(right_values).max()) * 1.08)
        key = f"{left}_vs_{right}"
        statistic = pairwise[key]
        axis.scatter(left_values, right_values, c=colors, alpha=0.8, edgecolor="white")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.plot([-limit, limit], [-limit, limit], color="grey", linestyle="--", linewidth=0.8)
        axis.set(
            xlabel=f"{left_label} signed effect",
            ylabel=f"{right_label} signed effect",
            xlim=(-limit, limit),
            ylim=(-limit, limit),
            title=(
                f"rho={statistic['rho']:.3f} "
                f"[{statistic['ci95'][0]:.3f}, {statistic['ci95'][1]:.3f}]\n"
                f"sign agreement={statistic['sign_concordance']['mean']:.1%}"
            ),
        )
    figure.suptitle(
        "H7 learned rank-one ablation: signed continuation log-probability effect per UTF-8 byte"
    )
    figure.savefig(args.output_dir / "cross_model_signed_causal_effect_alignment.png", dpi=180)
    plt.close(figure)

    manifest = {
        "schema_version": 1,
        "status": "FUNCTIONAL_CORRESPONDENCE_FIGURE_COMPLETE",
        "git_sha": _git_sha(),
        "sources": {
            str(args.summary): sha256_file(args.summary),
            str(args.component_effects): sha256_file(args.component_effects),
        },
        "figure": "cross_model_signed_causal_effect_alignment.png",
        "data_kind": "ACTUAL_H7_CAUSAL_EVALUATION",
        "scientific_boundary": (
            "signed pairwise alignment is necessary evidence but does not prove identical "
            "cross-model algorithms"
        ),
        "h7_verdict": summary["h7_verdict"],
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
