"""Validate and summarize saved P1--P5 multi-model analysis artifacts."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.config import MultiModelExperimentConfig

TAXONOMY_CATEGORIES = (
    "shared_activation_shared_mechanism",
    "shared_activation_concentrated_mechanism",
    "concentrated_activation_shared_mechanism",
    "concentrated_activation_concentrated_mechanism",
    "mixed_or_subset_mass",
)
ROBUSTNESS_EPSILONS = (0.05, 0.10, 0.15)
ROBUSTNESS_THRESHOLDS = (0.80, 0.90, 0.95)


def _require_shape(tensors: Mapping[str, Tensor], key: str, shape: tuple[int, ...]) -> Tensor:
    if key not in tensors:
        raise ValueError(f"missing tensor {key!r}")
    tensor = tensors[key]
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{key} has shape {tuple(tensor.shape)}, expected {shape}")
    return tensor


def validate_posthoc_artifacts(
    tensors: Mapping[str, Tensor],
    taxonomy: Mapping[str, list[int]],
    cfg: MultiModelExperimentConfig,
) -> dict[str, Any]:
    """Fail before interpretation if a saved P1--P5 invariant does not hold."""

    model_names = [model.name for model in cfg.models]
    n_models = len(model_names)
    n_features = cfg.sparsity.n_features
    if n_models != 2:
        raise ValueError(f"P1--P5 report currently requires two models, got {n_models}")

    activation_rho = _require_shape(tensors, "activation_rho_NC", (n_models, n_features))
    beta = _require_shape(tensors, "mechanism_beta_NC", (n_models, n_features))
    mechanism_rho = _require_shape(tensors, "mechanism_rho_NC", (n_models, n_features))
    fire_count = _require_shape(tensors, "fire_count_C", (n_features,))

    # Computation flow: model-normalized P1/P2 columns must sum to one. A zero-beta feature is the
    # sole P2 exception: beta[:,c]=[0,0] -> mechanism_rho[:,c]=[0,0], not an invented [.5,.5].
    if not torch.allclose(
        activation_rho.sum(dim=0), torch.ones(n_features), atol=1e-5, rtol=0
    ):
        raise ValueError("P1 activation_rho columns do not sum to one")
    positive_beta = beta.sum(dim=0) > 0
    if positive_beta.any() and not torch.allclose(
        mechanism_rho[:, positive_beta].sum(dim=0),
        torch.ones(int(positive_beta.sum())),
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("P2 mechanism_rho columns with positive beta do not sum to one")
    if (~positive_beta).any() and not torch.allclose(
        mechanism_rho[:, ~positive_beta], torch.zeros_like(mechanism_rho[:, ~positive_beta])
    ):
        raise ValueError("zero-beta features must have zero mechanism_rho")

    # BatchTopK fires K components per valid token. Example 65,536 tokens * K=32 = 2,097,152.
    expected_fires = cfg.data.validation_tokens * cfg.sparsity.top_k
    observed_fires = int(round(float(fire_count.sum())))
    if observed_fires != expected_fires:
        raise ValueError(
            f"fire-count invariant failed: observed {observed_fires}, expected {expected_fires}"
        )

    matrix_names = [matrix.name for matrix in cfg.models[0].matrices]
    if [matrix.name for matrix in cfg.models[1].matrices] != matrix_names:
        raise ValueError("P5 report requires aligned matrix names across the two models")
    for model_index, model_name in enumerate(model_names):
        loci = torch.stack(
            [
                _require_shape(tensors, f"locus/{model_name}/{matrix}", (n_features,))
                for matrix in matrix_names
            ]
        )
        active = beta[model_index] > 0
        if active.any() and not torch.allclose(
            loci[:, active].sum(dim=0),
            torch.ones(int(active.sum())),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError(f"P3 locus fractions do not sum to one for model {model_name!r}")

    for matrix in matrix_names:
        for metric in (
            "component_cosine",
            "relative_component_change",
            "read_cosine",
            "write_cosine",
        ):
            _require_shape(tensors, f"pair/{matrix}/{metric}", (n_features,))

    # P4 must be a true partition: concatenating its five lists gives exactly range(C).
    unknown = set(taxonomy) - set(TAXONOMY_CATEGORIES)
    if unknown:
        raise ValueError(f"unknown taxonomy categories: {sorted(unknown)}")
    missing = set(TAXONOMY_CATEGORIES) - set(taxonomy)
    if missing:
        raise ValueError(f"missing taxonomy categories: {sorted(missing)}")
    assigned = [feature for category in TAXONOMY_CATEGORIES for feature in taxonomy[category]]
    if len(assigned) != n_features or sorted(assigned) != list(range(n_features)):
        raise ValueError("P4 taxonomy does not partition every feature exactly once")

    for key, tensor in tensors.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite values in {key}")

    return {
        "expected_total_fires": expected_fires,
        "observed_total_fires": observed_fires,
        "mean_fires_per_feature": observed_fires / n_features,
        "zero_fire_features": int((fire_count == 0).sum()),
    }


def _category_by_feature(taxonomy: Mapping[str, list[int]], n_features: int) -> list[str]:
    result = [""] * n_features
    for category in TAXONOMY_CATEGORIES:
        for feature_id in taxonomy[category]:
            result[feature_id] = category
    return result


def build_feature_rows(
    tensors: Mapping[str, Tensor],
    taxonomy: Mapping[str, list[int]],
    cfg: MultiModelExperimentConfig,
    *,
    low_support_threshold: int = 32,
) -> list[dict[str, Any]]:
    """Create one transparent, sortable row per latent component."""

    model_names = [model.name for model in cfg.models]
    matrix_names = [matrix.name for matrix in cfg.models[0].matrices]
    activation_rho = tensors["activation_rho_NC"]
    beta = tensors["mechanism_beta_NC"]
    mechanism_rho = tensors["mechanism_rho_NC"]
    fire_count = tensors["fire_count_C"]
    categories = _category_by_feature(taxonomy, cfg.sparsity.n_features)
    locus = {
        model: torch.stack([tensors[f"locus/{model}/{matrix}"] for matrix in matrix_names])
        for model in model_names
    }

    rows: list[dict[str, Any]] = []
    for feature_id in range(cfg.sparsity.n_features):
        count = int(round(float(fire_count[feature_id])))
        total_beta = float(beta[:, feature_id].sum())
        # Threshold robustness flow: one feature is retested on the 3x3 (epsilon,tau) grid. A hit
        # means activation remains shared while its mechanism remains concentrated; max score is 9.
        robustness_hits = sum(
            abs(float(activation_rho[0, feature_id]) - 0.5) < epsilon
            and float(mechanism_rho[:, feature_id].max()) >= threshold
            for epsilon in ROBUSTNESS_EPSILONS
            for threshold in ROBUSTNESS_THRESHOLDS
        )
        row: dict[str, Any] = {
            "feature_id": feature_id,
            "category": categories[feature_id],
            f"activation_rho_{model_names[0]}": float(activation_rho[0, feature_id]),
            f"activation_rho_{model_names[1]}": float(activation_rho[1, feature_id]),
            f"mechanism_rho_{model_names[0]}": float(mechanism_rho[0, feature_id]),
            f"mechanism_rho_{model_names[1]}": float(mechanism_rho[1, feature_id]),
            f"beta_{model_names[0]}": float(beta[0, feature_id]),
            f"beta_{model_names[1]}": float(beta[1, feature_id]),
            "beta_total": total_beta,
            "fire_count": count,
            "fire_density": count / cfg.data.validation_tokens,
            "low_support": 0 < count < low_support_threshold,
            "dominant_mechanism_model": (
                model_names[int(mechanism_rho[:, feature_id].argmax())] if total_beta > 0 else ""
            ),
            "activation_mechanism_delta": abs(
                float(activation_rho[0, feature_id] - mechanism_rho[0, feature_id])
            ),
            "robustness_hits_of_9": robustness_hits,
        }
        for model_index, model_name in enumerate(model_names):
            row[f"top_matrix_{model_name}"] = (
                matrix_names[int(locus[model_name][:, feature_id].argmax())]
                if float(beta[model_index, feature_id]) > 0
                else ""
            )
        rows.append(row)
    return rows


def select_review_groups(
    rows: list[dict[str, Any]],
    cfg: MultiModelExperimentConfig,
    *,
    top_per_direction: int = 10,
    control_count: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Select target and control groups by explicit filters, then beta mass."""

    base, finetuned = (model.name for model in cfg.models)
    target = [
        row
        for row in rows
        if row["category"] == "shared_activation_concentrated_mechanism"
        and row["fire_count"] > 0
    ]
    # No opaque score: direction is a hard rho threshold, ordering is absolute beta_total only.
    base_dominant = sorted(
        [row for row in target if row["dominant_mechanism_model"] == base],
        key=lambda row: row["beta_total"],
        reverse=True,
    )[:top_per_direction]
    ft_dominant = sorted(
        [
            row
            for row in target
            if row["dominant_mechanism_model"] == finetuned
        ],
        key=lambda row: row["beta_total"],
        reverse=True,
    )[:top_per_direction]

    def controls(category: str) -> list[dict[str, Any]]:
        return sorted(
            [row for row in rows if row["category"] == category and row["fire_count"] > 0],
            key=lambda row: row["beta_total"],
            reverse=True,
        )[:control_count]

    return {
        "base_dominant_targets": base_dominant,
        "finetuned_dominant_targets": ft_dominant,
        "shared_shared_controls": controls("shared_activation_shared_mechanism"),
        "concentrated_concentrated_controls": controls(
            "concentrated_activation_concentrated_mechanism"
        ),
    }


def _configure_matplotlib() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        cache = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "aspd-matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)


def _rho_guides(axis: Any, shared_epsilon: float, concentrated_threshold: float) -> None:
    axis.axvspan(0.5 - shared_epsilon, 0.5 + shared_epsilon, color="tab:blue", alpha=0.08)
    axis.axhspan(0, 1 - concentrated_threshold, color="tab:red", alpha=0.08)
    axis.axhspan(concentrated_threshold, 1, color="tab:red", alpha=0.08)
    axis.axvline(0.5, color="black", linestyle=":", linewidth=1)
    axis.axhline(0.5, color="black", linestyle=":", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Activation rho (base)")
    axis.set_ylabel("Mechanism rho (base)")
    axis.grid(True, alpha=0.2)


def _write_figures(
    tensors: Mapping[str, Tensor],
    taxonomy: Mapping[str, list[int]],
    cfg: MultiModelExperimentConfig,
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    activation = tensors["activation_rho_NC"][0].float().numpy()
    mechanism = tensors["mechanism_rho_NC"][0].float().numpy()
    beta_total = tensors["mechanism_beta_NC"].sum(dim=0).float().numpy()
    fire_count = tensors["fire_count_C"].float().numpy()

    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.scatter(activation, mechanism, s=8, alpha=0.35, edgecolors="none")
    _rho_guides(axis, cfg.analysis.shared_epsilon, cfg.analysis.concentrated_threshold)
    axis.set_title("Activation ownership vs mechanism ownership")
    fig.savefig(output_dir / "rho_scatter.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Absolute-mass view: color and size use log(1+beta_base+beta_ft), so an extreme rho backed by
    # beta=1e-8 stays visually tiny while a strong mechanism is prominent.
    log_mass = torch.from_numpy(beta_total).log1p().numpy()
    scale = max(float(log_mass.max()), 1e-12)
    sizes = 6 + 54 * log_mass / scale
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    points = axis.scatter(
        activation, mechanism, c=log_mass, s=sizes, alpha=0.55, edgecolors="none", cmap="viridis"
    )
    _rho_guides(axis, cfg.analysis.shared_epsilon, cfg.analysis.concentrated_threshold)
    axis.set_title("Ownership mismatch weighted by absolute mechanism mass")
    fig.colorbar(points, ax=axis, label="log(1 + beta total)")
    fig.savefig(output_dir / "rho_mass_scatter.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    labels = [
        "shared/shared",
        "shared/concentrated",
        "concentrated/shared",
        "concentrated/concentrated",
        "mixed/subset",
    ]
    counts = [len(taxonomy[category]) for category in TAXONOMY_CATEGORIES]
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.bar(labels, counts)
    axis.set_ylabel("Feature count")
    axis.set_title("P4 taxonomy")
    axis.tick_params(axis="x", rotation=25)
    fig.savefig(output_dir / "taxonomy_counts.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    for values, name, title in (
        (activation, "activation_rho_hist.png", "Base activation rho"),
        (mechanism, "mechanism_rho_hist.png", "Base mechanism rho"),
    ):
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        axis.hist(values, bins=50)
        axis.axvline(0.5, color="black", linestyle=":")
        axis.set_xlabel("rho")
        axis.set_ylabel("Feature count")
        axis.set_title(title)
        fig.savefig(output_dir / name, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.hist(fire_count, bins=50)
    axis.axvline(float(fire_count.mean()), color="black", linestyle=":", label="mean")
    axis.set_xlabel("Validation fires per feature")
    axis.set_ylabel("Feature count")
    axis.set_yscale("log")
    axis.set_title("Feature firing support")
    axis.legend()
    fig.savefig(output_dir / "fire_count_hist.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(rows: list[dict[str, Any]], cfg: MultiModelExperimentConfig) -> str:
    base, finetuned = (model.name for model in cfg.models)
    header = (
        "| feature | beta total | activation rho base | mechanism rho base | fires | low support "
        "| top matrix base | top matrix FT | robust grids |\n"
        "|---:|---:|---:|---:|---:|:---:|:---|:---|---:|\n"
    )
    body = "".join(
        f"| {row['feature_id']} | {row['beta_total']:.6g} | "
        f"{row[f'activation_rho_{base}']:.4f} | {row[f'mechanism_rho_{base}']:.4f} | "
        f"{row['fire_count']} | {'yes' if row['low_support'] else 'no'} | "
        f"{row[f'top_matrix_{base}']} | {row[f'top_matrix_{finetuned}']} | "
        f"{row['robustness_hits_of_9']}/9 |\n"
        for row in rows
    )
    return header + body if rows else "_No features matched._\n"


def _write_candidates_markdown(
    path: Path,
    groups: Mapping[str, list[dict[str, Any]]],
    tensors: Mapping[str, Tensor],
    examples: Mapping[str, list[dict[str, Any]]],
    cfg: MultiModelExperimentConfig,
) -> None:
    base, finetuned = (model.name for model in cfg.models)
    matrix_names = [matrix.name for matrix in cfg.models[0].matrices]
    sections = [
        "# P1–P5 candidate review\n\n",
        "Targets are filtered by the configured P4 category and ranked only by `beta_total`. "
        "Features with fewer than 32 validation fires are flagged, not silently discarded.\n\n",
    ]
    for title, key in (
        ("Base-dominant target mechanisms", "base_dominant_targets"),
        ("Finetuned-dominant target mechanisms", "finetuned_dominant_targets"),
        ("Shared/shared controls", "shared_shared_controls"),
        ("Concentrated/concentrated controls", "concentrated_concentrated_controls"),
    ):
        sections.extend([f"## {title}\n\n", _markdown_table(groups[key], cfg), "\n"])

    targets = groups["base_dominant_targets"] + groups["finetuned_dominant_targets"]
    sections.append("## Target mechanism details\n\n")
    for row in targets:
        feature_id = row["feature_id"]
        sections.append(f"### Feature {feature_id}\n\n")
        sections.append(
            f"- beta total: `{row['beta_total']:.6g}`\n"
            f"- fire density: `{row['fire_density']:.6g}`\n"
            f"- activation/mechanism rho delta: `{row['activation_mechanism_delta']:.4f}`\n"
            f"- threshold robustness: `{row['robustness_hits_of_9']}/9`\n\n"
        )
        sections.append(
            f"| matrix | locus {base} | locus {finetuned} | component cosine | relative change "
            "| read cosine | write cosine |\n"
            "|:---|---:|---:|---:|---:|---:|---:|\n"
        )
        for matrix in matrix_names:
            sections.append(
                f"| {matrix} | {float(tensors[f'locus/{base}/{matrix}'][feature_id]):.4f} | "
                f"{float(tensors[f'locus/{finetuned}/{matrix}'][feature_id]):.4f} | "
                f"{float(tensors[f'pair/{matrix}/component_cosine'][feature_id]):.4f} | "
                f"{float(tensors[f'pair/{matrix}/relative_component_change'][feature_id]):.4f} | "
                f"{float(tensors[f'pair/{matrix}/read_cosine'][feature_id]):.4f} | "
                f"{float(tensors[f'pair/{matrix}/write_cosine'][feature_id]):.4f} |\n"
            )
        contexts = examples.get(str(feature_id), [])[:3]
        sections.append("\nTop saved contexts:\n\n")
        if not contexts:
            sections.append("_Not present in the analyzer's capped context sample._\n\n")
        else:
            for example in contexts:
                text = str(example.get("text", "")).replace("\n", " ").replace("|", "\\|")
                sections.append(
                    f"- `g_s={float(example['g_s']):.4g}`, center "
                    f"`{example.get('center_token', '?')}`: {text}\n"
                )
            sections.append("\n")
    path.write_text("".join(sections), encoding="utf-8")


def build_posthoc_report(
    cfg: MultiModelExperimentConfig,
    analysis_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_per_direction: int = 10,
    control_count: int = 5,
    low_support_threshold: int = 32,
    dpi: int = 160,
) -> Path:
    """Validate raw analysis, then write tables, plots, and review candidates."""

    if min(top_per_direction, control_count, low_support_threshold, dpi) < 1:
        raise ValueError("report counts and dpi must be positive")
    analysis_dir = Path(analysis_dir)
    output_dir = Path(output_dir) if output_dir is not None else analysis_dir

    from safetensors.torch import load_file

    tensors = load_file(str(analysis_dir / "posthoc.safetensors"))
    taxonomy = json.loads((analysis_dir / "taxonomy.json").read_text(encoding="utf-8"))
    examples_path = analysis_dir / "top_activation_examples.json"
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    invariants = validate_posthoc_artifacts(tensors, taxonomy, cfg)
    rows = build_feature_rows(
        tensors, taxonomy, cfg, low_support_threshold=low_support_threshold
    )
    groups = select_review_groups(
        rows, cfg, top_per_direction=top_per_direction, control_count=control_count
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "feature_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    threshold_grid = {
        f"epsilon={epsilon:.2f},tau={threshold:.2f}": sum(
            abs(row[f"activation_rho_{cfg.models[0].name}"] - 0.5) < epsilon
            and (
                row[f"mechanism_rho_{cfg.models[0].name}"] >= threshold
                or row[f"mechanism_rho_{cfg.models[1].name}"] >= threshold
            )
            for row in rows
        )
        for epsilon in ROBUSTNESS_EPSILONS
        for threshold in ROBUSTNESS_THRESHOLDS
    }
    summary = {
        "models": [model.name for model in cfg.models],
        "n_features": cfg.sparsity.n_features,
        "validation_tokens": cfg.data.validation_tokens,
        "top_k": cfg.sparsity.top_k,
        "shared_epsilon": cfg.analysis.shared_epsilon,
        "concentrated_threshold": cfg.analysis.concentrated_threshold,
        "low_support_threshold": low_support_threshold,
        "low_support_features": sum(row["low_support"] for row in rows),
        "taxonomy_counts": {
            category: len(taxonomy[category]) for category in TAXONOMY_CATEGORIES
        },
        "invariants": invariants,
        "robustness_target_counts": threshold_grid,
        "review_feature_ids": {
            key: [row["feature_id"] for row in values] for key, values in groups.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_figures(tensors, taxonomy, cfg, output_dir, dpi=dpi)
    _write_candidates_markdown(
        output_dir / "candidates.md", groups, tensors, examples, cfg
    )
    return output_dir
