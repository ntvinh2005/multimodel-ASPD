"""Training-curve plots for the mixed train/validation rows in ``metrics.jsonl``."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_training_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL metrics and report a malformed row with its line number."""

    if isinstance(path, str) and not path.strip():
        raise ValueError(
            "metrics path is empty; define $METRICS or pass the metrics.jsonl path directly"
        )
    path = Path(path)
    if path.is_dir():
        path = path / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object in {path} at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no metric rows found in {path}")
    return rows


def _numeric(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _series(rows: Iterable[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    # A resumed/restarted run can append the same step more than once; keep the latest occurrence.
    by_step: dict[int, float] = {}
    for row in rows:
        step = row.get("step")
        value = row.get(key)
        if _numeric(step) and _numeric(value):
            by_step[int(step)] = float(value)
    points = sorted(by_step.items())
    return [step for step, _ in points], [value for _, value in points]


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window == 1:
        return values
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        result.append(running / min(index + 1, window))
    return result


def _model_names(rows: Iterable[dict[str, Any]], family: str) -> list[str]:
    names: set[str] = set()
    for row in rows:
        for key in row:
            parts = key.split("/")
            # Keep aggregate act/<model> and internal/<model>, excluding per-matrix metrics.
            if len(parts) == 3 and parts[0] in {"train", "validation"} and parts[1] == family:
                names.add(parts[2])
    return sorted(names)


def _matrix_names(rows: Iterable[dict[str, Any]], model_name: str) -> list[str]:
    prefix = f"validation/internal/{model_name}/"
    return sorted(
        {
            key.removeprefix(prefix)
            for row in rows
            for key in row
            if key.startswith(prefix) and "/" not in key.removeprefix(prefix)
        }
    )


def _plot_key(
    axis: Any,
    rows: list[dict[str, Any]],
    key: str,
    label: str,
    color: object,
    smooth: int,
) -> bool:
    steps, values = _series(rows, key)
    if not steps:
        return False
    validation = key.startswith("validation/")
    if not validation:
        values = _rolling_mean(values, smooth)
    axis.plot(
        steps,
        values,
        label=label,
        color=color,
        linestyle="--" if validation else "-",
        marker="o" if validation else None,
        markersize=3,
        linewidth=1.8,
        alpha=0.95 if validation else 0.8,
    )
    return True


def _finish_axis(axis: Any, title: str, ylabel: str, plotted: bool) -> None:
    axis.set_title(title)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "No matching metrics", ha="center", va="center", alpha=0.6)


def plot_training_metrics(
    rows: list[dict[str, Any]],
    output: str | Path,
    *,
    smooth: int = 1,
    target_l0: float | None = None,
    title: str | None = None,
    dpi: int = 160,
) -> Path:
    """Write a four-panel training dashboard and return its final path."""

    if smooth < 1:
        raise ValueError("smooth must be at least 1")
    if dpi < 1:
        raise ValueError("dpi must be positive")

    output = Path(output)
    if not output.suffix:
        output = output.with_suffix(".png")
    supported = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"}
    if output.suffix.lower() not in supported:
        raise ValueError(f"unsupported output extension {output.suffix!r}; choose {sorted(supported)}")

    # Avoid slow or unwritable home-directory font caches on cluster/login nodes.
    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "aspd-matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors

    loss_axis = axes[0, 0]
    plotted = _plot_key(loss_axis, rows, "train/loss", "train", colors[0], smooth)
    plotted |= _plot_key(loss_axis, rows, "validation/loss", "validation", colors[0], smooth)
    _finish_axis(loss_axis, "Total objective", "Loss", plotted)

    activation_axis = axes[0, 1]
    plotted = False
    for index, model_name in enumerate(_model_names(rows, "act")):
        color = colors[index % len(colors)]
        plotted |= _plot_key(
            activation_axis,
            rows,
            f"train/act/{model_name}",
            f"{model_name} train",
            color,
            smooth,
        )
        plotted |= _plot_key(
            activation_axis,
            rows,
            f"validation/act/{model_name}",
            f"{model_name} validation",
            color,
            smooth,
        )
    _finish_axis(activation_axis, "Activation reconstruction", "FVU", plotted)

    internal_axis = axes[1, 0]
    plotted = False
    for index, model_name in enumerate(_model_names(rows, "internal")):
        color = colors[index % len(colors)]
        plotted |= _plot_key(
            internal_axis,
            rows,
            f"train/internal/{model_name}",
            f"{model_name} train",
            color,
            smooth,
        )
        plotted |= _plot_key(
            internal_axis,
            rows,
            f"validation/internal/{model_name}",
            f"{model_name} validation",
            color,
            smooth,
        )
    _finish_axis(internal_axis, "Parameter reconstruction", "FVU", plotted)

    sparsity_axis = axes[1, 1]
    plotted_l0 = _plot_key(
        sparsity_axis, rows, "train/sparsity/l0", "L0 train", colors[2], smooth
    )
    plotted_l0 |= _plot_key(
        sparsity_axis,
        rows,
        "validation/sparsity/l0",
        "L0 validation",
        colors[2],
        smooth,
    )
    if target_l0 is not None:
        sparsity_axis.axhline(target_l0, color=colors[2], linestyle=":", label=f"target {target_l0:g}")
        plotted_l0 = True
    sparsity_axis.set_title("Sparsity and feature health")
    sparsity_axis.set_xlabel("Optimizer step")
    sparsity_axis.set_ylabel("Mean L0")
    sparsity_axis.grid(True, alpha=0.25)

    dead_axis = sparsity_axis.twinx()
    plotted_dead = _plot_key(
        dead_axis,
        rows,
        "train/sparsity/dead_fraction",
        "dead fraction train",
        colors[3],
        smooth,
    )
    plotted_dead |= _plot_key(
        dead_axis,
        rows,
        "validation/sparsity/dead_fraction",
        "dead fraction validation",
        colors[3],
        smooth,
    )
    dead_axis.set_ylabel("Dead fraction")
    dead_axis.set_ylim(bottom=0)
    handles, labels = sparsity_axis.get_legend_handles_labels()
    dead_handles, dead_labels = dead_axis.get_legend_handles_labels()
    if plotted_l0 or plotted_dead:
        sparsity_axis.legend(handles + dead_handles, labels + dead_labels, fontsize=8)
    else:
        sparsity_axis.text(
            0.5, 0.5, "No matching metrics", ha="center", va="center", alpha=0.6
        )

    fig.suptitle(title or "Multi-model ASPD training", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_validation_internal_by_matrix(
    rows: list[dict[str, Any]],
    output: str | Path,
    *,
    title: str | None = None,
    dpi: int = 160,
) -> Path:
    """Plot validation internal FVU for every matrix, with one panel per model."""

    if dpi < 1:
        raise ValueError("dpi must be positive")

    output = Path(output)
    if not output.suffix:
        output = output.with_suffix(".png")
    supported = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"}
    if output.suffix.lower() not in supported:
        raise ValueError(f"unsupported output extension {output.suffix!r}; choose {sorted(supported)}")

    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "aspd-matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = [
        name for name in _model_names(rows, "internal") if _matrix_names(rows, name)
    ]
    if not model_names:
        raise ValueError("no validation per-matrix internal FVU metrics found")

    fig, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(7 * len(model_names), 5),
        squeeze=False,
        constrained_layout=True,
        sharey=True,
    )
    colors = plt.get_cmap("tab10").colors
    for model_index, model_name in enumerate(model_names):
        axis = axes[0, model_index]
        plotted = False
        for matrix_index, matrix_name in enumerate(_matrix_names(rows, model_name)):
            plotted |= _plot_key(
                axis,
                rows,
                f"validation/internal/{model_name}/{matrix_name}",
                matrix_name,
                colors[matrix_index % len(colors)],
                smooth=1,
            )
        _finish_axis(axis, model_name, "Validation internal FVU", plotted)

    fig.suptitle(title or "Validation internal FVU by matrix", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output
