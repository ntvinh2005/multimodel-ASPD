"""Figures for single-feature editing results."""

import json
import re
from pathlib import Path

from aspd.eval.plots.style import (
    INK_MUTED,
    REFERENCE,
    grid,
    legend,
    note,
    ordinal_colors,
    save,
    step_label,
)

_STEP_RE = re.compile(r"^attr_edit_step(?P<step>\d+)\.json$")
_CONTROL_STYLE = {"random": (1.4, (4, 2)), "norm_matched": (1.4, (1, 2))}


def _load(attr_dir: Path) -> dict[int, dict]:
    out = {}
    for path in sorted(Path(attr_dir).glob("attr_edit_step*.json")):
        match = _STEP_RE.match(path.name)
        assert match, f"unexpected name {path.name}"
        out[int(match["step"])] = json.loads(path.read_text())
    return dict(sorted(out.items()))


def _series(report: dict, kind: str, metric: str) -> tuple[list[int], list[float]]:
    ks, ys = [], []
    for key, row in report["by_k"].items():
        k_kind, k = key.split("/")
        if k_kind == kind and metric in row:
            ks.append(int(k))
            ys.append(row[metric])
    order = sorted(range(len(ks)), key=lambda i: ks[i])
    return [ks[i] for i in order], [ys[i] for i in order]


def _draw_vs_k(ax, per_step: dict[int, dict], metric: str, title: str, ylabel: str, *, log: bool):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("components removed (k)")
    colors = ordinal_colors(len(per_step))
    drawn: list[float] = []
    all_k: set[int] = set()
    for color, (step, report) in zip(colors, per_step.items(), strict=True):
        ks, ys = _series(report, "ranked", metric)
        if ks:
            ax.plot(ks, ys, color=color, linewidth=2.0, marker="o", markersize=4,
                    label=f"ranked · {step_label(step)}", zorder=3)
            drawn += ys
            all_k |= set(ks)
    last = list(per_step.values())[-1]
    for kind, (width, dashes) in _CONTROL_STYLE.items():
        ks, ys = _series(last, kind, metric)
        if ks:
            ax.plot(ks, ys, color=REFERENCE, linewidth=width, dashes=dashes,
                    label=f"{kind} control", zorder=2)
            drawn += ys
    if not all_k:
        note(ax, f"no {metric} in any by_k block")
        return
    if log and all(y > 0 for y in drawn):
        ax.set_yscale("log")
    _k_axis(ax, all_k)
    legend(ax)


def _k_axis(ax, ks) -> None:
    """A log x axis ticked at the MEASURED k values and nowhere else."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    ax.set_xscale("log")
    ax.set_xticks(sorted(ks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v)}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _scatter(ax, report: dict, x_of, y_of, *, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ks = sorted({row["k"] for row in report["edits"]})
    colors = dict(zip(ks, ordinal_colors(len(ks)), strict=True))
    for kind in ("random", "norm_matched"):
        pts = [(x_of(r), y_of(r)) for r in report["edits"] if r["kind"] == kind]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=10, color=REFERENCE,
                       alpha=0.35, linewidths=0, label="controls" if kind == "random" else None,
                       zorder=2)
    for k in ks:
        pts = [(x_of(r), y_of(r)) for r in report["edits"] if r["kind"] == "ranked" and r["k"] == k]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=16, color=colors[k],
                       linewidths=0, label=f"k = {k}", zorder=3)
    legend(ax)


def plot_dir(attr_dir: Path) -> list[Path]:
    attr_dir = Path(attr_dir)
    per_step = _load(attr_dir)
    if not per_step:
        print(f"[plots] no attr_edit_step*.json under {attr_dir}", flush=True)
        return []
    last_step, last = list(per_step.items())[-1]

    fig, axes = grid(
        6, ncol=3, width=5.0, height=3.8,
        title=f"Attribution-ranked component editing — {Path(last['run_dir']).name}",
        subtitle=(
            f"{len(last['features'])} output features at {last['site']} · "
            f"C = {last['n_components']} · estimator {last['estimator']} · "
            f"{last['n_tokens']:,} tokens (A_j) / {last['n_tokens_global']:,} (global) · "
            f"controls: {last['meta']['config']['n_control_reps']} draws per k"
        ),
    )

    _draw_vs_k(axes[0], per_step, "on_aj_delta_abs",
               "target feature change on A_j", "mean |Δf_j|", log=True)
    _draw_vs_k(axes[1], per_step, "on_aj_collateral_abs",
               "collateral on A_j", "mean Σ_{i≠j} |Δf_i|", log=True)
    _draw_vs_k(axes[2], per_step, "on_aj_localization",
               "localization on A_j", "|Δf_j| / Σ_i |Δf_i|", log=False)
    _draw_vs_k(axes[3], per_step, "on_global_delta_abs",
               "target change on the global sample", "mean |Δf_j|", log=True)

    if not last["edits"]:
        for ax in (axes[4], axes[5]):
            note(ax, last.get("edit_skipped") or "no edits in this report")
    else:
        _scatter(
            axes[4], last,
            x_of=lambda r: r["on_aj"]["collateral_abs"],
            y_of=lambda r: r["on_aj"]["delta_abs"],
            title=f"selectivity — {step_label(last_step)}",
            xlabel="collateral on A_j", ylabel="|Δf_j| on A_j",
        )
        for axis, key in ((axes[4].set_xscale, "collateral_abs"), (axes[4].set_yscale, "delta_abs")):
            if all(r["on_aj"][key] > 0 for r in last["edits"]):
                axis("log")

        _scatter(
            axes[5], last,
            x_of=lambda r: r["predicted_delta_unit"],
            y_of=lambda r: r["on_aj"]["delta_signed"],
            title=f"attribution faithfulness — {step_label(last_step)}",
            xlabel="predicted Δf_j  (−Σ_{c∈S} a_unit)", ylabel="measured Δf_j",
        )
        lo = min(min(r["predicted_delta_unit"] for r in last["edits"]),
                 min(r["on_aj"]["delta_signed"] for r in last["edits"]))
        hi = max(max(r["predicted_delta_unit"] for r in last["edits"]),
                 max(r["on_aj"]["delta_signed"] for r in last["edits"]))
        axes[5].plot([lo, hi], [lo, hi], color=REFERENCE, linewidth=1.2, zorder=1)
        rho = last["faithfulness"].get("spearman_pooled")
        if rho is not None:
            axes[5].text(0.03, 0.95, f"Spearman (ranked, pooled) = {rho:.3f}", transform=
                         axes[5].transAxes, ha="left", va="top", fontsize=8, color=INK_MUTED)

    written = [save(fig, attr_dir / "attr_edit.png")]
    written += _plot_structure(attr_dir, per_step)
    return written


def _plot_structure(attr_dir: Path, per_step: dict[int, dict]) -> list[Path]:
    """Diagnostics about the RANKING itself, which the effect panels cannot show."""
    fig, axes = grid(
        3, ncol=3, width=5.0, height=3.8,
        title="Attribution structure",
        subtitle="score concentration, selection overlap across features, and gate-vs-unit ranking",
    )
    colors = ordinal_colors(len(per_step))

    ax = axes[0]
    ax.set_title("score share captured by top-k", fontsize=10, loc="left")
    ax.set_xlabel("k")
    ax.set_ylabel("Σ_{c∈S} |a_gate| / Σ_c |a_gate|")
    for color, (step, report) in zip(colors, per_step.items(), strict=True):
        ks, ys = _series(report, "ranked", "score_share")
        ax.plot(ks, ys, color=color, linewidth=2.0, marker="o", markersize=4,
                label=step_label(step))
    _k_axis(ax, {k for r in per_step.values() for k in _series(r, "ranked", "score_share")[0]})
    legend(ax)

    ax = axes[1]
    ax.set_title("selection overlap across features", fontsize=10, loc="left")
    ax.set_xlabel("k")
    ax.set_ylabel("mean pairwise Jaccard")
    for color, (step, report) in zip(colors, per_step.items(), strict=True):
        items = sorted(((int(k), v) for k, v in report["selection_overlap"].items()))
        ax.plot([k for k, _ in items], [v for _, v in items], color=color, linewidth=2.0,
                marker="o", markersize=4, label=step_label(step))
    _k_axis(ax, {int(k) for r in per_step.values() for k in r["selection_overlap"]})
    legend(ax)

    ax = axes[2]
    ax.set_title("gate ranking vs unit ranking", fontsize=10, loc="left")
    ax.set_xlabel("k")
    ax.set_ylabel("mean Jaccard over features")
    last = list(per_step.values())[-1]
    ks = sorted(int(k) for k in last["features"][0]["rank_overlap_gate_vs_unit"])
    if ks:
        means = [
            sum(f["rank_overlap_gate_vs_unit"][str(k)] for f in last["features"])
            / len(last["features"])
            for k in ks
        ]
        ax.plot(ks, means, color=colors[-1], linewidth=2.0, marker="o", markersize=4)
        _k_axis(ax, ks)
        ax.set_ylim(0, 1.02)
    else:
        note(ax, "no ranking overlap recorded")
    return [save(fig, attr_dir / "attr_edit_structure.png")]


def headline(attr_dir: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for step, report in _load(Path(attr_dir)).items():
        ks = [int(key.split("/")[1]) for key in report["by_k"] if key.startswith("ranked/")]
        if not ks:
            continue
        k = max(ks)
        ranked = report["by_k"][f"ranked/{k}"]
        row = {
            "k": float(k),
            "delta_abs": ranked["on_aj_delta_abs"],
            "delta_relative": ranked["on_aj_delta_relative"],
            "localization": ranked["on_aj_localization"],
            "collateral_abs": ranked["on_aj_collateral_abs"],
            "spearman_pooled": report["faithfulness"].get("spearman_pooled", float("nan")),
        }
        for kind in ("random", "norm_matched"):
            control = report["by_k"].get(f"{kind}/{k}")
            if control:
                row[f"{kind}_delta_abs"] = control["on_aj_delta_abs"]
        out[step] = row
    return out
