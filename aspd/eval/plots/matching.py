"""Figures for matching results."""

import json
from pathlib import Path

from aspd.eval.matching import DEFAULT_MODE, MODES, REPORT_RE, mode_of
from aspd.eval.plots.style import (
    INK_MUTED,
    REFERENCE,
    compact,
    grid,
    legend,
    note,
    ordinal_colors,
    save,
    series_color,
    step_axis,
)

_SCORES = ("1", "2", "3")


def _pairing_color(name: str, order: list[str]) -> str:
    if name.startswith("random"):
        return REFERENCE
    entities = [n for n in order if not n.startswith("random")]
    return series_color(entities.index(name))


def _load(matching_dir: Path, mode: str) -> dict[int, dict]:
    """Every report of ONE mode in the directory, keyed by step."""
    assert mode in MODES, mode
    out = {}
    for path in sorted(Path(matching_dir).glob("matching_*step*.json")):
        match = REPORT_RE.match(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        found = match["mode"] or mode_of(data.get("meta", {}))
        if found == mode:
            out[int(match["step"])] = data
    return dict(sorted(out.items()))


def modes_present(matching_dir: Path) -> list[str]:
    """Which modes this directory holds, in `MODES` order."""
    return [m for m in MODES if _load(Path(matching_dir), m)]


def resolve_mode(matching_dir: Path, prefer: str = DEFAULT_MODE) -> str | None:
    """The one mode a single-number reader should use: `prefer` if present, else whatever is."""
    present = modes_present(matching_dir)
    if not present:
        return None
    return prefer if prefer in present else present[0]


def plot_dir(matching_dir: Path) -> list[Path]:
    """One figure per mode present -- `plots` draws whatever ran, and never mixes the two."""
    matching_dir = Path(matching_dir)
    present = modes_present(matching_dir)
    if not present:
        print(f"[plots] no matching_*step*.json under {matching_dir}", flush=True)
        return []
    return [p for mode in present for p in _plot_mode(matching_dir, mode)]


def _plot_mode(matching_dir: Path, mode: str) -> list[Path]:
    per_step = _load(matching_dir, mode)

    steps = list(per_step)
    order: list[str] = []
    for data in per_step.values():
        for result in data["results"]:
            if result["name"] not in order:
                order.append(result["name"])
    meta = per_step[steps[0]]["meta"]

    def value(step: int, name: str, field: str):
        for result in per_step[step]["results"]:
            if result["name"] == name:
                return result[field]
        return None

    fig, axes = grid(
        3, ncol=3, width=5.2, height=4.0,
        title=(("Component ↔ output-feature matching" if mode == "c2o"
                else "Input ↔ output feature matching")
               + f" ({mode}) — {Path(meta.get('run_dir', '')).name}"),
        subtitle=(
            f"judge {meta.get('judge_model', '?')} · {meta.get('n_subsample', '?')} components · "
            f"per configuration · ≥{meta.get('min_examples', '?')} harvested windows per side · "
            f"{meta.get('n_tokens', '?'):,} tokens"
            if isinstance(meta.get("n_tokens"), int) else str(meta.get("output_site", ""))
        ),
    )
    _plot_mean(axes[0], steps, order, value)
    _plot_distribution(axes[1], steps, order, value)
    _plot_counts(axes[2], steps, order, value)
    return [save(fig, matching_dir / f"matching_{mode}.png")]


def _plot_mean(ax, steps, order, value) -> None:
    ax.set_title("Mean judged score", fontsize=10, loc="left")
    ax.set_ylabel("mean score (1 = unrelated, 3 = same concept)")
    ax.set_ylim(0.95, 3.05)
    ax.axhline(1.0, color=REFERENCE, linewidth=1.0, linestyle="--", zorder=2)
    if len(steps) == 1:
        for i, name in enumerate(order):
            val = value(steps[0], name, "mean_score")
            ax.bar([i], [val], color=_pairing_color(name, order), width=0.6, zorder=3)
            ax.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=INK_MUTED)
        ax.set_xticks(range(len(order)), order, fontsize=8, color=INK_MUTED)
        ax.set_xlabel(f"pairing (step {steps[0]:,})")
        return
    for name in order:
        ys = [value(s, name, "mean_score") for s in steps]
        xs = [s for s, y in zip(steps, ys, strict=True) if y is not None]
        vals = [y for y in ys if y is not None]
        ax.plot(xs, vals, color=_pairing_color(name, order), linewidth=2.0, marker="o",
                markersize=6, label=name, zorder=3)
    ax.set_xlabel("checkpoint step")
    step_axis(ax, steps)
    legend(ax, loc="best")


def _plot_distribution(ax, steps, order, value) -> None:
    """Stacked fractions of the {1, 2, 3} histogram, one bar per (step, pairing)."""
    ax.set_title("Score distribution", fontsize=10, loc="left")
    ax.set_ylabel("fraction of judged pairs")
    colors = ordinal_colors(len(_SCORES))
    labels, xs = [], []
    groups: list[tuple[float, int]] = []  # (centre, step) for the second-level x labels
    bottoms: dict[int, float] = {}
    pos = 0
    for step in steps:
        first = pos
        for name in order:
            hist = value(step, name, "score_histogram") or {}
            total = sum(hist.get(s, 0) for s in _SCORES)
            if not total:
                continue
            xs.append(pos)
            labels.append(name)
            bottoms[pos] = 0.0
            for score, color in zip(_SCORES, colors, strict=True):
                frac = hist.get(score, 0) / total
                ax.bar([pos], [frac], bottom=[bottoms[pos]], color=color, width=0.72,
                       linewidth=1.5, edgecolor="white", zorder=3,
                       label=f"score {score}" if pos == 0 else None)
                bottoms[pos] += frac
            pos += 1
        if pos > first:
            groups.append(((first + pos - 1) / 2, step))
        pos += 0.6  # a wider gap between checkpoint groups than within one
    if not xs:
        note(ax, "no judged pairs")
        return
    ax.set_xticks(xs, labels, fontsize=7, color=INK_MUTED, rotation=45, ha="right")
    for centre, step in groups:
        ax.annotate(f"{step:,}", xy=(centre, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -30), textcoords="offset points", ha="center", va="top",
                    fontsize=7, color=INK_MUTED)
    ax.set_ylim(0, 1.0)
    legend(ax, inside_max=0, loc="upper center", ncol=3)


def _plot_counts(ax, steps, order, value) -> None:
    ax.set_title("Pairs judged", fontsize=10, loc="left")
    ax.set_ylabel("n_pairs")
    width = 0.8 / max(1, len(order))
    for i, name in enumerate(order):
        xs = [j + i * width for j in range(len(steps))]
        ys = [value(s, name, "n_pairs") or 0 for s in steps]
        ax.bar(xs, ys, width=width * 0.9, color=_pairing_color(name, order), label=name, zorder=3)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(steps))],
                  [compact(s) for s in steps], fontsize=8, color=INK_MUTED)
    ax.set_xlabel("checkpoint step")
    legend(ax, inside_max=0, loc="best")


def headline(matching_dir: Path, mode: str | None = None) -> dict[str, dict[int, float]]:
    """`{pairing: {step: mean_score}}` for ONE mode -- what the aggregate figure reads."""
    mode = mode or resolve_mode(Path(matching_dir))
    if mode is None:
        return {}
    out: dict[str, dict[int, float]] = {}
    for step, data in _load(Path(matching_dir), mode).items():
        for result in data["results"]:
            out.setdefault(result["name"], {})[step] = result["mean_score"]
    return out


# Re-exported so `plots/__init__` can treat every module the same way.
__all__ = ["headline", "modes_present", "plot_dir", "resolve_mode"]
