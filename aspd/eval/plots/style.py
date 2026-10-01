"""Shared figure style."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CATEGORICAL = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
MAX_SERIES = len(CATEGORICAL)

ORDINAL = (
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

REFERENCE = "#8a8983"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#fcfcfb"


def style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.title.set_color(INK)


def ordinal_colors(n: int) -> list[str]:
    """`n` steps of the blue ramp, light -> dark. For ordered series (checkpoints, k, scores)."""
    assert n >= 0
    if n == 0:
        return []
    if n == 1:
        return [ORDINAL[4]]
    idx = np.linspace(0, len(ORDINAL) - 1, n).round().astype(int)
    return [ORDINAL[i] for i in idx]


def series_color(i: int) -> str:
    assert 0 <= i < MAX_SERIES, (
        f"categorical slot {i} does not exist ({MAX_SERIES} slots). Split the panel with "
        "chunk_series() instead of cycling hues"
    )
    return CATEGORICAL[i]


def chunk_series(names: Sequence[str], size: int = MAX_SERIES) -> list[list[str]]:
    names = list(names)
    if len(names) <= size:
        return [names] if names else []
    n_panels = -(-len(names) // size)
    per = -(-len(names) // n_panels)  # balanced, so 9 -> 5 + 4 rather than 8 + 1
    return [names[i : i + per] for i in range(0, len(names), per)]


def grid(n_panels: int, *, ncol: int = 3, width: float = 4.6, height: float = 3.4,
         title: str | None = None, subtitle: str | None = None):
    assert n_panels >= 1, "nothing to plot"
    ncol = min(ncol, n_panels)
    nrow = -(-n_panels // ncol)
    fig_height = height * nrow + _HEADER_IN
    fig, axes = plt.subplots(nrow, ncol, figsize=(width * ncol, fig_height), squeeze=False)
    flat = list(axes.ravel())
    for ax in flat[n_panels:]:
        ax.remove()
    flat = flat[:n_panels]
    for ax in flat:
        style_axes(ax)
    if title:
        fig.suptitle(title, color=INK, fontsize=13, ha="left", x=0.01,
                     y=1 - 0.26 / fig_height, va="top")
    if subtitle:
        fig.text(0.01, 1 - (0.60 if title else 0.26) / fig_height, subtitle, color=INK_MUTED,
                 fontsize=9, ha="left", va="top")
    fig.sae_eval_top = 1 - _HEADER_IN / fig_height  # read by `save`
    return fig, flat


# Vertical inches reserved above the panels for the title + subtitle block.
_HEADER_IN = 0.85


def legend(ax, *, inside_max: int = 4, **kw) -> None:
    """A legend is present for >= 2 series and absent for one -- the title already names that
    one, and a one-entry box is pure noise. Pass `inside_max=0` to force the backed style on a
    panel whose marks fill the plot area (a stacked bar chart) whatever its series count.
    """
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) < 2:
        return
    style = (
        {"frameon": True, "facecolor": SURFACE, "edgecolor": GRID, "framealpha": 0.93}
        if len(handles) > inside_max else {"frameon": False}
    )
    leg = ax.legend(fontsize=7, labelcolor=INK_MUTED, **style, **kw)
    leg.set_zorder(6)
    leg.get_frame().set_linewidth(0.8)


def step_axis(ax, steps) -> None:
    """Checkpoint steps as ticks with compact labels."""
    steps = sorted(steps)
    if not steps:
        return
    ax.set_xticks(steps, [compact(s) for s in steps], fontsize=8)


def compact(n: int) -> str:
    n = int(n)
    if n and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return f"{n:,}"


def note(ax, text: str, *, width: int = 90) -> None:
    import textwrap

    wrapped = "\n".join(
        line for para in text.split("\n")
        for line in (textwrap.wrap(para, width) or [""])
    )
    ax.text(0.5, 0.5, wrapped, ha="center", va="center", color=INK_MUTED, fontsize=9,
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def step_label(step) -> str:
    return "frozen" if step is None else f"step {int(step):,}"


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, getattr(fig, "sae_eval_top", 0.965)))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plots] wrote {path}", flush=True)
    return path
