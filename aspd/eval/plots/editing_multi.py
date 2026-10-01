"""Figures for multiple-feature editing results."""

import json
import re
from pathlib import Path

from aspd.eval.plots.style import (
    REFERENCE,
    grid,
    legend,
    note,
    ordinal_colors,
    save,
    series_color,
    step_label,
)

_STEP_RE = re.compile(r"^attr_edit_multi_step(?P<step>\d+)\.json$")
_ARM_RE = re.compile(r"^m(?P<m>\d+)_(?P<setup>cond|global)$")
_CONTROL_STYLE = {"random": (1.4, (4, 2)), "norm_matched": (1.4, (1, 2))}
_SETUP_DASH = {"cond": None, "global": (5, 2)}


def _load_arm(arm_dir: Path) -> dict[int, dict]:
    out = {}
    for path in sorted(Path(arm_dir).glob("attr_edit_multi_step*.json")):
        match = _STEP_RE.match(path.name)
        assert match, f"unexpected name {path.name}"
        out[int(match["step"])] = json.loads(path.read_text())
    return dict(sorted(out.items()))


def _series(report: dict, kind: str, metric: str, block: str = "union"):
    ks, ys = [], []
    for key, row in report["by_k"].items():
        k_kind, k, k_block = key.split("/")
        if k_kind == kind and k_block == block and metric in row:
            ks.append(int(k))
            ys.append(row[metric])
    order = sorted(range(len(ks)), key=lambda i: ks[i])
    return [ks[i] for i in order], [ys[i] for i in order]


def _k_axis(ax, ks) -> None:
    """A log x axis ticked at the MEASURED k values and nowhere else -- `plots/attr_edit.py`'s."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    ax.set_xscale("log")
    ax.set_xticks(sorted(ks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v)}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _draw_vs_k(ax, per_step: dict[int, dict], metric: str, title: str, ylabel: str, *, log: bool):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("components per feature (k)")
    colors = ordinal_colors(len(per_step))
    drawn, all_k = [], set()
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


def plot_arm_dir(arm_dir: Path) -> list[Path]:
    """Six panels for one `(m, setup)` arm."""
    arm_dir = Path(arm_dir)
    per_step = _load_arm(arm_dir)
    if not per_step:
        print(f"[plots] no attr_edit_multi_step*.json under {arm_dir}", flush=True)
        return []
    last_step, last = list(per_step.items())[-1]
    m, setup = last["m"], last["setup"]

    fig, axes = grid(
        6, ncol=3, width=5.0, height=3.8,
        title=f"Multi-feature editing (m = {m}, {setup}) — {Path(last['run_dir']).name}",
        subtitle=(
            f"{last['n_combinations']} combinations of {m} output features at {last['site']} · "
            f"C = {last['n_components']} · estimator {last['estimator']} · "
            f"{last['n_tokens']:,} tokens (A_J) / {last['n_tokens_global']:,} (global) · "
            f"controls: {last['meta']['config']['n_control_reps']} draws per k on "
            f"{last['n_control_combos']}/{last['n_combinations']} combinations"
        ),
    )
    _draw_vs_k(axes[0], per_step, "on_a_delta_abs",
               "target change on A_J", "mean Σ_{j∈J} |Δf_j|", log=True)
    _draw_vs_k(axes[1], per_step, "on_a_collateral_abs",
               "collateral on A_J", "mean Σ_{i∉J} |Δf_i|", log=True)
    _draw_vs_k(axes[2], per_step, "on_a_localization",
               "localization on A_J", "Σ_{j∈J}|Δf_j| / Σ_i |Δf_i|", log=False)
    _draw_vs_k(axes[3], per_step, "on_global_delta_abs",
               "target change on the global sample", "mean Σ_{j∈J} |Δf_j|", log=True)

    union = [r for r in last["edits"] if r["block"] == "union"]
    if not union:
        for ax in (axes[4], axes[5]):
            note(ax, last.get("edit_skipped") or "no edits in this report")
    else:
        _scatter(axes[4], union,
                 x_of=lambda r: r["on_a"]["collateral_abs"], y_of=lambda r: r["on_a"]["delta_abs"],
                 title=f"selectivity — {step_label(last_step)}",
                 xlabel="collateral on A_J", ylabel="Σ_{j∈J}|Δf_j| on A_J")
        for setter, key in ((axes[4].set_xscale, "collateral_abs"),
                            (axes[4].set_yscale, "delta_abs")):
            if all(r["on_a"][key] > 0 for r in union):
                setter("log")
        _scatter(axes[5], union,
                 x_of=lambda r: r["predicted_delta_unit"],
                 y_of=lambda r: r["on_a"]["delta_signed"],
                 title=f"faithfulness — {step_label(last_step)}",
                 xlabel="predicted Δ (support-weighted)", ylabel="measured Δ on A_J")
        lo = min(min(r["predicted_delta_unit"] for r in union),
                 min(r["on_a"]["delta_signed"] for r in union))
        hi = max(max(r["predicted_delta_unit"] for r in union),
                 max(r["on_a"]["delta_signed"] for r in union))
        axes[5].plot([lo, hi], [lo, hi], color=REFERENCE, linewidth=1.0, dashes=(3, 3), zorder=1)
    return [save(fig, arm_dir / f"attr_edit_multi_m{m}_{setup}.png")]


def _scatter(ax, rows: list[dict], x_of, y_of, *, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ks = sorted({row["k"] for row in rows})
    colors = dict(zip(ks, ordinal_colors(len(ks)), strict=True))
    for kind in ("random", "norm_matched"):
        pts = [(x_of(r), y_of(r)) for r in rows if r["kind"] == kind]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=10, color=REFERENCE,
                       alpha=0.35, linewidths=0,
                       label="controls" if kind == "random" else None, zorder=2)
    for k in ks:
        pts = [(x_of(r), y_of(r)) for r in rows if r["kind"] == "ranked" and r["k"] == k]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=16, color=colors[k],
                       linewidths=0, label=f"k = {k}", zorder=3)
    legend(ax)


def load_arms(multi_dir: Path) -> dict[tuple[int, str], dict]:
    """`{(m, setup): last checkpoint's report}` for every arm present under `attr_edit_multi/`."""
    out: dict[tuple[int, str], dict] = {}
    for path in sorted(Path(multi_dir).iterdir()):
        match = _ARM_RE.match(path.name) if path.is_dir() else None
        if not match:
            continue
        per_step = _load_arm(path)
        if per_step:
            out[(int(match["m"]), match["setup"])] = list(per_step.values())[-1]
    return dict(sorted(out.items()))


def _vs_m(ax, arms: dict[tuple[int, str], dict], metric: str, k: int, title: str, ylabel: str,
          *, log: bool, block: str = "union", source: str = "by_k"):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("output features edited together (m)")
    drawn, any_point = [], False
    for i, setup in enumerate(("cond", "global")):
        ms, ys = [], []
        for (m, arm_setup), report in arms.items():
            if arm_setup != setup:
                continue
            if source == "structure":
                row = report["structure"].get(str(k), {})
            else:
                row = report["by_k"].get(f"ranked/{k}/{block}", {})
            if metric in row:
                ms.append(m)
                ys.append(row[metric])
        if ms:
            ax.plot(ms, ys, color=series_color(i), linewidth=2.0, marker="o", markersize=4,
                    dashes=_SETUP_DASH[setup] or (1, 0), label=f"{setup} · k = {k}", zorder=3)
            drawn += ys
            any_point = True
    if not any_point:
        note(ax, f"no {metric} at k = {k} in any arm")
        return
    if log and all(y > 0 for y in drawn):
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(sorted({m for m, _ in arms}))
    from matplotlib.ticker import FuncFormatter, NullFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v)}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    legend(ax)


def plot_multi_root(multi_dir: Path) -> list[Path]:
    """The cross-arm figure -- the one the `m` axis exists for. Also writes `sweep_multi.json`."""
    multi_dir = Path(multi_dir)
    arms = load_arms(multi_dir)
    if not arms:
        print(f"[plots] no m<m>_<setup>/ arms under {multi_dir}", flush=True)
        return []
    any_report = next(iter(arms.values()))
    ks = sorted({int(key.split("/")[1]) for key in any_report["by_k"]})
    k = 10 if 10 in ks else ks[-1]

    written = [_write_sweep_multi(multi_dir, arms)]
    fig, axes = grid(
        6, ncol=3, width=5.0, height=3.8,
        title=f"Multi-feature editing across m — {Path(any_report['run_dir']).name}",
        subtitle=(
            f"{len(arms)} arms · {any_report['n_combinations']} combinations each at "
            f"{any_report['site']} · union block, k = {k} components per feature · "
            f"last checkpoint of each arm"
        ),
    )
    _vs_m(axes[0], arms, "on_a_delta_abs", k,
          "target change on A_J", "mean Σ_{j∈J} |Δf_j|", log=True)
    _vs_m(axes[1], arms, "on_a_collateral_abs", k,
          "collateral on A_J", "mean Σ_{i∉J} |Δf_i|", log=True)
    _vs_m(axes[2], arms, "on_a_localization", k,
          "localization on A_J", "Σ_{j∈J}|Δf_j| / Σ_i |Δf_i|", log=False)
    _vs_m(axes[3], arms, "on_a_delta_abs", k, "per-target change on its own A_j",
          "mean |Δf_j|", log=True, block="target")
    _vs_m(axes[4], arms, "sharing_rate", k, "component sharing within a combination",
          "1 − |S| / (m·k)", log=False, source="structure")
    _vs_m(axes[5], arms, "rank_overlap_cond_vs_global", k,
          "cond vs global ranking overlap", "mean Jaccard of top-k", log=False, source="structure")
    written.append(save(fig, multi_dir / "attr_edit_multi.png"))
    return written


def _write_sweep_multi(multi_dir: Path, arms: dict[tuple[int, str], dict]) -> Path:
    path = multi_dir / "sweep_multi.json"
    path.write_text(json.dumps(
        {
            "arms": [
                {
                    "m": m, "setup": setup, "step": r["step"],
                    "n_combinations": r["n_combinations"],
                    "n_control_combos": r["n_control_combos"],
                    "estimator": r["estimator"],
                    "edit_skipped": r["edit_skipped"],
                    "global_skipped": r["global_skipped"],
                    "by_k": r["by_k"],
                    "faithfulness": r["faithfulness"],
                    "structure": r["structure"],
                }
                for (m, setup), r in arms.items()
            ],
        },
        indent=2, sort_keys=True,
    ))
    print(f"[plots] wrote {path}", flush=True)
    return path


def headline(multi_dir: Path) -> dict:
    """`{("m<m>_<setup>"): {metric: value}}` at the largest k -- what `plots/aggregate.py` reads."""
    arms = load_arms(multi_dir)
    out: dict[str, dict[str, float]] = {}
    for (m, setup), report in arms.items():
        keys = [key for key in report["by_k"] if key.startswith("ranked/")
                and key.endswith("/union")]
        if not keys:
            continue
        top = max(keys, key=lambda key: int(key.split("/")[1]))
        row = report["by_k"][top]
        out[f"m{m}_{setup}"] = {
            "k": float(top.split("/")[1]),
            "step": float(report["step"] or 0),
            **{name: row[name] for name in (
                "on_a_delta_abs", "on_a_delta_relative", "on_a_collateral_abs",
                "on_a_localization", "selection_size", "sharing_rate",
            ) if name in row},
        }
    return out
