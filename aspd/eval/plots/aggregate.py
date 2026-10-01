"""One summary figure per run: matching and editing headline panels."""

import csv
import json
from pathlib import Path

from aspd.eval.plots import editing as attr_edit_plots
from aspd.eval.plots import editing_multi as attr_edit_multi_plots
from aspd.eval.plots import matching as matching_plots
from aspd.eval.plots.style import (
    INK,
    INK_MUTED,
    REFERENCE,
    SURFACE,
    legend,
    note,
    series_color,
    step_axis,
    style_axes,
)

def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- stage collectors


def collect(run_dir: Path | None, sae_dir: Path | None) -> dict[str, dict]:
    """Everything the aggregate can find, keyed by stage. Missing stages carry a `reason`."""
    stages: dict[str, dict] = {}

    matching_dir = (run_dir / "matching") if run_dir else None
    matching_mode = (
        matching_plots.resolve_mode(matching_dir)
        if matching_dir and matching_dir.exists() else None
    )
    stages["matching"] = (
        {"dir": matching_dir, "mode": matching_mode,
         "headline": matching_plots.headline(matching_dir, matching_mode)}
        if matching_mode
        else {"reason": f"no matching_*step*.json under {matching_dir}"}
    )

    attr_dir = (run_dir / "attr_edit") if run_dir else None
    attr = attr_edit_plots.headline(attr_dir) if attr_dir and attr_dir.exists() else {}
    stages["attr_edit"] = (
        {"dir": attr_dir, "headline": attr} if attr
        else {"reason": f"no attr_edit_step*.json under {attr_dir}"}
    )

    multi_dir = (run_dir / "attr_edit_multi") if run_dir else None
    multi = attr_edit_multi_plots.headline(multi_dir) if multi_dir and multi_dir.exists() else {}
    stages["attr_edit_multi"] = (
        {"dir": multi_dir, "headline": multi} if multi
        else {"reason": f"no m<m>_<setup>/ arms under {multi_dir}"}
    )
    return stages


# --------------------------------------------------------------------------- panel drawing


def _panels_matching(stage: dict):
    headline = stage["headline"]
    mode = stage.get("mode", "?")

    def mean(ax):
        ax.set_title(f"Mean judged score ({mode})", fontsize=10, loc="left")
        ax.set_ylabel("score (1–3)")
        ax.set_xlabel("checkpoint step")
        ax.set_ylim(0.95, 3.05)
        for i, (name, by_step) in enumerate(sorted(headline.items())):
            steps = sorted(by_step)
            color = REFERENCE if name.startswith("random") else series_color(i)
            ax.plot(steps, [by_step[s] for s in steps], color=color, linewidth=2.0, marker="o",
                    markersize=6, label=name, zorder=3)
        step_axis(ax, sorted({s for v in headline.values() for s in v}))
        legend(ax, loc="best")

    def gap(ax):
        ax.set_title("Lift over the random control", fontsize=10, loc="left")
        ax.set_ylabel("mean score − random")
        ax.set_xlabel("checkpoint step")
        control = next((v for k, v in headline.items() if k.startswith("random")), None)
        if control is None:
            note(ax, "no random control in the report")
            return
        ax.axhline(0.0, color=REFERENCE, linewidth=1.2, zorder=2)
        for i, (name, by_step) in enumerate(
            sorted((k, v) for k, v in headline.items() if not k.startswith("random"))
        ):
            steps = [s for s in sorted(by_step) if s in control]
            ax.plot(steps, [by_step[s] - control[s] for s in steps], color=series_color(i),
                    linewidth=2.0, marker="o", markersize=6, label=f"{name} − random", zorder=3)
        step_axis(ax, sorted(control))
        legend(ax, loc="best")

    return [mean, gap]


def _panels_attr_edit_multi(stage: dict):
    """The `m` axis at each arm's largest k, `cond` beside `global`."""
    headline: dict[str, dict[str, float]] = stage["headline"]
    ms = sorted({int(arm.split("_")[0][1:]) for arm in headline})

    def series(ax, metric: str, title: str, ylabel: str):
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("output features edited together (m)")
        for i, setup in enumerate(("cond", "global")):
            pts = [(m, headline[f"m{m}_{setup}"][metric]) for m in ms
                   if f"m{m}_{setup}" in headline and metric in headline[f"m{m}_{setup}"]]
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], color=series_color(i),
                        linewidth=2.0, marker="o", markersize=5, label=setup, zorder=3)
        if not ms:
            note(ax, "no arms")
            return
        ax.set_xscale("log")
        ax.set_xticks(ms)
        from matplotlib.ticker import FuncFormatter, NullFormatter

        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v)}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        legend(ax)

    return [
        lambda ax: series(ax, "on_a_delta_abs", "target change on A_J", "mean Σ_{j∈J} |Δf_j|"),
        lambda ax: series(ax, "on_a_localization", "localization", "Σ_{j∈J}|Δf_j| / Σ_i |Δf_i|"),
        lambda ax: series(ax, "sharing_rate", "component sharing", "1 − |S| / (m·k)"),
    ]


def _panels_attr_edit(stage: dict):
    """Effect, localization and faithfulness at the LARGEST k, per checkpoint."""
    headline: dict[int, dict[str, float]] = stage["headline"]
    steps = sorted(headline)

    def series(ax, names: list[str], title: str, ylabel: str):
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("checkpoint step")
        for i, name in enumerate(names):
            ys = [(s, headline[s][name]) for s in steps if name in headline[s]]
            if not ys:
                continue
            control = name.startswith(("random", "norm_matched"))
            ax.plot([x for x, _ in ys], [y for _, y in ys],
                    color=REFERENCE if control else series_color(i),
                    linewidth=1.4 if control else 2.0, marker="o", markersize=5,
                    label=name, zorder=2 if control else 3)
        step_axis(ax, steps)
        legend(ax)

    def effect(ax):
        series(ax, ["delta_abs", "random_delta_abs", "norm_matched_delta_abs"],
               f"|Δf_j| on A_j at k = {int(headline[steps[-1]]['k'])}", "mean |Δf_j|")

    def localization(ax):
        series(ax, ["localization"], "localization", "|Δf_j| / Σ_i |Δf_i|")

    def faithfulness(ax):
        series(ax, ["spearman_pooled"], "attribution faithfulness", "Spearman ρ")

    return [effect, localization, faithfulness]


def _as_rows(built) -> list[list]:
    return built if built and isinstance(built[0], list) else [built]


_STAGES = (
    ("matching", "Meaning localization (judged component-feature matching)", _panels_matching),
    ("attr_edit", "Weight editing, single target feature", _panels_attr_edit),
    ("attr_edit_multi", "Weight editing, multiple target features", _panels_attr_edit_multi),
)


# --------------------------------------------------------------------------- entry point


def plot_aggregate(run_dir: Path | None, sae_dir: Path | None, out_dir: Path) -> list[Path]:
    """`aggregate.png` + `aggregate.json` + `aggregate.csv`, one section per evaluation stage."""
    import matplotlib.pyplot as plt

    assert run_dir or sae_dir, "aggregate needs a run dir, a dictionary dir, or both"
    stages = collect(run_dir, sae_dir)
    out_dir = Path(out_dir)

    built = []
    for key, title, builder in _STAGES:
        stage = stages[key]
        if "reason" in stage:
            built.append((key, title, stage["reason"], []))
        else:
            built.append((key, title, "", _as_rows(builder(stage))))

    heights = [3.9 * len(rows) if rows else 1.3 for _, _, _, rows in built]
    fig = plt.figure(figsize=(16.0, sum(heights) + 0.55))
    subfigs = list(fig.subfigures(len(built) + 1, 1, height_ratios=[0.55, *heights]))
    header, subfigs = subfigs[0], subfigs[1:]
    header.text(0.012, 0.35,
                f"Evaluation summary — run {Path(run_dir).name if run_dir else '(none)'} · "
                f"dictionary {Path(sae_dir).name if sae_dir else '(none)'}",
                color=INK, fontsize=15, ha="left", va="bottom")
    header.text(0.012, 0.30,
                "headline metrics only — every metric is in each stage's own figures",
                color=INK_MUTED, fontsize=9, ha="left", va="top")

    for subfig, (key, title, reason, rows) in zip(subfigs, built, strict=True):
        subfig.suptitle(title, color=INK, fontsize=12, ha="left", x=0.012)
        if not rows:
            ax = subfig.subplots(1, 1)
            style_axes(ax)
            note(ax, f"not run — {reason}")
            continue
        ncols = max(len(r) for r in rows)
        axes = subfig.subplots(len(rows), ncols, squeeze=False)
        if len(rows) > 1:
            subfig.subplots_adjust(hspace=0.55)
        for row_axes, panels in zip(axes, rows, strict=True):
            for i, ax in enumerate(row_axes):
                if i >= len(panels):
                    ax.set_axis_off()
                    continue
                style_axes(ax)
                panels[i](ax)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "aggregate.png"
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {path}", flush=True)

    return [path, *_write_tables(run_dir, sae_dir, stages, out_dir)]


def _write_tables(run_dir, sae_dir, stages: dict[str, dict], out_dir: Path) -> list[Path]:
    """The same headline numbers as text, because a PNG cannot be grepped or pasted."""
    rows: list[dict] = []

    for name, by_step in stages["matching"].get("headline", {}).items():
        rows += [{"stage": "matching", "source": f"{name}[{stages['matching'].get('mode')}]",
                  "step": s, "metric": "mean_score",
                  "value": v} for s, v in sorted(by_step.items())]

    for step, flat in sorted(stages["attr_edit"].get("headline", {}).items()):
        rows += [{"stage": "attr_edit", "source": "ranked_top_k", "step": step, "metric": m,
                  "value": v} for m, v in sorted(flat.items())]

    for arm, flat in sorted(stages["attr_edit_multi"].get("headline", {}).items()):
        rows += [{"stage": "attr_edit_multi", "source": arm, "step": int(flat.get("step", 0)),
                  "metric": m, "value": v} for m, v in sorted(flat.items())]

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aggregate.json"
    json_path.write_text(json.dumps(
        {
            "run_dir": str(Path(run_dir).resolve()) if run_dir else None,
            "sae_dir": str(Path(sae_dir).resolve()) if sae_dir else None,
            "missing": {k: v["reason"] for k, v in stages.items() if "reason" in v},
            "rows": rows,
        },
        indent=2, sort_keys=True, default=str,
    ))
    csv_path = out_dir / "aggregate.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stage", "source", "step", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[plots] wrote {json_path} and {csv_path}", flush=True)
    return [json_path, csv_path]
