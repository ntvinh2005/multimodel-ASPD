"""The paper tables: one row per method at its last checkpoint.

Collects intruder, diversity, matching and editing results from each run directory and writes a
compact CSV (value +- 95% CI) plus a full CSV.
"""

import csv
import json
import re
from collections.abc import Callable
from pathlib import Path

from aspd.eval.matching import DEFAULT_MODE, MODES, REPORT_RE, mode_of

_MODEL_STEP = re.compile(r"^model_(?P<step>\d+)\.pth$")
_H_STEP = re.compile(r"^h-step(?P<step>\d+)$")

_ATTR_METRICS = (("target_change", "delta_abs"), ("collateral", "collateral_abs"),
                 ("localization", "localization"))


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def checkpoint_steps(run_dir: Path) -> list[int]:
    """Every `model_<step>.pth` on disk, ascending. The checkpoint axis is the weights, not the
    evals: an eval directory can lag, and `last_step` must not follow it down.
    """
    return sorted(
        int(m["step"]) for m in
        (_MODEL_STEP.match(p.name) for p in Path(run_dir).glob("model_*.pth")) if m
    )


def last_step(run_dir: Path) -> int | None:
    steps = checkpoint_steps(run_dir)
    return steps[-1] if steps else None


def n_components(run_dir: Path) -> int | None:
    """`C` off the saved config, as text -- this module imports neither torch nor the lab."""
    cfg = Path(run_dir) / "experiment_config.yaml"
    if not cfg.exists():
        return None
    found = re.search(r"^\s*-?\s*C:\s*(\d+)\s*$", cfg.read_text(), re.MULTILINE)
    return int(found[1]) if found else None


def _pick(available: dict[int, object], step: int, *, nearest_below: bool):
    """`(step_used, payload)` for one stage, or `(None, None)`."""
    if step in available:
        return step, available[step]
    if not nearest_below:
        return None, None
    below = [s for s in available if s <= step]
    return (max(below), available[max(below)]) if below else (None, None)


# --------------------------------------------------------------------------- stage collectors


def _by_k(by_k: dict, kind: str, suffix: str = "") -> dict[int, dict]:
    """`{k: row}` for one estimator, keyed `<kind>/<k>[/<suffix>]`."""
    out = {}
    for key, row in by_k.items():
        parts = key.split("/")
        if parts[0] == kind and parts[2:] == ([suffix] if suffix else []):
            out[int(parts[1])] = row
    return out


def _localization_over_random(cols: dict, prefix: str, agg: str) -> dict:
    """`<prefix>_localization_over_random` -- localization in units of the `random` control's."""
    top = cols.get(f"{prefix}_localization{agg}")
    bottom = cols.get(f"{prefix}_localization_random{agg}")
    if top is None or not bottom:
        return {}
    return {f"{prefix}_localization_over_random": top / bottom}


def _attr_columns(by_k: dict, prefix: str, field: str, suffix: str = "") -> dict:
    """`<prefix>_<metric>_at_max_k` / `_mean_over_k`, plus `<prefix>_max_k` naming the `k` used."""
    ranked = _by_k(by_k, "ranked", suffix)
    if not ranked:
        return {}
    max_k = max(ranked)
    cols: dict[str, float | None] = {f"{prefix}_max_k": float(max_k)}
    for name, stem in _ATTR_METRICS:
        key = f"{field}_{stem}"
        values = [row[key] for row in ranked.values() if key in row]
        cols[f"{prefix}_{name}_at_max_k"] = ranked[max_k].get(key)
        cols[f"{prefix}_{name}_mean_over_k"] = sum(values) / len(values) if values else None
    key = f"{field}_localization"
    control = [row[key] for row in _by_k(by_k, "random", suffix).values() if key in row]
    cols[f"{prefix}_localization_random_mean_over_k"] = (
        sum(control) / len(control) if control else None
    )
    return cols | _localization_over_random(cols, prefix, "_mean_over_k")


def _sweep_by_step(path: Path) -> dict[int, dict]:
    """`{step: by_k}` out of an attr sweep file."""
    data = read_json(path)
    return {int(k): v["by_k"] for k, v in ((data or {}).get("per_step") or {}).items()
            if "by_k" in v}


def _attr_edit(run_dir: Path, step: int, *, nearest_below: bool):
    by_step = _sweep_by_step(run_dir / "attr_edit" / "sweep_attr_edit.json")
    used, by_k = _pick(by_step, step, nearest_below=nearest_below)
    if by_k is None:
        return used, {}
    return used, _attr_columns(by_k, "attr", "on_aj")


def _attr_edit_multi(run_dir: Path, step: int, *, nearest_below: bool):
    """One column group per `m<m>_<setup>` arm, on the `union` selection."""
    root = run_dir / "attr_edit_multi"
    if not root.is_dir():
        return None, {}
    cols: dict[str, float | None] = {}
    used_steps = set()
    for arm in sorted(p for p in root.iterdir() if p.is_dir()):
        by_step = _sweep_by_step(arm / "sweep_attr_edit_multi.json")
        used, by_k = _pick(by_step, step, nearest_below=nearest_below)
        if by_k is None:
            continue
        used_steps.add(used)
        cols |= _attr_columns(by_k, f"multi_{arm.name}", "on_a", "union")
    cols |= _mean_over_m(cols)
    for setup in sorted({found["setup"] for key in cols if (found := _MULTI_ARM.match(key))}):
        cols |= _localization_over_random(cols, f"multi_{setup}", "_mean_over_k_and_m")
    return (used_steps.pop() if len(used_steps) == 1 else None), cols


_MULTI_ARM = re.compile(r"^multi_m(?P<m>\d+)_(?P<setup>[a-z]+)_(?P<rest>.+)$")


def _mean_over_m(cols: dict) -> dict:
    """`multi_<setup>_<metric>_mean_over_k_and_m` -- the per-`m` means averaged across `m`."""
    grouped: dict[tuple[str, str], list[float]] = {}
    counts: dict[str, set[int]] = {}
    for key, value in cols.items():
        found = _MULTI_ARM.match(key)
        if not found or not found["rest"].endswith("_mean_over_k") or value is None:
            continue
        metric = found["rest"].removesuffix("_mean_over_k")
        grouped.setdefault((found["setup"], metric), []).append(value)
        counts.setdefault(found["setup"], set()).add(int(found["m"]))
    out: dict[str, float | None] = {
        f"multi_{setup}_{metric}_mean_over_k_and_m": sum(values) / len(values)
        for (setup, metric), values in grouped.items()
    }
    return out | {f"multi_{setup}_n_m": float(len(ms)) for setup, ms in counts.items()}


INTRUDER_CI_THRESHOLDS: tuple[float, ...] = (0.01, 0.1)
INTRUDER_THRESHOLDS: tuple[tuple[str, float], ...] = tuple(("ci", t) for t in INTRUDER_CI_THRESHOLDS)


def intruder_prefix(ci_threshold: float | None, criterion: str = "ci") -> str:
    """Column prefix for one threshold. `None` is the existing, unthresholded measurement."""
    return "intruder" if ci_threshold is None else f"intruder_{criterion}{ci_threshold:g}"


def _intruder_at(run_dir: Path, step: int, *, nearest_below: bool, ci_threshold: float | None,
                 criterion: str = "ci"):
    prefix = intruder_prefix(ci_threshold, criterion)
    stem = ("intruder_summary.json" if ci_threshold is None
            else f"intruder_summary_{criterion}{ci_threshold:g}.json")
    summaries = {}
    for path in (run_dir / "harvest").glob(f"h-step*/{stem}"):
        found = _H_STEP.match(path.parent.name)
        if found:
            summaries[int(found["step"])] = path
    used, path = _pick(summaries, step, nearest_below=nearest_below)
    if path is None:
        return used, {}
    data = read_json(path)
    return used, {f"{prefix}_mean": data.get("mean"), f"{prefix}_std": data.get("std"),
                  f"{prefix}_n_scored": data.get("n_scored")}


def _intruder(run_dir: Path, step: int, *, nearest_below: bool):
    return _intruder_at(run_dir, step, nearest_below=nearest_below, ci_threshold=None)


def _intruder_ci(ci_threshold: float, criterion: str = "ci"):
    """A collector for one threshold, bound at module scope so `COLUMNS` can list them all."""
    def collect(run_dir: Path, step: int, *, nearest_below: bool):
        return _intruder_at(run_dir, step, nearest_below=nearest_below,
                            ci_threshold=ci_threshold, criterion=criterion)

    collect.__name__ = f"_intruder_{criterion}{ci_threshold:g}"
    return collect


def _matching(run_dir: Path, step: int, *, nearest_below: bool):
    """The judged matching of ONE mode -- `c2o` when it is there, else whatever ran."""
    reports: dict[str, dict[int, Path]] = {m: {} for m in MODES}
    for path in (run_dir / "matching").glob("matching_*step*.json"):
        match = REPORT_RE.match(path.name)
        if not match:
            continue
        found = match["mode"] or mode_of((read_json(path) or {}).get("meta", {}))
        reports[found][int(match["step"])] = path
    mode = next((m for m in (DEFAULT_MODE, *MODES) if reports[m]), None)
    if mode is None:
        return None, {}
    used, path = _pick(reports[mode], step, nearest_below=nearest_below)
    if path is None:
        return used, {}
    data = read_json(path)
    scores = {r["name"]: r["mean_score"] for r in data["results"]}
    cols = {"matching_mode": mode}
    cols |= {f"matching_{name}": scores.get(name) for name in ("A_cond", "A_glob", "random")}
    control = scores.get("random")
    for name in ("A_cond", "A_glob"):
        score = scores.get(name)
        cols[f"matching_{name}_margin"] = (
            score - control if score is not None and control is not None else None)
    return used, cols


def _diversity_at(run_dir: Path, step: int, *, nearest_below: bool, suffix: str, prefix: str):
    found = {}
    for path in (run_dir / "diversity").glob(f"diversity_h-step*{suffix}.json"):
        m = re.match(rf"^diversity_h-step(\d+){re.escape(suffix)}\.json$", path.name)
        if m:
            found[int(m[1])] = path
    used, path = _pick(found, step, nearest_below=nearest_below)
    if path is None:
        return used, {}
    data = read_json(path)
    return used, {prefix: data["z_bar"], f"{prefix}_ci95": data["z_bar_ci95"],
                  f"{prefix}_n_sampled": data["n_sampled"]}


def _diversity(ci_threshold: float | None):
    """Sim (mean pairwise token-set overlap), at g > 0 or at g > `ci_threshold`."""
    suffix = "" if ci_threshold is None else f"_ci{ci_threshold:g}"
    prefix = "sim" if ci_threshold is None else f"sim_ci{ci_threshold:g}"

    def collect(run_dir: Path, step: int, *, nearest_below: bool):
        return _diversity_at(run_dir, step, nearest_below=nearest_below, suffix=suffix, prefix=prefix)

    collect.__name__ = f"_diversity{suffix}"
    return collect


Collector = Callable[..., tuple[int | None, dict]]
COLUMNS: tuple[tuple[str, Collector, tuple[str, ...]], ...] = (
    ("attr_edit", _attr_edit,
     ("attr_max_k", *(f"attr_{name}_{agg}" for name, _ in _ATTR_METRICS
                      for agg in ("at_max_k", "mean_over_k")),
      "attr_localization_random_mean_over_k", "attr_localization_over_random")),
    ("attr_edit_multi", _attr_edit_multi, ()),
    ("intruder", _intruder, ("intruder_mean", "intruder_std", "intruder_n_scored")),
    *(
        (
            intruder_prefix(tau, crit),
            _intruder_ci(tau, crit),
            tuple(f"{intruder_prefix(tau, crit)}_{field}" for field in ("mean", "std", "n_scored")),
        )
        for crit, tau in INTRUDER_THRESHOLDS
    ),
    ("diversity", _diversity(None), ("sim", "sim_ci95", "sim_n_sampled")),
    *(
        (f"diversity_ci{tau:g}", _diversity(tau), (f"sim_ci{tau:g}", f"sim_ci{tau:g}_ci95"))
        for tau in INTRUDER_CI_THRESHOLDS
    ),
    ("matching", _matching, ("matching_mode", "matching_A_cond", "matching_A_glob",
                             "matching_random", "matching_A_cond_margin",
                             "matching_A_glob_margin")),
)


# --------------------------------------------------------------------------- assembly


def collect_row(run_dir: Path, label: str, *, nearest_below: bool) -> dict:
    """One arm's row, plus a `_blanked` list naming the stages that contributed nothing."""
    run_dir = Path(run_dir)
    step = last_step(run_dir)
    assert step is not None, f"{run_dir} has no model_*.pth -- nothing to report a last step for"
    row: dict = {"arm": label, "run": run_dir.name, "C": n_components(run_dir), "step": step}
    blanked = []
    for stage, collect, _ in COLUMNS:
        used, cols = collect(run_dir, step, nearest_below=nearest_below)
        row[f"{stage}_step"] = used
        row |= cols
        if not cols:
            blanked.append(stage)
    row["_blanked"] = blanked
    return row


def stage_steps_on_disk(run_dir: Path) -> dict[str, int | None]:
    """Each stage's newest step, ignoring the checkpoint rule -- what a blanked cell DOES have."""
    run_dir = Path(run_dir)
    out = {}
    for stage, collect, _ in COLUMNS:
        used, _ = collect(run_dir, 1 << 62, nearest_below=True)
        out[stage] = used
    return out


def column_order(rows: list[dict]) -> list[str]:
    """Declared columns in `COLUMNS` order, then whatever the runs discovered, sorted."""
    order = ["arm", "run", "C", "step"]
    for stage, _, names in COLUMNS:
        order += [f"{stage}_step", *names]
    seen = set(order) | {"_blanked"}
    return order + sorted({k for row in rows for k in row if k not in seen})


_COMPACT = (
    "arm", "C", "step",
    # Interpretability and diversity
    "intruder_mean",
    *(f"intruder_ci{tau:g}_mean" for tau in INTRUDER_CI_THRESHOLDS),
    "sim",
    *(f"sim_ci{tau:g}" for tau in INTRUDER_CI_THRESHOLDS),
    # Meaning localization
    "matching_A_cond_margin",
    # Weight editing: single and multiple target features
    "attr_localization_over_random", "attr_localization_mean_over_k",
    "multi_cond_localization_over_random", "multi_cond_localization_mean_over_k_and_m",
)


def largest_multi(rows: list[dict]) -> str | None:
    """`multi_m<m>` for the largest `m` any run swept, or None if none did."""
    found = {int(m[1]) for row in rows for k in row
             if (m := re.match(r"^multi_m(\d+)_", k))}
    return f"multi_m{max(found)}" if found else None


def multi_m_swept(rows: list[dict]) -> list[int]:
    """Every `m` any run swept, ascending -- what the `_and_m` columns averaged over."""
    return sorted({int(m[1]) for row in rows for k in row
                   if (m := re.match(r"^multi_m(\d+)_", k))})


def compact_order(rows: list[dict]) -> list[str]:
    """`_COMPACT`, minus any column no run produced."""
    present = {k for row in rows for k in row}
    return [n for n in _COMPACT if n in ("arm", "C", "step") or n in present]


_STEP_OF = {"intruder": "intruder_step", "sim": "diversity_step", "matching": "matching_step",
            "attr": "attr_edit_step", "multi": "attr_edit_multi_step"}


def full_order(rows: list[dict]) -> list[str]:
    """The paper's metrics, each with its 95% CI, the CI's sample count and the checkpoint step it
    was computed at; minus any column no run produced."""
    present = {k for row in rows for k in row}
    order = ["arm", "run", "C", "step"]
    for name in _COMPACT[3:]:
        stage = next(s for p, s in _STEP_OF.items() if name.startswith(p))
        tau = re.search(r"_ci(0\.\d+)", name)
        if tau:
            stage = stage.replace("_step", f"_ci{tau[1]}_step")
        for col in (name, f"{name}_ci95", f"{name}_ci95_n", stage):
            if col in present and col not in order:
                order.append(col)
    for col in ("sim_n_sampled", "intruder_n_scored"):
        if col in present:
            order.append(col)
    return order


def build_rows(run_dirs: list[Path], labels: list[str], *, nearest_below: bool = False
               ) -> list[dict]:
    return [collect_row(d, label, nearest_below=nearest_below)
            for d, label in zip(run_dirs, labels, strict=True)]


def render_table(rows: list[dict], order: list[str], *, sig: int = 4) -> str:
    """The compact table as aligned text, for reading in a terminal rather than a spreadsheet."""
    def cell(value) -> str:
        if value is None or value == "":
            return "-"
        if isinstance(value, float):
            return f"{value:.{sig}g}"
        return str(value)

    table = [order] + [[cell(row.get(k)) for k in order] for row in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(order))]
    return "\n".join("  ".join(c.rjust(w) for c, w in zip(line, widths, strict=True)).rstrip()
                     for line in table)


def write_summary(rows: list[dict], out_path: Path, order: list[str] | None = None) -> Path:
    order = column_order(rows) if order is None else order
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=order, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in order})
    return out_path


def blanked_report(run_dirs: list[Path], rows: list[dict]) -> list[str]:
    """One line per arm that lost a stage, naming the step that stage does have on disk."""
    lines = []
    for run_dir, row in zip(run_dirs, rows, strict=True):
        if not row["_blanked"]:
            continue
        have = stage_steps_on_disk(run_dir)
        detail = ", ".join(
            f"{stage} (newest on disk: {have[stage] if have[stage] is not None else 'none'})"
            for stage in row["_blanked"]
        )
        lines.append(f"  {row['arm']:<34} step={row['step']:<8} blanked: {detail}")
    return lines


def run_arm(run_dir: Path) -> str:
    """The paper arm a run directory trained, derived from its own `experiment_config.yaml`."""
    from aspd.arms import derive_arm_name
    from aspd.config import LMInterpExperimentConfig

    return derive_arm_name(LMInterpExperimentConfig.from_file(run_dir / "experiment_config.yaml"))
