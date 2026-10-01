"""95% confidence intervals for the table columns, from the per-item records behind each mean."""

import math
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from scipy.stats import t as student_t

from aspd.eval.matching import DEFAULT_MODE, MODES, report_name
from aspd.eval.tables import (
    _ATTR_METRICS,
    _H_STEP,
    INTRUDER_THRESHOLDS,
    intruder_prefix,
    read_json,
)

CI_LEVEL = 0.95

NO_CI_REASON: dict[str, str] = {
    "intruder_std": "not a mean -- the spread ACROSS components, which has no interval",
    **{
        f"{intruder_prefix(tau, crit)}_{field}": reason
        for crit, tau in INTRUDER_THRESHOLDS
        for field, reason in (
            ("std", "not a mean -- the spread ACROSS components, which has no interval"),
            ("n_scored", "a count of components, not a mean over them"),
        )
    },
    "intruder_n_scored": "a count of components, not a mean over them",
}

COVERAGE_APPROX: dict[str, str] = {
    f"multi_{setup}_localization{tail}": "the control covers fewer combinations at large m"
    for setup in ("cond", "global")
    for tail in ("_random_mean_over_k_and_m", "_over_random")
}
COVERAGE_TOL = 0.25


def ci_half_width(values: list[float]) -> float | None:
    """Half-width of the two-sided 95% CI of the MEAN, or None below two units."""
    n = len(values)
    if n < 2:
        return None
    return float(student_t.ppf(0.5 + CI_LEVEL / 2, n - 1)) * statistics.stdev(values) / math.sqrt(n)


def _mean_over(rows: dict[object, list[float]]) -> list[float]:
    return [statistics.fmean(v) for v in rows.values()]


def _ratio_samples(top: dict[object, list[float]],
                   bottom: dict[object, list[float]]) -> list[float]:
    """Per-unit values whose mean is EXACTLY `mean(top) / mean(bottom)`, with the ratio's spread."""
    units = [u for u in top if u in bottom]
    if not units:
        return []
    x = [statistics.fmean(top[u]) for u in units]
    y = [statistics.fmean(bottom[u]) for u in units]
    y_bar = statistics.fmean(y)
    if not y_bar:
        return []
    ratio = statistics.fmean(x) / y_bar
    return [ratio + (xi - ratio * yi) / y_bar for xi, yi in zip(x, y, strict=True)]


def _per_unit(edits: list[dict], kind: str, field: str, stem: str, unit: str,
              block: str | None = None) -> dict[object, list[float]]:
    """`{unit: [every matching edit's value]}` -- the rows one unit's mean is taken over."""
    out: dict[object, list[float]] = defaultdict(list)
    for edit in edits:
        if edit["kind"] == kind and (block is None or edit["block"] == block):
            out[edit[unit]].append(edit[field][stem])
    return out


# --------------------------------------------------------------------------- populations


def attr_samples(run_dir: Path, step: int) -> dict[str, list[float]]:
    """One value per seed feature: its `ranked` edit averaged over the k sweep."""
    data = read_json(Path(run_dir) / "attr_edit" / f"attr_edit_step{step}.json")
    if data is None:
        return {}
    if not any(e["kind"] == "ranked" for e in data["edits"]):
        return {}
    out = {}
    ranked_localization: dict[object, list[float]] = {}
    for name, stem in _ATTR_METRICS:
        by_feature = _per_unit(data["edits"], "ranked", "on_aj", stem, "feature_id")
        counts = {len(v) for v in by_feature.values()}
        assert len(counts) == 1, f"unbalanced k sweep in {run_dir.name}/attr_edit: {counts}"
        out[f"attr_{name}_mean_over_k"] = _mean_over(by_feature)
        if name == "localization":
            ranked_localization = by_feature
    control = _per_unit(data["edits"], "random", "on_aj", "localization", "feature_id")
    if control:
        counts = {len(v) for v in control.values()}
        assert len(counts) == 1, (
            f"unbalanced random control in {run_dir.name}/attr_edit: {counts}"
        )
        out["attr_localization_random_mean_over_k"] = _mean_over(control)
        out["attr_localization_over_random"] = _ratio_samples(ranked_localization, control)
    return out


def attr_multi_samples(run_dir: Path, step: int) -> dict[str, list[float]]:
    """One value per combination: its `union` edit averaged over k, then over m."""
    root = Path(run_dir) / "attr_edit_multi"
    if not root.is_dir():
        return {}
    out: dict[str, list[float]] = {}
    for setup in ("cond", "global"):
        # {metric: {combo: [one mean-over-k per m]}}
        per_combo: dict[str, dict[int, list[float]]] = {
            name: defaultdict(list) for name, _ in _ATTR_METRICS
        }
        control: dict[object, list[float]] = defaultdict(list)
        found = False
        for arm in sorted(root.glob(f"m*_{setup}")):
            data = read_json(arm / f"attr_edit_multi_step{step}.json")
            if data is None:
                continue
            found = True
            for name, stem in _ATTR_METRICS:
                for combo, values in _per_unit(data["edits"], "ranked", "on_a", stem, "combo",
                                               block="union").items():
                    per_combo[name][combo].append(statistics.fmean(values))
            for combo, values in _per_unit(data["edits"], "random", "on_a", "localization",
                                           "combo", block="union").items():
                control[combo].append(statistics.fmean(values))
            del data  # ~35 MB per file on Gemma
        if not found:
            continue
        for name, _ in _ATTR_METRICS:
            out[f"multi_{setup}_{name}_mean_over_k_and_m"] = _mean_over(per_combo[name])
        if control:
            out[f"multi_{setup}_localization_random_mean_over_k_and_m"] = _mean_over(control)
            out[f"multi_{setup}_localization_over_random"] = _ratio_samples(
                per_combo["localization"], control)
    return out


def intruder_samples(run_dir: Path, step: int) -> dict[str, list[float]]:
    """One value per scored component."""
    return _intruder_samples_at(run_dir, step, ci_threshold=None)


def _intruder_samples_at(run_dir: Path, step: int, *, ci_threshold: float | None,
                         criterion: str = "ci"):
    stem = ("intruder_summary.json" if ci_threshold is None
            else f"intruder_summary_{criterion}{ci_threshold:g}.json")
    found = [p for p in (Path(run_dir) / "harvest").glob(f"h-step*/{stem}")
             if (m := _H_STEP.match(p.parent.name)) and int(m["step"]) == step]
    data = read_json(found[0]) if found else None
    if data is None or not data.get("scores"):
        return {}
    return {f"{intruder_prefix(ci_threshold, criterion)}_mean":
            [float(v) for v in data["scores"].values()]}


def intruder_ci_samples(ci_threshold: float, criterion: str = "ci"):
    """`intruder_samples` for one firing threshold -- the components still scored at it."""
    def sample(run_dir: Path, step: int) -> dict[str, list[float]]:
        return _intruder_samples_at(run_dir, step, ci_threshold=ci_threshold, criterion=criterion)

    sample.__name__ = f"intruder_{criterion}{ci_threshold:g}_samples"
    return sample


def matching_samples(run_dir: Path, step: int) -> dict[str, list[float]]:
    """One value per judged pair, for both scored arms, the `random` control, and the margins."""
    matching = Path(run_dir) / "matching"
    data = next(
        (d for d in (read_json(matching / report_name(m, step)) for m in (DEFAULT_MODE, *MODES))
         if d is not None),
        None,
    ) or read_json(matching / f"matching_step{step}.json")
    if data is None:
        return {}
    out = {f"matching_{r['name']}": [float(s) for s in r["scores"]]
           for r in data["results"] if r.get("scores")}
    components = {
        r["name"]: list(r["components"]) if r.get("components") else [p[0] for p in r["pairs"]]
        for r in data["results"] if r.get("scores") and (r.get("components") or r.get("pairs"))
    }
    control = out.get("matching_random")
    if control is None:
        return out
    for name in ("A_cond", "A_glob"):
        scored = out.get(f"matching_{name}")
        if scored is None:
            continue
        if name in components and "random" in components:
            assert components[name] == components["random"], (
                f"{run_dir.name}/matching step {step}: {name} and random judge different "
                "components, so their difference is not paired"
            )
        out[f"matching_{name}_margin"] = [
            s - c for s, c in zip(scored, control, strict=True)
        ]
    return out


Sampler = Callable[[Path, int], dict[str, list[float]]]
SAMPLERS: tuple[tuple[str, Sampler], ...] = (
    ("attr_edit_step", attr_samples),
    ("attr_edit_multi_step", attr_multi_samples),
    ("intruder_step", intruder_samples),
    *(
        (f"{intruder_prefix(tau, crit)}_step", intruder_ci_samples(tau, crit))
        for crit, tau in INTRUDER_THRESHOLDS
    ),
    ("matching_step", matching_samples),
)


# --------------------------------------------------------------------------- assembly


def attach_ci(run_dir: Path, row: dict, *, tol: float = 1e-9) -> dict:
    """Add `<column>_ci95` and `<column>_n` to `row`, in place; return the reconstructed means."""
    reconstructed = {}
    for step_column, sample in SAMPLERS:
        step = row.get(step_column)
        if step is None:
            continue
        for column, values in sample(Path(run_dir), int(step)).items():
            if not values:
                continue
            mean = statistics.fmean(values)
            reported = row.get(column)
            if column in COVERAGE_APPROX:
                agrees = reported is None or abs(mean - reported) <= COVERAGE_TOL * abs(reported)
            else:
                agrees = reported is None or abs(mean - reported) <= tol * max(1.0, abs(reported))
            assert agrees, (
                f"{row['arm']}: {column} CI population means {mean!r}, table says {reported!r}"
            )
            reconstructed[column] = mean
            row[f"{column}_ci95"] = ci_half_width(values)
            row[f"{column}_ci95_n"] = float(len(values))
    return reconstructed


def cell(value: float | None, half: float | None, *, sig: int = 6, ci_sig: int = 3) -> str:
    """`mean` alone, or `mean ± half`. Empty for a blank cell, so a gap stays a gap."""
    if value is None or value == "":
        return ""
    if not isinstance(value, float):
        return str(value)
    if half is None:
        return f"{value:.{sig}g}"
    return f"{value:.{sig}g} ± {half:.{ci_sig}g}"


def with_ci_cells(row: dict, order: list[str]) -> dict:
    out = dict(row)
    for column in order:
        if column in ("arm", "run", "C", "step"):
            continue
        out[column] = cell(row.get(column), row.get(f"{column}_ci95"))
    return out


def ci_report(rows: list[dict], order: list[str]) -> list[str]:
    """One line per column: how many arms carry an interval, and over how many units."""
    lines = []
    for column in order:
        if column in ("arm", "run", "C", "step"):
            continue
        ns = [int(r[f"{column}_ci95_n"]) for r in rows if r.get(f"{column}_ci95_n") is not None]
        if not ns:
            reason = NO_CI_REASON.get(column, "no per-item record on disk")
            lines.append(f"  {column:<44} no CI -- {reason}")
            continue
        span = str(ns[0]) if len(set(ns)) == 1 else f"{min(ns)}-{max(ns)}"
        note = f" -- approximate: {COVERAGE_APPROX[column]}" if column in COVERAGE_APPROX else ""
        lines.append(f"  {column:<44} n = {span:<8} on {len(ns)}/{len(rows)} arm(s){note}")
    return lines
