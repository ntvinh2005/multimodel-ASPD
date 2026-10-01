"""Helpers that turn result dictionaries into plot series."""

import re
from collections.abc import Iterable, Mapping

_THRESHOLD_RE = re.compile(r"^(?P<family>.+?)_threshold_(?P<k>\d+)(?:_(?P<suffix>.+))?$")
_TOPK_RE = re.compile(r"^(?P<family>.+?)_top_(?P<k>\d+)_(?P<suffix>.+)$")

_CONTROL_TOKENS = ("random", "rounded", "stoch", "zero", "control", "shuffled")


PROVENANCE = frozenset({
    "reference_fp32", "n_tokens", "n_tokens_unmasked", "n_batches", "n_runs", "seed", "index",
})


def numeric(d: Mapping[str, object], *, keep_provenance: bool = False) -> dict[str, float]:
    """The plottable entries of a metric dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, bool) or v is None:
            continue
        if not keep_provenance and k in PROVENANCE:
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def is_control(name: str) -> bool:
    return any(t in name.split("_") for t in _CONTROL_TOKENS)


def family(name: str) -> str:
    """The unit a metric is measured in, which is what may share an axis with what."""
    for prefix in ("ce_difference", "ce_unrecovered", "kl", "ce"):
        if name == prefix or name.startswith(prefix + "_"):
            return prefix
    return name.split("_")[0]


def panels_by_family(names: Iterable[str]) -> dict[tuple[str, bool], list[str]]:
    """`{(family, is_control): [metric, ...]}` -- one panel key per (unit, role) pair."""
    groups: dict[tuple[str, bool], list[str]] = {}
    for name in names:
        groups.setdefault((family(name), is_control(name)), []).append(name)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def sweep_axis(names: Iterable[str]) -> dict[str, dict[str, list[tuple[int, str]]]]:
    """`{panel: {series: [(k, metric_name), ...]}}` for metrics whose NAME carries its own x axis."""
    out: dict[str, dict[str, list[tuple[int, str]]]] = {}
    for name in names:
        match = _THRESHOLD_RE.match(name) or _TOPK_RE.match(name)
        if not match:
            continue
        parts = match.groupdict()
        suffix = parts.get("suffix")
        panel, series = (suffix, parts["family"]) if suffix else (parts["family"], "")
        out.setdefault(panel, {}).setdefault(series, []).append((int(parts["k"]), name))
    return {
        panel: {s: sorted(v) for s, v in sorted(subs.items())}
        for panel, subs in sorted(out.items())
    }


def sweep_references(names: Iterable[str],
                     panels: dict[str, dict[str, list[tuple[int, str]]]]) -> dict[tuple[str, str], str]:
    """`{(panel, series): metric}` for the unrestricted version of a swept metric."""
    names = set(names)
    out = {}
    for panel, subs in panels.items():
        for series in subs:
            candidate = f"{series}_{panel}" if series else panel
            if candidate in names:
                out[(panel, series)] = candidate
    return out


def sweep_names(panels: dict[str, dict[str, list[tuple[int, str]]]]) -> list[str]:
    """Every metric name routed into a `sweep_axis` result -- for `assert_all_covered`."""
    return [n for subs in panels.values() for entries in subs.values() for _, n in entries]


def assert_all_covered(available: Iterable[str], drawn: Iterable[str], what: str) -> None:
    """Every numeric metric must appear in some panel."""
    missing = sorted(set(available) - set(drawn))
    assert not missing, (
        f"{what}: {len(missing)} metric(s) were not routed to any panel -- {missing[:12]}"
        f"{' …' if len(missing) > 12 else ''}. Extend the grouping rules in plots/metrics.py "
        "rather than dropping them"
    )
