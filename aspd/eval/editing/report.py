"""The editing report schema written beside every result."""

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

HEADLINE = ("delta_abs", "delta_relative", "collateral_abs", "localization")


@dataclass
class EditRow:
    """One measured intervention against one target feature."""

    kind: str
    k: int
    rep: int
    feature_id: int
    selection: list[int]
    edit_norm: float
    edit_norm_rel: float
    predicted_delta_unit: float
    predicted_delta_gate: float
    score_share: float
    """Fraction of `sum_c |a_gate[j, c]|` captured by this selection."""
    on_aj: dict[str, float]
    on_global: dict[str, float]


@dataclass
class FeatureRow:
    """Per-feature facts that do not depend on the edit."""

    feature_id: int
    density: float
    support: int
    baseline_act: float
    top_components: list[int]
    """The top-50 ranking, so a reader can see WHICH components a feature selected."""
    top_scores: list[float]
    rank_overlap_gate_vs_unit: dict[str, float]


@dataclass
class AttrEditReport:
    site: str
    run_dir: str
    sae_dir: str
    step: int | None
    module: str
    n_components: int
    n_latents: int
    estimator: str
    """`analytic` or `autograd` -- which path produced the attribution table."""
    n_tokens: int
    n_tokens_global: int
    features: list[FeatureRow]
    edits: list[EditRow]
    by_k: dict[str, dict[str, float]]
    """`{kind}/{k}` -> mean over features of every metric, both token sets flattened with an
    `on_aj_` / `on_global_` prefix.
    """
    faithfulness: dict[str, float]
    """Spearman of predicted against measured `delta_signed`, per k and pooled."""
    selection_overlap: dict[str, float]
    """Mean pairwise Jaccard of the ranked selections across features, per k."""
    edit_skipped: str | None = None
    """Why the edit sweep did not run, or `None` if it did. Set on any arm whose attribution
    table is defined but for which `W - sum_S U_c (x) V_c` is not that arm's component.
    """
    meta: dict[str, object] = field(default_factory=dict)


@dataclass
class MultiEditRow:

    kind: str
    k: int
    rep: int
    combo: int
    m: int
    block: str
    feature_id: int | None
    selection_size: int
    """`|S(J,k)|`. Below `m*k` exactly when the combination's features share handles."""
    sharing_rate: float
    """`1 - |S|/(m*k)` -- 0 when every target picked disjoint components."""
    edit_norm: float
    edit_norm_rel: float
    predicted_delta_unit: float
    predicted_delta_gate: float
    score_share: float
    on_a: dict[str, float]
    """`A_J` on a union row, `A_j` on a target row."""
    on_global: dict[str, float]
    selection: list[int] = field(default_factory=list)
    """Recorded on union rows only -- the target rows of one edit share it."""


@dataclass
class MultiAttrEditReport:
    site: str
    run_dir: str
    sae_dir: str
    step: int | None
    module: str
    m: int
    setup: str
    n_combinations: int
    n_control_combos: int
    n_components: int
    n_latents: int
    estimator: str
    n_tokens: int
    n_tokens_global: int
    combinations: list[list[int]]
    edits: list[MultiEditRow]
    by_k: dict[str, dict[str, float]]
    """`{kind}/{k}/{block}` -> mean over combinations of every metric."""
    faithfulness: dict[str, float]
    structure: dict[str, dict[str, float]]
    edit_skipped: str | None = None
    global_skipped: str | None = None
    meta: dict[str, object] = field(default_factory=dict)


def write_multi_report(report: MultiAttrEditReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prior = json.loads(path.read_text())
        assert (prior["m"], prior["setup"]) == (report.m, report.setup), (
            f"{path} holds the m={prior['m']} {prior['setup']} arm and this is m={report.m} "
            f"{report.setup} -- pass --out-dir per arm, or drop it for the default layout"
        )
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
    print(f"[attr_edit_multi] wrote {path}", flush=True)
    return path


def write_multi_rows_csv(report: MultiAttrEditReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "m", "setup", "kind", "k", "rep", "combo", "block", "feature_id",
              "selection_size", "sharing_rate", "edit_norm", "edit_norm_rel",
              "predicted_delta_unit", "predicted_delta_gate", "score_share"]
    metric_names = sorted({m for row in report.edits for m in row.on_a})
    header = fields + [f"on_a_{m}" for m in metric_names] + \
        [f"on_global_{m}" for m in metric_names]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in report.edits:
            writer.writerow(
                [report.step, row.m, report.setup, row.kind, row.k, row.rep, row.combo,
                 row.block, "" if row.feature_id is None else row.feature_id,
                 row.selection_size, row.sharing_rate, row.edit_norm, row.edit_norm_rel,
                 row.predicted_delta_unit, row.predicted_delta_gate, row.score_share]
                + [row.on_a.get(m, "") for m in metric_names]
                + [row.on_global.get(m, "") for m in metric_names]
            )
    print(f"[attr_edit_multi] wrote {path}", flush=True)
    return path


def summarize_multi(report: MultiAttrEditReport) -> str:
    lines = [
        f"[attr_edit_multi] step {report.step} · m={report.m} · setup={report.setup} · "
        f"{report.n_combinations} combinations · {len(report.edits)} rows · "
        f"estimator {report.estimator}"
    ]
    if report.global_skipped:
        lines.append(f"  NO GLOBAL SETUP — {report.global_skipped}")
    if report.edit_skipped:
        lines.append(f"  NO EDIT SWEEP — {report.edit_skipped}")
        return "\n".join(lines)
    lines.append(
        f"{'kind':<13}{'k':>4}{'|S|':>7}{'share':>7}{'|Df| A_J':>12}{'rel':>9}"
        f"{'collat':>13}{'local':>8}{'||dW||rel':>11}"
    )
    union = {key: row for key, row in report.by_k.items() if key.endswith("/union")}
    for key in sorted(union, key=lambda s: (s.split("/")[0], int(s.split("/")[1]))):
        row, (kind, k, _) = union[key], key.split("/")
        lines.append(
            f"{kind:<13}{k:>4}{row.get('selection_size', float('nan')):>7.1f}"
            f"{row.get('sharing_rate', float('nan')):>7.2f}"
            f"{row.get('on_a_delta_abs', float('nan')):>12.4g}"
            f"{row.get('on_a_delta_relative', float('nan')):>9.3f}"
            f"{row.get('on_a_collateral_abs', float('nan')):>13.5g}"
            f"{row.get('on_a_localization', float('nan')):>8.3f}"
            f"{row.get('edit_norm_rel', float('nan')):>11.5f}"
        )
    return "\n".join(lines)


def write_multi_sweep(reports: list[MultiAttrEditReport], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {
            "site": reports[0].site if reports else None,
            "run_dir": reports[0].run_dir if reports else None,
            "sae_dir": reports[0].sae_dir if reports else None,
            "m": reports[0].m if reports else None,
            "setup": reports[0].setup if reports else None,
            "steps": [r.step for r in reports],
            "per_step": {
                str(r.step): {
                    "by_k": r.by_k,
                    "faithfulness": r.faithfulness,
                    "structure": r.structure,
                    "estimator": r.estimator,
                    "edit_skipped": r.edit_skipped,
                    "global_skipped": r.global_skipped,
                }
                for r in reports
            },
        },
        indent=2, sort_keys=True,
    ))
    print(f"[attr_edit_multi] wrote {path}", flush=True)
    return path


def write_report(report: AttrEditReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
    print(f"[attr_edit] wrote {path}", flush=True)
    return path


def write_rows_csv(report: AttrEditReport, path: Path) -> Path:
    """The edit rows as one flat table -- a PNG cannot be grepped and the JSON nests two levels."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "kind", "k", "rep", "feature_id", "edit_norm", "edit_norm_rel",
              "predicted_delta_unit", "predicted_delta_gate", "score_share"]
    metric_names = sorted({m for row in report.edits for m in row.on_aj})
    header = fields + [f"on_aj_{m}" for m in metric_names] + [f"on_global_{m}" for m in metric_names]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in report.edits:
            writer.writerow(
                [report.step, row.kind, row.k, row.rep, row.feature_id, row.edit_norm,
                 row.edit_norm_rel, row.predicted_delta_unit, row.predicted_delta_gate,
                 row.score_share]
                + [row.on_aj.get(m, "") for m in metric_names]
                + [row.on_global.get(m, "") for m in metric_names]
            )
    print(f"[attr_edit] wrote {path}", flush=True)
    return path


def write_sweep(reports: list[AttrEditReport], path: Path) -> Path:
    """One file carrying every checkpoint's `by_k`, which is what the figures read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {
            "site": reports[0].site if reports else None,
            "run_dir": reports[0].run_dir if reports else None,
            "sae_dir": reports[0].sae_dir if reports else None,
            "steps": [r.step for r in reports],
            "per_step": {
                str(r.step): {
                    "by_k": r.by_k,
                    "faithfulness": r.faithfulness,
                    "selection_overlap": r.selection_overlap,
                    "estimator": r.estimator,
                    "edit_skipped": r.edit_skipped,
                }
                for r in reports
            },
        },
        indent=2, sort_keys=True,
    ))
    print(f"[attr_edit] wrote {path}", flush=True)
    return path


def summarize(report: AttrEditReport) -> str:
    lines = [
        f"[attr_edit] step {report.step} · {len(report.features)} features · "
        f"{len(report.edits)} edits · estimator {report.estimator}",
    ]
    if report.edit_skipped:
        lines.append(f"  NO EDIT SWEEP — {report.edit_skipped}")
        return "\n".join(lines)
    lines.append(
        f"{'kind':<13}{'k':>4}{'|Df| A_j':>12}{'rel':>9}{'collat':>13}{'local':>8}{'||dW||rel':>11}"
    )
    for key in sorted(report.by_k, key=lambda s: (s.split("/")[0], int(s.split("/")[1]))):
        row = report.by_k[key]
        kind, k = key.split("/")
        lines.append(
            f"{kind:<13}{k:>4}{row.get('on_aj_delta_abs', float('nan')):>12.4g}"
            f"{row.get('on_aj_delta_relative', float('nan')):>9.3f}"
            f"{row.get('on_aj_collateral_abs', float('nan')):>13.5g}"
            f"{row.get('on_aj_localization', float('nan')):>8.3f}"
            f"{row.get('edit_norm_rel', float('nan')):>11.5f}"
        )
    return "\n".join(lines)
