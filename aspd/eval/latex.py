"""The paper's result tables as LaTeX, rendered from the per-model table CSVs of `aspd.cli.tables`.

Input: for each model, the `*_full.csv` written by `aspd.cli.tables --ci` (raw values and
`<column>_ci95` half-widths). Output: `interp_sim.tex`, `matching.tex` and `editing.tex`, one row
per method and one column group per model, every cell `mean \\pm 95% half-width`.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

ARM_NAMES = {
    "aspd": "ASPD (Ours)",
    "pdtc": "PD Transcoder (Ours)",
    "pdtc_param": "PD Transcoder + param",
    "pdtc_ablate": "PD Transcoder + ablate",
    "vpd": "VPD",
    "vpd_adaptive": "VPD with adaptive $L_0$",
    "vpd_internal": "VPD + internal",
    "vpd_internal_noparam": "VPD + internal + no param",
    "vpd_internal_noablate": "VPD + internal + no ablate",
}
ARM_ORDER = tuple(ARM_NAMES)
TAUS = (0.01, 0.1)


@dataclass(frozen=True)
class Metric:
    header: str
    column: str
    decimals: int


INTERP = Metric(r"Interp $\uparrow$", "intruder_mean", 2)
SIM = Metric(r"Sim $\downarrow$", "sim", 2)
MATCHING = Metric(r"Matching $\uparrow$", "matching_A_cond_margin", 2)
SINGLE = (
    Metric(r"ratio $\uparrow$", "attr_localization_over_random", 1),
    Metric(r"localization $\uparrow$", "attr_localization_mean_over_k", 3),
)
MULTIPLE = (
    Metric(r"ratio $\uparrow$", "multi_cond_localization_over_random", 1),
    Metric(r"localization $\uparrow$", "multi_cond_localization_mean_over_k_and_m", 3),
)


def load(path: Path) -> dict[str, dict[str, str]]:
    """`{arm: row}` from one model's full table CSV."""
    with Path(path).open() as fh:
        return {row["arm"]: row for row in csv.DictReader(fh)}


def cell(row: dict[str, str] | None, metric: Metric, column: str | None = None) -> str:
    column = column or metric.column
    if row is None or not row.get(column):
        return "--"
    value = float(row[column])
    half = row.get(f"{column}_ci95")
    text = f"{value:.{metric.decimals}f}"
    return f"{text} $\\pm$ {float(half):.{metric.decimals}f}" if half else text


def _arms(models: dict[str, dict[str, dict[str, str]]], arms: list[str] | None) -> list[str]:
    present = {a for rows in models.values() for a in rows}
    return [a for a in (arms or ARM_ORDER) if a in present]


def _table(header_rows: list[str], body: list[str], n_cols: int, caption: str) -> str:
    return "\n".join([
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
        r"\begin{tabular}{l" + "c" * (n_cols - 1) + "}", r"\toprule",
        *header_rows, r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def _model_header(models: list[str], per_model: int) -> str:
    groups = " & ".join(rf"\multicolumn{{{per_model}}}{{c}}{{{m}}}" for m in models)
    return f" & {groups} \\\\"


def interp_sim(models: dict[str, dict[str, dict[str, str]]], arms: list[str] | None = None,
               thresholds: bool = False) -> str:
    """Interp and Sim per model; with `thresholds`, extra rows for firing at g > tau."""
    names = list(models)
    header = [_model_header(names, 2),
              "Method & " + " & ".join(f"{INTERP.header} & {SIM.header}" for _ in names) + r" \\"]
    body = []
    for arm in _arms(models, arms):
        cells = [f"{cell(models[m].get(arm), INTERP)} & {cell(models[m].get(arm), SIM)}" for m in names]
        body.append(f"{ARM_NAMES[arm]} & " + " & ".join(cells) + r" \\")
        if thresholds and arm.startswith("vpd"):
            for tau in TAUS:
                cells = [
                    f"{cell(models[m].get(arm), INTERP, f'intruder_ci{tau:g}_mean')} & "
                    f"{cell(models[m].get(arm), SIM, f'sim_ci{tau:g}')}"
                    for m in names
                ]
                body.append(f"{ARM_NAMES[arm]} + ($g_{{t,c}} > {tau:g}$) & " + " & ".join(cells) + r" \\")
    return _table(header, body, 1 + 2 * len(names), "Interpretability and diversity.")


def matching(models: dict[str, dict[str, dict[str, str]]], arms: list[str] | None = None) -> str:
    names = list(models)
    header = [" & " + " & ".join(names) + r" \\", "Method & " + " & ".join(MATCHING.header for _ in names) + r" \\"]
    body = [
        f"{ARM_NAMES[arm]} & " + " & ".join(cell(models[m].get(arm), MATCHING) for m in names) + r" \\"
        for arm in _arms(models, arms)
    ]
    return _table(header, body, 1 + len(names), "Meaning localization (matching over random pairs).")


def editing(models: dict[str, dict[str, dict[str, str]]], arms: list[str] | None = None) -> str:
    names = list(models)
    header = [" &" + _model_header(names, 2),
              " & Method & " + " & ".join(" & ".join(m.header for m in SINGLE) for _ in names) + r" \\"]
    body = []
    for label, metrics in (("Single", SINGLE), ("Multiple", MULTIPLE)):
        for i, arm in enumerate(_arms(models, arms)):
            cells = [" & ".join(cell(models[m].get(arm), mt) for mt in metrics) for m in names]
            body.append(f"{label if i == 0 else ''} & {ARM_NAMES[arm]} & " + " & ".join(cells) + r" \\")
        body.append(r"\midrule")
    body.pop()
    return _table(header, body, 2 + 2 * len(names), "Weight editing localization.")


def write_all(model_csvs: dict[str, Path], out_dir: Path, arms: list[str] | None = None) -> list[Path]:
    models = {name: load(path) for name, path in model_csvs.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "interp_sim.tex": interp_sim(models, arms),
        "interp_sim_thresholds.tex": interp_sim(models, arms, thresholds=True),
        "matching.tex": matching(models, arms),
        "editing.tex": editing(models, arms),
    }
    paths = []
    for name, text in outputs.items():
        path = out_dir / name
        path.write_text(text)
        paths.append(path)
    return paths
