"""The paper table: the checkpoint rule, and the collapses that produce a column.

Every assertion here is about a decision that had an alternative -- which `k`, which estimator,
what happens to a stage that lagged. A column that silently changed convention would
still produce a plausible number, which is the failure this file exists to catch.
"""

import json
import re
from pathlib import Path

import pytest

from aspd.eval.tables import (
    COLUMNS,
    build_rows,
    collect_row,
    column_order,
    write_summary,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _attr_row(delta: float, collateral: float, localization: float, field: str = "on_aj") -> dict:
    return {f"{field}_delta_abs": delta, f"{field}_collateral_abs": collateral,
            f"{field}_localization": localization}


def make_run(root: Path, name: str, *, steps=(100, 200), eval_step: int = 200) -> Path:
    run = root / name
    run.mkdir(parents=True)
    for step in steps:
        (run / f"model_{step}.pth").write_bytes(b"")
    (run / "experiment_config.yaml").write_text("pd:\n  C: 1024\n")

    _write(run / "attr_edit" / "sweep_attr_edit.json", {"per_step": {str(eval_step): {"by_k": {
        "ranked/1": _attr_row(1.0, 10.0, 0.1),
        "ranked/5": _attr_row(3.0, 30.0, 0.3),
        "random/50": _attr_row(999.0, 999.0, 0.4),
        "norm_matched/50": _attr_row(999.0, 999.0, 999.0),
    }}}})
    _write(run / "attr_edit_multi" / "m5_cond" / "sweep_attr_edit_multi.json",
           {"per_step": {str(eval_step): {"by_k": {
               "ranked/1/union": _attr_row(2.0, 20.0, 0.2, "on_a"),
               "ranked/5/union": _attr_row(4.0, 40.0, 0.4, "on_a"),
               "random/1/union": _attr_row(999.0, 999.0, 0.3, "on_a"),
               "random/5/union": _attr_row(999.0, 999.0, 0.5, "on_a"),
               # `target` is the seed component, not the selected set.
               "ranked/5/target": _attr_row(999.0, 999.0, 999.0, "on_a"),
               "random/5/target": _attr_row(999.0, 999.0, 999.0, "on_a")}}}})

    _write(run / "harvest" / f"h-step{eval_step:06d}" / "intruder_summary.json",
           {"mean": 0.4, "std": 0.2, "n_scored": 199})
    _write(run / "harvest" / f"h-step{eval_step:06d}" / "intruder_summary_ci0.01.json",
           {"mean": 0.55, "std": 0.25, "n_scored": 120})
    _write(run / "harvest" / f"h-step{eval_step:06d}" / "intruder_summary_ci0.1.json",
           {"mean": None, "std": None, "n_scored": 0})
    _write(run / "harvest" / f"h-step{eval_step:06d}" / "intruder_summary_act0.01.json",
           {"mean": 0.61, "std": 0.2, "n_scored": 200})
    _write(run / "harvest" / f"h-step{eval_step:06d}" / "intruder_summary_act0.1.json",
           {"mean": 0.64, "std": 0.2, "n_scored": 198})

    for suffix, z in (("", 0.02), ("_ci0.01", 0.03), ("_ci0.1", 0.05)):
        _write(run / "diversity" / f"diversity_h-step{eval_step}{suffix}.json",
               {"z_bar": z, "z_bar_ci95": z / 10, "n_sampled": 500})

    _write(run / "matching" / f"matching_step{eval_step}.json", {
        "meta": {"scheme": "in_to_out"},
        "results": [
            {"name": "A_cond", "mean_score": 1.8}, {"name": "A_glob", "mean_score": 1.4},
            {"name": "random", "mean_score": 1.1}]})
    return run


def test_every_stage_lands_at_the_last_checkpoint(tmp_path):
    row = collect_row(make_run(tmp_path, "arm"), "arm", nearest_below=False)

    assert row["step"] == 200 and row["C"] == 1024 and row["_blanked"] == []

    # `ranked` only, max k = 5 (not the controls' 50), and the mean is over ranked's k alone.
    assert row["attr_max_k"] == 5.0
    assert row["attr_target_change_at_max_k"] == 3.0 and row["attr_target_change_mean_over_k"] == 2.0
    assert row["attr_collateral_at_max_k"] == 30.0 and row["attr_localization_at_max_k"] == 0.3
    assert row["multi_m5_cond_target_change_at_max_k"] == 4.0, "the `union` selection, not `target`"
    assert row["multi_m5_cond_target_change_mean_over_k"] == 3.0

    assert row["intruder_mean"] == 0.4 and row["intruder_std"] == 0.2
    assert row["sim"] == 0.02 and row["sim_ci0.01"] == 0.03 and row["sim_ci0.1"] == 0.05
    assert row["matching_A_cond"] == 1.8 and row["matching_A_glob"] == 1.4
    assert row["matching_random"] == 1.1, "the control the two judged scores are read against"


def test_localization_over_random_divides_the_two_means(tmp_path):
    """The one column built on a control. It is a ratio of the two mean-over-k columns beside it,
    not a mean of per-k ratios, and the control's own k sweep is what its denominator averages.
    """
    row = collect_row(make_run(tmp_path, "arm"), "arm", nearest_below=False)

    assert row["attr_localization_random_mean_over_k"] == 0.4, "the `random` rows, not `ranked`"
    assert row["attr_localization_over_random"] == pytest.approx((0.1 + 0.3) / 2 / 0.4)
    # The controls' 999s reach neither the metric columns nor `max_k`, ratio or no ratio.
    assert row["attr_collateral_mean_over_k"] == 20.0 and row["attr_max_k"] == 5.0

    assert row["multi_m5_cond_localization_random_mean_over_k"] == pytest.approx(0.4)
    assert row["multi_m5_cond_localization_over_random"] == pytest.approx(0.3 / 0.4)
    assert row["multi_cond_localization_random_mean_over_k_and_m"] == pytest.approx(0.4)
    assert row["multi_cond_localization_over_random"] == pytest.approx(0.3 / 0.4)


def test_an_arm_with_no_random_control_gets_no_ratio(tmp_path):
    """A sweep predating the control carries `ranked/` only. A blank ratio and a ratio of 1.0 are
    different claims, and only the first is true of a run that never measured the denominator.
    """
    run = make_run(tmp_path, "arm")
    path = run / "attr_edit" / "sweep_attr_edit.json"
    data = json.loads(path.read_text())
    by_k = data["per_step"]["200"]["by_k"]
    data["per_step"]["200"]["by_k"] = {k: v for k, v in by_k.items() if k.startswith("ranked/")}
    path.write_text(json.dumps(data))

    row = collect_row(run, "arm", nearest_below=False)
    assert row["attr_localization_random_mean_over_k"] is None
    assert "attr_localization_over_random" not in row
    assert row["attr_localization_mean_over_k"] == 0.2, "the rest of the stage is unaffected"


def test_a_lagging_stage_is_blank_not_its_own_newest_step(tmp_path):
    """The strict rule: every number in a row must describe the same weights."""
    run = make_run(tmp_path, "arm", steps=(100, 200), eval_step=100)
    row = collect_row(run, "arm", nearest_below=False)

    assert row["step"] == 200
    assert set(row["_blanked"]) == {stage for stage, _, _ in COLUMNS}
    assert row["matching_step"] is None and "matching_A_cond" not in row and "sim" not in row

    relaxed = collect_row(run, "arm", nearest_below=True)
    assert relaxed["_blanked"] == [] and relaxed["matching_A_cond"] == 1.8 and relaxed["sim"] == 0.02
    # The step columns are what keeps a relaxed row distinguishable from a strict one.
    assert relaxed["step"] == 200 and relaxed["matching_step"] == 100


def test_csv_columns_are_stable_when_an_arm_is_missing_a_stage(tmp_path):
    full = make_run(tmp_path, "full")
    lagging = make_run(tmp_path, "lags", steps=(100, 200), eval_step=100)
    rows = build_rows([full, lagging], ["full", "lags"], nearest_below=False)
    path = write_summary(rows, tmp_path / "summary.csv")

    header, *lines = path.read_text().splitlines()
    order = column_order(rows)
    assert header.split(",") == order
    assert order[:4] == ["arm", "run", "C", "step"]
    assert "multi_m5_cond_target_change_at_max_k" in order, "discovered arms reach the header"
    # Same width for both rows, and the missing cells are empty rather than absent or 0.
    assert all(len(line.split(",")) == len(order) for line in lines)
    assert lines[1].split(",")[order.index("matching_A_cond")] == ""


def test_at_max_k_is_the_value_at_the_largest_k_not_the_best_over_k():
    """The two readings the column name has to distinguish."""
    from aspd.eval.tables import _attr_columns

    by_k = {"ranked/1": _attr_row(1.0, 10.0, 0.1),
            "ranked/5": _attr_row(6.0, 60.0, 0.6),
            "ranked/10": _attr_row(2.0, 20.0, 0.2)}
    cols = _attr_columns(by_k, "attr", "on_aj")

    assert cols["attr_max_k"] == 10.0
    assert cols["attr_target_change_at_max_k"] == 2.0, "the k=10 row, not the k=5 maximum of 6.0"
    assert cols["attr_target_change_mean_over_k"] == 3.0, "(1 + 6 + 2) / 3"
    assert max(cols["attr_target_change_at_max_k"],
               cols["attr_target_change_mean_over_k"]) < 6.0, "no column is a max over k"


def test_multi_averages_over_m_as_a_mean_of_means(tmp_path):
    """`m` arms do not share a `k` sweep, so the aggregate weights each `m` once, not each `(m,k)`."""
    run = make_run(tmp_path, "arm")
    _write(run / "attr_edit_multi" / "m5_cond" / "sweep_attr_edit_multi.json",
           {"per_step": {"200": {"by_k": {
               "ranked/1/union": _attr_row(2.0, 20.0, 0.2, "on_a"),
               "ranked/5/union": _attr_row(6.0, 40.0, 0.6, "on_a")}}}})
    _write(run / "attr_edit_multi" / "m50_cond" / "sweep_attr_edit_multi.json",
           {"per_step": {"200": {"by_k": {
               "ranked/1/union": _attr_row(9.0, 90.0, 0.9, "on_a")}}}})
    row = collect_row(run, "arm", nearest_below=False)

    assert row["multi_m5_cond_target_change_mean_over_k"] == 4.0, "(2 + 6) / 2"
    assert row["multi_m50_cond_target_change_mean_over_k"] == 9.0
    assert row["multi_cond_target_change_mean_over_k_and_m"] == 6.5, "(4 + 9) / 2, not 5.667"
    assert row["multi_cond_n_m"] == 2.0

    # `cond` and `global` are different selection rules and are never merged.
    assert "multi_global_target_change_mean_over_k_and_m" not in row


def test_the_compact_table_is_means_only(tmp_path):
    """The compact view is the paper's columns: no `at_max_k`, no per-`m` arm."""
    from aspd.eval.tables import compact_order

    rows = build_rows([make_run(tmp_path, "arm")], ["arm"], nearest_below=False)
    compact = compact_order(rows)

    assert "attr_localization_mean_over_k" in compact and "sim" in compact
    assert "multi_cond_localization_mean_over_k_and_m" in compact
    assert not [c for c in compact if c.endswith("_at_max_k")], "at_max_k lives in the full CSV"
    assert not [c for c in compact if re.match(r"^multi_m\d+_", c)], "per-m arms are not compact"
    # ... but the full order keeps both, so nothing collected is discarded.
    full = column_order(rows)
    assert "attr_target_change_at_max_k" in full
    assert "multi_m5_cond_target_change_mean_over_k" in full


def test_a_threshold_with_no_eligible_component_is_nan_not_a_blank(tmp_path):
    """The two failure shapes a reader must be able to tell apart."""
    row = collect_row(make_run(tmp_path, "arm"), "arm", nearest_below=False)

    assert row["intruder_mean"] == 0.4 and row["intruder_n_scored"] == 199
    assert row["intruder_ci0.01_mean"] == 0.55 and row["intruder_ci0.01_n_scored"] == 120
    # Ran, scored nothing: present, null, and countable.
    assert "intruder_ci0.1_mean" in row
    assert row["intruder_ci0.1_mean"] is None and row["intruder_ci0.1_n_scored"] == 0
    assert row["intruder_ci0.1_step"] == row["step"], "a null mean still names its checkpoint"
    assert "intruder_ci0.1" not in row["_blanked"]

    # Never ran: no column at all, and the stage says so.
    bare_run = make_run(tmp_path / "bare", "bare", eval_step=200)
    for path in (bare_run / "harvest").glob("h-step*/intruder_summary_ci*.json"):
        path.unlink()
    bare = collect_row(bare_run, "bare", nearest_below=False)
    assert "intruder_ci0.1_mean" not in bare
    assert "intruder_ci0.1" in bare["_blanked"]
