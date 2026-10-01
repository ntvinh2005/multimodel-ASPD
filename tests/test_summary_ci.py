"""The 95% CI columns: each interval belongs to the mean it decorates."""

import json
import math
from pathlib import Path

import pytest

from aspd.eval.table_ci import (
    COVERAGE_APPROX,
    attach_ci,
    cell,
    ci_half_width,
    with_ci_cells,
)
from aspd.eval.tables import collect_row
from tests.test_summary_table import make_run

# Four units per population, offset so each one's mean is exactly the aggregate `make_run` wrote.
SPREAD = (-0.3, -0.1, 0.1, 0.3)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _units(mean: float, scale: float = 1.0) -> list[float]:
    return [mean + d * scale for d in SPREAD]


def _control_units(mean: float, scale: float) -> list[float]:
    """The same mean, spread the OTHER way across the units."""
    return list(reversed(_units(mean, scale)))


def _edits(field: str, by_k: dict[int, tuple[float, float, float]], **extra) -> list[dict]:
    rows = []
    for k, (delta, collateral, localization) in by_k.items():
        for i, (d, c, loc) in enumerate(zip(_units(delta), _units(collateral, 10.0),
                                            _units(localization, 0.01), strict=True)):
            rows.append({"k": k, "kind": "ranked", **extra, "unit": i,
                         field: {"delta_abs": d, "collateral_abs": c, "localization": loc}})
    return rows


def make_items(run: Path, eval_step: int = 200) -> Path:
    """Add the per-item files beside the aggregates `make_run` wrote, agreeing with them."""
    attr = _edits("on_aj", {1: (1.0, 10.0, 0.1), 5: (3.0, 30.0, 0.3)})
    for row in attr:
        row["feature_id"] = row.pop("unit")
    attr += [{"k": 50, "kind": "norm_matched", "feature_id": i,
              "on_aj": {"delta_abs": 999.0, "collateral_abs": 999.0, "localization": 999.0}}
             for i in range(4)]
    attr += [{"k": 50, "kind": "random", "feature_id": i, "rep": rep,
              "on_aj": {"delta_abs": 999.0, "collateral_abs": 999.0,
                        "localization": _control_units(0.4, 0.02)[i]}}
             for rep in range(2) for i in range(4)]
    _write(run / "attr_edit" / f"attr_edit_step{eval_step}.json", {"edits": attr})

    multi = _edits("on_a", {1: (2.0, 20.0, 0.2), 5: (4.0, 40.0, 0.4)}, block="union", m=5)
    for row in multi:
        row["combo"] = row.pop("unit")
    # `target` is the seed component, not the selected set -- same rows, different quantity.
    multi += [{"k": 5, "kind": kind, "block": "target", "combo": i, "m": 5,
               "on_a": {"delta_abs": 999.0, "collateral_abs": 999.0, "localization": 999.0}}
              for kind in ("ranked", "random") for i in range(4)]
    multi += [{"k": 5, "kind": "random", "block": "union", "combo": i, "m": 5, "rep": rep,
               "on_a": {"delta_abs": 999.0, "collateral_abs": 999.0,
                        "localization": _control_units(0.4, 0.02)[i]}}
              for rep in range(2) for i in range(4)]
    _write(run / "attr_edit_multi" / "m5_cond" / f"attr_edit_multi_step{eval_step}.json",
           {"edits": multi, "m": 5})

    summary = json.loads((run / "harvest" / f"h-step{eval_step:06d}" / "intruder_summary.json")
                         .read_text())
    summary["scores"] = {f"mod:{i}": v for i, v in enumerate(_units(0.4))}
    _write(run / "harvest" / f"h-step{eval_step:06d}" / "intruder_summary.json", summary)

    scr_averaged = {"scr_metric_threshold_2": 0.1, "scr_metric_threshold_10": 0.7,
                    "scr_metric_threshold_100": 0.4, "scr_dir1_threshold_10": 999.0}
    _write(run / "scr_tpp" / f"scr_components_step{eval_step}.json", {
        "averaged": scr_averaged,
        "per_run": {f"ds{i}": {k: _units(v)[i] for k, v in scr_averaged.items()}
                    for i in range(4)}})

    tpp_averaged = {"tpp_threshold_2_total_metric": 0.2, "tpp_threshold_10_total_metric": 0.6,
                    "tpp_threshold_2_intended_diff_only": 999.0}
    per_class = {f"ds{d}": {str(c): {k: _units(v)[2 * d + c] for k, v in tpp_averaged.items()}
                            for c in range(2)} for d in range(2)}
    # `per_run` is the dataset-level mean of those classes -- the coarser, consistent alternative.
    _write(run / "scr_tpp" / f"tpp_components_step{eval_step}.json", {
        "averaged": tpp_averaged,
        "per_run": {ds: {k: sum(c[k] for c in classes.values()) / len(classes)
                         for k in tpp_averaged} for ds, classes in per_class.items()},
        "meta": {"per_class": per_class}})

    matching = json.loads((run / "matching" / f"matching_step{eval_step}.json").read_text())
    for arm, result in enumerate(matching["results"]):
        result["scores"] = (_control_units(result["mean_score"], 1.0)
                            if result["name"] == "random" else _units(result["mean_score"]))
        # Pair `i` is the same COMPONENT in every arm; only the features it is paired with differ.
        result["components"] = list(range(len(SPREAD)))
        result["pairs"] = [[c, 100 * arm + c] for c in range(len(SPREAD))]
    _write(run / "matching" / f"matching_step{eval_step}.json", matching)
    return run


@pytest.fixture
def row_and_dir(tmp_path):
    run = make_items(make_run(tmp_path, "arm"))
    return run, collect_row(run, "arm", nearest_below=False)


def test_every_population_reconstructs_the_column_it_decorates(row_and_dir):
    """`attach_ci` asserts this internally; the point here is that it covers every CI column."""
    run, row = row_and_dir
    reconstructed = attach_ci(run, row)

    assert reconstructed == pytest.approx({
        "attr_target_change_mean_over_k": 2.0,
        "attr_collateral_mean_over_k": 20.0,
        "attr_localization_mean_over_k": 0.2,
        "attr_localization_random_mean_over_k": 0.4,
        "attr_localization_over_random": 0.5,
        "multi_cond_target_change_mean_over_k_and_m": 3.0,
        "multi_cond_collateral_mean_over_k_and_m": 30.0,
        "multi_cond_localization_mean_over_k_and_m": 0.3,
        "multi_cond_localization_random_mean_over_k_and_m": 0.4,
        "multi_cond_localization_over_random": 0.75,
        "intruder_mean": 0.4,
        "matching_A_cond": 1.8, "matching_A_glob": 1.4, "matching_random": 1.1,
        "matching_A_cond_margin": 0.7, "matching_A_glob_margin": 0.3,
    })


def test_the_ratio_column_is_a_ratio_of_means_not_a_mean_of_ratios(row_and_dir):
    """`localization_over_random` divides two means, so it has no population to average. Its CI
    rides on the linearised residuals, whose mean is the ratio EXACTLY and whose spread keeps the
    per-unit pairing -- both properties a naive mean of per-unit quotients would break.
    """
    run, row = row_and_dir
    reconstructed = attach_ci(run, row)

    assert reconstructed["attr_localization_over_random"] == pytest.approx(0.5)
    assert row["attr_localization_over_random_ci95_n"] == 4.0, "the features, not the k x rep rows"

    ranked = _units(0.2, 0.01)
    control = _control_units(0.4, 0.02)
    naive = sum(x / y for x, y in zip(ranked, control, strict=True)) / 4
    assert naive != pytest.approx(0.5, abs=1e-9)

    # The interval is the spread of `z_i = R + (x_i - R*y_i)/Y`, not of those quotients.
    expected = [0.5 + (x - 0.5 * y) / 0.4 for x, y in zip(ranked, control, strict=True)]
    assert row["attr_localization_over_random_ci95"] == pytest.approx(ci_half_width(expected))
    assert row["multi_cond_localization_over_random_ci95"] is not None


def _add_second_m(run: Path, localization: float, control: float = 0.4,
                  step: int = 200) -> None:
    """An `m2_cond` arm whose control reaches only half the combinations, as m20 / m50 do."""
    ranked = _edits("on_a", {1: (2.0, 20.0, localization)}, block="union", m=2)
    for edit in ranked:
        edit["combo"] = edit.pop("unit")
    rows = [{"k": 1, "kind": "random", "block": "union", "combo": i, "m": 2, "rep": 0,
             "on_a": {"delta_abs": 999.0, "collateral_abs": 999.0,
                      "localization": _control_units(control, 0.02)[i]}}
            for i in range(2)]
    _write(run / "attr_edit_multi" / "m2_cond" / f"attr_edit_multi_step{step}.json",
           {"edits": ranked + rows, "m": 2})
    _write(run / "attr_edit_multi" / "m2_cond" / "sweep_attr_edit_multi.json",
           {"per_step": {str(step): {"by_k": {
               "ranked/1/union": {"on_a_delta_abs": 2.0, "on_a_collateral_abs": 20.0,
                                  "on_a_localization": localization},
               "random/1/union": {"on_a_delta_abs": 999.0, "on_a_collateral_abs": 999.0,
                                  "on_a_localization": control}}}}})


def test_every_side_of_the_multi_ratio_averages_over_all_the_m_it_has(row_and_dir):
    """`n_control_combos` shrinks with m -- 50 / 25 / 10 in the real sweep -- so a combination has
    a control at only some m. Each side still averages over all the m it HAS: the ranked side over
    every one, the control over its own. The ranked side is NOT restricted to the control's
    combinations, which would align the two columns exactly and move the ratio by up to 47%.
    """
    run, _ = row_and_dir
    _add_second_m(run, 0.32)
    row = collect_row(run, "arm", nearest_below=False)
    reconstructed = attach_ci(run, row)

    assert "multi_cond_localization_over_random" in COVERAGE_APPROX, "the licensed exception"
    assert row["multi_cond_localization_over_random_ci95_n"] == 4.0, "all four combinations"
    assert reconstructed["multi_cond_localization_mean_over_k_and_m"] == pytest.approx(0.31)
    assert reconstructed["multi_cond_localization_over_random"] == pytest.approx(0.31 / 0.4)
    assert row["multi_cond_localization_over_random"] == pytest.approx(0.31 / 0.4)


def test_a_coverage_gap_far_larger_than_the_real_one_still_fires(tmp_path):
    """`COVERAGE_TOL` licenses the sweep's own imbalance, not any disagreement at all. A control
    read off the wrong m -- or off `target` -- lands orders of magnitude out, and must not pass.
    """
    run = make_items(make_run(tmp_path, "arm"))
    _add_second_m(run, 0.32, control=40.0)
    row = collect_row(run, "arm", nearest_below=False)

    with pytest.raises(AssertionError, match="multi_cond_localization"):
        attach_ci(run, row)


def test_the_matching_margin_is_differenced_per_component_not_between_two_means(row_and_dir):
    """`score - random` is a PAIRED difference: pair `i` is the same component in every arm, so
    the between-component spread -- which is most of it -- cancels. Subtracting the pooled control
    mean instead lands on the same number with the wrong interval.
    """
    run, row = row_and_dir
    attach_ci(run, row)

    assert row["matching_A_cond_margin_ci95_n"] == 4.0, "the judged pairs, not the arms"
    paired = [s - c for s, c in zip(_units(1.8), _control_units(1.1, 1.0), strict=True)]
    assert row["matching_A_cond_margin_ci95"] == pytest.approx(ci_half_width(paired))
    # The unpaired alternative on the same fixture: every pair against the same control mean.
    unpaired = [s - 1.1 for s in _units(1.8)]
    assert ci_half_width(unpaired) != pytest.approx(ci_half_width(paired))


def test_a_margin_over_a_control_judged_on_other_components_is_an_error(row_and_dir):
    """Both sides having 200 pairs does not make them paired. An unaligned file would give the
    same mean and an interval describing a difference nobody measured.
    """
    run, _ = row_and_dir
    path = run / "matching" / "matching_step200.json"
    data = json.loads(path.read_text())
    for result in data["results"]:
        if result["name"] == "random":
            result["components"] = [c + 100 for c in result["components"]]
    path.write_text(json.dumps(data))

    with pytest.raises(AssertionError, match="not paired"):
        attach_ci(run, collect_row(run, "arm", nearest_below=False))


def test_a_population_that_disagrees_with_its_column_is_an_error(row_and_dir):
    """The controls and the `target` block are the realistic wrong reads, so the check that would
    catch them has to fire rather than round away.
    """
    run, row = row_and_dir
    row["attr_localization_mean_over_k"] = 0.25

    with pytest.raises(AssertionError, match="attr_localization_mean_over_k"):
        attach_ci(run, row)


def test_intervals_are_student_t(row_and_dir):
    """At n = 8 the t factor is 2.365 against z's 1.96 -- a 20% difference on the SCR columns."""
    assert ci_half_width([0.0, 1.0] * 4) == pytest.approx(2.36462 * 0.5345225 / math.sqrt(8), rel=1e-4)
    assert ci_half_width([1.0]) is None, "a single unit has no interval"

    run, row = row_and_dir
    attach_ci(run, row)
    # SPREAD's stdev is 0.2582; four units, t(3) = 3.1824.
    assert row["intruder_mean_ci95"] == pytest.approx(3.18245 * 0.2581989 / 2.0, rel=1e-4)


def test_ce_and_fvu_keep_a_bare_mean(row_and_dir):
    """`ce_kl` accumulates running sums, so no per-batch spread exists to interval. The cell must
    then be the bare number -- an interval invented for it would be the one indefensible cell.
    """
    run, row = row_and_dir
    attach_ci(run, row)
    order = ["arm", "ce_spliced_nodelta", "fvu_nodelta", "intruder_mean", "intruder_std"]
    cells = with_ci_cells(row, order)

    assert "±" not in cells["ce_spliced_nodelta"] and "±" not in cells["fvu_nodelta"]
    assert "±" not in cells["intruder_std"], "a spread across components, not an interval"
    assert cells["intruder_mean"].startswith("0.4 ±")
    assert cells["arm"] == "arm", "identity columns are never reformatted"


def test_a_blank_cell_stays_blank(row_and_dir):
    """A stage that blanked under the checkpoint rule must not acquire a `nan ± nan`."""
    assert cell(None, None) == "" and cell(None, 0.1) == ""
    assert cell(1.5, None) == "1.5"
