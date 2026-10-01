"""Write the paper table for several runs (one row per method) as CSV."""

import argparse
from pathlib import Path

from aspd.eval.table_ci import attach_ci, ci_report, with_ci_cells
from aspd.eval.tables import (
    blanked_report,
    build_rows,
    compact_order,
    full_order,
    multi_m_swept,
    render_table,
    write_summary,
)


def arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", default=[], required=True,
                    help="repeat once per arm; row order follows this order")
    ap.add_argument("--out", required=True,
                    help="path of the CSV to write (absolute -- the launchers cd into this lib)")
    ap.add_argument("--nearest-below", action="store_true",
                    help="let a stage fall back to its newest step at or below the arm's last "
                         "checkpoint, instead of blanking. The *_step columns record which was "
                         "used, so a relaxed row stays distinguishable from a strict one")
    ap.add_argument("--ci", action="store_true",
                    help="write each compact cell as `mean ± half-width` of its 95% CI, rebuilt "
                         "from the per-item records on disk (features, combinations, components, "
                         "runs, classes, pairs). The full CSV keeps bare floats and gains a "
                         "<column>_ci95 and <column>_n beside each. Costs seconds per arm: it "
                         "reads the per-edit files the sweep files exist to avoid")
    return ap


def main() -> None:
    args = arg_parser().parse_args()
    run_dirs = [Path(d) for d in args.run_dir]
    for run_dir in run_dirs:
        assert run_dir.exists(), f"no such run dir: {run_dir}"

    from aspd.eval.tables import run_arm

    labels = [run_arm(d) for d in run_dirs]
    assert len(set(labels)) == len(labels), f"arm labels collide: {labels}"

    rows = build_rows(run_dirs, labels, nearest_below=args.nearest_below)

    compact = compact_order(rows)
    compact_rows = rows
    if args.ci:
        for run_dir, row in zip(run_dirs, rows, strict=True):
            attach_ci(run_dir, row)
        compact_rows = [with_ci_cells(row, compact) for row in rows]
    path = write_summary(compact_rows, Path(args.out), compact)
    full_path = write_summary(rows, Path(args.out).with_name(Path(args.out).stem + "_full.csv"),
                              full_order(rows))

    print(render_table(compact_rows, compact))
    swept = multi_m_swept(rows)
    print(f"\n[summary] {len(rows)} arm(s) x {len(compact)} column(s) -> {path}")
    print(f"[summary] {len(full_order(rows))} columns with intervals, counts and steps -> {full_path}")
    print("[summary] attr columns are MEANS, on the `ranked` estimator. attr_edit: mean over "
          "every k in\n"
          "          the sweep. The exception is `*_localization_over_random` -- localization "
          "divided by\n"
          "          the `random` control's, so 1.0 is `the ranking bought nothing a uniform "
          "draw of the\n"
          "          same k did not`. Its denominator is in the full CSV; `norm_matched` is not "
          "here at all.")
    if swept:
        print(f"[summary] attr_edit_multi: mean over k, then over m = {swept} -- a mean of means, "
              "so each m\n"
              "          counts once despite m1 sweeping more k. `cond` and `global` stay apart. "
              "Per-m arms,\n"
              "          and the value at the largest k, are in the full CSV.")
    if args.ci:
        print("[summary] cells are `mean ± half-width of the 95% CI` (Student-t) over:")
        print("\n".join(ci_report(rows, compact)))
        print("[summary] sim cells carry the bootstrap 95% interval (1.96 x SE over resampled components).")
        print("[summary] the multi `random` control is drawn on fewer combinations at large m, so "
              "the four\n"
              "          columns marked approximate are centred NEAR their column rather "
              "than exactly\n"
              "          on it -- a median 2.4% and at most 15%. `summary_ci.COVERAGE_APPROX` "
              "has the why.")
    report = blanked_report(run_dirs, rows)
    if report:
        rule = "nearest at or below" if args.nearest_below else "exact last checkpoint"
        print(f"[summary] stages with no eval at the {rule}; these cells are BLANK, not zero:")
        print("\n".join(report))


if __name__ == "__main__":
    main()
