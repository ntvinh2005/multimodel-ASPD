"""Diversity (mean pairwise token-set overlap) of the components of one or more runs."""

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import yaml

from aspd.paths import RUNS_DIR

TARGET_ALIASES = {
    "gpt2s": "openai-community/gpt2",
    "gpt2xl": "openai-community/gpt2-xl",
    "gemma2": "google/gemma-2-2b",
    "qwen3": "Qwen/Qwen3-8B",
    "ss2l": "goodfire/spd/runs/gf6rbga0",
}


def run_target(run_dir: Path) -> str | None:
    """The target model a run decomposes, from `experiment_config.yaml`."""
    cfg_path = run_dir / "experiment_config.yaml"
    if not cfg_path.exists():
        return None
    spec = yaml.safe_load(cfg_path.read_text()).get("target", {}).get("spec", {})
    return spec.get("model_name") or spec.get("params", {}).get("model_name") or spec.get("run_path")


def run_ci_mode(run_dir: Path) -> str | None:
    """The CI fn's `mode` from `experiment_config.yaml` -- which ARM FAMILY a run is."""
    cfg = yaml.safe_load((run_dir / "experiment_config.yaml").read_text())
    return ((cfg.get("pd") or {}).get("ci_config") or {}).get("mode")


def discover_runs(runs_root: Path, target: str) -> list[Path]:
    """Every non-symlink run under `runs_root` whose config names `target`."""
    wanted = TARGET_ALIASES.get(target, target)
    return sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and not d.is_symlink() and run_target(d) == wanted
    )


def final_harvest(run_dir: Path) -> Path | None:
    """The `h-step*` harvest with the highest step, or the newest `h-*` if none are stepped."""
    stepped = sorted(run_dir.glob("harvest/h-step*/harvest.db"),
                     key=lambda p: int(p.parent.name.removeprefix("h-step")))
    if stepped:
        return stepped[-1]
    other = sorted(run_dir.glob("harvest/h-*/harvest.db"))
    return other[-1] if other else None


def main() -> None:
    from aspd.eval.diversity import (
        bootstrap,
        jaccard_kernel,
        load_population,
        mean_off_diagonal,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", default=None, help="run directories")
    ap.add_argument("--target", default=None,
                    help=f"screen every run on disk decomposing this target: "
                         f"{'|'.join(TARGET_ALIASES)}, or a full model name")
    ap.add_argument("--runs-root", default=str(RUNS_DIR))
    ap.add_argument("--min-density", type=float, default=5e-5)
    ap.add_argument("--max-density", type=float, default=1e-3)
    ap.add_argument("--min-firings", type=int, default=2,
                    help="drop tokens a component fired on fewer than this many times")
    ap.add_argument("--n-sample", type=int, default=500,
                    help="components sampled per run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-band", type=int, default=20, help="skip runs with a smaller band")
    ap.add_argument("--out", default=None, help="write the result JSON here (CSV beside it)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="read each harvest as if it fired only above this; "
                         "omitted = the harvests' own threshold, 0")
    ap.add_argument("--criterion", choices=("ci", "act"), default="ci",
                    help="what --threshold applies to: `ci` the causal importance, `act` "
                         "|component activation| / its own peak")
    ap.add_argument("--ci-modes", nargs="+", default=None,
                    help="keep only runs whose config CI mode is one of these, e.g. "
                         "`pd_transcoder aspd`")
    ap.add_argument("--sort-by", choices=("z_bar",), default="z_bar", help="row order (ascending)")
    args = ap.parse_args()
    assert bool(args.run_dirs) != bool(args.target), "give exactly one of --run-dirs / --target"

    if args.target:
        run_dirs = discover_runs(Path(args.runs_root), args.target)
        assert run_dirs, f"no run under {args.runs_root} decomposes {args.target}"
        print(f"[diversity] target {args.target}: {len(run_dirs)} runs on disk")
    else:
        run_dirs = sorted({Path(r) for r in args.run_dirs})
    if args.ci_modes:
        run_dirs = [rd for rd in run_dirs if run_ci_mode(rd) in set(args.ci_modes)]
        assert run_dirs, f"no run has a CI mode in {args.ci_modes}"
        print(f"[diversity] {len(run_dirs)} runs with CI mode in {args.ci_modes}")
    if args.threshold is not None and args.threshold <= 0.0:
        args.threshold = None
    if args.threshold is not None:
        from aspd.eval.harvest_threshold import threshold_stats
        print(f"[diversity] {args.criterion} threshold {args.threshold:g}")

    rows, skipped = [], []
    for rd in run_dirs:
        db_path = final_harvest(rd)
        if db_path is None:
            print(f"[diversity] {rd.name}: no harvest -- skipped")
            skipped.append((rd.name, "no harvest"))
            continue
        step = db_path.parent.name
        stats = None
        if args.threshold is not None:
            stats = threshold_stats(db_path.parent, [args.threshold], criterion=args.criterion,
                                    min_examples=5)
        n_band = _band_size(db_path, args, stats)
        if n_band < args.min_band:
            print(f"[diversity] {rd.name} ({step}): band={n_band} < {args.min_band} -- skipped")
            skipped.append((rd.name, f"band={n_band} < {args.min_band}"))
            continue

        pop = load_population(
            db_path, min_density=args.min_density, max_density=args.max_density,
            min_firings=args.min_firings, n_sample=args.n_sample, seed=args.seed,
            threshold=args.threshold, stats=stats,
        )
        z = jaccard_kernel(pop.token_sets)
        z_bar = mean_off_diagonal(z)
        boot = bootstrap(z, pop.n_band, n_boot=args.n_boot, seed=args.seed)
        rows.append({
            "run": rd.name, "step": step, "target": run_target(rd),
            "ci_mode": run_ci_mode(rd),
            "criterion": args.criterion if args.threshold is not None else None,
            "threshold": args.threshold if args.threshold is not None else 0.0,
            "n_eligible": pop.n_eligible, "n_band": pop.n_band,
            "n_sampled": len(pop.keys), "n_dropped": pop.n_dropped,
            "median_tokens": int(sorted(len(s) for s in pop.token_sets)[len(pop.token_sets) // 2]),
            "z_bar": z_bar, "z_bar_std": boot["z_bar"]["std"],
            "z_bar_ci95": 1.96 * boot["z_bar"]["std"],
            "z_bar_lo": boot["z_bar"]["lo"], "z_bar_hi": boot["z_bar"]["hi"],
        })
        _write_run_result(rd, rows[-1])
        print(f"[diversity] {rd.name} ({step}): band={pop.n_band} n={len(pop.keys)} "
              f"sim={z_bar:.4f} +- {1.96 * boot['z_bar']['std']:.4f}", flush=True)

    if rows:
        _print_table(rows, args)
    else:
        print("\n[diversity] no run for this target has a usable band -- nothing to compare")
    if skipped:
        print("\nnot screened:")
        for name, why in skipped:
            print(f"  {name:52s} {why}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"config": vars(args), "runs": rows, "skipped": skipped}, indent=2, default=str))
        _write_csv(out.with_suffix(".csv"), rows, args.sort_by)
        print(f"\n[diversity] -> {out}\n[diversity] -> {out.with_suffix('.csv')}")
    assert rows or skipped, "no runs found at all -- check --target / --runs-root"


def _write_run_result(run_dir: Path, row: dict) -> None:
    """`<run>/diversity/diversity_<h-step...>[_<criterion><tau>].json`, read by `aspd.eval.tables`."""
    suffix = "" if not row["threshold"] else f"_{row['criterion']}{row['threshold']:g}"
    out = run_dir / "diversity" / f"diversity_{row['step']}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, default=str))


def _band_size(db_path: Path, args, stats: dict | None = None) -> int:
    if stats is not None:
        name = f"{args.threshold:g}"
        return sum(1 for h in stats["components"].values()
                   if h[name][0] >= 5 and args.min_density <= h[name][1] <= args.max_density)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    n = conn.execute(
        "SELECT COUNT(*) FROM components WHERE n_activation_examples >= 5 "
        "AND firing_density BETWEEN ? AND ?", (args.min_density, args.max_density)
    ).fetchone()[0]
    conn.close()
    return n


_FIELDS = ["run", "target", "ci_mode", "criterion", "threshold", "step", "n_eligible", "n_band", "n_sampled", "n_dropped",
           "median_tokens", "z_bar", "z_bar_std", "z_bar_ci95", "z_bar_lo", "z_bar_hi"]


def _sort_key(sort_by: str):
    return lambda x: x["z_bar"]


def _write_csv(path: Path, rows: list[dict], sort_by: str = "z_bar") -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        for r in sorted(rows, key=_sort_key(sort_by)):
            w.writerow({
                k: (f"{r[k]:.6f}" if k.startswith("z_bar") else r[k])
                for k in _FIELDS
            })


def _print_table(rows: list[dict], args) -> None:
    print(f"\nband [{args.min_density:g}, {args.max_density:g}]  min_firings={args.min_firings}  "
          f"n_sample={args.n_sample}  seed={args.seed}  n_boot={args.n_boot}\n")
    print(f"{'run':46s}{'C':>8s}{'n':>6s}{'tokens':>8s}{'sim +- 95%':>24s}")
    for r in sorted(rows, key=_sort_key(args.sort_by)):
        print(f"{r['run']:46s}{r['n_band']:>8}{r['n_sampled']:>6}{r['median_tokens']:>8}"
              f"{r['z_bar']:>14.4f} +- {r['z_bar_ci95']:.4f}")
    print("\nC = components in the density band, n = components sampled, tokens = median token-set size.")
    print("sim = mean pairwise Jaccard overlap of token sets; +- is 1.96 x the bootstrap SE.")


if __name__ == "__main__":
    main()
