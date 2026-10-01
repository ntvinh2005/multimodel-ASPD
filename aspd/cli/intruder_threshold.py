"""Intruder scores with firing redefined as g_{t,c} > tau, for several tau."""

import argparse
import json
from pathlib import Path


def main() -> None:
    from aspd.cli.harvest import discover_steps, step_subrun_id
    from aspd.config import LMInterpExperimentConfig
    from aspd.eval.autointerp_db import sample_keys
    from aspd.eval.harvest_threshold import threshold_stats
    from aspd.eval.intruder_db import intruder_harvest_db
    from aspd.eval.judge import OpenAICompatProvider, add_judge_args, judge_config_from_args
    from aspd.lab_compat import widen_lab_config_parsing

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="the run directory")
    ap.add_argument("--steps", nargs="+", default=["all"],
                    help="checkpoint steps with existing h-step<step>/ harvests (ints, 'all', or "
                         "'last' for the newest checkpoint -- which is the step the summary table "
                         "reports, and differs per run, so a multi-run launcher wants 'last')")
    ap.add_argument("--ci-thresholds", nargs="+", type=float, default=[0.01, 0.1],
                    help="CI firing thresholds to score at; 0 is the existing measurement and is "
                         "skipped rather than re-scored under a second score_type")
    ap.add_argument("--criterion", choices=("ci", "act"), default="ci",
                    help="what the threshold applies to: `ci` the causal importance (varies on VPD "
                         "runs), `act` |component activation| / its own peak (for PD Transcoder and "
                         "ASPD, whose gate is 0/1)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds each threshold's own draw from its own eligible population")
    ap.add_argument("--n-subsample", type=int, default=200,
                    help="components to score per threshold, drawn from those eligible AT it")
    add_judge_args(ap)
    ap.add_argument("--n-real", type=int, default=4)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--density-tolerance", type=float, default=0.05)
    ap.add_argument("--cost-limit-usd", type=float, default=None)
    ap.add_argument("--stats-only", action="store_true",
                    help="build the cached per-threshold stats, report how many keys survive at "
                         "each, then stop without calling a judge")
    ap.add_argument("--rebuild-stats", action="store_true",
                    help="recompute the cached ci_threshold_stats.json rather than reusing it")
    args = ap.parse_args()

    widen_lab_config_parsing()
    run_dir = Path(args.run_dir)
    cfg = LMInterpExperimentConfig.from_file(run_dir / "experiment_config.yaml")
    tokenizer_name = cfg.data.tokenizer_name

    taus = sorted({t for t in args.ci_thresholds if t > 0.0})
    assert taus, "every --ci-thresholds value is <= 0; those ARE `aspd.cli.intruder`, which owns them"

    if args.steps == ["all"]:
        steps = discover_steps(run_dir)
    elif args.steps == ["last"]:
        steps = [max(discover_steps(run_dir))]
    else:
        steps = [int(s) for s in args.steps]
    subruns = {step: run_dir / "harvest" / step_subrun_id(step) for step in steps}
    for step, subrun in subruns.items():
        assert (subrun / "harvest.db").exists(), (
            f"no harvest for step {step} at {subrun}; run "
            f"`python -m aspd.cli.harvest --run-dir {run_dir} --steps {' '.join(map(str, steps))}` first"
        )

    per_step: dict[int, dict] = {}
    for step, subrun in subruns.items():
        per_step[step] = threshold_stats(
            subrun, taus, criterion=args.criterion, min_examples=args.n_real + 1,
            rebuild=args.rebuild_stats,
        )["components"]

    eligible: dict[float, list[str]] = {}
    for tau in taus:
        name = f"{tau:g}"
        per_step_eligible = {
            step: {k for k, held in held_by_key.items() if held[name][0] >= args.n_real + 1}
            for step, held_by_key in per_step.items()
        }
        intersection = set.intersection(*per_step_eligible.values())

        print(f"[intruder-ci] {args.criterion} tau={name}: {len(intersection)} components "
              f"eligible at every step "
              f"{steps}; per step "
              + ", ".join(f"{s}: {len(v)}/{len(per_step[s])}"
                          for s, v in per_step_eligible.items()), flush=True)
        if args.stats_only:
            n = min(args.n_subsample, len(intersection))
            print(f"    would sample {n} -> {n * args.n_trials} trials per step (not persisted)")
            continue

        keys_path = (run_dir / "harvest"
                     / f"intruder_keys_seed{args.seed}_{args.criterion}{name}.json")
        if keys_path.exists():
            saved = json.loads(keys_path.read_text())
            assert (saved["seed"] == args.seed and saved["n_subsample"] == args.n_subsample
                    and saved["ci_threshold"] == tau
                    and saved.get("criterion", "ci") == args.criterion), (
                f"{keys_path} was sampled with seed={saved['seed']}, "
                f"n_subsample={saved['n_subsample']}, ci_threshold={saved['ci_threshold']}; "
                "delete it to re-sample under other settings"
            )
            eligible[tau] = saved["keys"]
            print(f"[intruder-ci] tau={name}: reusing {len(eligible[tau])} fixed keys from "
                  f"{keys_path} (sampled over steps {saved['steps']})", flush=True)
        else:
            eligible[tau] = sample_keys(sorted(intersection), args.n_subsample, args.seed)
            keys_path.write_text(json.dumps({
                "ci_threshold": tau,
                "criterion": args.criterion,
                "steps": steps,
                "seed": args.seed,
                "n_subsample": args.n_subsample,
                "per_step_eligible_counts": {str(s): len(v) for s, v in per_step_eligible.items()},
                "n_intersection": len(intersection),
                "n_eligible_at_tau0": {str(s): len(v) for s, v in per_step.items()},
                "keys": eligible[tau],
            }, indent=2))
            print(f"[intruder-ci] tau={name}: sampled {len(eligible[tau])} of {len(intersection)} "
                  f"-> {keys_path}", flush=True)
        print(f"    {len(eligible[tau]) * args.n_trials} trials per step")

    if args.stats_only:
        print("[intruder-ci] --stats-only: no judge called, no sample persisted")
        return

    llm_config = judge_config_from_args(args)
    provider = OpenAICompatProvider(llm_config)

    for tau in taus:
        for step, subrun in subruns.items():
            print(f"[intruder-ci] === {args.criterion} tau={tau:g} step {step} ===", flush=True)
            intruder_harvest_db(
                subrun,
                tokenizer_name,
                llm_config=llm_config,
                n_real=args.n_real,
                n_trials=args.n_trials,
                density_tolerance=args.density_tolerance,
                keys=eligible[tau],
                ci_threshold=tau,
                criterion=args.criterion,
                cost_limit_usd=args.cost_limit_usd,
                provider=provider,
            )


if __name__ == "__main__":
    main()
