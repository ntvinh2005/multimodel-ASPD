"""Interpretability (intruder score) of a run's components, per checkpoint."""

import argparse
import json
from pathlib import Path


def main() -> None:
    from param_decomp_lab.harvest.db import HarvestDB

    from aspd.cli.harvest import discover_steps, step_subrun_id
    from aspd.config import LMInterpExperimentConfig
    from aspd.eval.autointerp_db import sample_keys
    from aspd.eval.intruder_db import (
        eligible_density_entries,
        intruder_harvest_db,
    )
    from aspd.eval.judge import OpenAICompatProvider, add_judge_args, judge_config_from_args
    from aspd.lab_compat import widen_lab_config_parsing

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="the run directory")
    ap.add_argument("--steps", nargs="+", default=["all"],
                    help="checkpoint steps with existing h-step<step>/ harvests (ints or 'all')")
    ap.add_argument("--n-subsample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    add_judge_args(ap)
    ap.add_argument("--n-real", type=int, default=4)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--density-tolerance", type=float, default=0.05)
    ap.add_argument("--cost-limit-usd", type=float, default=None)
    args = ap.parse_args()

    widen_lab_config_parsing()
    run_dir = Path(args.run_dir)
    cfg = LMInterpExperimentConfig.from_file(run_dir / "experiment_config.yaml")
    tokenizer_name = cfg.data.tokenizer_name

    steps = discover_steps(run_dir) if args.steps == ["all"] else [int(s) for s in args.steps]
    subruns = {step: run_dir / "harvest" / step_subrun_id(step) for step in steps}
    for step, subrun in subruns.items():
        assert (subrun / "harvest.db").exists(), (
            f"no harvest for step {step} at {subrun}; run "
            f"`python -m aspd.cli.harvest --run-dir {run_dir} --steps {' '.join(map(str, steps))}` first"
        )

    llm_config = judge_config_from_args(args)
    provider = OpenAICompatProvider(llm_config)

    keys_path = run_dir / "harvest" / f"intruder_keys_seed{args.seed}.json"
    if keys_path.exists():
        saved = json.loads(keys_path.read_text())
        assert saved["seed"] == args.seed and saved["n_subsample"] == args.n_subsample, (
            f"{keys_path} was sampled with seed={saved['seed']}, n_subsample={saved['n_subsample']}; "
            "delete it to re-sample under different settings"
        )
        fixed = saved["keys"]
        print(f"[intruder] reusing {len(fixed)} fixed keys from {keys_path} "
              f"(sampled over steps {saved['steps']})", flush=True)
    else:
        per_step_eligible: dict[int, set[str]] = {}
        for step, subrun in subruns.items():
            db = HarvestDB(subrun / "harvest.db", readonly=True)
            per_step_eligible[step] = {
                k for k, _ in eligible_density_entries(db, min_examples=args.n_real + 1)
            }
            db.close()
        intersection = set.intersection(*per_step_eligible.values())
        assert intersection, "no component is eligible at every selected step"
        fixed = sample_keys(sorted(intersection), args.n_subsample, args.seed)
        keys_path.write_text(json.dumps({
            "steps": steps,
            "seed": args.seed,
            "n_subsample": args.n_subsample,
            "per_step_eligible_counts": {str(s): len(v) for s, v in per_step_eligible.items()},
            "n_intersection": len(intersection),
            "keys": fixed,
        }, indent=2))
        print(f"[intruder] sampled {len(fixed)} of {len(intersection)} intersection-eligible "
              f"components -> {keys_path}", flush=True)

    for step, subrun in subruns.items():
        print(f"[intruder] === step {step} ===", flush=True)
        intruder_harvest_db(
            subrun,
            tokenizer_name,
            llm_config=llm_config,
            n_real=args.n_real,
            n_trials=args.n_trials,
            density_tolerance=args.density_tolerance,
            keys=fixed,
            cost_limit_usd=args.cost_limit_usd,
            provider=provider,
        )


if __name__ == "__main__":
    main()
