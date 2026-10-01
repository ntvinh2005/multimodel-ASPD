"""Autointerp a harvested decomposition run's components with the lab's `run_interpret` worker.

The worker rebuilds the run through the lab's config parser, so the widened parse
(`aspd.lab_compat`) is applied in this process first; and it builds its LLM client through
`param_decomp_lab.autointerp.providers.create_provider`, which this CLI points at the
`aspd.eval.judge` endpoint for the duration of the call.

Above `--cap` components, a seeded random subset is labeled.

    python -m aspd.cli.autointerp --rid <run id> [--hsub h-YYYYMMDD_HHMMSS] [--cap 500]
"""

import argparse
import json
import random
import sys
from pathlib import Path

from aspd.cli.harvest import link_downstream_id
from aspd.eval.judge import OpenAICompatProvider, add_judge_args, judge_config_from_args
from aspd.lab_compat import widen_lab_config_parsing
from aspd.paths import RUNS_DIR

DEFAULT_STRATEGY = {
    "type": "compact_skeptical",
    "max_examples": 30,
    "include_pmi": True,
    "include_dataset_description": False,
    "label_max_words": 8,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rid", required=True, help="run dir name under the runs directory")
    ap.add_argument("--hsub", default=None,
                    help="harvest subrun id (default: latest h-* under the run's harvest/)")
    ap.add_argument("--config", type=Path, default=None,
                    help="AutointerpConfig JSON (default: compact_skeptical, 8-word labels)")
    ap.add_argument("--cap", type=int, default=500, help="max components to interpret")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (deterministic subset)")
    add_judge_args(ap)
    args = ap.parse_args()

    import param_decomp_lab.autointerp.providers as providers
    from param_decomp_lab.autointerp.schemas import get_autointerp_dir
    from param_decomp_lab.harvest.repo import HarvestRepo
    from param_decomp_lab.harvest.schemas import get_harvest_dir

    did = link_downstream_id(RUNS_DIR / args.rid)
    hsub = args.hsub
    if hsub is None:
        subruns = sorted(d.name for d in get_harvest_dir(did).glob("h-*"))
        assert subruns, f"no harvest subruns under {get_harvest_dir(did)}; run aspd.cli.harvest first"
        hsub = subruns[-1]

    judge = judge_config_from_args(args, structured=True)
    cfg = json.loads(args.config.read_text()) if args.config else {"template_strategy": DEFAULT_STRATEGY}
    # A lab-parseable LLM entry carrying the pacing the worker reads; the client itself is ours.
    cfg["llm"] = {"type": "openai", "model": judge.model, "max_concurrent": judge.max_concurrent,
                  "max_requests_per_minute": judge.max_requests_per_minute}
    keys = HarvestRepo(did, subrun_id=hsub, readonly=True).get_component_keys()
    total = len(keys)
    if cfg.get("component_keys_path") is not None:
        print(f"[autointerp] config pins component_keys_path; not sampling ({total} components)",
              file=sys.stderr)
    elif total > args.cap:
        chosen = sorted(random.Random(args.seed).sample(keys, args.cap))
        keys_out = get_autointerp_dir(did) / f"sampled_keys_{hsub}.txt"
        keys_out.parent.mkdir(parents=True, exist_ok=True)
        keys_out.write_text("\n".join(chosen) + "\n")
        cfg["component_keys_path"] = str(keys_out.resolve())
        print(f"[autointerp] {total} components > cap {args.cap}: sampled {args.cap} "
              f"(seed {args.seed}) -> {keys_out}", file=sys.stderr)
    else:
        print(f"[autointerp] {total} components <= cap {args.cap}: labeling all", file=sys.stderr)

    widen_lab_config_parsing()
    providers.create_provider = lambda _config: OpenAICompatProvider(judge)
    from param_decomp_lab.autointerp.scripts.run_interpret import main as run_interpret_main

    run_interpret_main(did, cfg, hsub)


if __name__ == "__main__":
    main()
