"""Harvest a trained decomposition: activation examples and firing statistics per component, per checkpoint."""

import argparse
import dataclasses
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal, override

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from aspd.harvest_patch import (
    install_component_token_stats,
    install_low_memory_token_stats,
    install_no_cooccurrence,
)
from aspd.lab_compat import widen_lab_config_parsing

_PD_RUN_ID_RE = re.compile(r"^p-[a-z0-9]{8}$")


def downstream_id_for(run_id: str) -> str:
    """Deterministic parse_wandb_run_path-valid id for a local run name (identity if already one)."""
    if _PD_RUN_ID_RE.match(run_id):
        return run_id
    return "p-" + hashlib.sha1(run_id.encode()).hexdigest()[:8]


def link_downstream_id(run_dir: Path) -> str:
    """Link `runs/<p-id>` to the run dir (the lab addresses runs by that id). Returns the id.

    Also links the id each of the run's harvests records, so a harvest made under another run name
    resolves to this run.
    """
    from param_decomp_lab.harvest.db import HarvestDB

    did = downstream_id_for(run_dir.name)
    ids = {did}
    for db in run_dir.glob("harvest/*/harvest.db"):
        recorded = HarvestDB(db, readonly=True).get_config_dict()["method_config"].get("wandb_path")
        if recorded and _PD_RUN_ID_RE.match(recorded):
            ids.add(recorded)
    for i in ids - {run_dir.name}:
        link = run_dir.parent / i
        if not (link.is_symlink() and link.resolve() == run_dir.resolve()):
            link.unlink(missing_ok=True)
            link.symlink_to(run_dir.name)  # relative: runs/<id> -> <run_dir.name>
    return did


def _local_adapter(run_dir: Path, run_id: str, step: int | None = None):
    """PDAdapter over a local run dir with an explicit run_id (skips the W&B-path parse)."""
    from functools import cached_property

    from param_decomp_lab.adapters.pd import PDAdapter
    from param_decomp_lab.experiments.lm.run import SavedLMRun

    class LocalPDAdapter(PDAdapter):
        def __init__(self) -> None:
            self._wandb_path = str(run_dir)  # SavedLMRun.from_path accepts a local run dir
            self._run_id = run_id

        @cached_property
        @override
        def pd_run(self) -> SavedLMRun:
            run = SavedLMRun.from_path(str(run_dir))
            if step is None:
                return run
            ckpt = run_dir / f"model_{step}.pth"
            assert ckpt.exists(), f"no checkpoint {ckpt}"
            return dataclasses.replace(run, checkpoint_path=ckpt)

        @override
        def dataloader(self, batch_size: int) -> DataLoader[Tensor]:
            from param_decomp_lab.experiments.lm.run import build_lm_loader

            cfg = self.pd_run.cfg
            return build_lm_loader(
                cfg.target, cfg.data, split="train", device="cpu",
                batch_size=batch_size, seed=cfg.pd.seed,
            )

    return LocalPDAdapter()


def step_subrun_id(step: int) -> str:
    return f"h-step{step:06d}"


def discover_steps(run_dir: Path) -> list[int]:
    steps = sorted(int(p.stem.removeprefix("model_")) for p in run_dir.glob("model_*.pth"))
    assert steps, f"no model_*.pth in {run_dir}"
    return steps


def run_local_harvest(run_dir: Path, step: int, n_batches: int = 400, batch_size: int = 16,
                      context_tokens_per_side: int = 20,
                      examples_per_component: int = 400,
                      activation_threshold: float = 0.0,
                      token_stats: Literal["topk", "full"] = "topk",
                      correlations: Literal["off", "full"] = "off") -> Path:
    from param_decomp_lab.harvest.config import HarvestConfig, ParamDecompHarvestConfig
    from param_decomp_lab.harvest.harvest_fn import make_harvest_fn
    from param_decomp_lab.harvest.pipeline import harvest
    from param_decomp_lab.harvest.schemas import get_harvest_subrun_dir

    assert (run_dir / "experiment_config.yaml").exists(), f"no run at {run_dir}"
    widen_lab_config_parsing()
    install_component_token_stats(token_stats, correlations=correlations)
    install_no_cooccurrence(
        correlations=correlations == "full", token_stats=token_stats == "full"
    )
    install_low_memory_token_stats()
    run_id = run_dir.name
    adapter = _local_adapter(run_dir, run_id, step)
    did = link_downstream_id(run_dir)
    config = HarvestConfig(
        method_config=ParamDecompHarvestConfig(wandb_path=did, activation_threshold=activation_threshold),
        n_batches=n_batches,
        batch_size=batch_size,
        activation_context_tokens_per_side=context_tokens_per_side,
        activation_examples_per_component=examples_per_component,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = get_harvest_subrun_dir(adapter.decomposition_id, step_subrun_id(step))
    print(f"[harvest] run={run_id} ckpt={adapter.pd_run.checkpoint_path.name} "
          f"n_batches={n_batches} thresh={activation_threshold} dev={device}\n"
          f"  downstream id {did} (autointerp DID={did}; app run local/local/{did})\n"
          f"  -> {out}", flush=True)
    harvest(
        layers=adapter.layer_activation_sizes,
        vocab_size=adapter.vocab_size,
        dataloader=adapter.dataloader(config.batch_size),
        harvest_fn=make_harvest_fn(device, config.method_config, adapter),
        config=config,
        output_dir=out,
        rank_world_size=None,
        device=device,
    )
    (out / "harvest_meta.json").write_text(json.dumps({
        "step": step,
        "checkpoint": adapter.pd_run.checkpoint_path.name,
        "activation_threshold": activation_threshold,
        "run_dir": str(run_dir),
        "created": datetime.now().isoformat(),
    }, indent=2))
    print(f"[harvest] DONE -> {out / 'harvest.db'}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="the run directory")
    ap.add_argument("--n-batches", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--context-tokens-per-side", type=int, default=20,
                    help="activation-example half-window; reservoir is [C, examples, 2*this+1] on GPU")
    ap.add_argument("--examples-per-component", type=int, default=400)
    ap.add_argument("--steps", nargs="+", default=None,
                    help="checkpoint steps to harvest into h-step<step>/ subruns (ints or 'all'); "
                         "default: the latest checkpoint")
    ap.add_argument("--activation-threshold", type=float, default=0.0,
                    help="CI firing threshold stamped into the harvest (VPD paper uses 0.0 or 0.1)")
    ap.add_argument("--token-stats", choices=("topk", "full"), default="topk",
                    help="'full' writes token_stats.pt -- REQUIRED by autointerp, every strategy")
    ap.add_argument("--correlations", choices=("off", "full"), default="off",
                    help="'full' writes component_correlations.pt ([C, C]) -- pd-graph-interp needs it")
    ap.add_argument("--force", action="store_true",
                    help="re-harvest steps whose h-step*/harvest.db already exists")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    kwargs = dict(n_batches=args.n_batches, batch_size=args.batch_size,
                  context_tokens_per_side=args.context_tokens_per_side,
                  examples_per_component=args.examples_per_component,
                  activation_threshold=args.activation_threshold,
                  token_stats=args.token_stats, correlations=args.correlations)

    if args.steps is None:
        steps = discover_steps(run_dir)[-1:]
    elif args.steps == ["all"]:
        steps = discover_steps(run_dir)
    else:
        steps = [int(s) for s in args.steps]
    for step in steps:
        existing = run_dir / "harvest" / step_subrun_id(step) / "harvest.db"
        if existing.exists() and not args.force:
            print(f"[harvest] step {step}: {existing} exists, skipping (--force to redo)", flush=True)
            continue
        run_local_harvest(run_dir, step=step, **kwargs)
        torch.cuda.empty_cache()  # 5 sequential harvests: free each step's model/reservoir statics


if __name__ == "__main__":
    main()
