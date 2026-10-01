"""Dataset attribution for one target-layer shard of a model-wide run."""

import argparse
from pathlib import Path

from aspd.paths import RUNS_DIR


def target_layers(run_dir: Path) -> list[str]:
    """Every decomposed module, in config order. The shard axis."""
    from aspd.config import LMInterpExperimentConfig

    cfg = LMInterpExperimentConfig.from_file(run_dir / "experiment_config.yaml")
    return [t.module_pattern for t in cfg.pd.decomposition_targets]


def _unembed_path(run_dir: Path) -> str:
    """The model's unembed module path (`lm_head` on GPT-2), from the topology."""
    from param_decomp_lab.experiments.lm.run import build_target
    from param_decomp_lab.topology import TransformerTopology

    from aspd.config import LMInterpExperimentConfig

    cfg = LMInterpExperimentConfig.from_file(run_dir / "experiment_config.yaml")
    return TransformerTopology(build_target(cfg.target)).path_schema.unembed_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-root", default=str(RUNS_DIR))
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--n-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--hsub", default=None,
                    help="harvest subrun id (default: latest h-* under the run's harvest/)")
    ap.add_argument("--merge", action="store_true", help="merge finished shards and exit")
    args = ap.parse_args()

    from aspd.analysis.circuits.sharding import (
        install_optional_unembed,
        install_target_shard,
        install_union_merge,
    )
    from aspd.lab_compat import widen_lab_config_parsing

    widen_lab_config_parsing()
    install_union_merge()
    install_optional_unembed()

    run_dir = Path(args.runs_root) / args.run
    out_dir = run_dir / "dataset_attributions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        from param_decomp_lab.dataset_attributions.pipeline import merge_attributions

        merge_attributions(out_dir)
        print(f"[attr] merged -> {out_dir / 'dataset_attributions.pt'}")
        return

    assert args.shard is not None and args.n_shards is not None, "--shard and --n-shards required"
    layers = target_layers(run_dir)
    mine = layers[args.shard :: args.n_shards]
    assert mine, f"shard {args.shard} of {args.n_shards} covers no layers ({len(layers)} total)"
    if args.shard == 0:
        mine = mine + [_unembed_path(run_dir)]
    print(f"[attr] shard {args.shard}/{args.n_shards}: {len(mine)} target layers -> {mine}")
    install_target_shard(mine)

    hsub = args.hsub
    if hsub is None:
        subs = sorted(d.name for d in (run_dir / "harvest").glob("h-*"))
        assert subs, f"no harvest under {run_dir}/harvest -- run slurm/harvest.sbatch first"
        hsub = subs[-1]
    print(f"[attr] harvest subrun {hsub}")

    from param_decomp_lab.dataset_attributions.config import DatasetAttributionConfig
    from param_decomp_lab.dataset_attributions.pipeline import harvest_attributions

    from aspd.cli.harvest import downstream_id_for

    did = downstream_id_for(args.run)
    link = run_dir.parent / did
    assert link.exists(), (
        f"no run alias {link} -- it is created by the harvest. Run slurm/harvest.sbatch first."
    )
    print(f"[attr] run reference {did}")
    config = DatasetAttributionConfig(
        wandb_path=did, n_batches=args.n_batches, batch_size=args.batch_size
    )
    shard_dir = out_dir / f"_shard_{args.shard}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    harvest_attributions(
        config=config,
        output_dir=shard_dir,
        harvest_subrun_id=hsub,
        rank=0,
        world_size=1,
    )
    produced = shard_dir / "worker_states" / "dataset_attributions_rank_0.pt"
    assert produced.exists(), f"shard produced no file at {produced}"
    final_dir = out_dir / "worker_states"
    final_dir.mkdir(parents=True, exist_ok=True)
    final = final_dir / f"dataset_attributions_rank_{args.shard}.pt"
    produced.replace(final)
    shard_dir.joinpath("worker_states").rmdir()
    shard_dir.rmdir()
    print(f"[attr] shard {args.shard} done -> {final.name}")


if __name__ == "__main__":
    main()
