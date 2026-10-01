"""Train (or load) the SAEs for one config ahead of the evaluations."""

import argparse
from pathlib import Path

from aspd.sae.config import SAERunConfig
from aspd.sites import (
    SITE_BUILDERS,
    adopt_output_dictionary,
    loader_to_token_stream,
    sites_for_module,
    sites_for_run,
    stamped_input_take,
    train_or_load,
)

__all__ = [
    "SITE_BUILDERS",
    "adopt_output_dictionary",
    "loader_to_token_stream",
    "sites_for_module",
    "sites_for_run",
    "stamped_input_take",
    "train_or_load",
]

def main() -> None:
    """Pretrain the SAE pair for one config, ahead of any decomposition run."""
    from param_decomp_lab.distributed import get_device
    from param_decomp_lab.experiments.lm.run import build_lm_loader, build_target

    from aspd.loader_patch import install_bos_for_tokenizers_that_add_it
    from aspd.config import LMInterpExperimentConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("sae_config", help="configs/sae/<target>.yaml")
    args = ap.parse_args()

    sae_cfg = SAERunConfig.from_file(args.sae_config)
    cfg = LMInterpExperimentConfig.from_file(sae_cfg.experiment_config)
    device = get_device()
    model = build_target(cfg.target).to(device)
    install_bos_for_tokenizers_that_add_it()
    loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        seed=cfg.pd.seed,
    )

    train_or_load(
        model,
        cfg.pd.decomposition_targets[0].module_pattern,
        loader_to_token_stream(loader),
        Path(sae_cfg.sae_dir),
        device=device,
        sae_cfg=sae_cfg,
    )


if __name__ == "__main__":
    main()
