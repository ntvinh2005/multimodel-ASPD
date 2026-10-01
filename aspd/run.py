"""Training entry point: one decomposition run of any method in the paper.

Procedure: parse the config, derive the arm (`aspd.arms`), build the target model and data
loaders, install the CI functions and the component parameterization, calibrate the CI function
(ASPD only), build the lab `Trainer`, and train. Checkpoints, metrics and `provenance.json` go to
the run directory.
"""

import argparse
import dataclasses
import json
import os
from itertools import islice
from pathlib import Path
from typing import Any

from aspd.losses import ComponentAliveTrackerConfig
from param_decomp.distributed import is_main_process
from param_decomp.log import logger
from param_decomp.metrics.base import Metric
from param_decomp.optimize import Trainer
from param_decomp.run_sink import RunSink
from param_decomp.training_state import TrainingState
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.distributed import get_device, init_distributed
from param_decomp_lab.experiments.lm.run import (
    _build_eval_loop,
    build_lm_loader,
    build_target,
    make_run_batch,
)
from param_decomp_lab.experiments.utils import init_pd_run
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.seed import set_seed

from aspd.arms import INJECTED_LOSS_CONFIGS, arm_summary, derive_arm_name
from aspd.component_setup import install_component_parameterization
from aspd.ci.setup import (
    calibrate_ci_scale,
    install_ci_fns,
    seed_ci_fn,
    uses_custom_ci_fn,
)
from aspd.config import LMInterpExperimentConfig
from aspd.loader_patch import install_bos_for_tokenizers_that_add_it
from aspd.resume import apply_resume, resolve_resume_step, save_loader_state
from aspd.sites import loader_to_token_stream

# Token budget of the CI-function calibration loader (ASPD reads at most `CALIBRATION_BATCHES` of it).
CALIB_TOKENS = 500_000


def calib_batches(cfg: LMInterpExperimentConfig, override: int | None) -> int:
    """Eval batches needed for `CALIB_TOKENS`, or `override` when one is given."""
    if override is not None:
        return override
    batch = cfg.eval.batch_size if cfg.eval is not None else cfg.pd.batch_size
    per_batch = batch * cfg.data.max_seq_len
    return max(1, -(-CALIB_TOKENS // per_batch))  # ceil


def _all_coeffs(cfg: LMInterpExperimentConfig) -> dict[str, float]:
    """`{term: coeff}` over EVERY loss entry, core's included, with repeats disambiguated."""
    seen: dict[str, int] = {}
    for entry in cfg.pd.loss_metrics:
        seen[entry.type] = seen.get(entry.type, 0) + 1
    repeated = {t for t, n in seen.items() if n > 1}

    out: dict[str, float] = {}
    for index, entry in enumerate(cfg.pd.loss_metrics):
        coeff = getattr(entry, "coeff", None)
        if coeff is None:
            continue
        key = entry.type
        if entry.type in repeated:
            key = f"{entry.type}[{getattr(entry, 'name', None) or index}]"
        out[key] = float(coeff)
    return out


def inject_runtime_objects(cfg: LMInterpExperimentConfig, module: str) -> None:
    """Fill `module` and `train_diag_every` on every loss entry that needs them."""

    def _set_module(entry) -> None:
        if not entry.module:
            object.__setattr__(entry, "module", module)

    for entry in cfg.pd.loss_metrics:
        if isinstance(entry, INJECTED_LOSS_CONFIGS):
            _set_module(entry)
            object.__setattr__(entry, "train_diag_every", cfg.cadence.train_log_every)
        elif isinstance(entry, ComponentAliveTrackerConfig):
            _set_module(entry)


class TrainDiagSink:
    """A `RunSink` that merges the losses' per-component diagnostics into the TRAIN stream."""

    def __init__(
        self,
        inner: RunSink,
        metrics: dict[str, Metric[Any]],
        train_loader: Any = None,
        loader_state_dir: Path | None = None,
    ) -> None:
        self._inner = inner
        self._metrics = metrics
        self._train_loader = train_loader
        self._loader_state_dir = loader_state_dir

    @property
    def out_dir(self) -> Path | None:
        return getattr(self._inner, "out_dir", None)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if any(key.startswith("train/") for key in metrics):
            for metric in self._metrics.values():
                stashed = getattr(metric, "pop_train_log", dict)()
                for key, value in stashed.items():
                    metrics[f"train/{metric.log_namespace}/{metric.instance_key}/{key}"] = (
                        value.item()
                    )
        self._inner.log(metrics, step)

    def console(self, *lines: str) -> None:
        self._inner.console(*lines)

    def checkpoint(self, snapshot: TrainingState) -> None:
        self._inner.checkpoint(snapshot)
        if self._train_loader is not None:
            save_loader_state(
                self._loader_state_dir or self.out_dir, snapshot.step, self._train_loader
            )

    def finish(self) -> None:
        self._inner.finish()


def _wandb_log_data_only() -> None:
    """Stop the harness from uploading multi-GB checkpoints to wandb — log metrics/tables only."""
    import wandb

    if getattr(wandb.save, "_p2_data_only", False):
        return
    _real_save = wandb.save

    def _save(glob_str=None, *args, **kwargs):
        if glob_str is not None and str(glob_str).endswith(".pth"):
            return []  # keep checkpoints local; never upload model weights to wandb
        return _real_save(glob_str, *args, **kwargs)

    _save._p2_data_only = True
    wandb.save = _save


def main() -> None:
    _wandb_log_data_only()  # log metrics/tables to wandb, but never upload the multi-GB checkpoints
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--tags", default=None)
    ap.add_argument(
        "--keep-last-n", type=int, default=None,
        help="override cadence.keep_last_n_checkpoints. A checkpoint PAIR is model_<step>.pth + "
             "training_<step>.pth, and the training half carries optimizer + metric state -- on "
             "the Gemma runs that is ~14.5 GB against ~11.3 GB of weights, so keeping 5 is ~129 GB "
             "per run. 1 keeps only the newest pair, which is still resumable; what it gives up is "
             "resuming from an INTERMEDIATE step. Left unset the config's own value wins, so this "
             "does not silently change any existing arm.",
    )
    ap.add_argument(
        "--probe-calib-batches",
        type=int,
        default=None,
        help=f"eval-split batches for the CI-function calibration pass; derived from "
        f"{CALIB_TOKENS:,} tokens and the eval batch size unless given. Unused by VPD.",
    )
    ap.add_argument(
        "--resume",
        default="off",
        help="'off' (default), 'auto' -- newest complete checkpoint in the run dir, starting "
        "fresh when there is none, so one launch line serves both a first submission and a "
        "requeue -- or an explicit step. Needs --run-id: without it every launch allocates a "
        "new run dir and there is nothing to resume from.",
    )
    args = ap.parse_args()

    cfg = LMInterpExperimentConfig.from_file(Path(args.config_path))
    if args.keep_last_n is not None:
        logger.info(
            f"--keep-last-n {args.keep_last_n} overrides "
            f"cadence.keep_last_n_checkpoints={cfg.cadence.keep_last_n_checkpoints}"
        )
    arm = derive_arm_name(cfg)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    set_seed(cfg.pd.seed)
    device = get_device()
    assert not (cfg.runtime.dp is not None and dist_state is None), (
        f"config declares runtime.dp={cfg.runtime.dp} but this process is not distributed. "
        f"Launch it with `torchrun --nproc_per_node={cfg.runtime.dp}` (the launcher's DP= knob), "
        "or drop `dp` from the config."
    )
    cfg = cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(
                update={
                    "device": device,
                    "dp": dist_state.world_size if dist_state is not None else None,
                }
            )
        }
    )

    target_model = build_target(cfg.target)
    install_bos_for_tokenizers_that_add_it()
    train_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        dist_state=dist_state,
        seed=cfg.pd.seed,
    )

    module = cfg.pd.decomposition_targets[0].module_pattern

    # Before any forward: the ASPD calibration pass runs the target on device batches.
    target_model = target_model.to(device)

    inject_runtime_objects(cfg, module)

    # Must precede `Trainer(...)`, which is where `ComponentModel` calls `make_ci_fn_wrapper`.
    install_ci_fns()
    parameterization = install_component_parameterization(cfg)
    if is_main_process() and parameterization != "vpd":
        logger.info(f"component parameterization: {parameterization}")
    ci_scale, ci_scale_stats = None, {}
    if uses_custom_ci_fn(cfg):
        n_calib_batches = calib_batches(cfg, args.probe_calib_batches)
        calib_loader = build_lm_loader(
            cfg.target,
            cfg.data,
            split="eval",
            device=device,
            batch_size=cfg.eval.batch_size if cfg.eval is not None else cfg.pd.batch_size,
            dist_state=dist_state,
            seed=cfg.pd.seed,
        )
        ci_scale, ci_scale_stats = calibrate_ci_scale(
            cfg,
            target_model,
            module,
            islice(loader_to_token_stream(calib_loader), n_calib_batches),
            device,
        )
        if is_main_process():
            logger.info(
                f"{cfg.pd.ci_config.mode} per-latent scale calibrated over {n_calib_batches} eval "
                f"batches: {ci_scale_stats}"
            )

    eval_loop = _build_eval_loop(cfg, device, dist_state)

    tags = ",".join(t for t in (args.tags, f"arm={arm}") if t)

    run_dir = PARAM_DECOMP_OUT_DIR / "runs" / args.run_id if args.run_id else None
    resume_step: int | None = None
    if args.resume != "off":
        assert run_dir is not None, "--resume needs --run-id to know which run dir to look in"
        resume_step = resolve_resume_step(args.resume, run_dir)
        if resume_step is None:
            logger.info(f"--resume {args.resume}: no complete checkpoint in {run_dir}; fresh start")
        else:
            os.environ.setdefault("WANDB_RESUME", "allow")
            logger.info(f"--resume {args.resume}: resuming {args.run_id} from step {resume_step}")

    sink = init_pd_run(cfg, group=args.group, tags=tags, run_id=args.run_id)
    if args.keep_last_n is not None:
        sink = dataclasses.replace(sink, keep_last_n_checkpoints=args.keep_last_n)
        logger.info(
            f"EFFECTIVE keep_last_n_checkpoints={args.keep_last_n} (experiment_config.yaml still "
            f"records the config's {cfg.cadence.keep_last_n_checkpoints}; wandb locks it on resume)"
        )
    if is_main_process():
        logger.info(f"arm={arm}\n{arm_summary(cfg)}")
    if sink.out_dir is not None:
        (sink.out_dir / "provenance.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "module": module,
                    "ci_fn": cfg.pd.ci_config.mode,
                    # What the SINK enforces, which is not always what the config says.
                    "keep_last_n_checkpoints_effective": sink.keep_last_n_checkpoints,
                    # The CI function's calibration statistics (empty on VPD).
                    "ci_calibration": ci_scale_stats,
                    "coeffs": {
                        e.type: e.coeff
                        for e in cfg.pd.loss_metrics
                        if isinstance(e, INJECTED_LOSS_CONFIGS)
                    },
                    # Every term's coefficient, core's included.
                    "all_coeffs": _all_coeffs(cfg),
                },
                indent=2,
            )
        )
    try:
        trainer = Trainer(
            target_model=target_model,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=recon_loss_kl,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        if ci_scale is not None:
            seed_ci_fn(cfg, trainer.component_model, ci_scale)
        train_stream = train_loader
        if resume_step is not None:
            assert run_dir is not None
            train_stream = apply_resume(trainer, train_loader, run_dir, resume_step)
        diag_sink = TrainDiagSink(sink, trainer.loss_metrics, train_loader, run_dir)
        trainer.run(
            train_stream,
            diag_sink,
            cfg.cadence,
            eval_loop,
        )
    finally:
        sink.finish()


if __name__ == "__main__":
    main()
