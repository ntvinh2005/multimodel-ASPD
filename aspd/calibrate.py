"""Choose coefficients for this package's loss terms from gradient norms at step 0.

After the faithfulness warmup, on `--n-batches` batches (median):

    for each other term L_i:  N_i = || grad(coeff_i * L_i) ||   (as configured, e.g. VPD's losses)
    for each own term L_x:    R_x = || grad(L_x) ||             (L_internal, L_act, AuxK; coeff 1)

    targets T = 0.1 * min_i N_i, ..., 10 * max_i N_i in steps of 10x;  lambda_x(T) = T / R_x

reported separately for the component parameters and the CI-function parameters. The warmup ramp
is disabled for the measurement (it is 0 at step 0).

    python -m aspd.calibrate configs/gpt2/vpd_internal.yaml --out calibration/gpt2_vpd_internal.json
"""

import argparse
import json
import math
from itertools import islice
from pathlib import Path

import torch
from param_decomp.faithfulness_warmup import run_faithfulness_warmup
from param_decomp.log import logger
from param_decomp.optimize import Trainer, _build_metric_context
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.batch_and_loss_fns import make_run_batch, recon_loss_kl
from param_decomp_lab.distributed import get_device
from param_decomp_lab.experiments.lm.run import build_lm_loader, build_target
from param_decomp_lab.seed import set_seed

from aspd.arms import INJECTED_LOSS_CONFIGS, assert_config
from aspd.ci.setup import (
    calibrate_ci_scale,
    install_ci_fns,
    seed_ci_fn,
    uses_custom_ci_fn,
)
from aspd.component_setup import install_component_parameterization
from aspd.config import LMInterpExperimentConfig
from aspd.run import calib_batches, inject_runtime_objects
from aspd.sites import loader_to_token_stream

GROUPS = ("components", "ci_fn")

NEW_TERMS = {c.model_fields["type"].default for c in INJECTED_LOSS_CONFIGS}


def _group_params(trainer: Trainer, group: str):
    return (
        trainer._components_optimizer_named_params()
        if group == "components"
        else trainer._ci_fn_optimizer_named_params()
    )


def group_grad_norm(trainer: Trainer, group: str) -> float:
    """L2 norm of the currently-accumulated gradient over one parameter group."""
    total = torch.zeros((), device="cpu")
    for _, p in _group_params(trainer, group):
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().cpu()
    return total.sqrt().item()


def group_is_trainable(trainer: Trainer, group: str) -> bool:
    """Whether ANY parameter in the group can receive a gradient at all."""
    return any(p.requires_grad for _, p in _group_params(trainer, group))


def measure(trainer: Trainer, ctx_builder, n_batches: int) -> dict[str, dict[str, float]]:
    """Per-loss, per-group gradient norms, MEDIANED over `n_batches` independent draws."""
    samples: dict[str, dict[str, list[float]]] = {}
    for metric in trainer.loss_metrics.values():
        metric.reset()

    for i in range(n_batches):
        ctx = ctx_builder(i)
        for name, metric in trainer.loss_metrics.items():
            with bf16_autocast(enabled=trainer.runtime_config.autocast_bf16):
                loss = metric.update(ctx)
            if loss is None:  # pure measurement (the alive tracker) -- nothing to anchor
                continue
            if not loss.requires_grad:  # a term that reaches no trainable parameter
                for group in GROUPS:
                    samples.setdefault(name, {}).setdefault(group, []).append(0.0)
                continue
            trainer.component_model.zero_grad(set_to_none=True)
            coeff = trainer.loss_metrics[name].cfg.coeff
            assert coeff is not None, f"{name} has no coeff to measure with"
            scale = 1.0 if name in NEW_TERMS else coeff
            (scale * loss).backward(retain_graph=True)
            for group in GROUPS:
                samples.setdefault(name, {}).setdefault(group, []).append(
                    group_grad_norm(trainer, group)
                )
        trainer.component_model.zero_grad(set_to_none=True)

    return {
        name: {g: float(torch.tensor(v).median()) for g, v in per_group.items()}
        for name, per_group in samples.items()
    }


def sweep_targets(existing: dict[str, float]) -> list[float]:
    """`0.1 * min` to `10 * max` of the existing terms' norms, stepping by 10x."""
    live = [v for v in existing.values() if v > 0.0]
    assert live, "every existing VPD term measured a zero gradient -- nothing to anchor against"
    start, stop = 0.1 * min(live), 10.0 * max(live)
    n = int(math.floor(math.log10(stop / start)))
    targets = [start * 10.0**j for j in range(n + 1)]
    if targets[-1] < stop / 1.5:
        targets.append(stop)
    return targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--calib-batches", type=int, default=None,
                    help="eval batches for the CI-function calibration (default: as in training)")
    ap.add_argument("--out", default=None, help="defaults to calibration/<config>.json")
    args = ap.parse_args()

    cfg = LMInterpExperimentConfig.from_file(Path(args.config_path))
    assert_config(cfg)
    set_seed(cfg.pd.seed)
    device = get_device()
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})

    target_model = build_target(cfg.target).to(device)
    from aspd.loader_patch import install_bos_for_tokenizers_that_add_it

    install_bos_for_tokenizers_that_add_it()
    loader = build_lm_loader(
        cfg.target, cfg.data, split="train", device=device,
        batch_size=cfg.pd.batch_size, seed=cfg.pd.seed,
    )
    module = cfg.pd.decomposition_targets[0].module_pattern
    inject_runtime_objects(cfg, module)
    install_component_parameterization(cfg)
    install_ci_fns()  # must precede `Trainer`, which builds the ComponentModel

    trainer = Trainer(
        target_model=target_model,
        run_batch=make_run_batch(cfg.target.output_extract),
        reconstruction_loss=recon_loss_kl,
        pd_config=cfg.pd,
        runtime_config=cfg.runtime,
    )
    probe_stats: dict[str, float] = {}
    if uses_custom_ci_fn(cfg):
        calib_loader = build_lm_loader(
            cfg.target, cfg.data, split="eval", device=device,
            batch_size=cfg.eval.batch_size if cfg.eval else cfg.pd.batch_size, seed=cfg.pd.seed,
        )
        n_calib_batches = calib_batches(cfg, args.calib_batches)
        probe_scale, probe_stats = calibrate_ci_scale(
            cfg, target_model, module,
            islice(loader_to_token_stream(calib_loader), n_calib_batches), device,
        )
        seed_ci_fn(cfg, trainer.component_model, probe_scale)
        logger.info(f"{cfg.pd.ci_config.mode} CI function calibrated: {probe_stats}")

    logger.info(f"faithfulness warmup: {cfg.pd.faithfulness_warmup_steps} steps")
    run_faithfulness_warmup(trainer.component_model, trainer._component_params, cfg.pd)

    for name, metric in trainer.loss_metrics.items():
        if name in NEW_TERMS:
            metric._scale = lambda _ctx: 1.0  # type: ignore[method-assign]

    it = iter(loader)
    weight_deltas = trainer.component_model.calc_weight_deltas()

    def ctx_builder(_i: int):
        return _build_metric_context(
            next(it), step=0, is_eval=False, device=device,
            wrapped_model=trainer._wrapped_model,
            component_model=trainer.component_model,
            config=cfg.pd, reconstruction_loss=recon_loss_kl, weight_deltas=weight_deltas,
        )

    norms = measure(trainer, ctx_builder, args.n_batches)
    existing = {n: v for n, v in norms.items() if n not in NEW_TERMS}
    new = {n: v for n, v in norms.items() if n in NEW_TERMS}

    report: dict = {"config": args.config_path, "n_batches": args.n_batches, "norms": norms,
                    "groups": {}}
    if probe_stats:
        report["ci_calibration"] = {
            **probe_stats, "split": "eval", "n_batches": n_calib_batches
        }
    for group in GROUPS:
        by_group = {n: v[group] for n, v in existing.items()}
        if not group_is_trainable(trainer, group):
            report["groups"][group] = {
                "frozen": f"every parameter in the `{group}` group has requires_grad=False",
                "existing": by_group,
                "targets": [],
                "alphas": {},
            }
            continue
        targets = sweep_targets(by_group)
        report["groups"][group] = {
            "existing": by_group,
            "targets": targets,
            "alphas": {
                n: [t / v[group] if v[group] > 0 else None for t in targets] for n, v in new.items()
            },
        }

    report["dominant_group"] = {
        n: max(GROUPS, key=lambda g: v[g]) for n, v in new.items()
    }

    out = Path(args.out or f"calibration/{Path(args.config_path).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\n=== gradient norms at step 0 ({args.n_batches} batches, median) ===")
    for name, v in norms.items():
        tag = "own term (coeff=1)" if name in NEW_TERMS else "other term (coeff applied)"
        print(f"  {name:28s} components={v['components']:.4e}  ci_fn={v['ci_fn']:.4e}   {tag}")
    for group in GROUPS:
        g = report["groups"][group]
        print(f"\n=== sweep targets, {group} group ===")
        if "frozen" in g:
            print(f"  skipped: {g['frozen']}")
            continue
        print("  " + "  ".join(f"{t:.3e}" for t in g["targets"]))
        for name, alphas in g["alphas"].items():
            marker = " <- dominant" if report["dominant_group"][name] == group else ""
            cells = "  ".join("      n/a" if a is None else f"{a:.3e}" for a in alphas)
            print(f"  {name:22s} " + cells + marker)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
