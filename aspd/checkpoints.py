"""Load a trained decomposition from a run directory at a given step."""

import re
from pathlib import Path

from torch import nn

from aspd.ci.setup import attach_ci_fn, install_ci_fns
from aspd.component_setup import install_component_parameterization
from aspd.config import LMInterpExperimentConfig

install_ci_fns()


def install_decomposition_only_checkpoints() -> None:
    """Let `ComponentModel.load_state_dict` take a checkpoint without the frozen target's weights.

    Released checkpoints hold only the decomposition (`_components.*`, `ci_fn.*`); the target model
    is built from the config. Missing `target_model.*` entries are taken from the built model; the
    load is otherwise strict. Idempotent.
    """
    from param_decomp.component_model import ComponentModel

    if getattr(ComponentModel.load_state_dict, "_aspd_patched", False):
        return
    stock = ComponentModel.load_state_dict

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if not any(k.startswith("target_model.") for k in state_dict):
            target = {k: v for k, v in self.state_dict().items() if k.startswith("target_model.")}
            state_dict = {**target, **state_dict}
        return stock(self, state_dict, strict=strict, assign=assign)

    load_state_dict._aspd_patched = True  # pyright: ignore[reportFunctionMemberAccess]
    ComponentModel.load_state_dict = load_state_dict  # pyright: ignore[reportAttributeAccessIssue]


install_decomposition_only_checkpoints()


def load_run_model(
    run_dir: Path,
    step: int,
    device: str,
    target_model: nn.Module | None = None,
):
    """(ComponentModel on `device` in eval mode, run config). Pass `target_model` to reuse it across steps."""
    from param_decomp_lab.component_model_io import load_component_model
    from param_decomp_lab.experiments.lm.run import build_target, make_run_batch

    cfg = LMInterpExperimentConfig.from_file(run_dir / "experiment_config.yaml")
    install_component_parameterization(cfg)
    ckpt = run_dir / f"model_{step}.pth"
    assert ckpt.exists(), f"no checkpoint {ckpt}"
    model = load_component_model(
        pd_config=cfg.pd,
        checkpoint_path=ckpt,
        target_model=target_model if target_model is not None else build_target(cfg.target),
        run_batch=make_run_batch(cfg.target),
    )
    attach_ci_fn(model)
    return model.to(device).eval(), cfg


def resolve_steps(run_dir: Path, spec: str) -> list[int]:
    """`all` -> every `model_<step>.pth` in the run dir; otherwise an explicit list."""
    if spec == "all":
        steps = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("model_*.pth"))
        assert steps, f"no model_*.pth in {run_dir}"
        return steps
    steps = [int(s) for s in re.split(r"[,:\s]+", spec.strip()) if s]
    assert steps, f"could not parse --steps {spec!r}"
    return steps
