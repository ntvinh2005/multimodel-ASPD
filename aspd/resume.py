"""Resume training from the newest complete checkpoint, including the data loader's position."""

import os
from pathlib import Path
from typing import Any

import torch

from param_decomp.distributed import get_distributed_state
from param_decomp.log import logger
from param_decomp.optimize import Trainer
from param_decomp.training_state import TrainingState

LOADER_STATE_PREFIX = "loader"


def current_rank() -> int:
    """Global rank, 0 when not distributed."""
    state = get_distributed_state()
    return 0 if state is None else state.rank


def _steps_for(run_dir: Path, prefix: str) -> dict[int, Path]:
    """`{step: path}` for `<prefix>_<step>.pth` in `run_dir`, ignoring unparseable names."""
    out: dict[int, Path] = {}
    for path in run_dir.glob(f"{prefix}_*.pth"):
        try:
            out[int(path.stem.removeprefix(f"{prefix}_"))] = path
        except ValueError:
            continue
    return out


def resumable_steps(run_dir: Path) -> list[int]:
    """Steps with a COMPLETE (model, training) pair, ascending."""
    if not run_dir.is_dir():
        return []
    models = _steps_for(run_dir, "model")
    trainings = _steps_for(run_dir, "training")
    return sorted(
        step
        for step in models.keys() & trainings.keys()
        if models[step].stat().st_size > 0 and trainings[step].stat().st_size > 0
    )


def resolve_resume_step(spec: str, run_dir: Path) -> int | None:
    """Turn a `--resume` value into a step, or `None` for "start fresh"."""
    if spec == "off":
        return None
    steps = resumable_steps(run_dir)
    if spec == "auto":
        return steps[-1] if steps else None
    try:
        step = int(spec)
    except ValueError:
        raise AssertionError(f"--resume must be 'off', 'auto' or a step; got {spec!r}") from None
    assert step in steps, (
        f"--resume {step}: no complete (model, training) pair at step {step} in {run_dir}. "
        f"Resumable steps: {steps or 'none'}"
    )
    return step


def load_training_state(run_dir: Path, step: int) -> TrainingState:
    """The `TrainingState` written alongside `model_<step>.pth`."""
    path = run_dir / f"training_{step}.pth"
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(state, TrainingState), f"{path} holds {type(state).__name__}, not TrainingState"
    assert state.step == step, f"{path} says step {state.step}, expected {step}"
    return state


def _loader_state_path(run_dir: Path, step: int, rank: int) -> Path:
    """Per-RANK, and that is load-bearing under DP."""
    return run_dir / f"{LOADER_STATE_PREFIX}_{step}_rank{rank}.pth"


def save_loader_state(out_dir: Path | None, step: int, loader: Any) -> None:
    """Write this rank's train-loader stream position beside checkpoint `step`."""
    if out_dir is None or loader is None or not out_dir.is_dir():
        return
    dataset = getattr(loader, "dataset", None)
    state_dict = getattr(dataset, "state_dict", None)
    if state_dict is None:
        return
    rank = current_rank()
    tmp = out_dir / f"{LOADER_STATE_PREFIX}_{step}_rank{rank}.pth.tmp"
    torch.save(state_dict(), tmp)
    os.replace(tmp, _loader_state_path(out_dir, step, rank))

    live = set(_steps_for(out_dir, "training"))
    for path in out_dir.glob(f"{LOADER_STATE_PREFIX}_*_rank{rank}.pth"):
        try:
            orphan_step = int(path.stem.removeprefix(f"{LOADER_STATE_PREFIX}_").split("_rank")[0])
        except ValueError:
            continue
        if orphan_step not in live and orphan_step != step:
            path.unlink(missing_ok=True)


def load_loader_state(run_dir: Path, step: int) -> Any | None:
    """This rank's saved stream position for `step`, or `None` if there is no usable sidecar."""
    path = _loader_state_path(run_dir, step, current_rank())
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_loader_position(loader: Any, run_dir: Path, step: int) -> bool:
    """Fast-forward `loader` to where checkpoint `step` left off."""
    state = load_loader_state(run_dir, step)
    if state is None:
        return False
    dataset = getattr(loader, "dataset", None)
    load = getattr(dataset, "load_state_dict", None)
    if load is None:
        return False
    load(state)
    return True


class PreAdvancedLoader:
    """A `DataLoader` stand-in whose first `skip` batches cost nothing."""

    def __init__(self, inner: Any, skip: int) -> None:
        assert skip >= 0, f"skip must be non-negative, got {skip}"
        self._inner = inner
        self._skip = skip
        self._skipped = False

    def __iter__(self) -> Any:
        if not self._skipped:
            self._skipped = True
            for _ in range(self._skip):
                yield None  # consumed and discarded by Trainer.run's replay; never reaches a step
        yield from self._inner

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


def apply_resume(trainer: Trainer, train_loader: Any, run_dir: Path, step: int) -> Any:
    """Restore `trainer` to `step` and return the loader to hand to `Trainer.run`."""
    state = load_training_state(run_dir, step)
    trainer._load_state(state)  # noqa: SLF001 -- see module docstring
    logger.info(f"resume: trainer restored to step {step} from training_{step}.pth")

    if restore_loader_position(train_loader, run_dir, step):
        logger.info(f"resume: stream position restored from {LOADER_STATE_PREFIX}_{step}.pth")
        return PreAdvancedLoader(train_loader, skip=step)

    logger.warning(
        f"resume: no {LOADER_STATE_PREFIX}_{step}_rank{current_rank()}.pth sidecar -- replaying "
        f"{step} batches to reach the same stream position. Exact, but on a streaming loader "
        "expect ~1h at step 200k, and under DP the other ranks idle at the first barrier until "
        "this one catches up."
    )
    return train_loader
