"""Open a run and trace one prompt."""

from pathlib import Path

from aspd.analysis.prompt.engine import PromptEngine
from aspd.analysis.prompt.trace import PromptTrace
from aspd.paths import RUNS_DIR

DEFAULT_RUNS_ROOT = RUNS_DIR


def latest_step(run_dir: Path) -> int:
    steps = [int(p.stem.split("_")[1]) for p in run_dir.glob("model_*.pth")]
    assert steps, f"no model_*.pth in {run_dir}"
    return max(steps)


def open_prompt(
    run: str,
    prompt: str,
    *,
    runs_root: Path | str = DEFAULT_RUNS_ROOT,
    step: int | None = None,
    device: str = "cpu",
    verify: bool = True,
) -> PromptTrace:
    """Load a decomposition run and trace one prompt through it."""
    run_dir = Path(runs_root) / run
    assert run_dir.exists(), f"no run dir {run_dir}"
    engine = PromptEngine(run_dir, step, device)
    trace = engine.trace(prompt)
    if verify:
        from aspd.analysis.circuits.replacement import assert_forward_is_exact

        assert_forward_is_exact(trace.model, trace.tokens, sampling=trace.sampling)
    return trace
