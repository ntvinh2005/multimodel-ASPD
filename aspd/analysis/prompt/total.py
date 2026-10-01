"""Total-effect attribution of every component to the target (multi-hop)."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

from aspd.analysis.prompt.trace import PromptTrace


@dataclass(frozen=True)
class TotalEffect:
    """`components[module]` is `[P, C]`, one total-effect score per position and component."""

    target: float
    components: dict[str, Tensor]


@contextmanager
def _multipliers(trace: PromptTrace, mults: dict[str, Tensor]) -> Iterator[None]:
    """Hang a zero-valued, differentiable handle off every decomposed module's output."""
    handles = []
    for module in trace.modules:
        eff = trace.effective(module)[None]  # [1, P, C], detached
        u = trace.directions(module, "write")  # [C, d_out]
        mult = torch.ones_like(eff).requires_grad_(True)
        mults[module] = mult

        def hook(_mod, _args, output, eff=eff, u=u, mult=mult):
            delta = ((mult - 1.0) * eff) @ u
            return output + delta.to(output.dtype)

        handles.append(trace.model.target_model.get_submodule(module)  # pyright: ignore[reportAttributeAccessIssue]
                       .register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def total_effect(trace: PromptTrace, target: Callable[[Tensor], Tensor]) -> TotalEffect:
    """One forward and one backward: the multi-hop score of every component at every position."""
    from aspd.capture_guard import disarmed

    mults: dict[str, Tensor] = {}
    with torch.enable_grad(), disarmed(), _multipliers(trace, mults):
        logits = trace.model(trace.tokens)  # pyright: ignore[reportCallIssue]
    t = target(logits)
    assert t.ndim == 0, f"target must be a scalar, got shape {tuple(t.shape)}"

    names = list(mults)
    grads = torch.autograd.grad(t, [mults[n] for n in names], allow_unused=True)
    out = {}
    for name, g in zip(names, grads, strict=True):
        out[name] = (torch.zeros_like(mults[name][0]) if g is None else g[0].detach().float())
    return TotalEffect(float(t.item()), out)
