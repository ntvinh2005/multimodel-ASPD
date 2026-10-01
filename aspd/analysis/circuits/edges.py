"""Edge weights: gradient times activation between connected nodes."""

import torch
from torch import Tensor


def grad_times_act(
    target: Tensor,
    leaves: dict[str, Tensor],
    *,
    retain_graph: bool = True,
) -> dict[str, Tensor]:
    """`{name: (dT/du) * u}` for every leaf, from ONE backward."""
    names = list(leaves)
    tensors = [leaves[n] for n in names]
    grads = torch.autograd.grad(
        target, tensors, retain_graph=retain_graph, allow_unused=True
    )
    out: dict[str, Tensor] = {}
    for name, g, u in zip(names, grads, tensors, strict=True):
        out[name] = torch.zeros_like(u) if g is None else g * u
    return out


def contract_error(weighted: Tensor) -> Tensor:
    """`[B, S, d_out] -> [B, S]`. The error node is one node per (module, position)."""
    return weighted.sum(dim=-1)
