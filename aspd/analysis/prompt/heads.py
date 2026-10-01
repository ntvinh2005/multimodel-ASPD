"""Which attention heads a component acts through (per-head mass)."""

import torch
from torch import Tensor

from aspd.analysis.pairs.spaces import ModuleSpaces, Side


def headed_side(spaces: ModuleSpaces) -> Side:
    """The side of this module that carries head structure. Raises if neither does."""
    for side in ("write", "read"):
        if spaces.space(side).heads is not None:  # pyright: ignore[reportArgumentType]
            return side  # pyright: ignore[reportReturnType]
    raise AssertionError(
        f"{spaces.module} ({spaces.role}) has no head-structured side: "
        f"read={spaces.read.key}, write={spaces.write.key}"
    )


def head_mass(directions: Tensor, spaces: ModuleSpaces, side: Side) -> Tensor:
    """`[C, n_heads]`, each row a distribution over heads summing to 1."""
    heads = spaces.space(side).heads
    assert heads is not None, f"{spaces.module}:{side} is not head-structured"
    assert directions.shape[1] == heads.n_heads * heads.head_dim, (
        f"{spaces.module}:{side} directions are {directions.shape[1]}d but the layout is "
        f"{heads.n_heads} x {heads.head_dim}"
    )
    blocks = directions.unflatten(1, (heads.n_heads, heads.head_dim))  # [C, H, d_h]
    sq = blocks.pow(2).sum(dim=-1)  # [C, H]
    return sq / sq.sum(dim=1, keepdim=True).clamp_min(torch.finfo(sq.dtype).tiny)
