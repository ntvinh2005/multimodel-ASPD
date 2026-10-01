"""The scalar a circuit explains (e.g. a logit difference)."""

import torch
from torch import Tensor


def top_k_vs_rest(logits: Tensor, *, k: int = 10, seq_pos: int = -1) -> Tensor:
    """`mean(top-k logits) - mean(the remaining logits)` at one position. Returns a scalar."""
    assert logits.ndim == 3, f"expected [B, S, V], got {tuple(logits.shape)}"
    assert logits.shape[0] == 1, "one prompt at a time"
    z = logits[0, seq_pos]
    v = z.shape[0]
    assert 0 < k < v, f"k={k} out of range for vocab {v}"
    top = torch.topk(z, k)
    mask = torch.ones_like(z, dtype=torch.bool)
    mask[top.indices] = False
    return top.values.mean() - z[mask].mean()


def logit_diff(logits: Tensor, *, correct: int, wrong: int, seq_pos: int = -1) -> Tensor:
    """`z_correct - z_wrong` at one position. Also shift-invariant."""
    assert logits.ndim == 3, f"expected [B, S, V], got {tuple(logits.shape)}"
    return logits[0, seq_pos, correct] - logits[0, seq_pos, wrong]


def single_logit(logits: Tensor, *, token: int, seq_pos: int = -1) -> Tensor:
    """One logit. **Not shift-invariant** -- see the module docstring before using it as a target."""
    assert logits.ndim == 3, f"expected [B, S, V], got {tuple(logits.shape)}"
    return logits[0, seq_pos, token]
