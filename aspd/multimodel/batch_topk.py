"""BatchTopK ``sigma_K`` for the shared code ``G^s``.

The function ranks entries over all valid token positions in a batch.  It keeps the original
non-negative pre-activation ``a_{t,c}``; the score ``a_{t,c} omega_c`` is used only to choose the
support.  Consequently S1 and S2 differ only through ``omega_c`` as specified in the method note.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from aspd.multimodel.config import SparsitySpec

# Worked examples in this file use B=1 sequence, T=2 valid tokens, and C=4 latents.
# Example pre-activations: a = [[[1, 4, 2, 3], [5, 0, 2, 1]]].


@dataclass(frozen=True)
class SparseCode:
    """Dense storage of sparse ``g^s`` plus the binary ASPD gate ``g = 1[g^s > 0]``."""

    values: Tensor
    gate: Tensor
    preactivations: Tensor
    n_selected: int


def _select_block(
    preactivations: Tensor,
    valid_tokens: Tensor,
    feature_start: int,
    feature_stop: int,
    k_per_token: int,
    ranking_weights: Tensor,
    pool_tokens: int | None,
) -> Tensor:
    """Return a boolean support for one feature block and one BatchTopK budget."""

    # Restrict a_{t,c} to this block. Example S={0,1}: block=[[1,4],[5,0]], shape [1,2,2].
    block = preactivations[..., feature_start:feature_stop]
    # Restrict omega_c to the same c. Example omega_S=[1,2].
    weights = ranking_weights[feature_start:feature_stop].to(block)
    # s_{t,c}=a_{t,c}*omega_c. Example scores=[[1,8],[5,0]].
    scores = block * weights
    # Pool every valid (t,c), as BatchTopK does. Example flat_scores=[1,8,5,0].
    valid_scores = scores[valid_tokens]
    # Start with no selected (t,c). Example support is a 1x2x2 all-False tensor.
    support = torch.zeros_like(block, dtype=torch.bool)
    if valid_scores.numel() == 0:
        return support
    valid_support = torch.zeros_like(valid_scores, dtype=torch.bool)
    block_tokens = valid_scores.shape[0] if pool_tokens is None else pool_tokens
    # Keep K times the number of valid t. Example K=1,T_valid=2 gives n_keep=2.
    # Select the largest scores. Example top-2 values are 8 and 5, at flattened indices 1 and 2.
    # Build the pooled boolean support before restoring [B,T,C_block].
    # Set the chosen pairs True. Example [False,True,True,False].
    for start in range(0, valid_scores.shape[0], block_tokens):
        pool_scores = valid_scores[start:start + block_tokens]
        flat_scores = pool_scores.reshape(-1)
        n_keep = min(k_per_token * pool_scores.shape[0], flat_scores.numel())
        if n_keep == 0:
            continue
        chosen = torch.topk(flat_scores, n_keep, sorted=False).indices
        pool_support = torch.zeros_like(flat_scores, dtype=torch.bool)
        pool_support.scatter_(0, chosen, True)
        valid_support[start:start + block_tokens] = pool_support.reshape_as(pool_scores)
    # Restore token/feature axes. Example support=[[[F,T],[T,F]]].
    support[valid_tokens] = valid_support
    return support


def batch_topk(
    preactivations: Tensor,
    valid_tokens: Tensor,
    cfg: SparsitySpec,
    ranking_weights: Tensor | None = None,
) -> SparseCode:
    """Apply D0 or Dual-K D1/D2 selection to ``a`` with shape ``[B,T,C]``.

    ``valid_tokens[B,T]`` excludes padding from both the BatchTopK pool and all downstream losses.
    For S2, callers pass ``sum_n ||d_c^(n)||_2`` as ``ranking_weights``.  The weights are detached:
    TopK chooses a discrete support and must not become an unintended decoder-norm objective.
    """

    if preactivations.ndim != 3:
        raise ValueError(f"preactivations must have shape [B,T,C], got {preactivations.shape}")
    if valid_tokens.shape != preactivations.shape[:2]:
        raise ValueError("valid_tokens must match the [B,T] token grid")
    if preactivations.shape[-1] != cfg.n_features:
        raise ValueError("the feature axis does not match sparsity.n_features")
    if ranking_weights is None:
        # S1 sets omega_c=1. Example omega=[1,1,1,1], so scores equal a_{t,c}.
        ranking_weights = torch.ones(cfg.n_features, device=preactivations.device)
    if tuple(ranking_weights.shape) != (cfg.n_features,):
        raise ValueError("ranking_weights must have shape [C]")
    # TopK support is discrete: detach omega_c and reject negative ranking weights.
    # Example omega=[1,-2,3,4] becomes [1,0,3,4] without an omega-gradient.
    ranking_weights = ranking_weights.detach().clamp_min(0)

    # Allocate support for all g^s_{t,c}; blocks below fill it with selected pairs.
    support = torch.zeros_like(preactivations, dtype=torch.bool)
    if cfg.diffing == "D0":
        # D0 runs one pool over c in [0,C); example C=4,K=2 selects 2*T_valid pairs.
        support = _select_block(
            preactivations,
            valid_tokens,
            0,
            cfg.n_features,
            cfg.top_k,
            ranking_weights,
            cfg.pool_tokens,
        )
    else:
        assert cfg.top_k_shared is not None and cfg.top_k_exclusive is not None
        # Dual-K selects S=[0,C_S) independently; example C_S=3,K_S=1.
        support[..., : cfg.n_shared] = _select_block(
            preactivations,
            valid_tokens,
            0,
            cfg.n_shared,
            cfg.top_k_shared,
            ranking_weights,
            cfg.pool_tokens,
        )
        # Then select E=[C_S,C) independently; example one exclusive coordinate at K_E=1.
        support[..., cfg.n_shared:] = _select_block(
            preactivations,
            valid_tokens,
            cfg.n_shared,
            cfg.n_features,
            cfg.top_k_exclusive,
            ranking_weights,
            cfg.pool_tokens,
        )

    # sigma_K(a) keeps original a, not score a*omega. If a=4 is selected with omega=2, g^s=4.
    values = preactivations * support.to(preactivations.dtype)
    # ASPD Eq. 5: the component gate is binary even though activation reconstruction uses g^s.
    # Example g^s=[0,4,0,0] gives g=[False,True,False,False].
    gate = values > 0
    return SparseCode(
        values=values,
        gate=gate,
        preactivations=preactivations,
        # Diagnostic total active pairs. Example two valid tokens at K=1 give n_selected=2.
        n_selected=int(gate.sum().item()),
    )


class BatchTopKController(nn.Module):
    """Checkpointed training BatchTopK statistics and inference JumpReLU threshold."""

    def __init__(self, cfg: SparsitySpec, dead_after_pools: int):
        super().__init__()
        self.cfg = cfg
        self.dead_after_pools = dead_after_pools
        self.register_buffer("threshold", torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "n_pools_not_active", torch.zeros(cfg.n_features, dtype=torch.float32)
        )

    @property
    def dead(self) -> Tensor:
        return self.n_pools_not_active >= self.dead_after_pools

    @torch.no_grad()
    def _observe(self, code: SparseCode, valid_tokens: Tensor) -> None:
        values = code.values[valid_tokens]
        pool_tokens = self.cfg.pool_tokens or max(1, values.shape[0])
        for start in range(0, values.shape[0], pool_tokens):
            pool = values[start:start + pool_tokens]
            positive = pool > 0
            if positive.any():
                smallest = pool[positive].float().min()
                if self.cfg.sync_statistics and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(smallest, op=torch.distributed.ReduceOp.MIN)
                self.threshold.mul_(1.0 - self.cfg.threshold_lr).add_(
                    self.cfg.threshold_lr * smallest
                )
            fired = positive.any(dim=0).to(self.n_pools_not_active.dtype)
            if self.cfg.sync_statistics and torch.distributed.is_initialized():
                torch.distributed.all_reduce(fired, op=torch.distributed.ReduceOp.MAX)
            fired = fired > 0
            self.n_pools_not_active.add_((~fired).to(self.n_pools_not_active.dtype))
            self.n_pools_not_active[fired] = 0

    def forward(
        self,
        preactivations: Tensor,
        valid_tokens: Tensor,
        ranking_weights: Tensor | None = None,
        *,
        update_statistics: bool = True,
    ) -> SparseCode:
        if self.training and torch.is_grad_enabled():
            code = batch_topk(preactivations, valid_tokens, self.cfg, ranking_weights)
            if update_statistics:
                self._observe(code, valid_tokens)
            return code
        gate = (preactivations > self.threshold.to(preactivations)) & valid_tokens[..., None]
        values = preactivations * gate.to(preactivations.dtype)
        return SparseCode(values, gate, preactivations, int(gate.sum().item()))
