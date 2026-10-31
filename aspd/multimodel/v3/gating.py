"""V3 adapter around the audited ASPD BatchTopK implementation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from aspd.multimodel.batch_topk import SparseCode, batch_topk
from aspd.multimodel.config import SparsitySpec

ORIGINAL_ASPD_THRESHOLD_LR = 0.01
ORIGINAL_ASPD_N_POOLS_TO_DEAD = 320


class V3BatchTopK(nn.Module):
    def __init__(self, n_features: int, top_k: int, *, pool_anchors: int | None = None):
        super().__init__()
        self.spec = SparsitySpec(
            n_features=n_features,
            selection_score="S1",
            diffing="D0",
            top_k=top_k,
            pool_tokens=pool_anchors,
        )
        self.register_buffer("feature_fire_counts", torch.zeros(n_features, dtype=torch.long))
        self.register_buffer("accepted_anchor_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("threshold", torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "n_pools_not_active", torch.zeros(n_features, dtype=torch.float32)
        )

    @property
    def dead_features(self) -> Tensor:
        return self.n_pools_not_active >= ORIGINAL_ASPD_N_POOLS_TO_DEAD

    @torch.no_grad()
    def _observe(self, code: SparseCode, valid_anchors: Tensor) -> None:
        values = code.values[valid_anchors]
        pool_anchors = self.spec.pool_tokens or max(1, values.shape[0])
        for start in range(0, values.shape[0], pool_anchors):
            pool = values[start : start + pool_anchors]
            positive = pool > 0
            local_minimum = (
                pool[positive].float().min()
                if positive.any()
                else torch.tensor(float("inf"), device=pool.device)
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    local_minimum, op=torch.distributed.ReduceOp.MIN
                )
            if torch.isfinite(local_minimum):
                self.threshold.mul_(1.0 - ORIGINAL_ASPD_THRESHOLD_LR).add_(
                    ORIGINAL_ASPD_THRESHOLD_LR * local_minimum
                )
            fired = positive.any(dim=0).to(self.n_pools_not_active.dtype)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(fired, op=torch.distributed.ReduceOp.MAX)
            fired_mask = fired > 0
            self.n_pools_not_active.add_((~fired_mask).to(self.n_pools_not_active.dtype))
            self.n_pools_not_active[fired_mask] = 0

    def forward(
        self, scores: Tensor, valid_anchors: Tensor, *, update_statistics: bool = True
    ) -> SparseCode:
        if torch.any(scores < 0):
            raise ValueError("v3 BatchTopK requires non-negative feature scores")
        code = batch_topk(scores, valid_anchors, self.spec)
        if update_statistics:
            with torch.no_grad():
                self.feature_fire_counts.add_(code.gate[valid_anchors].sum(dim=0))
                self.accepted_anchor_count.add_(valid_anchors.sum())
                self._observe(code, valid_anchors)
        return code

    def utilization(self) -> dict[str, Tensor | int]:
        return {
            "accepted_anchor_count": int(self.accepted_anchor_count.item()),
            "feature_fire_counts": self.feature_fire_counts.detach().clone(),
            "used_feature_count": int((self.feature_fire_counts > 0).sum().item()),
            "dead_feature_count": int(self.dead_features.sum().item()),
            "threshold": self.threshold.detach().clone(),
        }
