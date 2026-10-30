"""V3 adapter around the audited ASPD BatchTopK implementation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from aspd.multimodel.batch_topk import SparseCode, batch_topk
from aspd.multimodel.config import SparsitySpec


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
        return code

    def utilization(self) -> dict[str, Tensor | int]:
        return {
            "accepted_anchor_count": int(self.accepted_anchor_count.item()),
            "feature_fire_counts": self.feature_fire_counts.detach().clone(),
            "used_feature_count": int((self.feature_fire_counts > 0).sum().item()),
        }
