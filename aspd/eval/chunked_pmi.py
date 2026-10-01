"""Top/bottom-k token PMI per latent, accumulated one vocabulary slice at a time."""

import torch
from jaxtyping import Float, Int
from torch import Tensor

NEG_INF = float("-inf")


class ChunkedTokenPmiRanker:
    """Running exact top/bottom-k PMI per component, fed one vocabulary slice at a time."""

    def __init__(self, n_components: int, vocab_size: int, top_k: int, device: torch.device | str):
        self.n_components = n_components
        self.vocab_size = vocab_size
        self.top_k = top_k
        self.device = torch.device(device)
        self.n_valid: Int[Tensor, " C"] = torch.zeros(
            n_components, dtype=torch.long, device=self.device
        )
        self._top_vals: Float[Tensor, "C k"] = torch.full(
            (n_components, top_k), NEG_INF, device=self.device
        )
        self._top_idx: Int[Tensor, "C k"] = torch.full(
            (n_components, top_k), -1, dtype=torch.long, device=self.device
        )
        self._bot_vals: Float[Tensor, "C k"] = torch.full(
            (n_components, top_k), float("inf"), device=self.device
        )
        self._bot_idx: Int[Tensor, "C k"] = torch.full(
            (n_components, top_k), -1, dtype=torch.long, device=self.device
        )

    @torch.no_grad()
    def add_chunk(
        self,
        cooccurrence: Float[Tensor, "C w"],
        marginals: Float[Tensor, " w"],
        firing_counts: Float[Tensor, " C"],
        total_tokens: int,
        vocab_offset: int,
    ) -> None:
        """Fold one finished vocabulary slice `[vocab_offset, vocab_offset + w)` into the ranking."""
        cooc = cooccurrence.to(self.device).float()
        marg = marginals.to(self.device).float()
        firings = firing_counts.to(self.device).float()
        w = cooc.shape[-1]
        assert cooc.shape == (self.n_components, w), cooc.shape
        assert marg.shape == (w,), marg.shape

        valid = (cooc > 0) & (marg > 0).unsqueeze(0)
        pmi = torch.log(cooc * total_tokens / (firings.unsqueeze(1) * marg.unsqueeze(0) + 1e-10))
        pmi = torch.where(valid, pmi, torch.full_like(pmi, NEG_INF))

        self.n_valid += (pmi > NEG_INF).sum(dim=1)

        idx = torch.arange(vocab_offset, vocab_offset + w, device=self.device)
        idx_row = idx.unsqueeze(0).expand(self.n_components, w)

        k = min(self.top_k, w)
        # --- top side: merge this slice's k best with the running k best
        chunk_top = torch.topk(pmi, k, dim=1, largest=True)
        cand_vals = torch.cat([self._top_vals, chunk_top.values], dim=1)
        cand_idx = torch.cat([self._top_idx, idx_row.gather(1, chunk_top.indices)], dim=1)
        merged = torch.topk(cand_vals, self.top_k, dim=1, largest=True)
        self._top_vals = merged.values
        self._top_idx = cand_idx.gather(1, merged.indices)

        finite_pmi = torch.where(pmi > NEG_INF, pmi, torch.full_like(pmi, float("inf")))
        chunk_bot = torch.topk(finite_pmi, k, dim=1, largest=False)
        cand_vals = torch.cat([self._bot_vals, chunk_bot.values], dim=1)
        cand_idx = torch.cat([self._bot_idx, idx_row.gather(1, chunk_bot.indices)], dim=1)
        merged = torch.topk(cand_vals, self.top_k, dim=1, largest=False)
        self._bot_vals = merged.values
        self._bot_idx = cand_idx.gather(1, merged.indices)

    def finalize(self, flat_idx: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """`(top, bottom)` for one component, matching `top_k_pmi`'s contract exactly."""
        n_valid = int(self.n_valid[flat_idx])
        k = min(self.top_k, n_valid)
        if k == 0:
            return [], []

        top_vals = self._top_vals[flat_idx].tolist()
        top_idx = self._top_idx[flat_idx].tolist()
        top_items = [(int(i), float(v)) for v, i in zip(top_vals, top_idx, strict=True) if v > NEG_INF][:k]

        n_invalid = self.vocab_size - n_valid
        n_bottom = k - n_invalid
        if n_bottom <= 0:
            return top_items, []

        bot_vals = self._bot_vals[flat_idx].tolist()
        bot_idx = self._bot_idx[flat_idx].tolist()
        bottom_items = [
            (int(i), float(v))
            for v, i in zip(bot_vals, bot_idx, strict=True)
            if v < float("inf")
        ][:n_bottom]
        return top_items, bottom_items
