"""Matryoshka BatchTopK SAE.

Encoder relu((x - b_dec) W_enc) with batch top-k selection during training and a learned
JumpReLU threshold at inference; the loss sums the reconstruction errors of nested prefixes of
the dictionary (`group_fracs`) plus an AuxK term on dead latents.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn

# Bussmann's group increments: prefixes M then double. G = 5.
DEFAULT_GROUP_FRACS = (1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2)


def raw_preacts_from(
    x: Float[Tensor, "... d"], w_enc: Float[Tensor, "d f"], b_dec: Float[Tensor, " d"]
) -> Float[Tensor, "... f"]:
    return (x - b_dec) @ w_enc


def preacts_from(
    x: Float[Tensor, "... d"], w_enc: Float[Tensor, "d f"], b_dec: Float[Tensor, " d"]
) -> Float[Tensor, "... f"]:
    return F.relu(raw_preacts_from(x, w_enc, b_dec))


def features_from(
    x: Float[Tensor, "... d"],
    w_enc: Float[Tensor, "d f"],
    b_dec: Float[Tensor, " d"],
    threshold: Tensor,
) -> Float[Tensor, "... f"]:
    pre = preacts_from(x, w_enc, b_dec)
    return torch.where(pre > threshold, pre, torch.zeros_like(pre))


@dataclass(frozen=True)
class MatryoshkaSAEConfig:
    d_in: int
    n_features: int
    group_fracs: tuple[float, ...] = DEFAULT_GROUP_FRACS
    top_k: int = 32
    top_k_aux: int = 512
    aux_penalty: float = 1.0 / 32.0
    n_batches_to_dead: int = 20
    l1_coeff: float = 0.0  # upstream default; the L1 term is present but inert
    threshold_lr: float = 0.01  # hardcoded upstream, exposed here
    encoder_init: Literal["reference", "unit_norm"] = "reference"
    """How `W_enc` is scaled at init. `reference` is upstream's and stays the default."""

    def group_sizes(self) -> list[int]:
        """Increments, not cumulative prefixes. The last group absorbs any rounding remainder."""
        sizes = [int(self.n_features * f) for f in self.group_fracs]
        sizes[-1] += self.n_features - sum(sizes)
        assert all(s > 0 for s in sizes), f"empty group in {sizes}"
        assert sum(sizes) == self.n_features
        return sizes


class MatryoshkaBatchTopKSAE(nn.Module):
    """Nested-prefix BatchTopK SAE. Trained once, then frozen and shared across all VPD arms."""

    threshold: Tensor
    n_batches_not_active: Tensor

    def __init__(self, cfg: MatryoshkaSAEConfig):
        super().__init__()
        self.cfg = cfg
        sizes = cfg.group_sizes()
        self.group_indices = [0] + torch.cumsum(torch.tensor(sizes), dim=0).tolist()

        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(cfg.d_in, cfg.n_features))
        )
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(cfg.n_features, cfg.d_in))
        )
        # Tied at init only; W_enc is left un-normalized thereafter.
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True))
            if cfg.encoder_init == "unit_norm":
                # After the tie, so this makes the init exactly `W_enc == W_dec.T`.
                self.W_enc.div_(self.W_enc.norm(dim=0, keepdim=True).clamp_min(1e-8))

        self.register_buffer("threshold", torch.zeros(()))
        self.register_buffer("n_batches_not_active", torch.zeros(cfg.n_features))

    # ---- encode / decode -------------------------------------------------------------------

    def raw_preacts(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... f"]:
        """`(x - b_dec) @ W_enc` -- pre-ReLU, pre-threshold."""
        return raw_preacts_from(x, self.W_enc, self.b_dec)

    def preacts(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... f"]:
        """`relu((x - b_dec) @ W_enc)`. No encoder bias -- see module docstring."""
        return preacts_from(x, self.W_enc, self.b_dec)

    def features(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... f"]:
        """Inference-time features: the BatchTopK->JumpReLU conversion via the EMA threshold."""
        return features_from(x, self.W_enc, self.b_dec, self.threshold)

    def active_mask(self, x: Float[Tensor, "... d"]) -> Bool[Tensor, "... f"]:
        return self.preacts(x) > self.threshold

    @property
    def _decoder_bias(self) -> Tensor:
        """The bias added in the RECONSTRUCTION space. `b_dec` here; `b_out` on a transcoder."""
        return self.b_dec

    def decode(self, f: Float[Tensor, "... f"]) -> Float[Tensor, "... d"]:
        return f @ self.W_dec + self._decoder_bias

    # ---- frozen derived matrices for the footprint math ------------------------------------

    def encoder_normalized(self) -> Float[Tensor, "d f"]:
        return self.W_enc / self.W_enc.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def freeze(self) -> "MatryoshkaBatchTopKSAE":
        self.requires_grad_(False)
        self.eval()
        return self

    def unfreeze(self) -> "MatryoshkaBatchTopKSAE":
        """The joint-training inverse of `freeze()`. Deliberately explicit and deliberately loud."""
        self.requires_grad_(True)
        self.train()
        return self

    # ---- training --------------------------------------------------------------------------

    @torch.no_grad()
    def _update_threshold(self, acts_topk: Tensor) -> None:
        positive = acts_topk > 0
        if positive.any():
            lr = self.cfg.threshold_lr
            self.threshold.mul_(1.0 - lr).add_(lr * acts_topk[positive].min())

    @torch.no_grad()
    def _update_inactive(self, acts_topk: Tensor) -> None:
        fired = acts_topk.sum(0) > 0
        self.n_batches_not_active += (~fired).float()
        self.n_batches_not_active[fired] = 0.0

    def _batch_topk(self, acts: Float[Tensor, "n f"]) -> Float[Tensor, "n f"]:
        """One global top-(top_k * n) over the flattened batch x dictionary -- prefixes compete."""
        n = acts.shape[0]
        flat = acts.flatten()
        top = torch.topk(flat, self.cfg.top_k * n, dim=-1)
        return torch.zeros_like(flat).scatter(-1, top.indices, top.values).reshape(acts.shape)

    def _auxiliary_loss(
        self, x: Tensor, x_reconstruct: Tensor, acts: Tensor
    ) -> Tensor:
        """Gao et al. AuxK over dead latents, against the residual of the FULL reconstruction."""
        residual = x.float() - x_reconstruct.float()
        dead = self.n_batches_not_active >= self.cfg.n_batches_to_dead
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return torch.zeros((), device=x.device)

        acts_dead = acts[:, dead]
        top = torch.topk(acts_dead, min(self.cfg.top_k_aux, n_dead), dim=-1)
        acts_aux = torch.zeros_like(acts_dead).scatter(-1, top.indices, top.values)
        aux_reconstruct = acts_aux @ self.W_dec[dead]
        if aux_reconstruct.abs().sum() == 0:
            return torch.zeros((), device=x.device)
        return (aux_reconstruct.float() - residual).pow(2).mean()

    def loss(self, x: Float[Tensor, "... d"]) -> dict[str, Tensor]:
        """Repo-exact training objective. `x` is flattened to `[n_tokens, d_in]` first."""
        return self.encode_and_loss(x)[1]

    def encode_and_loss(
        self, x: Float[Tensor, "... d"]
    ) -> tuple[Float[Tensor, "n f"], dict[str, Tensor]]:
        """`loss(x)` plus the BatchTopK features it computed, still carrying gradient."""
        x = x.reshape(-1, x.shape[-1])
        acts = self.preacts(x)
        acts_topk = self._batch_topk(acts)
        self._update_threshold(acts_topk)
        return acts_topk, self._loss_from(x, acts, acts_topk)

    def _loss_from(
        self, x: Float[Tensor, "n d"], acts: Float[Tensor, "n f"], acts_topk: Float[Tensor, "n f"]
    ) -> dict[str, Tensor]:
        # Incremental prefix reconstructions: r_g = b_dec + f[:, :m_g] @ W_dec[:m_g].
        reconstruct = self._decoder_bias.expand_as(x)
        prefix_recons: list[Tensor] = []
        for i in range(len(self.group_indices) - 1):
            lo, hi = self.group_indices[i], self.group_indices[i + 1]
            reconstruct = acts_topk[:, lo:hi] @ self.W_dec[lo:hi] + reconstruct
            prefix_recons.append(reconstruct)

        self._update_inactive(acts_topk)

        # The m=0 term is the empty-prefix reconstruction; hence the division by G+1.
        l2_terms = [(self._decoder_bias - x.float()).pow(2).mean()]
        l2_terms += [(r.float() - x.float()).pow(2).mean() for r in prefix_recons]
        l2_loss = torch.stack(l2_terms).sum() / len(l2_terms)

        l1_norm = acts_topk.float().abs().sum(-1).mean()
        aux_raw = self._auxiliary_loss(x, prefix_recons[-1], acts)
        aux_loss = self.cfg.aux_penalty * aux_raw
        total = l2_loss + self.cfg.l1_coeff * l1_norm + aux_loss

        with torch.no_grad():
            resid = prefix_recons[-1].float() - x.float()
            fvu = resid.pow(2).sum() / (x.float() - x.float().mean(0)).pow(2).sum().clamp_min(1e-8)
        return {
            "loss": total,
            "_l2_live": l2_loss + self.cfg.l1_coeff * l1_norm,
            "_aux_live": aux_loss,
            "_aux_raw_live": aux_raw,
            "l2_loss": l2_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "l1_norm": l1_norm.detach(),
            "l0_norm": (acts_topk > 0).float().sum(-1).mean().detach(),
            "fvu": fvu,
            "n_dead": (self.n_batches_not_active >= self.cfg.n_batches_to_dead).sum().detach(),
            "threshold": self.threshold.detach().clone(),
        }

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Project out the radial gradient, then renormalize rows. Call between `backward()` and
        `optimizer.step()` -- the radial component gets projected away anyway, so leaving it in
        would pollute Adam's second-moment estimates.
        """
        normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        if self.W_dec.grad is not None:
            radial = (self.W_dec.grad * normed).sum(-1, keepdim=True) * normed
            self.W_dec.grad -= radial
        self.W_dec.data = normed
