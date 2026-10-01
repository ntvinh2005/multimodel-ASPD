"""Accumulates the component-feature effect matrix E_{t in A_j}[|zeta_c(t)|] M_{j,c} over the evaluation tokens."""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE


def raw_footprint_matrix(
    sae_out: MatryoshkaBatchTopKSAE, u: Float[Tensor, "c d_out"]
) -> Float[Tensor, "f c"]:
    enc = sae_out.W_enc
    assert enc.shape[0] == u.shape[1], (
        f"output SAE d_in {enc.shape[0]} != U d_out {u.shape[1]}; "
        "this dictionary is not on the module's output site"
    )
    return enc.t() @ u.t()


@dataclass
class AlignmentAccumulator:
    """Streams `E_t[|ζ_c|]` and `E_{t:Λ_j}[|ζ_c|]` over an arbitrary number of batches."""

    n_features: int
    n_components: int
    device: torch.device

    def __post_init__(self) -> None:
        f, c = self.n_features, self.n_components
        self.sum_abs_zeta = torch.zeros(c, dtype=torch.float64, device=self.device)
        self.n_tokens = torch.zeros((), dtype=torch.float64, device=self.device)
        self.cond_sum = torch.zeros(f, c, dtype=torch.float64, device=self.device)
        self.cond_count = torch.zeros(f, dtype=torch.float64, device=self.device)
        self.row_block = max(1, 2**26 // max(c, 1))

    @torch.no_grad()
    def update(
        self, zeta: Float[Tensor, "t c"], active: Bool[Tensor, "t f"]
    ) -> None:
        """`zeta` is `g_c z_c` per kept token; `active` is the output dictionary's fired mask."""
        assert zeta.shape[0] == active.shape[0], (zeta.shape, active.shape)
        assert zeta.shape[1] == self.n_components and active.shape[1] == self.n_features

        abs_zeta = zeta.abs().float()
        self.sum_abs_zeta += abs_zeta.sum(dim=0).double()
        self.n_tokens += zeta.shape[0]

        mask = active.float()
        mask_t = mask.t()
        for start in range(0, self.n_features, self.row_block):
            stop = min(start + self.row_block, self.n_features)
            self.cond_sum[start:stop] += (mask_t[start:stop] @ abs_zeta).double()
        self.cond_count += mask.sum(dim=0).double()

    @torch.no_grad()
    def alignments(
        self, footprint: Float[Tensor, "f c"], *, out_device: torch.device | None = None
    ) -> tuple[Float[Tensor, "c f"], Float[Tensor, "c f"]]:
        """`(A_glob, A_cond)`, both `[C, F]` and fp32, **on the CPU by default**."""
        assert footprint.shape == (self.n_features, self.n_components), footprint.shape
        assert self.n_tokens > 0, "no tokens accumulated"

        out = out_device if out_device is not None else torch.device("cpu")
        m = footprint.float()  # [F, C]
        mean_abs_zeta = (self.sum_abs_zeta / self.n_tokens).float()[:, None]
        denom = self.cond_count.clamp_min(1.0)[:, None]
        dead = self.dead_output_features

        shape = (self.n_components, self.n_features)
        glob = torch.empty(shape, dtype=torch.float32, device=out)
        cond = torch.empty(shape, dtype=torch.float32, device=out)
        for start in range(0, self.n_features, self.row_block):
            stop = min(start + self.row_block, self.n_features)
            block = m[start:stop]
            glob[:, start:stop] = (mean_abs_zeta * block.t()).to(out)
            rows = (self.cond_sum[start:stop] / denom[start:stop]).float()
            rows *= block
            rows[dead[start:stop]] = 0.0
            cond[:, start:stop] = rows.t().to(out)

        return glob, cond

    @property
    def dead_output_features(self) -> Bool[Tensor, " f"]:
        return self.cond_count == 0

    @property
    def dead_components(self) -> Bool[Tensor, " c"]:
        return self.sum_abs_zeta == 0


def raw_input_footprint_matrix(
    sae_in: MatryoshkaBatchTopKSAE, v: Float[Tensor, "d_in c"]
) -> Float[Tensor, "c f_in"]:
    dec = sae_in.W_dec
    assert dec.shape[1] == v.shape[0], (
        f"input SAE d_in {dec.shape[1]} != V d_in {v.shape[0]}; "
        "this dictionary is not on the module's input site"
    )
    return v.t().to(dec.dtype) @ dec.t()


@dataclass
class InputAlignmentAccumulator:

    n_input_features: int
    n_components: int
    device: torch.device

    def __post_init__(self) -> None:
        f, c = self.n_input_features, self.n_components
        self.sum_f = torch.zeros(f, dtype=torch.float64, device=self.device)
        self.active_count = torch.zeros(f, dtype=torch.float64, device=self.device)
        self.n_tokens = torch.zeros((), dtype=torch.float64, device=self.device)
        self.cond_sum = torch.zeros(c, f, dtype=torch.float64, device=self.device)
        self.fire_count = torch.zeros(c, dtype=torch.float64, device=self.device)
        self.row_block = max(1, 2**26 // max(f, 1))

    @torch.no_grad()
    def update(
        self, features: Float[Tensor, "t f_in"], fires: Bool[Tensor, "t c"]
    ) -> None:
        assert features.shape[0] == fires.shape[0], (features.shape, fires.shape)
        assert features.shape[1] == self.n_input_features and fires.shape[1] == self.n_components

        feats = features.float()
        assert (feats >= 0).all(), "input features went negative; JumpReLU output must be >= 0"
        self.sum_f += feats.sum(dim=0).double()
        self.active_count += (feats > 0).sum(dim=0).double()
        self.n_tokens += feats.shape[0]

        mask = fires.float()
        mask_t = mask.t()
        for start in range(0, self.n_components, self.row_block):
            stop = min(start + self.row_block, self.n_components)
            self.cond_sum[start:stop] += (mask_t[start:stop] @ feats).double()
        self.fire_count += mask.sum(dim=0).double()

    @torch.no_grad()
    def alignments(
        self, footprint: Float[Tensor, "c f_in"], *, out_device: torch.device | None = None
    ) -> tuple[Float[Tensor, "c f_in"], Float[Tensor, "c f_in"]]:
        """`(A_in_glob, A_in_cond)`, both `[C, F_in]` and fp32, on the CPU by default."""
        assert footprint.shape == (self.n_components, self.n_input_features), footprint.shape
        assert self.n_tokens > 0, "no tokens accumulated"

        out = out_device if out_device is not None else torch.device("cpu")
        m = footprint.float()
        mean_f = (self.sum_f / self.n_tokens).float()[None, :]
        denom = self.fire_count.clamp_min(1.0)[:, None]
        dead = self.dead_input_features

        glob = torch.empty(m.shape, dtype=torch.float32, device=out)
        cond = torch.empty(m.shape, dtype=torch.float32, device=out)
        for start in range(0, self.n_components, self.row_block):
            stop = min(start + self.row_block, self.n_components)
            block = m[start:stop]
            glob[start:stop] = (mean_f * block).to(out)
            rows = (self.cond_sum[start:stop] / denom[start:stop]).float()
            rows *= block
            rows[:, dead] = 0.0
            cond[start:stop] = rows.to(out)

        return glob, cond

    @property
    def dead_input_features(self) -> Bool[Tensor, " f_in"]:
        return self.active_count == 0

    @property
    def dead_components(self) -> Bool[Tensor, " c"]:
        return self.fire_count == 0
