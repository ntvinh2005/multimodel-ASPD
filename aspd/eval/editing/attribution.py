"""Attribution of output-SAE feature activations to component gates (effect_{j,c})."""

from dataclasses import dataclass

import torch
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE
from aspd.transcoder_components import TranscoderLinearComponents
from jaxtyping import Bool, Float, Int
from param_decomp.component_model import ComponentModel
from param_decomp.components import LinearComponents
from torch import Tensor

_RANK_ONE_CLASSES: tuple[type, ...] = (LinearComponents, TranscoderLinearComponents)


def is_rank_one(components) -> bool:
    """Whether `v_c(x) = (V_c . x) U_c` holds, which is what the analytic path needs."""
    return type(components) in _RANK_ONE_CLASSES


def bias_residue_for(
    components, selection: Int[Tensor, " s"]
) -> Float[Tensor, " d_out"] | None:
    """`sum_{c in S} (b_dec . V_c) U_c` -- what a WEIGHT-ONLY edit cannot remove. `None` off a
    transcoder arm.
    """
    b_dec = getattr(components, "b_dec", None)
    if b_dec is None:
        return None
    u = components.U.detach()[selection].float()  # [s, d_out]
    v = components.V.detach()[:, selection].float()  # [d_in, s]
    return (b_dec.detach().float() @ v) @ u


def m_raw(
    sae: MatryoshkaBatchTopKSAE, u: Float[Tensor, "c d_out"], feature_ids: Int[Tensor, " j"]
) -> Float[Tensor, "j c"]:
    """`M_raw[j, c] = <W_enc[:, j], U_c>` for the sampled features only."""
    enc = sae.W_enc.detach()  # [d_out, F]
    assert enc.shape[0] == u.shape[1], (
        f"encoder d_out {enc.shape[0]} != U d_out {u.shape[1]}; this dictionary is not on the "
        "decomposed module's output site"
    )
    return enc[:, feature_ids].t().double() @ u.t().double()


@torch.no_grad()
def gate_and_acts(
    model: ComponentModel, module_path: str, tokens: Int[Tensor, "b l"], sampling: str = "continuous"
) -> tuple[Tensor, Tensor, Tensor]:
    """`(g, z, x)` for one token batch -- the gate, the component activations, and the input."""
    out = model.forward(tokens, cache_type="input")
    x = out.cache[module_path]
    z = model.components[module_path].get_component_acts(x)
    g = model.calc_causal_importances({module_path: x}, sampling=sampling).lower_leaky[module_path]
    assert g.shape == z.shape, (g.shape, z.shape)
    assert (g >= 0).all(), (
        "causal importance went negative; the gate patch `df/dm * g` would change meaning"
    )
    return g, z, x


@dataclass
class AttributionAccumulator:
    """`sum_{t in A_j}` of the two per-token quantities, plus `|A_j|`."""

    n_features: int
    n_components: int
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        shape = (self.n_features, self.n_components)
        self.sum_gate = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.sum_unit = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.count = torch.zeros(self.n_features, dtype=torch.float64, device=self.device)

    def add(
        self,
        active: Bool[Tensor, "t j"],
        gate_term: Float[Tensor, "t c"],
        unit_term: Float[Tensor, "t c"],
    ) -> None:
        """Accumulate one flattened token block. `active[t, j]` is `t in A_j`."""
        weight = active.to(gate_term.dtype).t()  # [j, t]
        self.sum_gate += (weight @ gate_term).double()
        self.sum_unit += (weight @ unit_term).double()
        self.count += weight.sum(dim=1).double()

    def add_sums(
        self, sum_gate: Float[Tensor, "j c"], sum_unit: Float[Tensor, "j c"], count: Tensor
    ) -> None:
        """Accumulate already-reduced sums -- the autograd path's shape, which has no `[t, c]`
        term to hand over (the gradient is consumed one feature at a time).
        """
        self.sum_gate += sum_gate.double()
        self.sum_unit += sum_unit.double()
        self.count += count.double()

    def finalize(self) -> tuple[Tensor, Tensor, Tensor]:
        """`(a_gate, a_unit, count)`, the sums divided by each feature's own support."""
        assert (self.count > 0).all(), (
            f"features {(self.count == 0).nonzero(as_tuple=True)[0].tolist()} had no active token; "
            "the support filter in `sample.py` is what is supposed to make that impossible"
        )
        denom = self.count[:, None]
        return self.sum_gate / denom, self.sum_unit / denom, self.count.clone()


@dataclass
class GlobalAccumulator:

    n_components: int
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        self.sum_gate = torch.zeros(self.n_components, dtype=torch.float64, device=self.device)
        self.sum_unit = torch.zeros_like(self.sum_gate)
        self.count = 0.0

    def add(
        self,
        keep: Bool[Tensor, " t"],
        gate_term: Float[Tensor, "t c"],
        unit_term: Float[Tensor, "t c"],
    ) -> None:
        """Accumulate one flattened token block, restricted to the pad/bos/eos mask."""
        w = keep.to(gate_term.dtype)
        self.sum_gate += (w @ gate_term).double()
        self.sum_unit += (w @ unit_term).double()
        self.count += float(keep.sum())

    def finalize(self) -> tuple[Tensor, Tensor, float]:
        assert self.count > 0, "the global attribution set is empty"
        return self.sum_gate / self.count, self.sum_unit / self.count, self.count


def jaccard(a: Tensor, b: Tensor) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


def analytic_terms(
    g: Float[Tensor, "t c"], z: Float[Tensor, "t c"]
) -> tuple[Tensor, Tensor]:
    """The two per-token factors of the analytic path, before `M_raw` is applied."""
    return g * z, z


def autograd_terms(
    model: ComponentModel,
    module_path: str,
    sae: MatryoshkaBatchTopKSAE,
    *,
    x: Float[Tensor, "b l d_in"],
    y_true: Float[Tensor, "b l d_out"],
    g: Float[Tensor, "b l c"],
    feature_ids: Int[Tensor, " j"],
    active: Bool[Tensor, "b l j"],
) -> tuple[Tensor, Tensor]:
    """`(sum_gate, sum_unit)` over this batch, `[j, c]` each, by differentiating the real forward."""
    components = model.components[module_path]
    delta = model.calc_weight_deltas()[module_path]
    ones = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)

    with torch.enable_grad():
        m = g.detach().clone().requires_grad_(True)
        y_m = components.forward(x, mask=m, weight_delta_and_mask=(delta, ones))
        with torch.no_grad():
            y_g = components.forward(x, mask=g.detach(), weight_delta_and_mask=(delta, ones))
        y = y_true.detach() + (y_m - y_g)

        enc = sae.W_enc.detach()[:, feature_ids]  # [d_out, j]
        assert y.dtype == enc.dtype, (y.dtype, enc.dtype)
        preacts = torch.relu((y - sae.b_dec) @ enc)  # [b, l, j]
        features = torch.where(preacts > sae.threshold, preacts, torch.zeros_like(preacts))

        n_feat, n_comp = feature_ids.numel(), g.shape[-1]
        sum_gate = torch.zeros(n_feat, n_comp, dtype=torch.float64, device=g.device)
        sum_unit = torch.zeros_like(sum_gate)
        for i in range(n_feat):
            selected = features[..., i] * active[..., i]
            if not torch.any(active[..., i]):
                continue
            (grad,) = torch.autograd.grad(selected.sum(), m, retain_graph=i < n_feat - 1)
            keep = active[..., i].unsqueeze(-1)
            sum_unit[i] = (grad * keep).reshape(-1, n_comp).sum(dim=0).double()
            sum_gate[i] = (grad * g.detach() * keep).reshape(-1, n_comp).sum(dim=0).double()
    return sum_gate, sum_unit


def ranking_order(scores: Float[Tensor, " c"]) -> Tensor:
    return scores.abs().sort(descending=True, stable=True).indices


def top_k_selection(scores: Float[Tensor, " c"], k: int) -> Tensor:
    """The nested top-`k` as an ascending index set -- what the edit sums over."""
    assert 0 < k <= scores.numel(), (k, scores.numel())
    return ranking_order(scores)[:k].sort().values
