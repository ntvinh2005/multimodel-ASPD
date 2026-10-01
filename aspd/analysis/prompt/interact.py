"""Prompt-specific interaction scores for every pairing the viewer offers."""

from typing import Literal

import torch
from torch import Tensor

from aspd.analysis.pairs import endpoints as ep
from aspd.analysis.pairs import features as ft
from aspd.analysis.pairs.spaces import Side, compatibility, head_pairs
from aspd.analysis.pairs.suggest import LINKS, link_for
from aspd.analysis.prompt.features import feature_acts
from aspd.analysis.prompt.trace import PromptTrace

Mode = Literal["same_pos", "bilinear", "routed"]


class _Side:
    """One endpoint resolved: its reconciled directions and its per-position factor."""

    def __init__(self, trace: PromptTrace, saes, key: str, side: Side):
        self.key, self.side = key, side
        self.is_sae = ft.is_sae_key(key)
        if self.is_sae:
            assert saes is not None, f"{key} is a dictionary endpoint but no SaeStore was given"
            self.ep = ep.sae_endpoint(saes.endpoint(key), side)
            acts = feature_acts(trace, saes, key)
            self.factor = acts if side == "write" else torch.ones_like(acts)
            self.factor_name = "a_f" if side == "write" else "1"
            self.dirs = saes.directions(key, side).mat
        else:
            spec = trace.spaces[key]
            self.ep = ep.module_endpoint(key, side, spec, trace.acts[key].shape[1])
            self.factor = trace.effective(key) if side == "write" else trace.gates[key]
            self.factor_name = "g*a" if side == "write" else "g"
            self.dirs = trace.directions(key, side)


_OV = {("attn.v", "write"), ("attn.o", "read")}


def _mode(a: _Side, b: _Side, link: str | None) -> Mode:
    """Which positional template this pairing follows."""
    if link == "bilinear_form":
        return "bilinear"
    if {(a.ep.role, a.side), (b.ep.role, b.side)} == _OV and a.ep.layer == b.ep.layer:
        return "routed"
    return "same_pos"


def _free_side(a: _Side, b: _Side, mode: Mode) -> _Side | None:
    """The endpoint read at `src` -- the KEY for `bilinear`, the VALUE for `routed`, none for
    `same_pos`.
    """
    if mode == "same_pos":
        return None
    want = "attn.k" if mode == "bilinear" else "attn.v"
    return a if a.ep.role == want else b


def _prepare(trace: PromptTrace, saes, norms, a_key, a_side, b_key, b_side, head):
    a = _Side(trace, saes, a_key, a_side)
    b = _Side(trace, saes, b_key, b_side)
    link = (ep.sae_link(a.ep, b.ep, trace.model_name) if (a.is_sae or b.is_sae)
            else link_for(trace.spaces[a_key], a_side, trace.spaces[b_key], b_side))
    mode = _mode(a, b, link)

    rec = ep.reconcile(
        a.ep, b.ep,
        ft.Directions(mat=a.dirs, norms=torch.zeros(1)),
        ft.Directions(mat=b.dirs, norms=torch.zeros(1)),
        model_name=trace.model_name, norms=norms or ft.TargetNorms(trace.model_name),
    )
    da, db = rec.a.mat, rec.b.mat
    sa, sb = a.ep.space, b.ep.space
    compat = compatibility(sa, sb)
    if head is not None:
        assert sa.heads is not None and sb.heads is not None, "this pairing has no head structure"
        pairs = head_pairs(sa.heads, sb.heads)
        assert 0 <= head < len(pairs), f"head {head} is outside 0..{len(pairs) - 1}"
        ha, hb = pairs[head]
        da, db = da[:, sa.heads.head_slice(ha)], db[:, sb.heads.head_slice(hb)]
    else:
        assert compat["flat"], str(compat["reason"])
        assert mode != "routed", "the OV pairing routes through one head; pass `head`"
        assert link != "bilinear_form", (
            "the attention logit is per head; a flat score sums the heads' logits, which the model "
            "never forms. Pass `head`."
        )
    scale = 1.0
    if mode == "bilinear":
        cfg = trace.model.target_model.config  # pyright: ignore[reportAttributeAccessIssue]
        assert getattr(cfg, "scale_attn_weights", True), "attention logits are unscaled here"
        assert not getattr(cfg, "scale_attn_by_inverse_layer_idx", False), (
            "this model scales attention logits by 1/(layer+1); this scale does not"
        )
        scale = trace.head_dim**-0.5
    return a, b, da, db, link, mode, _free_side(a, b, mode), rec.applied, scale


def series(
    trace: PromptTrace, a_key: str, b_key: str, a_idx: int, b_idx: int, *,
    a_side: Side = "write", b_side: Side = "read", head: int | None = None,
    saes=None, norms=None,
) -> Tensor:
    """One component pair's score over positions, UN-SUMMED."""
    a, b, da, db, _, mode, free, _, scale = _prepare(
        trace, saes, norms, a_key, a_side, b_key, b_side, head
    )
    g = scale * float(da[a_idx] @ db[b_idx])
    fa, fb = a.factor[:, a_idx], b.factor[:, b_idx]
    if mode == "same_pos":
        return fa * g * fb
    assert free is not None
    f_src, f_dest = (fa, fb) if free is a else (fb, fa)
    if mode == "bilinear":
        return torch.outer(f_dest, f_src) * g  # [query, key] -- the query is read at `dest`
    attn = trace.attn_probs[free.ep.layer][head]  # pyright: ignore[reportArgumentType]
    return attn * f_src.unsqueeze(0) * g * f_dest.unsqueeze(1)


def interact(
    trace: PromptTrace, a_key: str, b_key: str, *,
    a_side: Side = "write", b_side: Side = "read",
    pos: int | None = None, src: int | None = None, head: int | None = None,
    a_idx: int | None = None, b_idx: int | None = None,
    saes=None, norms=None, k: int = 25,
) -> dict[str, object]:
    """Rank the component pairs interacting most strongly at one position."""
    a, b, da, db, link, mode, free, applied, scale = _prepare(
        trace, saes, norms, a_key, a_side, b_key, b_side, head
    )
    p = trace.n_pos - 1 if pos is None else pos
    trace.check_pos(p)
    if src is not None:
        trace.check_pos(src)
        assert src <= p, f"source {src} is after {p}; attention is causal"
    fired: dict[int, bool] = {}
    # Stated before any score is computed, so an empty table always says why it is empty.
    reason: str | None = None
    if mode == "same_pos" and src is not None and src != p:
        reason = (
            f"{link or 'this pairing'} reads both sides at one position, so there is no term "
            f"linking {src} to {p} -- not a weak one, none at all"
        )
    # `routed` carries the attention weight `A[h, pos, src]` on the value's source position.
    weight = (trace.attn_probs[free.ep.layer][head][p] if mode == "routed"  # pyright: ignore[reportArgumentType,reportOptionalMemberAccess]
              else None)

    def _live(side: _Side, pin: int | None) -> Tensor:
        if side is not free:
            nonzero = side.factor[p] != 0
        elif src is not None:
            nonzero = side.factor[src] != 0
        else:
            nonzero = side.factor[: p + 1].abs().sum(0) != 0
        if pin is not None:
            assert 0 <= pin < nonzero.numel(), f"component {pin} is outside 0..{nonzero.numel() - 1}"
            fired[id(side)] = bool(nonzero[pin])
            return torch.tensor([pin])
        return nonzero.nonzero(as_tuple=False).flatten()

    def _factor(side: _Side, live: Tensor) -> tuple[Tensor, Tensor | None]:
        if side is not free:
            return side.factor[p, live], None
        f = side.factor[: p + 1, live]  # [pos+1, n] -- causal, so never past `pos`
        if weight is not None:
            f = f * weight[: p + 1].unsqueeze(1)
        if src is not None:
            return f[src], None
        best = f.abs().argmax(dim=0)  # strongest source per component, reported not summed
        return f[best, torch.arange(f.shape[1])], best

    live_a, live_b = _live(a, a_idx), _live(b, b_idx)
    for name, side, pin in (("a", a, a_idx), ("b", b, b_idx)):
        if pin is not None and not fired[id(side)] and reason is None:
            where = p if side is not free else (src if src is not None else "any source position")
            reason = (
                f"{name}:{pin} did not fire at {where}, the position this pairing reads it at, "
                "so every score here would be exactly 0"
            )
    rows: list[dict[str, object]] = []
    if reason is None and live_a.numel() and live_b.numel():
        geometry = da[live_a] @ db[live_b].T  # [na, nb]
        fa, at_a = _factor(a, live_a)
        fb, at_b = _factor(b, live_b)
        score = scale * geometry * fa.unsqueeze(1) * fb.unsqueeze(0)
        flat = score.abs().flatten()
        for i in torch.topk(flat, min(k, flat.numel())).indices.tolist():
            r, c = divmod(int(i), score.shape[1])
            row = {"a_idx": int(live_a[r]), "b_idx": int(live_b[c]),
                   "score": float(score[r, c]), "geometry": float(geometry[r, c])}
            if at_a is not None:
                row["src"] = int(at_a[r])
            elif at_b is not None:
                row["src"] = int(at_b[c])
            elif src is not None:
                row["src"] = src
            rows.append(row)
        rows.sort(key=lambda x: -abs(x["score"]))  # pyright: ignore[reportArgumentType]

    return {
        "a": a_key, "a_side": a_side, "b": b_key, "b_side": b_side,
        "link": link, "link_label": LINKS[link]["label"] if link else None,
        "caveat": LINKS[link]["caveat"] if link else
                  "No template describes this pairing; the score is a raw inner product between "
                  "two spaces whose relationship is unstated.",
        "mode": mode, "head": head, "pos": p, "src": src, "piece": trace.pieces[p],
        "applied": applied,
        "factors": {"a": a.factor_name, "b": b.factor_name},
        "n_live_a": int(live_a.numel()), "n_live_b": int(live_b.numel()),
        "pinned": {"a": a_idx, "b": b_idx},
        "fired": {"a": fired.get(id(a)), "b": fired.get(id(b))},
        # Which position each side is READ at, which the mode decides and the caller cannot assume.
        "reads_at": {"a": ("src" if a is free else "pos"), "b": ("src" if b is free else "pos")},
        "reason": reason,
        "rows": rows,
    }
