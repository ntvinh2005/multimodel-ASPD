"""Per-prompt scores.

- attribution patching: attrib(c, t) = dLOGITDIFF / dxi_{t,c} at xi = 1, with xi_{t,c} scaling
  e_{t,c} u_c;
- QK contribution of a query-key component pair in head h:
  e_{c1}(x_{t1}) e_{c2}(x_{t2}) <u^h_{c1}, u^h_{c2}> / sqrt(d_h);
- OV and cross-layer contributions: e_{c1} <u_{c1}, v_{c2}> g_{c2} (times the attention pattern for OV);
- the QK weight edit: remove components from W_Q and W_K and recompute the attention pattern.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from aspd.analysis.prompt.trace import PromptTrace


@dataclass(frozen=True)
class NodeScores:
    """`components[module]` is `[P, C]`; `errors[module]` and `embed` are `[P]`."""

    target: float
    components: dict[str, Tensor]
    errors: dict[str, Tensor]
    embed: Tensor | None
    unreachable: frozenset[str]

    def top(self, k: int = 25) -> list[dict[str, object]]:
        """The `k` largest `|score|` component nodes, as plain rows."""
        rows = []
        for module, s in self.components.items():
            flat = s.abs().flatten()
            n = min(k, flat.numel())
            for i in torch.topk(flat, n).indices.tolist():
                p, c = divmod(int(i), s.shape[1])
                rows.append({"module": module, "pos": p, "idx": c, "score": float(s[p, c])})
        rows.sort(key=lambda r: -abs(r["score"]))  # pyright: ignore[reportArgumentType]
        return rows[:k]


def component_target(trace: PromptTrace, module: str, pos: int, idx: int) -> Tensor:
    """The scalar `a^tau_{pos,idx}` -- one component's activation, as a TARGET to explain."""
    assert module in trace.cache.component_acts_pre, f"{module} is not a decomposed module"  # pyright: ignore[reportAttributeAccessIssue]
    return trace.cache.component_acts_pre[module][0, pos, idx]  # pyright: ignore[reportAttributeAccessIssue]


def node_scores(trace: PromptTrace, target: Callable[[Tensor], Tensor] | Tensor) -> NodeScores:
    """grad x act for every component, error and embedding node reachable from one scalar target."""
    cache = trace.cache
    t = target(cache.logits) if callable(target) else target  # pyright: ignore[reportAttributeAccessIssue]
    assert t.ndim == 0, f"target must be a scalar, got shape {tuple(t.shape)}"

    leaves: dict[str, Tensor] = {f"c::{k}": v for k, v in cache.component_acts.items()}  # pyright: ignore[reportAttributeAccessIssue]
    leaves |= {f"e::{k}": v for k, v in cache.errors.items()}  # pyright: ignore[reportAttributeAccessIssue]
    if cache.embed is not None:  # pyright: ignore[reportAttributeAccessIssue]
        leaves["m::embed"] = cache.embed  # pyright: ignore[reportAttributeAccessIssue]

    names = list(leaves)
    grads = torch.autograd.grad(
        t, [leaves[n] for n in names], retain_graph=True, allow_unused=True
    )
    comps, errs, embed, unreachable = {}, {}, None, set()
    for name, g in zip(names, grads, strict=True):
        kind, _, module = name.partition("::")
        if g is None:
            # Structurally unreachable, NOT zero. See `NodeScores`.
            if kind == "c":
                unreachable.add(module)
            continue
        w = (g * leaves[name])[0].detach().float()
        if kind == "c":
            comps[module] = w
        elif kind == "e":
            errs[module] = w.sum(dim=-1)
        else:
            embed = w.sum(dim=-1)
    return NodeScores(float(t.item()), comps, errs, embed, frozenset(unreachable))


# ---------------------------------------------------------------- QK: the attention score itself


@dataclass(frozen=True)
class QKSetup:
    """Everything `qk_*` needs for one (layer, head), resolved once."""

    layer: int
    head: int
    q_module: str
    k_module: str
    scale: float
    uq: Tensor  # [C_q, d_h]  this head's slice of every q component's write direction
    uk: Tensor  # [C_k, d_h]
    eq: Tensor  # [P, C_q]    effective activation g * a
    ek: Tensor  # [P, C_k]


def qk_setup(trace: PromptTrace, layer: int, head: int) -> QKSetup:
    q_module, k_module = trace.module_at(layer, "attn.q"), trace.module_at(layer, "attn.k")
    d_h = trace.head_dim
    trace.check_head(head)
    assert "gpt2" in trace.model_name.lower(), (
        f"the QK decomposition is written against GPT-2 attention; {trace.model_name} needs its "
        "own scale, mask and logit-capping rules before these numbers mean anything"
    )
    cfg = trace.model.target_model.config  # pyright: ignore[reportAttributeAccessIssue]
    assert getattr(cfg, "scale_attn_weights", True), "attention logits are unscaled on this model"
    assert not getattr(cfg, "scale_attn_by_inverse_layer_idx", False), (
        "this model scales attention logits by 1/(layer+1); the DDIS scale here does not"
    )
    sl = slice(head * d_h, (head + 1) * d_h)
    return QKSetup(
        layer=layer,
        head=head,
        q_module=q_module,
        k_module=k_module,
        scale=d_h**-0.5,
        uq=trace.directions(q_module, "write")[:, sl],
        uk=trace.directions(k_module, "write")[:, sl],
        eq=trace.effective(q_module),
        ek=trace.effective(k_module),
    )


def qk_pair(setup: QKSetup, c_q: int, c_k: int) -> Tensor:
    """`[P, P]` -- one component pair's contribution to this head's attention score, un-summed."""
    g = float(setup.uq[c_q] @ setup.uk[c_k])
    return setup.scale * g * torch.outer(setup.eq[:, c_q], setup.ek[:, c_k])


def qk_term(
    setup: QKSetup, trace: PromptTrace, kind: str, c_q: int | None, c_k: int | None
) -> Tensor:
    """`[P, P]` -- ANY row of `qk_top_pairs`, un-summed, indexed `[query, key]`."""
    d_h = trace.head_dim
    sl = slice(setup.head * d_h, (setup.head + 1) * d_h)
    qb, kb = trace.out_bias(setup.q_module)[sl], trace.out_bias(setup.k_module)[sl]
    ones = torch.ones(trace.n_pos)
    if kind == "pair":
        assert c_q is not None and c_k is not None, "a `pair` row needs both component indices"
        return qk_pair(setup, c_q, c_k)
    if kind == "k_bias":
        assert c_q is not None, "a `k_bias` row is a QUERY component against the key bias"
        return setup.scale * float(setup.uq[c_q] @ kb) * torch.outer(setup.eq[:, c_q], ones)
    if kind == "q_bias":
        assert c_k is not None, "a `q_bias` row is the query bias against a KEY component"
        return setup.scale * float(qb @ setup.uk[c_k]) * torch.outer(ones, setup.ek[:, c_k])
    if kind == "bias_bias":
        return setup.scale * float(qb @ kb) * torch.outer(ones, ones)
    if kind == "error":
        terms = qk_reconstruct(setup, trace)
        return terms.full - terms.explained
    raise AssertionError(
        f"unknown row kind {kind!r}; use pair, k_bias, q_bias, bias_bias or error"
    )


def qk_top_pairs(
    setup: QKSetup, trace: PromptTrace, t: int, t_key: int, k: int | None = 25
) -> list[dict[str, object]]:
    """What drives `Z[t, t_key]`, largest `|contribution|` first. `k=None` returns every row."""
    d_h, head = trace.head_dim, setup.head
    sl = slice(head * d_h, (head + 1) * d_h)
    qm, km = trace.out_bias(setup.q_module)[sl], trace.out_bias(setup.k_module)[sl]

    cq, ck = trace.fired(setup.q_module, t), trace.fired(setup.k_module, t_key)
    rows: list[dict[str, object]] = []
    if cq.numel() and ck.numel():
        gram = setup.uq[cq] @ setup.uk[ck].T
        contrib = setup.scale * torch.outer(setup.eq[t, cq], setup.ek[t_key, ck]) * gram
        flat = contrib.abs().flatten()
        for i in torch.topk(flat, flat.numel() if k is None else min(k, flat.numel())).indices.tolist():
            a, b = divmod(int(i), contrib.shape[1])
            rows.append({"kind": "pair", "q_idx": int(cq[a]), "k_idx": int(ck[b]),
                         "contribution": float(contrib[a, b]), "gram": float(gram[a, b])})

    def against_bias(kind: str, idx: Tensor, weights: Tensor, dirs: Tensor, bias: Tensor) -> None:
        """One side's components against the OTHER side's `b_out`."""
        if idx.numel() == 0:
            return
        vals = setup.scale * weights[idx] * (dirs[idx] @ bias)
        for j in torch.topk(vals.abs(), vals.numel() if k is None else min(k, vals.numel())).indices.tolist():
            c = int(idx[j])
            rows.append({
                "kind": kind,
                "q_idx": c if kind == "k_bias" else None,
                "k_idx": c if kind == "q_bias" else None,
                "contribution": float(vals[j]), "gram": None,
            })

    against_bias("k_bias", cq, setup.eq[t], setup.uq, km)
    against_bias("q_bias", ck, setup.ek[t_key], setup.uk, qm)
    rows.append({"kind": "bias_bias", "q_idx": None, "k_idx": None,
                 "contribution": float(setup.scale * (qm @ km)), "gram": None})

    terms = qk_reconstruct(setup, trace)
    rows.append({"kind": "error", "q_idx": None, "k_idx": None,
                 "contribution": float(terms.full[t, t_key] - terms.explained[t, t_key]),
                 "gram": None})
    rows.sort(key=lambda r: -abs(r["contribution"]))  # pyright: ignore[reportArgumentType]
    return rows if k is None else rows[:k]


@dataclass(frozen=True)
class QKTerms:
    """`Z` split by NODE KIND. All `[P, P]`, indexed `[query, key]`."""

    cc: Tensor  # component x component -- the sum of every DDIS pair
    explained: Tensor
    full: Tensor

    def _centred(self, z: Tensor) -> Tensor:
        """Row-centre over the causally allowed keys."""
        mask = torch.ones_like(z, dtype=torch.bool).tril()
        z = z.masked_fill(~mask, 0.0)
        mean = z.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True)
        return (z - mean).masked_fill(~mask, 0.0)

    def _ratio(self, a: Tensor, b: Tensor) -> float:
        num, den = self._centred(a).norm(), self._centred(b).norm()
        return float(num / den.clamp_min(torch.finfo(den.dtype).tiny))

    def pair_magnitude(self) -> float:
        """`||cc|| / ||full||` on the softmax-relevant part. **Not a share, not capped at 1.**"""
        return self._ratio(self.cc, self.full)

    def residual(self) -> float:
        """`||full - explained|| / ||full||`: the ERROR node's share, the honest coverage number."""
        return self._ratio(self.full - self.explained, self.full)


def qk_reconstruct(setup: QKSetup, trace: PromptTrace) -> QKTerms:
    """Split this head's attention score by node kind."""
    d_h, head = trace.head_dim, setup.head
    sl = slice(head * d_h, (head + 1) * d_h)

    def side(module: str, e: Tensor, u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return e @ u, trace.out_bias(module)[sl].expand(trace.n_pos, -1), trace.errors[module][:, sl]

    q_c, q_m, q_e = side(setup.q_module, setup.eq, setup.uq)
    k_c, k_m, k_e = side(setup.k_module, setup.ek, setup.uk)
    s = setup.scale
    cc = s * (q_c @ k_c.T)
    explained = cc + s * ((q_c @ k_m.T) + (q_m @ k_c.T) + (q_m @ k_m.T))
    full = s * ((q_c + q_m + q_e) @ (k_c + k_m + k_e).T)
    return QKTerms(cc=cc, explained=explained, full=full)


def qk_weight_edit(setup: QKSetup, trace: PromptTrace, c_q: int, c_k: int) -> Tensor:
    """`[P, P]` -- this head's score with `P_c = u_c v_c^T` DELETED from the Q and K weights."""
    d_h = trace.head_dim
    sl = slice(setup.head * d_h, (setup.head + 1) * d_h)

    def side(module: str, e: Tensor, u: Tensor, c: int) -> Tensor:
        full = e @ u + trace.out_bias(module)[sl] + trace.errors[module][:, sl]
        v = trace.model.components[module].V[:, c].detach().float()  # pyright: ignore[reportAttributeAccessIssue]
        proj = trace.inputs[module] @ v  # [P]
        return full - torch.outer(proj, u[c])

    q = side(setup.q_module, setup.eq, setup.uq, c_q)
    k = side(setup.k_module, setup.ek, setup.uk, c_k)
    return setup.scale * (q @ k.T)


def attn_from_scores(z: Tensor) -> Tensor:
    """Causal-mask and softmax a `[P, P]` score matrix, as the model does."""
    mask = torch.ones_like(z, dtype=torch.bool).tril()
    return torch.softmax(z.masked_fill(~mask, float("-inf")), dim=-1)


# ---------------------------------------------------------------- OV: what the attention moves


def ov_alignment(trace: PromptTrace, layer: int, head: int, c_v: int, c_o: int) -> Tensor:
    """`[P, P]` -- `attn[q, k] * e_v[k, c_v] * <U_v, V_o> * g_o[q, c_o]`, un-summed over positions."""
    v_module, o_module = trace.module_at(layer, "attn.v"), trace.module_at(layer, "attn.o")
    d_h = trace.head_dim
    trace.check_head(head)
    sl = slice(head * d_h, (head + 1) * d_h)
    g = float(trace.directions(v_module, "write")[c_v, sl] @ trace.directions(o_module, "read")[c_o, sl])
    delivered = trace.attn_probs[layer][head] * trace.effective(v_module)[:, c_v]  # [q, k]
    return delivered * g * trace.gates[o_module][:, c_o].unsqueeze(1)


def qk_head_profile(
    trace: PromptTrace, layer: int, kind: str, c_q: int | None, c_k: int | None,
    t: int, t_key: int,
) -> Tensor:
    """`[H]` -- one `qk_top_pairs` row's contribution to `Z[t, t_key]` on EVERY head of the layer."""
    q_module, k_module = trace.module_at(layer, "attn.q"), trace.module_at(layer, "attn.k")
    d_h = trace.head_dim
    s = d_h**-0.5
    per_head = lambda v: v.unflatten(0, (v.shape[0] // d_h, d_h))  # [H, d_h]

    if kind == "error":
        # Every cross-term touching the error node, head by head: `full - explained` at one cell.
        def sides(module: str, pos: int) -> tuple[Tensor, Tensor]:
            comp = trace.effective(module)[pos] @ trace.directions(module, "write")
            explained = comp + trace.out_bias(module)
            return explained + trace.errors[module][pos], explained

        q_full, q_exp = sides(q_module, t)
        k_full, k_exp = sides(k_module, t_key)
        dot = lambda a, b: (per_head(a) * per_head(b)).sum(dim=-1)
        return s * (dot(q_full, k_full) - dot(q_exp, k_exp))

    if kind == "pair":
        assert c_q is not None and c_k is not None, "a `pair` row needs both component indices"
    if kind == "k_bias":
        assert c_q is not None and c_k is None, "a `k_bias` row is a query component vs the key bias"
    if kind == "q_bias":
        assert c_k is not None and c_q is None, "a `q_bias` row is the query bias vs a key component"
    if kind == "bias_bias":
        assert c_q is None and c_k is None, "a `bias_bias` row takes no component index"
    if kind not in ("pair", "k_bias", "q_bias", "bias_bias"):
        raise AssertionError(
            f"unknown row kind {kind!r}; use pair, k_bias, q_bias, bias_bias or error"
        )

    qv = trace.directions(q_module, "write")[c_q] if c_q is not None else trace.out_bias(q_module)
    kv = trace.directions(k_module, "write")[c_k] if c_k is not None else trace.out_bias(k_module)
    w = 1.0
    if c_q is not None:
        w *= float(trace.effective(q_module)[t, c_q])
    if c_k is not None:
        w *= float(trace.effective(k_module)[t_key, c_k])
    return s * w * (per_head(qv) * per_head(kv)).sum(dim=-1)


def qk_listed_total(setup: QKSetup, trace: PromptTrace, t: int, t_key: int) -> Tensor:
    """`[P, P]` -- the sum of EVERY row `qk_top_pairs(t, t_key)` lists, as one matrix."""
    d_h = trace.head_dim
    sl = slice(setup.head * d_h, (setup.head + 1) * d_h)
    qb, kb = trace.out_bias(setup.q_module)[sl], trace.out_bias(setup.k_module)[sl]
    cq, ck = trace.fired(setup.q_module, t), trace.fired(setup.k_module, t_key)
    ones = torch.ones(trace.n_pos)
    s = setup.scale

    # Each side's contribution from ONLY the components the table lists.
    q_c = setup.eq[:, cq] @ setup.uq[cq] if cq.numel() else torch.zeros(trace.n_pos, len(qb))
    k_c = setup.ek[:, ck] @ setup.uk[ck] if ck.numel() else torch.zeros(trace.n_pos, len(kb))

    pairs = s * (q_c @ k_c.T)
    k_bias = s * torch.outer(q_c @ kb, ones)  # every listed `k_bias` row
    q_bias = s * torch.outer(ones, k_c @ qb)  # every listed `q_bias` row
    bias_bias = s * float(qb @ kb) * torch.outer(ones, ones)
    terms = qk_reconstruct(setup, trace)
    error = terms.full - terms.explained  # the `error` row IS the whole error group
    return pairs + k_bias + q_bias + bias_bias + error


def atp_to_module(
    trace: PromptTrace, a_module: str, a_idx: int, a_pos: int, b_module: str
) -> dict[str, object]:
    """grad x act from one component of `a` BACKWARD to every component of `b`, every position."""
    assert a_module in trace.acts, f"{a_module} is not a decomposed module"
    assert b_module in trace.acts, f"{b_module} is not a decomposed module"
    trace.check_pos(a_pos)
    n_a = int(trace.acts[a_module].shape[1])
    assert 0 <= a_idx < n_a, f"component {a_idx} is outside 0..{n_a - 1}"

    scores = node_scores(trace, component_target(trace, a_module, a_pos, a_idx))
    reason = None
    if b_module not in scores.components:
        reason = (
            f"no gradient path from {b_module} to {a_module}:{a_idx} -- nothing in {b_module} "
            f"feeds it, so every score here is absent rather than zero"
        )
    act = float(trace.acts[a_module][a_pos, a_idx])
    return {
        "src_module": a_module, "src_idx": a_idx, "src_pos": a_pos,
        "src_act": act, "src_gate": float(trace.gates[a_module][a_pos, a_idx]),
        "src_effective": float(trace.effective(a_module)[a_pos, a_idx]),
        "dst_module": b_module, "n_pos": trace.n_pos,
        "n_components": int(trace.acts[b_module].shape[1]),
        "target": float(scores.target),
        "reason": reason,
        "scores": scores.components.get(b_module),
    }


def atp_rows(
    trace: PromptTrace, res: dict[str, object], *, sort: str = "abs",
    offset: int = 0, limit: int = 100,
) -> dict[str, object]:
    """One page of `atp_to_module`'s `[P, C]` scores, as rows, largest first."""
    scores = res["scores"]
    if scores is None:
        return {"rows": [], "n_total": 0, "offset": 0, "limit": limit}
    assert isinstance(scores, Tensor)
    dst = str(res["dst_module"])
    gates = trace.gates[dst]
    live = (scores != 0).nonzero(as_tuple=False)  # [(pos, idx)]
    vals = scores[live[:, 0], live[:, 1]]
    key = vals.abs() if sort == "abs" else vals
    order = torch.argsort(key, descending=True)
    n_total = int(order.numel())
    page = order[offset : offset + limit] if limit else order[offset:]
    density = (gates > 0).float().mean(dim=0)
    rows = []
    for i in page.tolist():
        p, c = int(live[i, 0]), int(live[i, 1])
        rows.append({
            "idx": c, "pos": p, "piece": trace.pieces[p],
            "score": float(scores[p, c]),
            "act": float(trace.acts[dst][p, c]),
            "gate": float(gates[p, c]),
            "effective": float(trace.effective(dst)[p, c]),
            "density": float(density[c]),
        })
    return {"rows": rows, "n_total": n_total, "offset": offset, "limit": limit}
