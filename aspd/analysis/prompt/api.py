"""`/api/prompt/*`: the per-prompt endpoints installed onto the pair viewer."""

from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from fastapi import HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from torch import Tensor

from aspd.analysis.circuits.targets import logit_diff, top_k_vs_rest
from aspd.analysis.pairs import features as ft
from aspd.analysis.prompt.chain import attribution_tree, roles_reached
from aspd.analysis.prompt.engine import PromptEngine
from aspd.analysis.prompt.features import feature_acts
from aspd.analysis.prompt.heads import head_mass, headed_side
from aspd.analysis.prompt.interact import interact, series
from aspd.analysis.prompt.page import PROMPT_HTML
from aspd.analysis.prompt.scores import (
    NodeScores,
    atp_rows,
    atp_to_module,
    attn_from_scores,
    node_scores,
    ov_alignment,
    qk_head_profile,
    qk_listed_total,
    qk_reconstruct,
    qk_setup,
    qk_term,
    qk_weight_edit,
    qk_top_pairs,
)
from aspd.analysis.prompt.total import TotalEffect, total_effect
from aspd.analysis.prompt.trace import PromptTrace


@contextmanager
def _bad_input() -> Iterator[None]:
    """Turn a rejected argument into 422, not 500."""
    try:
        yield
    except (AssertionError, IndexError, KeyError) as exc:
        raise HTTPException(422, str(exc) or type(exc).__name__) from exc


def _target_fn(kind: str, engine: PromptEngine, correct: str, wrong: str) -> Callable:
    """The scalar a circuit explains. `logit_diff` is IOI's metric; `topk` needs no arguments."""
    if kind == "topk":
        return lambda z: top_k_vs_rest(z, k=10)
    if kind == "logit_diff":
        if not (correct and wrong):
            raise HTTPException(422, "logit_diff needs `correct` and `wrong` token strings")
        try:
            c, w = engine.single_token(correct), engine.single_token(wrong)
        except AssertionError as exc:
            raise HTTPException(422, str(exc)) from exc
        return lambda z: logit_diff(z, correct=c, wrong=w)
    raise HTTPException(422, f"unknown target {kind!r}; use 'topk' or 'logit_diff'")


def install_prompt_api(app, engine: PromptEngine, run_name: str, saes=None, norms=None,
                       harvest=None, qk_edit: str = "gated") -> None:
    """Mount the prompt routes and the `/prompt` page onto an existing FastAPI app."""

    def _has_saes() -> bool:
        return saes is not None and norms is not None

    def need_saes():
        if not _has_saes():
            raise HTTPException(501, "this server was started without a dictionary store")
        return saes, norms

    _score_cache: OrderedDict[tuple[str, str, str, str], NodeScores] = OrderedDict()

    def resolved(prompt: str, target: str, correct: str, wrong: str
                 ) -> tuple[PromptTrace, NodeScores]:
        """The trace and its node scores, both cached."""
        trace = engine.trace(prompt)
        key = (prompt, target, correct, wrong)
        hit = _score_cache.get(key)
        if hit is None:
            hit = node_scores(trace, _target_fn(target, engine, correct, wrong))
            _score_cache[key] = hit
            while len(_score_cache) > 8:
                _score_cache.popitem(last=False)
        else:
            _score_cache.move_to_end(key)
        return trace, hit

    _total_cache: OrderedDict[tuple[str, str, str, str], TotalEffect] = OrderedDict()

    def resolved_total(prompt: str, target: str, correct: str, wrong: str
                       ) -> tuple[PromptTrace, TotalEffect]:
        """The trace and its MULTI-HOP scores, cached the same way and keyed the same way."""
        trace = engine.trace(prompt)
        key = (prompt, target, correct, wrong)
        hit = _total_cache.get(key)
        if hit is None:
            hit = total_effect(trace, _target_fn(target, engine, correct, wrong))
            _total_cache[key] = hit
            while len(_total_cache) > 4:
                _total_cache.popitem(last=False)
        else:
            _total_cache.move_to_end(key)
        return trace, hit

    @app.get("/api/prompt/trace")
    def trace_route(
        prompt: str,
        target: str = "topk",
        correct: str = "",
        wrong: str = "",
        k: int = Query(25, ge=1, le=200),
    ) -> JSONResponse:
        """Tokens, the target, and where the movement sits -- the page's first request."""
        trace, scores = resolved(prompt, target, correct, wrong)
        per_pos = torch.zeros(trace.n_pos)
        for s in scores.components.values():
            per_pos += s.abs().sum(dim=1)
        err_pos = torch.zeros(trace.n_pos)
        for s in scores.errors.values():
            err_pos += s.abs()
        top = torch.topk(trace.logits[-1], 10)
        return JSONResponse({
            "run": run_name,
            "prompt": prompt,
            "tokens": [{"pos": i, "piece": p, "score": float(per_pos[i]),
                        "error": float(err_pos[i])} for i, p in enumerate(trace.pieces)],
            "target": scores.target,
            "target_kind": target,
            "layers": trace.layers,
            "n_heads": trace.n_heads,
            "predictions": [
                {"token": engine.decode(t), "logit": float(v)}
                for t, v in zip(top.indices.tolist(), top.values.tolist(), strict=True)
            ],
            "top_nodes": _node_rows(scores.top(k), trace),
            "scored_roles": sorted({trace.spaces[m].role for m in scores.components}),
            "unreachable_roles": sorted({trace.spaces[m].role for m in scores.unreachable}),
        })

    def _node_rows(rows: list[dict], trace: PromptTrace) -> list[dict]:
        out = []
        for r in rows:
            module = str(r["module"])
            spec = trace.spaces[module]
            row = dict(r) | {"piece": trace.pieces[int(r["pos"])], "role": spec.role,
                             "layer": spec.layer}
            row["head_mass"] = _head_mass_row(trace, module, int(r["idx"]))
            out.append(row)
        return out

    _mass_cache: dict[str, Tensor | None] = {}

    def _head_mass_all(trace: PromptTrace, module: str) -> Tensor | None:
        """`[C, H]` for a head-structured module, `None` for an MLP or `attn.o`'s write."""
        if module not in _mass_cache:
            spec = trace.spaces[module]
            _mass_cache[module] = (
                None if spec.read.heads is None and spec.write.heads is None
                else head_mass(trace.directions(module, (side := headed_side(spec))), spec, side)
            )
        return _mass_cache[module]

    def _head_mass_row(trace: PromptTrace, module: str, idx: int) -> list[float] | None:
        mass = _head_mass_all(trace, module)
        return None if mass is None else mass[idx].tolist()

    @app.get("/api/prompt/fired")
    def fired_route(
        prompt: str,
        pos: int,
        target: str = "topk",
        correct: str = "",
        wrong: str = "",
        module: str = "",
        k: int = Query(50, ge=1, le=500),
    ) -> JSONResponse:
        """Every component that fired at one position, ranked by |node score|."""
        trace, scores = resolved(prompt, target, correct, wrong)
        if not 0 <= pos < trace.n_pos:
            raise HTTPException(404, f"position {pos} out of range (0..{trace.n_pos - 1})")
        modules = [module] if module else trace.modules
        rows = []
        for m in modules:
            if m not in trace.spaces:
                raise HTTPException(404, f"{m} is not a decomposed module of this run")
            idx = trace.fired(m, pos)
            if idx.numel() == 0:
                continue
            dens = (harvest.densities(m, trace.acts[m].shape[1])
                    if harvest is not None and harvest.available else None)
            s = scores.components.get(m)
            eff = trace.effective(m)[pos]
            for c in idx.tolist():
                rows.append({"module": m, "idx": int(c), "pos": pos,
                             "score": None if s is None else float(s[pos, c]),
                             "unreachable": s is None,
                             "act": float(trace.acts[m][pos, c]),
                             "density": None if dens is None else float(dens[c]),
                             "effective": float(eff[c]),
                             "gate": float(trace.gates[m][pos, c]),
                             "role": trace.spaces[m].role, "layer": trace.spaces[m].layer})
        # Unscorable rows sort last, ordered by what they actually contributed at this token.
        rows.sort(key=lambda r: (r["score"] is None, -abs(r["score"] if r["score"] is not None
                                                          else r["effective"])))
        return JSONResponse({"pos": pos, "piece": trace.pieces[pos], "n_fired": len(rows),
                             "rows": rows[:k]})

    @app.get("/api/prompt/browse")
    def browse_route(
        prompt: str,
        pos: int,
        target: str = "topk",
        correct: str = "",
        wrong: str = "",
        module: str = "",
        k: int = Query(50, ge=1, le=500),
    ) -> JSONResponse:
        """`/api/prompt/fired` with a TOTAL-effect score in place of the one-hop one. The browse panel."""
        trace = engine.trace(prompt)
        if not 0 <= pos < trace.n_pos:
            raise HTTPException(404, f"position {pos} out of range (0..{trace.n_pos - 1})")
        modules = [module] if module else trace.modules
        for m in modules:
            if m not in trace.spaces:
                raise HTTPException(404, f"{m} is not a decomposed module of this run")
        _, totals = resolved_total(prompt, target, correct, wrong)
        rows = []
        for m in modules:
            idx = trace.fired(m, pos)
            if idx.numel() == 0:
                continue
            dens = (harvest.densities(m, trace.acts[m].shape[1])
                    if harvest is not None and harvest.available else None)
            tot, eff = totals.components[m][pos], trace.effective(m)[pos]
            for c in idx.tolist():
                rows.append({"module": m, "idx": int(c), "pos": pos,
                             "total": float(tot[c]),
                             "act": float(trace.acts[m][pos, c]),
                             "density": None if dens is None else float(dens[c]),
                             "effective": float(eff[c]),
                             "gate": float(trace.gates[m][pos, c]),
                             "role": trace.spaces[m].role, "layer": trace.spaces[m].layer})
        rows.sort(key=lambda r: -abs(r["total"]))
        return JSONResponse({"pos": pos, "piece": trace.pieces[pos], "n_fired": len(rows),
                             "effect": "total", "target": totals.target,
                             "rows": rows[:k]})

    @app.get("/api/prompt/node")
    def node_route(
        prompt: str,
        module: str,
        idx: int,
        target: str = "topk",
        correct: str = "",
        wrong: str = "",
    ) -> JSONResponse:
        """One component's or feature's whole trace on THIS prompt, position by position."""
        trace = engine.trace(prompt)
        if ft.is_sae_key(module):
            store, _ = need_saes()
            acts = feature_acts(trace, store, module)[:, idx]
            endpoint = store.endpoint(module)
            return JSONResponse({
                "module": module, "idx": idx, "kind": "sae",
                "role": endpoint.site, "layer": endpoint.layer,
                "n_components": int(endpoint.d_sae),
                "pieces": trace.pieces,
                "act": acts.tolist(), "gate": None, "effective": acts.tolist(),
                "score": None, "unreachable": True, "head_mass": None,
                "reason": "a dictionary is not part of the replacement model: no gate, no gradient",
            })
        if module not in trace.spaces:
            raise HTTPException(404, f"{module} is not a decomposed module of this run")
        spaces = trace.spaces[module]
        with _bad_input():
            assert 0 <= idx < trace.acts[module].shape[1], (
                f"component {idx} is outside 0..{trace.acts[module].shape[1] - 1} of {module}"
            )
        _, scores = resolved(prompt, target, correct, wrong)
        s = scores.components.get(module)
        mass = None
        try:
            side = headed_side(spaces)
            mass = head_mass(trace.directions(module, side), spaces, side)[idx].tolist()
        except AssertionError:
            pass
        return JSONResponse({
            "module": module, "idx": idx, "kind": "component",
            "role": spaces.role, "layer": spaces.layer,
            "n_components": int(trace.acts[module].shape[1]),
            "pieces": trace.pieces,
            "act": trace.acts[module][:, idx].tolist(),
            "gate": trace.gates[module][:, idx].tolist(),
            "effective": trace.effective(module)[:, idx].tolist(),
            "score": None if s is None else s[:, idx].tolist(),
            "unreachable": s is None,
            "head_mass": mass,
        })

    @app.get("/api/prompt/attn")
    def attn_route(prompt: str, layer: int, head: int) -> JSONResponse:
        """The head's attention pattern, and how much of it the decomposition accounts for."""
        trace = engine.trace(prompt)
        with _bad_input():
            terms = qk_reconstruct(qk_setup(trace, layer, head), trace)
            probs = trace.attn_probs[layer][head]
        return JSONResponse({
            "layer": layer, "head": head,
            "pieces": trace.pieces,
            "attn": probs.tolist(),
            "z": terms.full.tolist(),
            "residual": terms.residual(),
            "pair_magnitude": terms.pair_magnitude(),
            "recon_error": float((attn_from_scores(terms.full) - probs).abs().max()),
        })

    @app.get("/api/prompt/qk")
    def qk_route(
        prompt: str, layer: int, head: int, t: int, t_key: int,
        k: int = Query(25, ge=1, le=5000),
    ) -> JSONResponse:
        """What drives `Z[t, t_key]`. Rows sum to `Z`; `kind` says which node kind each row is."""
        trace = engine.trace(prompt)
        if not (0 <= t < trace.n_pos and 0 <= t_key < trace.n_pos):
            raise HTTPException(404, "position out of range")
        if t_key > t:
            raise HTTPException(422, f"key {t_key} is after query {t}; attention is causal")
        with _bad_input():
            setup = qk_setup(trace, layer, head)
        terms = qk_reconstruct(setup, trace)
        every = qk_top_pairs(setup, trace, t, t_key, k=None)
        rows = every[:k]
        for r in rows:
            for side, mod in (("q_idx", setup.q_module), ("k_idx", setup.k_module)):
                if r.get(side) is not None:
                    r[f"{side[0]}_head_mass"] = _head_mass_row(trace, mod, int(r[side]))  # pyright: ignore[reportArgumentType]
        return JSONResponse({
            "layer": layer, "head": head, "t": t, "t_key": t_key,
            "q_piece": trace.pieces[t], "k_piece": trace.pieces[t_key],
            "q_module": setup.q_module, "k_module": setup.k_module,
            "z": float(terms.full[t, t_key]),
            "attn": float(trace.attn_probs[layer][head][t, t_key]),
            "residual": terms.residual(),
            "n_rows": len(every),
            "omitted": sum(float(r["contribution"]) for r in every[k:]),
            "rows": rows,
        })

    @app.get("/api/prompt/qk_pair")
    def qk_pair_route(
        prompt: str, layer: int, head: int,
        c_q: int | None = None, c_k: int | None = None, kind: str = "pair",
        t: int | None = None, t_key: int | None = None,
        edit: str | None = None,
    ) -> JSONResponse:
        """ANY `qk` row's `[P, P]` contribution, un-summed -- the picture that names the behaviour."""
        trace = engine.trace(prompt)
        with _bad_input():
            setup = qk_setup(trace, layer, head)
            term = qk_term(setup, trace, kind, c_q, c_k)
            profile = (
                None if t is None or t_key is None
                else qk_head_profile(trace, layer, kind, c_q, c_k, t, t_key).tolist()
            )
        full = qk_reconstruct(setup, trace).full
        causal = torch.ones_like(full, dtype=torch.bool).tril()
        mode = (edit or qk_edit).lower()
        with _bad_input():
            assert mode in ("gated", "weight"), f"edit must be gated or weight, got {mode!r}"
        weighted = mode == "weight" and kind == "pair"
        with _bad_input():
            z_off = (qk_weight_edit(setup, trace, c_q, c_k)  # pyright: ignore[reportArgumentType]
                     if weighted else full - term)
        return JSONResponse({
            "layer": layer, "head": head, "kind": kind, "c_q": c_q, "c_k": c_k,
            "edit": "weight" if weighted else "gated",
            "pieces": trace.pieces,
            "matrix": term.tolist(),
            "peak": float(term.masked_fill(~causal, 0.0).abs().max()),
            "z_max": float(full.masked_fill(~causal, 0.0).abs().max()),
            "attn_true": attn_from_scores(full).tolist(),
            "attn_off": attn_from_scores(z_off).tolist(),
            "per_head": profile,
            "at": None if profile is None else {
                "t": t, "t_key": t_key, "head": head, "value": profile[head],
            },
        })

    @app.get("/api/prompt/subset")
    def subset_route(
        prompt: str, layer: int, head: int, rows: str, t: int, t_key: int,
        with_error: bool = False, with_bias: bool = False, complete: bool = False,
    ) -> JSONResponse:
        """Re-softmax the attention pattern from a CHOSEN SUBSET of `qk` rows."""
        trace = engine.trace(prompt)
        with _bad_input():
            setup = qk_setup(trace, layer, head)
            picked = []
            for spec in rows.split(","):
                if not spec.strip():
                    continue
                parts = spec.split(":")
                assert len(parts) == 3, f"{spec!r} is not <kind>:<c_q>:<c_k>"
                kind, cq, ck = parts[0], parts[1], parts[2]
                picked.append((kind, int(cq) if cq else None, int(ck) if ck else None))
            assert picked, "choose at least one row"
            total = torch.zeros(trace.n_pos, trace.n_pos)
            for kind, cq, ck in picked:
                total = total + qk_term(setup, trace, kind, cq, ck)
        terms = qk_reconstruct(setup, trace)
        if complete:
            total = terms.full - (qk_listed_total(setup, trace, t, t_key) - total)
        else:
            if with_bias:
                total = total + (terms.explained - terms.cc)
            if with_error:
                total = total + (terms.full - terms.explained)
        built, true = attn_from_scores(total), trace.attn_probs[layer][head]
        return JSONResponse({
            "layer": layer, "head": head, "n_rows": len(picked),
            "with_error": with_error and not complete, "with_bias": with_bias and not complete,
            "complete": complete,
            "pieces": trace.pieces,
            "z": total.tolist(),
            "attn": built.tolist(),
            # The model's own pattern, so the page never has to guess what it is comparing against.
            "attn_true": true.tolist(),
            "attn_off": attn_from_scores(terms.full - total).tolist(),
            "z_true": terms.full.tolist(),
            "gap": float((built - true).abs().max()),
            "at": {"t": t, "t_key": t_key, "z": float(total[t, t_key]),
                   "z_true": float(terms.full[t, t_key]),
                   "omitted": float(terms.full[t, t_key]) - float(total[t, t_key])},
        })

    @app.get("/api/prompt/ov")
    def ov_route(prompt: str, layer: int, head: int, c_v: int, c_o: int) -> JSONResponse:
        trace = engine.trace(prompt)
        with _bad_input():
            matrix = ov_alignment(trace, layer, head, c_v, c_o).tolist()
        return JSONResponse({
            "layer": layer, "head": head, "c_v": c_v, "c_o": c_o,
            "pieces": trace.pieces,
            "matrix": matrix,
        })

    @app.get("/api/prompt/chain")
    def chain_route(
        prompt: str,
        target: str = "topk",
        correct: str = "",
        wrong: str = "",
        depth: int = Query(2, ge=1, le=4),
        width: int = Query(5, ge=1, le=20),
    ) -> JSONResponse:
        """Multi-hop attribution. **Rank by `effect`, not `score`** -- see `ChainEdge`."""
        trace = engine.trace(prompt)
        edges, stats = attribution_tree(
            trace, _target_fn(target, engine, correct, wrong), depth=depth, width=width
        )
        return JSONResponse({
            "prompt": prompt, "depth": depth, "width": width,
            "pieces": trace.pieces,
            "stats": stats,
            "roles_reached": roles_reached(trace, edges),
            "units": {
                "score": "the local edge, in units of its PARENT; NOT comparable across depths",
                "effect": "the same edge composed back to the root target; rank with this",
            },
            "edges": [e.as_row() for e in edges],
        })

    @app.get("/api/prompt/features")
    def features_route(
        prompt: str, sae: str, pos: int, k: int = Query(50, ge=1, le=500)
    ) -> JSONResponse:
        """Every feature live at one token, strongest first."""
        store, _ = need_saes()
        trace = engine.trace(prompt)
        if not 0 <= pos < trace.n_pos:
            raise HTTPException(404, f"position {pos} out of range (0..{trace.n_pos - 1})")
        acts = feature_acts(trace, store, sae)[pos]
        live = (acts != 0).nonzero(as_tuple=False).flatten()
        top = torch.topk(acts[live].abs(), min(k, live.numel())) if live.numel() else None
        rows = [] if top is None else [
            {"idx": int(live[i]), "act": float(acts[live[i]])} for i in top.indices.tolist()
        ]
        endpoint = store.endpoint(sae)
        return JSONResponse({
            "sae": sae, "site": endpoint.site, "layer": endpoint.layer,
            "d_sae": endpoint.d_sae, "pos": pos, "piece": trace.pieces[pos],
            "n_live": int(live.numel()), "rows": rows,
        })

    @app.get("/api/prompt/interact")
    def interact_route(
        prompt: str,
        a: str,
        b: str,
        pos: int,
        a_side: str = "write",
        b_side: str = "read",
        src: int | None = None,
        head: int | None = None,
        a_idx: int | None = None,
        b_idx: int | None = None,
        k: int = Query(25, ge=1, le=200),
    ) -> JSONResponse:
        """`(g_a a_a) · ⟨dir_a, dir_b⟩ · g_b` — the pair viewer's `dot_coact` on one prompt."""
        store, target_norms = (saes, norms) if _has_saes() else (None, None)
        trace = engine.trace(prompt)
        for key in (a, b):
            if not ft.is_sae_key(key) and key not in trace.spaces:
                raise HTTPException(404, f"{key} is not a decomposed module of this run")
        if (ft.is_sae_key(a) or ft.is_sae_key(b)) and store is None:
            raise HTTPException(501, "this server was started without a dictionary store")
        if not 0 <= pos < trace.n_pos:
            raise HTTPException(404, f"position {pos} out of range (0..{trace.n_pos - 1})")
        try:
            return JSONResponse(interact(
                trace, a, b, a_side=a_side, b_side=b_side, pos=pos, src=src, head=head,  # pyright: ignore[reportArgumentType]
                a_idx=a_idx, b_idx=b_idx, saes=store, norms=target_norms, k=k,
            ))
        except AssertionError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/prompt/atp")
    def atp_route(
        prompt: str,
        a: str,
        b: str,
        a_idx: int,
        pos: int,
        sort: str = "abs",
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=0, le=1000),
    ) -> JSONResponse:
        """grad × act from one component of `a` BACKWARD to every component of `b`."""
        trace = engine.trace(prompt)
        for key in (a, b):
            if key not in trace.spaces:
                raise HTTPException(404, f"{key} is not a decomposed module of this run")
        if not 0 <= pos < trace.n_pos:
            raise HTTPException(404, f"position {pos} out of range (0..{trace.n_pos - 1})")
        if sort not in ("abs", "signed"):
            raise HTTPException(422, f"sort must be 'abs' or 'signed', got {sort!r}")
        try:
            res = atp_to_module(trace, a, a_idx, pos, b)
        except AssertionError as exc:
            raise HTTPException(422, str(exc)) from exc
        page = atp_rows(trace, res, sort=sort, offset=offset, limit=limit)
        return JSONResponse({k: v for k, v in res.items() if k != "scores"} | page
                            | {"piece": trace.pieces[pos], "sort": sort})

    @app.get("/api/prompt/interact_series")
    def interact_series_route(
        prompt: str,
        a: str,
        b: str,
        a_idx: int,
        b_idx: int,
        a_side: str = "write",
        b_side: str = "read",
        head: int | None = None,
    ) -> JSONResponse:
        """One pair's score over positions, **un-summed**."""
        store, target_norms = (saes, norms) if _has_saes() else (None, None)
        trace = engine.trace(prompt)
        try:
            out = series(trace, a, b, a_idx, b_idx, a_side=a_side, b_side=b_side,  # pyright: ignore[reportArgumentType]
                         head=head, saes=store, norms=target_norms)
        except AssertionError as exc:
            raise HTTPException(422, str(exc)) from exc
        return JSONResponse({"a": a, "b": b, "a_idx": a_idx, "b_idx": b_idx, "head": head,
                             "pieces": trace.pieces, "shape": list(out.shape),
                             "values": out.tolist()})

    @app.post("/api/prompt/clientlog")
    def clientlog_route(msg: str, where: str = "") -> JSONResponse:
        """Browser-side failures, printed to the server's stdout."""
        print(f"[client] {where + ': ' if where else ''}{msg}", flush=True)
        return JSONResponse({"ok": True})

    @app.get("/prompt", response_class=HTMLResponse)
    def prompt_page() -> HTMLResponse:
        return HTMLResponse(PROMPT_HTML.replace("__RUN__", run_name))
