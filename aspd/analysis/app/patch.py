"""Runtime patches to the lab's attribution app for model-wide decompositions."""

import contextvars

DEFAULT_EDGE_CAP = 200_000


def install_graph_edge_cap(limit: int = DEFAULT_EDGE_CAP) -> None:
    """Keep only the `limit` strongest edges per list before a graph is written to SQLite."""
    from param_decomp_lab.app.backend.database import PromptAttrDB

    if getattr(PromptAttrDB.save_graph, "_aspd_edge_capped", False):
        return

    stock_save_graph = PromptAttrDB.save_graph

    def _save_graph(self, prompt_id, graph):
        def _cap(edges):
            if edges is None or len(edges) <= limit:
                return edges
            return sorted(edges, key=lambda e: -abs(e.strength))[:limit]

        n_before = len(graph.edges)
        capped = graph.model_copy(
            update={"edges": _cap(graph.edges), "edges_abs": _cap(graph.edges_abs)}
        )
        if n_before > limit:
            from param_decomp.log import logger

            logger.info(
                f"[aspd] edge cap: {n_before} -> {limit} edges kept for save "
                f"(display limit is 50k; re-run the prompt to recompute the full set)"
            )
        return stock_save_graph(self, prompt_id, capped)

    _rebind_save_graph(_save_graph, stock_save_graph, "_aspd_edge_capped")


def install_attribution_mask_is_ci() -> None:
    """Evaluate attribution gradients at `m = g`, not at the lab's `m = 1`."""
    from param_decomp_lab.app.backend import compute as _compute

    if getattr(_compute.compute_edges_from_ci, "_aspd_mask_is_ci", False):
        return

    stock_edges_from_ci = _compute.compute_edges_from_ci

    def _compute_edges_from_ci(*args, **kwargs):
        ci = kwargs["ci_lower_leaky"] if "ci_lower_leaky" in kwargs else args[3]
        stock_make_mask_infos = _compute.make_mask_infos

        def _make_mask_infos(component_masks=None, *a, **kw):
            if kw.get("weight_deltas_and_masks") is not None and component_masks is not None:
                assert set(component_masks) == set(ci), (
                    "gradient-forward masks and CI cover different modules: "
                    f"{sorted(set(component_masks) ^ set(ci))}"
                )
                component_masks = {k: ci[k] for k in component_masks}
                from param_decomp.log import logger

                logger.info(
                    f"[aspd] attribution mask m = g over {len(component_masks)} modules"
                )
            return stock_make_mask_infos(component_masks, *a, **kw)

        _compute.make_mask_infos = _make_mask_infos
        try:
            return stock_edges_from_ci(*args, **kwargs)
        finally:
            _compute.make_mask_infos = stock_make_mask_infos

    _compute_edges_from_ci._aspd_mask_is_ci = True
    _compute.compute_edges_from_ci = _compute_edges_from_ci


def _rebind_save_graph(new_fn, stock_fn, marker: str) -> None:
    """Install `new_fn` as `PromptAttrDB.save_graph`, keeping every marker already on the chain."""
    from param_decomp_lab.app.backend.database import PromptAttrDB

    setattr(new_fn, marker, True)
    for carried in (
        "_aspd_edge_capped",
        "_aspd_sparse_ci",
        "_aspd_graph_cached",
        "_aspd_method",
    ):
        if carried != marker and getattr(stock_fn, carried, False):
            setattr(new_fn, carried, True)
    PromptAttrDB.save_graph = new_fn


DEFAULT_NODE_CAP = 500
DEFAULT_MAX_DENSITY = 1.0

_RANKINGS: "list[tuple[tuple, dict]]" = []
_RANKING_MEMO_SIZE = 4

VIEW_MAX_DENSITY: "contextvars.ContextVar[float | None]" = contextvars.ContextVar(
    "lm_interp_view_max_density", default=None
)
VIEW_NODE_CAP: "contextvars.ContextVar[int | None]" = contextvars.ContextVar(
    "lm_interp_view_node_cap", default=None
)

# id(RunState) -> {canonical_key: firing_density}. One run at a time, so this holds one entry.
_DENSITY_CACHE: "dict[int, dict[str, float]]" = {}


def lookup_ranking(raw_edges: list, ci_threshold: float) -> "dict | None":
    """The most recent ranking for this edge list at this threshold, whatever density/cap made it."""
    for key, ranking in reversed(_RANKINGS):
        if key[0] is raw_edges and key[1] == ci_threshold:
            return ranking
    return None


def current_firing_densities() -> "dict[str, float]":
    """Firing densities for the run the app currently has loaded."""
    from param_decomp_lab.app.backend.state import StateManager

    return canonical_firing_densities(StateManager.get().run_state)


def canonical_firing_densities(run_state) -> "dict[str, float]":
    """`{canonical_key: firing_density}` over every harvested component, memoised per run."""
    key = id(run_state)
    hit = _DENSITY_CACHE.get(key)
    if hit is not None:
        return hit

    from time import perf_counter

    from param_decomp.log import logger

    assert run_state.harvest is not None, (
        "density filtering needs harvest.db, and this run has no harvest loaded"
    )
    t0 = perf_counter()
    canon: dict[str, str] = {}
    out: dict[str, float] = {}
    for summary in run_state.harvest.get_summary().values():
        layer = summary.layer
        if layer not in canon:
            canon[layer] = run_state.topology.target_to_canon(layer)
        out[f"{canon[layer]}:{summary.component_idx}"] = summary.firing_density
    _DENSITY_CACHE.clear()
    _DENSITY_CACHE[key] = out
    logger.info(
        f"[aspd] firing densities: {len(out)} components over {len(canon)} modules "
        f"in {perf_counter() - t0:.1f}s"
    )
    return out


def output_reachable_influence(
    edges: "list[tuple[str, str, float]]",
    seeds: "dict[str, float]",
) -> "dict[str, float]":
    """Path-weighted influence of each node on the output nodes. Torch-free, exact, one pass."""
    from collections import defaultdict

    incoming: dict[str, list[tuple[str, float]]] = defaultdict(list)
    adj_out: dict[str, list[str]] = defaultdict(list)
    indeg: dict[str, int] = defaultdict(int)
    nodes: set[str] = set(seeds)

    for src, tgt, strength in edges:
        w = abs(strength)
        incoming[tgt].append((src, w))
        adj_out[src].append(tgt)
        indeg[tgt] += 1
        nodes.add(src)
        nodes.add(tgt)

    # Kahn's algorithm, sources first.
    order: list[str] = []
    queue = [n for n in nodes if indeg[n] == 0]
    while queue:
        n = queue.pop()
        order.append(n)
        for t in adj_out[n]:
            indeg[t] -= 1
            if indeg[t] == 0:
                queue.append(t)
    assert len(order) == len(nodes), (
        f"attribution edge set is not a DAG: {len(nodes) - len(order)} of {len(nodes)} nodes lie "
        "on a cycle"
    )

    influence: dict[str, float] = dict(seeds)
    for tgt in reversed(order):
        got = influence.get(tgt, 0.0)
        if got == 0.0:
            continue
        srcs = incoming.get(tgt)
        if not srcs:
            continue
        total = sum(w for _, w in srcs)
        if total == 0.0:
            continue
        for src, w in srcs:
            influence[src] = influence.get(src, 0.0) + got * (w / total)
    return influence


def install_output_influence_pruning(
    node_cap: int = DEFAULT_NODE_CAP, max_density: float = DEFAULT_MAX_DENSITY
) -> None:
    """Cut the displayed graph to `node_cap` components, after dropping ones that fire too often."""
    from param_decomp_lab.app.backend.routers import graphs as _graphs

    if getattr(_graphs.filter_graph_for_display, "_aspd_influence_pruned", False):
        return

    stock = _graphs.filter_graph_for_display

    def _cached_rank(key, compute):
        for cached_key, ranking in reversed(_RANKINGS):
            if cached_key[0] is key[0] and cached_key[1:] == key[1:]:
                return ranking, True
        ranking = compute()
        _RANKINGS.append((key, ranking))
        del _RANKINGS[:-_RANKING_MEMO_SIZE]
        return ranking, False

    def _filter_graph_for_display(
        raw_edges,
        node_ci_vals,
        node_subcomp_acts,
        ci_masked_out_logits,
        target_out_logits,
        tok_display,
        num_tokens,
        ci_threshold,
        normalize,
        raw_edges_abs=None,
        edge_limit=_graphs.GLOBAL_EDGE_LIMIT,
    ):
        from dataclasses import replace

        from param_decomp.log import logger

        def _stock(ci_vals):
            return stock(
                raw_edges=raw_edges,
                node_ci_vals=ci_vals,
                node_subcomp_acts=node_subcomp_acts,
                ci_masked_out_logits=ci_masked_out_logits,
                target_out_logits=target_out_logits,
                tok_display=tok_display,
                num_tokens=num_tokens,
                ci_threshold=ci_threshold,
                normalize=normalize,
                raw_edges_abs=raw_edges_abs,
                edge_limit=edge_limit,
            )

        req_density = VIEW_MAX_DENSITY.get()
        req_cap = VIEW_NODE_CAP.get()
        density_limit = max_density if req_density is None else req_density
        cap = node_cap if req_cap is None else req_cap

        alive = {k: v for k, v in node_ci_vals.items() if v > ci_threshold}
        if len(alive) <= cap and density_limit >= 1.0:
            return _stock(node_ci_vals)

        def _rank():
            out_probs = _graphs._build_out_probs(
                ci_masked_out_logits, target_out_logits, tok_display
            )
            seeds = {
                f"output:{key.split(':')[0]}:{key.split(':')[1]}": p.prob
                for key, p in out_probs.items()
            }
            node_keys = set(alive) | set(seeds) | {f"embed:{s}:0" for s in range(num_tokens)}
            edges = [
                (str(e.source), str(e.target), e.strength)
                for e in raw_edges
                if str(e.source) in node_keys and str(e.target) in node_keys
            ]
            influence = output_reachable_influence(edges, seeds)

            density: dict[str, float] = {}
            eligible = list(alive)
            if density_limit < 1.0:
                densities = current_firing_densities()
                for k in alive:
                    layer, _, c_idx = k.split(":")
                    density[k] = densities.get(f"{layer}:{c_idx}", 0.0)
                eligible = [k for k in alive if density[k] <= density_limit]

            ordered = sorted(eligible, key=lambda k: -influence.get(k, 0.0))
            kept = ordered[:cap]

            total_mass = sum(influence.get(k, 0.0) for k in alive)
            kept_mass = sum(influence.get(k, 0.0) for k in kept)
            reached = sum(1 for k in alive if influence.get(k, 0.0) > 0.0)
            dropped = len(alive) - len(eligible)
            logger.info(
                f"[aspd] influence prune: {len(alive)} alive"
                + (f" -> {len(eligible)} under density {density_limit}" if dropped else "")
                + f" -> {len(kept)} drawn ({reached} had a path to an output), keeping "
                f"{100 * kept_mass / total_mass if total_mass else 0:.1f}% of output influence"
            )
            return {
                "alive": list(alive),
                "influence": influence,
                "rank": {k: i + 1 for i, k in enumerate(ordered)},
                "density": density,
                "kept": kept,
                "n_alive": len(alive),
                "n_eligible": len(eligible),
                "n_drawn": len(kept),
                "max_density": density_limit,
                "node_cap": cap,
                "kept_influence_frac": kept_mass / total_mass if total_mass else 0.0,
            }

        ranking, was_cached = _cached_rank(
            (raw_edges, ci_threshold, density_limit, cap), _rank
        )
        if was_cached:
            logger.info(
                f"[aspd] influence prune: {ranking['n_drawn']} nodes drawn (cached ranking)"
            )

        fg = _stock({k: alive[k] for k in ranking["kept"]})
        return replace(fg, l0_total=len(alive))

    _filter_graph_for_display._aspd_influence_pruned = True
    _graphs.filter_graph_for_display = _filter_graph_for_display


def install_node_ranking_api() -> None:
    """Serve each node's output-influence rank at `GET /api/aspd/node_ranking/{prompt_id}`."""
    from param_decomp_lab.app.backend.server import app

    if getattr(app, "_aspd_node_ranking_api", False):
        return

    from typing import Annotated

    from fastapi import Query

    from param_decomp_lab.app.backend.dependencies import DepStateManager

    @app.get("/api/aspd/node_ranking/{prompt_id}")
    def node_ranking(  # pyright: ignore[reportUnusedFunction]
        prompt_id: int,
        ci_threshold: Annotated[float, Query(ge=0)],
        manager: DepStateManager,
    ) -> dict:
        out: dict[str, dict] = {}
        for graph in manager.db.get_graphs(prompt_id):
            ranking = lookup_ranking(graph.edges, ci_threshold)
            if ranking is None:
                out[str(graph.id)] = {"ranked": False}
                continue
            rank, influence, density = (
                ranking["rank"],
                ranking["influence"],
                ranking["density"],
            )
            out[str(graph.id)] = {
                "ranked": True,
                "nAlive": ranking["n_alive"],
                "nEligible": ranking["n_eligible"],
                "nDrawn": ranking["n_drawn"],
                "maxDensity": ranking["max_density"],
                "keptInfluenceFrac": ranking["kept_influence_frac"],
                "nodes": {
                    k: [rank.get(k), influence.get(k, 0.0), density.get(k)]
                    for k in ranking["alive"]
                },
            }
        return out

    app._aspd_node_ranking_api = True


def install_sparse_node_ci_vals() -> None:
    """Drop the zero CI values before a graph is written to SQLite."""
    from param_decomp_lab.app.backend.database import PromptAttrDB

    if getattr(PromptAttrDB.save_graph, "_aspd_sparse_ci", False):
        return

    stock_save_graph = PromptAttrDB.save_graph

    def _save_graph(self, prompt_id, graph):
        sparse = {k: v for k, v in graph.node_ci_vals.items() if v != 0.0}
        if len(sparse) < len(graph.node_ci_vals):
            from param_decomp.log import logger

            logger.info(
                f"[aspd] node_ci_vals: {len(graph.node_ci_vals)} -> {len(sparse)} entries "
                "for save (zeros are dropped at display anyway)"
            )
        return stock_save_graph(self, prompt_id, graph.model_copy(update={"node_ci_vals": sparse}))

    _rebind_save_graph(_save_graph, stock_save_graph, "_aspd_sparse_ci")


DEFAULT_HOVER_CACHE_ENTRIES = 4096
_HOVER_CACHE_MAX_BODY = 2 * 1024 * 1024


def install_component_data_cache(maxsize: int = DEFAULT_HOVER_CACHE_ENTRIES) -> None:
    """Serve repeated per-component tooltip GETs from memory."""
    from collections import OrderedDict

    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    from param_decomp_lab.app.backend.server import app

    if getattr(app, "_aspd_hover_cached", False):
        return
    assert app.middleware_stack is None, (
        "the middleware stack is already built; install this before the server starts"
    )

    cache: "OrderedDict[tuple, tuple[bytes, dict[str, str]]]" = OrderedDict()
    stats = {"hit": 0, "miss": 0}

    def _cacheable(path: str) -> bool:
        if path.startswith("/api/correlations/interpretations"):
            return False
        return path.startswith("/api/activation_contexts/") or path.startswith("/api/correlations/")

    async def _component_data_cache(request, call_next):
        from starlette.responses import Response

        if request.method != "GET" or not _cacheable(request.url.path):
            return await call_next(request)

        from param_decomp_lab.app.backend.state import StateManager

        key = (id(StateManager.get().run_state), request.url.path, request.url.query)

        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            stats["hit"] += 1
            body, headers = hit
            return Response(content=body, status_code=200, headers=dict(headers))

        response = await call_next(request)
        if response.status_code != 200:
            return response

        body = b"".join([chunk async for chunk in response.body_iterator])
        headers = {
            k: v for k, v in response.headers.items() if k.lower() != "content-length"
        }
        if len(body) <= _HOVER_CACHE_MAX_BODY:
            cache[key] = (body, headers)
            cache.move_to_end(key)
            while len(cache) > maxsize:
                cache.popitem(last=False)
            stats["miss"] += 1
            total = stats["hit"] + stats["miss"]
            if total % 500 == 0:
                from param_decomp.log import logger

                logger.info(
                    f"[aspd] tooltip cache: {stats['hit']}/{total} hits, "
                    f"{len(cache)} entries"
                )
        return Response(content=body, status_code=200, headers=headers)

    app.user_middleware.append(
        Middleware(BaseHTTPMiddleware, dispatch=_component_data_cache)
    )
    app._aspd_hover_cached = True


DEFAULT_GRAPH_CACHE_ENTRIES = 4


def install_stored_graph_cache(maxsize: int = DEFAULT_GRAPH_CACHE_ENTRIES) -> None:
    """Keep parsed graphs in memory so re-reading one is not re-parsing it."""
    from param_decomp_lab.app.backend.database import PromptAttrDB

    if getattr(PromptAttrDB.get_graphs, "_aspd_graph_cached", False):
        return

    from collections import OrderedDict

    cache: "OrderedDict[tuple, list]" = OrderedDict()

    stock_get_graphs = PromptAttrDB.get_graphs

    def _get_graphs(self, prompt_id):
        key = (id(self), prompt_id)
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        graphs = stock_get_graphs(self, prompt_id)
        cache[key] = graphs
        cache.move_to_end(key)
        while len(cache) > maxsize:
            cache.popitem(last=False)
        return graphs

    _get_graphs._aspd_graph_cached = True
    _get_graphs._aspd_cache = cache
    PromptAttrDB.get_graphs = _get_graphs

    stock_save_graph = PromptAttrDB.save_graph

    def _save_graph(self, prompt_id, graph):
        cache.pop((id(self), prompt_id), None)
        return stock_save_graph(self, prompt_id, graph)

    _rebind_save_graph(_save_graph, stock_save_graph, "_aspd_graph_cached")
