"""Stores several attribution methods per prompt and lets the app choose one."""

import contextvars
import sqlite3
from contextlib import contextmanager

from param_decomp.log import logger

from aspd.analysis.circuits.lab_api import (
    ERROR_LAYER_SUFFIX,
    install_error_nodes_are_not_interventable,
)

METHODS = ("lab", "err")
DEFAULT_METHOD = "lab"

CURRENT_METHOD: contextvars.ContextVar[str] = contextvars.ContextVar(
    "lm_interp_graph_method", default=DEFAULT_METHOD
)

_INDEXES = {
    "idx_graphs_standard": (
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_graphs_standard ON graphs(prompt_id, method) "
        "WHERE graph_type = 'standard'"
    ),
    "idx_graphs_optimized": (
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_graphs_optimized ON graphs(prompt_id, method, "
        "imp_min_coeff, steps, pnorm, beta, mask_type, loss_config_hash, adv_pgd_n_steps, "
        "adv_pgd_step_size) WHERE graph_type = 'optimized'"
    ),
    "idx_graphs_manual": (
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_graphs_manual ON graphs(prompt_id, method, "
        "included_nodes_hash) WHERE graph_type = 'manual'"
    ),
}


@contextmanager
def method(name: str):
    """Run a block with `name` as the active attribution method."""
    assert name in METHODS, f"unknown method {name!r}, expected one of {METHODS}"
    token = CURRENT_METHOD.set(name)
    try:
        yield
    finally:
        CURRENT_METHOD.reset(token)


def migrate(conn: sqlite3.Connection) -> None:
    """Add `method` and rebuild the three unique indexes to include it. Idempotent."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(graphs)")}
    if "method" not in cols:
        conn.execute("ALTER TABLE graphs ADD COLUMN method TEXT")
        n = conn.execute("UPDATE graphs SET method = ? WHERE method IS NULL", (DEFAULT_METHOD,))
        logger.info(f"[aspd] graphs.method added; {n.rowcount} existing rows -> 'lab'")
    for name, create in _INDEXES.items():
        existing = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (name,)
        ).fetchone()
        if existing is not None and "method" in (existing["sql"] or ""):
            continue
        conn.execute(f"DROP INDEX IF EXISTS {name}")
        conn.execute(create)
        logger.info(f"[aspd] rebuilt {name} to include method")
    conn.commit()


def install_graph_method_column() -> None:
    """Migrate the schema on startup and stamp every saved graph with the active method."""
    from param_decomp_lab.app.backend import database as db_mod

    if getattr(db_mod.PromptAttrDB.save_graph, "_aspd_method", False):
        return

    stock_init = db_mod.PromptAttrDB.init_schema
    stock_save = db_mod.PromptAttrDB.save_graph

    def init_schema(self) -> None:
        stock_init(self)
        migrate(self._get_conn())

    def save_graph(self, prompt_id, graph):
        name = CURRENT_METHOD.get()
        graph_id = stock_save(self, prompt_id, graph)
        conn = self._get_conn()
        with self._write_lock():
            try:
                conn.execute("UPDATE graphs SET method = ? WHERE id = ?", (name, graph_id))
                conn.commit()
            except sqlite3.IntegrityError as e:
                conn.rollback()
                if graph.graph_type == "manual":
                    row = conn.execute(
                        "SELECT id FROM graphs WHERE prompt_id = ? AND graph_type = 'manual' "
                        "AND method = ? AND included_nodes_hash IS NOT NULL "
                        "AND id != ? ORDER BY id LIMIT 1",
                        (prompt_id, name, graph_id),
                    ).fetchone()
                    conn.execute("DELETE FROM graphs WHERE id = ?", (graph_id,))
                    conn.commit()
                    if row:
                        return row["id"]
                conn.execute("DELETE FROM graphs WHERE id = ?", (graph_id,))
                conn.commit()
                raise ValueError(
                    f"A {graph.graph_type} graph with these parameters already exists for "
                    f"prompt_id={prompt_id} under method={name!r}."
                ) from e
        return graph_id

    from aspd.analysis.app.patch import _rebind_save_graph

    init_schema._aspd_method = True  # pyright: ignore[reportFunctionMemberAccess]
    db_mod.PromptAttrDB.init_schema = init_schema
    _rebind_save_graph(save_graph, stock_save, "_aspd_method")


def install_view_query_params() -> None:
    """Read `?method=`, `?max_density=` and `?node_cap=` off any route into ContextVars."""
    from param_decomp_lab.app.backend.server import app
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    from aspd.analysis.app.patch import VIEW_MAX_DENSITY, VIEW_NODE_CAP

    if getattr(app, "_aspd_view_params", False):
        return

    async def dispatch(request, call_next):
        raw = request.query_params.get("method")
        if raw is not None:
            assert raw in METHODS, f"unknown method {raw!r}, expected one of {METHODS}"
        CURRENT_METHOD.set(DEFAULT_METHOD if raw is None else raw)

        density = request.query_params.get("max_density")
        VIEW_MAX_DENSITY.set(None if density is None else float(density))
        cap = request.query_params.get("node_cap")
        VIEW_NODE_CAP.set(None if cap is None else int(cap))
        return await call_next(request)

    app.user_middleware.append(Middleware(BaseHTTPMiddleware, dispatch=dispatch))
    app.middleware_stack = None
    app._aspd_view_params = True


def install_method_context_propagation() -> None:
    """Carry the active method into the compute worker thread."""
    from param_decomp_lab.app.backend.routers import graphs as graphs_mod

    if getattr(graphs_mod.stream_computation, "_aspd_method_ctx", False):
        return

    stock = graphs_mod.stream_computation

    def stream_computation(work, *args, **kwargs):
        ctx = contextvars.copy_context()

        def in_context(*a, **k):
            return ctx.run(work, *a, **k)

        return stock(in_context, *args, **kwargs)

    stream_computation._aspd_method_ctx = True  # pyright: ignore[reportFunctionMemberAccess]
    graphs_mod.stream_computation = stream_computation


def install_method_dispatch() -> None:
    """Route `compute_prompt_attributions` to the lab's or ours, per request."""
    from param_decomp_lab.app.backend import compute as compute_mod
    from param_decomp_lab.app.backend.routers import graphs as graphs_mod

    from aspd.analysis.circuits.lab_api import compute_prompt_attributions_with_errors

    if getattr(compute_mod.compute_prompt_attributions, "_aspd_dispatch", False):
        return

    stock = compute_mod.compute_prompt_attributions

    def dispatch(*args, **kwargs):
        if CURRENT_METHOD.get() == "err":
            return compute_prompt_attributions_with_errors(*args, **kwargs)
        return stock(*args, **kwargs)

    def reject_optimized(*args, **kwargs):
        assert CURRENT_METHOD.get() != "err", (
            "method='err' has no optimizer: the error-node path computes a circuit at m = g and "
            "nothing else. Use method=lab for optimized graphs, or drop ?method=err."
        )
        return stock_optimized(*args, **kwargs)

    stock_optimized = compute_mod.compute_prompt_attributions_optimized

    dispatch._aspd_dispatch = True  # pyright: ignore[reportFunctionMemberAccess]
    compute_mod.compute_prompt_attributions = dispatch
    compute_mod.compute_prompt_attributions_optimized = reject_optimized
    # `routers/graphs.py` binds both names at import time, so patch that namespace too.
    graphs_mod.compute_prompt_attributions = dispatch
    graphs_mod.compute_prompt_attributions_optimized = reject_optimized


def install_graph_method_api() -> None:
    """`GET /api/aspd/graph_methods/{prompt_id}` -> `{graph_id: method}`."""
    from param_decomp_lab.app.backend.server import app
    from param_decomp_lab.app.backend.dependencies import DepStateManager

    if getattr(app, "_aspd_graph_method_api", False):
        return

    @app.get("/api/aspd/graph_methods/{prompt_id}")
    def graph_methods(prompt_id: int, manager: DepStateManager) -> dict[str, str]:
        rows = manager.db._get_conn().execute(
            "SELECT id, method FROM graphs WHERE prompt_id = ?", (prompt_id,)
        ).fetchall()
        return {str(r["id"]): (r["method"] or DEFAULT_METHOD) for r in rows}

    app._aspd_graph_method_api = True


def install_thread_local_connections() -> None:
    """Give every thread its own SQLite connection."""
    import sqlite3
    import threading

    from param_decomp_lab.app.backend import database as db_mod

    if getattr(db_mod.PromptAttrDB._get_conn, "_aspd_thread_local", False):
        return

    def _get_conn(self) -> sqlite3.Connection:
        local = getattr(self, "_aspd_local", None)
        if local is None:
            local = threading.local()
            self._aspd_local = local
        conn = getattr(local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread)
            conn.row_factory = sqlite3.Row
            local.conn = conn
            self._conn = conn
        return conn

    _get_conn._aspd_thread_local = True  # pyright: ignore[reportFunctionMemberAccess]
    db_mod.PromptAttrDB._get_conn = _get_conn


def install_no_orphan_graphs() -> None:
    """Delete a graph whose base intervention run failed, instead of storing it half-written."""
    from param_decomp_lab.app.backend.database import PromptAttrDB
    from param_decomp_lab.app.backend.routers import graphs as graphs_mod

    if getattr(graphs_mod._save_base_intervention_run, "_aspd_no_orphans", False):
        return

    stock = graphs_mod._save_base_intervention_run

    def _save_base_intervention_run(*args, **kwargs):
        try:
            return stock(*args, **kwargs)
        except Exception as e:
            graph_id, db = kwargs.get("graph_id"), kwargs.get("db")
            if graph_id is None or db is None:
                raise
            conn = db._get_conn()
            with db._write_lock():
                conn.execute("DELETE FROM intervention_runs WHERE graph_id = ?", (graph_id,))
                conn.execute("DELETE FROM graphs WHERE id = ?", (graph_id,))
                conn.commit()
            cache = getattr(PromptAttrDB.get_graphs, "_aspd_cache", None)
            if cache is not None:
                cache.clear()
            logger.warning(
                f"[aspd] graph {graph_id} deleted: its base intervention run failed "
                f"({type(e).__name__}: {e}). A stored graph with no base run cannot be opened."
            )
            raise

    _save_base_intervention_run._aspd_no_orphans = True  # pyright: ignore[reportAny]
    for carried in ("_aspd_skips_errors",):
        if getattr(stock, carried, False):
            setattr(_save_base_intervention_run, carried, True)
    graphs_mod._save_base_intervention_run = _save_base_intervention_run


def install_density_graph_api() -> None:
    """`POST /api/aspd/graphs/density_filtered/{prompt_id}` -- the filter as an ABLATION."""
    from typing import Annotated

    from fastapi import Query
    from param_decomp_lab.app.backend.server import app
    from param_decomp_lab.app.backend.dependencies import DepLoadedRun, DepStateManager

    if getattr(app, "_aspd_density_graph_api", False):
        return

    @app.post("/api/aspd/graphs/density_filtered/{prompt_id}")
    def density_filtered_graph(
        prompt_id: int,
        normalize: Annotated[str, Query()],
        ci_threshold: Annotated[float, Query()],
        loaded: DepLoadedRun,
        manager: DepStateManager,
        max_density: Annotated[float, Query(gt=0, le=1.0)] = 0.05,
        node_cap: Annotated[int, Query(ge=1, le=10000)] = 5000,
        source_graph_id: Annotated[int | None, Query()] = None,
    ):
        from param_decomp_lab.app.backend.routers.graphs import (
            ComputeGraphRequest,
            compute_graph_stream,
        )

        from aspd.analysis.app.patch import current_firing_densities

        graphs = manager.db.get_graphs(prompt_id)
        assert graphs, (
            f"prompt {prompt_id} has no stored graph; compute one first -- the node set and its "
            "ranking both come from an existing graph's edges"
        )
        if source_graph_id is None:
            source = graphs[0]
        else:
            matches = [g for g in graphs if g.id == source_graph_id]
            assert matches, f"graph {source_graph_id} is not a graph of prompt {prompt_id}"
            source = matches[0]

        densities = current_firing_densities()
        score: dict[str, float] = {}
        for edge in source.edges:
            if edge.target.layer != "output":
                continue
            key = str(edge.source)
            score[key] = score.get(key, 0.0) + abs(edge.strength)

        eligible: list[str] = []
        n_dense = 0
        for key, ci_val in source.node_ci_vals.items():
            layer, _, c_idx = key.split(":")
            if ci_val <= 0.0 or layer.endswith(ERROR_LAYER_SUFFIX) or layer in ("embed", "output"):
                continue
            if densities.get(f"{layer}:{c_idx}", 0.0) > max_density:
                n_dense += 1
                continue
            eligible.append(key)

        eligible.sort(key=lambda k: -score.get(k, 0.0))
        included = eligible[:node_cap]
        logger.info(
            f"[aspd] density ablation: graph {source.id} -> {len(eligible)} under density "
            f"{max_density} ({n_dense} dropped as too dense) -> {len(included)} included; the "
            "excluded components are switched OFF in the forwards, not merely hidden"
        )
        return compute_graph_stream(
            prompt_id=prompt_id,
            normalize=normalize,  # pyright: ignore[reportArgumentType]
            loaded=loaded,
            manager=manager,
            ci_threshold=ci_threshold,
            body=ComputeGraphRequest(included_nodes=included),
        )

    app._aspd_density_graph_api = True


def install_all_methods() -> None:
    """Every patch this module owns, in dependency order."""
    install_thread_local_connections()
    install_graph_method_column()
    install_method_dispatch()
    install_error_nodes_are_not_interventable()
    install_no_orphan_graphs()
    install_method_context_propagation()
    install_view_query_params()
    install_graph_method_api()
    install_density_graph_api()
