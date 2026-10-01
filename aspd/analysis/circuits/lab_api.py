"""Convert circuits to the lab app's prompt-attribution format."""

import os

import torch
from torch import Tensor

from param_decomp.log import logger

from aspd.analysis.circuits.edges import contract_error, grad_times_act
from aspd.analysis.circuits.replacement import run_replacement

ERROR_LAYER_SUFFIX = ".err"

DEFAULT_COMPONENT_TARGET_CAP = int(os.environ.get("LM_INTERP_COMPONENT_TARGETS", "0"))

DEFAULT_EDGES_PER_TARGET = int(os.environ.get("LM_INTERP_EDGES_PER_TARGET", "64"))


def compute_prompt_attributions_with_errors(
    model,
    topology,
    tokens: Tensor,
    sources_by_target: dict[str, list[str]],
    output_prob_threshold: float,
    sampling,
    device: str,
    on_progress=None,
    included_nodes: set[str] | None = None,
    loss_seq_pos: int | None = None,
    component_target_cap: int | None = None,
    edges_per_target: int | None = None,
):
    """The lab's `compute_prompt_attributions` signature, computed our way."""
    from param_decomp_lab.app.backend.compute import Edge, Node, PromptAttributionResult

    cache = run_replacement(
        model, tokens, sampling=sampling, topology=topology, error_nodes=True
    )
    assert cache.logits is not None
    seq = tokens.shape[1]
    tgt_pos = seq - 1 if loss_seq_pos is None else loss_seq_pos

    with torch.no_grad():
        target_logits = model(tokens)
        target_probs = torch.softmax(target_logits, dim=-1)
        ci_masked_logits, ci_masked_probs = target_logits, target_probs

    # One target per output token above the threshold, matching the stock graph's shape.
    probs_at = target_probs[0, tgt_pos]
    out_tokens = (probs_at > output_prob_threshold).nonzero(as_tuple=False).flatten().tolist()
    if not out_tokens:
        out_tokens = [int(probs_at.argmax())]

    leaves: dict[str, Tensor] = {f"c::{k}": v for k, v in cache.component_acts.items()}
    leaves.update({f"e::{k}": v for k, v in cache.errors.items()})

    def sources_into(target_scalar: Tensor, target_node, keep_top: int | None):
        """`(edges, edges_abs)` into one target node, from one backward."""
        sign = float(torch.sign(target_scalar).item())
        weighted = grad_times_act(target_scalar, leaves)
        found: list[tuple[float, object]] = []
        for key, w in weighted.items():
            kind, _, path = key.partition("::")
            canon = topology.target_to_canon(path)
            if kind == "c":
                fired = cache.ci[path][0] > 0
                for p, c in fired.nonzero(as_tuple=False).tolist():
                    val = float(w[0, p, c].item())
                    if val == 0.0:
                        continue
                    src = Node(layer=canon, seq_pos=p, component_idx=c)
                    if included_nodes is not None and str(src) not in included_nodes:
                        continue
                    found.append((val, src))
            else:
                for p, val in enumerate(contract_error(w)[0].tolist()):
                    if val == 0.0:
                        continue
                    found.append((val, Node(canon + ERROR_LAYER_SUFFIX, p, 0)))
        if keep_top is not None and len(found) > keep_top:
            found.sort(key=lambda vs: -abs(vs[0]))
            del found[keep_top:]
        out, out_abs = [], []
        for val, src in found:
            if src == target_node:
                continue
            cross = src.seq_pos != target_node.seq_pos
            out.append(Edge(src, target_node, val, cross))
            out_abs.append(Edge(src, target_node, val * sign, cross))
        return out, out_abs

    edges: list[Edge] = []
    edges_abs: list[Edge] = []

    # Pass 1 -- output targets. Also produces the ranking pass 2 selects with, at no extra cost.
    influence: dict[str, float] = {}
    for vocab_idx in out_tokens:
        out_node = Node(layer="output", seq_pos=tgt_pos, component_idx=int(vocab_idx))
        e, e_abs = sources_into(cache.logits[0, tgt_pos, vocab_idx], out_node, None)
        edges.extend(e)
        edges_abs.extend(e_abs)
        for edge in e:
            key = str(edge.source)
            influence[key] = influence.get(key, 0.0) + abs(edge.strength)

    cap = DEFAULT_COMPONENT_TARGET_CAP if component_target_cap is None else component_target_cap
    top_k = DEFAULT_EDGES_PER_TARGET if edges_per_target is None else edges_per_target
    top_k = None if top_k <= 0 else top_k
    n_component_targets = 0
    if cap != 0:
        canon_to_path = {topology.target_to_canon(path): path for path in cache.ci}
        candidates = [
            f"{topology.target_to_canon(path)}:{p}:{c}"
            for path, ci in cache.ci.items()
            for p, c in (ci[0] > 0).nonzero(as_tuple=False).tolist()
        ]
        if included_nodes is not None:
            candidates = [k for k in candidates if k in included_nodes]
        ranked = sorted(candidates, key=lambda k: -influence.get(k, 0.0))
        if cap > 0:
            ranked = ranked[:cap]
        for i, key in enumerate(ranked):
            layer, q, d = key.split(":")
            path = canon_to_path.get(layer)
            if path is None:
                continue
            tgt_node = Node(layer=layer, seq_pos=int(q), component_idx=int(d))
            # The PRE-detach copy. Differentiating the leaf would give zero for everything.
            scalar = cache.component_acts_pre[path][0, int(q), int(d)]
            e, e_abs = sources_into(scalar, tgt_node, top_k)
            edges.extend(e)
            edges_abs.extend(e_abs)
            n_component_targets += 1
            if on_progress is not None and i % 25 == 0:
                on_progress(i, len(ranked), "edges")
    logger.info(
        f"[aspd] error-node graph: {len(out_tokens)} output targets, "
        f"{n_component_targets} component targets "
        f"(cap {'all' if cap < 0 else cap}, {'all' if top_k is None else f'top-{top_k}'} edges "
        f"each), {len(edges)} edges"
    )

    node_ci_vals: dict[str, float] = {}
    node_subcomp_acts: dict[str, float] = {}
    for path, ci in cache.ci.items():
        canon = topology.target_to_canon(path)
        acts = cache.component_acts[path]
        for p, c in (ci[0] > 0).nonzero(as_tuple=False).tolist():
            key = f"{canon}:{p}:{c}"
            node_ci_vals[key] = float(ci[0, p, c].item())
            node_subcomp_acts[key] = float(acts[0, p, c].item())
    for path, eps in cache.errors.items():
        canon = topology.target_to_canon(path) + ERROR_LAYER_SUFFIX
        for p in range(seq):
            key = f"{canon}:{p}:0"
            node_ci_vals[key] = 1.0
            node_subcomp_acts[key] = float(eps[0, p].norm().item())

    return PromptAttributionResult(
        edges=edges,
        edges_abs=edges_abs,
        ci_masked_out_probs=ci_masked_probs[0],
        ci_masked_out_logits=ci_masked_logits[0],
        target_out_probs=target_probs[0],
        target_out_logits=target_logits[0],
        node_ci_vals=node_ci_vals,
        node_subcomp_acts=node_subcomp_acts,
    )


def install_error_nodes_are_not_interventable() -> None:
    """Keep `<layer>.err` nodes out of the base intervention run the graph encoder saves."""
    from param_decomp_lab.app.backend.routers import graphs as graphs_mod

    if getattr(graphs_mod._save_base_intervention_run, "_aspd_skips_errors", False):
        return

    stock = graphs_mod._save_base_intervention_run

    def _save_base_intervention_run(*args, **kwargs):
        vals = kwargs.get("node_ci_vals")
        if vals is not None:
            kept = {
                k: v for k, v in vals.items() if not k.split(":")[0].endswith(ERROR_LAYER_SUFFIX)
            }
            if len(kept) != len(vals):
                logger.info(
                    f"[aspd] base intervention: skipping {len(vals) - len(kept)} error nodes "
                    "(nothing to ablate -- an error node is the unexplained residual)"
                )
            kwargs["node_ci_vals"] = kept
        return stock(*args, **kwargs)

    _save_base_intervention_run._aspd_skips_errors = True  # pyright: ignore[reportFunctionMemberAccess]
    graphs_mod._save_base_intervention_run = _save_base_intervention_run


def install_error_node_attributions() -> None:
    """Route the app's graph encoder through the error-node computation. Idempotent."""
    if os.environ.get("LM_INTERP_ERROR_NODES") == "0":
        return
    from param_decomp_lab.app.backend import compute as compute_mod

    if getattr(compute_mod.compute_prompt_attributions, "_aspd_errors", False):
        return
    compute_prompt_attributions_with_errors._aspd_errors = True  # pyright: ignore[reportFunctionMemberAccess]
    compute_mod.compute_prompt_attributions = compute_prompt_attributions_with_errors
    try:
        from param_decomp_lab.app.backend.routers import graphs as graphs_mod

        graphs_mod.compute_prompt_attributions = compute_prompt_attributions_with_errors
    except ImportError:
        pass
