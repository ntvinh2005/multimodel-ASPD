"""The circuit for one prompt: which components and error terms move the target scalar."""

from collections.abc import Callable

import torch
from torch import Tensor

from aspd.analysis.circuits.edges import contract_error, grad_times_act
from aspd.analysis.circuits.nodes import Edge, Node
from aspd.analysis.circuits.replacement import run_replacement
from aspd.analysis.circuits.targets import top_k_vs_rest


def prompt_circuit(
    model,
    tokens: Tensor,
    *,
    sampling: str,
    target_fn: Callable[[Tensor], Tensor] | None = None,
    topology=None,
    edge_threshold: float = 0.0,
    error_nodes: bool = True,
    freeze_layernorm: bool = False,
) -> tuple[list[Edge], dict[str, float]]:
    """`(edges, stats)`. `target_fn` defaults to `mean(top-10 logits) - mean(the rest)`."""
    cache = run_replacement(
        model,
        tokens,
        sampling=sampling,
        topology=topology,
        error_nodes=error_nodes,
        freeze_layernorm=freeze_layernorm,
    )
    assert cache.logits is not None
    target = (target_fn or (lambda z: top_k_vs_rest(z, k=10)))(cache.logits)
    sign = float(torch.sign(target).item())

    leaves: dict[str, Tensor] = {f"c::{k}": v for k, v in cache.component_acts.items()}
    leaves.update({f"e::{k}": v for k, v in cache.errors.items()})
    if cache.embed is not None:
        leaves["m::embed"] = cache.embed

    weighted = grad_times_act(target, leaves)

    logit_node = Node(kind="logit", layer="target")
    edges: list[Edge] = []
    n_fired = 0
    for key, w in weighted.items():
        kind, _, layer = key.partition("::")
        if kind == "c":
            fired = cache.ci[layer][0] > 0
            n_fired += int(fired.sum().item())
            idx = fired.nonzero(as_tuple=False)
            for p, c in idx.tolist():
                val = float(w[0, p, c].item())
                if abs(val) < edge_threshold:
                    continue
                edges.append(Edge(
                    source=Node("component", layer, seq_pos=p, component_idx=c),
                    target=logit_node, weight=val, weight_abs=val * sign,
                ))
        else:
            per_pos = contract_error(w)[0]
            node_layer = "embed" if kind == "m" else layer
            for p, val_t in enumerate(per_pos.tolist()):
                if abs(val_t) < edge_threshold:
                    continue
                edges.append(Edge(
                    source=Node("embed" if kind == "m" else "error", node_layer, seq_pos=p),
                    target=logit_node, weight=val_t, weight_abs=val_t * sign,
                ))

    edges.sort(key=lambda e: -abs(e.weight))
    total = sum(e.weight for e in edges)
    err = sum(e.weight for e in edges if e.source.kind == "error")
    stats = {
        "target": float(target.item()),
        "n_edges": float(len(edges)),
        "n_fired_components": float(n_fired),
        "sum_edges": total,
        "error_share": (err / total) if total != 0 else float("nan"),
    }
    return edges, stats
