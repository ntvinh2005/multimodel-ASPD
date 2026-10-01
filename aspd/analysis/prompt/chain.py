"""Multi-hop attribution: what feeds the component that feeds the target."""

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from aspd.analysis.prompt.scores import component_target, node_scores
from aspd.analysis.prompt.trace import PromptTrace


@dataclass(frozen=True)
class ChainEdge:
    """`source` feeds `target`. Two numbers, and using the wrong one is the easy mistake."""

    source: str  # "<module>:<pos>:<idx>", or "error:<parent>"
    target: str | None
    score: float
    effect: float
    depth: int
    kind: str = "component"

    def as_row(self) -> dict[str, object]:
        return {"source": self.source, "target": self.target, "score": self.score,
                "effect": self.effect, "depth": self.depth, "kind": self.kind}


@dataclass
class _Node:
    key: str
    module: str
    pos: int
    idx: int
    depth: int
    effect: float  # this node's own effect on the ROOT target
    act: float


def node_key(module: str, pos: int, idx: int) -> str:
    return f"{module}:{pos}:{idx}"


def attribution_tree(
    trace: PromptTrace,
    target: Callable[[Tensor], Tensor] | Tensor,
    *,
    depth: int = 2,
    width: int = 5,
    include_error: bool = True,
) -> tuple[list[ChainEdge], dict[str, float]]:
    """`(edges, stats)`. Expand the `width` strongest sources of the target, then of each of those."""
    assert depth >= 1 and width >= 1
    edges: list[ChainEdge] = []
    expanded: set[str] = set()
    frontier: list[_Node | None] = [None]  # None is the root scalar
    n_backwards = 0

    for d in range(depth):
        nxt: list[_Node] = []
        for node in frontier:
            key = node.key if node is not None else None
            if key is not None and key in expanded:
                continue
            if key is not None:
                expanded.add(key)
            scalar = target if node is None else component_target(
                trace, node.module, node.pos, node.idx
            )
            scores = node_scores(trace, scalar)
            n_backwards += 1
            if node is None:
                scale = 1.0  # the root: a child's effect IS its score
            elif abs(node.act) > 1e-9:
                scale = node.effect / node.act
            else:
                scale = 0.0
            for row in scores.top(width):
                module, pos, idx = str(row["module"]), int(row["pos"]), int(row["idx"])
                score = float(row["score"])
                child = _Node(node_key(module, pos, idx), module, pos, idx, d + 1,
                              effect=scale * score, act=float(trace.acts[module][pos, idx]))
                edges.append(ChainEdge(child.key, key, score, child.effect, d + 1))
                nxt.append(child)
            if include_error:
                err = sum(float(s.abs().sum()) for s in scores.errors.values())
                if err:
                    edges.append(ChainEdge(f"error:{key or 'target'}", key, err, scale * err,
                                           d + 1, "error"))
        frontier = nxt

    return edges, {"n_backwards": float(n_backwards), "n_edges": float(len(edges)),
                   "n_nodes": float(len(expanded))}


def roles_reached(trace: PromptTrace, edges: list[ChainEdge]) -> dict[str, int]:
    """How many edges landed on each module role -- the check that chaining bought what it should."""
    out: dict[str, int] = {}
    for e in edges:
        if e.kind != "component":
            continue
        module = e.source.rsplit(":", 2)[0]
        role = trace.spaces[module].role
        out[role] = out.get(role, 0) + 1
    return out
