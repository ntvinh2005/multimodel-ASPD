"""Node types of a circuit graph."""

from dataclasses import dataclass
from typing import Literal

NodeKind = Literal["component", "error", "embed", "logit"]


@dataclass(frozen=True, order=True)
class Node:
    """`seq_pos` is None for a node with no position (a logit target aggregated over the vocab)."""

    kind: NodeKind
    layer: str
    seq_pos: int | None = None
    component_idx: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "component":
            assert self.component_idx is not None, "a component node needs an index"
        else:
            assert self.component_idx is None, f"{self.kind} nodes have no component index"

    @property
    def key(self) -> str:
        parts = [self.kind, self.layer]
        if self.seq_pos is not None:
            parts.append(f"p{self.seq_pos}")
        if self.component_idx is not None:
            parts.append(f"c{self.component_idx}")
        return ":".join(parts)


@dataclass(frozen=True)
class Edge:
    """`weight` is the signed grad x act; `weight_abs` is the same against `|target|`."""

    source: Node
    target: Node
    weight: float
    weight_abs: float
