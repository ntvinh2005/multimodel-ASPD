"""Parent sets (formulation, "Parent set definition" and "Parent Set Comparison").

    Pa_{K_par,tau}^{(n)}(a; alpha) = { b in TopK_{K_par}( |E_{.->a}(alpha)| ) : |E_{b->a}(alpha)| > tau }

    G_{p,i}^{(n)}(alpha) = union_{a in R_{p,i}^{(n)}} { gamma(b) : b in Pa(a; alpha) }

The candidate set ``U^{(n)}(a)`` is whatever the edge stage stored for ``a`` (every
firing upstream component; see ``edges``).  Ties in ``|E|`` are broken by
``ComponentId`` order so the result is deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from aspd.multimodel.posthoc.edges import EdgeRecord
from aspd.multimodel.posthoc.ids import ComponentId, GateId


@dataclass(frozen=True)
class ParentRule:
    k_par: int
    tau: float

    def __post_init__(self) -> None:
        if self.k_par < 1:
            raise ValueError("k_par must be positive")
        if not self.tau >= 0:
            raise ValueError("tau must be non-negative")


@dataclass(frozen=True)
class Parent:
    component: ComponentId
    score: float  # signed E_{b->a}(alpha)


def parent_set(record: EdgeRecord, rule: ParentRule) -> tuple[Parent, ...]:
    """``Pa(a; alpha)`` ordered by decreasing ``|E|``."""

    ranked = sorted(
        zip(record.upstream, record.scores.tolist(), strict=True),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    return tuple(
        Parent(component, score)
        for component, score in ranked[: rule.k_par]
        if abs(score) > rule.tau
    )


def parent_gate_set(parents_by_representative: Iterable[Iterable[Parent]]) -> frozenset[GateId]:
    """``G_{p,i}^{(n)}(alpha)``: union of ``gamma(b)`` over all representatives' parents."""

    return frozenset(
        parent.component.gate for parents in parents_by_representative for parent in parents
    )
