"""Cross-model parent comparison (formulation, "Set Overlap" and "Weighted Parent Distribution").

For downstream gate ``(p,i)``, anchor ``alpha`` and models ``n, m``::

    J^{(n,m)}         = |G^n ∩ G^m| / |G^n ∪ G^m|
    w_{b->a}          = |E_{b->a}| * ||u_{j_a,i_a}||_2 / sigma_{j_a}
    S^{(n)}(xi)       = (1/|R^n|) sum_{a in R^n} sum_{b in Pa(a), gamma(b)=xi} w_{b->a}
    pi^{(n)}(xi)      = S^{(n)}(xi) / sum_xi' S^{(n)}(xi')
    D_TV^{(n,m)}      = 1/2 sum_xi | pi^{(n)}(xi) - pi^{(m)}(xi) |
    M^{(n)}           = sum_xi S^{(n)}(xi)

Undefined quantities are returned as ``None`` rather than a number: the Jaccard index
when both gate sets are empty, ``pi`` when ``M = 0``, and ``D_TV`` unless both models
have a non-empty representative set and ``M > 0`` (the formulation's validity rule).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aspd.multimodel.posthoc.ids import ComponentId, GateId
from aspd.multimodel.posthoc.parents import Parent, parent_gate_set


def jaccard(left: frozenset[GateId], right: frozenset[GateId]) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def edge_weight(score: float, write_norm: float, sigma: float) -> float:
    """``w_{b->a} = |E| ||u_a|| / sigma_{j_a}``; requires ``sigma > 0``."""

    if not sigma > 0:
        raise ValueError("edge weight needs a positive downstream output scale")
    return abs(score) * write_norm / sigma


def aggregated_parent_scores(
    parents: Mapping[ComponentId, Sequence[Parent]],
    representatives: Sequence[ComponentId],
    write_norms: Mapping[ComponentId, float],
    sigmas: Mapping[ComponentId, float],
) -> dict[GateId, float]:
    """``S^{(n)}_{p,i}(xi; alpha)`` for every upstream gate identity ``xi``.

    ``parents[a]`` is ``Pa(a; alpha)``; a representative without a stored edge record
    contributes nothing but still counts in ``|R|`` (the formulation's normalization).
    """

    if not representatives:
        return {}
    totals: dict[GateId, float] = {}
    for a in representatives:
        for parent in parents.get(a, ()):
            weight = edge_weight(parent.score, write_norms[a], sigmas[a])
            gate = parent.component.gate
            totals[gate] = totals.get(gate, 0.0) + weight
    scale = 1.0 / len(representatives)
    return {gate: value * scale for gate, value in sorted(totals.items())}


def parent_distribution(scores: Mapping[GateId, float]) -> dict[GateId, float] | None:
    """``pi`` over gate identities, or ``None`` when ``M = 0``."""

    total = sum(scores.values())
    if not total > 0:
        return None
    return {gate: value / total for gate, value in scores.items()}


def total_variation(left: Mapping[GateId, float], right: Mapping[GateId, float]) -> float:
    support = set(left) | set(right)
    return 0.5 * sum(abs(left.get(xi, 0.0) - right.get(xi, 0.0)) for xi in support)


@dataclass(frozen=True)
class ModelParentView:
    """Everything one model contributes for one downstream gate at one anchor."""

    model: str
    representatives: tuple[ComponentId, ...]
    parents: Mapping[ComponentId, tuple[Parent, ...]]
    write_norms: Mapping[ComponentId, float]
    sigmas: Mapping[ComponentId, float]

    @property
    def gate_set(self) -> frozenset[GateId]:
        return parent_gate_set(self.parents.get(a, ()) for a in self.representatives)

    @property
    def scores(self) -> dict[GateId, float]:
        return aggregated_parent_scores(
            self.parents, self.representatives, self.write_norms, self.sigmas
        )


@dataclass(frozen=True)
class PairComparison:
    gate: GateId
    anchor_id: str
    left: str
    right: str
    jaccard: float | None
    total_variation: float | None
    strength_left: float  # M^{(n)}
    strength_right: float  # M^{(m)}
    gate_set_left: tuple[str, ...]
    gate_set_right: tuple[str, ...]
    distribution_valid: bool

    def to_json(self) -> dict[str, object]:
        return {
            "gate": self.gate.key,
            "anchor_id": self.anchor_id,
            "models": [self.left, self.right],
            "jaccard": self.jaccard,
            "total_variation": self.total_variation,
            "strength": [self.strength_left, self.strength_right],
            "parent_gate_sets": [list(self.gate_set_left), list(self.gate_set_right)],
            "distribution_valid": self.distribution_valid,
        }


def compare_pair(
    gate: GateId, anchor_id: str, left: ModelParentView, right: ModelParentView
) -> PairComparison:
    scores_left, scores_right = left.scores, right.scores
    m_left, m_right = sum(scores_left.values()), sum(scores_right.values())
    pi_left, pi_right = parent_distribution(scores_left), parent_distribution(scores_right)
    valid = (
        bool(left.representatives)
        and bool(right.representatives)
        and pi_left is not None
        and pi_right is not None
    )
    tv = total_variation(pi_left, pi_right) if valid else None  # type: ignore[arg-type]
    return PairComparison(
        gate=gate,
        anchor_id=anchor_id,
        left=left.model,
        right=right.model,
        jaccard=jaccard(left.gate_set, right.gate_set),
        total_variation=tv,
        strength_left=m_left,
        strength_right=m_right,
        gate_set_left=tuple(sorted(g.key for g in left.gate_set)),
        gate_set_right=tuple(sorted(g.key for g in right.gate_set)),
        distribution_valid=valid,
    )


def compare_all_pairs(
    gate: GateId,
    anchor_id: str,
    views: Sequence[ModelParentView],
    pairs: Sequence[tuple[str, str]] | None = None,
) -> list[PairComparison]:
    """Compare every requested model pair (default: all unordered pairs in model order)."""

    by_model = {view.model: view for view in views}
    if pairs is None:
        pairs = list(itertools.combinations([view.model for view in views], 2))
    return [compare_pair(gate, anchor_id, by_model[a], by_model[b]) for a, b in pairs]
