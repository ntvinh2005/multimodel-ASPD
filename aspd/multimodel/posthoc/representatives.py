"""Representative component selection (formulation, "Representative Component Selection").

    R_{p,i}^{(n)} = { (j,i) : j in TopK_{K_rep}( score_{.,i}^{(n)} ),  score_{j,i}^{(n)} > tau_rep }

with ``j`` ranging over ``J_p^{(n)}`` (the matrices of model ``n`` controlled by site
``p``).  The formulation uses ``beta`` as the score and leaves the choice between
``beta`` and ``rho`` open, so the metric is an explicit argument with no default.

Ties are broken deterministically by ``MatrixId`` order (layer, then logical name), and
``NaN`` scores are never selected.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from aspd.multimodel.posthoc.ids import ComponentId, GateId
from aspd.multimodel.posthoc.importance import ImportanceTable

SCREENING_METRICS = ("beta", "rho")


@dataclass(frozen=True)
class RepresentativeRule:
    metric: str  # "beta" or "rho"
    k_rep: int
    tau_rep: float

    def __post_init__(self) -> None:
        if self.metric not in SCREENING_METRICS:
            raise ValueError(f"screening metric must be one of {SCREENING_METRICS}")
        if self.k_rep < 1:
            raise ValueError("k_rep must be positive")
        if not self.tau_rep >= 0:
            raise ValueError("tau_rep must be non-negative")


def select_representatives(
    table: ImportanceTable, model: str, gate: GateId, rule: RepresentativeRule
) -> tuple[ComponentId, ...]:
    """Return ``R_{p,i}^{(n)}`` ordered by decreasing score."""

    scored: list[tuple[float, ComponentId]] = []
    for matrix in table.matrices(model=model, site=gate.site):
        value = float(table[matrix].score(rule.metric)[gate.feature])
        if math.isnan(value) or not value > rule.tau_rep:
            continue
        scored.append((value, ComponentId(matrix, gate.feature)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(component for _value, component in scored[: rule.k_rep])


def representative_sets(
    table: ImportanceTable,
    models: Iterable[str],
    gates: Iterable[GateId],
    rule: RepresentativeRule,
) -> dict[tuple[str, GateId], tuple[ComponentId, ...]]:
    """``R_{p,i}^{(n)}`` for every requested model and gate (empty sets included)."""

    models = tuple(models)
    return {
        (model, gate): select_representatives(table, model, gate, rule)
        for gate in gates
        for model in models
    }
