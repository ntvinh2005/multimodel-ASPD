"""Identities used throughout the post-hoc analysis.

Formulation reference: ``docs/multimodel/multimodel_aspd_formulation.tex``
(sections "Setup" and "Post-hoc analysis").

* ``MatrixId``  -- one selected weight matrix ``j = (l, m)`` of model ``n``.
* ``ComponentId`` -- one rank-one component ``P^{(n)}_{j,i}``; ``i`` is the shared
  feature index.
* ``GateId`` -- one shared gate ``(p, i)``; ``p`` is the PRE/MID site.  The map
  ``gamma(b) = (p(j_b), i_b)`` from a component to its gate is ``ComponentId.gate``.

Compute order inside one decoder layer (used for "upstream of"):

    q, k, v  (stage 0, read the PRE residual)
    o        (stage 1, reads the attention output)
    gate, up (stage 2, read the MID residual)
    down     (stage 3, reads the MLP hidden state)

Matrices in the same stage cannot influence each other within one forward pass, so
``b`` is upstream of ``a`` iff it belongs to the same model and
``(layer_b, stage_b) < (layer_a, stage_a)`` lexicographically.
"""

from __future__ import annotations

from dataclasses import dataclass

SITES: tuple[str, ...] = ("pre", "mid")

#: Logical matrix names, identical to ``aspd.multimodel.v3.config.V3_LOGICAL_MATRICES``.
MATRIX_NAMES: tuple[str, ...] = ("q", "k", "v", "o", "mlp_gate", "mlp_up", "mlp_down")

#: ``p(j)``: the gate site that controls each logical matrix (v3.1 specification).
MATRIX_SITE: dict[str, str] = {
    "q": "pre",
    "k": "pre",
    "v": "pre",
    "o": "mid",
    "mlp_gate": "mid",
    "mlp_up": "mid",
    "mlp_down": "mid",
}

#: Position of each logical matrix in the forward computation of one layer.
MATRIX_STAGE: dict[str, int] = {
    "q": 0,
    "k": 0,
    "v": 0,
    "o": 1,
    "mlp_gate": 2,
    "mlp_up": 2,
    "mlp_down": 3,
}


@dataclass(frozen=True, order=True)
class MatrixId:
    """Selected weight matrix ``j = (layer, matrix)`` of model ``model``."""

    model: str
    layer: int
    matrix: str

    def __post_init__(self) -> None:
        if self.matrix not in MATRIX_SITE:
            raise ValueError(f"unknown logical matrix {self.matrix!r}")
        if self.layer < 0:
            raise ValueError("layer index must be non-negative")

    @property
    def key(self) -> str:
        """Same spelling as ``LogicalMatrixTarget.key`` in the v3 registry."""

        return f"{self.model}.L{self.layer:02d}.{self.matrix}"

    @classmethod
    def from_key(cls, key: str) -> MatrixId:
        model, layer, matrix = key.split(".", 2)
        if not layer.startswith("L"):
            raise ValueError(f"malformed matrix key {key!r}")
        return cls(model, int(layer[1:]), matrix)

    @property
    def site(self) -> str:
        """``p(j)``."""

        return MATRIX_SITE[self.matrix]

    @property
    def order(self) -> tuple[int, int]:
        return (self.layer, MATRIX_STAGE[self.matrix])

    def is_upstream_of(self, other: MatrixId) -> bool:
        """True iff an output of ``self`` can change the input of ``other``."""

        return self.model == other.model and self.order < other.order


@dataclass(frozen=True, order=True)
class GateId:
    """Shared gate identity ``(p, i)``; identical across models and layers."""

    site: str
    feature: int

    def __post_init__(self) -> None:
        if self.site not in SITES:
            raise ValueError(f"unknown site {self.site!r}")
        if self.feature < 0:
            raise ValueError("feature index must be non-negative")

    @property
    def key(self) -> str:
        return f"{self.site}/{self.feature}"

    @classmethod
    def from_key(cls, key: str) -> GateId:
        site, feature = key.split("/", 1)
        return cls(site, int(feature))


@dataclass(frozen=True, order=True)
class ComponentId:
    """Rank-one component ``P^{(n)}_{j,i}``: matrix ``j`` of model ``n``, feature ``i``."""

    matrix: MatrixId
    feature: int

    @property
    def model(self) -> str:
        return self.matrix.model

    @property
    def gate(self) -> GateId:
        """``gamma(b) = (p(j_b), i_b)``."""

        return GateId(self.matrix.site, self.feature)

    @property
    def key(self) -> str:
        return f"{self.matrix.key}/{self.feature}"

    @classmethod
    def from_key(cls, key: str) -> ComponentId:
        matrix, feature = key.rsplit("/", 1)
        return cls(MatrixId.from_key(matrix), int(feature))
