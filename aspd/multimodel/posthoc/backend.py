"""Backend contract: what a data/model source must provide to the post-hoc pipeline.

The analysis modules never import a checkpoint format, a tokenizer or a stream.  They
consume the small objects below, so a new source (a toy model, the v3.1 500M
checkpoint, a future v3.2 run) only needs to implement ``PosthocBackend``.

Shapes
------
``ModelInputs`` stores one model's padded documents and the gather map from the shared
anchor axis ``alpha`` to that model's native token positions ``t^{(n)}(alpha)``.  The
anchor axis is shared: row ``r`` of every model and every gate tensor is the same
canonical anchor.

``AnchorPool.gates[site]`` is the binary shared gate ``g_{p,alpha,i}`` with shape
``[A, C]``.  It must come from the natural pool-wide BatchTopK of the trained encoder
(or a synthetic stand-in in tests); the pipeline never forces support.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from aspd.multimodel.posthoc.ids import MatrixId


@dataclass(frozen=True)
class MatrixHookSpec:
    """Where the logical matrix ``j`` lives inside the frozen source model."""

    matrix: MatrixId
    module_path: str
    #: ``(start, stop)`` for a logical slice of a fused physical projection.
    output_slice: tuple[int, int] | None
    d_in: int
    d_out: int


@dataclass(frozen=True)
class ModelInputs:
    """Padded documents of one model plus the anchor gather map."""

    input_ids: Tensor  # [D, T] long
    attention_mask: Tensor  # [D, T] long
    anchor_document: Tensor  # [A] long, document row of each shared anchor
    anchor_token: Tensor  # [A] long, native token position of each shared anchor

    def __post_init__(self) -> None:
        if self.input_ids.shape != self.attention_mask.shape or self.input_ids.ndim != 2:
            raise ValueError("input_ids and attention_mask must both be [D,T]")
        if self.anchor_document.shape != self.anchor_token.shape or self.anchor_token.ndim != 1:
            raise ValueError("anchor gather maps must be matching [A] tensors")

    @property
    def anchor_count(self) -> int:
        return int(self.anchor_token.numel())

    def to(self, device: torch.device | str) -> ModelInputs:
        return ModelInputs(*(t.to(device) for t in (
            self.input_ids, self.attention_mask, self.anchor_document, self.anchor_token
        )))

    def subset(self, rows: Sequence[int]) -> ModelInputs:
        """Keep only the documents of ``rows`` and truncate to the last needed token.

        Causal models make positions after the last anchor irrelevant, so truncation
        does not change any captured value.  The returned anchors are ``rows`` in order.
        """

        index = torch.as_tensor(list(rows), dtype=torch.long, device=self.anchor_token.device)
        if index.numel() == 0:
            raise ValueError("subset requires at least one anchor row")
        documents = self.anchor_document[index]
        unique, local = torch.unique(documents, sorted=True, return_inverse=True)
        tokens = self.anchor_token[index]
        length = int(tokens.max()) + 1
        return ModelInputs(
            input_ids=self.input_ids[unique, :length],
            attention_mask=self.attention_mask[unique, :length],
            anchor_document=local,
            anchor_token=tokens,
        )


@dataclass(frozen=True)
class AnchorPool:
    """One natural BatchTopK pool on the shared anchor axis."""

    anchor_ids: tuple[str, ...]
    gates: Mapping[str, Tensor]  # site -> [A, C] bool
    models: Mapping[str, ModelInputs]

    def __post_init__(self) -> None:
        count = len(self.anchor_ids)
        if len(set(self.anchor_ids)) != count:
            raise ValueError("anchor IDs must be unique within a pool")
        for site, gate in self.gates.items():
            if gate.ndim != 2 or gate.shape[0] != count:
                raise ValueError(f"gate {site!r} must be [A,C] with A={count}")
        for key, inputs in self.models.items():
            if inputs.anchor_count != count:
                raise ValueError(f"model {key!r} gather map does not match the anchor axis")

    @property
    def anchor_count(self) -> int:
        return len(self.anchor_ids)


@dataclass(frozen=True)
class EncodedAnchorPool:
    """Binary post-hoc pool plus the exact selected continuous code ``g^s``.

    Existing dependency code consumes :attr:`pool` and is unchanged.  Evaluation tools
    that need activation magnitudes consume :attr:`continuous`; the invariant below
    prevents a positive candidate score from being mistaken for selected BatchTopK support.
    """

    pool: AnchorPool
    continuous: Mapping[str, Tensor]  # site -> [A,C], zero outside selected support

    def __post_init__(self) -> None:
        for site, values in self.continuous.items():
            if site not in self.pool.gates or values.shape != self.pool.gates[site].shape:
                raise ValueError(f"continuous gate {site!r} does not match binary support")
            if not torch.equal(values > 0, self.pool.gates[site].bool()):
                raise ValueError(f"continuous/binary gate mismatch at site {site!r}")


class PosthocBackend(Protocol):
    """Source of frozen models, trained factors and natural anchor pools."""

    #: Deterministic model order ``n = 1..N``.
    model_keys: tuple[str, ...]
    #: Dictionary size ``C`` per site.
    n_features: int
    device: torch.device

    def hook_specs(self, model: str) -> tuple[MatrixHookSpec, ...]:
        """Every selected logical matrix of ``model``."""

    def factors(self, model: str) -> Mapping[MatrixId, tuple[Tensor, Tensor]]:
        """``(U [C,d_out], V [C,d_in])`` for every matrix of ``model`` on ``device``."""

    def source_model(self, model: str) -> nn.Module:
        """The frozen source model; called with ``input_ids`` and ``attention_mask``."""

    def pools(self, count: int, *, skip: int = 0) -> Iterator[AnchorPool]:
        """Yield pools ``skip, ..., skip + count - 1`` of one deterministic pool sequence.

        The sequence must be reproducible across processes: the pipeline runs the
        importance stage on pools ``[0, n_importance)`` and the edge stage on the next
        pools, possibly in separate jobs, and relies on that to keep anchors disjoint.
        """

    def provenance(self) -> dict[str, Any]:
        """Immutable identifiers recorded in every artifact manifest."""

    def close(self) -> None:
        """Release models, streams and temporary files."""
