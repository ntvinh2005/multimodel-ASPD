"""Edge scores ``E_{b -> a}`` (formulation, "Edge score").

Definition (first-order Taylor estimate of ablating upstream component ``b``)::

    E_{b->a}(alpha) = (d e_a / d e_b) * e_b  ~=  e_a(s) - e_a(s | e_b = 0)

Implementation in the *native* source model
-------------------------------------------
Component ``b = (j_b, i_b)`` writes ``e_b u_b`` into ``y_{j_b,alpha}``.  Setting ``e_b`` to
zero removes exactly that vector from the native output, so

    d e_a / d e_b = u_b^T  d e_a / d y_{j_b,alpha}.

``probed_forward`` adds a zero probe to every upstream ``y_j`` at the anchor and keeps
the graph.  One backward pass from ``e_a`` therefore gives ``d e_a / d y_j`` for all
upstream matrices at once, and

    E_{.->a}(alpha)[j, :] = e_{j,alpha,:} * (U_j  d e_a / d y_{j,alpha})       # [C]

for every component of every upstream matrix.  Only components with ``g_b = 1`` have
``e_b != 0``; only those are stored.

Conventions (the only implemented values of the required ``edges.conventions``
config fields; see ``config.EDGE_CONVENTIONS`` and ``docs/multimodel/posthoc_formulation_map.md``):

* Gates are frozen at their natural values: ``e_a = g_a (v_a^T x_a)`` with ``g_a``
  constant.  The encoder is not re-run under ablation.
* "Same anchor": ``e_b`` is taken at the anchor's own token position, as in the
  formulation.  Influence of ``b`` at earlier positions through attention is not
  attributed to ``E_{b->a}(alpha)``.
* Candidate upstream set ``U^{(n)}(a)``: every component of every selected matrix of
  the same model that is upstream of ``j_a`` in compute order (``MatrixId.is_upstream_of``).

Batching: several anchors can share one forward/backward pass only if they belong to
different documents.  Within one document a later anchor attends to an earlier one,
so summing their ``e_a`` would mix their gradients.  ``group_rows`` enforces this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from aspd.multimodel.posthoc.backend import MatrixHookSpec, ModelInputs
from aspd.multimodel.posthoc.ids import ComponentId, MatrixId
from aspd.multimodel.posthoc.source_hooks import Ablation, probed_forward


@dataclass(frozen=True)
class EdgeRecord:
    """All edge scores into one downstream component at one anchor.

    Stored compactly (a full run has ~10^6 records of ~700 entries): entry ``k`` is the
    upstream component ``ComponentId(matrices[matrix_index[k]], features[k])`` with edge
    score ``scores[k]``.  ``upstream`` materializes the ComponentIds for small uses.
    """

    anchor_id: str
    downstream: ComponentId
    coefficient: float  # e_a at this anchor
    matrices: tuple[MatrixId, ...]
    matrix_index: Tensor  # [n] int32 into ``matrices``
    features: Tensor  # [n] int32
    scores: Tensor  # [n] float32, E_{b->a}(alpha)

    def __len__(self) -> int:
        return int(self.scores.numel())

    def component(self, k: int) -> ComponentId:
        return ComponentId(self.matrices[int(self.matrix_index[k])], int(self.features[k]))

    @property
    def upstream(self) -> tuple[ComponentId, ...]:
        index, features = self.matrix_index.tolist(), self.features.tolist()
        return tuple(ComponentId(self.matrices[m], f) for m, f in zip(index, features, strict=True))

    @classmethod
    def from_components(
        cls, anchor_id: str, downstream: ComponentId, coefficient: float,
        upstream: Sequence[ComponentId], scores: Tensor,
    ) -> EdgeRecord:
        matrices = tuple(sorted({c.matrix for c in upstream}))
        lookup = {m: i for i, m in enumerate(matrices)}
        return cls(
            anchor_id, downstream, coefficient, matrices,
            torch.tensor([lookup[c.matrix] for c in upstream], dtype=torch.int32),
            torch.tensor([c.feature for c in upstream], dtype=torch.int32),
            scores.float().reshape(-1),
        )


def group_rows(documents: Tensor, rows: Sequence[int], max_group: int) -> list[list[int]]:
    """Greedy deterministic grouping of anchor rows with one row per document."""

    if max_group < 1:
        raise ValueError("max_group must be positive")
    groups: list[list[int]] = []
    group_documents: list[set[int]] = []
    for row in rows:
        document = int(documents[row])
        for group, seen in zip(groups, group_documents, strict=True):
            if len(group) < max_group and document not in seen:
                group.append(row)
                seen.add(document)
                break
        else:
            groups.append([row])
            group_documents.append({document})
    return groups


def edge_scores_for_group(
    model: nn.Module,
    inputs: ModelInputs,
    anchor_ids: Sequence[str],
    gates: Mapping[str, Tensor],
    specs: Sequence[MatrixHookSpec],
    factors: Mapping[MatrixId, tuple[Tensor, Tensor]],
    requests: Mapping[ComponentId, Tensor],
) -> list[EdgeRecord]:
    """Compute ``E_{.->a}`` for every requested downstream component.

    ``inputs`` holds ``G`` anchors from ``G`` distinct documents (``ModelInputs.subset``);
    ``gates[site]`` is ``[G, C]``; ``requests[a]`` is a ``[G]`` bool mask of the rows at
    which ``a`` is analysed.  Rows with ``g_a = 0`` produce ``e_a = 0`` and empty edges.
    """

    if len(anchor_ids) != inputs.anchor_count:
        raise ValueError("anchor_ids must match the group")
    if not requests:
        return []
    model_key = next(iter(requests)).model
    if any(component.model != model_key for component in requests):
        raise ValueError("one group call handles one source model")
    downstream = {component.matrix for component in requests}
    spec_matrices = {spec.matrix for spec in specs}
    upstream = sorted(
        matrix
        for matrix in spec_matrices
        if any(matrix.is_upstream_of(target) for target in downstream)
    )
    capture = set(upstream) | downstream

    with torch.enable_grad():
        forward = probed_forward(model, inputs, specs, probe=upstream, capture=capture)
        coefficients = {}
        with torch.no_grad():
            for matrix in upstream:
                U, V = factors[matrix]
                x = forward.inputs[matrix].detach().to(V.dtype)
                coefficients[matrix] = (x @ V.T) * gates[matrix.site].to(V.dtype)
        records: list[EdgeRecord] = []
        for component, mask in sorted(requests.items()):
            mask = mask.to(torch.bool)
            if not bool(mask.any()):
                continue
            _U_a, V_a = factors[component.matrix]
            g_a = gates[component.matrix.site][:, component.feature].to(V_a.dtype)
            x_a = forward.inputs[component.matrix].to(V_a.dtype)
            e_a = g_a * (x_a @ V_a[component.feature])  # [G], graph-attached
            parents = [m for m in upstream if m.is_upstream_of(component.matrix)]
            target = (e_a * mask.to(e_a.dtype)).sum()
            if parents and target.requires_grad:
                grads = torch.autograd.grad(
                    target, [forward.probes[m] for m in parents],
                    retain_graph=True, allow_unused=True,
                )
            else:
                grads = [None] * len(parents)
            parent_tuple = tuple(parents)
            for row in mask.nonzero(as_tuple=True)[0].tolist():
                blocks: list[Tensor] = []
                features: list[Tensor] = []
                values: list[Tensor] = []
                for block, (matrix, grad) in enumerate(zip(parents, grads, strict=True)):
                    e_b = coefficients[matrix][row]
                    active = (e_b != 0).nonzero(as_tuple=True)[0]
                    if active.numel() == 0:
                        continue
                    U_b, _V_b = factors[matrix]
                    if grad is None:
                        scores = torch.zeros(active.numel(), dtype=torch.float32)
                    else:
                        direction = U_b[active] @ grad[row].to(U_b.dtype)
                        scores = (e_b[active] * direction).float().cpu()
                    blocks.append(torch.full((active.numel(),), block, dtype=torch.int32))
                    features.append(active.to(torch.int32).cpu())
                    values.append(scores)
                empty_int = torch.zeros(0, dtype=torch.int32)
                records.append(
                    EdgeRecord(
                        anchor_id=anchor_ids[row],
                        downstream=component,
                        coefficient=float(e_a[row].detach()),
                        matrices=parent_tuple,
                        matrix_index=torch.cat(blocks) if blocks else empty_int,
                        features=torch.cat(features) if features else empty_int,
                        scores=torch.cat(values) if values else torch.zeros(0),
                    )
                )
    return records


@torch.no_grad()
def exact_ablation_effect(
    model: nn.Module,
    inputs: ModelInputs,
    gates: Mapping[str, Tensor],
    specs: Sequence[MatrixHookSpec],
    factors: Mapping[MatrixId, tuple[Tensor, Tensor]],
    downstream: ComponentId,
    upstream: ComponentId,
    *,
    fraction: float = 1.0,
) -> Tensor:
    """Return ``e_a(s) - e_a(s | e_b -> (1 - fraction) e_b)`` per anchor row ``[G]``.

    With ``fraction = 1`` this is the exact ablation that ``E_{b->a}`` approximates.
    For small ``fraction`` it equals ``fraction * E_{b->a}`` to first order, which is the
    finite-difference check used by the tests.  Gates stay frozen, as in ``E``.
    """

    if not upstream.matrix.is_upstream_of(downstream.matrix):
        raise ValueError(f"{upstream.key} is not upstream of {downstream.key}")
    _U, V_a = factors[downstream.matrix]
    g_a = gates[downstream.matrix.site][:, downstream.feature].to(V_a.dtype)

    def e_a(ablations: tuple[Ablation, ...]) -> Tensor:
        forward = probed_forward(
            model, inputs, specs, probe=(), capture={downstream.matrix},
            gates=gates, factors=factors, ablations=ablations,
        )
        x_a = forward.inputs[downstream.matrix].to(V_a.dtype)
        return g_a * (x_a @ V_a[downstream.feature])

    return (e_a(()) - e_a((Ablation(upstream, fraction),))).float().cpu()
