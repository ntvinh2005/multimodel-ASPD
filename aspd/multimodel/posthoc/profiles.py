"""Parent and edge profiles: readable per-component, per-gate and per-edge-type outputs.

The compare stage reduces everything to pairwise Jaccard / TV numbers. This stage keeps
the objects behind them. It is CPU-only and reads the verified ``importance``,
``representatives`` and ``edges`` stages.

``parents_per_anchor.safetensors`` (+ ``parents_vocab.json``)
    ``Pa_{K_par,tau}(a; alpha)`` for every analysed (downstream component, anchor): each
    parent's component, signed ``E_{b->a}``, weight ``w_{b->a}`` and rank. Formulation
    objects, stored in full.

``component_parent_profile.jsonl``
    One line per representative component ``a``, aggregated over its anchors: for each
    parent ``b``, how often it is in ``Pa`` and its mean signed ``E``, mean ``|E|``,
    mean ``w`` and fraction of positive ``E``. Plus parent-gate counts by ``gamma(b)``.
    Aggregation across anchors is ADDED (descriptive); the top ``top_parents`` parents
    are listed, ranked by frequency and then mean ``w``. The full lists are in
    ``parents_per_anchor``.

``parent_distributions.safetensors`` (+ ``distributions_vocab.json``)
    ``S^{(n)}_{p,i}(xi; alpha)`` and ``pi^{(n)}_{p,i}(xi; alpha)`` for every analysed (gate,
    model, anchor) and parent gate ``xi``, with ``|R|``, ``M`` and the validity flag per
    (gate, model, anchor).  Formulation objects, stored in full.

``gate_parent_profile.jsonl``
    One line per analysed gate ``(p, i)``. For each model: the mean of
    ``pi^{(n)}_{p,i}(xi; alpha)`` over valid anchors, the mean total strength ``M``, and the
    counts of anchors and valid anchors. Also the parent gates in every model's top list.
    The means over anchors are ADDED (descriptive).

``edge_profile.json``
    Distribution of all stored edge scores (and of parents only), per model and per
    (upstream matrix type, downstream matrix type, layer gap): count, mean ``|E|``, mean
    ``E``, positive fraction, and a log10 ``|E|`` histogram with approximate quantiles.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.posthoc.comparison import (
    aggregated_parent_scores,
    edge_weight,
    parent_distribution,
)
from aspd.multimodel.posthoc.edges import EdgeRecord
from aspd.multimodel.posthoc.ids import MATRIX_NAMES, ComponentId, GateId
from aspd.multimodel.posthoc.parents import Parent

#: log10 |E| histogram edges used by the edge profile.
LOG_BINS = torch.arange(-10.0, 3.0001, 0.1)


# ----------------------------------------------------------- component profile
@dataclass
class _ParentStats:
    count: int = 0
    sum_score: float = 0.0
    sum_abs: float = 0.0
    sum_weight: float = 0.0
    positive: int = 0


@dataclass
class ComponentProfile:
    """Aggregates ``Pa(a; alpha)`` of one downstream component over its anchors."""

    component: ComponentId
    write_norm: float
    sigma: float
    anchors: int = 0
    anchors_with_parents: int = 0
    sum_coefficient: float = 0.0
    parents: dict[ComponentId, _ParentStats] = field(default_factory=dict)
    parent_gates: dict[GateId, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, coefficient: float, parents: Sequence[Parent]) -> None:
        self.anchors += 1
        self.sum_coefficient += coefficient
        if parents:
            self.anchors_with_parents += 1
        for parent in parents:
            stats = self.parents.setdefault(parent.component, _ParentStats())
            stats.count += 1
            stats.sum_score += parent.score
            stats.sum_abs += abs(parent.score)
            stats.sum_weight += edge_weight(parent.score, self.write_norm, self.sigma)
            stats.positive += parent.score > 0
            self.parent_gates[parent.component.gate] += 1

    def to_json(self, top_parents: int) -> dict[str, Any]:
        ranked = sorted(
            self.parents.items(),
            key=lambda item: (-item[1].count, -item[1].sum_weight / item[1].count, item[0]),
        )
        own_gate = self.component.gate
        return {
            "component": self.component.key,
            "model": self.component.model,
            "gate": own_gate.key,
            "anchors": self.anchors,
            "anchors_with_parents": self.anchors_with_parents,
            "mean_coefficient": self.sum_coefficient / self.anchors if self.anchors else None,
            "distinct_parents": len(self.parents),
            "top_parents": [
                {
                    "parent": b.key,
                    "parent_gate": b.gate.key,
                    "same_gate": b.gate == own_gate,
                    "frequency": s.count / self.anchors,
                    "anchors": s.count,
                    "mean_E": s.sum_score / s.count,
                    "mean_abs_E": s.sum_abs / s.count,
                    "mean_w": s.sum_weight / s.count,
                    "positive_fraction": s.positive / s.count,
                }
                for b, s in ranked[:top_parents]
            ],
            "parent_gate_counts": dict(sorted(
                ((g.key, n) for g, n in self.parent_gates.items()), key=lambda kv: (-kv[1], kv[0])
            )[:top_parents]),
        }


# ---------------------------------------------------------------- gate profile
@dataclass
class GateModelProfile:
    """Mean ``pi^{(n)}_{p,i}`` and ``M`` of one gate in one model over valid anchors."""

    representatives: tuple[ComponentId, ...]
    anchors: int = 0
    valid: int = 0
    sum_strength: float = 0.0
    sum_pi: dict[GateId, float] = field(default_factory=lambda: defaultdict(float))

    def add(
        self,
        parents: Mapping[ComponentId, Sequence[Parent]],
        write_norms: Mapping[ComponentId, float],
        sigmas: Mapping[ComponentId, float],
    ) -> tuple[dict[GateId, float], dict[GateId, float] | None]:
        """Add one anchor; return its ``S`` and ``pi`` (``None`` when invalid)."""

        scores = aggregated_parent_scores(parents, self.representatives, write_norms, sigmas)
        pi = parent_distribution(scores)
        self.add_distribution(scores, pi)
        return scores, pi

    def add_distribution(
        self, scores: Mapping[GateId, float], pi: Mapping[GateId, float] | None
    ) -> None:
        self.anchors += 1
        if not self.representatives or pi is None:
            return
        self.valid += 1
        self.sum_strength += sum(scores.values())
        for gate, value in pi.items():
            self.sum_pi[gate] += value

    def mean_pi(self) -> dict[GateId, float]:
        return {g: v / self.valid for g, v in self.sum_pi.items()} if self.valid else {}

    def to_json(self, top: int) -> dict[str, Any]:
        mean_pi = sorted(self.mean_pi().items(), key=lambda kv: (-kv[1], kv[0]))
        return {
            "representatives": [a.key for a in self.representatives],
            "anchors": self.anchors,
            "valid_anchors": self.valid,
            "mean_M": self.sum_strength / self.valid if self.valid else None,
            "distinct_parent_gates": len(mean_pi),
            "top_parent_gates": [{"gate": g.key, "mean_pi": v} for g, v in mean_pi[:top]],
        }


def shared_top_parent_gates(profiles: Mapping[str, GateModelProfile], top: int) -> list[str]:
    """Parent gates in the top-``top`` mean-pi list of every model that has valid anchors."""

    tops = [
        {g for g, _ in sorted(p.mean_pi().items(), key=lambda kv: (-kv[1], kv[0]))[:top]}
        for p in profiles.values() if p.valid
    ]
    if len(tops) < 2:
        return []
    return sorted(g.key for g in set.intersection(*tops))


# ------------------------------------------------- per-anchor S and pi rows
class DistributionRows:
    """Columnar store of ``S`` and ``pi`` per (gate, model, anchor, xi) (flushed per gate)."""

    HEADER = (("gate", torch.int32), ("model", torch.int16), ("anchor", torch.int32),
              ("representatives", torch.int16), ("M", torch.float64), ("valid", torch.bool))
    ENTRY = (("row", torch.int64), ("xi", torch.int32), ("S", torch.float64),
             ("pi", torch.float64))

    def __init__(self, models: Sequence[str]) -> None:
        self.models = {m: i for i, m in enumerate(models)}
        self.gates: dict[GateId, int] = {}
        self.anchor_ids: dict[str, int] = {}
        self._rows = 0
        self._pending: dict[str, list[Any]] = defaultdict(list)
        self._chunks: dict[str, list[Tensor]] = defaultdict(list)

    def _id(self, table: dict, key: Any) -> int:
        return table.setdefault(key, len(table))

    def add(self, gate: GateId, model: str, anchor_id: str, representatives: int,
            scores: Mapping[GateId, float], pi: Mapping[GateId, float] | None) -> None:
        row = self._rows
        self._rows += 1
        valid = bool(representatives) and pi is not None
        for name, value in (("gate", self._id(self.gates, gate)), ("model", self.models[model]),
                            ("anchor", self._id(self.anchor_ids, anchor_id)),
                            ("representatives", representatives),
                            ("M", float(sum(scores.values()))), ("valid", valid)):
            self._pending[name].append(value)
        for xi, s in scores.items():
            self._pending["row"].append(row)
            self._pending["xi"].append(self._id(self.gates, xi))
            self._pending["S"].append(s)
            self._pending["pi"].append(pi[xi] if pi is not None else math.nan)

    def flush(self) -> None:
        if not self._pending:
            return
        for name, dtype in (*self.HEADER, *self.ENTRY):
            self._chunks[name].append(torch.tensor(self._pending[name], dtype=dtype))
        self._pending = defaultdict(list)

    def tensors(self) -> dict[str, Tensor]:
        self.flush()
        out = {}
        for prefix, columns in (("header_", self.HEADER), ("entry_", self.ENTRY)):
            for name, dtype in columns:
                chunks = self._chunks[name]
                out[prefix + name] = torch.cat(chunks) if chunks else torch.zeros(0, dtype=dtype)
        return out

    def vocab(self) -> dict[str, Any]:
        return {
            "models": [m for m, _ in sorted(self.models.items(), key=lambda kv: kv[1])],
            "gates": [g.key for g, _ in sorted(self.gates.items(), key=lambda kv: kv[1])],
            "anchor_ids": [a for a, _ in sorted(self.anchor_ids.items(), key=lambda kv: kv[1])],
        }


# ------------------------------------------------------- per-anchor parent rows
class ParentRows:
    """Columnar store of every ``Pa(a; alpha)`` entry (flushed to tensors per gate)."""

    COLUMNS = (("anchor", torch.int32), ("downstream", torch.int32), ("parent", torch.int32),
               ("rank", torch.int16), ("E", torch.float32), ("w", torch.float32))

    def __init__(self) -> None:
        self.anchor_ids: dict[str, int] = {}
        self.components: dict[ComponentId, int] = {}
        self._pending: dict[str, list[float | int]] = defaultdict(list)
        self._chunks: dict[str, list[Tensor]] = defaultdict(list)

    def _id(self, table: dict, key: Any) -> int:
        return table.setdefault(key, len(table))

    def add(self, anchor_id: str, a: ComponentId, parents: Sequence[Parent],
            write_norm: float, sigma: float) -> None:
        anchor = self._id(self.anchor_ids, anchor_id)
        down = self._id(self.components, a)
        for rank, parent in enumerate(parents):
            self._pending["anchor"].append(anchor)
            self._pending["downstream"].append(down)
            self._pending["parent"].append(self._id(self.components, parent.component))
            self._pending["rank"].append(rank)
            self._pending["E"].append(parent.score)
            self._pending["w"].append(edge_weight(parent.score, write_norm, sigma))

    def flush(self) -> None:
        if not self._pending:
            return
        for name, dtype in self.COLUMNS:
            self._chunks[name].append(torch.tensor(self._pending[name], dtype=dtype))
        self._pending = defaultdict(list)

    def tensors(self) -> dict[str, Tensor]:
        self.flush()
        return {
            name: torch.cat(self._chunks[name]) if self._chunks[name] else torch.zeros(0, dtype=d)
            for name, d in self.COLUMNS
        }

    def vocab(self) -> dict[str, list[str]]:
        return {
            "anchor_ids": [a for a, _ in sorted(self.anchor_ids.items(), key=lambda kv: kv[1])],
            "components": [c.key for c, _ in sorted(self.components.items(), key=lambda kv: kv[1])],
        }

    def edge_profile(self, models: Sequence[str]) -> EdgeProfile:
        """The edge profile restricted to parents (entries of some ``Pa``)."""

        profile = EdgeProfile(models)
        cols = self.tensors()
        if cols["E"].numel() == 0:
            return profile
        model_index = {m: i for i, m in enumerate(models)}
        type_index = {t: i for i, t in enumerate(EdgeProfile.TYPES)}
        ordered = sorted(self.components.items(), key=lambda kv: kv[1])
        c_model = torch.tensor([model_index[c.model] for c, _ in ordered], dtype=torch.int16)
        c_type = torch.tensor([type_index[c.matrix.matrix] for c, _ in ordered], dtype=torch.int16)
        c_layer = torch.tensor([c.matrix.layer for c, _ in ordered], dtype=torch.int16)
        down, up = cols["downstream"].long(), cols["parent"].long()
        profile.add(c_model[down], c_type[up], c_type[down], c_layer[down] - c_layer[up], cols["E"])
        return profile


# ----------------------------------------------------------------- edge profile
class EdgeProfile:
    """Vectorized aggregates of edge scores by (model, upstream type, downstream type, gap)."""

    TYPES = MATRIX_NAMES

    def __init__(self, models: Sequence[str]):
        self.models = list(models)
        size = len(self.models) * len(self.TYPES) ** 2 * 64
        self.count = torch.zeros(size, dtype=torch.float64)
        self.sum_abs = torch.zeros(size, dtype=torch.float64)
        self.sum = torch.zeros(size, dtype=torch.float64)
        self.positive = torch.zeros(size, dtype=torch.float64)
        self.hist = {m: torch.zeros(len(LOG_BINS) + 1, dtype=torch.float64) for m in self.models}

    def _key(self, model: Tensor, up_type: Tensor, down_type: Tensor, gap: Tensor) -> Tensor:
        t = len(self.TYPES)
        return ((model * t + up_type) * t + down_type) * 64 + gap.clamp(0, 63)

    def add(self, model: Tensor, up_type: Tensor, down_type: Tensor, gap: Tensor,
            scores: Tensor) -> None:
        """All arguments are aligned 1-D tensors, one entry per edge."""

        key = self._key(model.long(), up_type.long(), down_type.long(), gap.long())
        size = self.count.numel()
        s = scores.double()
        self.count += torch.bincount(key, minlength=size).double()
        self.sum_abs += torch.bincount(key, weights=s.abs(), minlength=size)
        self.sum += torch.bincount(key, weights=s, minlength=size)
        self.positive += torch.bincount(key, weights=(s > 0).double(), minlength=size)
        log = torch.log10(s.abs().clamp_min(1e-30)).float()
        bucket = torch.bucketize(log, LOG_BINS)
        for index, name in enumerate(self.models):
            mask = model == index
            if bool(mask.any()):
                self.hist[name] += torch.bincount(bucket[mask], minlength=len(LOG_BINS) + 1).double()

    def add_records(self, records: Sequence[EdgeRecord]) -> None:
        """Convenience path for small inputs (tests, parents-only profile)."""

        model_index = {m: i for i, m in enumerate(self.models)}
        type_index = {t: i for i, t in enumerate(self.TYPES)}
        cols: dict[str, list[int]] = defaultdict(list)
        scores: list[Tensor] = []
        for r in records:
            for k in range(len(r)):
                b = r.component(k)
                cols["model"].append(model_index[r.downstream.model])
                cols["up"].append(type_index[b.matrix.matrix])
                cols["down"].append(type_index[r.downstream.matrix.matrix])
                cols["gap"].append(r.downstream.matrix.layer - b.matrix.layer)
            scores.append(r.scores)
        if not scores or not cols["model"]:
            return
        self.add(*(torch.tensor(cols[c]) for c in ("model", "up", "down", "gap")),
                 torch.cat(scores))

    def to_json(self) -> dict[str, Any]:
        t = len(self.TYPES)
        cells = []
        for key in self.count.nonzero(as_tuple=True)[0].tolist():
            gap = key % 64
            down = (key // 64) % t
            up = (key // 64 // t) % t
            model = key // 64 // t // t
            n = float(self.count[key])
            cells.append({
                "model": self.models[model], "upstream": self.TYPES[up],
                "downstream": self.TYPES[down], "layer_gap": gap, "count": int(n),
                "mean_abs_E": float(self.sum_abs[key]) / n, "mean_E": float(self.sum[key]) / n,
                "positive_fraction": float(self.positive[key]) / n,
            })
        return {"cells": cells, "abs_E_quantiles_log10": {
            m: _hist_quantiles(h) for m, h in self.hist.items()
        }, "log10_bins": [round(float(x), 2) for x in LOG_BINS]}


def _hist_quantiles(hist: Tensor) -> dict[str, float | None]:
    total = float(hist.sum())
    if total == 0:
        return {}
    cdf = torch.cumsum(hist, 0) / total
    out: dict[str, float | None] = {"count": total}
    for q in (0.1, 0.5, 0.9, 0.99):
        index = int(torch.searchsorted(cdf, torch.tensor(q, dtype=cdf.dtype)))
        # upper edge of the bin; None if the quantile lies above the last bin (|E| > 10^3)
        out[f"q{int(q * 100)}"] = float(LOG_BINS[index]) if index < len(LOG_BINS) else None
    return out


def edge_profile_from_table(table: Any, models: Sequence[str]) -> EdgeProfile:
    """Vectorized pass over every stored edge of an ``EdgeTable``."""

    profile = EdgeProfile(models)
    model_index = {m: i for i, m in enumerate(models)}
    type_index = {t: i for i, t in enumerate(EdgeProfile.TYPES)}
    for part in table.parts:
        t = part.tensors
        if t["score"].numel() == 0:
            continue
        up_type = torch.tensor([type_index[m.matrix] for m in part.matrices], dtype=torch.int16)
        up_layer = torch.tensor([m.layer for m in part.matrices], dtype=torch.int16)
        down_model = torch.tensor([model_index[c.model] for c in part.downstream], dtype=torch.int16)
        down_type = torch.tensor([type_index[c.matrix.matrix] for c in part.downstream],
                                 dtype=torch.int16)
        down_layer = torch.tensor([c.matrix.layer for c in part.downstream], dtype=torch.int16)
        lengths = (t["record_stop"] - t["record_start"])
        per_entry_down = torch.repeat_interleave(t["record_downstream"].long(), lengths)
        upm = t["upstream_matrix"].long()
        profile.add(
            down_model[per_entry_down], up_type[upm], down_type[per_entry_down],
            down_layer[per_entry_down] - up_layer[upm], t["score"],
        )
        del per_entry_down, upm
    return profile


def dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, allow_nan=False)
