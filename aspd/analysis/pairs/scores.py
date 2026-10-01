"""Pair scores between components (or features).

`dot`: <u_c1, v_c2>; `cosine`; `dot_coact`: Interact(c1, c2) = kappa(c1, c2) <u_c1, v_c2>; for
query-key pairs the head-restricted <u^h_q, u^h_k>.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from aspd.analysis.pairs.spaces import Space, head_pairs
from aspd.analysis.pairs.weights import Directions

Metric = Literal["cosine", "dot", "dot_coact"]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    source: Literal["weights", "kappa", "db"]
    formula: str
    note: str


METRIC_SPECS: list[MetricSpec] = [
    MetricSpec(
        "dot_coact", "co-activation-weighted dot", "kappa",
        "kappa(a, b) * <x_a, y_b>,  kappa(a, b) = E_x[g_a g_b a_a]",
        "The attribution patch in closed form: geometry weighted by how much the two are actually "
        "on together. Needs a data pass; module x module only.",
    ),
    MetricSpec(
        "cosine", "cosine similarity", "weights",
        "<x_a, y_b> / (|x_a| |y_b|)",
        "Scale-free. Always available.",
    ),
    MetricSpec(
        "dot", "raw geometry (dot product)", "weights",
        "<x_a, y_b>",
        "Magnitude-aware: ranks by the size of the first-order contribution, not its direction.",
    ),
    MetricSpec(
        "jacobian", "Jacobian-weighted composition", "db",
        "E_x[ <x_a * phi'(h(x)), y_b> ]",
        "First-order composition through the intervening pointwise nonlinearity. Needs a data pass.",
    ),
    MetricSpec(
        "coactivation", "co-activation (Jaccard)", "db",
        "|T_a ^ T_b| / |T_a v T_b| over firing token sets",
        "Whether the two components are ever on together. Needs a data pass.",
    ),
    MetricSpec(
        "causal", "attribution edge (grad x act)", "db",
        "E_x[ (dz_b / dz_a) z_a ]",
        "Measured composition rather than geometric. Needs a data pass.",
    ),
]
WEIGHT_METRICS = {m.key for m in METRIC_SPECS if m.source == "weights"}
KAPPA_METRICS = {m.key for m in METRIC_SPECS if m.source == "kappa"}
#: Everything this module computes from directions, as opposed to reading whole out of a sidecar DB.
DIRECTION_METRICS = WEIGHT_METRICS | KAPPA_METRICS


def _slices(sa: Space, sb: Space) -> list[tuple[slice, slice]]:
    assert sa.heads is not None and sb.heads is not None
    return [
        (sa.heads.head_slice(ha), sb.heads.head_slice(hb))
        for ha, hb in head_pairs(sa.heads, sb.heads)
    ]


@dataclass(frozen=True)
class RowScores:
    """One component's scores against every component of the other endpoint."""

    top: Tensor
    bottom: Tensor
    top_head: Tensor | None
    bottom_head: Tensor | None
    mean: float
    std: float
    n_population: int


def score_row(
    xa: Directions,
    yb: Directions,
    idx: int,
    *,
    metric: Metric,
    space_a: Space,
    space_b: Space,
    per_head: bool,
    head: int | None,
    keep: Tensor | None = None,
    kappa: Tensor | None = None,
) -> RowScores:
    """Scores for component `idx` of endpoint A against every component of endpoint B."""
    assert metric in DIRECTION_METRICS, f"{metric} is not computed from directions"
    assert keep is None or (keep.dtype == torch.bool and keep.numel() == yb.mat.shape[0]), (
        "keep must be a bool mask over endpoint B's components"
    )
    assert (kappa is not None) == (metric in KAPPA_METRICS), (
        f"{metric} requires kappa exactly when it is a kappa metric"
    )
    assert kappa is None or kappa.numel() == yb.mat.shape[0], (
        "kappa must cover every component of endpoint B"
    )
    if not per_head:
        assert space_a.dim == space_b.dim, "flat scoring needs equal dimensions"
        s = yb.mat @ xa.mat[idx]
        if metric == "cosine":
            s = s / (xa.norms[idx].clamp_min(1e-12) * yb.norms.clamp_min(1e-12))
        if kappa is not None:
            s = s * kappa
        pop = s if keep is None else s[keep]
        return RowScores(s, s, None, None, float(pop.mean()), float(pop.std()), int(pop.numel()))

    pairs = _slices(space_a, space_b)
    head_ids = list(range(len(pairs)))
    if head is not None:
        assert 0 <= head < len(pairs), f"head {head} out of range (0..{len(pairs) - 1})"
        pairs, head_ids = [pairs[head]], [head]
    rows = []
    for sl_a, sl_b in pairs:
        v = xa.mat[idx, sl_a]
        m = yb.mat[:, sl_b]
        s = m @ v
        if metric == "cosine":
            s = s / (v.norm().clamp_min(1e-12) * m.norm(dim=1).clamp_min(1e-12))
        rows.append(s)
    stacked = torch.stack(rows)  # [H, C_B]
    if kappa is not None:
        stacked = stacked * kappa[None, :]
    pop = stacked if keep is None else stacked[:, keep]
    mean, std, n = float(pop.mean()), float(pop.std()), int(pop.numel())
    heads = torch.tensor(head_ids)
    top_v, top_i = stacked.max(dim=0)
    bot_v, bot_i = stacked.min(dim=0)
    return RowScores(top_v, bot_v, heads[top_i], heads[bot_i], mean, std, n)


def top_bottom(
    row: RowScores, *, k: int, exclude: int | None = None, covered: Tensor | None = None
) -> dict[str, object]:
    """Top-k and bottom-k, with both null models attached."""
    mean, std = row.mean, row.std
    n_b = int(row.top.numel())
    n_covered = n_b if covered is None else int(covered.sum())

    def side(vals: Tensor, heads: Tensor | None, largest: bool) -> list[dict[str, object]]:
        v = vals.clone()
        if covered is not None:
            v[~covered] = float("-inf") if largest else float("inf")
        masked = exclude is not None and 0 <= exclude < v.numel()
        if masked:
            v[exclude] = float("-inf") if largest else float("inf")
        n_avail = n_covered - (1 if masked and (covered is None or bool(covered[exclude])) else 0)
        kk = min(k, n_avail)
        if kk <= 0:
            return []
        sel_v, sel_i = torch.topk(v if largest else -v, kk)
        if not largest:
            sel_v = -sel_v
        out = []
        for val, i in zip(sel_v.tolist(), sel_i.tolist(), strict=True):
            entry: dict[str, object] = {
                "idx": int(i),
                "score": float(val),
                "z_empirical": (float(val) - mean) / std if std > 0 else 0.0,
            }
            if heads is not None:
                entry["head"] = int(heads[i])
            out.append(entry)
        return out

    return {
        "top": side(row.top, row.top_head, True),
        "bottom": side(row.bottom, row.bottom_head, False),
        "row_mean": mean,
        "row_std": std,
        "n_scored": row.n_population,
        "n_covered": n_covered,
        "n_components": n_b,
    }


def best_match_summary(
    xa: Directions,
    yb: Directions,
    *,
    metric: Metric,
    space_a: Space,
    space_b: Space,
    per_head: bool,
    chunk: int = 1024,
) -> dict[str, list[float]]:
    """Per-component best (max) and worst (min) score over all of B, for the overview histogram."""
    best, worst = [], []
    n_a = xa.mat.shape[0]
    for start in range(0, n_a, chunk):
        stop = min(start + chunk, n_a)
        if not per_head:
            block = xa.mat[start:stop] @ yb.mat.t()
            if metric == "cosine":
                block = block / (
                    xa.norms[start:stop, None].clamp_min(1e-12) * yb.norms[None, :].clamp_min(1e-12)
                )
        else:
            acc_hi, acc_lo = None, None
            for sl_a, sl_b in _slices(space_a, space_b):
                xs, ys = xa.mat[start:stop, sl_a], yb.mat[:, sl_b]
                blk = xs @ ys.t()
                if metric == "cosine":
                    blk = blk / (
                        xs.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        * ys.norm(dim=1)[None, :].clamp_min(1e-12)
                    )
                acc_hi = blk if acc_hi is None else torch.maximum(acc_hi, blk)
                acc_lo = blk if acc_lo is None else torch.minimum(acc_lo, blk)
            assert acc_hi is not None and acc_lo is not None
            best.extend(acc_hi.max(dim=1).values.tolist())
            worst.extend(acc_lo.min(dim=1).values.tolist())
            continue
        best.extend(block.max(dim=1).values.tolist())
        worst.extend(block.min(dim=1).values.tolist())
    return {"best": best, "worst": worst}
