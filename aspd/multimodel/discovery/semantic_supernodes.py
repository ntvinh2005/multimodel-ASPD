"""Semantic partitions and same-anchor evidence for ASPD representative components."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import isfinite

import numpy as np
from scipy.cluster.hierarchy import cut_tree, fcluster, linkage
from scipy.optimize import OptimizeWarning, linprog
from scipy.spatial.distance import cdist, squareform
from scipy.special import logsumexp
from scipy.stats import rankdata

from aspd.multimodel.discovery.semantic_ot import marginals, normalized
from aspd.multimodel.posthoc.ids import ComponentId


@dataclass(frozen=True)
class Distribution:
    anchors: np.ndarray
    mass: np.ndarray

    def __post_init__(self):
        if len(self.anchors) != len(self.mass) or not np.all(self.mass > 0):
            raise ValueError("A distribution requires one positive mass per support point")
        object.__setattr__(self, "mass", normalized(self.mass))


def mixture(distributions, weights=None):
    if not distributions:
        raise ValueError("An empty supernode has no semantic distribution")
    weights = normalized(np.ones(len(distributions)) if weights is None else weights)
    if len(weights) != len(distributions):
        raise ValueError("Provide one weight per member")
    merged = defaultdict(float)
    for item, weight in zip(distributions, weights, strict=True):
        for anchor, mass in zip(item.anchors, item.mass, strict=True):
            merged[int(anchor)] += float(weight * mass)
    keys = sorted(key for key, mass in merged.items() if mass > 0)
    return Distribution(np.asarray(keys), np.asarray([merged[key] for key in keys]))


def transport(cost, left, right, *, epsilon=None):
    """Return ground transport cost <T,C> and feasible T, including under regularization."""
    left, right = normalized(left), normalized(right)
    if (
        cost.shape != (len(left), len(right))
        or not np.isfinite(cost).all()
        or np.any(cost < 0)
        or np.any(left <= 0)
        or np.any(right <= 0)
    ):
        raise ValueError("Invalid balanced transport inputs")
    if epsilon is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Unrecognized options detected.*", OptimizeWarning)
            result = linprog(
                cost.ravel(),
                A_eq=marginals(*cost.shape),
                b_eq=np.r_[left, right],
                bounds=(0, None),
                method="highs",
                options={"threads": 1},
            )
        if not result.success:
            raise RuntimeError(result.message)
        plan = result.x.reshape(cost.shape)
    else:
        if not isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Sinkhorn epsilon must be finite and positive")
        kernel = -cost / epsilon
        v = np.zeros_like(right)
        for _ in range(5000):
            u = np.log(left) - logsumexp(kernel + v, axis=1)
            v = np.log(right) - logsumexp(kernel + u[:, None], axis=0)
            plan = np.exp(kernel + u[:, None] + v)
            if np.max(abs(plan.sum(1) - left)) <= 1e-8:
                break
    if not (
        np.allclose(plan.sum(1), left, atol=1e-7, rtol=0)
        and np.allclose(plan.sum(0), right, atol=1e-7, rtol=0)
    ):
        raise RuntimeError("Transport plan violates the balanced marginal constraints")
    return float(np.sum(plan * cost)), plan


def cost_without_pairs(cost, left, right, excluded):
    """Exact transport after excluding same-document cells. None if infeasible."""
    bounds = [(0, 0) if value else (0, None) for value in excluded.ravel()]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Unrecognized options detected.*", OptimizeWarning)
        solution = linprog(
            cost.ravel(),
            A_eq=marginals(*cost.shape),
            b_eq=np.r_[normalized(left), normalized(right)],
            bounds=bounds,
            method="highs",
            options={"threads": 1},
        )
    if solution.status == 2:
        return None
    if not solution.success:
        raise RuntimeError(solution.message)
    return float(solution.fun)


def compare(left, right, vectors, *, epsilon=None):
    if left is right and epsilon is None:
        return 0.0, np.diag(left.mass)
    return transport(
        cdist(vectors[left.anchors], vectors[right.anchors]), left.mass, right.mass, epsilon=epsilon
    )


def pairwise(distributions, vectors, *, epsilon=None):
    result = np.zeros((len(distributions), len(distributions)))
    for a, b in combinations(range(len(distributions)), 2):
        result[a, b] = result[b, a] = compare(
            distributions[a],
            distributions[b],
            vectors,
            epsilon=epsilon,
        )[0]
    return result


def tree_and_partition(matrix, count=None, threshold=None, *, method="average"):
    if (count is None) == (threshold is None):
        raise ValueError("Use exactly one cluster cut")
    condensed = squareform(matrix, checks=True)
    if not len(matrix) or np.any(~np.isfinite(condensed)) or np.any(condensed < 0):
        raise ValueError("A complete nonnegative distance matrix is required")
    if count is not None and not 1 <= count <= len(matrix):
        raise ValueError("Cluster count exceeds cohort")
    if threshold is not None and (not isfinite(threshold) or threshold < 0):
        raise ValueError("Cluster threshold must be finite and nonnegative")
    if len(matrix) == 1:
        return np.empty((0, 4)), [[0]]
    tree = linkage(condensed, method=method)
    labels = (
        cut_tree(tree, n_clusters=[count])[:, 0]
        if count is not None
        else fcluster(tree, t=threshold, criterion="distance")
    )
    return tree, groups_by(labels)


def groups_by(labels):
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    return list(groups.values())


def cost_summary(groups, distance):
    """Report Eq. (5) and per-pair average without counting singletons as pairs."""
    costs = [distance[a, b] for group in groups for a, b in combinations(group, 2)]
    objective = sum(
        distance[np.ix_(group, group)].sum() / (len(group) * (len(group) - 1))
        for group in groups
        if len(group) > 1
    )
    return {
        "objective": float(objective),
        "pairs": len(costs),
        "mean": float(np.mean(costs)) if costs else None,
    }


def coassignment(left, right):
    def pairs(groups):
        return {tuple(sorted(pair)) for g in groups for pair in combinations(g, 2)}

    a, b = pairs(left), pairs(right)
    return len(a & b) / len(a | b) if a or b else None


def rank_correlation(left, right):
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def aggregate_taylor(source, targets, observed, scales):
    """Evaluate one exact model/anchor scope. An absent record conveys no coefficient."""
    ids = {key: ComponentId.from_key(key) for key in set(source) | set(targets)}
    eligible = {
        a: [b for b in source if ids[b].matrix.is_upstream_of(ids[a].matrix)] for a in targets
    }
    eligible = {a: bs for a, bs in eligible.items() if bs}
    needed = sum(map(len, eligible.values()))
    details = []
    for a, bs in eligible.items():
        known = {b: float(observed[b, a]) for b in bs if (b, a) in observed}
        if known and (a not in scales or not isfinite(scales[a]) or scales[a] <= 0):
            raise ValueError(f"Missing or invalid target normalization for {a}")
        if not all(isfinite(effect) for effect in known.values()):
            raise ValueError("Nonfinite Taylor effect")
        if known:
            signed = sum(known.values())
            details.append(
                {
                    "target": a,
                    "expected": len(bs),
                    "effects": known,
                    "signed": signed,
                    "weight": abs(signed) * scales[a],
                }
            )
    measured = sum(len(item["effects"]) for item in details)
    complete = needed > 0 and measured == needed
    score = float(np.mean([item["weight"] for item in details])) if details else None
    return {
        "status": (
            "undefined"
            if not needed
            else "complete"
            if complete
            else "truncated"
            if measured
            else "unavailable"
        ),
        "score": score if complete else None,
        "observedTargetMean": score if measured and not complete else None,
        "eligibleTargets": len(eligible),
        "observedTargets": len(details),
        "eligiblePairs": needed,
        "observedPairs": measured,
        "targets": details,
    }


def concatenate_hidden_layers(layers, selected=None):
    """Assemble one model's reference rows, keeping layer units and order as supplied."""
    chosen = layers if selected is None else [layers[i] for i in selected]
    if not chosen or any(np.ndim(x) != 2 or not np.isfinite(x).all() for x in chosen):
        raise ValueError("Expected finite [aligned anchors, hidden dimensions] rows per layer")
    return np.concatenate(chosen, axis=1)
