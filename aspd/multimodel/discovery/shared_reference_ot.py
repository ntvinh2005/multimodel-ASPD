"""Exact cosine OT and complete linkage in the pooled ASPD shared representation."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from aspd.multimodel.discovery.semantic_supernodes import (
    Distribution,
    transport,
    tree_and_partition,
)


def top_anchors(strengths, count):
    """Keep distinct positive anchors, breaking equal strengths by anchor ID."""
    best = {}
    for anchor, strength in strengths:
        if not np.isfinite(strength) or strength < 0:
            raise ValueError(f"Invalid contribution at {anchor}")
        if strength > 0:
            best[anchor] = max(best.get(anchor, 0.0), strength)
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))[:count]


def unit_reference(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Expected finite [anchor, PRE + MID] reference vectors")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms <= 0).any():
        raise ValueError("Cosine cost is undefined for a zero shared reference vector")
    return vectors / norms[:, None]


def cosine_cost(left, right):
    return np.clip(1 - left @ right.T, 0, 2)


def pair_cost(left: Distribution, right: Distribution, unit_vectors):
    cost = cosine_cost(unit_vectors[left.anchors], unit_vectors[right.anchors])
    # A reference row is one canonical anchor. Its cosine dissimilarity to itself is zero.
    cost[left.anchors[:, None] == right.anchors[None, :]] = 0
    return transport(cost, left.mass, right.mass)


def dissimilarities(distributions, unit_vectors):
    result = np.full((len(distributions), len(distributions)), np.nan)
    indices = [i for i, item in enumerate(distributions) if item is not None]
    result[indices, indices] = 0
    for i, j in combinations(indices, 2):
        result[i, j] = result[j, i] = pair_cost(distributions[i], distributions[j], unit_vectors)[0]
    return result


def complete_partition(distance, threshold):
    tree, groups = tree_and_partition(distance, threshold=threshold, method="complete")
    if any(np.max(distance[np.ix_(group, group)]) > threshold for group in groups):
        raise RuntimeError("Complete linkage produced a group outside the requested diameter")
    return tree, groups
