"""Diversity: mean pairwise Jaccard overlap between the token sets components fire on.

T_c is the set of token types component c fires on at least `min_firings` times among its
examples; sim_{c,d} = |T_c & T_d| / |T_c | T_d|; sim is the mean over pairs of a uniform sample of
components inside a firing-density band. A bootstrap over components gives its interval.
"""

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson
from scipy import sparse


@dataclass(frozen=True)
class Population:
    """One harvest's band-filtered components and their token sets."""

    keys: list[str]
    token_sets: list[np.ndarray]
    n_band: int
    """C -- the full band population, of which `keys` is a uniform sample."""
    n_eligible: int
    n_dropped: int


def load_population(
    db_path: Path,
    *,
    min_density: float,
    max_density: float,
    min_firings: int,
    n_sample: int,
    seed: int,
    min_examples: int = 5,
    threshold: float | None = None,
    stats: dict | None = None,
) -> Population:
    """Band-filter a harvest.db and read a uniform sample of components' token sets."""
    if threshold is not None:
        assert stats is not None, "a threshold needs the stats its densities and scales come from"
        return _load_population_thresholded(
            db_path, min_density=min_density, max_density=max_density, min_firings=min_firings,
            n_sample=n_sample, seed=seed, min_examples=min_examples, threshold=threshold,
            stats=stats,
        )
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    n_eligible = conn.execute(
        "SELECT COUNT(*) FROM components WHERE n_activation_examples >= ?", (min_examples,)
    ).fetchone()[0]
    band = [
        r["component_key"]
        for r in conn.execute(
            "SELECT component_key FROM components WHERE n_activation_examples >= ? "
            "AND firing_density BETWEEN ? AND ? ORDER BY component_key",
            (min_examples, min_density, max_density),
        )
    ]

    rng = np.random.default_rng(seed)
    sampled = band
    if len(band) > n_sample:
        sampled = sorted(band[i] for i in rng.choice(len(band), size=n_sample, replace=False))

    keys, token_sets = [], []
    for key in sampled:
        row = conn.execute(
            "SELECT activation_examples FROM components WHERE component_key = ?", (key,)
        ).fetchone()
        hits: defaultdict[int, int] = defaultdict(int)
        for ex in orjson.loads(row["activation_examples"]):
            for tid, firing in zip(ex["token_ids"], ex["firings"], strict=True):
                if firing:
                    hits[tid] += 1
        kept = np.array(sorted(t for t, n in hits.items() if n >= min_firings), dtype=np.int64)
        if len(kept) < 2:
            continue  # a one-token set carries no overlap information
        keys.append(key)
        token_sets.append(kept)
    conn.close()

    assert keys, f"no component in [{min_density:g}, {max_density:g}] has a usable token set"
    return Population(
        keys=keys,
        token_sets=token_sets,
        n_band=len(band),
        n_eligible=n_eligible,
        n_dropped=len(sampled) - len(keys),
    )


def _load_population_thresholded(
    db_path: Path,
    *,
    min_density: float,
    max_density: float,
    min_firings: int,
    n_sample: int,
    seed: int,
    min_examples: int,
    threshold: float,
    stats: dict,
) -> Population:
    """`load_population` at a raised firing threshold; the sampling is the same, the inputs are not."""
    from param_decomp_lab.harvest.schemas import ActivationExample

    from aspd.eval.harvest_threshold import anchor_value, surviving_mask

    criterion = stats.get("criterion", "ci")
    name = f"{threshold:g}"
    assert name in stats["thresholds"], f"stats hold {stats['thresholds']}, not {name}"
    half = int(stats["half"])
    held_by_key: dict[str, dict] = stats["components"]

    eligible = {k: h for k, h in held_by_key.items() if h[name][0] >= min_examples}
    band = sorted(k for k, h in eligible.items() if min_density <= h[name][1] <= max_density)

    # The same draw as the unthresholded path, over this path's band.
    rng = np.random.default_rng(seed)
    sampled = band
    if len(band) > n_sample:
        sampled = sorted(band[i] for i in rng.choice(len(band), size=n_sample, replace=False))

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    keys, token_sets = [], []
    for key in sampled:
        scale = held_by_key[key].get("scale", 1.0)
        row = conn.execute(
            "SELECT activation_examples FROM components WHERE component_key = ?", (key,)
        ).fetchone()
        hits: defaultdict[int, int] = defaultdict(int)
        for raw in orjson.loads(row["activation_examples"]):
            ex = ActivationExample(**raw)
            if anchor_value(ex, criterion, half)[0] <= threshold * scale:
                continue
            for tid, alive in zip(ex.token_ids,
                                  surviving_mask(ex, criterion, threshold, scale), strict=True):
                if alive:
                    hits[tid] += 1
        kept = np.array(sorted(t for t, n in hits.items() if n >= min_firings), dtype=np.int64)
        if len(kept) < 2:
            continue  # a one-token set carries no overlap information
        keys.append(key)
        token_sets.append(kept)
    conn.close()

    assert keys, f"no component in [{min_density:g}, {max_density:g}] has a usable token set"
    return Population(
        keys=keys,
        token_sets=token_sets,
        n_band=len(band),
        n_eligible=len(eligible),
        n_dropped=len(sampled) - len(keys),
    )


def jaccard_kernel(token_sets: list[np.ndarray]) -> np.ndarray:
    """Dense [n, n] Jaccard similarity over token sets, unit diagonal."""
    sizes = np.array([len(s) for s in token_sets], dtype=np.float64)
    vocab = int(max(s.max() for s in token_sets)) + 1
    rows = np.repeat(np.arange(len(token_sets)), sizes.astype(int))
    membership = sparse.csr_matrix(
        (np.ones(int(sizes.sum()), dtype=np.float32), (rows, np.concatenate(token_sets))),
        shape=(len(token_sets), vocab),
    )
    inter = (membership @ membership.T).toarray().astype(np.float64)
    union = sizes[:, None] + sizes[None, :] - inter
    z = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(z, 1.0)
    return z


def mean_off_diagonal(z: np.ndarray) -> float:
    """`z_bar` -- mean Jaccard over all pairs c != d."""
    n = z.shape[0]
    assert n > 1, "diversity is undefined for a single component"
    return float(z[~np.eye(n, dtype=bool)].mean())


def bootstrap(
    z: np.ndarray, n_band: int, *, n_boot: int, seed: int, alpha: float = 0.05
) -> dict[str, dict[str, float]]:
    """Bootstrap of `z_bar` over resampled components: mean, std and the percentile interval."""
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    draws: dict[str, list[float]] = {"z_bar": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        zb = z[np.ix_(idx, idx)]
        dup = idx[:, None] == idx[None, :]
        np.fill_diagonal(dup, True)
        off = ~dup
        if not off.any():
            continue
        zbar_b = float(zb[off].mean())
        draws["z_bar"].append(zbar_b)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)),
            "lo": float(np.percentile(v, lo)),
            "hi": float(np.percentile(v, hi)),
        }
        for k, v in draws.items()
    }
