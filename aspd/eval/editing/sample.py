"""Which output-SAE features are edited: the eligible set, the sample and the reusable pool."""

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

DEFAULT_MAX_DENSITY = 0.2
DEFAULT_MIN_SUPPORT = 100
DEFAULT_N_FEATURES = 50
DEFAULT_SEED = 42


def harvest_densities(harvest_db: Path, site: str, n_features: int) -> Tensor:
    """Per-latent firing density at `site`, as a dense `[F]` tensor with 0 for absent rows."""
    harvest_db = Path(harvest_db)
    assert harvest_db.exists(), f"no harvest.db at {harvest_db}"
    density = torch.zeros(n_features, dtype=torch.float64)
    seen = 0
    with sqlite3.connect(f"file:{harvest_db}?immutable=1", uri=True) as con:
        rows = con.execute(
            "SELECT component_idx, firing_density FROM components WHERE layer = ?", (site,)
        ).fetchall()
    for idx, value in rows:
        assert 0 <= idx < n_features, (
            f"{harvest_db} has latent index {idx} at site {site!r}, but the loaded dictionary has "
            f"{n_features} latents -- this harvest describes a different dictionary"
        )
        density[idx] = value
        seen += 1
    assert seen, (
        f"{harvest_db} holds no rows for site {site!r}; it has "
        f"{sorted({r[0] for r in con.execute('SELECT layer FROM components')})}"
    )
    return density


def dictionary_fingerprint(w_enc: Tensor, w_dec: Tensor, threshold: Tensor) -> str:
    """A short identity for one dictionary, so a sample file cannot be read against another."""
    parts = [
        torch.tensor(list(w_enc.shape) + list(w_dec.shape), dtype=torch.int64),
        w_enc.detach().float().norm(dim=0).cpu(),
        w_dec.detach().float().norm(dim=1).cpu(),
        threshold.detach().float().reshape(-1).cpu(),
    ]
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


@dataclass
class FeatureSample:
    """The drawn features plus every filter that produced them, so a reader can reproduce it."""

    site: str
    n_latents: int
    fingerprint: str
    seed: int
    max_density: float
    min_support: int
    n_tokens: int
    feature_ids: list[int]
    density: list[float]
    """Harvest firing density of each drawn feature, index-aligned with `feature_ids`."""
    support: list[int]
    """`|A_j|` on the study's own batch, index-aligned with `feature_ids`."""
    n_eligible: int
    """Features passing BOTH filters. The draw is uniform over these."""
    n_density_eligible: int
    """Features passing the density band alone -- the support filter's cost, made visible."""
    source: dict[str, str] = field(default_factory=dict)

    def ids_tensor(self, device: torch.device | str = "cpu") -> Tensor:
        return torch.tensor(self.feature_ids, dtype=torch.long, device=device)


def draw_sample(
    *,
    density: Tensor,
    support: Tensor,
    site: str,
    fingerprint: str,
    n_features: int = DEFAULT_N_FEATURES,
    seed: int = DEFAULT_SEED,
    max_density: float = DEFAULT_MAX_DENSITY,
    min_support: int = DEFAULT_MIN_SUPPORT,
    n_tokens: int = 0,
    source: dict[str, str] | None = None,
) -> FeatureSample:
    """Uniform draw over `{j : 0 < density_j < max_density and support_j >= min_support}`."""
    assert density.shape == support.shape, (density.shape, support.shape)
    density_ok = (density > 0) & (density < max_density)
    eligible = (density_ok & (support >= min_support)).nonzero(as_tuple=True)[0]
    assert eligible.numel() >= n_features, (
        f"only {eligible.numel()} features pass density in (0, {max_density}) AND support >= "
        f"{min_support} over {n_tokens} tokens, but {n_features} were asked for. Lower "
        f"--min-support (density alone leaves {int(density_ok.sum())}) or raise --n-tokens"
    )
    generator = torch.Generator().manual_seed(seed)
    picked = eligible[torch.randperm(eligible.numel(), generator=generator)[:n_features]]
    picked = picked.sort().values
    return FeatureSample(
        site=site,
        n_latents=int(density.numel()),
        fingerprint=fingerprint,
        seed=seed,
        max_density=max_density,
        min_support=min_support,
        n_tokens=int(n_tokens),
        feature_ids=[int(i) for i in picked],
        density=[float(density[i]) for i in picked],
        support=[int(support[i]) for i in picked],
        n_eligible=int(eligible.numel()),
        n_density_eligible=int(density_ok.sum()),
        source=source or {},
    )


DEFAULT_POOL_SIZE = 600
DEFAULT_N_COMBINATIONS = 50
DEFAULT_M_VALUES = (1, 5, 10, 20, 50)


@dataclass
class FeaturePool:

    site: str
    n_latents: int
    fingerprint: str
    seed: int
    max_density: float
    min_support: int
    n_tokens: int
    feature_ids: list[int]
    density: list[float]
    support: list[int]
    n_seed_features: int
    n_eligible: int
    n_density_eligible: int
    source: dict[str, str] = field(default_factory=dict)

    def index_of(self) -> dict[int, int]:
        return {j: i for i, j in enumerate(self.feature_ids)}


def draw_pool(
    *,
    density: Tensor,
    support: Tensor,
    site: str,
    fingerprint: str,
    seed_features: list[int],
    pool_size: int = DEFAULT_POOL_SIZE,
    seed: int = DEFAULT_SEED,
    max_density: float = DEFAULT_MAX_DENSITY,
    min_support: int = DEFAULT_MIN_SUPPORT,
    n_tokens: int = 0,
    source: dict[str, str] | None = None,
) -> FeaturePool:
    """`seed_features` verbatim, then a uniform draw from the eligible set to fill `pool_size`."""
    assert density.shape == support.shape, (density.shape, support.shape)
    density_ok = (density > 0) & (density < max_density)
    eligible_mask = density_ok & (support >= min_support)
    for j in seed_features:
        assert bool(eligible_mask[j]), (
            f"sample feature {j} is not eligible under density in (0, {max_density}) and "
            f"support >= {min_support} over {n_tokens} tokens -- the pool is being drawn against a "
            "different token budget than the sample was"
        )
    eligible = eligible_mask.nonzero(as_tuple=True)[0]
    assert eligible.numel() >= pool_size, (
        f"only {eligible.numel()} eligible features, but a pool of {pool_size} was asked for. "
        f"Lower --pool-size (density alone leaves {int(density_ok.sum())}) or raise --n-tokens"
    )
    seen = set(seed_features)
    generator = torch.Generator().manual_seed(seed)
    order = eligible[torch.randperm(eligible.numel(), generator=generator)]
    extra = [int(j) for j in order if int(j) not in seen][: pool_size - len(seed_features)]
    ids = list(seed_features) + sorted(extra)
    assert len(ids) == pool_size and len(set(ids)) == pool_size, (len(ids), len(set(ids)))
    return FeaturePool(
        site=site,
        n_latents=int(density.numel()),
        fingerprint=fingerprint,
        seed=seed,
        max_density=max_density,
        min_support=min_support,
        n_tokens=int(n_tokens),
        feature_ids=ids,
        density=[float(density[j]) for j in ids],
        support=[int(support[j]) for j in ids],
        n_seed_features=len(seed_features),
        n_eligible=int(eligible.numel()),
        n_density_eligible=int(density_ok.sum()),
        source=source or {},
    )


def draw_combinations(
    pool: FeaturePool,
    m: int,
    *,
    n_combinations: int = DEFAULT_N_COMBINATIONS,
    seed: int = DEFAULT_SEED,
) -> list[list[int]]:
    assert 0 < m <= len(pool.feature_ids), (m, len(pool.feature_ids))
    if m == 1:
        assert pool.n_seed_features >= n_combinations, (
            f"the m=1 arm is the sample verbatim, but the pool carries only "
            f"{pool.n_seed_features} seed features and {n_combinations} combinations were asked for"
        )
        return [[j] for j in pool.feature_ids[:n_combinations]]
    generator = torch.Generator().manual_seed(seed * 1_000_003 + m * 1009)
    ids = torch.tensor(pool.feature_ids, dtype=torch.long)
    return [
        sorted(int(j) for j in ids[torch.randperm(ids.numel(), generator=generator)[:m]])
        for _ in range(n_combinations)
    ]


def write_pool(pool: FeaturePool, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(pool), indent=2, sort_keys=True))
    print(f"[attr_edit] wrote {path} ({len(pool.feature_ids)} features)", flush=True)
    return path


def load_pool(path: Path, *, fingerprint: str | None, site: str | None = None) -> FeaturePool:
    pool = FeaturePool(**json.loads(Path(path).read_text()))
    if site is not None:
        assert pool.site == site, (
            f"{path} was drawn at site {pool.site!r}, this run evaluates {site!r}"
        )
    if fingerprint is not None:
        assert pool.fingerprint == fingerprint, (
            f"{path} was drawn from dictionary {pool.fingerprint}, the loaded one is "
            f"{fingerprint}. Feature ids are meaningless across dictionaries"
        )
    return pool


def write_sample(sample: FeatureSample, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(sample), indent=2, sort_keys=True))
    print(f"[attr_edit] wrote {path} ({len(sample.feature_ids)} features)", flush=True)
    return path


def load_sample(path: Path, *, fingerprint: str | None, site: str | None = None) -> FeatureSample:
    """Read a sample and refuse it if it describes a different dictionary."""
    sample = FeatureSample(**json.loads(Path(path).read_text()))
    if site is not None:
        assert sample.site == site, (
            f"{path} was drawn at site {sample.site!r}, this run evaluates {site!r}"
        )
    if fingerprint is not None:
        assert sample.fingerprint == fingerprint, (
            f"{path} was drawn from dictionary {sample.fingerprint}, the loaded one is "
            f"{fingerprint}. Feature ids are meaningless across dictionaries -- point --features "
            "at the right file, or drop it to draw a fresh sample"
        )
    return sample
