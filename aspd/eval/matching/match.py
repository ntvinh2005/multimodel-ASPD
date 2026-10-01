"""Top-1 pairing of components to features, and the random-pairing control."""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor


@dataclass(frozen=True)
class Pairing:

    name: str
    pairs: list[tuple[int, int]]
    """What the judge is shown: `(c, j)` on `c2o`, `(i, j)` on `i2o`."""

    scores: list[float]

    components: list[int]

    input_scores: list[float]
    """The INPUT leg's selection score `A_in[c, i]` -- EMPTY on `c2o`, which has no input leg."""

    def __post_init__(self) -> None:
        n = len(self.pairs)
        assert len(self.scores) == n and len(self.components) == n
        assert len(self.input_scores) in (0, n), (len(self.input_scores), n)

    @property
    def identity_rate(self) -> float:
        """Fraction of `i2o` pairs whose input match is the component's own index."""
        if not self.pairs or not self.input_scores:
            return float("nan")
        return sum(i == c for (i, _), c in zip(self.pairs, self.components)) / len(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)


def eligible(
    dead_components: Bool[Tensor, " c"],
    dead_features: Bool[Tensor, " f"],
    judgeable_components: set[int],
    judgeable_features: set[int],
    dead_input_features: Bool[Tensor, " f_in"] | None = None,
    judgeable_input_features: set[int] | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Indices on every side in play that are alive AND have enough harvested examples to judge."""

    def alive(dead: Bool[Tensor, " n"], judgeable: set[int]) -> Tensor:
        return torch.tensor(
            [i for i in range(dead.numel()) if not dead[i] and i in judgeable], dtype=torch.long
        )

    assert (dead_input_features is None) == (judgeable_input_features is None), (
        "the input side needs both its dead mask and its judgeable set, or neither"
    )
    comps = alive(dead_components, judgeable_components)
    feats = alive(dead_features, judgeable_features)
    assert comps.numel() > 0, "no eligible components; check min_examples and the harvest"
    assert feats.numel() > 0, "no eligible output features; check min_examples and the harvest"
    if dead_input_features is None or judgeable_input_features is None:
        return comps, feats, None
    in_feats = alive(dead_input_features, judgeable_input_features)
    assert in_feats.numel() > 0, "no eligible input features; check min_examples and the harvest"
    return comps, feats, in_feats


def subsample_components(components: Tensor, *, n_subsample: int, seed: int) -> Tensor:
    """The ONE component subsample every pairing is built on."""
    generator = torch.Generator().manual_seed(seed)
    n = min(n_subsample, components.numel())
    return components[torch.randperm(components.numel(), generator=generator)[:n]].sort().values


def top1(
    alignment: Float[Tensor, "c f"],
    chosen: Tensor,
    features: Tensor,
    *,
    name: str,
) -> Pairing:
    """`c2o`: `π(c) = argmax_j A[c, j]` over eligible `j`, for the shared component subsample."""
    block = alignment[chosen][:, features]
    best = block.argmax(dim=1)
    comps = [int(c) for c in chosen.tolist()]
    return Pairing(
        name=name,
        pairs=[(c, int(features[j])) for c, j in zip(comps, best.tolist())],
        scores=block.gather(1, best[:, None]).squeeze(1).tolist(),
        components=comps,
        input_scores=[],
    )


def chained_top1(
    alignment_in: Float[Tensor, "c f_in"],
    alignment_out: Float[Tensor, "c f"],
    chosen: Tensor,
    input_features: Tensor,
    features: Tensor,
    *,
    name: str,
) -> Pairing:
    out_block = alignment_out[chosen][:, features]
    in_block = alignment_in[chosen][:, input_features]
    best_out = out_block.argmax(dim=1)
    best_in = in_block.argmax(dim=1)
    pairs = [
        (int(input_features[i]), int(features[j]))
        for i, j in zip(best_in.tolist(), best_out.tolist())
    ]
    return Pairing(
        name=name,
        pairs=pairs,
        scores=out_block.gather(1, best_out[:, None]).squeeze(1).tolist(),
        components=[int(c) for c in chosen.tolist()],
        input_scores=in_block.gather(1, best_in[:, None]).squeeze(1).tolist(),
    )


def random_control(matched: Pairing, features: Tensor, *, seed: int) -> Pairing:
    """`matched`'s own LEFT side, paired with a uniformly drawn eligible OUTPUT feature."""
    generator = torch.Generator().manual_seed(seed + 1)
    partners = features[torch.randint(features.numel(), (len(matched),), generator=generator)]
    return Pairing(
        name="random",
        pairs=[(i, int(j)) for (i, _), j in zip(matched.pairs, partners.tolist())],
        scores=[float("nan")] * len(matched),
        components=list(matched.components),
        input_scores=list(matched.input_scores),
    )
