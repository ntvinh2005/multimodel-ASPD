"""Activation examples for the judge from a `harvest.db`, binned by activation interval."""

from collections.abc import Callable
from dataclasses import dataclass

from param_decomp_lab.harvest.db import HarvestDB
from param_decomp_lab.harvest.schemas import ComponentData

ACT_THRESHOLD = 1e-3
NO_EXAMPLES = "NO EXAMPLE."


@dataclass(frozen=True)
class Example:
    """One window. `to_str` is upstream's `Example.to_str` character for character."""

    str_toks: list[str]
    acts: list[float]

    @property
    def peak(self) -> float:
        return max(self.acts) if self.acts else 0.0

    def to_str(self, mark_toks: bool = False) -> str:
        return (
            "".join(
                f"<<{tok}>>" if (mark_toks and act > ACT_THRESHOLD) else tok
                for tok, act in zip(self.str_toks, self.acts, strict=True)
            )
            .replace("�", "")
            .replace("\n", "↵")
        )


def format_feature_examples(examples: list[Example]) -> str:
    """Upstream's `format_feature_examples`. The empty case is load-bearing: the judge's prompt
    instructs it to answer DIFFERENT whenever either side shows this string.
    """
    if not examples:
        return NO_EXAMPLES
    return "\n".join(f"{i + 1}. {ex.to_str(mark_toks=True)}" for i, ex in enumerate(examples))


def feature_examples(
    comp: ComponentData,
    decode: Callable[[list[int]], list[str]],
    activation_key: str,
    *,
    n_intervals: int = 3,
    k_per_interval: int = 3,
    buffer: int = 10,
) -> list[Example]:
    keys = sorted({k for ex in comp.activation_examples for k in ex.activations})
    assert activation_key in keys or not keys, (
        f"{comp.component_key} has no {activation_key!r} series; harvested keys are {keys}. "
        "A decomposition harvest stores causal_importance / component_activation; an SAE "
        "harvest stores activation -- the wrong key yields an empty, plausible-looking prompt"
    )

    candidates: list[tuple[float, Example]] = []
    for stored in comp.activation_examples:
        acts = stored.activations.get(activation_key, [])
        if not acts:
            continue
        peak = max(acts)
        if peak <= 0:
            continue
        centre = acts.index(peak)
        toks = decode(stored.token_ids)
        assert len(toks) == len(acts), (len(toks), len(acts), comp.component_key)
        lo, hi = max(0, centre - buffer), min(len(toks), centre + buffer + 1)
        candidates.append((peak, Example(toks[lo:hi], acts[lo:hi])))

    if not candidates:
        return []
    candidates.sort(key=lambda c: -c[0])
    max_act = candidates[0][0]

    step = (max_act - ACT_THRESHOLD) / n_intervals
    selected: list[Example] = []
    for i in range(n_intervals, 0, -1):
        lo_b = ACT_THRESHOLD + step * (i - 1)
        hi_b = ACT_THRESHOLD + step * i
        # `candidates` is already peak-descending, so the first `k` in a bin ARE the k highest.
        taken = 0
        for peak, example in candidates:
            if taken == k_per_interval:
                break
            in_bin = lo_b <= peak <= hi_b + 1e-4 if i == n_intervals else lo_b <= peak < hi_b
            if in_bin:
                selected.append(example)
                taken += 1

    selected.sort(key=lambda e: -e.peak)
    return selected


def example_counts(db: HarvestDB, key_prefix: str, min_examples: int) -> set[int]:
    return {
        int(key.rsplit(":", 1)[1])
        for key in db.get_eligible_component_keys(min_examples)
        if key.rsplit(":", 1)[0] == key_prefix
    }
