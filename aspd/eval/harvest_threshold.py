"""Read a finished harvest as if firing required g_{t,c} > tau (or an activation above tau).

Used for the VPD tau-filtered interpretability and diversity numbers.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, override

from param_decomp_lab.harvest.db import HarvestDB
from param_decomp_lab.harvest.schemas import ActivationExample, ComponentData

Criterion = Literal["ci", "act"]

CI_ACT_KEY = "causal_importance"
COMPONENT_ACT_KEY = "component_activation"

STATS_FILE: dict[Criterion, str] = {
    "ci": "ci_threshold_stats.json",
    "act": "act_threshold_stats.json",
}

VALUE_KEY: dict[Criterion, str] = {"ci": CI_ACT_KEY, "act": COMPONENT_ACT_KEY}


def anchor_index(firings: list[bool], half: int) -> int | None:
    n = len(firings)
    centre = n > half and firings[half]
    j = n - half - 1
    left = 0 <= j < n and firings[j]
    if centre and left and j != half:
        return None
    if centre:
        return half
    return j if left else None


def example_values(example: ActivationExample, criterion: Criterion) -> list[float]:
    """Per-position value the threshold is applied to, on the scale the criterion uses."""
    key = VALUE_KEY[criterion]
    assert key in example.activations, (
        f"example has no `{key}` -- the `{criterion}` criterion cannot be applied to this harvest"
    )
    values = example.activations[key]
    return [abs(v) for v in values] if criterion == "act" else list(values)


def component_scale(comp: ComponentData, criterion: Criterion) -> float:
    """The denominator the threshold is relative to; `1.0` for the absolute `ci` criterion."""
    if criterion == "ci":
        return 1.0
    peak = 0.0
    for e in comp.activation_examples:
        for v, f in zip(example_values(e, criterion), e.firings, strict=True):
            if f and v > peak:
                peak = v
    return peak


def surviving_mask(
    example: ActivationExample, criterion: Criterion, threshold: float, scale: float
) -> list[bool]:
    """The firing mask at `threshold`, always a SUBSET of the example's own."""
    cut = threshold * scale
    return [v > cut and f for v, f in
            zip(example_values(example, criterion), example.firings, strict=True)]


def anchor_value(
    example: ActivationExample, criterion: Criterion, half: int
) -> tuple[float, bool]:
    """`(value at the anchor, whether the anchor had to be guessed)` for one stored example."""
    values = example_values(example, criterion)
    i = anchor_index(example.firings, half)
    if i is not None:
        return values[i], False
    peak = max((v for v, f in zip(values, example.firings, strict=True) if f), default=None)
    assert peak is not None, "activation example with no firing token -- not a harvest window"
    return peak, True


def threshold_stats(
    harvest_dir: Path,
    thresholds: list[float],
    *,
    criterion: Criterion = "ci",
    min_examples: int,
    rebuild: bool = False,
) -> dict:
    """Per-component surviving example count and firing density at each threshold, cached."""
    harvest_dir = Path(harvest_dir)
    cache = harvest_dir / STATS_FILE[criterion]
    taus = [float(t) for t in thresholds]
    names = [f"{t:g}" for t in taus]
    if cache.exists() and not rebuild:
        held = json.loads(cache.read_text())
        # Caches without a `criterion` field are `ci` caches.
        if (held.get("criterion", "ci") == criterion
                and held.get("min_examples") == min_examples
                and all(n in held["thresholds"] for n in names)):
            return held

    db = HarvestDB(harvest_dir / "harvest.db", readonly=True)
    half = int(db.get_config_dict()["activation_context_tokens_per_side"])
    keys = [k for k, _ in db.get_component_densities(min_examples=min_examples)]
    print(
        f"[{criterion}-stats] one pass over {len(keys)} components of {harvest_dir} "
        f"for tau in {names} (minutes, cached to {STATS_FILE[criterion]})",
        flush=True,
    )
    components: dict[str, dict] = {}
    n_examples = n_guessed = n_no_scale = 0
    for i, key in enumerate(keys):
        comp = db.get_component(key)
        assert comp is not None
        scale = component_scale(comp, criterion)
        if scale <= 0.0:
            n_no_scale += 1
            components[key] = {"n0": len(comp.activation_examples),
                               "density0": comp.firing_density, "scale": 0.0,
                               **{n: [0, 0.0] for n in names}}
            continue
        pairs = [anchor_value(e, criterion, half) for e in comp.activation_examples]
        n_examples += len(pairs)
        n_guessed += sum(1 for _, guessed in pairs if guessed)
        values = [v for v, _ in pairs]
        entry: dict = {"n0": len(values), "density0": comp.firing_density, "scale": scale}
        for tau, name in zip(taus, names, strict=True):
            n = sum(1 for v in values if v > tau * scale)
            entry[name] = [n, comp.firing_density * n / len(values)]
        components[key] = entry
        if (i + 1) % 5000 == 0:
            print(f"[{criterion}-stats]   {i + 1}/{len(keys)}", flush=True)
    db.close()

    stats = {
        "criterion": criterion,
        "half": half,
        "min_examples": min_examples,
        "thresholds": names,
        "n_components": len(components),
        "n_examples": n_examples,
        "n_anchors_guessed": n_guessed,
        "n_zero_scale": n_no_scale,
        "created": datetime.now(timezone.utc).isoformat(),
        "components": components,
    }
    cache.write_text(json.dumps(stats))
    print(
        f"[{criterion}-stats] {len(components)} components, {n_examples} examples, "
        f"{n_guessed} ({n_guessed / max(n_examples, 1):.2%}) anchors guessed"
        + (f", {n_no_scale} with no positive activation" if n_no_scale else ""),
        flush=True,
    )
    return stats


class ThresholdedHarvestDB(HarvestDB):
    """A HarvestDB read as if the harvest had run at `threshold` under `criterion`."""

    def __init__(self, db_path: Path, threshold: float, stats: dict, readonly: bool = False):
        super().__init__(db_path, readonly)
        self.threshold = threshold
        self.criterion: Criterion = stats.get("criterion", "ci")
        self._name = f"{threshold:g}"
        self._half = int(stats["half"])
        self._stats: dict[str, dict] = stats["components"]

    @override
    def get_component(self, component_key: str) -> ComponentData | None:
        comp = super().get_component(component_key)
        if comp is None:
            return None
        held = self._stats.get(component_key)
        scale = (held or {}).get("scale")
        if scale is None:
            scale = component_scale(comp, self.criterion)
        kept = []
        if scale > 0.0:
            for e in comp.activation_examples:
                if anchor_value(e, self.criterion, self._half)[0] <= self.threshold * scale:
                    continue
                kept.append(
                    ActivationExample(
                        token_ids=e.token_ids,
                        firings=surviving_mask(e, self.criterion, self.threshold, scale),
                        activations=e.activations,
                    )
                )
        return comp.model_copy(
            update={
                "activation_examples": kept,
                "firing_density": held[self._name][1] if held else comp.firing_density,
            }
        )

    @override
    def get_component_densities(self, min_examples: int) -> list[tuple[str, float]]:
        return [
            (key, held[self._name][1])
            for key, held in self._stats.items()
            if held[self._name][0] >= min_examples
        ]
