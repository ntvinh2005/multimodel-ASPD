"""Matched held-out comparison for the ASPD v3 gate-collapse recovery pilot."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

MODELS = ("gemma", "smol")
SITES = ("pre", "mid")
PRE_MATRICES = {"q", "k", "v"}
ACTIVATION_KEY = re.compile(
    r"^activation_normalized/(?P<model>gemma|smol)/(?P<site>pre|mid)/L\d+/prefix_5$"
)
INTERNAL_KEY = re.compile(r"^internal/(?P<model>gemma|smol)\.L\d+\.(?P<matrix>.+)$")


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return sum(materialized) / len(materialized)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_document_summary(
    control_metrics: Mapping[str, Any],
    treatment_metrics: Mapping[str, Any],
    keys: list[str],
    *,
    site_balanced: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not keys:
        raise ValueError("paired comparison selected no metrics")
    document_sets: list[set[str]] = []
    for key in keys:
        for metrics in (control_metrics, treatment_metrics):
            documents = metrics[key].get("document_fvu_by_id")
            if not isinstance(documents, dict):
                raise ValueError(f"metric lacks document_fvu_by_id: {key}")
            document_sets.append(set(documents))
    document_ids = sorted(set.intersection(*document_sets))
    if not document_ids:
        raise ValueError("paired comparison has no common documents")

    keys_by_site = {
        site: [
            key
            for key in keys
            if (
                (match := ACTIVATION_KEY.match(key)) is not None
                and match.group("site") == site
            )
            or (
                (internal := INTERNAL_KEY.match(key)) is not None
                and ((internal.group("matrix") in PRE_MATRICES) == (site == "pre"))
            )
        ]
        for site in SITES
    }
    if site_balanced and any(not keys_by_site[site] for site in SITES):
        raise ValueError("site-balanced comparison requires PRE and MID metrics")

    deltas: list[float] = []
    for document_id in document_ids:
        if site_balanced:
            site_deltas = []
            for site in SITES:
                site_deltas.append(
                    _mean(
                        treatment_metrics[key]["document_fvu_by_id"][document_id]
                        - control_metrics[key]["document_fvu_by_id"][document_id]
                        for key in keys_by_site[site]
                    )
                )
            deltas.append(_mean(site_deltas))
        else:
            deltas.append(
                _mean(
                    treatment_metrics[key]["document_fvu_by_id"][document_id]
                    - control_metrics[key]["document_fvu_by_id"][document_id]
                    for key in keys
                )
            )

    generator = random.Random(seed)
    bootstrapped = [
        _mean(deltas[generator.randrange(len(deltas))] for _ in deltas)
        for _ in range(bootstrap_samples)
    ]
    return {
        "unit": "source_document",
        "paired_documents": len(document_ids),
        "metric_count": len(keys),
        "treatment_minus_control_macro_fvu": _mean(deltas),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_95_percentile_interval": [
            _quantile(bootstrapped, 0.025),
            _quantile(bootstrapped, 0.975),
        ],
    }


def compare(
    control_dir: Path,
    treatment_dir: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    control_reconstruction = _load(control_dir / "heldout_reconstruction.json")
    treatment_reconstruction = _load(treatment_dir / "heldout_reconstruction.json")
    control_dictionary = _load(control_dir / "dictionary_health.json")
    treatment_dictionary = _load(treatment_dir / "dictionary_health.json")
    control_metrics = control_reconstruction["metrics"]
    treatment_metrics = treatment_reconstruction["metrics"]
    if set(control_metrics) != set(treatment_metrics):
        raise ValueError("control and treatment reconstruction metric identities differ")

    activation: dict[str, Any] = {}
    activation_deltas = []
    for model in MODELS:
        activation[model] = {}
        for site in SITES:
            control_fvu = control_reconstruction["activation_summary"][model]["normalized"][site][
                "full_reconstruction"
            ]["fvu"]
            treatment_fvu = treatment_reconstruction["activation_summary"][model]["normalized"][
                site
            ]["full_reconstruction"]["fvu"]
            delta = treatment_fvu - control_fvu
            activation[model][site] = {
                "control_fvu": control_fvu,
                "treatment_fvu": treatment_fvu,
                "treatment_minus_control": delta,
            }
            activation_deltas.append(delta)

    internal: dict[str, Any] = {}
    internal_site_values: dict[str, dict[str, list[float]]] = {
        "control": {site: [] for site in SITES},
        "treatment": {site: [] for site in SITES},
    }
    internal_keys = []
    activation_keys = []
    for key in sorted(control_metrics):
        if ACTIVATION_KEY.match(key):
            activation_keys.append(key)
        match = INTERNAL_KEY.match(key)
        if match is None:
            continue
        internal_keys.append(key)
        site = "pre" if match.group("matrix") in PRE_MATRICES else "mid"
        internal_site_values["control"][site].append(control_metrics[key]["fvu"])
        internal_site_values["treatment"][site].append(treatment_metrics[key]["fvu"])
    for site in SITES:
        control_fvu = _mean(internal_site_values["control"][site])
        treatment_fvu = _mean(internal_site_values["treatment"][site])
        internal[site] = {
            "control_fvu": control_fvu,
            "treatment_fvu": treatment_fvu,
            "treatment_minus_control": treatment_fvu - control_fvu,
        }
    control_internal = _mean(internal[site]["control_fvu"] for site in SITES)
    treatment_internal = _mean(internal[site]["treatment_fvu"] for site in SITES)
    internal["equal_site_mean"] = {
        "control_fvu": control_internal,
        "treatment_fvu": treatment_internal,
        "treatment_minus_control": treatment_internal - control_internal,
    }

    dictionary: dict[str, Any] = {}
    for model in MODELS:
        dictionary[model] = {}
        for site in SITES:
            control = control_dictionary[model][site]
            treatment = treatment_dictionary[model][site]
            dictionary[model][site] = {
                name: {"control": control[name], "treatment": treatment[name]}
                for name in (
                    "positive_candidates_per_anchor_mean",
                    "mean_active_features_per_anchor",
                    "effective_dictionary_size_entropy",
                    "used_features",
                    "never_fired_features",
                    "top_1_percent_event_share",
                    "top_10_percent_event_share",
                    "training_recently_dead_features",
                )
            }

    promotion = {
        "each_activation_delta_lte_0_03": all(delta <= 0.03 for delta in activation_deltas),
        "equal_model_site_activation_delta_lte_0_02": _mean(activation_deltas) <= 0.02,
        "equal_site_internal_delta_lte_0_02": internal["equal_site_mean"][
            "treatment_minus_control"
        ]
        <= 0.02,
        "heldout_used_features_gte_1000": all(
            treatment_dictionary[model][site]["used_features"] >= 1000
            for model in MODELS
            for site in SITES
        ),
    }
    promotion["heldout_reconstruction_and_coverage_pass"] = all(promotion.values())

    return {
        "schema_version": 1,
        "comparison": "per_anchor_zscore_minus_full_only_control",
        "control_dir": str(control_dir),
        "treatment_dir": str(treatment_dir),
        "activation_normalized_full_fvu": activation,
        "activation_equal_model_site_delta": _mean(activation_deltas),
        "internal_fvu": internal,
        "dictionary": dictionary,
        "paired_document_macro_fvu": {
            "activation": _paired_document_summary(
                control_metrics,
                treatment_metrics,
                activation_keys,
                site_balanced=False,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
            "internal_equal_site": _paired_document_summary(
                control_metrics,
                treatment_metrics,
                internal_keys,
                site_balanced=True,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
        },
        "promotion_checks": promotion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--treatment-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                args.control_dir,
                args.treatment_dir,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
