"""Compare paired baseline/treatment training and held-out artifacts without GPU work."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--treatment-run", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--treatment-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_step = {int(row["step"]): row for row in rows}
    if sorted(by_step) != list(range(1, 1222)):
        raise ValueError(f"pilot metrics are incomplete: {path}")
    return [by_step[step] for step in range(1, 1222)]


def _paired_bootstrap(values: list[float], *, seed: int = 20261106) -> dict[str, float | int]:
    if not values:
        raise ValueError("paired bootstrap requires document effects")
    rng = random.Random(seed)
    count = len(values)
    draws = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(5000)
    )
    return {
        "documents": count,
        "mean_treatment_minus_baseline": sum(values) / count,
        "lower_95": draws[124],
        "upper_95": draws[4874],
        "seed": seed,
        "replicates": 5000,
    }


def _site_document_effects(
    baseline: dict[str, Any], treatment: dict[str, Any], model: str, site: str
) -> dict[str, Any]:
    baseline_layers = {
        key: value
        for key, value in baseline["metrics"].items()
        if key.startswith(f"activation/{model}/{site}/") and key.endswith("/prefix_5")
    }
    treatment_layers = {
        key: value
        for key, value in treatment["metrics"].items()
        if key.startswith(f"activation/{model}/{site}/") and key.endswith("/prefix_5")
    }
    if set(baseline_layers) != set(treatment_layers) or len(baseline_layers) != 8:
        raise ValueError(f"held-out layer coverage mismatch for {model}/{site}")
    common_documents: set[str] | None = None
    for key in baseline_layers:
        left = set(baseline_layers[key]["document_fvu_by_id"])
        right = set(treatment_layers[key]["document_fvu_by_id"])
        if left != right:
            raise ValueError(f"held-out document coverage differs at {key}")
        common_documents = left if common_documents is None else common_documents & left
    assert common_documents is not None
    effects = []
    for document in sorted(common_documents):
        left = sum(value["document_fvu_by_id"][document] for value in baseline_layers.values()) / 8
        right = (
            sum(value["document_fvu_by_id"][document] for value in treatment_layers.values()) / 8
        )
        effects.append(right - left)
    return _paired_bootstrap(effects)


def _aggregate_internal(payload: dict[str, Any], model: str) -> float:
    rows = [
        value for key, value in payload["metrics"].items() if key.startswith(f"internal/{model}.")
    ]
    return sum(row["sse"] for row in rows) / sum(row["centered_target_ss"] for row in rows)


def main() -> None:
    args = _args()
    baseline_rows = _metrics(args.baseline_run / "metrics.jsonl")
    treatment_rows = _metrics(args.treatment_run / "metrics.jsonl")
    batch_mismatch = [
        step
        for step, (left, right) in enumerate(zip(baseline_rows, treatment_rows, strict=True), 1)
        if left["ordered_anchor_ids_sha256"] != right["ordered_anchor_ids_sha256"]
    ]
    if batch_mismatch:
        raise ValueError(f"paired training batches differ at steps {batch_mismatch[:10]}")
    baseline_manifest = _json(args.baseline_run / "run_manifest.json")
    treatment_manifest = _json(args.treatment_run / "run_manifest.json")
    if (
        baseline_manifest["initial_common_state_by_rank"]
        != treatment_manifest["initial_common_state_by_rank"]
    ):
        raise ValueError("baseline/treatment common initialization hashes differ")
    baseline = _json(args.baseline_evaluation / "heldout_reconstruction.json")
    treatment = _json(args.treatment_evaluation / "heldout_reconstruction.json")
    baseline_dictionary = _json(args.baseline_evaluation / "dictionary_health.json")
    treatment_dictionary = _json(args.treatment_evaluation / "dictionary_health.json")
    paired = {
        model: {
            site: _site_document_effects(baseline, treatment, model, site)
            for site in ("pre", "mid")
        }
        for model in ("gemma", "smol")
    }
    full_fvu = {
        arm: {
            model: {
                site: payload["activation_summary"][model][site]["full_reconstruction"]["fvu"]
                for site in ("pre", "mid")
            }
            for model in ("gemma", "smol")
        }
        for arm, payload in (("baseline", baseline), ("treatment", treatment))
    }
    internal = {
        arm: {model: _aggregate_internal(payload, model) for model in ("gemma", "smol")}
        for arm, payload in (("baseline", baseline), ("treatment", treatment))
    }
    dictionary = {
        arm: {
            model: {
                site: {
                    key: source[model][site][key]
                    for key in (
                        "used_features",
                        "effective_dictionary_size_entropy",
                        "top_10_percent_event_share",
                    )
                }
                for site in ("pre", "mid")
            }
            for model in ("gemma", "smol")
        }
        for arm, source in (
            ("baseline", baseline_dictionary),
            ("treatment", treatment_dictionary),
        )
    }
    gemma_direction_and_uncertainty = all(
        paired["gemma"][site]["upper_95"] < 0 for site in ("pre", "mid")
    )
    payload = {
        "schema_version": 1,
        "status": "PAIRED_SCALE_PILOT_COMPLETE_RESEARCHER_DECISION_REQUIRED",
        "batch_identity": {
            "status": "PASS",
            "steps": 1221,
            "ordered_anchor_digest_mismatches": 0,
        },
        "common_initialization": "PASS",
        "full_activation_fvu": full_fvu,
        "paired_document_bootstrap": paired,
        "aggregate_internal_fvu": internal,
        "dictionary_health": dictionary,
        "learned_scales": _json(args.treatment_evaluation / "experiment_manifest.json")[
            "reconstruction_scales_by_rank"
        ],
        "scale_hypothesis_direction_and_uncertainty_gate": (
            "PASS" if gemma_direction_and_uncertainty else "FAIL"
        ),
        "decision_boundary": (
            "This report does not silently invent a material-regression threshold. "
            "Promotion additionally requires researcher review of SmolLM2, internal FVU, "
            "and dictionary-health changes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
