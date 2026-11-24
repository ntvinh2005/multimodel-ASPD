#!/usr/bin/env python3
"""Export measured induction shared-gate results into a bounded frontend bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aspd.multimodel.posthoc.artifacts import require_complete, sha256_file
from aspd.multimodel.posthoc.ids import ComponentId, GateId
from aspd.multimodel.posthoc.importance import ImportanceTable

MODELS = ("gemma", "smol", "granite")
MATRIX_LABEL = {
    "q": "Q", "k": "K", "v": "V", "o": "O",
    "mlp_gate": "gate", "mlp_up": "up", "mlp_down": "down",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _verify_evaluation(root: Path) -> dict[str, Any]:
    complete = _json(root / "COMPLETE")
    manifest_path = root / "manifest.json"
    if complete["status"] != "COMPLETE" or complete["manifest_sha256"] != sha256_file(
        manifest_path
    ):
        raise ValueError(f"induction evaluation is not immutable COMPLETE: {root}")
    manifest = _json(manifest_path)
    for name, digest in manifest["artifacts"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"induction artifact changed: {root / name}")
    return manifest


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def _load_references(root: Path):
    importance = ImportanceTable.load(require_complete(root, "importance"))
    reps_payload = _json(require_complete(root, "representatives") / "representatives.json")
    representatives = {
        (GateId.from_key(gate), model): tuple(ComponentId.from_key(value) for value in values)
        for gate, by_model in reps_payload["representatives"].items()
        for model, values in by_model.items()
    }
    return importance, representatives


def _gate_activation(position: dict[str, Any], site: str, feature: int) -> float:
    active = position[site]["active_gate_ids"]
    try:
        index = active.index(feature)
    except ValueError:
        return 0.0
    return float(position[site]["values"][index])


def _component_payload(
    gate: GateId, importance: ImportanceTable,
    representatives: dict[tuple[GateId, str], tuple[ComponentId, ...]],
) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        selected = set(representatives.get((gate, model), ()))
        for matrix in importance.matrices(model=model, site=gate.site):
            beta = float(importance[matrix].beta[gate.feature])
            rows.append(
                {
                    "id": f"{matrix.key}/{gate.feature}",
                    "model": model,
                    "layer": matrix.layer,
                    "matrix": MATRIX_LABEL[matrix.matrix],
                    "beta": beta if math.isfinite(beta) else None,
                    "representative": ComponentId(matrix, gate.feature) in selected,
                }
            )
    return rows


def _split_payload(
    root: Path,
    manifest: dict[str, Any],
    importance: ImportanceTable,
    representatives: dict[tuple[GateId, str], tuple[ComponentId, ...]],
    *,
    context_gate_limit: int,
    groups_per_gate: int,
) -> dict[str, Any]:
    with (root / "gate_summary.csv").open(encoding="utf-8", newline="") as handle:
        summaries = list(csv.DictReader(handle))
    examples = list(_jsonl(root / "activation_examples.jsonl"))
    examples_by_group = defaultdict(list)
    for example in examples:
        examples_by_group[example["group_id"]].append(example)

    final_counts: dict[tuple[str, int, str], int] = defaultdict(int)
    final_strength: dict[tuple[str, int, str], float] = defaultdict(float)
    for example in examples:
        final = next(position for position in example["positions"] if position["final_query"])
        condition = example["condition"]
        for site in ("pre", "mid"):
            for feature, value in zip(
                final[site]["active_gate_ids"], final[site]["values"], strict=True
            ):
                final_counts[(site, int(feature), condition)] += 1
                final_strength[(site, int(feature), condition)] += float(value)

    gates = []
    for row in summaries:
        site = str(row["site"])
        feature = int(row["feature"])
        gate = GateId(site, feature)
        beta_by_model = {}
        for model in MODELS:
            values = [
                float(importance[component.matrix].beta[feature])
                for component in representatives.get((gate, model), ())
            ]
            values = [value for value in values if math.isfinite(value)]
            beta_by_model[model] = {
                "max": max(values, default=0.0),
                "mean": sum(values) / len(values) if values else 0.0,
            }
        gates.append(
            {
                "id": f"{site.upper()}:{feature}",
                "site": site.upper(),
                "feature": feature,
                "contrast": _float(row, "induction_minus_control_final_activation"),
                "meanActivation": _float(row, "mean_continuous_activation"),
                "meanAnnotatedActivation": _float(row, "mean_activation_at_annotated_positions"),
                "sequenceFires": {
                    "induction": int(row["induction_contexts_fired"]),
                    "counterfactual": int(row["counterfactual_contexts_fired"]),
                    "no_match": int(row["no_match_contexts_fired"]),
                    "repeated_no_association": int(row["other_control_contexts_fired"]),
                },
                "finalQueryFires": {
                    condition: final_counts[(site, feature, condition)]
                    for condition in (
                        "induction", "counterfactual", "no_match", "repeated_no_association"
                    )
                },
                "finalQueryActivationSum": {
                    condition: final_strength[(site, feature, condition)]
                    for condition in (
                        "induction", "counterfactual", "no_match", "repeated_no_association"
                    )
                },
                "betaByModel": beta_by_model,
            }
        )
    gates.sort(
        key=lambda row: (
            abs(row["contrast"]),
            row["finalQueryFires"]["induction"]
            + row["finalQueryFires"]["counterfactual"],
        ),
        reverse=True,
    )
    context_gates = [
        gate for gate in gates
        if gate["finalQueryFires"]["induction"] + gate["finalQueryFires"]["counterfactual"] > 0
    ][:context_gate_limit]
    contexts: dict[str, list[dict[str, Any]]] = {}
    components: dict[str, list[dict[str, Any]]] = {}
    for gate_row in context_gates:
        gate = GateId(gate_row["site"].lower(), gate_row["feature"])
        scored_groups = []
        for group_id, group in examples_by_group.items():
            positive = [
                example for example in group
                if example["condition"] in ("induction", "counterfactual")
            ]
            score = max(
                (
                    _gate_activation(
                        next(p for p in example["positions"] if p["final_query"]),
                        gate.site,
                        gate.feature,
                    )
                    for example in positive
                ),
                default=0.0,
            )
            if score > 0:
                scored_groups.append((score, group_id))
        scored_groups.sort(reverse=True)
        selected_group_ids = {value for _score, value in scored_groups[:groups_per_gate]}
        selected_examples = []
        for group_id in selected_group_ids:
            for example in sorted(
                examples_by_group[group_id],
                key=lambda value: (
                    "induction", "counterfactual", "no_match", "repeated_no_association"
                ).index(value["condition"]),
            ):
                selected_examples.append(
                    {
                        "exampleId": example["example_id"],
                        "groupId": group_id,
                        "family": example["family"],
                        "condition": example["condition"],
                        "prompt": example["prompt"],
                        "expectedContinuation": example["expected_continuation"],
                        "nativePredictions": example["native_predictions"],
                        "spans": example["spans"],
                        "positions": [
                            {
                                "boundary": int(position["character_boundary"]),
                                "activation": _gate_activation(
                                    position, gate.site, gate.feature
                                ),
                                "finalQuery": bool(position["final_query"]),
                                "roles": position["roles_ending"],
                            }
                            for position in example["positions"]
                        ],
                    }
                )
        contexts[gate_row["id"]] = selected_examples
        components[gate_row["id"]] = _component_payload(gate, importance, representatives)
    return {
        "status": "measured",
        "split": manifest["split"],
        "sourceRevision": manifest["git_revision"],
        "evaluationScope": manifest.get("evaluation_scope", "gates-and-components"),
        "artifactRoot": str(root),
        "manifestSha256": sha256_file(root / "manifest.json"),
        "groups": len({example["group_id"] for example in examples}),
        "examples": len(examples),
        "pools": len(_json(root / "pools.json")),
        "gates": gates,
        "contextGateIds": [row["id"] for row in context_gates],
        "contexts": contexts,
        "components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--discovery", type=Path)
    parser.add_argument("--reference-posthoc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-gate-limit", type=int, default=64)
    parser.add_argument("--groups-per-gate", type=int, default=3)
    args = parser.parse_args()

    importance, representatives = _load_references(args.reference_posthoc_root)
    smoke_manifest = _verify_evaluation(args.smoke)
    splits = {
        "smoke": _split_payload(
            args.smoke, smoke_manifest, importance, representatives,
            context_gate_limit=args.context_gate_limit, groups_per_gate=args.groups_per_gate,
        )
    }
    if args.discovery is not None:
        discovery_manifest = _verify_evaluation(args.discovery)
        splits["discovery"] = _split_payload(
            args.discovery, discovery_manifest, importance, representatives,
            context_gate_limit=args.context_gate_limit, groups_per_gate=args.groups_per_gate,
        )
    else:
        splits["discovery"] = {"status": "pending"}
    payload = {
        "schemaVersion": 1,
        "label": "ASPD v3.1 induction shared-gate evaluation",
        "checkpointStep": 61036,
        "checkpointPath": smoke_manifest["backend"]["checkpoint"],
        "scientificBoundary": (
            "Shared gate activation and imported beta are exploratory screening evidence; "
            "no induction circuit or universal mechanism is asserted."
        ),
        "referencePosthocRoot": str(args.reference_posthoc_root),
        "referenceImportanceCompleteSha256": sha256_file(
            args.reference_posthoc_root / "importance/COMPLETE"
        ),
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
