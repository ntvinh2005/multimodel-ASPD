"""Evaluate frozen source models and the final ASPD v3.1 encoder on induction prompts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from aspd.multimodel.posthoc.artifacts import require_complete, sha256_file
from aspd.multimodel.posthoc.ids import ComponentId, GateId, MatrixId
from aspd.multimodel.posthoc.importance import ImportanceTable
from aspd.multimodel.posthoc.source_hooks import capture_matrix_io
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend, _pad_id
from aspd.multimodel.v3.induction import (
    MODEL_ORDER,
    behavioral_metrics,
    merge_task_and_background,
    task_pool_from_verified,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "discovery"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


class _JsonlWriter:
    def __init__(self, path: Path):
        self.handle = path.open("x", encoding="utf-8")
        self.count = 0

    def write(self, payload: Any) -> None:
        self.handle.write(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        )
        self.count += 1

    def close(self) -> None:
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_preparation(path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = path / "manifest.json"
    expected = (path / "PREPARED").read_text(encoding="utf-8").strip()
    if sha256_file(manifest_path) != expected:
        raise ValueError("preparation manifest changed after PREPARED")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, digest in manifest["artifacts"].items():
        if sha256_file(path / relative) != digest:
            raise ValueError(f"prepared artifact changed: {relative}")
    # Deliberately never open heldout_sealed in this executable.
    records = []
    with (path / "smoke_discovery.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["example"]["split"] == split:
                records.append(row)
    expected_examples = 128 if split == "smoke" else 1024
    if len(records) != expected_examples:
        raise ValueError(f"{split} preparation has {len(records)} examples, expected {expected_examples}")
    return records, manifest


def _native_rows(
    backend: V31CheckpointBackend,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for model_key in MODEL_ORDER:
        model = backend.source_model(model_key)
        tokenizer = backend.tokenizers[model_key]
        pad = _pad_id(tokenizer)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            maximum = max(len(row["models"][model_key]["input_ids"]) for row in batch)
            ids = torch.full(
                (len(batch), maximum), pad, dtype=torch.long, device=backend.device
            )
            mask = torch.zeros_like(ids)
            positions = []
            for index, row in enumerate(batch):
                model_record = row["models"][model_key]
                sequence = torch.tensor(
                    model_record["input_ids"], dtype=torch.long, device=backend.device
                )
                ids[index, : sequence.numel()] = sequence
                mask[index, : sequence.numel()] = torch.tensor(
                    model_record["attention_mask"], dtype=torch.long, device=backend.device
                )
                positions.append(int(model_record["prediction_token_index"]))
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask, use_cache=False)
                logits = output.logits[
                    torch.arange(len(batch), device=backend.device),
                    torch.tensor(positions, device=backend.device),
                ].float()
            probabilities = torch.softmax(logits, dim=-1)
            top = torch.topk(logits, k=top_k, dim=-1)
            predicted = top.indices[:, 0]
            for index, record in enumerate(batch):
                example = record["example"]
                model_record = record["models"][model_key]
                expected = int(model_record["expected_token_id"])
                distractor = int(model_record["distractor_token_id"])
                rows.append(
                    {
                        "model": model_key,
                        "example_id": example["example_id"],
                        "group_id": example["group_id"],
                        "split": example["split"],
                        "family": example["family"],
                        "condition": example["condition"],
                        "prompt": example["prompt"],
                        "expected_continuation": example["expected_continuation"],
                        "distractor_continuation": example["distractor_continuation"],
                        "expected_is_ground_truth": example["expected_is_ground_truth"],
                        "prediction_token_index": positions[index],
                        "expected_token_id": expected,
                        "distractor_token_id": distractor,
                        "predicted_token_id": int(predicted[index]),
                        "predicted_token": tokenizer.decode(
                            [int(predicted[index])], clean_up_tokenization_spaces=False
                        ),
                        "expected_probability": float(probabilities[index, expected]),
                        "distractor_probability": float(probabilities[index, distractor]),
                        "logit_margin": float(logits[index, expected] - logits[index, distractor]),
                        "top_predictions": [
                            {
                                "token_id": int(token_id),
                                "token": tokenizer.decode(
                                    [int(token_id)], clean_up_tokenization_spaces=False
                                ),
                                "logit": float(value),
                                "probability": float(probabilities[index, int(token_id)]),
                            }
                            for value, token_id in zip(
                                top.values[index].tolist(), top.indices[index].tolist(), strict=True
                            )
                        ],
                        "repetition_distance_tokens": example["metadata"].get(
                            "repetition_distance_tokens"
                        ),
                    }
                )
            del output, logits, probabilities, top, ids, mask
    return rows


def _paired_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["model"], row["group_id"], row["condition"]): row for row in rows}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    models_groups = sorted({(row["model"], row["group_id"]) for row in rows})
    for model, group_id in models_groups:
        clean = by_key[(model, group_id, "induction")]
        no_match = by_key[(model, group_id, "no_match")]
        counter = by_key[(model, group_id, "counterfactual")]
        other = by_key[(model, group_id, "repeated_no_association")]
        groups[(model, clean["family"])].append(
            {
                "clean_minus_no_match": clean["logit_margin"] - no_match["logit_margin"],
                "clean_minus_other_control": clean["logit_margin"] - other["logit_margin"],
                "clean_pairwise": clean["logit_margin"] > 0,
                "counterfactual_pairwise": counter["logit_margin"] > 0,
                "pair_reversal": clean["logit_margin"] > 0 and counter["logit_margin"] > 0,
            }
        )
    result = {}
    for key, values in sorted(groups.items()):
        model, family = key
        result[f"{model}/{family}"] = {
            "groups": len(values),
            "clean_pairwise_accuracy": sum(v["clean_pairwise"] for v in values) / len(values),
            "counterfactual_pairwise_accuracy": sum(v["counterfactual_pairwise"] for v in values)
            / len(values),
            "paired_reversal_rate": sum(v["pair_reversal"] for v in values) / len(values),
            "mean_clean_minus_no_match_margin": sum(v["clean_minus_no_match"] for v in values)
            / len(values),
            "mean_clean_minus_other_control_margin": sum(
                v["clean_minus_other_control"] for v in values
            )
            / len(values),
        }
    return result


def _load_references(root: Path):
    importance = ImportanceTable.load(require_complete(root, "importance"))
    representatives_path = require_complete(root, "representatives") / "representatives.json"
    payload = json.loads(representatives_path.read_text(encoding="utf-8"))
    if tuple(payload["model_order"]) != MODEL_ORDER:
        raise ValueError("reference representative model order differs")
    representatives = {
        (GateId.from_key(gate), model): tuple(ComponentId.from_key(key) for key in components)
        for gate, by_model in payload["representatives"].items()
        for model, components in by_model.items()
    }
    return importance, representatives


def _anchor_metadata(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for record in records:
        example = record["example"]
        spans = example["spans"]
        required = set(int(value) for value in record["required_boundaries"])
        for anchor in record["alignment"]["anchors"]:
            boundary = int(anchor["character_boundary"])
            result[anchor["anchor_id"]] = {
                "example_id": example["example_id"],
                "group_id": example["group_id"],
                "family": example["family"],
                "condition": example["condition"],
                "character_boundary": boundary,
                "roles_ending": [span["role"] for span in spans if int(span["stop"]) == boundary],
                "annotated": boundary in required,
                "final_query": anchor["anchor_id"] == record["final_anchor_id"],
                "model_positions": {
                    key: int(anchor["positions"][key]["token_index"]) for key in MODEL_ORDER
                },
            }
    return result


def _component_rows(
    *,
    backend: V31CheckpointBackend,
    encoded: Any,
    anchor_meta: dict[str, dict[str, Any]],
    importance: ImportanceTable,
    representatives: dict[tuple[GateId, str], tuple[ComponentId, ...]],
    pool_id: str,
    writer: _JsonlWriter,
) -> dict[tuple[str, str, int], list[float]]:
    aggregate: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    selected_rows = [
        row for row, anchor_id in enumerate(encoded.pool.anchor_ids)
        if anchor_id in anchor_meta and anchor_meta[anchor_id]["annotated"]
    ]
    if not selected_rows:
        return aggregate
    for model in MODEL_ORDER:
        inputs = encoded.pool.models[model].to(backend.device)
        captured = capture_matrix_io(
            backend.source_model(model), inputs, backend.hook_specs(model)
        )
        requests: dict[MatrixId, list[tuple[int, int, int]]] = defaultdict(list)
        for row in selected_rows:
            for site in ("pre", "mid"):
                active = encoded.pool.gates[site][row].nonzero(as_tuple=True)[0].tolist()
                for feature in active:
                    gate = GateId(site, int(feature))
                    for rank, component in enumerate(representatives.get((gate, model), ())):
                        requests[component.matrix].append((row, int(feature), rank))
        factors = backend.factors(model)
        for matrix, triples in requests.items():
            rows = torch.tensor([value[0] for value in triples], device=backend.device)
            features = torch.tensor([value[1] for value in triples], device=backend.device)
            x = captured[matrix][0][rows].float()
            U, V = factors[matrix]
            coefficients = (x * V[features]).sum(dim=-1)
            norms = coefficients.abs() * torch.linalg.vector_norm(U[features], dim=-1)
            sigma = float(importance[matrix].sigma)
            normalized = norms / sigma if sigma > 0 else torch.full_like(norms, torch.nan)
            for local, (row, feature, rank) in enumerate(triples):
                anchor_id = encoded.pool.anchor_ids[row]
                meta = anchor_meta[anchor_id]
                norm = float(norms[local])
                normalized_value: float | None = float(normalized[local])
                if not math.isfinite(normalized_value):
                    normalized_value = None
                beta: float | None = float(importance[matrix].beta[feature])
                if not math.isfinite(beta):
                    beta = None
                writer.write(
                    {
                        "pool_id": pool_id,
                        "anchor_id": anchor_id,
                        **meta,
                        "model": model,
                        "site": matrix.site,
                        "feature": feature,
                        "component": f"{matrix.key}/{feature}",
                        "matrix": matrix.key,
                        "representative_rank": rank,
                        "coefficient": float(coefficients[local]),
                        "contribution_norm": norm,
                        "matrix_output_sigma": sigma,
                        "normalized_contribution_magnitude": normalized_value,
                        "reference_beta": beta,
                        "reference_beta_fire_count": int(
                            importance[matrix].fire_count[feature]
                        ),
                    }
                )
                aggregate[(model, matrix.site, feature)].append(norm)
        del captured
    return aggregate


def _gate_summary(
    examples: dict[str, dict[str, Any]],
    contribution: dict[tuple[str, str, int], list[float]],
    importance: ImportanceTable,
    representatives: dict[tuple[GateId, str], tuple[ComponentId, ...]],
) -> list[dict[str, Any]]:
    context_sets: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    active_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    relevant_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    condition_counts: dict[str, int] = defaultdict(int)
    final_sums: dict[tuple[str, int, str], float] = defaultdict(float)
    for example_id, example in examples.items():
        condition = example["condition"]
        condition_counts[condition] += 1
        sequence_seen: dict[str, set[int]] = {"pre": set(), "mid": set()}
        for position in example["positions"]:
            for site in ("pre", "mid"):
                activations = dict(
                    zip(position[site]["active_gate_ids"], position[site]["values"], strict=True)
                )
                sequence_seen[site].update(int(feature) for feature in activations)
                for feature, value in activations.items():
                    active_values[(site, int(feature))].append(float(value))
                    if position["annotated"]:
                        relevant_values[(site, int(feature))].append(float(value))
                if position["final_query"]:
                    for feature, value in activations.items():
                        final_sums[(site, int(feature), condition)] += float(value)
        for site, features in sequence_seen.items():
            for feature in features:
                context_sets[(site, feature, condition)].add(example_id)
    keys = sorted({(site, feature) for site, feature, _condition in context_sets})
    rows = []
    for site, feature in keys:
        positive_total = final_sums[(site, feature, "induction")] + final_sums[
            (site, feature, "counterfactual")
        ]
        positive_n = condition_counts["induction"] + condition_counts["counterfactual"]
        control_total = final_sums[(site, feature, "no_match")] + final_sums[
            (site, feature, "repeated_no_association")
        ]
        control_n = condition_counts["no_match"] + condition_counts["repeated_no_association"]
        positive_mean = positive_total / max(1, positive_n)
        control_mean = control_total / max(1, control_n)
        row: dict[str, Any] = {
            "site": site,
            "feature": feature,
            "induction_contexts_fired": len(context_sets.get((site, feature, "induction"), set())),
            "counterfactual_contexts_fired": len(
                context_sets.get((site, feature, "counterfactual"), set())
            ),
            "no_match_contexts_fired": len(context_sets.get((site, feature, "no_match"), set())),
            "other_control_contexts_fired": len(
                context_sets.get((site, feature, "repeated_no_association"), set())
            ),
            "distinct_firing_contexts": len(
                set().union(
                    *(context_sets.get((site, feature, condition), set())
                      for condition in ("induction", "counterfactual", "no_match",
                                        "repeated_no_association"))
                )
            ),
            "mean_continuous_activation": sum(active_values[(site, feature)])
            / len(active_values[(site, feature)]),
            "mean_activation_at_annotated_positions": (
                sum(relevant_values[(site, feature)]) / len(relevant_values[(site, feature)])
                if relevant_values[(site, feature)] else 0.0
            ),
            "induction_minus_control_final_activation": positive_mean - control_mean,
        }
        for model in MODEL_ORDER:
            values = contribution.get((model, site, feature), [])
            components = representatives.get((GateId(site, feature), model), ())
            beta_pairs = [
                (component.key, float(importance[component.matrix].beta[feature]))
                for component in components
                if math.isfinite(float(importance[component.matrix].beta[feature]))
            ]
            row[f"{model}_representative_components"] = json.dumps(
                [key for key, _value in beta_pairs], separators=(",", ":")
            )
            row[f"{model}_max_reference_beta"] = max(
                (value for _key, value in beta_pairs), default=0.0
            )
            row[f"{model}_mean_reference_beta"] = (
                sum(value for _key, value in beta_pairs) / len(beta_pairs) if beta_pairs else 0.0
            )
            row[f"{model}_representative_contribution_n"] = len(values)
            row[f"{model}_mean_representative_contribution_norm"] = (
                sum(values) / len(values) if values else 0.0
            )
            row[f"{model}_max_representative_contribution_norm"] = max(values, default=0.0)
        rows.append(row)
    return rows


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {args.output_dir}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records, preparation_manifest = _load_preparation(args.preparation, args.split)
    if preparation_manifest["checkpoint_manifest_sha256"] != config["checkpoint_manifest_sha256"]:
        raise ValueError("preparation/evaluation checkpoint provenance differs")
    reference_root = Path(config["components"]["reference_posthoc_root"])
    importance, representatives = _load_references(reference_root)
    options = {
        "training_config": config["training_config"],
        "checkpoint": config["checkpoint"],
        "config_file_sha256": config["config_file_sha256"],
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "split": "validation",
        "source_dtype": config["source_dtype"],
        "device": config["device"],
        "local_files_only": config["local_files_only"],
        "pool_anchors": config["batch_topk"]["pool_anchors"],
        "skip_training_launch_evidence": config["skip_training_launch_evidence"],
    }
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    backend = V31CheckpointBackend(options, work_dir=args.output_dir / "stream_state")
    try:
        native = _native_rows(
            backend, records, batch_size=int(config["behavior"]["batch_size"]),
            top_k=int(config["behavior"]["top_predictions"]),
        )
        with _JsonlWriter(args.output_dir / "native_predictions.jsonl") as writer:
            for row in native:
                writer.write(row)
        behavior_summary = {
            "by_model_family_condition": behavioral_metrics(native),
            "paired": _paired_behavior(native),
        }
        _write_json(args.output_dir / "behavior_summary.json", behavior_summary)
        predictions = defaultdict(dict)
        for row in native:
            predictions[row["example_id"]][row["model"]] = {
                key: row[key]
                for key in (
                    "predicted_token_id", "predicted_token", "expected_probability",
                    "distractor_probability", "logit_margin", "top_predictions",
                )
            }

        plan = json.loads((args.preparation / "pool_plan.json").read_text(encoding="utf-8"))
        by_id = {row["example"]["example_id"]: row for row in records}
        anchor_meta = _anchor_metadata(records)
        example_outputs: dict[str, dict[str, Any]] = {}
        contribution_aggregate: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        pool_manifests = []
        activation_writer = _JsonlWriter(args.output_dir / "anchor_activations.jsonl")
        component_writer = _JsonlWriter(args.output_dir / "component_contributions.jsonl")
        try:
            backgrounds = backend.aligned_pools(len(plan["splits"][args.split]))
            for pool_spec, background in zip(plan["splits"][args.split], backgrounds, strict=True):
                pool_records = [by_id[value] for value in pool_spec["example_ids"]]
                task = task_pool_from_verified(pool_records)
                if task.anchor_count != int(pool_spec["task_anchor_count"]):
                    raise ValueError("prepared task anchor count changed")
                aligned = merge_task_and_background(
                    task, background, pool_anchors=int(config["batch_topk"]["pool_anchors"])
                )
                pool_id = f"{args.split}-{int(pool_spec['pool_index']):04d}"
                encoded = backend.encode_aligned_pool(aligned)
                task_rows = range(task.anchor_count)
                for row in task_rows:
                    anchor_id = encoded.pool.anchor_ids[row]
                    meta = anchor_meta[anchor_id]
                    site_payload = {}
                    for site in ("pre", "mid"):
                        active = encoded.pool.gates[site][row].nonzero(as_tuple=True)[0]
                        values = encoded.continuous[site][row, active]
                        site_payload[site] = {
                            "active_gate_ids": [int(value) for value in active.tolist()],
                            "values": [float(value) for value in values.tolist()],
                        }
                    activation_writer.write(
                        {"pool_id": pool_id, "anchor_id": anchor_id, **meta, **site_payload}
                    )
                    example = example_outputs.setdefault(
                        meta["example_id"],
                        {
                            "example_id": meta["example_id"],
                            "group_id": meta["group_id"],
                            "family": meta["family"],
                            "condition": meta["condition"],
                            "prompt": by_id[meta["example_id"]]["example"]["prompt"],
                            "expected_continuation": by_id[meta["example_id"]]["example"][
                                "expected_continuation"
                            ],
                            "distractor_continuation": by_id[meta["example_id"]]["example"][
                                "distractor_continuation"
                            ],
                            "spans": by_id[meta["example_id"]]["example"]["spans"],
                            "native_predictions": predictions[meta["example_id"]],
                            "pool_id": pool_id,
                            "positions": [],
                        },
                    )
                    example["positions"].append(
                        {"anchor_id": anchor_id, **meta, **site_payload}
                    )
                pool_contributions = _component_rows(
                    backend=backend, encoded=encoded, anchor_meta=anchor_meta,
                    importance=importance, representatives=representatives,
                    pool_id=pool_id, writer=component_writer,
                )
                for key, values in pool_contributions.items():
                    contribution_aggregate[key].extend(values)
                pool_manifests.append(
                    {
                        "pool_id": pool_id,
                        "task_anchor_count": task.anchor_count,
                        "background_anchor_count": aligned.anchor_count - task.anchor_count,
                        "total_anchor_count": aligned.anchor_count,
                        "group_ids": pool_spec["group_ids"],
                        "pre_selected": int(encoded.pool.gates["pre"].sum()),
                        "mid_selected": int(encoded.pool.gates["mid"].sum()),
                    }
                )
                del encoded, aligned, task
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            activation_writer.close()
            component_writer.close()

        with _JsonlWriter(args.output_dir / "activation_examples.jsonl") as writer:
            for example_id in sorted(example_outputs):
                writer.write(example_outputs[example_id])
        gate_rows = _gate_summary(
            example_outputs, contribution_aggregate, importance, representatives
        )
        with (args.output_dir / "gate_summary.csv").open("x", newline="", encoding="utf-8") as handle:
            fieldnames = list(gate_rows[0]) if gate_rows else ["site", "feature"]
            csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(gate_rows)
        _write_json(args.output_dir / "pools.json", pool_manifests)
    finally:
        backend_provenance = backend.provenance()
        backend.close()

    files = sorted(
        path for path in args.output_dir.iterdir()
        if path.is_file() and path.name not in ("manifest.json", "COMPLETE")
    )
    runtime = {
        "wall_seconds": time.perf_counter() - started,
        "cuda_max_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "artifact_bytes_before_manifest": sum(path.stat().st_size for path in files),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "split": args.split,
        "scientific_status": "MEASURED_EXPLORATORY_NO_UNIVERSAL_MECHANISM_CLAIM",
        "git_revision": _git_sha(),
        "git_tracked_changes": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
        ),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "preparation_manifest_sha256": sha256_file(args.preparation / "manifest.json"),
        "backend": backend_provenance,
        "reference_posthoc_root": str(reference_root),
        "reference_importance_complete_sha256": sha256_file(reference_root / "importance/COMPLETE"),
        "reference_representatives_complete_sha256": sha256_file(
            reference_root / "representatives/COMPLETE"
        ),
        "component_scope": config["components"]["sparse_scope"],
        "dependency_edges_computed": False,
        "heldout_opened": False,
        "runtime": runtime,
        "environment": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "artifacts": {path.name: sha256_file(path) for path in files},
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    _write_json(
        args.output_dir / "COMPLETE",
        {
            "status": "COMPLETE",
            "manifest_sha256": sha256_file(args.output_dir / "manifest.json"),
            "artifact_count": len(files),
        },
    )


if __name__ == "__main__":
    main()
