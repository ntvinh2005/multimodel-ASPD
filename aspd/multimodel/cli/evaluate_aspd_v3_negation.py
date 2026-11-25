"""Measure native negation behavior and frozen natural ASPD gates on smoke/discovery only."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from aspd.multimodel.cli.evaluate_aspd_v3_induction import (
    _anchor_metadata,
    _component_rows,
    _JsonlWriter,
    _load_references,
)
from aspd.multimodel.cli.prepare_aspd_v3_negation import read_contract
from aspd.multimodel.posthoc.artifacts import runtime_provenance, sha256_file, write_json
from aspd.multimodel.posthoc.ids import SITES, GateId
from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend, _pad_id
from aspd.multimodel.v3.induction import (
    MODEL_ORDER,
    merge_task_and_background,
    task_pool_from_verified,
)
from aspd.multimodel.v3.negation import (
    NEGATIVE_CONDITIONS,
    candidate_behavior,
    failure_class,
    validate_group,
)


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def verify_evaluation(root, split):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads((root / "COMPLETE").read_text())
    if (
        complete["manifest_sha256"] != sha256_file(manifest_path)
        or manifest["mechanism"] != "negation"
        or manifest["split"] != split
        or manifest["heldout_opened"]
    ):
        raise ValueError(f"{split} evaluation provenance mismatch")
    for name, digest in manifest["artifacts"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"{split} artifact changed: {name}")
    return manifest


def load_preparation(root, config, split):
    if split not in {"smoke", "discovery"}:
        raise ValueError("held-out evaluation is sealed")
    digest = sha256_file(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        (root / "PREPARED").read_text().strip() != digest
        or manifest["mechanism"] != "negation"
        or manifest["config_sha256"] != sha256_file(config)
    ):
        raise ValueError("prepared provenance changed")
    for name in ("smoke_discovery.jsonl", "pool_plan.json", "validation.json"):
        if sha256_file(root / name) != manifest["artifacts"][name]:
            raise ValueError(f"prepared file changed: {name}")
    rows = [
        row for row in jsonl(root / "smoke_discovery.jsonl") if row["example"]["split"] == split
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[row["example"]["group_id"]].append(row["example"])
    for group in groups.values():
        validate_group(group)
    return rows, json.loads((root / "pool_plan.json").read_text())[split], manifest


def backend_options(reference):
    return {
        **{
            key: reference[key]
            for key in (
                "training_config",
                "checkpoint",
                "config_file_sha256",
                "dataset_manifest_sha256",
                "source_dtype",
                "device",
                "local_files_only",
                "skip_training_launch_evidence",
            )
        },
        "split": "validation",
        "pool_anchors": 2048,
    }


def native_predictions(backend, rows, batch_size, top_k):
    predictions = defaultdict(dict)
    for key in MODEL_ORDER:
        tokenizer = backend.tokenizers[key]
        model = backend.source_model(key)
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            data = [row["models"][key] for row in batch]
            length = max(len(record["input_ids"]) for record in data)
            ids = torch.full(
                (len(batch), length), _pad_id(tokenizer), dtype=torch.long, device=backend.device
            )
            attention = torch.zeros_like(ids)
            for index, record in enumerate(data):
                size = len(record["input_ids"])
                ids[index, :size] = torch.tensor(record["input_ids"], device=backend.device)
                attention[index, :size] = torch.tensor(
                    record["attention_mask"], device=backend.device
                )
            positions = torch.tensor(
                [record["prediction_token_index"] for record in data], device=backend.device
            )
            with torch.inference_mode():
                scores = (
                    model(input_ids=ids, attention_mask=attention, use_cache=False)
                    .logits[torch.arange(len(batch), device=backend.device), positions]
                    .float()
                )
                if not torch.isfinite(scores).all():
                    raise ValueError(f"non-finite native logits for {key}/{start}")
                probs = scores.softmax(-1)
                top = scores.topk(top_k)
            for index, (row, record) in enumerate(zip(batch, data, strict=True)):
                example = row["example"]
                tokens = record["candidate_token_ids"]
                logits = {name: float(scores[index, token]) for name, token in tokens.items()}
                probabilities = {name: float(probs[index, token]) for name, token in tokens.items()}
                prediction_id = int(top.indices[index, 0])
                predicted = next(
                    (name for name, token in tokens.items() if token == prediction_id), None
                )
                result = candidate_behavior(example, logits, probabilities, predicted)
                predictions[example["example_id"]][key] = {
                    **result,
                    "predicted_token_id": prediction_id,
                    "predicted_token": tokenizer.decode(
                        [prediction_id], clean_up_tokenization_spaces=False
                    ),
                    "prediction_token_index": record["prediction_token_index"],
                    "candidates": {
                        name: {
                            "token_id": tokens[name],
                            "logit": value,
                            "probability": probabilities[name],
                        }
                        for name, value in logits.items()
                    },
                    "top_predictions": [
                        {
                            "token_id": token,
                            "token": tokenizer.decode([token], clean_up_tokenization_spaces=False),
                            "logit": value,
                            "probability": float(probs[index, token]),
                        }
                        for token, value in zip(
                            top.indices[index].tolist(), top.values[index].tolist(), strict=True
                        )
                    ],
                }
        print(f"Finished native scoring: {key}", flush=True)
    by_id = {row["example"]["example_id"]: row["example"] for row in rows}
    for example_id, by_model in predictions.items():
        example = by_id[example_id]
        condition = example["condition"]
        if condition in NEGATIVE_CONDITIONS:
            counterpart = (
                "counterfactual_affirmative"
                if condition.startswith("counterfactual")
                else "affirmative"
            )
            bases = predictions[f"{example['group_id']}/{counterpart}"]
            for model, result in by_model.items():
                result["negation_outcome"] = failure_class({**example, **result}, bases[model])
    return predictions


def behavior_summary(rows, predictions):
    partitions = defaultdict(list)
    grouped = defaultdict(dict)
    for row in rows:
        ex = row["example"]
        for model, prediction in predictions[ex["example_id"]].items():
            partitions[(model, ex["family"], ex["condition"])].append(prediction)
            grouped[(model, ex["family"], ex["group_id"])][ex["condition"]] = prediction
    metrics = {}
    for key, values in partitions.items():
        count = len(values)
        metrics["/".join(key)] = {
            "denominator": count,
            "candidate_correct": sum(x["candidate_correct"] for x in values),
            "exact_correct": sum(x["exact_correct"] for x in values),
            "outcomes": dict(
                Counter(x["negation_outcome"] for x in values if "negation_outcome" in x)
            ),
            **{
                f"mean_{field}": sum(x[field] for x in values) / count
                for field in (
                    "fixed_margin",
                    "correctness_margin",
                    "valid_mass",
                    "excluded_mass",
                    "unrelated_mass",
                )
            },
        }
    paired = defaultdict(Counter)
    for (model, family, _group), by_condition in grouped.items():
        affirmative, negative = by_condition["affirmative"], by_condition["negative"]
        counter_a, counter_n = (
            by_condition["counterfactual_affirmative"],
            by_condition["counterfactual_negative"],
        )
        paired[f"{model}/{family}"].update(
            groups=1,
            base_margin_moves_away_from_a=negative["fixed_margin"] < affirmative["fixed_margin"],
            counterfactual_margin_moves_away_from_b=counter_n["fixed_margin"]
            > counter_a["fixed_margin"],
            all_four_candidate_correct=all(
                x["candidate_correct"] for x in (affirmative, negative, counter_a, counter_n)
            ),
            all_four_exact_correct=all(
                x["exact_correct"] for x in (affirmative, negative, counter_a, counter_n)
            ),
        )
    return {"by_model_family_condition": metrics, "paired": dict(paired)}


def gate_statistics(examples, importance, representatives, component_stats):
    observed = defaultdict(lambda: defaultdict(Counter))
    denominators = defaultdict(Counter)
    for ex in examples:
        labels = [ex["condition"]]
        labels.extend(
            f"{model}/{p['negation_outcome']}"
            for model, p in ex["predictions"].items()
            if "negation_outcome" in p
        )
        sequence = set()
        for row in ex["positions"]:
            roles = set(row["roles_ending"])
            for label in labels:
                totals = denominators[label]
                totals["anchors"] += 1
                totals["queries"] += row["final_query"]
                for role in roles:
                    totals[f"role/{role}"] += 1
            for site in SITES:
                for feature, value in zip(
                    row[site]["active_gate_ids"], row[site]["values"], strict=True
                ):
                    gate = GateId(site, feature)
                    sequence.add(gate)
                    for label in labels:
                        data = observed[gate][label]
                        data.update(fires=1, sum=value, sq=value * value)
                        if row["final_query"]:
                            data.update(query_fires=1, query_sum=value)
                        for role in roles:
                            data[f"role/{role}/fires"] += 1
                            data[f"role/{role}/sum"] += value
        for label in labels:
            denominators[label]["examples"] += 1
            for gate in sequence:
                observed[gate][label]["sequence_fires"] += 1
    rows = []
    for gate, labels in sorted(observed.items()):
        beta = {}
        task_components = {}
        for model in MODEL_ORDER:
            reps = representatives.get((gate, model), ())
            values = [float(importance[rep.matrix].beta[rep.feature]) for rep in reps]
            beta[model] = max(values) if values else None
            stats = component_stats.get((model, gate.site, gate.feature), ())
            if stats:
                task_components[model] = {"n": len(stats), "mean_norm": sum(stats) / len(stats)}
        rows.append(
            {
                "gate": gate.key,
                "conditions": dict(labels),
                "reference_max_beta": beta,
                "available_component_stats": task_components,
            }
        )
    return {"denominators": dict(denominators), "gates": rows}


def extract(backend, rows, plan, predictions, output, importance, reps, split):
    by_id = {row["example"]["example_id"]: row for row in rows}
    planned = [example_id for p in plan for example_id in p["example_ids"]]
    if len(planned) != len(by_id) or set(planned) != set(by_id):
        raise ValueError("frozen pool plan does not cover the evaluated examples exactly once")
    meta = _anchor_metadata(rows)
    examples = {}
    pools = []
    component_stats = defaultdict(list)
    with (
        _JsonlWriter(output / "anchor_activations.jsonl") as activations,
        _JsonlWriter(output / "component_contributions.jsonl") as components,
    ):
        for index, (pool, background) in enumerate(
            zip(plan, backend.aligned_pools(len(plan)), strict=True)
        ):
            records = [by_id[key] for key in pool["example_ids"]]
            task = task_pool_from_verified(records, mechanism="negation")
            if task.anchor_count != pool["task_anchor_count"]:
                raise ValueError("task anchor inventory changed")
            aligned = merge_task_and_background(task, background, pool_anchors=2048)
            encoded = backend.encode_aligned_pool(aligned)
            pool_id = f"negation-{split}-pool-{index:03d}"
            for row, anchor in enumerate(encoded.pool.anchor_ids[: task.anchor_count]):
                metadata = meta[anchor]
                payload = {"pool_id": pool_id, "anchor_id": anchor, **metadata}
                for site in SITES:
                    active = encoded.pool.gates[site][row].nonzero(as_tuple=True)[0]
                    values = encoded.continuous[site][row, active]
                    if not bool(values.isfinite().all()) or not bool((values > 0).all()):
                        raise ValueError("natural continuous gate support is invalid")
                    payload[site] = {"active_gate_ids": active.tolist(), "values": values.tolist()}
                activations.write(payload)
                key = metadata["example_id"]
                if key not in examples:
                    examples[key] = {
                        **by_id[key]["example"],
                        "pool_id": pool_id,
                        "positions": [],
                        "predictions": predictions[key],
                    }
                examples[key]["positions"].append(payload)
            if split == "smoke" and index == 0:
                sample = {row["example"]["group_id"] for row in records[:4]}
                sample_meta = {
                    a: {**m, "annotated": True}
                    for a, m in meta.items()
                    if m["final_query"] and m["group_id"] in sample
                }
                stats = _component_rows(
                    backend=backend,
                    encoded=encoded,
                    anchor_meta=sample_meta,
                    importance=importance,
                    representatives=reps,
                    pool_id=pool_id,
                    writer=components,
                )
                for key, values in stats.items():
                    component_stats[key].extend(values)
            for site in SITES:
                if int(encoded.pool.gates[site].sum()) != 2048 * 32:
                    raise ValueError("BatchTopK pool does not contain K × anchor_count selections")
            pools.append(
                {
                    "pool_id": pool_id,
                    "anchors": 2048,
                    "k": 32,
                    "task_anchors": task.anchor_count,
                    "background_anchors": 2048 - task.anchor_count,
                    "example_ids": pool["example_ids"],
                    "family_counts": dict(Counter(row["example"]["family"] for row in records)),
                    "condition_counts": dict(
                        Counter(row["example"]["condition"] for row in records)
                    ),
                }
            )
            print(
                f"Natural pool {index + 1}/{len(plan)} with {task.anchor_count} task anchors",
                flush=True,
            )
    return list(examples.values()), pools, component_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "preparation", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "discovery"), required=True)
    args = parser.parse_args()
    started = time.monotonic()
    spec, reference = read_contract(args.config)
    rows, plan, preparation = load_preparation(args.preparation, args.config, args.split)
    if len({r["example"]["group_id"] for r in rows}) != spec["dataset"]["split_groups"][args.split]:
        raise ValueError("prepared group count changed")
    reference_root = Path(reference["components"]["reference_posthoc_root"])
    importance, reps = _load_references(reference_root)
    args.output.mkdir(parents=True, exist_ok=False)
    backend = V31CheckpointBackend(
        backend_options(reference), work_dir=args.output / "stream_state"
    )
    try:
        backend_provenance = backend.provenance()
        if backend_provenance["checkpoint_step"] != spec["checkpoint_step"]:
            raise ValueError("runtime checkpoint step changed")
        for key, pinned in preparation["models"].items():
            native = backend_provenance["models"][key]
            if (native["model_id"], native["revision"], native["tokenizer_revision"]) != (
                pinned["id"],
                pinned["revision"],
                pinned["tokenizer_revision"],
            ):
                raise ValueError(f"native source/tokenizer identity changed: {key}")
        preds = native_predictions(
            backend,
            rows,
            reference["behavior"]["batch_size"],
            reference["behavior"]["top_predictions"],
        )
        with _JsonlWriter(args.output / "native_predictions.jsonl") as handle:
            for row in rows:
                ex = row["example"]
                handle.write({"example": ex, "models": preds[ex["example_id"]]})
        behavior = behavior_summary(rows, preds)
        write_json(args.output / "behavior_summary.json", behavior)
        if any(
            not math.isfinite(s["mean_fixed_margin"])
            for s in behavior["by_model_family_condition"].values()
        ):
            raise ValueError("native behavior validity failed")
        examples, pools, sparse = extract(
            backend, rows, plan, preds, args.output, importance, reps, args.split
        )
        with _JsonlWriter(args.output / "activation_examples.jsonl") as handle:
            for example in examples:
                handle.write(example)
        write_json(
            args.output / "gate_summary.json", gate_statistics(examples, importance, reps, sparse)
        )
        write_json(args.output / "pools.json", pools)
    finally:
        backend.close()
    files = [path for path in args.output.iterdir() if path.is_file()]
    manifest = {
        "mechanism": "negation",
        "split": args.split,
        "heldout_opened": False,
        "preparation_sha256": sha256_file(args.preparation / "manifest.json"),
        "config_sha256": sha256_file(args.config),
        "checkpoint_sha256": preparation["checkpoint_sha256"],
        "reference_representatives_sha256": sha256_file(
            reference_root / "representatives/COMPLETE"
        ),
        "reference_importance_sha256": sha256_file(reference_root / "importance/COMPLETE"),
        "source": runtime_provenance(),
        "backend": backend_provenance,
        "groups": spec["dataset"]["split_groups"][args.split],
        "examples": len(rows),
        "task_anchors": sum(p["task_anchors"] for p in pools),
        "runtime": {
            "seconds": time.monotonic() - started,
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
            "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "flat_artifact_bytes": sum(p.stat().st_size for p in files),
        },
        "artifacts": {p.name: sha256_file(p) for p in files},
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(
        args.output / "COMPLETE", {"manifest_sha256": sha256_file(args.output / "manifest.json")}
    )


if __name__ == "__main__":
    main()
