"""RCCAS Stage 4 decision-flip audit and frozen held-out confirmation.

This module is intentionally isolated from the Stage 1--3 implementation.  It reuses
the real clean-gate and ``o_proj`` patching primitives, but never screens or reranks
components on held-out inputs.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch

from aspd.multimodel.causal import p6_patched_forward
from aspd.multimodel.mechanism_discovery import capture_model_forward
from aspd.multimodel.original_recipe import sha256_file
from aspd.multimodel.risk_attribution import (
    FROZEN_MATRIX,
    _atomic_new_json,
    _atomic_new_text,
    _git_head,
    _last_logits,
    _load_runtime,
    _norm_validity,
    _source_weight_fingerprint,
    _verify_screening_artifact_hashes,
    build_choice_items,
    decision_margin,
    load_risk_attribution_config,
    load_scenario_pairs,
    tokenize_choice_items,
)

EXPECTED_ARMS = (
    "learned",
    "gate_off",
    "active_matched",
    "orthogonal_matched",
    "random_same_l2_seed_0",
    "random_same_l2_seed_1",
    "random_same_l2_seed_2",
)
PRIMARY_OUTCOMES = (
    "target_risk_effect",
    "target_risk_minus_safe_effect",
    "target_minus_base_risk_safe_differential",
    "target_risk_learned_minus_active_control",
    "target_risk_learned_minus_orthogonal_control",
    "target_risk_learned_minus_pooled_random_control",
)


def load_heldout_contract(path: str | Path) -> dict[str, Any]:
    """Load the preregistered contract and reject any drift in its frozen structure."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported held-out contract schema")
    if value.get("status") != "FROZEN_BEFORE_HELDOUT_MODEL_OUTCOMES":
        raise ValueError("held-out analysis contract is not frozen")
    if value.get("split") != "heldout" or value.get("alpha") != 1.0:
        raise ValueError("held-out contract must use split=heldout and alpha=1")
    if value.get("random_control_seeds") != [0, 1, 2]:
        raise ValueError("held-out random-control seeds changed")
    expected = [(2883, "negative", 2248), (4742, "negative", 776), (4300, "positive", 309)]
    observed = [
        (
            int(row["component_id"]),
            str(row["expected_direction"]),
            int(row["active_control_component_id"]),
        )
        for row in value.get("candidates", [])
    ]
    if observed != expected:
        raise ValueError("held-out candidate identities, directions, or controls changed")
    if tuple(value.get("primary_outcomes", [])) != PRIMARY_OUTCOMES:
        raise ValueError("held-out primary outcomes changed")
    if value.get("expected_scenario_pairs") != 12 or value.get("expected_choice_items") != 48:
        raise ValueError("held-out item-count contract changed")
    if value.get("expected_intervention_records") != 2016:
        raise ValueError("held-out intervention-grid size changed")
    return value


def _choice(margin: float) -> Literal["PROCEED", "POSTPONE", "TIE"]:
    if margin > 0:
        return "PROCEED"
    if margin < 0:
        return "POSTPONE"
    return "TIE"


def _flip_counts(clean: Sequence[float], patched: Sequence[float]) -> dict[str, int]:
    clean_choices = [_choice(value) for value in clean]
    patched_choices = [_choice(value) for value in patched]
    return {
        "clean_proceed": clean_choices.count("PROCEED"),
        "clean_postpone": clean_choices.count("POSTPONE"),
        "clean_tie": clean_choices.count("TIE"),
        "patched_proceed": patched_choices.count("PROCEED"),
        "patched_postpone": patched_choices.count("POSTPONE"),
        "patched_tie": patched_choices.count("TIE"),
        "proceed_to_postpone": sum(
            left == "PROCEED" and right == "POSTPONE"
            for left, right in zip(clean_choices, patched_choices, strict=True)
        ),
        "postpone_to_proceed": sum(
            left == "POSTPONE" and right == "PROCEED"
            for left, right in zip(clean_choices, patched_choices, strict=True)
        ),
        "net_proceed_change": patched_choices.count("PROCEED")
        - clean_choices.count("PROCEED"),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def decision_flip_audit(records_path: str | Path) -> list[dict[str, Any]]:
    """Audit Stage 3 choices without treating A/B orders as independent scenarios."""

    records = _load_jsonl(Path(records_path))
    if len(records) != 100_800:
        raise ValueError(f"expected 100800 Stage 3 records, observed {len(records)}")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (
            int(row["component_id"]),
            float(row["alpha"]),
            str(row["model_name"]),
            str(row["condition"]),
            str(row["arm"]),
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        component_id, alpha, model_name, condition, arm = key
        for scope in ("primary", "swapped", "pair_aggregated"):
            invalid = Counter()
            clean_values: list[float] = []
            patched_values: list[float] = []
            if scope != "pair_aggregated":
                selected = [row for row in rows if row["order_variant"] == scope]
                for row in selected:
                    if not bool(row["causal_comparison_valid"]):
                        invalid[str(row.get("invalid_reason") or "UNSPECIFIED_INVALID") ] += 1
                        continue
                    clean_values.append(float(row["clean_margin"]))
                    patched_values.append(float(row["patched_margin"]))
                expected = 24
            else:
                by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    by_pair[str(row["pair_id"])].append(row)
                for _pair_id, pair_rows in sorted(by_pair.items()):
                    variants = {str(row["order_variant"]): row for row in pair_rows}
                    if set(variants) != {"primary", "swapped"} or len(pair_rows) != 2:
                        invalid["MISSING_OR_DUPLICATE_ORDER_VARIANT"] += 1
                        continue
                    bad = [row for row in pair_rows if not bool(row["causal_comparison_valid"])]
                    if bad:
                        for row in bad:
                            invalid[str(row.get("invalid_reason") or "UNSPECIFIED_INVALID")] += 1
                        continue
                    clean_values.append(sum(float(row["clean_margin"]) for row in pair_rows) / 2)
                    patched_values.append(sum(float(row["patched_margin"]) for row in pair_rows) / 2)
                expected = 24
            output.append(
                {
                    "component_id": component_id,
                    "alpha": alpha,
                    "model_name": model_name,
                    "condition": condition,
                    "arm": arm,
                    "order_scope": scope,
                    "expected_units": expected,
                    "valid_units": len(clean_values),
                    "invalid_units": expected - len(clean_values),
                    "invalid_reason_counts": json.dumps(dict(sorted(invalid.items())), sort_keys=True),
                    **_flip_counts(clean_values, patched_values),
                }
            )

    # Pool the three random controls within each scenario/order before aggregation.
    random_rows = [row for row in records if str(row["arm"]).startswith("random_same_l2_seed_")]
    random_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in random_rows:
        key = (
            int(row["component_id"]), float(row["alpha"]), str(row["model_name"]),
            str(row["condition"]), str(row["pair_id"]), str(row["order_variant"]),
        )
        random_groups[key].append(row)
    pooled_order: list[dict[str, Any]] = []
    for key, rows in random_groups.items():
        if len(rows) != 3 or not all(bool(row["causal_comparison_valid"]) for row in rows):
            continue
        component_id, alpha, model_name, condition, pair_id, order = key
        pooled_order.append(
            {
                "component_id": component_id, "alpha": alpha, "model_name": model_name,
                "condition": condition, "pair_id": pair_id, "order_variant": order,
                "clean_margin": sum(float(row["clean_margin"]) for row in rows) / 3,
                "patched_margin": sum(float(row["patched_margin"]) for row in rows) / 3,
            }
        )
    pooled_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pooled_order:
        pooled_groups[(row["component_id"], row["alpha"], row["model_name"], row["condition"])].append(row)
    for key, rows in sorted(pooled_groups.items()):
        component_id, alpha, model_name, condition = key
        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_pair[str(row["pair_id"])].append(row)
        clean_values, patched_values = [], []
        for pair_rows in by_pair.values():
            if len(pair_rows) == 2 and {row["order_variant"] for row in pair_rows} == {"primary", "swapped"}:
                clean_values.append(sum(float(row["clean_margin"]) for row in pair_rows) / 2)
                patched_values.append(sum(float(row["patched_margin"]) for row in pair_rows) / 2)
        output.append(
            {
                "component_id": component_id, "alpha": alpha, "model_name": model_name,
                "condition": condition, "arm": "random_same_l2_pooled", "order_scope": "pair_aggregated",
                "expected_units": 24, "valid_units": len(clean_values),
                "invalid_units": 24 - len(clean_values), "invalid_reason_counts": "{}",
                **_flip_counts(clean_values, patched_values),
            }
        )
    return output


def write_decision_flip_audit(
    records_path: str | Path, csv_path: str | Path, markdown_path: str | Path
) -> dict[str, Any]:
    rows = decision_flip_audit(records_path)
    csv_output, markdown_output = Path(csv_path), Path(markdown_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = [
        row for row in rows
        if row["alpha"] == 1.0 and row["model_name"] == "target"
        and row["condition"] == "risk" and row["order_scope"] == "pair_aggregated"
        and row["arm"] in {"learned", "active_matched", "orthogonal_matched", "random_same_l2_pooled"}
    ]
    by_component = defaultdict(dict)
    for row in primary:
        by_component[int(row["component_id"])][str(row["arm"])] = row
    lines = [
        "# RCCAS Stage 4 decision-flip audit", "", "Date: 2026-10-31", "",
        "This CPU-only audit reconstructs choices from the de-ordered Stage 3 margin. "
        "A/B orders are shown separately in the CSV and averaged within each scenario pair here. "
        "A nonzero margin change is not counted as a decision flip.", "",
        f"- Stage 3 records: `{Path(records_path)}`", f"- Records SHA-256: `{sha256_file(records_path)}`",
        "- Primary audit: alpha=1; lower doses remain exploratory in the CSV.",
        "- Invalid same-prefix controls are excluded without replacement and retain reasons in the CSV.", "",
        "## Target-risk alpha=1 pair-level decisions", "",
        "| Component | Learned clean PROCEED | Learned P→PST | Learned PST→P | Net risky choices | Active net | Orthogonal net | Random net |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for component_id in sorted(by_component):
        group = by_component[component_id]
        learned = group.get("learned", {})
        def metric(arm: str, name: str, *, source: Mapping[str, Any] = group) -> str:
            row = source.get(arm)
            return "NA" if row is None else str(row[name])
        lines.append(
            f"| {component_id} | {learned.get('clean_proceed', 'NA')}/24 | "
            f"{learned.get('proceed_to_postpone', 'NA')} | {learned.get('postpone_to_proceed', 'NA')} | "
            f"{learned.get('net_proceed_change', 'NA')} | {metric('active_matched', 'net_proceed_change')} | "
            f"{metric('orthogonal_matched', 'net_proceed_change')} | "
            f"{metric('random_same_l2_pooled', 'net_proceed_change')} |"
        )
    lines.extend([
        "", "## Interpretation boundary", "",
        "Zero flips do not imply zero causal margin effect. Conversely, a flip is descriptive unless it "
        "survives the same-prefix norm-matched controls. Discovery prompts remain selection-conditioned; "
        "held-out confirmation is analyzed separately.", "",
    ])
    _atomic_new_text(markdown_output, "\n".join(lines))
    return {"row_count": len(rows), "csv_sha256": sha256_file(csv_output), "markdown_sha256": sha256_file(markdown_output)}


def exact_one_sided_sign_flip(values: Sequence[float], direction: str) -> float:
    """Exact randomization p-value for a directional paired mean."""

    signed = [(-value if direction == "negative" else value) for value in values]
    if not signed:
        return math.nan
    observed = sum(signed) / len(signed)
    exceed = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(signed)):
        permuted = sum(sign * value for sign, value in zip(signs, signed, strict=True)) / len(signed)
        exceed += permuted >= observed - 1e-15
    return exceed / (2 ** len(signed))


def paired_bootstrap_ci(
    values: Sequence[float], *, replicates: int, seed: int
) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * (replicates - 1))]
    hi = means[int(0.975 * (replicates - 1))]
    return lo, hi


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def verify_heldout_preflight(
    config_path: str | Path,
    contract_path: str | Path,
    readout_validation_report: str | Path,
    screening_dir: str | Path,
    stage3_report: str | Path,
    stage3_raw_report: str | Path,
    stage3_records: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Verify frozen provenance without loading either 7B model."""

    config_file, contract_file = Path(config_path), Path(contract_path)
    readout_path, screening_root = Path(readout_validation_report), Path(screening_dir)
    published_stage3, raw_stage3, records_path = (
        Path(stage3_report), Path(stage3_raw_report), Path(stage3_records)
    )
    config = load_risk_attribution_config(config_file)
    contract = load_heldout_contract(contract_file)
    paths_and_hashes = (
        (readout_path, "readout_validation_sha256"),
        (screening_root / "screening_manifest.json", "screening_manifest_sha256"),
        (screening_root / "frozen_candidates.json", "frozen_candidates_sha256"),
        (published_stage3, "stage3_report_sha256"),
        (raw_stage3, "stage3_raw_report_sha256"),
        (records_path, "stage3_records_sha256"),
    )
    for path, field in paths_and_hashes:
        if not path.is_file() or sha256_file(path) != contract[field]:
            raise ValueError(f"frozen provenance mismatch: {field}")
    if config.dataset_sha256 != contract["dataset_sha256"]:
        raise ValueError("config and held-out contract dataset hashes differ")
    if sha256_file(config.dataset_file) != contract["dataset_sha256"]:
        raise ValueError("held-out dataset checksum mismatch")
    if config.expected_checkpoint_sha256 != contract["checkpoint_sha256"]:
        raise ValueError("checkpoint checksum differs from held-out contract")
    readout = json.loads(readout_path.read_text(encoding="utf-8"))
    if readout.get("status") != "READOUT_VALIDATED":
        raise ValueError("held-out confirmation requires READOUT_VALIDATED")
    screening_manifest = json.loads(
        (screening_root / "screening_manifest.json").read_text(encoding="utf-8")
    )
    artifact_audit = _verify_screening_artifact_hashes(screening_root, screening_manifest)
    if screening_manifest.get("git_commit") != contract["source_experiment_commit"]:
        raise ValueError("screening source commit mismatch")
    if screening_manifest.get("checkpoint_sha256") != contract["checkpoint_sha256"]:
        raise ValueError("screening checkpoint mismatch")
    if screening_manifest.get("dataset_sha256") != contract["dataset_sha256"]:
        raise ValueError("screening dataset mismatch")
    frozen = json.loads(
        (screening_root / "frozen_candidates.json").read_text(encoding="utf-8")
    )
    frozen_index = {int(row["component_id"]): row for row in frozen["candidates"]}
    for candidate in contract["candidates"]:
        component_id = int(candidate["component_id"])
        row = frozen_index.get(component_id)
        if row is None or int(row["active_control_component_id"]) != int(
            candidate["active_control_component_id"]
        ):
            raise ValueError(f"frozen candidate/control mismatch for {component_id}")
    rows = load_scenario_pairs(config.dataset_file)
    discovery_ids = {row["pair_id"] for row in rows if row["split"] == "discovery"}
    heldout_ids = {row["pair_id"] for row in rows if row["split"] == "heldout"}
    items = build_choice_items(rows, split="heldout")
    if discovery_ids & heldout_ids or len(heldout_ids) != 12 or len(items) != 48:
        raise ValueError("discovery/held-out split contract failed")
    expected_records = (
        len(contract["candidates"]) * len(items) * 2 * len(EXPECTED_ARMS)
    )
    if expected_records != contract["expected_intervention_records"]:
        raise ValueError("held-out intervention-grid count mismatch")
    report = {
        "status": "HELDOUT_SCIENTIFIC_PREFLIGHT_GO",
        "execution_code_commit": _git_head(),
        "original_source_experiment_commit": contract["source_experiment_commit"],
        "contract_sha256": sha256_file(contract_file),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "dataset_sha256": contract["dataset_sha256"],
        "stage3_report_sha256": contract["stage3_report_sha256"],
        "frozen_candidates_sha256": contract["frozen_candidates_sha256"],
        "heldout_pair_count": len(heldout_ids),
        "heldout_choice_item_count": len(items),
        "expected_intervention_record_count": expected_records,
        "candidate_ids": [int(row["component_id"]) for row in contract["candidates"]],
        "screening_artifact_hash_audit": artifact_audit,
        "heldout_model_outcomes_accessed": False,
    }
    if output is not None:
        _atomic_new_json(Path(output), report)
    return report


def run_heldout_confirmation(
    config_path: str | Path,
    contract_path: str | Path,
    readout_validation_report: str | Path,
    screening_dir: str | Path,
    stage3_report: str | Path,
    stage3_raw_report: str | Path,
    stage3_records: str | Path,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Execute the single frozen held-out grid using the real L14 ``o_proj`` patch."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite held-out output {destination}")
    preflight = verify_heldout_preflight(
        config_path, contract_path, readout_validation_report, screening_dir,
        stage3_report, stage3_raw_report, stage3_records,
    )
    config = load_risk_attribution_config(config_path)
    contract = load_heldout_contract(contract_path)
    rows = load_scenario_pairs(config.dataset_file)
    items = build_choice_items(rows, split="heldout")
    aspd, tokenizer, models, provenance, decomposition, handoff = _load_runtime(config, device)
    ids, valid, label_ids = tokenize_choice_items(tokenizer, items)
    source_before = _source_weight_fingerprint(models, aspd)
    runtime_device = next(models[0].parameters()).device
    if runtime_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime_device)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch_start in range(0, len(items), config.gpu_batch_size):
        batch_stop = min(batch_start + config.gpu_batch_size, len(items))
        batch_items = items[batch_start:batch_stop]
        batch_ids, batch_valid = ids[batch_start:batch_stop], valid[batch_start:batch_stop]
        clean = [
            capture_model_forward(model, spec, batch_ids, batch_valid, capture_matrices=True)
            for model, spec in zip(models, aspd.models, strict=True)
        ]
        decomposition_device = next(decomposition.parameters()).device
        code = decomposition.encode(
            [record["R"].to(decomposition_device).float() for record in clean],
            batch_valid.to(decomposition_device),
            update_statistics=False,
        )
        clean_margins = [
            decision_margin(
                _last_logits(record["logits"], batch_valid.to(record["logits"].device)),
                batch_items,
                label_ids,
            ).float().cpu()
            for record in clean
        ]
        for candidate in contract["candidates"]:
            component_id = int(candidate["component_id"])
            active_control_id = int(candidate["active_control_component_id"])
            selected = torch.tensor([component_id], device=code.gate.device)
            learned_gate = code.gate.index_select(-1, selected).detach().cpu()
            arms: list[tuple[str, dict[str, Any]]] = [
                ("learned", {}),
                ("gate_off", {"gate_override": torch.zeros_like(learned_gate)}),
                ("active_matched", {"matched_write_component_ids": [active_control_id]}),
                ("orthogonal_matched", {"norm_control": True}),
                *[
                    (f"random_same_l2_seed_{seed}", {"random_control_seed": seed})
                    for seed in contract["random_control_seeds"]
                ],
            ]
            for model_index, (model, spec) in enumerate(zip(models, aspd.models, strict=True)):
                for arm_name, arm_kwargs in arms:
                    kwargs = dict(arm_kwargs)
                    arm_gate = kwargs.pop("gate_override", learned_gate)
                    patched = p6_patched_forward(
                        model, spec, decomposition, model_index, batch_ids, batch_valid,
                        arm_gate, [component_id], strength=1.0, **kwargs,
                    )
                    patched_margin = decision_margin(
                        _last_logits(patched.logits, batch_valid.to(patched.logits.device)),
                        batch_items,
                        label_ids,
                    ).float().cpu()
                    diagnostics = patched.matrix_diagnostics["o_proj"]["per_sequence"]
                    for row_index, item in enumerate(batch_items):
                        diagnostic = diagnostics[row_index]
                        valid_comparison, invalid_reason = _norm_validity(
                            diagnostic,
                            float(contract["post_bf16_per_position_norm_relative_tolerance"]),
                        )
                        if arm_name == "learned":
                            actual = float(diagnostic["actual_applied_output_difference_l2"])
                            intended = float(diagnostic["intended_fp32_subtraction_l2"])
                            valid_comparison = actual > 0.0
                            invalid_reason = None if actual > 0 else (
                                "INEFFECTIVE_AFTER_DTYPE_ROUNDING"
                                if intended > 0 else "ZERO_ACTIVATION_OR_ZERO_WRITE"
                            )
                        elif arm_name == "gate_off":
                            valid_comparison, invalid_reason = True, None
                        records.append(
                            {
                                "item_id": item.item_id,
                                "pair_id": item.pair_id,
                                "split": "heldout",
                                "domain": item.domain,
                                "condition": item.condition,
                                "order_variant": item.order_variant,
                                "model_name": spec.name,
                                "component_id": component_id,
                                "expected_direction": candidate["expected_direction"],
                                "active_control_component_id": active_control_id,
                                "arm": arm_name,
                                "alpha": 1.0,
                                "clean_margin": float(clean_margins[model_index][row_index]),
                                "patched_margin": float(patched_margin[row_index]),
                                "exact_delta_margin": float(
                                    patched_margin[row_index] - clean_margins[model_index][row_index]
                                ),
                                "intended_fp32_subtraction_l2": diagnostic[
                                    "intended_fp32_subtraction_l2"
                                ],
                                "actual_applied_output_difference_l2": diagnostic[
                                    "actual_applied_output_difference_l2"
                                ],
                                "candidate_actual_applied_output_difference_l2": diagnostic[
                                    "candidate_actual_applied_output_difference_l2"
                                ],
                                "rounded_to_zero_position_count": diagnostic[
                                    "rounded_to_zero_position_count"
                                ],
                                "causal_comparison_valid": valid_comparison,
                                "invalid_reason": invalid_reason,
                            }
                        )
    runtime = time.perf_counter() - started
    source_after = _source_weight_fingerprint(models, aspd)
    if source_after != source_before:
        raise RuntimeError("source model weights changed during held-out confirmation")
    if any(
        model.get_submodule(FROZEN_MATRIX)._forward_hooks
        or model.get_submodule(FROZEN_MATRIX)._forward_pre_hooks
        for model in models
    ):
        raise RuntimeError("held-out intervention hooks were not removed")
    if len(records) != int(contract["expected_intervention_records"]):
        raise RuntimeError(f"held-out record count mismatch: {len(records)}")
    gate_off = [row for row in records if row["arm"] == "gate_off"]
    max_noop_error = max(abs(float(row["exact_delta_margin"])) for row in gate_off)
    if max_noop_error > 1e-6:
        raise RuntimeError(f"gate-off no-op mismatch: {max_noop_error}")
    if not all(math.isfinite(float(row["exact_delta_margin"])) for row in records):
        raise RuntimeError("non-finite held-out margin effect")
    destination.mkdir(parents=True, exist_ok=False)
    records_path = destination / "heldout_exact_margin_records.jsonl"
    _atomic_new_text(
        records_path,
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records),
    )
    report = {
        "status": "HELDOUT_EXACT_CONFIRMATION_COMPLETED_UNVERIFIED",
        "execution_code_commit": _git_head(),
        "original_source_experiment_commit": contract["source_experiment_commit"],
        "contract_sha256": sha256_file(contract_path),
        "checkpoint_sha256": handoff["checkpoint_sha256"],
        "dataset_sha256": contract["dataset_sha256"],
        "stage3_report_sha256": contract["stage3_report_sha256"],
        "frozen_candidates_sha256": contract["frozen_candidates_sha256"],
        "candidate_ids": [int(row["component_id"]) for row in contract["candidates"]],
        "choice_item_count": len(items),
        "record_count": len(records),
        "runtime_seconds": runtime,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(runtime_device))
            if runtime_device.type == "cuda" else None
        ),
        "invalid_learned_records": sum(
            row["arm"] == "learned" and not row["causal_comparison_valid"] for row in records
        ),
        "invalid_control_records": sum(
            row["arm"] not in {"learned", "gate_off"}
            and not row["causal_comparison_valid"] for row in records
        ),
        "invalid_reason_counts": dict(sorted(Counter(
            str(row["invalid_reason"]) for row in records if row["invalid_reason"]
        ).items())),
        "maximum_gate_off_margin_error": max_noop_error,
        "source_weights_unchanged": True,
        "hook_cleanup_passed": True,
        "source_model_provenance": provenance,
        "records_sha256": sha256_file(records_path),
        "scientific_boundary": (
            "Held-out fixed-prefix causal influence is not proof of mediation of the original fine-tuning process."
        ),
        "preflight": preflight,
    }
    _atomic_new_json(destination / "heldout_stage_report.json", report)
    return report


def _pair_effects(
    records: Sequence[Mapping[str, Any]], *, component_id: int, model_name: str,
    condition: str, arm: str,
) -> dict[str, float]:
    selected = [
        row for row in records
        if int(row["component_id"]) == component_id
        and row["model_name"] == model_name and row["condition"] == condition
        and (row["arm"] == arm if arm != "pooled_random" else str(row["arm"]).startswith("random_same_l2_seed_"))
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["pair_id"])].append(row)
    expected = 6 if arm == "pooled_random" else 2
    output = {}
    for pair_id, pair_rows in grouped.items():
        if len(pair_rows) != expected or not all(bool(row["causal_comparison_valid"]) for row in pair_rows):
            continue
        if {row["order_variant"] for row in pair_rows} != {"primary", "swapped"}:
            continue
        output[pair_id] = sum(float(row["exact_delta_margin"]) for row in pair_rows) / len(pair_rows)
    return output


def _combine_pair_maps(operation: Any, *maps: Mapping[str, float]) -> dict[str, float]:
    common = set.intersection(*(set(value) for value in maps)) if maps else set()
    return {key: float(operation(*(value[key] for value in maps))) for key in sorted(common)}


def analyze_heldout_confirmation(
    records_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    stage3_metrics_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Apply only the preregistered pair-level tests to immutable held-out records."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite held-out analysis {destination}")
    contract = load_heldout_contract(contract_path)
    records = _load_jsonl(Path(records_path))
    if len(records) != contract["expected_intervention_records"]:
        raise ValueError("held-out records are incomplete")
    analyses: list[dict[str, Any]] = []
    pair_payload: dict[str, Any] = {}
    for candidate in contract["candidates"]:
        component_id = int(candidate["component_id"])
        direction = str(candidate["expected_direction"])
        tr = _pair_effects(records, component_id=component_id, model_name="target", condition="risk", arm="learned")
        ts = _pair_effects(records, component_id=component_id, model_name="target", condition="safe_control", arm="learned")
        br = _pair_effects(records, component_id=component_id, model_name="base", condition="risk", arm="learned")
        bs = _pair_effects(records, component_id=component_id, model_name="base", condition="safe_control", arm="learned")
        active = _pair_effects(records, component_id=component_id, model_name="target", condition="risk", arm="active_matched")
        orthogonal = _pair_effects(records, component_id=component_id, model_name="target", condition="risk", arm="orthogonal_matched")
        random_control = _pair_effects(records, component_id=component_id, model_name="target", condition="risk", arm="pooled_random")
        outcomes = {
            "target_risk_effect": tr,
            "target_risk_minus_safe_effect": _combine_pair_maps(lambda a, b: a - b, tr, ts),
            "target_minus_base_risk_safe_differential": _combine_pair_maps(
                lambda a, b, c, d: (a - b) - (c - d), tr, ts, br, bs
            ),
            "target_risk_learned_minus_active_control": _combine_pair_maps(lambda a, b: a - b, tr, active),
            "target_risk_learned_minus_orthogonal_control": _combine_pair_maps(lambda a, b: a - b, tr, orthogonal),
            "target_risk_learned_minus_pooled_random_control": _combine_pair_maps(lambda a, b: a - b, tr, random_control),
        }
        pair_payload[str(component_id)] = outcomes
        for outcome_name in PRIMARY_OUTCOMES:
            values = list(outcomes[outcome_name].values())
            lo, hi = paired_bootstrap_ci(
                values,
                replicates=int(contract["statistical_contract"]["bootstrap_replicates"]),
                seed=int(contract["statistical_contract"]["bootstrap_seed"]),
            )
            analyses.append(
                {
                    "component_id": component_id,
                    "expected_direction": direction,
                    "outcome": outcome_name,
                    "valid_pairs": len(values),
                    "mean": sum(values) / len(values) if values else math.nan,
                    "bootstrap_ci_lower": lo,
                    "bootstrap_ci_upper": hi,
                    "unadjusted_p_value": exact_one_sided_sign_flip(values, direction),
                }
            )
    finite_p = [float(row["unadjusted_p_value"]) for row in analyses]
    adjusted = holm_adjust(finite_p)
    for row, value in zip(analyses, adjusted, strict=True):
        row["holm_adjusted_p_value"] = value
        expected_interval = (
            row["bootstrap_ci_upper"] < 0
            if row["expected_direction"] == "negative"
            else row["bootstrap_ci_lower"] > 0
        )
        row["expected_direction_interval_excludes_zero"] = expected_interval

    summaries: list[dict[str, Any]] = []
    by_candidate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in analyses:
        by_candidate[int(row["component_id"])].append(row)
    for candidate in contract["candidates"]:
        component_id = int(candidate["component_id"])
        rows = by_candidate[component_id]
        complete = all(row["valid_pairs"] == 12 for row in rows)
        passing = [
            row for row in rows
            if row["holm_adjusted_p_value"] <= 0.05
            and row["expected_direction_interval_excludes_zero"]
        ]
        primary = next(row for row in rows if row["outcome"] == "target_risk_effect")
        if not complete:
            classification = "INCONCLUSIVE"
        elif len(passing) == len(PRIMARY_OUTCOMES):
            classification = "HELDOUT_CONTROLLED_EFFECT_CONFIRMED"
        elif primary in passing:
            classification = "HELDOUT_EXACT_EFFECT_ONLY"
        else:
            classification = "NOT_REPLICATED"
        learned_rows = [
            row for row in records
            if int(row["component_id"]) == component_id and row["model_name"] == "target"
            and row["condition"] == "risk" and row["arm"] == "learned"
        ]
        by_pair = defaultdict(list)
        for row in learned_rows:
            by_pair[str(row["pair_id"])].append(row)
        clean, patched = [], []
        for pair_rows in by_pair.values():
            if len(pair_rows) == 2:
                clean.append(sum(float(row["clean_margin"]) for row in pair_rows) / 2)
                patched.append(sum(float(row["patched_margin"]) for row in pair_rows) / 2)
        flips = _flip_counts(clean, patched)
        outcome_index = {row["outcome"]: row for row in rows}
        summaries.append(
            {
                "component_id": component_id,
                "direction": candidate["expected_direction"],
                "classification": classification,
                "target_risk": outcome_index["target_risk_effect"]["mean"],
                "target_safe": sum(_pair_effects(records, component_id=component_id, model_name="target", condition="safe_control", arm="learned").values()) / max(1, len(_pair_effects(records, component_id=component_id, model_name="target", condition="safe_control", arm="learned"))),
                "target_minus_base_did": outcome_index["target_minus_base_risk_safe_differential"]["mean"],
                "active_control_difference": outcome_index["target_risk_learned_minus_active_control"]["mean"],
                "orthogonal_control_difference": outcome_index["target_risk_learned_minus_orthogonal_control"]["mean"],
                "random_control_difference": outcome_index["target_risk_learned_minus_pooled_random_control"]["mean"],
                "valid_pairs": min(int(row["valid_pairs"]) for row in rows),
                "adjusted_p_value_target_risk": primary["holm_adjusted_p_value"],
                "beneficial_flips": flips["proceed_to_postpone"] if candidate["expected_direction"] == "negative" else flips["postpone_to_proceed"],
                "harmful_flips": flips["postpone_to_proceed"] if candidate["expected_direction"] == "negative" else flips["proceed_to_postpone"],
                "net_proceed_change": flips["net_proceed_change"],
            }
        )

    discovery = {}
    if stage3_metrics_csv is not None:
        with Path(stage3_metrics_csv).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["component_id"]) in {2883, 4742, 4300} and float(row["alpha"]) == 1.0:
                    discovery[int(row["component_id"])] = row
    destination.mkdir(parents=True, exist_ok=False)
    order_rows = [
        {
            key: row[key]
            for key in (
                "component_id", "pair_id", "condition", "order_variant", "model_name",
                "clean_margin", "patched_margin", "exact_delta_margin",
                "causal_comparison_valid", "invalid_reason",
            )
        }
        for row in records
        if row["arm"] == "learned"
    ]
    order_csv = destination / "rccas_stage4_heldout_learned_order_effects.csv"
    with order_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(order_rows[0]))
        writer.writeheader()
        writer.writerows(order_rows)
    invalid_fields = (
        "component_id", "pair_id", "condition", "order_variant", "model_name",
        "arm", "invalid_reason", "actual_applied_output_difference_l2",
        "candidate_actual_applied_output_difference_l2",
    )
    invalid_rows = [
        {
            key: row.get(key)
            for key in invalid_fields
        }
        for row in records
        if not row["causal_comparison_valid"]
    ]
    invalid_csv = destination / "rccas_stage4_heldout_invalid_records.csv"
    with invalid_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(invalid_fields))
        writer.writeheader()
        writer.writerows(invalid_rows)
    stats_csv = destination / "rccas_stage4_heldout_statistics.csv"
    with stats_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analyses[0]))
        writer.writeheader()
        writer.writerows(analyses)
    summary_json = destination / "rccas_stage4_heldout_summary.json"
    _atomic_new_json(summary_json, {"summaries": summaries, "outcome_tests": analyses})
    pair_json = destination / "rccas_stage4_heldout_pair_effects.json"
    _atomic_new_json(pair_json, pair_payload)
    lines = [
        "# RCCAS Stage 4 held-out causal confirmation", "", "Date: 2026-10-31", "",
        "Primary inference uses 12 held-out scenario pairs, averages A/B answer orders before testing, "
        "and applies one global Holm correction to all 18 preregistered directional tests.", "",
        "| Component | Direction | Exact Target risk | Target safe | Target−Base DID | Active difference | Orthogonal difference | Random difference | Valid pairs | Adjusted p (Target risk) | Decision flips | Classification |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['component_id']} | {row['direction']} | {row['target_risk']:.6f} | "
            f"{row['target_safe']:.6f} | {row['target_minus_base_did']:.6f} | "
            f"{row['active_control_difference']:.6f} | {row['orthogonal_control_difference']:.6f} | "
            f"{row['random_control_difference']:.6f} | {row['valid_pairs']}/12 | "
            f"{row['adjusted_p_value_target_risk']:.6g} | "
            f"beneficial {row['beneficial_flips']}, harmful {row['harmful_flips']} | {row['classification']} |"
        )
    lines.extend([
        "", "## Per-pair Target-risk learned effects", "",
        "| Pair | Component 2883 | Component 4742 | Component 4300 |",
        "|---|---:|---:|---:|",
    ])
    pair_ids = sorted(pair_payload["2883"]["target_risk_effect"])
    for pair_id in pair_ids:
        lines.append(
            f"| {pair_id} | {pair_payload['2883']['target_risk_effect'][pair_id]:.6f} | "
            f"{pair_payload['4742']['target_risk_effect'][pair_id]:.6f} | "
            f"{pair_payload['4300']['target_risk_effect'][pair_id]:.6f} |"
        )
    invalid_scopes = Counter(
        (row["component_id"], row["model_name"], row["condition"], row["arm"])
        for row in invalid_rows
    )
    lines.extend([
        "", "## Validity and decision audit", "",
        f"- Invalid records: `{len(invalid_rows)}`; every one remains in the immutable raw file and the compact invalid-record CSV.",
        f"- Invalid reasons: `{dict(sorted(Counter(str(row['invalid_reason']) for row in invalid_rows).items()))}`.",
        f"- Invalid-record scopes: `{dict(sorted(invalid_scopes.items()))}`.",
        "- The learned-order CSV preserves primary and swapped A/B measurements separately; inference averages them within pair.",
    ])
    for row in summaries:
        lines.append(
            f"- Component {row['component_id']}: beneficial flips `{row['beneficial_flips']}`, "
            f"harmful flips `{row['harmful_flips']}`, net PROCEED change `{row['net_proceed_change']}`."
        )
    lines.extend(["", "## Discovery comparison", ""])
    for row in summaries:
        prior = discovery.get(int(row["component_id"]))
        if prior:
            lines.append(
                f"- Component {row['component_id']}: discovery Target-risk effect "
                f"`{float(prior['exact_target_risk']):.6f}`; held-out `{row['target_risk']:.6f}`."
            )
    lines.extend([
        "", "## Scientific boundary", "",
        "A confirmed result establishes causal influence on this fixed-prefix choice readout. It does not "
        "by itself prove mediation of the original fine-tuning process, semantic specificity outside these "
        "scenarios, or free-generation behavior.", "",
    ])
    report_path = destination / "rccas_stage4_heldout_confirmation.md"
    _atomic_new_text(report_path, "\n".join(lines))
    manifest = {
        "status": "HELDOUT_ANALYSIS_COMPLETE",
        "execution_code_commit": _git_head(),
        "contract_sha256": sha256_file(contract_path),
        "records_sha256": sha256_file(records_path),
        "artifacts": {
            path.name: sha256_file(path)
            for path in (
                stats_csv, summary_json, pair_json, order_csv, invalid_csv, report_path
            )
        },
    }
    _atomic_new_json(destination / "analysis_manifest.json", manifest)
    return {"manifest": manifest, "summaries": summaries}

