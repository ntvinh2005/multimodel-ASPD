"""Stage orchestration: importance -> representatives -> edges -> compare.

Each ``run_<stage>`` function reads its prerequisites from ``root`` (verified through
``COMPLETE`` hashes), writes one new immutable stage directory and returns its path.
``run_all`` runs the four stages in order with one backend instance.

Only ``importance`` and ``edges`` need the backend (source models, factors, pools).
``representatives`` and ``compare`` are CPU-only functions of earlier artifacts, so
selection and parent rules can be changed later by re-running them into a new root
that symlinks or copies the earlier stages (see the runbook).
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.posthoc.artifacts import (
    StageDirectory,
    load_edges,
    require_complete,
    runtime_provenance,
    save_edges,
    write_json,
)
from aspd.multimodel.posthoc.backend import PosthocBackend
from aspd.multimodel.posthoc.comparison import ModelParentView, compare_all_pairs
from aspd.multimodel.posthoc.config import STAGES, PosthocConfig
from aspd.multimodel.posthoc.edges import (
    EdgeRecord,
    edge_scores_for_group,
    exact_ablation_effect,
    group_rows,
)
from aspd.multimodel.posthoc.ids import SITES, ComponentId, GateId
from aspd.multimodel.posthoc.importance import ImportanceTable, MatrixImportanceAccumulator
from aspd.multimodel.posthoc.parents import parent_set
from aspd.multimodel.posthoc.representatives import representative_sets
from aspd.multimodel.posthoc.source_hooks import capture_matrix_io

FORMULATION = "docs/multimodel/multimodel_aspd_formulation.tex"


def build_backend(config: PosthocConfig, work_dir: Path) -> PosthocBackend:
    """Instantiate the configured backend (imports are lazy: the v3.1 backend is heavy)."""

    if config.backend_kind == "toy":
        from aspd.multimodel.posthoc.toy_backend import ToyBackend

        return ToyBackend(config.backend_options, config.backend_options.get("device", "cpu"))
    if config.backend_kind == "v31_checkpoint":
        from aspd.multimodel.posthoc.v31_backend import V31CheckpointBackend

        return V31CheckpointBackend(config.backend_options, work_dir=work_dir)
    raise ValueError(f"unknown backend {config.backend_kind!r}")


def _manifest(stage: str, config: PosthocConfig, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "experiment_id": config.experiment_id,
        "formulation": FORMULATION,
        "config_path": config.source_path,
        "config_sha256": config.source_sha256,
        "config_fingerprint": config.fingerprint(),
        "runtime": runtime_provenance(),
        **extra,
    }


# ===================================================================== importance
def run_importance(config: PosthocConfig, root: Path, backend: PosthocBackend) -> Path:
    """Stream ``importance_pools`` natural pools and finalize sigma, beta, q, rho."""

    config.require_ready("importance")
    stage = StageDirectory(root, "importance")
    accumulators: dict[Any, MatrixImportanceAccumulator] = {}
    anchors = 0
    for pool in backend.pools(int(config.importance_pools), skip=0):  # type: ignore[arg-type]
        anchors += pool.anchor_count
        gates = {site: pool.gates[site].to(backend.device) for site in SITES}
        for model in backend.model_keys:
            specs = backend.hook_specs(model)
            factors = backend.factors(model)
            captured = capture_matrix_io(
                backend.source_model(model), pool.models[model].to(backend.device), specs
            )
            for spec in specs:
                x, y = captured[spec.matrix]
                U, V = factors[spec.matrix]
                accumulator = accumulators.setdefault(
                    spec.matrix,
                    MatrixImportanceAccumulator(spec.matrix, backend.n_features, spec.d_out),
                )
                accumulator.update(x.float(), y.float(), gates[spec.matrix.site], U, V)
            del captured
    table = ImportanceTable(acc.finalize() for acc in accumulators.values())
    table.save(stage.path)
    stage.complete(
        _manifest(
            "importance", config,
            {"backend": backend.provenance(), "model_order": list(backend.model_keys),
             "anchors": anchors, "pools": config.importance_pools,
             "matrices": len(table.entries)},
        )
    )
    return stage.path


# ================================================================ representatives
def run_representatives(config: PosthocConfig, root: Path, model_keys: Iterable[str]) -> Path:
    """Compute ``R_{p,i}^{(n)}`` for every gate and model; CPU only."""

    rule = config.representative_rule()
    table = ImportanceTable.load(require_complete(root, "importance"))
    model_keys = tuple(model_keys)
    gates = [GateId(site, c) for site in SITES for c in range(table.n_features)]
    sets = representative_sets(table, model_keys, gates, rule)
    payload = {
        "rule": {"metric": rule.metric, "k_rep": rule.k_rep, "tau_rep": rule.tau_rep},
        "model_order": list(model_keys),
        "representatives": {
            gate.key: {model: [c.key for c in sets[(model, gate)]] for model in model_keys}
            for gate in gates
            if any(sets[(model, gate)] for model in model_keys)
        },
    }
    stage = StageDirectory(root, "representatives")
    write_json(stage.path / "representatives.json", payload)
    stage.complete(_manifest("representatives", config, {"gates_with_representatives":
                                                          len(payload["representatives"])}))
    return stage.path


def load_representatives(root: Path) -> tuple[tuple[str, ...], dict[tuple[str, GateId], tuple[ComponentId, ...]]]:
    payload = json.loads(
        (require_complete(root, "representatives") / "representatives.json").read_text()
    )
    models = tuple(payload["model_order"])
    sets: dict[tuple[str, GateId], tuple[ComponentId, ...]] = {}
    for gate_key, by_model in payload["representatives"].items():
        gate = GateId.from_key(gate_key)
        for model in models:
            sets[(model, gate)] = tuple(ComponentId.from_key(k) for k in by_model[model])
    return models, sets


# ========================================================================= edges
def run_edges(config: PosthocConfig, root: Path, backend: PosthocBackend) -> Path:
    """Edge scores into every representative of every selected gate at natural anchors."""

    config.require_ready("edges")
    table = ImportanceTable.load(require_complete(root, "importance"))
    models, reps = load_representatives(root)
    if models != backend.model_keys:
        raise ValueError("representative model order differs from the backend")
    fire_counts = {site: table.gate_fire_counts(site).tolist() for site in SITES}
    selected_gates = [
        gate for gate in config.downstream_gates.resolve(fire_counts)
        if any(reps.get((model, gate)) for model in models)
    ]
    quota = {gate: int(config.max_anchors_per_gate) for gate in selected_gates}  # type: ignore[arg-type]
    anchors_by_gate: dict[GateId, list[str]] = defaultdict(list)
    records: list[EdgeRecord] = []
    checks: list[dict[str, Any]] = []
    check_budget = int(config.ablation_check_records)  # type: ignore[arg-type]
    write_norms: dict[str, float] = {}
    for model in models:
        factors = backend.factors(model)
        for gate in selected_gates:
            for a in reps.get((model, gate), ()):
                write_norms[a.key] = float(factors[a.matrix][0][a.feature].float().norm())

    pools_used = 0
    for pool in backend.pools(int(config.edge_pools), skip=int(config.importance_pools)):  # type: ignore[arg-type]
        if not any(quota.values()):
            break
        pools_used += 1
        rows_by_gate: dict[GateId, list[int]] = {}
        for gate in selected_gates:
            if quota[gate] == 0:
                continue
            rows = pool.gates[gate.site][:, gate.feature].nonzero(as_tuple=True)[0].tolist()
            rows = rows[: quota[gate]]
            if rows:
                rows_by_gate[gate] = rows
                quota[gate] -= len(rows)
                anchors_by_gate[gate].extend(pool.anchor_ids[r] for r in rows)
        needed = sorted({r for rows in rows_by_gate.values() for r in rows})
        if not needed:
            continue
        for model in models:
            source = backend.source_model(model)
            specs = backend.hook_specs(model)
            factors = backend.factors(model)
            inputs = pool.models[model]
            for group in group_rows(inputs.anchor_document, needed,
                                    int(config.anchors_per_backward)):  # type: ignore[arg-type]
                sub = inputs.subset(group).to(backend.device)
                sub_gates = {s: pool.gates[s][group].to(backend.device) for s in SITES}
                requests = {}
                for gate, rows in rows_by_gate.items():
                    mask = torch.tensor([r in rows for r in group], device=backend.device)
                    if bool(mask.any()):
                        for a in reps.get((model, gate), ()):
                            requests[a] = mask
                group_ids = [pool.anchor_ids[r] for r in group]
                new = edge_scores_for_group(source, sub, group_ids, sub_gates, specs, factors,
                                            requests)
                records.extend(new)
                for record in new:
                    if len(checks) >= check_budget or not record.upstream:
                        continue
                    top = int(record.scores.abs().argmax())
                    b = record.upstream[top]
                    row = group_ids.index(record.anchor_id)
                    exact = exact_ablation_effect(source, sub, sub_gates, specs, factors,
                                                  record.downstream, b)[row]
                    checks.append({
                        "anchor_id": record.anchor_id, "downstream": record.downstream.key,
                        "upstream": b.key, "taylor": float(record.scores[top]),
                        "exact": float(exact), "coefficient": record.coefficient,
                    })

    stage = StageDirectory(root, "edges")
    save_edges(stage.path, records)
    write_json(stage.path / "selection.json", {
        "mode": config.downstream_gates.mode,
        "gates": [g.key for g in selected_gates],
        "anchors": {g.key: anchors_by_gate[g] for g in selected_gates},
        "quota_shortfall": {g.key: quota[g] for g in selected_gates if quota[g]},
        "pools_used": pools_used,
        "downstream_write_norms": write_norms,
    })
    write_json(stage.path / "ablation_check.json", {
        "description": "Taylor E_{b->a} versus exact ablation for the strongest parent of "
                       "the first records; diagnostic of the first-order approximation.",
        "checks": checks,
    })
    stage.complete(_manifest("edges", config, {
        "backend": backend.provenance(), "records": len(records),
        "conventions": {
            "jacobian_path": "native_source_model",
            "gates": "frozen_natural_binary_gates",
            "upstream_positions": "same_anchor_token_only",
            "candidate_upstream_set": "all_selected_matrices_upstream_in_compute_order",
        },
    }))
    return stage.path


# ======================================================================= compare
def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def run_compare(config: PosthocConfig, root: Path) -> Path:
    """Parent sets, Jaccard, weighted parent distributions, TV and strength; CPU only."""

    rule = config.parent_rule()
    table = ImportanceTable.load(require_complete(root, "importance"))
    models, reps = load_representatives(root)
    edges_dir = require_complete(root, "edges")
    selection = json.loads((edges_dir / "selection.json").read_text())
    write_norms = selection["downstream_write_norms"]
    by_key: dict[tuple[str, ComponentId], EdgeRecord] = {
        (r.anchor_id, r.downstream): r for r in load_edges(edges_dir)
    }
    pairs = config.model_pairs
    stage = StageDirectory(root, "compare")
    per_gate: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows = 0
    with (stage.path / "comparisons.jsonl").open("x", encoding="utf-8") as handle:
        for gate_key in selection["gates"]:
            gate = GateId.from_key(gate_key)
            for anchor_id in selection["anchors"][gate_key]:
                views = []
                for model in models:
                    representatives = reps.get((model, gate), ())
                    parents = {
                        a: parent_set(by_key[(anchor_id, a)], rule)
                        for a in representatives if (anchor_id, a) in by_key
                    }
                    views.append(ModelParentView(
                        model=model,
                        representatives=representatives,
                        parents=parents,
                        write_norms={a: write_norms[a.key] for a in representatives},
                        sigmas={a: table[a.matrix].sigma for a in representatives},
                    ))
                for comparison in compare_all_pairs(gate, anchor_id, views, pairs):
                    handle.write(json.dumps(comparison.to_json(), sort_keys=True) + "\n")
                    rows += 1
                    pair = f"{comparison.left}|{comparison.right}"
                    if comparison.jaccard is not None:
                        per_gate[gate_key][f"{pair}/jaccard"].append(comparison.jaccard)
                    if comparison.total_variation is not None:
                        per_gate[gate_key][f"{pair}/tv"].append(comparison.total_variation)
    summary = {
        "status": "DESCRIPTIVE_ANCHOR_MEANS_NOT_A_VALIDATED_STATISTIC",
        "rule": {"k_par": rule.k_par, "tau": rule.tau},
        "comparisons": rows,
        "per_gate": {
            gate: {name: {"mean": _mean(v), "n": len(v)} for name, v in sorted(stats.items())}
            for gate, stats in sorted(per_gate.items())
        },
    }
    for name in sorted({n for stats in per_gate.values() for n in stats}):
        values = [x for stats in per_gate.values() for x in stats.get(name, [])]
        summary.setdefault("overall", {})[name] = {"mean": _mean(values), "n": len(values)}
    write_json(stage.path / "summary.json", _finite(summary))
    stage.complete(_manifest("compare", config, {"comparisons": rows}))
    return stage.path


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


# ========================================================================= driver
def run_stages(
    config: PosthocConfig,
    root: Path,
    stages: Iterable[str],
    *,
    backend_factory: Callable[[PosthocConfig, Path], PosthocBackend] = build_backend,
) -> list[Path]:
    """Run ``stages`` in canonical order, building the backend only if needed."""

    stages = [s for s in STAGES if s in set(stages)]
    for stage in stages:  # fail before any compute if a requested stage is blocked
        config.require_ready(stage)
    backend: PosthocBackend | None = None
    outputs = []
    try:
        for stage in stages:
            if stage in ("importance", "edges") and backend is None:
                backend = backend_factory(config, root)
            if stage == "importance":
                outputs.append(run_importance(config, root, backend))  # type: ignore[arg-type]
            elif stage == "representatives":
                outputs.append(run_representatives(config, root, model_order(root)))
            elif stage == "edges":
                outputs.append(run_edges(config, root, backend))  # type: ignore[arg-type]
            else:
                outputs.append(run_compare(config, root))
    finally:
        if backend is not None:
            backend.close()
    return outputs


def model_order(root: Path) -> tuple[str, ...]:
    """Model order recorded by the importance stage."""

    manifest = json.loads((require_complete(root, "importance") / "manifest.json").read_text())
    return tuple(manifest["model_order"])
