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

import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.posthoc.artifacts import (
    StageDirectory,
    load_edge_table,
    require_complete,
    runtime_provenance,
    save_edges,
    sha256_file,
    shard_stage_name,
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
from aspd.multimodel.posthoc.profiles import (
    ComponentProfile,
    DistributionRows,
    GateModelProfile,
    ParentRows,
    dumps,
    edge_profile_from_table,
    shared_top_parent_gates,
)
from aspd.multimodel.posthoc.representatives import representative_sets
from aspd.multimodel.posthoc.source_hooks import capture_matrix_io

FORMULATION = "docs/multimodel/multimodel_aspd_formulation.tex"


def _log(message: str) -> None:
    print(f"[posthoc {time.strftime('%H:%M:%S')}] {message}", flush=True)


class _Timing:
    """Wall-clock and pass counters recorded in stage manifests (used to size full runs)."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.seconds: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, name: str, since: float) -> float:
        now = time.perf_counter()
        self.seconds[name] += now - since
        return now

    def timed_pools(self, pools: Any) -> Any:
        """Yield from ``pools`` while charging the wait (stream + alignment + gates)."""

        iterator = iter(pools)
        while True:
            start = time.perf_counter()
            try:
                pool = next(iterator)
            except StopIteration:
                return
            self.add("pool_preparation", start)
            self.counts["pools"] += 1
            yield pool

    def report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "total_seconds": time.perf_counter() - self.started,
            "seconds": dict(self.seconds),
            "counts": dict(self.counts),
        }
        if torch.cuda.is_available():
            report["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        return report


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
    timing = _Timing()
    pools = backend.pools(int(config.importance_pools), skip=0)  # type: ignore[arg-type]
    for pool in timing.timed_pools(pools):
        anchors += pool.anchor_count
        _log(f"importance pool {timing.counts['pools']}/{config.importance_pools}: "
             f"{pool.anchor_count} anchors")
        gates = {site: pool.gates[site].to(backend.device) for site in SITES}
        for model in backend.model_keys:
            start = time.perf_counter()
            specs = backend.hook_specs(model)
            factors = backend.factors(model)
            captured = capture_matrix_io(
                backend.source_model(model), pool.models[model].to(backend.device), specs
            )
            timing.counts["capture_forwards"] += 1
            for spec in specs:
                x, y = captured[spec.matrix]
                U, V = factors[spec.matrix]
                accumulator = accumulators.setdefault(
                    spec.matrix,
                    MatrixImportanceAccumulator(spec.matrix, backend.n_features, spec.d_out),
                )
                accumulator.update(x.float(), y.float(), gates[spec.matrix.site], U, V)
            del captured
            timing.add("capture_and_accumulate", start)
    table = ImportanceTable(acc.finalize() for acc in accumulators.values())
    table.save(stage.path)
    stage.complete(
        _manifest(
            "importance", config,
            {"backend": backend.provenance(), "model_order": list(backend.model_keys),
             "anchors": anchors, "pools": config.importance_pools,
             "matrices": len(table.entries), "timing": timing.report()},
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
def selected_downstream_gates(config: PosthocConfig, root: Path) -> list[GateId]:
    """The analysed gates, in canonical order (identical for every shard)."""

    table = ImportanceTable.load(require_complete(root, "importance"))
    models, reps = load_representatives(root)
    fire_counts = {site: table.gate_fire_counts(site).tolist() for site in SITES}
    support = {
        gate: sum(1 for model in models if reps.get((model, gate)))
        for gate in {gate for (_model, gate) in reps}
    }
    return [
        gate for gate in config.downstream_gates.resolve(fire_counts, support)
        if support.get(gate, 0) >= 1
    ]


def run_edges(
    config: PosthocConfig, root: Path, backend: PosthocBackend,
    shard: tuple[int, int] | None = None,
) -> Path:
    """Edge scores into every representative of every selected gate at natural anchors.

    With ``shard = (i, n)`` only gates ``i, i+n, i+2n, ...`` of the canonical gate list are
    processed and the output goes to ``edges-shard-i-of-n``; ``finalize_edge_shards``
    then publishes the ``edges`` stage.  Anchor sampling is per gate and per pool, so a
    sharded run selects exactly the anchors of an unsharded one.
    """

    config.require_ready("edges")
    if shard is not None:
        index, count = shard
        if not 0 <= index < count:
            raise ValueError(f"invalid shard {index}/{count}")
    models, reps = load_representatives(root)
    if models != backend.model_keys:
        raise ValueError("representative model order differs from the backend")
    all_gates = selected_downstream_gates(config, root)
    selected_gates = all_gates if shard is None else all_gates[shard[0]::shard[1]]
    quota = {gate: int(config.max_anchors_per_gate) for gate in selected_gates}  # type: ignore[arg-type]
    anchors_by_gate: dict[GateId, list[str]] = defaultdict(list)
    records: list[EdgeRecord] = []
    checks: list[dict[str, Any]] = []
    check_budget = int(config.ablation_check_records)  # type: ignore[arg-type]
    if shard is not None:
        check_budget = math.ceil(check_budget / shard[1])
    write_norms: dict[str, float] = {}
    for model in models:
        factors = backend.factors(model)
        for gate in selected_gates:
            for a in reps.get((model, gate), ()):
                write_norms[a.key] = float(factors[a.matrix][0][a.feature].float().norm())

    pools_used = 0
    skip = config.edge_skip()
    timing = _Timing()
    _log(f"edges: {len(selected_gates)} selected gates"
         + ("" if shard is None else f" (shard {shard[0]}/{shard[1]} of {len(all_gates)})"))
    pools = backend.pools(int(config.edge_pools), skip=skip)  # type: ignore[arg-type]
    for offset, pool in enumerate(timing.timed_pools(pools)):
        if not any(quota.values()):
            break
        pools_used += 1
        rows_by_gate: dict[GateId, list[int]] = {}
        for gate in selected_gates:
            if quota[gate] == 0:
                continue
            rows = pool.gates[gate.site][:, gate.feature].nonzero(as_tuple=True)[0].tolist()
            rows = sample_anchor_rows(rows, quota[gate], config, pool_index=skip + offset,
                                      gate=gate)
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
                start = time.perf_counter()
                new = edge_scores_for_group(source, sub, group_ids, sub_gates, specs, factors,
                                            requests)
                timing.add("edge_forward_backward", start)
                timing.counts["edge_forwards"] += 1
                timing.counts["edge_backwards"] += sum(bool(m.any()) for m in requests.values())
                timing.counts["edge_records"] += len(new)
                records.extend(new)
                for record in new:
                    if len(checks) >= check_budget or not record.upstream:
                        continue
                    top = int(record.scores.abs().argmax())
                    b = record.component(top)
                    row = group_ids.index(record.anchor_id)
                    start = time.perf_counter()
                    exact = exact_ablation_effect(source, sub, sub_gates, specs, factors,
                                                  record.downstream, b)[row]
                    timing.add("ablation_checks", start)
                    timing.counts["ablation_checks"] += 1
                    checks.append({
                        "anchor_id": record.anchor_id, "downstream": record.downstream.key,
                        "upstream": b.key, "taylor": float(record.scores[top]),
                        "exact": float(exact), "coefficient": record.coefficient,
                    })

    _log(f"edges: {len(records)} records, {timing.counts['edge_backwards']} backward passes, "
         f"{timing.seconds['edge_forward_backward']:.1f} s in forward/backward")
    stage = StageDirectory(root, "edges" if shard is None else shard_stage_name(*shard))
    save_edges(stage.path, records)
    write_json(stage.path / "selection.json", {
        "mode": config.downstream_gates.mode,
        "shard": None if shard is None else list(shard),
        "all_selected_gates": [g.key for g in all_gates],
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
        "first_pool_index": skip,
        "shard": None if shard is None else list(shard),
        "timing": timing.report(),
        "conventions": config.recorded_conventions(),
    }))
    return stage.path


def sample_anchor_rows(
    rows: list[int], quota: int, config: PosthocConfig, *, pool_index: int, gate: GateId
) -> list[int]:
    """Pick at most ``quota`` of the firing ``rows`` of one pool for one gate.

    ``first_in_pool`` keeps pool order; ``seeded_random`` draws without replacement
    with a generator seeded by ``(anchor_seed, pool_index, gate)``, so the choice is
    reproducible and independent of which other gates are analysed.
    """

    if len(rows) <= quota:
        return rows
    if config.anchor_sampling == "first_in_pool":
        return rows[:quota]
    if config.anchor_sampling == "seeded_random":
        key = f"{config.anchor_seed}:{pool_index}:{gate.key}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))
        return sorted(rows[i] for i in order[:quota].tolist())
    raise ValueError(f"unknown anchor sampling {config.anchor_sampling!r}")


# ======================================================================= compare
def finalize_edge_shards(config: PosthocConfig, root: Path, count: int) -> Path:
    """Publish the ``edges`` stage from ``count`` completed shards (no data is copied).

    Checks that every shard completed with this config, that all shards saw the same
    canonical gate list, and that their gate subsets are disjoint and cover it.  The
    ``edges`` stage then lists each shard with the SHA-256 of its ``COMPLETE`` file.
    """

    shards = []
    for index in range(count):
        name = shard_stage_name(index, count)
        path = require_complete(root, name)
        manifest = json.loads((path / "manifest.json").read_text())
        selection = json.loads((path / "selection.json").read_text())
        if manifest["config_fingerprint"] != config.fingerprint():
            raise ValueError(f"{name} was produced by a different config")
        if selection["shard"] != [index, count]:
            raise ValueError(f"{name} records shard {selection['shard']}")
        shards.append((name, path, manifest, selection))
    reference = shards[0][3]["all_selected_gates"]
    covered: list[str] = []
    for name, _path, _shard_manifest, selection in shards:
        if selection["all_selected_gates"] != reference:
            raise ValueError(f"{name} saw a different canonical gate list")
        covered.extend(selection["gates"])
    if sorted(covered) != sorted(reference) or len(set(covered)) != len(covered):
        raise ValueError("edge shards do not partition the selected gates")

    merged: dict[str, Any] = {
        "mode": shards[0][3]["mode"], "shard": None, "all_selected_gates": reference,
        "gates": reference, "anchors": {}, "quota_shortfall": {},
        "pools_used": max(s[3]["pools_used"] for s in shards), "downstream_write_norms": {},
    }
    checks: list[dict[str, Any]] = []
    timing: dict[str, Any] = {"seconds": defaultdict(float), "counts": defaultdict(int),
                              "total_seconds": 0.0}
    for _name, path, manifest, selection in shards:
        merged["anchors"].update(selection["anchors"])
        merged["quota_shortfall"].update(selection["quota_shortfall"])
        merged["downstream_write_norms"].update(selection["downstream_write_norms"])
        checks.extend(json.loads((path / "ablation_check.json").read_text())["checks"])
        for key, value in manifest["timing"]["seconds"].items():
            timing["seconds"][key] += value
        for key, value in manifest["timing"]["counts"].items():
            timing["counts"][key] += value
        timing["total_seconds"] += manifest["timing"]["total_seconds"]
    stage = StageDirectory(root, "edges")
    write_json(stage.path / "shards.json", {"shards": [
        {"stage": name, "complete_sha256": sha256_file(path / "COMPLETE")}
        for name, path, _shard_manifest, _selection in shards
    ]})
    write_json(stage.path / "selection.json", merged)
    write_json(stage.path / "ablation_check.json", {
        "description": "Taylor E_{b->a} versus exact ablation, concatenated over shards.",
        "checks": checks,
    })
    stage.complete(_manifest("edges", config, {
        "shards": count,
        "records": sum(int(s[2]["records"]) for s in shards),
        "conventions": shards[0][2]["conventions"],
        "timing_summed_over_shards": {k: dict(v) if isinstance(v, defaultdict) else v
                                      for k, v in timing.items()},
    }))
    return stage.path


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
    edges = load_edge_table(root)
    pairs = config.pairs()
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
                    parents = {}
                    for a in representatives:
                        record = edges.get(anchor_id, a)
                        if record is not None:
                            parents[a] = parent_set(record, rule)
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


def run_profiles(config: PosthocConfig, root: Path, *, top: int = 20) -> Path:
    """Write Pa(a; alpha), component/gate parent profiles and the edge profile; CPU only.

    Recomputes ``Pa`` with the same rule and records as ``compare`` (it reads the same
    verified artifacts), so the profiles are consistent with ``comparisons.jsonl``.
    """

    from safetensors.torch import save_file

    rule = config.parent_rule()
    table = ImportanceTable.load(require_complete(root, "importance"))
    models, reps = load_representatives(root)
    selection = json.loads((require_complete(root, "edges") / "selection.json").read_text())
    write_norms = selection["downstream_write_norms"]
    edges = load_edge_table(root)
    stage = StageDirectory(root, "profiles")
    rows = ParentRows()
    distributions = DistributionRows(models)
    timing = _Timing()
    components_written = gates_written = 0
    with (stage.path / "component_parent_profile.jsonl").open("x", encoding="utf-8") as comp_out, \
            (stage.path / "gate_parent_profile.jsonl").open("x", encoding="utf-8") as gate_out:
        for index, gate_key in enumerate(selection["gates"]):
            gate = GateId.from_key(gate_key)
            anchors = selection["anchors"][gate_key]
            gate_profiles: dict[str, GateModelProfile] = {}
            for model in models:
                representatives = reps.get((model, gate), ())
                norms = {a: write_norms[a.key] for a in representatives}
                sigmas = {a: table[a.matrix].sigma for a in representatives}
                components = {a: ComponentProfile(a, norms[a], sigmas[a]) for a in representatives}
                gate_profile = GateModelProfile(representatives)
                for anchor_id in anchors:
                    parents = {}
                    for a in representatives:
                        record = edges.get(anchor_id, a)
                        if record is None:
                            continue
                        parents[a] = parent_set(record, rule)
                        components[a].add(record.coefficient, parents[a])
                        rows.add(anchor_id, a, parents[a], norms[a], sigmas[a])
                    scores, pi = gate_profile.add(parents, norms, sigmas)
                    distributions.add(gate, model, anchor_id, len(representatives), scores, pi)
                gate_profiles[model] = gate_profile
                for profile in components.values():
                    if profile.anchors:
                        comp_out.write(dumps(profile.to_json(top)) + "\n")
                        components_written += 1
            gate_out.write(dumps({
                "gate": gate_key,
                "anchors": len(anchors),
                "models": {m: p.to_json(top) for m, p in gate_profiles.items()},
                "shared_top_parent_gates": shared_top_parent_gates(gate_profiles, top),
            }) + "\n")
            gates_written += 1
            rows.flush()
            distributions.flush()
            if (index + 1) % 1000 == 0:
                _log(f"profiles: {index + 1}/{len(selection['gates'])} gates")
    timing.add("parent_profiles", timing.started)
    parent_tensors = rows.tensors()
    save_file(parent_tensors, str(stage.path / "parents_per_anchor.safetensors"))
    (stage.path / "parents_per_anchor.safetensors").chmod(0o664)
    write_json(stage.path / "parents_vocab.json", {
        "description": "rows of parents_per_anchor.safetensors: Pa(a; alpha) entries; "
                       "anchor/downstream/parent index anchor_ids/components",
        **rows.vocab(),
    })
    save_file(distributions.tensors(), str(stage.path / "parent_distributions.safetensors"))
    (stage.path / "parent_distributions.safetensors").chmod(0o664)
    write_json(stage.path / "distributions_vocab.json", {
        "description": "header_* rows: one per (gate, model, anchor) with |R|, M and validity; "
                       "entry_* rows: S and pi per parent gate xi (entry_row -> header row; "
                       "pi is NaN when the anchor is invalid). Indices into models/gates/anchor_ids.",
        **distributions.vocab(),
    })
    start = time.perf_counter()
    write_json(stage.path / "edge_profile.json", _finite({
        "description": "edge-score distribution by model, upstream/downstream matrix type "
                       "and layer gap (downstream layer - upstream layer)",
        "all_stored_edges": edge_profile_from_table(edges, models).to_json(),
        "parents_only": rows.edge_profile(models).to_json(),
    }))
    timing.add("edge_profile", start)
    stage.complete(_manifest("profiles", config, {
        "rule": {"k_par": rule.k_par, "tau": rule.tau},
        "top_listed": top,
        "components": components_written,
        "gates": gates_written,
        "parent_rows": int(parent_tensors["E"].numel()),
        "timing": timing.report(),
        "status": "Pa(a;alpha) entries are formulation objects; profiles are descriptive "
                  "aggregates over anchors (ADDED)",
    }))
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
    shard: tuple[int, int] | None = None,
    finalize_shards: int | None = None,
) -> list[Path]:
    """Run ``stages`` in canonical order, building the backend only if needed.

    ``shard = (i, n)`` runs only shard ``i`` of the edge stage (``stages`` must be
    ``("edges",)``).  ``finalize_shards = n`` publishes ``edges`` from ``n`` completed shards
    before any later stage (use with ``stages = ("compare",)`` or alone).
    """

    stages = [s for s in STAGES if s in set(stages)]
    if shard is not None and stages != ["edges"]:
        raise ValueError("--shard runs the edge stage only")
    for stage in stages:  # fail before any compute if a requested stage is blocked
        config.require_ready(stage)
    backend: PosthocBackend | None = None
    outputs = []
    try:
        for stage in stages:
            if stage in ("importance", "edges") and backend is None:
                start = time.perf_counter()
                backend = backend_factory(config, root)
                _log(f"backend {config.backend_kind} ready in {time.perf_counter() - start:.1f} s")
            _log(f"stage {stage} started")
            if stage == "importance":
                outputs.append(run_importance(config, root, backend))  # type: ignore[arg-type]
            elif stage == "representatives":
                outputs.append(run_representatives(config, root, model_order(root)))
            elif stage == "edges":
                outputs.append(run_edges(config, root, backend, shard))  # type: ignore[arg-type]
            else:
                if finalize_shards is not None:
                    outputs.append(finalize_edge_shards(config, root, finalize_shards))
                    finalize_shards = None
                if stage == "compare":
                    outputs.append(run_compare(config, root))
                else:
                    outputs.append(run_profiles(config, root))
        if finalize_shards is not None:
            outputs.append(finalize_edge_shards(config, root, finalize_shards))
    finally:
        if backend is not None:
            backend.close()
    return outputs


def model_order(root: Path) -> tuple[str, ...]:
    """Model order recorded by the importance stage."""

    manifest = json.loads((require_complete(root, "importance") / "manifest.json").read_text())
    return tuple(manifest["model_order"])
