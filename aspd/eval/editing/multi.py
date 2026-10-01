"""Multiple-feature editing: target sets J of size m, the union of per-feature top-k components."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE
from param_decomp.component_model import ComponentModel
from torch import Tensor, nn

from aspd.eval.editing import measure
from aspd.eval.editing.attribution import (
    AttributionAccumulator,
    GlobalAccumulator,
    analytic_terms,
    autograd_terms,
    gate_and_acts,
    is_rank_one,
    jaccard,
    m_raw,
    ranking_order,
)
from aspd.eval.editing.edit import (
    component_delta_weight,
    norm_matched_delta,
    random_selection,
)
from aspd.eval.editing.measure import (
    DeltaAccumulator,
    MeasureGroup,
    TokenPlan,
    assert_cached_forward_matches,
    measure_edit,
)
from aspd.eval.editing.report import MultiAttrEditReport, MultiEditRow
from aspd.eval.editing.run import _true_site_output
from aspd.eval.editing.sample import FeaturePool

_BASELINE_ACT_CHUNK = 8192
DEFAULT_TOP_K_MULTI = (1, 5, 10)
DEFAULT_TOP_K_SINGLE = (1, 5, 10, 20, 50)
SETUPS = ("cond", "global")


def default_top_k(m: int) -> tuple[int, ...]:
    return DEFAULT_TOP_K_SINGLE if m == 1 else DEFAULT_TOP_K_MULTI


def default_control_combos(m: int, n_combinations: int) -> int:
    return min(n_combinations, max(10, 500 // m))


@dataclass(frozen=True)
class MultiConfig:
    m: int
    setup: str = "cond"
    top_k: tuple[int, ...] = DEFAULT_TOP_K_MULTI
    n_combinations: int = 50
    n_control_combos: int = 0
    """0 means `default_control_combos(m)`."""
    n_control_reps: int = 5
    n_tokens_global: int = 200_000
    batch_size: int = 16
    seed: int = 42
    sampling: str = "continuous"
    estimator: str = "auto"
    cache_module_inputs: bool = True
    """Measure each edit by applying the patched module to its cached input instead of reaching it
    through the whole model. Bitwise identical (see `measure.measure_edit`) and ~60x cheaper: the
    ~180 edits per arm stop being ~180 forwards of an 8B stack. `False` restores the hooked
    forward, which is what `verify_cached_forward` compares against.
    """
    verify_cached_forward: bool = True

    def __post_init__(self) -> None:
        assert self.setup in SETUPS, (self.setup, SETUPS)
        assert self.top_k == tuple(sorted(self.top_k)), self.top_k

    @property
    def control_combos(self) -> int:
        return self.n_control_combos or default_control_combos(self.m, self.n_combinations)

    def control_seed(self, k: int, rep: int) -> int:
        return self.seed * 1_000_003 + k * 1009 + rep + (self.m - 1) * 10007


@dataclass
class Combination:
    """One `J_i`: its members, their rows in the arm's feature list, and its union group."""

    index: int
    feature_ids: list[int]
    rows: list[int]
    positions: Tensor
    group: MeasureGroup


@dataclass
class MultiBaseline:

    plan: TokenPlan
    pool: FeaturePool
    features: list[int]
    """Distinct features this arm touches, ascending. Up to `n_combinations * m`, capped by the
    pool -- the baseline cache is built over these and nothing else.
    """
    row_of: dict[int, int]
    positions: dict[int, Tensor]
    feature_groups: dict[int, MeasureGroup]
    combinations: list[Combination]
    global_group: MeasureGroup
    active: Tensor
    baseline_act: dict[int, float]
    x_cache: Tensor | None = None
    """The decomposed module's input for every sequence in the plan, or `None` when
    `cfg.cache_module_inputs` is off. `[n_seq, L, d_in]` on the cache device.
    """


def prepare_multi(
    target_model: nn.Module,
    module_path: str,
    sae: MatryoshkaBatchTopKSAE,
    plan: TokenPlan,
    pool: FeaturePool,
    combos: list[list[int]],
    cfg: MultiConfig,
    *,
    device: torch.device | str,
    cache_device: torch.device | str,
) -> MultiBaseline:
    features = sorted({j for combo in combos for j in combo})
    row_of = {j: i for i, j in enumerate(features)}
    ids = torch.tensor(features, dtype=torch.long)
    positions, y_cache = measure.collect_positions(
        target_model, module_path, sae, plan, ids.to(device),
        batch_size=cfg.batch_size, device=device, cache_device=cache_device,
    )
    union = torch.unique(torch.cat([positions[j] for j in features]))
    gb = union.numel() * sae.cfg.d_in * 4 / 2**30
    print(f"[attr_edit_multi] {len(features)} distinct features · "
          f"{int(union.numel()):,} cached baseline positions ({gb:.1f} GB fp32)", flush=True)

    def group(name: str, pos: Tensor, targets: list[int]) -> MeasureGroup:
        return MeasureGroup(
            name=name, positions=pos, targets=torch.tensor(targets, dtype=torch.long),
            y_base=y_cache, cache_rows=torch.searchsorted(union, pos),
        )

    feature_groups = {}
    for j in features:
        assert positions[j].numel() >= pool.min_support, (
            f"feature {j} has support {positions[j].numel()} < {pool.min_support}; the pool file "
            "was drawn against a different token budget"
        )
        feature_groups[j] = group(f"f{j}", positions[j], [j])

    combinations = []
    for i, combo in enumerate(combos):
        pos = torch.unique(torch.cat([positions[j] for j in combo]))
        combinations.append(Combination(
            index=i, feature_ids=list(combo), rows=[row_of[j] for j in combo],
            positions=pos, group=group(f"c{i}", pos, list(combo)),
        ))

    global_positions = plan.global_positions(cfg.n_tokens_global)
    global_group = MeasureGroup(
        name="global", positions=global_positions, targets=ids.clone(),
        y_base=measure.gather_site_activations(
            target_model, module_path, plan, global_positions,
            batch_size=cfg.batch_size, device=device, cache_device=cache_device,
        ),
    )

    active = torch.zeros(int(plan.tokens.numel()), len(features), dtype=torch.bool)
    baseline_act = {}
    for i, j in enumerate(features):
        active[positions[j], i] = True
        rows = torch.searchsorted(union, positions[j])
        with torch.no_grad():
            total, seen = 0.0, 0
            for start in range(0, int(rows.numel()), _BASELINE_ACT_CHUNK):
                block = y_cache[rows[start : start + _BASELINE_ACT_CHUNK]].to(device).float()
                total += float(sae.features(block)[:, j].double().sum())
                seen += int(block.shape[0])
            assert seen == int(rows.numel()), (seen, int(rows.numel()))
            baseline_act[j] = total / seen
    x_cache = None
    if cfg.cache_module_inputs:
        x_cache = measure.collect_module_inputs(
            target_model, module_path, plan,
            batch_size=cfg.batch_size, device=device, cache_device=cache_device,
        )
        xgb = x_cache.numel() * x_cache.element_size() / 2**30
        print(f"[attr_edit_multi] module-input cache {tuple(x_cache.shape)} "
              f"({xgb:.1f} GB on {x_cache.device}); edits skip the model forward", flush=True)
    return MultiBaseline(
        plan=plan, pool=pool, features=features, row_of=row_of, positions=positions,
        feature_groups=feature_groups, combinations=combinations, global_group=global_group,
        active=active, baseline_act=baseline_act, x_cache=x_cache,
    )


def attribution_tables(
    model: ComponentModel,
    module_path: str,
    sae: MatryoshkaBatchTopKSAE,
    base: MultiBaseline,
    cfg: MultiConfig,
    *,
    estimator: str,
    want_global: bool,
    device: torch.device | str,
) -> dict[str, Tensor]:
    """`cond` always, `global` when the analytic factorization exists -- one pass for both."""
    plan = base.plan
    ids = torch.tensor(base.features, dtype=torch.long, device=device)
    components = model.components[module_path]
    n_seq, seq_len = plan.tokens.shape
    acc = AttributionAccumulator(len(base.features), components.C, device=device)
    gacc = GlobalAccumulator(components.C, device=device) if want_global else None

    for start in range(0, n_seq, cfg.batch_size):
        chunk = plan.tokens[start : start + cfg.batch_size].to(device)
        lo, hi = start * seq_len, min(start + cfg.batch_size, n_seq) * seq_len
        active = base.active[lo:hi].to(device)
        if gacc is None and not active.any():
            continue
        g, z, x = gate_and_acts(model, module_path, chunk, cfg.sampling)
        if estimator == "analytic":
            gate_term, unit_term = analytic_terms(g.float(), z.float())
            gate_term = gate_term.reshape(-1, components.C)
            unit_term = unit_term.reshape(-1, components.C)
            if active.any():
                acc.add(active, gate_term, unit_term)
            if gacc is not None:
                keep = plan.keep[start : start + cfg.batch_size].reshape(-1).to(device)
                gacc.add(keep, gate_term, unit_term)
        elif active.any():
            sum_gate, sum_unit = autograd_terms(
                model, module_path, sae,
                x=x, y_true=_true_site_output(model, module_path, x), g=g, feature_ids=ids,
                active=active.reshape(*chunk.shape, -1),
            )
            acc.add_sums(sum_gate, sum_unit, active.sum(dim=0))

    mean_gate, mean_unit, support = acc.finalize()
    out = {"support": support}
    if estimator == "analytic":
        m = m_raw(sae, components.U.detach(), ids).double()
        out |= {"cond_gate": mean_gate * m, "cond_unit": mean_unit * m}
        if gacc is not None:
            g_gate, g_unit, _ = gacc.finalize()
            out |= {"global_gate": m * g_gate[None, :], "global_unit": m * g_unit[None, :]}
    else:
        out |= {"cond_gate": mean_gate, "cond_unit": mean_unit}
    return out


def _median(values: list[float]) -> float:
    return float(torch.tensor(values, dtype=torch.float64).median())


@dataclass
class _Sweep:
    rows: list[MultiEditRow] = field(default_factory=list)
    selections: dict[tuple[int, int], Tensor] = field(default_factory=dict)
    norms: dict[tuple[int, int], float] = field(default_factory=dict)


def run_step_multi(
    target_model: nn.Module,
    model: ComponentModel,
    module_path: str,
    sae: MatryoshkaBatchTopKSAE,
    base: MultiBaseline,
    cfg: MultiConfig,
    *,
    step: int | None,
    run_dir: Path,
    sae_dir: Path,
    site: str,
    device: torch.device | str,
) -> MultiAttrEditReport:
    components = model.components[module_path]
    rank_one = is_rank_one(components)
    estimator = cfg.estimator if cfg.estimator != "auto" else (
        "analytic" if rank_one else "autograd"
    )
    if not rank_one:
        global_skipped = (
            f"{type(components).__name__} has no rank-1 form, so `df_j/dm_c = L_j z_c M[j,c]` does "
            "not factorize and the global token set has no estimator: off A_j the true derivative "
            "of a JumpReLU feature is identically zero. The cond table below is "
            "architecture-agnostic"
        )
    elif estimator != "analytic":
        global_skipped = (
            "--estimator autograd was forced on a rank-1 arm. The global token set exists only "
            "through the analytic factorization; autograd would return zeros off A_j, which is "
            "the CORRECT derivative of f_j and the wrong quantity for the global setup"
        )
    else:
        global_skipped = None
    edit_skipped = None if rank_one else (
        f"{type(components).__name__} has no rank-1 component weight, so `W - sum_S U_c (x) V_c` "
        "is not this arm's edit. The attribution table is reported; the edit sweep is not run"
    )
    want_global = global_skipped is None
    if cfg.setup == "global" and global_skipped:
        assert edit_skipped, (
            f"--setup global cannot run here and the edit sweep still can, so this job would "
            f"write COND results into the global arm's directory. {global_skipped}"
        )
        print(f"[attr_edit_multi] step {step}: NO GLOBAL SETUP — {global_skipped}", flush=True)

    print(f"[attr_edit_multi] step {step}: attribution ({estimator}, setup={cfg.setup}) over "
          f"{base.plan.n_tokens_total:,} tokens, {len(base.features)} features", flush=True)
    tables = attribution_tables(
        model, module_path, sae, base, cfg,
        estimator=estimator, want_global=want_global, device=device,
    )
    rank_key = "global_gate" if (cfg.setup == "global" and not global_skipped) else "cond_gate"
    a_rank, a_unit, a_gate = tables[rank_key], tables["cond_unit"], tables["cond_gate"]
    orders = {j: ranking_order(a_rank[i]) for j, i in base.row_of.items()}

    target_norm = float(model.target_weight(module_path).norm())
    sweep = _Sweep()
    verify_pending = cfg.verify_cached_forward
    n_control = cfg.control_combos
    if n_control < cfg.n_combinations:
        print(f"[attr_edit_multi] controls are measured against the FIRST {n_control} of "
              f"{cfg.n_combinations} combinations (m={cfg.m}); ranked edits use all of them",
              flush=True)

    # --- ranked edits: one per (combination, k).
    for combo in (base.combinations if not edit_skipped else []):
        for k in cfg.top_k:
            selection = torch.unique(torch.cat([orders[j][:k] for j in combo.feature_ids]))
            delta_w = component_delta_weight(components, selection)
            groups = [combo.group, *_member_groups(base, combo.feature_ids, cfg.m),
                      base.global_group]
            accs = measure_edit(
                target_model, module_path, sae, base.plan, groups, delta_w,
                batch_size=cfg.batch_size, device=device, x_cache=base.x_cache,
            )
            if verify_pending and base.x_cache is not None:
                verify_pending = False
                assert_cached_forward_matches(
                    target_model, module_path, sae, base.plan, groups, delta_w, accs,
                    batch_size=cfg.batch_size, device=device,
                    label=f"step {step}, combination {combo.index}, k={k}",
                )
            sweep.selections[(combo.index, k)] = selection.cpu()
            sweep.norms[(combo.index, k)] = float(delta_w.norm())
            sweep.rows += _rows_for(
                base, cfg, combo, selection, delta_w, target_norm, k,
                a_rank, a_unit, a_gate, accs, kind="ranked", rep=0,
            )
        print(f"[attr_edit_multi] step {step}: combination {combo.index + 1}/"
              f"{len(base.combinations)} done", flush=True)

    # --- controls: one draw per (k, rep), measured against the first `n_control` combinations.
    control_combos = base.combinations[:n_control]
    for k in (cfg.top_k if not edit_skipped else ()):
        sizes = [int(sweep.selections[(c.index, k)].numel()) for c in base.combinations]
        median_size = int(_median([float(s) for s in sizes]))
        median_norm = _median([sweep.norms[(c.index, k)] for c in base.combinations])
        members = sorted({j for c in control_combos for j in c.feature_ids})
        groups = [c.group for c in control_combos] + \
            _member_groups(base, members, cfg.m) + [base.global_group]
        for rep in range(cfg.n_control_reps):
            generator = torch.Generator().manual_seed(cfg.control_seed(k, rep))
            selection = random_selection(components.C, median_size, generator)
            base_delta = component_delta_weight(components, selection.to(device))
            scaled = norm_matched_delta(base_delta, median_norm)
            for kind, delta_w, scale in (
                ("random", base_delta, 1.0),
                ("norm_matched", scaled, median_norm / float(base_delta.norm())),
            ):
                accs = measure_edit(
                    target_model, module_path, sae, base.plan, groups, delta_w,
                    batch_size=cfg.batch_size, device=device, x_cache=base.x_cache,
                )
                for combo in control_combos:
                    sweep.rows += _rows_for(
                        base, cfg, combo, selection, delta_w, target_norm, k,
                        a_rank, a_unit, a_gate, accs, kind=kind, rep=rep,
                        prediction_scale=scale,
                    )
        print(f"[attr_edit_multi] step {step}: controls at k={k} done "
              f"(|S| {median_size}, norm target {median_norm:.4g})", flush=True)

    return MultiAttrEditReport(
        site=site,
        run_dir=str(Path(run_dir).resolve()),
        sae_dir=str(Path(sae_dir).resolve()),
        step=step,
        module=module_path,
        m=cfg.m,
        setup=cfg.setup,
        n_combinations=cfg.n_combinations,
        n_control_combos=n_control,
        n_components=int(components.C),
        n_latents=int(sae.cfg.n_features),
        estimator=estimator,
        n_tokens=base.plan.n_tokens_total,
        n_tokens_global=base.global_group.n_positions,
        combinations=[c.feature_ids for c in base.combinations],
        edits=sweep.rows,
        by_k=_aggregate(sweep.rows),
        faithfulness=_faithfulness(sweep.rows),
        structure=_structure(base, cfg, orders, tables),
        edit_skipped=edit_skipped,
        global_skipped=global_skipped if cfg.setup == "global" else None,
        meta={
            "config": {**cfg.__dict__, "top_k": list(cfg.top_k)},
            "pool": {
                "path": base.pool.source.get("path", ""),
                "size": len(base.pool.feature_ids),
                "n_seed_features": base.pool.n_seed_features,
                "seed": base.pool.seed,
                "max_density": base.pool.max_density,
                "min_support": base.pool.min_support,
                "n_eligible": base.pool.n_eligible,
                "fingerprint": base.pool.fingerprint,
            },
            "n_features_touched": len(base.features),
            "target_weight_norm": target_norm,
            "baseline_act": {str(j): v for j, v in base.baseline_act.items()},
            "support": {str(j): int(base.positions[j].numel()) for j in base.features},
            "control_sizes": {
                str(k): int(_median([float(sweep.selections[(c.index, k)].numel())
                                     for c in base.combinations]))
                for k in (cfg.top_k if not edit_skipped else ())
            },
            "control_norm_targets": {
                str(k): _median([sweep.norms[(c.index, k)] for c in base.combinations])
                for k in (cfg.top_k if not edit_skipped else ())
            },
        },
    )


def _member_groups(base: MultiBaseline, features: list[int], m: int) -> list[MeasureGroup]:
    """The per-target groups. Empty at `m = 1`, where the union group IS the member's group."""
    return [] if m == 1 else [base.feature_groups[j] for j in features]


def _rows_for(
    base: MultiBaseline,
    cfg: MultiConfig,
    combo: Combination,
    selection: Tensor,
    delta_w: Tensor,
    target_norm: float,
    k: int,
    a_rank: Tensor,
    a_unit: Tensor,
    a_gate: Tensor,
    accs: dict[str, DeltaAccumulator],
    *,
    kind: str,
    rep: int,
    prediction_scale: float = 1.0,
) -> list[MultiEditRow]:
    sel = selection.to(a_rank.device)
    rows, ids = combo.rows, combo.feature_ids
    glob = accs["global"]
    g_rows = [base.row_of[j] for j in ids]
    n_union = float(combo.positions.numel())
    weights = [float(base.positions[j].numel()) / n_union for j in ids]

    def predict(table: Tensor, per_row_weight: list[float]) -> float:
        return -prediction_scale * sum(
            w * float(table[r, sel].sum()) for r, w in zip(rows, per_row_weight, strict=True)
        )

    total_rank = float(a_rank[rows].abs().sum())
    common = dict(
        kind=kind, k=k, rep=rep, combo=combo.index, m=cfg.m,
        selection_size=int(selection.numel()),
        sharing_rate=1.0 - int(selection.numel()) / (cfg.m * k),
        edit_norm=float(delta_w.norm()),
        edit_norm_rel=float(delta_w.norm()) / target_norm,
    )
    out = [MultiEditRow(
        **common, block="union", feature_id=None,
        predicted_delta_unit=predict(a_unit, weights),
        predicted_delta_gate=predict(a_gate, weights),
        score_share=float(a_rank[rows][:, sel].abs().sum()) / total_rank if total_rank > 0 else 0.0,
        on_a=accs[combo.group.name].summarize_group(list(range(len(ids))), ids),
        on_global=glob.summarize_group(g_rows, ids),
        selection=[int(c) for c in selection],
    )]
    if cfg.m == 1:
        return out
    for j, r, g_row in zip(ids, rows, g_rows, strict=True):
        own = float(a_rank[r].abs().sum())
        out.append(MultiEditRow(
            **common, block="target", feature_id=j,
            predicted_delta_unit=-prediction_scale * float(a_unit[r, sel].sum()),
            predicted_delta_gate=-prediction_scale * float(a_gate[r, sel].sum()),
            score_share=float(a_rank[r, sel].abs().sum()) / own if own > 0 else 0.0,
            on_a=accs[f"f{j}"].summarize(0, j, exclude=ids),
            on_global=glob.summarize(g_row, j, exclude=ids),
        ))
    return out


def _aggregate(rows: list[MultiEditRow]) -> dict[str, dict[str, float]]:
    """Mean over combinations per `(kind, k, block)`. Controls average over their reps as well."""
    buckets: dict[str, list[MultiEditRow]] = {}
    for row in rows:
        buckets.setdefault(f"{row.kind}/{row.k}/{row.block}", []).append(row)
    return {key: _bucket_means(group) for key, group in buckets.items()}


def _bucket_means(group: list[MultiEditRow]) -> dict[str, float]:
    """Every scalar on the row, plus both token-set dicts flattened under their own prefix."""
    n = len(group)
    summary = {"n_rows": float(n)}
    for name in ("selection_size", "sharing_rate", "edit_norm", "edit_norm_rel",
                 "predicted_delta_unit", "predicted_delta_gate", "score_share"):
        summary[name] = sum(getattr(row, name) for row in group) / n
    for prefix, attr in (("on_a_", "on_a"), ("on_global_", "on_global")):
        for metric in sorted(getattr(group[0], attr)):
            summary[prefix + metric] = sum(getattr(row, attr)[metric] for row in group) / n
    return summary


def _faithfulness(rows: list[MultiEditRow]) -> dict[str, float]:
    from aspd.eval.editing.run import _spearman

    out: dict[str, float] = {}
    ranked = [r for r in rows if r.kind == "ranked" and r.block == "union"]
    for k in sorted({r.k for r in ranked}):
        at_k = [r for r in ranked if r.k == k]
        out[f"spearman_k{k}"] = _spearman(
            [r.predicted_delta_unit for r in at_k], [r.on_a["delta_signed"] for r in at_k]
        )
        nonzero = [r for r in at_k if r.predicted_delta_unit != 0]
        out[f"mean_ratio_k{k}"] = sum(
            r.on_a["delta_signed"] / r.predicted_delta_unit for r in nonzero
        ) / max(len(nonzero), 1)
    out["spearman_pooled"] = _spearman(
        [r.predicted_delta_unit for r in ranked], [r.on_a["delta_signed"] for r in ranked]
    )
    return out


def _structure(
    base: MultiBaseline,
    cfg: MultiConfig,
    orders: dict[int, Tensor],
    tables: dict[str, Tensor],
) -> dict[str, dict[str, float]]:
    unit_orders = {j: ranking_order(tables["cond_unit"][i]) for j, i in base.row_of.items()}
    cond_orders = {j: ranking_order(tables["cond_gate"][i]) for j, i in base.row_of.items()}
    glob_orders = (
        {j: ranking_order(tables["global_gate"][i]) for j, i in base.row_of.items()}
        if "global_gate" in tables else {}
    )
    out: dict[str, dict[str, float]] = {}
    for k in cfg.top_k:
        picked = [
            torch.unique(torch.cat([orders[j][:k] for j in c.feature_ids]))
            for c in base.combinations
        ]
        entry = {
            "selection_size": _mean([float(s.numel()) for s in picked]),
            "sharing_rate": _mean([1.0 - s.numel() / (cfg.m * k) for s in picked]),
            "selection_overlap": _mean([
                jaccard(picked[a], picked[b])
                for a in range(len(picked)) for b in range(a + 1, len(picked))
            ]),
            "rank_overlap_gate_vs_unit": _mean(
                [jaccard(orders[j][:k], unit_orders[j][:k]) for j in base.features]
            ),
        }
        if glob_orders:
            entry["rank_overlap_cond_vs_global"] = _mean(
                [jaccard(cond_orders[j][:k], glob_orders[j][:k]) for j in base.features]
            )
        out[str(k)] = entry
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
