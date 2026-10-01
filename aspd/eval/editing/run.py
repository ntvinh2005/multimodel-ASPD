"""Single-feature editing: attribution, the edit sweep over k, and one report per checkpoint."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE
from param_decomp.component_model import ComponentModel
from torch import Tensor, nn

from aspd.eval.adapters.component import target_module_bias
from aspd.eval.editing import measure
from aspd.eval.editing.attribution import (
    AttributionAccumulator,
    analytic_terms,
    autograd_terms,
    gate_and_acts,
    is_rank_one,
    m_raw,
    ranking_order,
)
from aspd.eval.editing.edit import (
    EditSpec,
    component_delta_weight,
    norm_matched_delta,
    random_selection,
)
from aspd.eval.editing.measure import (
    DeltaAccumulator,
    MeasureGroup,
    TokenPlan,
    measure_edit,
)
from aspd.eval.editing.report import AttrEditReport, EditRow, FeatureRow
from aspd.eval.editing.sample import FeatureSample

DEFAULT_TOP_K = (1, 5, 10, 20, 50)


@dataclass(frozen=True)
class AttrEditConfig:
    top_k: tuple[int, ...] = DEFAULT_TOP_K
    n_control_reps: int = 5
    n_tokens_global: int = 200_000
    batch_size: int = 16
    """Sequences per forward. `batch_size * seq_len` also sizes the `[p, F]` encode block, which
    at 16x512 and F=24576 is ~0.8 GB per tensor -- raise it and the encode, not the forward, is
    what runs out of memory.
    """
    seed: int = 42
    sampling: str = "continuous"
    estimator: str = "auto"
    """`auto` picks analytic on a rank-1 arm and autograd everywhere else."""
    cache_module_inputs: bool = True
    """Measure each edit by applying the patched module to its cached input instead of reaching it
    through the whole model. Bitwise identical (see `measure.measure_edit`) and far cheaper: the
    edits stop being that many forwards of the whole stack. `False` restores the hooked forward.
    """
    verify_cached_forward: bool = True
    """Re-measure the first `(feature, k)` cell BOTH ways and require exact agreement. One extra
    forward, which turns the equivalence from an argument into a per-run check.
    """


@dataclass
class Baseline:

    plan: TokenPlan
    sample: FeatureSample
    positions: dict[int, Tensor]
    """`A_j` per sampled feature, ascending flat positions."""
    local_groups: dict[int, MeasureGroup]
    global_group: MeasureGroup
    active: Tensor
    """`[T, n_features]` bool over the flattened plan -- `A_j` as a mask, for the attribution
    pass. 50 MB at GPT2-small's budget, which is cheaper than rebuilding it per chunk.
    """
    baseline_act: dict[int, float]
    x_cache: Tensor | None = None
    """The decomposed module's input for every sequence in the plan, or `None` when
    `cfg.cache_module_inputs` is off. `[n_seq, L, d_in]` on the cache device.
    """


def prepare(
    target_model: nn.Module,
    module_path: str,
    sae: MatryoshkaBatchTopKSAE,
    plan: TokenPlan,
    sample: FeatureSample,
    cfg: AttrEditConfig,
    *,
    device: torch.device | str,
    cache_device: torch.device | str,
) -> Baseline:
    """Stage 1: the frozen-side fixture. No decomposition is loaded here."""
    ids = sample.ids_tensor()
    positions, y_local = measure.collect_positions(
        target_model, module_path, sae, plan, ids.to(device),
        batch_size=cfg.batch_size, device=device, cache_device=cache_device,
    )
    union = torch.unique(torch.cat([p for p in positions.values()]))
    local_groups: dict[int, MeasureGroup] = {}
    for j in sample.feature_ids:
        pos = positions[j]
        assert pos.numel() >= sample.min_support, (
            f"feature {j} has support {pos.numel()} < {sample.min_support}; the sample file was "
            "drawn against a different token budget"
        )
        rows = torch.searchsorted(union, pos)
        local_groups[j] = MeasureGroup(
            name=f"local_f{j}",
            positions=pos,
            targets=torch.tensor([j], dtype=torch.long),
            y_base=y_local[rows],
        )

    global_positions = plan.global_positions(cfg.n_tokens_global)
    global_group = MeasureGroup(
        name="global",
        positions=global_positions,
        targets=ids.clone(),
        y_base=measure.gather_site_activations(
            target_model, module_path, plan, global_positions,
            batch_size=cfg.batch_size, device=device, cache_device=cache_device,
        ),
    )

    n_flat = int(plan.tokens.numel())
    active = torch.zeros(n_flat, len(sample.feature_ids), dtype=torch.bool)
    baseline_act: dict[int, float] = {}
    for i, j in enumerate(sample.feature_ids):
        active[positions[j], i] = True
        with torch.no_grad():
            f_base = sae.features(local_groups[j].y_base.to(device).float())[:, j]
        baseline_act[j] = float(f_base.mean())
    x_cache = None
    if cfg.cache_module_inputs:
        x_cache = measure.collect_module_inputs(
            target_model, module_path, plan,
            batch_size=cfg.batch_size, device=device, cache_device=cache_device,
        )
        xgb = x_cache.numel() * x_cache.element_size() / 2**30
        print(f"[attr_edit] module-input cache {tuple(x_cache.shape)} "
              f"({xgb:.1f} GB on {x_cache.device}); edits skip the model forward", flush=True)
    return Baseline(
        plan=plan, sample=sample, positions=positions, local_groups=local_groups,
        global_group=global_group, active=active, baseline_act=baseline_act, x_cache=x_cache,
    )


def resolve_estimator(cfg: AttrEditConfig, components) -> str:
    if cfg.estimator != "auto":
        return cfg.estimator
    return "analytic" if is_rank_one(components) else "autograd"


def attribution_table(
    model: ComponentModel,
    module_path: str,
    sae: MatryoshkaBatchTopKSAE,
    base: Baseline,
    cfg: AttrEditConfig,
    *,
    estimator: str,
    device: torch.device | str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Stage 2: `(a_gate, a_unit, support)` for this checkpoint, `[n_features, C]` each."""
    plan, sample = base.plan, base.sample
    ids = sample.ids_tensor().to(device)
    components = model.components[module_path]
    n_seq, seq_len = plan.tokens.shape
    acc = AttributionAccumulator(len(sample.feature_ids), components.C, device=device)

    for start in range(0, n_seq, cfg.batch_size):
        chunk = plan.tokens[start : start + cfg.batch_size].to(device)
        lo, hi = start * seq_len, min(start + cfg.batch_size, n_seq) * seq_len
        active = base.active[lo:hi].to(device)
        if not active.any():
            continue
        g, z, x = gate_and_acts(model, module_path, chunk, cfg.sampling)
        if estimator == "analytic":
            gate_term, unit_term = analytic_terms(g.float(), z.float())
            acc.add(
                active,
                gate_term.reshape(-1, components.C),
                unit_term.reshape(-1, components.C),
            )
        else:
            y_true = _true_site_output(model, module_path, x)
            sum_gate, sum_unit = autograd_terms(
                model, module_path, sae,
                x=x, y_true=y_true, g=g, feature_ids=ids,
                active=active.reshape(*chunk.shape, -1),
            )
            acc.add_sums(sum_gate, sum_unit, active.sum(dim=0))

    mean_gate, mean_unit, support = acc.finalize()
    if estimator == "analytic":
        m = m_raw(sae, components.U.detach(), ids).double()
        return mean_gate * m, mean_unit * m, support
    return mean_gate, mean_unit, support


@torch.no_grad()
def _true_site_output(model: ComponentModel, module_path: str, x: Tensor) -> Tensor:
    """`W x + b` from the FROZEN target module -- the clean point the gradient is taken at."""
    weight = model.target_weight(module_path)
    y = x.to(weight.dtype) @ weight.t()
    bias = target_module_bias(model, module_path)
    return y if bias is None else y + bias.to(y.dtype)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, hand-rolled to keep scipy out of the dependency set."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = mean_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else float("nan")


def _jaccard(a: Tensor, b: Tensor) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


@dataclass
class _Sweep:
    """Working state for one checkpoint's edit sweep."""

    rows: list[EditRow] = field(default_factory=list)
    selections: dict[tuple[int, int], Tensor] = field(default_factory=dict)
    norms: dict[tuple[int, int], float] = field(default_factory=dict)


def run_step(
    target_model: nn.Module,
    model: ComponentModel,
    module_path: str,
    sae: MatryoshkaBatchTopKSAE,
    base: Baseline,
    cfg: AttrEditConfig,
    *,
    step: int | None,
    run_dir: Path,
    sae_dir: Path,
    site: str,
    device: torch.device | str,
) -> AttrEditReport:
    """Stages 2 and 3 for one checkpoint."""
    components = model.components[module_path]
    estimator = resolve_estimator(cfg, components)
    ids = base.sample.feature_ids
    print(f"[attr_edit] step {step}: attribution ({estimator}) over "
          f"{base.plan.n_tokens_total:,} tokens", flush=True)
    a_gate, a_unit, support = attribution_table(
        model, module_path, sae, base, cfg, estimator=estimator, device=device
    )

    target_norm = float(model.target_weight(module_path).norm())
    sweep = _Sweep()
    orders = {j: ranking_order(a_gate[i]) for i, j in enumerate(ids)}
    orders_unit = {j: ranking_order(a_unit[i]) for i, j in enumerate(ids)}

    edit_skipped = None if is_rank_one(components) else (
        f"{type(components).__name__} has no rank-1 component weight, so `W - sum_S U_c (x) V_c` "
        "is not this arm's edit. "
        "The attribution table above is architecture-agnostic and is reported; the edit sweep is "
        "not run"
    )
    if edit_skipped:
        print(f"[attr_edit] step {step}: NO EDIT SWEEP — {edit_skipped}", flush=True)

    verify_pending = cfg.verify_cached_forward
    for i, j in enumerate(ids if not edit_skipped else []):
        for k in cfg.top_k:
            selection = orders[j][:k].sort().values
            delta_w = component_delta_weight(components, selection)
            groups = [base.local_groups[j], base.global_group]
            accs = measure_edit(
                target_model, module_path, sae, base.plan, groups, delta_w,
                batch_size=cfg.batch_size, device=device, x_cache=base.x_cache,
            )
            if verify_pending and base.x_cache is not None:
                verify_pending = False
                measure.assert_cached_forward_matches(
                    target_model, module_path, sae, base.plan, groups, delta_w, accs,
                    batch_size=cfg.batch_size, device=device,
                    label=f"step {step}, feature {j}, k={k}",
                )
            sweep.selections[(j, k)] = selection.cpu()
            sweep.norms[(j, k)] = float(delta_w.norm())
            sweep.rows.append(_row(
                EditSpec("ranked", k, j), i, j, selection, delta_w, target_norm,
                a_gate, a_unit, accs, base,
            ))
        print(f"[attr_edit] step {step}: feature {j} ({i + 1}/{len(ids)}) done", flush=True)

    # --- controls: drawn once per (k, rep) and measured against EVERY feature in one pass.
    all_groups = [*base.local_groups.values(), base.global_group]
    for k in (cfg.top_k if not edit_skipped else ()):
        median_norm = float(torch.tensor([sweep.norms[(j, k)] for j in ids]).median())
        for rep in range(cfg.n_control_reps):
            generator = torch.Generator().manual_seed(cfg.seed * 1_000_003 + k * 1009 + rep)
            selection = random_selection(components.C, k, generator)
            base_delta = component_delta_weight(components, selection.to(device))
            scaled = norm_matched_delta(base_delta, median_norm)
            for kind, delta_w, scale in (
                ("random", base_delta, 1.0),
                ("norm_matched", scaled, median_norm / float(base_delta.norm())),
            ):
                accs = measure_edit(
                    target_model, module_path, sae, base.plan, all_groups, delta_w,
                    batch_size=cfg.batch_size, device=device, x_cache=base.x_cache,
                )
                for i, j in enumerate(ids):
                    sweep.rows.append(_row(
                        EditSpec(kind, k, None, rep), i, j, selection, delta_w, target_norm,
                        a_gate, a_unit, accs, base, prediction_scale=scale,
                    ))
        print(f"[attr_edit] step {step}: controls at k={k} done "
              f"(norm target {median_norm:.4g})", flush=True)

    features = [
        FeatureRow(
            feature_id=j,
            density=base.sample.density[i],
            support=int(support[i]),
            baseline_act=base.baseline_act[j],
            top_components=[int(c) for c in orders[j][: max(cfg.top_k)]],
            top_scores=[float(a_gate[i, c]) for c in orders[j][: max(cfg.top_k)]],
            rank_overlap_gate_vs_unit={
                str(k): _jaccard(orders[j][:k], orders_unit[j][:k]) for k in cfg.top_k
            },
        )
        for i, j in enumerate(ids)
    ]
    return AttrEditReport(
        site=site,
        run_dir=str(Path(run_dir).resolve()),
        sae_dir=str(Path(sae_dir).resolve()),
        step=step,
        module=module_path,
        n_components=int(components.C),
        n_latents=int(sae.cfg.n_features),
        estimator=estimator,
        n_tokens=base.plan.n_tokens_total,
        n_tokens_global=base.global_group.n_positions,
        features=features,
        edits=sweep.rows,
        by_k=_aggregate(sweep.rows),
        faithfulness=_faithfulness(sweep.rows),
        selection_overlap={
            str(k): _mean_pairwise_overlap([orders[j][:k] for j in ids]) for k in cfg.top_k
        },
        edit_skipped=edit_skipped,
        meta={
            "config": {**cfg.__dict__, "top_k": list(cfg.top_k)},
            "sample": {
                "path": base.sample.source.get("path", ""),
                "seed": base.sample.seed,
                "max_density": base.sample.max_density,
                "min_support": base.sample.min_support,
                "n_eligible": base.sample.n_eligible,
                "n_density_eligible": base.sample.n_density_eligible,
                "fingerprint": base.sample.fingerprint,
            },
            "target_weight_norm": target_norm,
            "control_norm_targets": {
                str(k): float(torch.tensor([sweep.norms[(j, k)] for j in ids]).median())
                for k in (cfg.top_k if not edit_skipped else ())
            },
        },
    )


def _row(
    spec: EditSpec,
    index: int,
    feature_id: int,
    selection: Tensor,
    delta_w: Tensor,
    target_norm: float,
    a_gate: Tensor,
    a_unit: Tensor,
    accs: dict[str, DeltaAccumulator],
    base: Baseline,
    *,
    prediction_scale: float = 1.0,
) -> EditRow:
    local = accs[f"local_f{feature_id}"]
    glob = accs["global"]
    sel = selection.to(a_gate.device)
    total = float(a_gate[index].abs().sum())
    assert int(glob.targets[index]) == feature_id, (glob.targets[index], feature_id)
    return EditRow(
        kind=spec.kind,
        k=spec.k,
        rep=spec.rep,
        feature_id=feature_id,
        selection=[int(c) for c in selection],
        edit_norm=float(delta_w.norm()),
        edit_norm_rel=float(delta_w.norm()) / target_norm,
        predicted_delta_unit=-prediction_scale * float(a_unit[index, sel].sum()),
        predicted_delta_gate=-prediction_scale * float(a_gate[index, sel].sum()),
        score_share=float(a_gate[index, sel].abs().sum()) / total if total > 0 else 0.0,
        on_aj=local.summarize(0, feature_id),
        on_global=glob.summarize(index, feature_id),
    )


def _aggregate(rows: list[EditRow]) -> dict[str, dict[str, float]]:
    """Mean over features per `(kind, k)`. Controls average over their reps as well."""
    buckets: dict[str, list[EditRow]] = {}
    for row in rows:
        buckets.setdefault(f"{row.kind}/{row.k}", []).append(row)
    out: dict[str, dict[str, float]] = {}
    for key, group in buckets.items():
        summary = {
            "n_rows": float(len(group)),
            "edit_norm": sum(r.edit_norm for r in group) / len(group),
            "edit_norm_rel": sum(r.edit_norm_rel for r in group) / len(group),
            "predicted_delta_unit": sum(r.predicted_delta_unit for r in group) / len(group),
            "predicted_delta_gate": sum(r.predicted_delta_gate for r in group) / len(group),
            "score_share": sum(r.score_share for r in group) / len(group),
        }
        for prefix, attr in (("on_aj_", "on_aj"), ("on_global_", "on_global")):
            for metric in sorted(getattr(group[0], attr)):
                summary[prefix + metric] = sum(
                    getattr(r, attr)[metric] for r in group
                ) / len(group)
        out[key] = summary
    return out


def _faithfulness(rows: list[EditRow]) -> dict[str, float]:
    """Spearman of the first-order prediction against the measured signed change, per `k`."""
    out: dict[str, float] = {}
    ranked = [r for r in rows if r.kind == "ranked"]
    for k in sorted({r.k for r in ranked}):
        at_k = [r for r in ranked if r.k == k]
        out[f"spearman_k{k}"] = _spearman(
            [r.predicted_delta_unit for r in at_k],
            [r.on_aj["delta_signed"] for r in at_k],
        )
        out[f"mean_ratio_k{k}"] = sum(
            r.on_aj["delta_signed"] / r.predicted_delta_unit
            for r in at_k if r.predicted_delta_unit != 0
        ) / max(sum(1 for r in at_k if r.predicted_delta_unit != 0), 1)
    out["spearman_pooled"] = _spearman(
        [r.predicted_delta_unit for r in ranked],
        [r.on_aj["delta_signed"] for r in ranked],
    )
    return out


def _mean_pairwise_overlap(selections: list[Tensor]) -> float:
    """Mean Jaccard between every pair of features' selections at one `k`."""
    pairs = [
        _jaccard(selections[i], selections[j])
        for i in range(len(selections))
        for j in range(i + 1, len(selections))
    ]
    return sum(pairs) / len(pairs) if pairs else 0.0
