# Multimodel ASPD post-hoc analysis: edge scores and parent-set comparison

Status (2026-11-21)

| Surface | Status |
|---|---|
| Implementation (`aspd/multimodel/posthoc/`) | `LOCAL_CODE_READY` |
| Toy end-to-end pipeline (CPU) | `SYNTHETIC_TESTED` |
| v3.1 checkpoint backend | `REAL_CHECKPOINT_SMOKE_PASS` (job 52378170, 2026-11-22) |
| Scientific decisions | `UNRESOLVED` (see [Open decisions](#open-decisions)) |
| Results | none — nothing has been run on a real checkpoint |

Mathematics: [`multimodel_aspd_formulation.tex`](multimodel_aspd_formulation.tex), section
"Post-hoc analysis". Canonical training notation:
[`aspd_v3_1_mathematical_specification.md`](aspd_v3_1_mathematical_specification.md).

## 1. Quick start

```bash
# from the worktree root; one thread avoids contention on many-core login nodes
export PYTHONPATH=$PWD OMP_NUM_THREADS=1
PY=/blue/thai/share/multimodel-ASPD/repo/.venv/bin/python

# (a) what is still blocked in a config (no compute)
$PY -m aspd.multimodel.cli.posthoc_analysis --config configs/multimodel/posthoc/v31_500m_template.yaml --check

# (b) full synthetic run, ~10 s on CPU
$PY -m aspd.multimodel.cli.posthoc_analysis \
    --config configs/multimodel/posthoc/toy_smoke.yaml --output-dir out/posthoc/toy_$(date +%Y%m%d_%H%M%S)

# (c) tests
$PY -m pytest -q tests/test_multimodel_posthoc_*.py
```

Full v3.1 run on 2 GPUs. This submits three dependent jobs: importance and
representatives on 1 GPU; the edge stage as 8 gate shards, at most 2 at a time; then a
CPU-only job that publishes the shards and runs compare.

```bash
CONFIG=configs/multimodel/posthoc/v31_500m_validation_v1.yaml SHARDS=8 PARALLEL=2 \
  EDGE_TIME=05:00:00 bash slurm/multimodel/posthoc_submit_sharded.sh
```

A shard processes gates `i, i+N, …` of the canonical gate list and writes
`edges-shard-i-of-N/`. `--finalize-shards N` checks that the shards partition the gates
and come from the same config, then writes `edges/` as an index of the shards; no edge
data is copied. Anchors are sampled per gate and pool, so a sharded run analyses exactly
the anchors of an unsharded one. Scores differ only by float32 rounding from different
batching. This is tested.

Stages can be run one at a time (`--stage importance`, repeatable). Each stage writes an
immutable `<root>/<stage>/` directory with `manifest.json` (git SHA, config SHA-256,
backend provenance, Slurm job ID) and a `COMPLETE` file of SHA-256 hashes; later stages
refuse to read a prerequisite whose hashes changed. A stage directory is never
overwritten — use a new root.

To try a different parent rule (`K_par`, `tau`) or comparison without recomputing
gradients, copy (`cp -r`) `importance/`, `representatives/` and `edges/` into a new root
and run `--stage compare` with the new config. Changing `K_rep`/`tau_rep`/metric needs
`representatives` and `edges` to be recomputed.

## 2. Pipeline

```
importance        backend: source models + factors + natural pools   -> sigma_j, beta, q, rho
representatives   CPU: importance                                    -> R_{p,i}^{(n)} for every gate
edges             backend: models + factors + new natural pools      -> E_{b->a}(alpha), ablation check
compare           CPU: importance + representatives + edges          -> J, D_TV, M, G per anchor/pair
profiles          CPU: importance + representatives + edges          -> Pa(a;alpha) with E,w; component,
                                                                        gate (mean pi, M) and edge profiles
```

`importance` uses pools `[0, importance_pools)` of the backend's deterministic pool
sequence. With `edge_anchor_source: disjoint_after_importance`, `edges` uses the following
`edge_pools` pools, so the two stages never share anchors even when run as separate jobs.
With `same_as_importance`, it starts again at pool 0.

## 3. Equation-to-code map

Short form below. The full walk through the `.tex` — every definition, its exact code location, EXACT/INTERPRETED/ADDED status and the notation differences — is in [`posthoc_formulation_map.md`](posthoc_formulation_map.md).

| Formulation | Code |
|---|---|
| `p(j)`, compute order, "b upstream of a" | `ids.MATRIX_SITE`, `ids.MatrixId.is_upstream_of` |
| `e_{j,α,i} = g_{p(j),α,i} (v^T x)` | `contributions.component_coefficients` |
| `Δy = e u`, `‖Δy‖²` | `contributions.component_contributions`, `contribution_sq_norms` |
| `σ_j`, `β_{j,i}`, `q_{j,i}`, `ρ_{j,i}` | `importance.MatrixImportanceAccumulator.finalize` |
| `R_{p,i}^{(n)}` | `representatives.select_representatives` |
| `E_{b→a}(α) = (∂e_a/∂e_b) e_b` | `edges.edge_scores_for_group` |
| `e_a(s) − e_a(s | e_b = 0)` | `edges.exact_ablation_effect` |
| `Pa_{K_par,τ}(a; α)` | `parents.parent_set` |
| `γ(b) = (p(j_b), i_b)` | `ids.ComponentId.gate` |
| `G_{p,i}^{(n)}(α)` | `parents.parent_gate_set` |
| `J^{(n,m)}_{p,i}(α)` | `comparison.jaccard` |
| `w_{b→a} = |E| ‖u_a‖ / σ_{j_a}` | `comparison.edge_weight` |
| `S`, `π`, `D_TV`, `M` | `comparison.aggregated_parent_scores`, `parent_distribution`, `total_variation`, `PairComparison.strength_*` |

Undefined values are `null`, never `0`: `β` for a never-firing component, `ρ` when
`Σ_k q = 0`, the Jaccard index of two empty sets, and `D_TV` unless both models have a
non-empty `R` and `M > 0` (`distribution_valid`).

## 4. How the edge score is computed

Component `b` adds `e_b u_b` to the native output `y_{j_b,α}`. Setting `e_b = 0`
removes exactly that vector, so `∂e_a/∂e_b = u_bᵀ ∂e_a/∂y_{j_b,α}`.
`source_hooks.probed_forward` adds a zero probe to each upstream `y_j` at the anchor
token and keeps the autograd graph. One backward pass from `e_a` therefore gives
`∂e_a/∂y_j` for **all** upstream matrices, and

```
E_{·→a}(α)[j, :] = e_{j,α,:} ⊙ (U_j ∂e_a/∂y_{j,α})
```

for every upstream component at once. Only firing components (`g_b = 1`, so `e_b ≠ 0`)
are stored. Anchors from different documents share one forward/backward pass; two
anchors from the same document never do (`edges.group_rows`), because a later anchor
attends to an earlier one.

The forward pass stops as soon as the deepest needed matrix input is captured. Hooks
touch only the anchor rows and are removed on exit; source weights are never modified.

### Edge-score conventions (fixed by the formulation, not settings)

These follow from the formulation's own definitions and are recorded in every
`edges/manifest.json` (`config.FORMULATION_FIXED_CONVENTIONS`):

- **Real model.** $E \approx e_a(s) - e_a(s \mid e_b = 0)$ is evaluated on $s$ through the
  real model, so removing $b$ subtracts $e_b u_b$ from the native $y_{j_b}$.
- **Gates held fixed.** $g$ is a 0/1 step function, so $\partial g / \partial e_b = 0$
  almost everywhere and $E$ is the same whether or not the gate would be recomputed.
- **Same anchor.** $e_a$ and $e_b$ carry the same $\alpha$.
- **Candidate set.** $\mathcal U^{(n)}(a)$ is all selected components; non-upstream ones
  have $E = 0$ and never pass $|E| > \tau$. Components of $a$'s own gate are allowed
  (researcher decision, 2026-11-21).
- **Total effect.** Everything downstream of $b$ responds to its removal.

`edges/ablation_check.json` compares the Taylor score with the exact ablation for the
strongest parent of the first `ablation_check_records` records. On linear paths (for
example `v → o` within one attention block, with attention weights fixed) the two agree
exactly. Through nonlinearities they diverge for large `e_b`. This is the first-order
error, not a bug.

## 5. Open decisions

Every decision, who made it and why: [`posthoc_decisions.md`](posthoc_decisions.md).
Approved values and their derivation: [`posthoc_thresholds.md`](posthoc_thresholds.md).
Run configs: `configs/multimodel/posthoc/v31_500m_validation_v1.yaml` (analysis) and
`v31_500m_validation_smoke_v1.yaml` (engineering smoke run).

Every row is a required config field with no default. A `null` blocks only the stages
that need it; `--check` lists them.

| Field | Stage | Formulation status |
|---|---|---|
| `representatives.screening_metric` (`beta`/`rho`) | representatives | "ambiguity on what is used on the screening step" |
| `representatives.k_rep`, `tau_rep` | representatives | free parameters |
| `edges.downstream_gates` (`screened` [decided], `most_frequent`, `explicit`; optional `count` cap) | edges | formulation says "each shared gate"; `screened` keeps the gates that pass its screening step |
| `edges.max_anchors_per_gate`, `anchors.edge_pools` | edges | not specified (sample size) |
| `edges.anchor_sampling` (`first_in_pool`/`seeded_random`) + `anchor_seed` | edges | not specified; which firing anchors are kept when a gate exceeds its quota |
| `anchors.edge_anchor_source` (`disjoint_after_importance`/`same_as_importance`) | edges | not specified; whether `E` uses the anchors that define `β`, `σ` |
| `anchors.importance_pools` | importance | expectation set for `β`, `σ` |
| `parents.k_par`, `tau` | compare | free parameters |
| `backend.options.split` (validation/test) | importance, edges | evaluation protocol |
| `backend.options.source_dtype` (bfloat16/float32) | importance, edges | bf16 reproduces the evaluation-time gates; fp32 gives more precise gradients but slightly different residuals |

`edges.anchors_per_backward` and `edges.ablation_check_records` are cost and diagnostic
knobs only; they do not change any score.

`compare/summary.json` reports plain means over anchors per gate and per model pair. It
is labelled `DESCRIPTIVE_ANCHOR_MEANS_NOT_A_VALIDATED_STATISTIC`; the formulation
defines per-anchor quantities only.

## 6. Real-run blockers

Observed on 2026-11-21 from the `minhdatle` account. This branch is based on
`aspd-v3.1-feature-coherence-v1` (`6d998c8`), the latest evaluation branch. It contains
the checkpoint's training commit `5ca3b43`, and its
`configs/multimodel/aspd_v3/main500m_preparation_v1.yaml` matches the checkpoint
byte-for-byte: file SHA-256 `f7a6d874…63be`, config fingerprint `7e44ead3…3902`,
Gemma-2-2B / SmolLM2-1.7B / Granite-3.3-2B, 168 targets.

1. **Config loading from a non-owner account (resolved for the v1 runs).** The YAML sets
   `full_training_authorized: true`, so `V31Main500MConfig`'s model validator re-verifies
   the evidence that authorized *launching training*. `validate_preflight_evidence()`
   reads `…/preflight/preflight_report.json` and `validate_exact_resume_evidence()` reads
   `…/resume_validation_job_50550194/benchmark_report.json`. Both files are `-rw-------`
   and owned by the training owner, and neither feeds any value into the analysis. With
   researcher authorization (2026-11-22), the v1 run configs set
   `skip_training_launch_evidence: true`: `v31_backend.load_frozen_config` skips exactly
   those two checks and proves the config is the checkpoint's (file SHA-256 and
   fingerprint equal to the checkpoint manifest). The alternative remains the owner
   running `chmod g+r` on the two files.
2. **This base is not merged into `main`.** CONTRIBUTING requires official evaluations
   to run from a merged commit. A run from this branch is a validation run until the
   evaluation branches are merged.
3. **The v3.1 backend has never run on a GPU.** Its first run should be a small smoke run,
   e.g. `importance_pools: 1`, two explicit gates, `max_anchors_per_gate: 2`, written to a
   validation namespace.

## 7. Guide for agents extending this code

- **New model or checkpoint family.** Implement `backend.PosthocBackend`. Provide
  `MatrixHookSpec`s (module path, optional fused slice), `(U [C,d_out], V [C,d_in])`
  factors, and `AnchorPool`s whose `gates` are natural binary gates on the shared
  anchor axis. Register it in `pipeline.build_backend` and `config.required_backend_options`.
  Mirror `toy_backend.py`, and add a CPU contract test like `test_multimodel_posthoc_v31_contract.py`.
- **New comparison statistic.** Add a pure function to `comparison.py` that takes
  `ModelParentView`s, test it with hand-computed values in `test_multimodel_posthoc_math.py`,
  and emit it from `run_compare`. Don't touch `edges` artifacts: stats are recomputable.
- **Changing an edge convention** (for example, gate recomputation or multi-position
  attribution) is a scientific change. Add it as a new explicit config field with no
  default, record it under `conventions` in the edge manifest, and keep the old path.
  Extend the finite-difference test in `test_multimodel_posthoc_edges.py`: that identity
  is what defines correctness.
- Keep the stage contract: each stage reads only verified artifacts plus the config, writes
  one new directory, and never deletes or overwrites.

## 8. Verification record

| Check | Result |
|---|---|
| `pytest -q tests/test_multimodel_posthoc_math.py tests/test_multimodel_posthoc_edges.py tests/test_multimodel_posthoc_pipeline.py tests/test_multimodel_posthoc_v31_contract.py` | 36 passed in 15.5 s, CPU, `OMP_NUM_THREADS=1`, on base `6d998c8` (2026-11-22) |
| `ruff check` (repository config) on all new Python files | passed |
| Finite-difference identity `Δe_a/ε ≈ E_{b→a}` on tiny random Llama models and a fused-slice model (float64) | passed (tests) |
| Exact equality of Taylor and full ablation on the linear `v → o` path | passed (tests) |
| v3.1 backend contract on this base (`aspd-v3.1-feature-coherence-v1`) | API symbols pass; frozen config loads through `load_frozen_config` from a non-owner account, matches the checkpoint manifest (file SHA-256, fingerprint `7e44ead3…`), 168 targets; an altered config is rejected |
| Real checkpoint / GPU smoke (job 52378170, commit `5625def`, 1 B200) | `COMPLETED 0:0` in 4 min 57 s; all four stages; 36 edge records; 12 of 12 pair comparisons valid; Taylor vs exact within 1–12% (engineering only, not a result) |
