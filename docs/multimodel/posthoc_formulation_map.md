# Formulation → code map for the post-hoc analysis

This walks through [`multimodel_aspd_formulation.tex`](multimodel_aspd_formulation.tex) in
order. For every definition it gives the code that implements it and the match status:

- **EXACT** — the formula is implemented as written.
- **INTERPRETED** — the formulation leaves something open; the choice is a required
  config field (§5).
- **ADDED** — the code computes something the formulation does not define.
- **REUSED** — implemented by the trained v3.1 code, not by the post-hoc package.

Paths are relative to `aspd/multimodel/posthoc/` unless stated otherwise. Line numbers
refer to branch `analysis/posthoc-parent-sets` at the commit that adds this file.

## 1. Notation: formulation symbol → code name

| Formulation | Code | Note |
|---|---|---|
| model $n$, $m$ ($N$ models) | model key string (`gemma`, `smol`, `granite`); order `backend.model_keys` | |
| layer $\ell \in \mathcal L^{(n)}$ | `MatrixId.layer` (absolute layer index) | |
| site type $p \in \{\mathrm{PRE},\mathrm{MID}\}$ | `site` ∈ `{"pre","mid"}` | |
| matrix type $m \in \{Q,K,V,O,\mathrm{gate},\mathrm{up},\mathrm{down}\}$ | `MatrixId.matrix` ∈ `q,k,v,o,mlp_gate,mlp_up,mlp_down` | repo registry spelling; the letter $m$ is not reused for models in code |
| $j = (\ell, m)$ | `MatrixId(model, layer, matrix)`, key `gemma.L12.mlp_up` | `ids.py:57` |
| $p(j)$ | `MatrixId.site` / `MATRIX_SITE` | `ids.py:84`, `ids.py:34` |
| component index $i$ ($1..C$) | `feature` ($0..C-1$) | zero-based |
| component $(j, i)$, $P^{(n)}_{j,i}$ | `ComponentId(matrix, feature)`, key `gemma.L12.mlp_up/77` | `ids.py:123` |
| shared gate $(p, i)$ | `GateId(site, feature)`, key `mid/77` | `ids.py:100` |
| anchor $\alpha$ / token $t$ | row on the shared anchor axis + `anchor_id`; $t^{(n)}(\alpha)$ = `ModelInputs.anchor_document/anchor_token` | `backend.py` |
| $u^{(n)}_{j,i}$, $v^{(n)}_{j,i}$ | `U[i]` of `U [C,d_out]`, `V[i]` of `V [C,d_in]` | v3 storage layout, `components.py:35–36` |
| $a$ (downstream), $b$ (upstream) | `downstream` / `component`, `upstream` / `b` | same letters in `edges.py` |
| $\xi$ (upstream gate identity) | `GateId` keys of the score dicts | |

## 2. Setup

| Formulation | Code | Status |
|---|---|---|
| $\mathcal M^{(1..N)}$ | `PosthocBackend.source_model(model)`; v3.1: frozen HF models from `load_frozen_v3_model` (`v31_backend.py:46`) | REUSED |
| $\mathbf s \sim \mathcal D$, shared text | aligned Pile documents of the selected split (`DiagnosticAlignedAnchorStream`, `v31_backend.py:156`) | REUSED |
| $\tau^{(n)}(\mathbf s)$, $T_n$, unaligned positions | each model's own token IDs in `ModelInputs.input_ids`; the anchor gather map gives $t^{(n)}(\alpha)$ | REUSED (exact canonical-prefix alignment) |
| $\Sigma^{(n)}$, $J^{(n)}$ | `PosthocBackend.hook_specs(model)` → one `MatrixHookSpec` per selected matrix (56 per model in v3.1) | EXACT |
| $H^{(n)}_\sigma$, $h^{(n)}_{\sigma,t}$ | residuals captured only to recompute the gate (`capture_family_pool` inside `v31_backend._gates`, `:180`) | REUSED |
| $W_j$, $x_{j,t}$, $y_{j,t} = W_j x_{j,t}$ | `source_hooks.capture_matrix_io` (`:81`) reads the real matrix input and logical output at the anchors | EXACT. The three v3.1 models have no biases, so "omit biases" holds; a model with a bias would include it in $y$. |

## 3. Architecture (current) — produces $g_{p,\alpha,i}$

The post-hoc package does not re-implement the encoder. It runs the **trained** v3.1
modules exactly as the evaluation extractors do (`v31_backend._gates`, `:180–202`).

| Formulation | Trained module (`aspd/multimodel/v3/`) | Status |
|---|---|---|
| $r^{(n)}_{p,t} = [h_{l_1,p,t}, \ldots, h_{l_8,p,t}]$ | `encoder.concatenate_layer_activations` (`encoder.py:16`), called by `LocalSiteEncoder` (`model_owned.py:82`) | REUSED |
| $v = P^{(n)}_p r + b$, $E = \mathrm{Reshape}_{8\times512}(v)$ | `LearnedProjectionPacker` (`encoder.py:55`) | REUSED |
| $Z^{(n)}_{p,t} = F^{(n)}_p(E)$, 2 blocks | `IntraModelTransformer` (`encoder.py:79`) | REUSED |
| $Z^{\mathrm{shared}} = \mathrm{Transformer}(\mathrm{Concat}(Z^1, \mathrm{SEP}, \ldots))$ | `pack_cross_model_tokens` (`encoder.py:121`; adds one-hot model/SEP identity, 512+5 = 517) → 517→512 adapter → `CrossModelTransformer` (`encoder.py:189`), inside `SharedSiteEncoder` (`model_owned.py:118`) | REUSED. The `.tex` omits the identity channels and adapter. |
| $\bar z = \frac1L \sum_k Z_k$, $a = W^{\mathrm{gate}}_p \bar z + b$ | `FeatureReadout` (`encoder.py:224`) | REUSED |
| z-score + ReLU, $\epsilon$ | `FeatureReadout`, feature-axis `layer_norm`, $\epsilon = 10^{-5}$, population variance (`encoder.py:272`) | REUSED |
| $g^s = \mathrm{BatchTopK}_K(\{\tilde a\}_{\alpha\in\mathcal B})$ | `V3BatchTopK` (`gating.py:15`), $K = 32$, pool = 2,048 anchors, `update_statistics=False` | REUSED |
| $g_{p,\alpha,i} = 1[g^s > 0]$ | `v31_backend._gates`: `(code.values[0] > 0)` (`:201`) | EXACT |
| $P_{j,i} = u v^\top$ | `RankOneComponents.U`, `.V` loaded from the checkpoint (`v31_backend._build_factors`, `:134`) | REUSED |
| $e_{j,\alpha,i} = g_{p(j),\alpha,i}\,(v^\top_{j,i} x_{j,\alpha})$ | `contributions.component_coefficients` (`:28`); inline in `edges.py:130` and `:142–153` | EXACT (binary $g$, not $g^s$) |
| $\Delta y_{j,\alpha,i} = e\,u$ | `contributions.component_contributions` (`:48`); $\lVert\Delta y\rVert^2 = e^2 \lVert u\rVert^2$ in `contribution_sq_norms` (`:40`) | EXACT |

## 4. Post-hoc analysis

### 4.1 Edge score

| Formulation | Code | Status |
|---|---|---|
| $E_{b\to a} = \frac{\partial e_a}{\partial e_b} e_b$ | `edges.edge_scores_for_group` (`:82`). Because $e_b$ enters only through $e_b u_b$ in $y_{j_b}$, $\partial e_a / \partial e_b = u_b^\top \partial e_a / \partial y_{j_b,\alpha}$. One backward pass per downstream $a$ (`:134`) gives $\partial e_a / \partial y_j$ for every upstream matrix via zero probes (`source_hooks.probed_forward`, `:137`). Then $E = e_b \cdot (U_b \cdot \mathrm{grad})$ (`:152–153`). | EXACT (§5.1) |
| $\approx e_a(s) - e_a(s \mid e_b = 0)$ | `edges.exact_ablation_effect` (`:169`) subtracts $e_b u_b$ from the native output and re-runs. Same sign as $E$. | EXACT. Used as a diagnostic (`edges/ablation_check.json`) and in tests. |
| sign reading ($E > 0$: ablating $b$ decreases $e_a$) | scores stored **signed** (`EdgeRecord.scores`) | EXACT |
| "at anchor $\alpha$" | the model's token $t^{(n)}(\alpha)$; $e_b$ at the same token | EXACT (§5.1) |

### 4.2 Component importance

| Formulation | Code | Status |
|---|---|---|
| $J^{(n)}_p$ | `ImportanceTable.matrices(model, site)` (`importance.py:137`) | EXACT |
| $\sigma_j = \sqrt{\mathbb E_\alpha \lVert y - \mathbb E y\rVert^2}$ | `MatrixImportanceAccumulator.finalize` (`importance.py:92–98`), float64 streaming sums | EXACT |
| $\beta_{j,i} = \sqrt{\mathbb E[\lVert\Delta y\rVert^2 \mid g = 1]} / \sigma_j$ | `importance.py:100–104`: $\sum \lVert\Delta y\rVert^2 / \#\{g=1\}$, since $\Delta y = 0$ when $g = 0$ | EXACT; `NaN` if never fired or $\sigma = 0$ |
| $q_{j,i} = \sqrt{\mathbb E_\alpha \lVert\Delta y\rVert^2}$ | `importance.py:99` | EXACT |
| $\rho_{j,i} = q_{j,i} / \sum_k q_{j,k}$ | `importance.py:105–106` | EXACT; `NaN` if $\sum = 0$ |
| $\mathbb E_\alpha$ over which anchors | the `importance_pools` natural pools of the selected split | INTERPRETED (`anchors.importance_pools`, `backend.options.split`) |
| "ambiguity on what is used on the screening step" | `representatives.screening_metric`; researcher chose `beta` (2026-11-21), $\rho$ not used | DECIDED |

### 4.3 Parent sets

| Formulation | Code | Status |
|---|---|---|
| correspondence = same shared gate | every model's representatives are keyed by the same `GateId` (`pipeline.run_representatives`, `:120`) | EXACT |
| $\mathcal R^{(n)}_{p,i} = \{j \in \mathrm{TopK}_{K_{rep}}(\beta), \beta > \tau_{rep}\}$ | `representatives.select_representatives` (`:40`). Filter-then-TopK equals TopK-then-filter for one score. Ties go by layer, then matrix. `NaN` is never selected (`:48`). | EXACT |
| $a^{(n)}, a^{(m)}$ corresponding downstream | at each selected anchor, edges are computed for every $a \in \mathcal R^{(n)}_{p,i}$ in every model (`pipeline.run_edges`, `:158`) | EXACT |
| $\mathcal U^{(n)}(a)$ | `MatrixId.is_upstream_of` (`ids.py:93`): same model, $(\ell, \mathrm{stage})$ earlier; stages q,k,v < o < gate,up < down; same-gate components included | EXACT (§5.1) |
| $\mathrm{Pa}_{K_{par},\tau}(a;\alpha) = \{b \in \mathrm{TopK}(\lvert E\rvert) : \lvert E\rvert > \tau\}$ | `parents.parent_set` (`:39`); TopK then strict threshold (`:48`). Only $b$ with $g_b = 1$ are stored; the rest have $E = 0$ and can never pass $\lvert E\rvert > \tau \ge 0$. | EXACT |
| $\mathcal P^{(n)}_{a,\alpha}$, $\mathcal P^{(m)}_{a,\alpha}$ | `parents` dict built per model in `pipeline.run_compare` (`:286`) | EXACT |

### 4.4 Parent-set comparison

| Formulation | Code | Status |
|---|---|---|
| $\gamma(b) = (p(j_b), i_b)$ | `ComponentId.gate` (`ids.py:134`) | EXACT |
| $\mathcal G^{(n)}_{p,i}(\alpha) = \bigcup_{a\in\mathcal R} \{\gamma(b)\}$ | `parents.parent_gate_set` (`:53`), `ModelParentView.gate_set` (`comparison.py:91`) | EXACT |
| $J^{(n,m)}_{p,i}(\alpha)$ | `comparison.jaccard` (`:27`) | EXACT; `null` when both sets are empty (ADDED: 0/0 is undefined in the `.tex`) |
| $w_{b\to a} = \lvert E\rvert \lVert u_{j_a,i_a}\rVert / \sigma_{j_a}$ | `comparison.edge_weight` (`:34`); $\lVert u\rVert$ from the checkpoint factors (`selection.json: downstream_write_norms`), $\sigma$ from the importance stage | EXACT. v3.1 keeps $\lVert u\rVert = 1$, so in practice $w = \lvert E\rvert / \sigma$. |
| $S(\xi) = \frac{1}{\lvert\mathcal R\rvert}\sum_a \sum_{b:\gamma(b)=\xi} w$ | `comparison.aggregated_parent_scores` (`:42`); representatives without parents still count in $\lvert\mathcal R\rvert$ | EXACT |
| $\pi(\xi) = S(\xi) / \sum S$ | `comparison.parent_distribution` (`:66`) | EXACT; `null` when $\sum S = 0$ |
| $D_{TV} = \frac12 \sum \lvert\pi^n - \pi^m\rvert$ | `comparison.total_variation` (`:75`) | EXACT |
| $M = \sum_\xi S(\xi)$ | `PairComparison.strength_left/right` (`comparison.py:109`, set in `compare_pair` `:128`) | EXACT |
| defined only for nonempty $\mathcal R$ and positive totals in both models | `compare_pair` → `distribution_valid`; `D_TV = null` otherwise | EXACT |
| which pairs $(n, m)$ | `comparison.model_pairs`, default `all` | EXACT (all pairs of the $N$ models) |

### 4.5 Where each formulation object is written

- `profiles/parents_per_anchor.safetensors` stores every $\mathrm{Pa}(a;\alpha)$ entry:
  parent, signed $E$, $w$ and rank (`parents_vocab.json` for the ids).
- `compare/comparisons.jsonl` stores $\mathcal G^{(n)}_{p,i}(\alpha)$, $J$, $D_{TV}$ and
  $M$ per gate, anchor and model pair.
- `profiles/parent_distributions.safetensors` stores $S^{(n)}_{p,i}(\xi;\alpha)$ and
  $\pi^{(n)}_{p,i}(\xi;\alpha)$ for every gate, model, anchor and parent gate $\xi$, with
  $|\mathcal R|$, $M$ and validity per (gate, model, anchor). `distributions_vocab.json`
  has the ids. Their means over valid anchors are in `profiles/gate_parent_profile.jsonl`.
- Raw $E_{b\to a}$ for all firing upstream components are in the edge shards.

Until 2026-11-23 only `compare` existed: Pa, $S$ and $\pi$ were computed but not written.
The `profiles` stage writes them. It is part of the default pipeline (`--stage all`) and of
the CPU job of the sharded chain.

### 4.6 Computed but not in the formulation (ADDED)

- `profiles/component_parent_profile.jsonl`: per component, parent frequency, mean $E$,
  mean $|E|$, mean $w$ and sign consistency over anchors. The top 20 are listed; the full
  data is in `parents_per_anchor`.
- `profiles/gate_parent_profile.jsonl`: per gate and model, mean $\pi$ (top 20), mean $M$
  and valid-anchor counts, plus the parent gates that every model has in its top 20.
- `profiles/edge_profile.json`: the distribution of edge scores by model,
  upstream → downstream matrix type and layer gap, for all stored edges and for parents
  only.


| Output | Why |
|---|---|
| `edges/ablation_check.json`: Taylor $E$ vs exact $e_a(s) - e_a(s \mid e_b = 0)$ for the strongest parent of the first `ablation_check_records` records | measures the first-order error of the estimate the formulation itself defines |
| `compare/summary.json`: per-gate and overall means of $J$ and $D_{TV}$ over anchors | convenience; labelled `DESCRIPTIVE_ANCHOR_MEANS_NOT_A_VALIDATED_STATISTIC`. The formulation defines only per-anchor quantities. |
| `selection.json`: analysed gates, anchors, quota shortfall | provenance |

## 5. Decision register

Everything the formulation leaves open is a **required config field with no default**.
A `null` blocks only the stage that needs it (`--check` lists them), and a value that is
not implemented is rejected by name. The toy config sets every field, labelled SMOKE
ONLY; `v31_500m_template.yaml` leaves all of them `null`.

### 5.1 Edge-score conventions — fixed by the formulation

These were first recorded as open decisions. On review they follow from the
formulation's definitions and are not settings. Each reason is given in
`config.FORMULATION_FIXED_CONVENTIONS` and in `posthoc_analysis.md` §4:
- real model ($E$ is defined through $e_a(s \mid e_b = 0)$ on the real input);
- gates held fixed (a step-function gate has zero derivative almost everywhere);
- same anchor (shared $\alpha$ in the notation);
- candidate set = all selected components (non-upstream ones have $E = 0$), including
  the downstream gate's own components (researcher decision, 2026-11-21);
- total effect.

### 5.2 Sampling and data

| Field | Values | Meaning |
|---|---|---|
| `anchors.edge_anchor_source` | `disjoint_after_importance` \| `same_as_importance` | whether $E$ uses different anchors from those defining $\beta$ and $\sigma$ |
| `edges.anchor_sampling` (+ `anchor_seed`) | `first_in_pool` \| `seeded_random` | which firing anchors are kept when a gate fires more often than `max_anchors_per_gate`. The first implementation silently used `first_in_pool`. |
| `edges.downstream_gates` | `screened` (decided) \| `most_frequent` \| `explicit`; optional `count` | which gates $(p, i)$ to analyse. `most_frequent` = the `count` most-firing gates per site; ties go to the lower index; never-firing gates are excluded. |
| `edges.max_anchors_per_gate`, `anchors.importance_pools`, `anchors.edge_pools` | integers | sample sizes |
| `backend.options.split` | `validation` \| `test` | document-disjoint evaluation split |
| `backend.options.source_dtype` | `bfloat16` \| `float32` | bf16 reproduces the evaluation-time gates; fp32 gives more precise gradients |

### 5.3 Formulation parameters

`representatives.screening_metric`, `k_rep`, `tau_rep`, `parents.k_par`, `parents.tau`,
`comparison.model_pairs` (default `all`). Proposed values: [`posthoc_thresholds.md`](posthoc_thresholds.md).

### 5.4 Technical choices that do not change any defined quantity

- Deterministic tie-breaking in TopK.
- `null` instead of a number for undefined values.
- Batching several anchors per backward pass, one per document. Equality with single-anchor runs is tested.
- Early stop of the forward pass.
- `anchors_per_backward`, a memory knob only.
