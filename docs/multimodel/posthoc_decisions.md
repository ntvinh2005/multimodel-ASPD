# Post-hoc analysis: decision log

Every choice behind the post-hoc analysis, who made it, and why. The mathematics is in
[`multimodel_aspd_formulation.tex`](multimodel_aspd_formulation.tex); the code map is in
[`posthoc_formulation_map.md`](posthoc_formulation_map.md); the numbers and their
derivation are in [`posthoc_thresholds.md`](posthoc_thresholds.md).

To change a decision, add a dated entry here, create a new versioned config (`…_v2.yaml`)
and run into a new output root. Configs and outputs that already produced results are
never edited.

Run configs that implement this log:
- `configs/multimodel/posthoc/v31_500m_validation_v1.yaml`: the analysis;
- `configs/multimodel/posthoc/v31_500m_validation_smoke_v1.yaml`: engineering smoke run,
  same decisions, tiny sample sizes.

## A. Researcher decisions (2026-11-21, in conversation)

1. **Screening metric: β only.** Representatives $\mathcal R^{(n)}_{p,i}$ are ranked and
   thresholded by β. ρ is not used, neither as the ranking nor as an extra filter.
   This resolves the formulation's note "ambiguity on what is used on the screening
   step". (`representatives.screening_metric: beta`)
2. **Same-gate parents are allowed.** A component whose gate equals the downstream
   gate, $\gamma(b) = (p, i)$ (for example the same MID feature in an earlier layer), may
   be a parent. So $\mathcal G_{p,i}$ may contain $(p, i)$ itself.
3. **Gates are chosen by the screening step.** Analyse every gate that the formulation's
   own screening keeps, with no hand-picked list (`edges.downstream_gates.mode: screened`).
4. **Edge anchors are disjoint from importance anchors.** β and σ (which choose the
   representatives) are computed on pools 0–49; edge scores on the following pools. This
   avoids choosing representatives on the same anchors that are then analysed. The
   researcher accepted this proposal. (`anchors.edge_anchor_source: disjoint_after_importance`)
5. **Validation split** (document-disjoint from training). (`backend.options.split: validation`)
6. **bf16 source models** for gates, matrix captures and edge gradients.
   (`backend.options.source_dtype: bfloat16`)
7. **Proposed thresholds and sample sizes approved** ("ok", 2026-11-21) as starting
   values, to be tuned. See C.

## B. Fixed by the formulation itself (not settings)

These were first listed as open; on review each follows from the formulation's
definitions. They are recorded in every `edges/manifest.json`
(`config.FORMULATION_FIXED_CONVENTIONS`).

1. **Real model.** $E \approx e_a(s) - e_a(s \mid e_b = 0)$ is evaluated on the input $s$
   through the real model, so removing $b$ subtracts $e_b u_b$ from the native output
   $y_{j_b}$. A replacement model (each matrix swapped for its decomposition) is a
   different quantity.
2. **Gates held at their natural values.** $g$ is a 0/1 step function, so its
   derivative with respect to $e_b$ is zero almost everywhere. $E$ is identical whether
   or not the gate would be recomputed.
3. **Same anchor.** $e_a$ and $e_b$ carry the same $\alpha$, so $e_b$ is taken at the
   anchor's own token in each model.
4. **Candidate set $\mathcal U^{(n)}(a)$ = all selected components.** Components exist
   only for $j \in J^{(n)}$. Any component not upstream of $a$ in compute order
   (q,k,v → o → gate,up → down, then the next layer) has $E = 0$ exactly and can never
   pass $|E| > \tau$. Parents can come from both sites.
5. **Total effect.** Removing $b$ lets everything downstream of $b$ respond.
6. **Gates analysed at anchors where they fire.** Elsewhere every $e_a = 0$, so the parent
   set is empty and the formulation's validity rule excludes the anchor anyway.
7. **All model pairs.** "Two models $n$ and $m$" with $N = 3$: (gemma, smol),
   (gemma, granite), (smol, granite). (`comparison.model_pairs: all`)

## C. Approved starting values (proposed by Claude, approved 2026-11-21)

Derivations are in `posthoc_thresholds.md`.

1. `representatives.tau_rep = 0.07`: half of the typical-component scale
   $\beta_{\mathrm{ref}} = \sqrt{(1-\mathrm{FVU})/K} \approx 0.13$–$0.15$.
2. `representatives.k_rep = 3`.
3. `parents.k_par = 10`.
4. `parents.tau = 0.0`. Rely on `k_par` until real edge-score distributions exist (see E.1).
5. `anchors.importance_pools = 50` (102,400 anchors; an average gate fires ~400 times).
6. `edges.downstream_gates.min_importance_fires = 50`: a gate must fire at least 50 times
   in the importance stage to be analysed, so no β rests on a handful of samples.
7. `edges.max_anchors_per_gate = 20`, `anchor_sampling = seeded_random`,
   `anchor_seed = 0`: at most 20 firing anchors per gate, chosen reproducibly at random
   (not the first rows, which favour early documents).
8. `anchors.edge_pools = 25` (51,200 anchors; an average gate fires ~200 times, so the cap
   of 20 fills for almost every gate).
9. `edges.anchors_per_backward = 16` and `edges.ablation_check_records = 50`. Cost and
   diagnostic only; they do not change any score.

## D. Implementation choices (made by Claude; none changes a defined quantity)

1. A gate counts as screened if its $\mathcal R$ is non-empty in **at least two** models.
   With fewer, no valid pair exists. Pairs with an empty side are still reported, marked
   invalid.
2. TopK ties are broken deterministically: by score, then layer, then matrix, then index.
   Filter-then-TopK is used, which equals the formulation's TopK-then-filter for one score.
3. Undefined values are written as `null`, never as a number: β of a never-firing
   component, ρ with $\sum q = 0$, the Jaccard index of two empty sets, π with $M = 0$,
   and $D_{TV}$ when the validity rule fails.
4. Gates are reproduced exactly as in the v3.1 evaluation extractors: 2,048-anchor
   pools, bf16 encoder, `update_statistics=False`, $g = 1[g^s > 0]$.
5. Several anchors share one backward pass only if they come from different documents.
   Equality with single-anchor runs is tested.
6. Each stage writes an immutable directory with SHA-256 hashes and provenance. A later
   stage refuses a prerequisite whose hashes changed.
7. Added outputs not defined by the formulation: the Taylor-vs-exact ablation check
   (`edges/ablation_check.json`) and plain per-gate means of $J$ and $D_{TV}$ over anchors
   (`compare/summary.json`, labelled descriptive).
8. The smoke config caps the screened gates at 1 per site and has no minimum-firings
   floor. It is for engineering checks only and is never reported as a result.

## E. Open (needs a decision later)

1. **Units of $\tau$.** $E_{b\to a}$ is in the output units of matrix $j_a$, which differ
   between matrices and models. One absolute $\tau$ is not equally strict everywhere.
   Option: threshold $|E|/\sigma_{j_a}$. That would be a formulation change; decide after
   seeing real edge scores.
2. **Edge-stage cost.** Up to ~2.9 M backward passes if every gate passes screening. Decide
   whether to cap screened gates per site (`edges.downstream_gates.count`) after the smoke
   run measures speed.
3. **Read access for the config loader.** The frozen 500M config re-checks two
   training-launch reports that only the training owner can read. The owner must
   `chmod g+r` them, or approve an evaluation-only loader that skips those two checks
   (see `posthoc_analysis.md`, "Real-run blockers").
4. **Merge into `main`.** Runs from this branch are validation runs under CONTRIBUTING
   until the evaluation branches are merged.
