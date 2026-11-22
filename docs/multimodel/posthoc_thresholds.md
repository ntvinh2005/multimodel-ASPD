# Post-hoc analysis: pinned thresholds and sample sizes

**Status: PROPOSED (2026-11-21), awaiting researcher approval.** These values are not in
any executable config yet. `configs/multimodel/posthoc/v31_500m_template.yaml` keeps them
`null` and points here. When a value is approved or retuned, add a row to the
[tuning log](#tuning-log) and copy the value into a new, versioned run config. Never edit
a config that already produced results.

Decided by the researcher on 2026-11-21 (not proposals):
- representatives are ranked by **β only** (ρ not used);
- components of the downstream gate itself **may** be parents;
- edge anchors are **disjoint** from the anchors used for β and σ;
- **validation** split;
- **bf16** source models;
- gates are chosen by the **screening step** (`mode: screened`).

## 1. Reference scale for β

β compares a component's typical contribution, *when it fires*, with the spread of
the matrix output:

$$\beta_{j,i} = \sqrt{\mathbb E[\lVert\Delta y_{j,\alpha,i}\rVert^2 \mid g = 1]}\,/\,\sigma_j.$$

The reconstruction of matrix $j$ is the sum of the firing components:
$\hat y = \sum_i g_i e_i u_i$. If the firing contributions were roughly orthogonal, the
explained variance would split over the components that fire:

$$(1-\mathrm{FVU}_j)\,\sigma_j^2 \approx \sum_i P(g_i = 1)\,\beta_{j,i}^2\,\sigma_j^2,
\qquad \sum_i P(g_i = 1) = K = 32.$$

So the firing-weighted mean of $\beta^2$ is $(1 - \mathrm{FVU}_j)/K$, and a *typical
active component* has

$$\beta_{\mathrm{ref}} = \sqrt{(1-\mathrm{FVU}_j)/K}.$$

The held-out reconstruction of the 500M checkpoint
(`results/aspd_v3_1_interpretability_failure_investigation.md`: 100,352 validation
anchors, all 168 targets) reports matrix FVU 0.318–0.325 for PRE (q/k/v) and
0.439–0.470 for MID (o/MLP):

- PRE: $\sqrt{0.68/32} \approx 0.146$
- MID: $\sqrt{0.53/32}$ to $\sqrt{0.56/32} \approx 0.129$–$0.132$

Caveats: the orthogonality assumption ignores cross terms, and FVU is a mean over
matrices. This gives a **scale**, not a cut-off derived from data. Recompute it from the
importance stage once it has run (`importance.safetensors` has β for every component).

## 2. Proposed values

**`representatives.tau_rep = 0.07`**. This is about half of $\beta_{\mathrm{ref}}$ at
both sites. A component counts as a representative only if, when it fires, its
contribution is at least half that of a typical active component. Through the
screening step, this also decides **which gates are analysed**: a gate whose components
all fall below 0.07 in a model has an empty $\mathcal R$ there. With `mode: screened`, a
gate needs a non-empty $\mathcal R$ in at least two models. Raising `tau_rep` both
tightens the representatives and shrinks the edge-stage cost.

**`representatives.k_rep = 3`**. A PRE gate controls 24 matrices per model and a MID gate
32. Three representatives per model cover the dominant loci without letting weak
matrices into $\mathcal G$. The edge-stage cost is proportional to `k_rep`.

**`parents.k_par = 10`**. For a late downstream matrix, $\mathcal U(a)$ holds up to about
55 upstream matrices × 32 firing components, roughly 1,700 candidates. `k_par = 10` keeps
the top ~0.6%. $\mathcal G$ then has at most `k_rep × k_par` = 30 gate identities per
model, enough for the Jaccard index to take more than a few values. With `k_par = 1`,
Jaccard is almost always 0 or 1.

**`parents.tau = 0.0`**, relying on `k_par` alone for now. Units caveat for the
formulation: $E_{b\to a}$ is a change in $e_a$, and $e_a$ is measured in the output units
of matrix $j_a$ ($|e_a| = \lVert\Delta y_a\rVert$, since $\lVert u\rVert = 1$). Those units
differ between matrices and between models; $\sigma_j$ differs. One absolute $\tau$ is
therefore stricter for some downstream matrices than for others. The weights $w$ divide
by $\sigma_{j_a}$, but the Pa threshold does not. Possible formulation change, if wanted:
threshold the normalized score $|E_{b\to a}|/\sigma_{j_a}$, or $|E|/|e_a|$. Pick $\tau$
from the edge-score distribution after the smoke run.

**`anchors.importance_pools = 50`** (102,400 anchors, about the size of the existing
held-out evaluation). Each anchor fires 32 of 8,192 gates per site, so an average gate
fires about 8 times per 2,048-anchor pool, or about 400 times in 50 pools. A gate ten
times rarer than average still has about 40 firings to estimate its β. Almost every gate
fires at 500M (8,191 PRE and 8,189 MID active in the quick Intruder/Jaccard report), but
frequency is skewed: the top 1% of gates take about 10–12% of firings.

**`edges.max_anchors_per_gate = 20`**, `anchor_sampling = seeded_random`,
`anchor_seed = 0`, **`anchors.edge_pools = 25`**. That's 51,200 anchors, where an average
gate fires about 200 times, so the cap of 20 fills for all but the rarest gates.

## 3. Cost check before a full run

The edge stage needs about one backward pass per (anchor, representative, model).
Upper bound with every screened gate: 16,384 gates × 20 anchors × 3 representatives ×
3 models ≈ 2.9 M backward passes. The time per backward pass on these models has not
been measured. Run the smoke configuration first (1 importance pool, 2 explicit gates,
2 anchors per gate), then decide whether to set `edges.downstream_gates.count` to cap
the number of screened gates per site.

## Tuning log

| Date | Field | Old → new | Evidence | Approved by |
|---|---|---|---|---|
| 2026-11-21 | all of §2 | — → proposed | §1 calculation; 500M held-out FVU and firing statistics | pending |
