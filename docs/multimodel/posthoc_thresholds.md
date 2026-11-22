# Post-hoc analysis: pinned thresholds and sample sizes

**Status: APPROVED as starting values (researcher, 2026-11-21).** They are in
`configs/multimodel/posthoc/v31_500m_validation_v1.yaml`. When a value is retuned, add a
row to the [tuning log](#tuning-log), record the reason in
[`posthoc_decisions.md`](posthoc_decisions.md), and create a new versioned config
(`…_v2.yaml`). Never edit a config that already produced results.

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

## 2. Approved starting values

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

**`edges.downstream_gates.min_importance_fires = 50`**. A gate must fire at least 50 times
in the importance stage to be analysed, so no representative is chosen from a β that
rests on a handful of samples.

**`edges.anchors_per_backward = 16`**, **`edges.ablation_check_records = 50`**. Cost and
diagnostic settings only; they do not change any score.

## 3. Cost check before a full run

The edge stage needs about one backward pass per (anchor, representative, model).
Upper bound with every screened gate: 16,384 gates × 20 anchors × 3 representatives ×
3 models ≈ 2.9 M backward passes. The time per backward pass on these models has not
been measured. Run the smoke configuration first (1 importance pool, 2 explicit gates,
2 anchors per gate), then decide whether to set `edges.downstream_gates.count` to cap
the number of screened gates per site.

### Measured on the smoke run (job 52378170, 2026-11-22, engineering only)

Configuration `v31_500m_validation_smoke_v1.yaml`, commit `5625def`, 1 B200. Output:
`/blue/thai/share/multimodel-ASPD/evaluations/minhdatle/posthoc/v31_500m_validation_smoke_v1_job52378170`.
- Job: `COMPLETED 0:0`, 4 min 57 s wall time. Peak CUDA allocation 59.9 GB; host RSS 1.2 GB.
- Backend load (3 models + 3 checkpoint owners): 246 s, once per job.
- Pool preparation (stream + 3-way alignment + natural gates): 13.9 s for the importance
  pool, 6.9 s for the edge pool. Importance capture and accumulation: 0.9 s per pool for
  all three models.
- Edge stage: 18 backward passes in 0.72 s, about 40 ms each, including the 3 forward
  passes. Each pass covered 2 anchors.
- Taylor vs exact ablation (4 checks): within 1–12%, e.g. +0.678 vs +0.686, −1.47 vs −1.65.
- Screening from one pool: 6,517 PRE and 6,519 MID gates fired. 5,308 PRE and 5,013 MID
  gates had non-empty R in at least two models at τ_rep = 0.07. Finite-β quantiles
  (10/50/90/99%): 0.017 / 0.063 / 0.156 / 0.480, consistent with the β_ref scale of §1.

Projection for `v31_500m_validation_v1.yaml` on 1 B200 (to be re-measured):
- About 8,000–10,000 analysed gates after `min_importance_fires = 50`.
- × 20 anchors × 3 representatives × 3 models gives about 1.5–1.8 M backward passes.
  In the full run, anchors in one backward pass rarely share a gate, so expect about one
  pass per (anchor, representative, model).
- At about 40 ms each, that's about 17–20 GPU-hours for edges. Importance (50 pools) is
  about 15 min, edge pools about 5 min, load 4 min.
- Total: roughly 18–22 GPU-hours on one GPU, or about 5–7 hours wall time split by gate
  over 4 GPUs.

## Tuning log

| Date | Field | Old → new | Evidence | Approved by |
|---|---|---|---|---|
| 2026-11-21 | all of §2 | — → proposed | §1 calculation; 500M held-out FVU and firing statistics | — |
| 2026-11-21 | all of §2 | proposed → approved | researcher "ok" | researcher |
