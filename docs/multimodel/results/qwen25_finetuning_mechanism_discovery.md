# Qwen2.5 Fine-Tuning Mechanism Discovery

Status: `RESULTS_AUDITED_NO_P6`

This exploratory follow-up is anchored on Delta-Crosscoder feature 10122. It does not alter the
frozen V3 evaluation or upgrade its `BORDERLINE_PENDING_MANUAL_REVIEW` decision. The objective is
to discover candidate ASPD computations changed by post-training, not to compare methods or prove
that feature 10122 is a risk-seeking mechanism.

## Frozen inputs

- Delta checkpoint: `checkpoint_00002048.pt`, SHA256
  `838010d85b5793b9c0e2882dd0bc3ce427ebae5652cb84423acaa14289504477`.
- ASPD checkpoint: `checkpoint_00000200.pt`, SHA256
  `b5c27fdf641924ffe4511ac4b75ff1468274593c250d878682527df9e1383800`.
- Both decompositions use the exact pinned Qwen2.5 Base/Target pair and layer 14 input residual.
- Discovery set: 24 new fixed prompt/continuation sequences (8 sports-risk, 8 sports-control,
  8 neutral), with explicit lexical controls and no exact V3 prompt overlap.

## Delta 10122 causal pilot

The intervention uses the learned representation-difference contribution
`z_10122 * (decoder_target - decoder_base)` in raw residual-stream units. It subtracts this vector
from Target or adds it to Base at the layer-14 input. Conditions include alpha 0, 0.5, and 1.0 and
a deterministic orthogonal direction with matched L2 norm.

The feature value is computed from clean Base and Target activations on identical token prefixes
and held fixed during each patched forward. Qwen attention is causal and thresholded Delta encoding
is token-local, so a token's patch does not use later teacher-forced tokens. This is nevertheless a
teacher-forced fixed-continuation pilot, not an autoregressive mediation claim.

Artifact:
`out/multimodel/mechanism_discovery/delta_10122_causal_pilot/delta_10122_causal_pilot.json`.

### Delta result

The no-op path reproduced clean logits exactly (`max_abs_diff = 0`). The learned and orthogonal
control directions had equal norm `0.156341` and cosine approximately zero. Feature 10122 fired at
40/1,544 token positions in 10/24 sequences:

| Category | Sequences with a fire | Prompt fires | Continuation fires |
| --- | ---: | ---: | ---: |
| Sports risk | 3/8 | 6 | 3 |
| Sports control | 4/8 | 13 | 3 |
| Neutral | 3/8 | 13 | 2 |

No sequence fired at the prompt decision point. Exact windows put the strongest fires directly on
`overthink`, `overcomplicating`, `worrying`, `second-guessing`, and nearby tokens. The same pattern
appeared in risk, benign sports, and neutral text.

On the ten firing sequences at alpha 1, Target subtraction produced mean continuation KL
`0.000851` and logit RMS change `0.04309`; the norm-matched control produced `0.000844` and
`0.04181`. Base addition produced KL `0.000725` and RMS `0.04806`, versus control `0.000714` and
`0.04523`. These are measurable perturbation effects, but the learned direction was not reliably
larger than the control. Target subtraction did not consistently move Target toward Base, and Base
addition was on average farther from Target (`+0.00770` RMS versus `+0.00290` for control). Relevant
continuation log-probability changes varied in sign and were not risk-specific.

Conclusion: feature 10122 is best supported as a lexical/response-style-sensitive representation
around hesitation and simplification. This pilot does **not** support risk-specific mediation or a
feature-specific causal effect beyond a norm-matched residual perturbation. The frozen V3 status
remains `BORDERLINE_PENDING_MANUAL_REVIEW`.

## ASPD correspondence and mechanism measurement

The common exact-token pass measured thresholded Delta-10122 activation and all 8,192 ASPD codes at
the same token positions. Candidate ranking uses positive Spearman/Pearson association, sparse
Jaccard overlap, binary phi, co-activation excess above each component's marginal fire rate, and
stability across risk/control/neutral subsets. Correlation is only candidate correspondence.

For the top 20 candidates, the analysis joins existing P1–P5 tensors with measured Base and Target
rank-one contributions over q/k/v/o/gate/up/down projections. It separates the exact identity

`target(x_target) - base(x_base)`

into an operator-on-Base-input term and a Target-operator input-change term. Relative P5 geometry
change is weighted by P2 mechanism mass before selection, preventing a near-zero denominator from
promoting a component by itself.

Artifacts:

- `out/multimodel/mechanism_discovery/delta_10122_aspd_correspondence/mechanism_discovery.json`
- `out/multimodel/mechanism_discovery/delta_10122_aspd_correspondence/component_cards.json`
- `out/multimodel/mechanism_discovery/delta_10122_aspd_correspondence/full_correspondence_ranking.csv`

### Correspondence result

The measurement used 1,544 exact aligned token positions. Delta 10122 fired 40 times and ASPD
selected exactly `1,544 * 32 = 49,408` component/token pairs. Associations were modest. Component
310 was strongest: Pearson `0.276`, Spearman `0.279`, binary phi `0.262`, Jaccard `0.125`, and
`P(ASPD active | Delta active) = 0.675` versus marginal fire rate `0.131`. Its subset Spearman was
`0.206` risk, `0.350` sports control, and `0.284` neutral, again arguing against risk specificity.

The first automated score used raw conditional co-activation, which favors dense ASPD components.
The audit replaced that term with marginal-rate excess and binary phi. The complete top-20 set was
unchanged, but the generated automatic final-five label is treated as superseded because the final
order was sensitive to composite weights.

The following five-component inspection set is retained transparently; it is not a confirmed
mechanism set:

| ASPD ID | Role in the audit | Key observation |
| ---: | --- | --- |
| 310 | strongest anchor correspondence | Stable across all subsets; shared activation, slightly Base-heavy mechanism; q_proj is the largest measured change. |
| 295 | lexical candidate | Dense but repeatedly active around “go for it/commit/overthink”; largest changes at gate_proj, k_proj, q_proj. |
| 328 | sparse lexical candidate | Fires on about 5.4% of positions and includes “go for it” contexts; lower absolute computation mass. |
| 390 | anti-overthinking candidate | Exact high-anchor contexts include neutral/control “overcomplicating”; risk-subset association is negative, making it a useful specificity test. |
| 267 | high-mass negative control | Largest measured mass/change, but top contexts are repeated system-template tokens rather than the anchor semantics. |

Other automatically proposed dense components (7, 227, 482) remain in the full audit trail but are
not promoted: their representative maxima are dominated by the common Qwen system template.

### ASPD mechanism fidelity and geometry

Activation reconstruction on this small discovery set was strong (FVU `0.0117` Base and `0.0120`
Target). Parameterized matrix reconstruction was weak: matrix FVU ranged from `0.829` to `1.685`,
with several values above 1. The original validation metrics were similarly weak (aggregate
internal FVU about `0.903` Base and `0.897` Target at step 200), so this is a limitation of the
engineering pilot checkpoint rather than a new inference from the selected prompts.

The measured contribution changes most often involved q_proj and the MLP gate/up projections, but
these are observational changes inside a low-fidelity approximation. The candidates all had nearly
shared P1 activation attribution, and none supplied evidence for a concentrated Target mechanism.
No causal circuit edge is claimed. The only candidate subgraph justified by the measurements is a
speculative local cluster connecting shared lexical/style activations to q_proj and gate/up sites;
component 267 acts as a template-sensitive negative control.

### P6 decision

No ASPD P6 job is submitted. A P6 ablation would remove approximate rank-one terms from matrix
outputs whose reconstruction FVU is close to or worse than the mean baseline, while the anchor's
own learned intervention did not beat its norm-matched control. That combination does not support
a clear component-specific causal hypothesis. A defensible next causal step requires a better
fidelity ASPD checkpoint or an independently confirmed hypothesis frozen from this discovery set.

## Scientific boundaries

- V3 informed selection of the anchor, so these measurements are discovery evidence only.
- ASPD BatchTopK support is measured over the supplied discovery set and is teacher-forced.
- ASPD rank-one terms reconstruct configured matrix computations; they are not an exact
  decomposition of the fine-tuning weight update.
- The reported computation differences are algebraically exact for learned ASPD terms but are not
  faithful explanations of original matrix outputs when reconstruction FVU is high.
- No co-activation is called a circuit and no edge is called causal.

## Run status

Submitted from local Git commit `c5950fc9af93472ab3d26c311107cb18edffd8d4` at
`2026-10-19T18:26:20-04:00`:

| Job | Slurm ID | Requested resources | Final state | Time limit | Active runtime |
| --- | ---: | --- | --- | --- | --- |
| Delta 10122 causal pilot | 46736349 | 1 B200, 8 CPU, 80 GB RAM | `COMPLETED 0:0` | 1 hour | 1 minute 32 seconds |
| Delta→ASPD discovery | 46736350 | 1 B200, 8 CPU, 128 GB RAM | `COMPLETED 0:0` | 2 hours | 2 minutes 18 seconds |

Logs:

- `slurm/logs/mm-delta10122-causal-46736349.out` and `.err`.
- `slurm/logs/mm-delta-aspd-discovery-46736350.out` and `.err`.

No cache generation, decomposition training, ASPD P6 ablation, autoregressive causal campaign, or
dependent Slurm job was submitted.

The exact next scientific action is to freeze the lexical/style and q_proj/gate-up hypotheses, then
test them on new held-out examples after improving or retraining an ASPD decomposition with
materially better internal reconstruction fidelity. Re-running P6 on the current pilot is not
recommended.
