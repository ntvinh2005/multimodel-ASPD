# ASPD v3.1 z-score mechanistic validation

Status: `MECHANISTIC_VALIDATION_PROMISING — LONGER CONFIRMATION REQUIRED`

Date: 2026-11-09

This report evaluates per-anchor feature-axis z-scoring at the matched 10M-exposure
checkpoint. The current Gemma + SmolLM2 system is a two-model diagnostic configuration.
The scientific objective remains three-model ASPD; no third model is approved, and no Qwen
model was executed.

## Executive summary

Per-anchor z-scoring does more than increase firing counts. On a source-document-disjoint
test set it greatly expands natural feature coverage, removes most near-duplicate firing
pairs, and produces learned rank-one operators whose context-specific logit effects exceed
matched controls more often than the original gate. These results support longer
confirmation, but they do not yet establish semantic interpretability or cross-model
functional correspondence.

The strongest new causal result is the learned-minus-random specificity of last-token logit
RMS. Relative to the original-gate arm, the z-score arm improved this contrast by 0.01318
for Gemma (95% component-stratified bootstrap interval [0.00223, 0.02652]) and 0.01136 for
SmolLM2 ([0.00475, 0.01766]). Within the z-score arm, 13/16 Gemma and 12/16 SmolLM2
components favored the learned operator over the mean of three norm-matched random
rank-one controls. The result survives removal of the largest absolute component contrast.

The evidence is still incomplete:

- The validated local Llama judge protocols failed on the scientific Intruder set: the
  forced-choice protocol had severe position bias, while the position-decoupled protocol
  abstained on 94.5% of trials. Semantic coherence is therefore `INCONCLUSIVE`.
- Within-arm causal intervals exclude zero for Gemma, but the SmolLM2 interval against the
  random-rank-one control narrowly includes zero. SmolLM2 also fails against the random
  learned-component control.
- Z-score cross-model effect-magnitude correlations are weak and do not test behavioral
  direction. Shared component IDs are not evidence of shared mechanisms.
- This is one training seed, 16 components per arm, four active and four inactive documents
  per component, and C1 single-matrix ablation only.

The checkpoint is suitable for a staged 25M-exposure confirmation with the same frozen test
protocol. It is not ready for a 500M campaign or for adding an unapproved third model.

## 1. Provenance and comparison validity

The comparison is matched at optimizer step 1,221 and 10,002,432 accepted aligned-anchor
exposures. Both arms use identical ordered batches, source models, revisions, dataset and
alignment manifests, encoder topology, full-only activation objective, original AuxK,
optimizer settings, and initial model-owned states. The primary training intervention is
`gate_logit_normalization: none` versus `per_anchor_feature_zscore`.

| Item | Value |
|---|---|
| Training Git SHA | `5b6365107a72f224bfa3c6a224a54f96372d546c` |
| Control checkpoint manifest | `8f075f9b1dc264b043b649378761b24321fd1433c293ac5627bde75108c37750` |
| Z-score checkpoint manifest | `c18e93a7f86771164acdedc47390eed11560bf492ce5d7201190d63edc5a45f1` |
| Dataset manifest | `a258974d21ceac3457529644be07459518fde7bc28501abde5c67e5e3719286f` |
| Alignment manifest | `bfd825e51bd4b48602c7822b2ed97ffb919cde588886b72352699e7e077d3c56` |
| Gemma revision | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` |
| SmolLM2 revision | `31b70e2e869a7173562077fd711b654946d38674` |

The previous 306-document reconstruction evaluation influenced intervention selection and
is development evidence. It is not called untouched test data. This validation instead
uses test buckets 10–19: 100,352 exactly paired anchors from 296 source documents.

Full machine-readable provenance is in
`docs/multimodel/analysis/aspd_v3_zscore_mechanistic_validation_v1/provenance_audit.json`.

## 2. Architecture and semantics audit

For packed representation (h_i) and shared readout (R), the control gate computes

\[
z_{i,c}=R_c h_i+b_c,\qquad s_{i,c}=\operatorname{ReLU}(z_{i,c}).
\]

The treatment inserts feature-axis normalization for every anchor:

\[
\tilde z_{i,c}=\frac{z_{i,c}-\operatorname{mean}_{c'}z_{i,c'}}
{\sqrt{\operatorname{var}_{c'}z_{i,c'}+10^{-5}}},\qquad
s_{i,c}=\operatorname{ReLU}(\tilde z_{i,c}).
\]

Pool-wide BatchTopK selects continuous values (a_{i,c}) from (s). Activation
reconstruction consumes those continuous codes. Internal reconstruction uses detached
binary support (g_{i,c}=1[a_{i,c}>0]) with model-specific operators

\[
P_{j,c}^{(m)}=u_{j,c}^{(m)}(v_{j,c}^{(m)})^\top.
\]

The intervention adds no learned parameters and forces no support events. It preserves
feature ordering within an anchor, shared component identities, continuous activation
codes, detached binary internal support, and frozen source weights. It does change
cross-anchor score comparability under pool-wide BatchTopK and couples projection-logit
gradients across features. This is a substantive gate formulation change, not a harmless
logging transform.

Implementation evidence:

- z-score gate: `aspd/multimodel/v3/encoder.py:265`;
- frozen intervention configuration:
  `configs/multimodel/aspd_v3/two_model_gate_collapse_zscore_v1.yaml:96`;
- evaluation contracts: `aspd/multimodel/v3/mechanistic_validation.py:33-286`;
- exact reversible parameter edit:
  `aspd/multimodel/v3/mechanistic_validation.py:387`.

## 3. Reconstruction and natural-gate health

The matched development evaluation established that improved utilization was not purchased
by worse reconstruction.

| Metric | Control | Z-score | Treatment − control |
|---|---:|---:|---:|
| Gemma PRE activation FVU | 0.68260 | 0.54783 | -0.13477 |
| Gemma MID activation FVU | 0.68213 | 0.57154 | -0.11059 |
| SmolLM2 PRE activation FVU | 0.70144 | 0.53692 | -0.16453 |
| SmolLM2 MID activation FVU | 0.69975 | 0.56124 | -0.13851 |
| Equal-site internal FVU | 0.64259 | 0.57804 | -0.06454 |

Document-bootstrap intervals for the macro activation and internal deltas were wholly
below zero. Those values are development metrics because the 306 documents were already
inspected when the intervention was selected.

On the disjoint test extraction, each arm/site contains 3,211,264 natural firing events,
exactly 32 per anchor on average. No forced support contributes to the following results.

| Site | Control firing | Z-score firing | Control eligible | Z-score eligible |
|---|---:|---:|---:|---:|
| PRE | 2,396 | 5,522 | 693 | 5,451 |
| MID | 2,975 | 5,506 | 1,503 | 5,398 |

`OBSERVED`: source-model signatures were unchanged, ASPD state was finite, owner-rank
checkpoint restore was correct, and anchor/document identities were exactly paired.

## 4. Evaluation A — Intruder and semantic coherence

### 4.1 Protocols

Two local, unpaid, non-Qwen protocols used
`meta-llama/Llama-3.3-70B-Instruct` revision
`6f6073b423013f6a7d4d9f39144961bfbfbc386b`, greedy BF16 inference.

1. The v2 forced-choice judge evaluates five cyclic rotations per 5-context task and
   requires parent consensus. Semantic paraphrase positive and negative controls calibrate
   accuracy and abstention.
2. The leave-one-out judge independently scores the coherence of each four-context subset.
   It selects an omitted context only when the top coherence is at least 3/4 and the margin
   over the runner-up is at least one point. The candidate's original position is not shown.

Features were not selected only from the most active tail. Each arm/site has 50 tasks
stratified across low, medium, and high frequency, and the eligibility denominators are
reported above.

### 4.2 Results

| Judge | Positive control | Negative control | Scientific coverage | Scientific result | Status |
|---|---:|---:|---:|---:|---|
| Forced-choice v2 | 1.00 | 0.94 | 123/200 consensus | 0.3008 accuracy | Failed: answer position 2 received 52.0% |
| Leave-one-out | 0.80 | 1.00 | 11/200 answered | 5/11 correct | Failed: 94.5% abstention and max position share 45.5% |

The leave-one-out answered subset was control 0/4 and z-score 5/7 (PRE 1/2, MID 4/5).
This is hypothesis-generating only: coverage was post-selection by the judge, the protocol
failed its predeclared position threshold, and the denominators are too small for a stable
arm comparison.

A correctly answered z-score example (feature 2044, PRE, high-frequency stratum) was
described by the judge as a weak formal-attribution/affiliation pattern. An abstained
z-score example (feature 516, PRE, low-frequency stratum) mixed legal, construction,
market, television, and energy contexts. This illustrates the dominant error taxonomy:

- many features have no visible semantic commonality at the retained context length;
- some apparent coherence is document-style or formatting rather than mechanism content;
- forced-choice judgments are susceptible to positional heuristics;
- a strict position-decoupled protocol becomes too conservative to provide coverage.

`CONCLUSION — INCONCLUSIVE`: z-score semantic coherence is not established. The 5/7 versus
0/4 answered-subset difference cannot be used as a confirmatory effect.

## 5. Evaluation B — Jaccard, redundancy, and diversity

Jaccard pairs were sampled within firing-frequency strata, and high-overlap candidates were
found with MinHash/LSH rather than a dense 8,192 by 8,192 matrix.

| Arm/site | Mean Jaccard | Mean excess coactivation | Near-duplicate pairs (J ≥ 0.5) |
|---|---:|---:|---:|
| Control PRE | 0.007019 | 2.98e-5 | 442 |
| Z-score PRE | 0.002913 | 2.57e-6 | 35 |
| Control MID | 0.002015 | 2.16e-6 | 370 |
| Z-score MID | 0.002924 | 3.45e-6 | 82 |

`SUPPORTED`: z-score increases feature and document coverage, sharply reduces domination
by a few features, and removes most near-duplicate pairs detected by the same MinHash/LSH
search at both sites. PRE mean Jaccard
and excess coactivation also improve strongly.

`PARTIALLY SUPPORTED`: diversity does not improve uniformly. MID mean Jaccard and excess
coactivation are slightly higher under z-score. Frequency-stratified results explain part
of this: the collapsed control's low/medium features are extremely rare and therefore have
near-zero overlap, whereas z-score gives them enough support to coactivate. High-frequency
overlap falls under z-score, but low/medium overlap can rise. Lower overlap is therefore not
treated as a standalone interpretability result.

## 6. Evaluation C — direct parameter causal ablation

### 6.1 Design and invariants

The screen sampled 16 Intruder-disjoint components per arm: eight PRE and eight MID, with
3/3/2 low/medium/high-frequency quotas. Every component used four naturally active and four
naturally inactive, source-document-disjoint contexts. Selection did not use causal output.

For each model and component, C1 target selection chose the logical matrix with maximum
learned operator Frobenius norm within the component's site. The source model was edited
temporarily as

\[
W'_j=W_j-\lambda P_{j,c}^{(m)},\quad \lambda\in\{0,0.5,1\}.
\]

Controls were three seeded random rank-one operators and one random learned component,
each scaled to the learned operator's FP32 Frobenius norm on the same target matrix. Every
logical slice was restored by exact copy. All four final tasks passed source-signature and
zero-strength invariance checks.

The primary uncertainty interval resamples components, then the four documents within each
activity condition. The arm contrast additionally preserves site and frequency-stratum
sampling. It does not treat anchors as independent observations.

### 6.2 BF16 realized-edit audit

Because source weights are BF16, the realized weight delta was measured after subtraction.
Gemma's realized/intended norm ratio is about 1.001 and cosine about 0.9986 for learned and
random interventions. SmolLM2 has larger quantization distortion: mean ratios 1.020–1.028
and mean cosines 0.938–0.961. Crucially, learned and random controls are affected similarly.
Raw and per-realized-norm analyses agree qualitatively, but SmolLM2 evidence remains
exploratory because direction distortion is nontrivial.

### 6.3 Context-specific effects versus random controls

The table reports learned-minus-control active/inactive specificity for logit RMS at
strength 1. Positive values mean the learned operator has a larger effect on naturally
active than inactive contexts relative to the matched control.

| Arm/model | Versus mean random rank-one | 95% hierarchical interval | Positive components |
|---|---:|---:|---:|
| Control Gemma | 0.00132 | [-0.00428, 0.00814] | 8/16 |
| Z-score Gemma | 0.01450 | [0.00164, 0.04097] | 13/16 |
| Control SmolLM2 | -0.00382 | [-0.01082, 0.00311] | 4/16 |
| Z-score SmolLM2 | 0.00754 | [-0.00008, 0.01489] | 12/16 |

The z-score-minus-control contrast is positive for both models:

| Model | Difference in specificity contrast | 95% stratum-preserving interval |
|---|---:|---:|
| Gemma | 0.01318 | [0.00223, 0.02652] |
| SmolLM2 | 0.01136 | [0.00475, 0.01766] |

Against the random learned-component control, z-score Gemma specificity is 0.01505
([0.00091, 0.04044]), while SmolLM2 is 0.00437 ([-0.00710, 0.01472]). KL contrasts point
mostly in the same direction but their intervals include zero.

The Gemma mean is influenced by one large MID component, but it is not solely an outlier
result. Removing the largest absolute component leaves a mean specificity advantage of
0.00708; the median is 0.00414. For SmolLM2 the corresponding trimmed mean is 0.00594 and
median 0.00697.

Dose response is not uniform. Logit-RMS effects increase from strength 0.5 to 1.0 for 7/8
z-score Gemma MID, 5/8 Gemma PRE, 3/8 SmolLM2 MID, and 6/8 SmolLM2 PRE components. This
limits claims of stable directional causality.

`PARTIALLY SUPPORTED`: z-score rank-one operators show more context-specific causal signal
than the original-gate operators and norm-matched random rank-one controls, especially in
Gemma. The effect is not yet a reproducible behavioral mechanism: it is a last-token
distribution effect from a small, single-seed, single-matrix screen.

### 6.4 Cross-model correspondence

For z-score, cross-model Spearman correlations across the 16 shared component IDs are 0.497
for active logit-RMS magnitude and 0.218 for active/inactive specificity; Pearson values are
-0.027 and 0.104. Site-level estimates vary in sign.

`NOT SUPPORTED`: these results do not establish functional correspondence. The metric does
not specify a semantic direction, no held-out behavioral prediction was made, and the two
models may express mechanisms at different loci. Shared IDs plus weak magnitude correlation
are insufficient.

## 7. Answers to the required scientific questions

**Q1. Does z-score produce more coherent features at equal exposure?**

`INCONCLUSIVE`. The unbiased judge did not achieve sufficient scientific coverage. The
answered subset favors z-score but is exploratory.

**Q2. Does it increase meaningful diversity rather than merely redistribute firing?**

`SUPPORTED WITH LIMITATIONS`. Coverage, concentration, high-frequency overlap, and
near-duplicate counts improve strongly. Low/medium overlap and MID mean Jaccard do not
uniformly improve, so fragmentation/redundancy is not eliminated.

**Q3. Do components have reproducible, context-specific causal effects?**

`PARTIALLY SUPPORTED`. Z-score improves active/inactive logit-effect specificity in both
models relative to the control arm. Gemma's within-arm interval excludes zero; SmolLM2's
does not quite do so. One seed and four documents per activity are insufficient for a
reproducibility claim.

**Q4. Are effects stronger or more specific than norm-matched random controls?**

`PARTIALLY SUPPORTED`. Yes for z-score Gemma against both control types. SmolLM2 is positive
against mean random rank-one but inconclusive against a random learned component. BF16
realized-edit differences do not explain the arm result, though SmolLM2 direction distortion
adds uncertainty.

**Q5. Is cross-model correspondence meaningful beyond shared IDs?**

`NOT SUPPORTED`. No functional-direction or behavior-level correspondence has been shown.

**Q6. Does interpretability improve with accepted exposures?**

`INCONCLUSIVE`. Only the matched 10M checkpoint has been evaluated.

**Q7. Is evidence sufficient for longer confirmation or eventual three-model training?**

It is sufficient for a staged 25M confirmation of the two-model z-score arm and a matched
control subset. It is not sufficient for 500M training, for a three-model claim, or for
choosing a third model.

## 8. Scientific decision and readiness

Final label: `MECHANISTIC_VALIDATION_PROMISING`.

What is demonstrated:

- natural gate collapse is substantially mitigated through 10M exposures;
- held-out activation and internal reconstruction improve;
- the expanded dictionary is not merely a collection of identical firing patterns;
- z-score components exhibit stronger context-specific parameter-ablation effects than the
  matched original-gate components.

What is unresolved:

- semantic coherence at usable coverage;
- behavioral direction and causal specificity beyond last-token distribution shift;
- cross-model functional correspondence;
- stability at 25M/50M exposures and across training seeds;
- C2 layer-level and C3 component-group behavior.

Readiness:

| Question | Decision |
|---|---|
| Sustained collapse addressed through 10M? | Yes |
| Held-out activation reconstruction acceptable? | Yes, on development reconstruction protocol |
| Internal parameter reconstruction preserved? | Yes |
| Dictionary diversity scientifically meaningful? | Promising, not semantically established |
| Ready for longer confirmation? | Yes |
| Ready for 500M? | No |
| Ready for three-model training? | No; third model remains unapproved |

## 9. Recommended next stage

Continue the existing z-score checkpoint to approximately 25M accepted exposures. Do not
restart the identical trajectory. At that milestone:

1. repeat the same natural-gate extraction and reconstruction protocol;
2. rerun the fixed causal component-sampling frame without selecting on current effect;
3. add a small predeclared behavior-level test for components with discovery-only semantic
   hypotheses, using held-out prompts;
4. require stronger SmolLM2 evidence and cross-model functional direction before 50M;
5. start one independent z-score seed only if the 25M trajectory remains healthy.

Abort or revise before 50M if natural diversity collapses, any model/site activation FVU
regresses materially, internal FVU regresses, or causal specificity becomes indistinguishable
from both random controls. Do not launch 500M automatically.

## 10. Resource and reliability report

- Total new B200 usage for extraction, judges, causal screens, and validation reruns:
  approximately **2.080 B200 GPU-hours**.
- Final causal array: `50363891`, four B200 tasks, all `COMPLETED 0:0` in 1:14–1:35.
- One four-second task in array `50363056` failed the clean-worktree precondition while the
  array was being replaced; remaining pending tasks were cancelled. It produced no
  scientific conclusion. The replacement used the same frozen plan at a clean commit.
- No checkpoint was created, changed, or deleted by this evaluation.
- Evaluation namespace size: approximately 193 MiB.
- Final observed `/orange/thai` headroom: about 203 GiB; inode use 42%.
- All complete control and z-score checkpoints were preserved.

## 11. Reproducible artifacts

Primary code and contracts:

- `aspd/multimodel/cli/analyze_aspd_v3_mechanistic.py`;
- `aspd/multimodel/cli/judge_aspd_v3_intruder.py`;
- `aspd/multimodel/cli/judge_aspd_v3_intruder_loo.py`;
- `aspd/multimodel/cli/plan_aspd_v3_causal.py`;
- `aspd/multimodel/cli/run_aspd_v3_causal_screen.py`;
- `aspd/multimodel/cli/analyze_aspd_v3_causal.py`;
- `configs/multimodel/aspd_v3/zscore_causal_validation_v1.yaml`.

Final causal analysis:

`docs/multimodel/analysis/aspd_v3_zscore_mechanistic_validation_v1/causal_screen_git_f692522/`

The directory contains per-component effects, component contrasts, hierarchical primary
intervals, robust summaries, arm contrasts, BF16 realized-perturbation audit, dose response,
cross-model correlations, and complete input manifest identities.

External immutable evaluation artifacts remain under:

`/orange/thai/vinhnguyen1/aspd_v3/zscore_mechanistic_validation_v1/`

No raw checkpoint or evaluation artifact was overwritten.
