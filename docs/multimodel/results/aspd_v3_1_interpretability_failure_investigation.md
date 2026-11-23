# ASPD v3.1 interpretability-failure investigation

Vietnamese hypothesis synthesis: `aspd_v3_1_interpretability_hypothesis_report_vi.md`.

Status: `COMPLETE_WITH_EXPLORATORY_NONHUMAN_H4_AUDIT`

Date: 2026-11-14

Checkpoint: step `61,036`, `500,006,912` accepted aligned-anchor exposures, checkpoint-manifest SHA-256 `735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969`.

## Executive conclusion

The current evidence does **not** support reading the reported 37% Intruder accuracy as a calibrated estimate of semantic interpretability. The most strongly established failure is in the measurement instrument: the Llama judge has severe position dependence, and an independent Mistral-family judge also failed its preregistered reliability gates. A paper-faithful task construction produced 33.65% macro accuracy, but that number remains exploratory because the judge is invalid. The direction and magnitude of any bias relative to human judgement are unknown; the data establish distortion, not specifically downward bias.

There is also exploratory evidence of representation heterogeneity. In 120 frequency-stratified features with 50--100 natural-firing contexts each, 78/120 (65%) passed the predeclared embedding-cluster stability rule. This is consistent with multi-pattern or polysemantic firing, but it is not yet a human-validated estimate of polysemanticity. A blinded nonhuman H4 audit classified 23/59 uncontaminated items (39.0%) as coherent and 18/59 (30.5%) as coherently syntactic, formatting, or token-boundary features. This supports the existence of non-topical features, but most were already in the old judge's high-score band. A bounded three-model causal test found positive signed-effect correspondence only for SmolLM2--Granite, not for either pair involving Gemma; the predeclared H7 verdict is `INCONCLUSIVE`. Capacity shortage and a causal effect of detached internal supervision remain unproven.

The best present classification is therefore:

1. **Primary verified problem:** invalid semantic measurement.
2. **Secondary plausible representation problem:** stable within-feature context heterogeneity.
3. **Unresolved mechanism/training problems:** cross-model correspondence is mixed, while capacity and detached-gradient causal effects require controlled experiments.

If human ratings remain out of scope, the single most informative next measurement experiment is an independently executed, structure-aware local-judge rerating of the same frozen H4 tasks, compared with the original prompt on identical positions. This would test whether prompt scope rather than component quality explains missed structural patterns, without retraining ASPD.

## Provenance and immutable inputs

- Research branch: `aspd-v3.1-interpretability-failure-v1`.
- Final checkpoint: `/blue/thai/vinhnguyen1/aspd_v3/main500m_preparation_v1/runs/aspd_v3_1_three_model_168targets_c8192_k32_500m_v1_resume_step16000_authfix_5ca3b43/checkpoints/step-000061036`.
- Stage-1 extraction: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/extraction/job_51341200`.
- Frozen paper tasks: `extraction/job_51341200/frozen_tasks` (4,000 independent paper-style trials and 2,000 balanced-rotation diagnostic trials).
- Llama evaluation: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/judge/llama/job_51341991`.
- Independent Mistral calibration: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/calibration/mistral/job_51343141`.
- Stage-2 preparation: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/stage2/preparation_git_fccfdc9`.
- Valid context embeddings: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/stage2/embeddings/job_51368322`.
- Cluster analysis: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/stage2/clusters_git_f7688c7`.
- Held-out reconstruction: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/heldout/job_51367637/heldout`.
- Held-out synthesis: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/stage2/heldout_analysis_git_979c8ca`.
- H7 frozen plan: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/h7_functional_correspondence/plan_git_f50b1d7.json`.
- H7 model runs: `h7_functional_correspondence/job_51380041_{0,1,2}`.
- H7 corrected analysis: `h7_functional_correspondence/analysis_git_3f028ee`.
- H7 causal figure: `h7_functional_correspondence/figures_git_fb81273`.
- H4 frozen nonhuman responses: `docs/multimodel/analysis/aspd_v3_1_interpretability_failure_v1/h4_agent_blinded_responses.json`.
- H4 nonhuman analysis: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/stage2/h4_agent_audit_git_5d6114a`.
- No Qwen model was loaded or trained. Source-model weights and the ASPD checkpoint were read-only.

The exact step-13k checkpoint weights no longer exist. Only immutable extraction/judge artifacts and deleted-checkpoint manifests survive. The step-13k and final feature samples overlap in only three component IDs, so a fixed-component checkpoint comparison cannot be reconstructed.

## Was the original ASPD protocol followed?

The original quick evaluation was not paper-faithful. Stage 1 repairs most task-construction differences but remains a declared v3.1 adaptation rather than an exact reproduction.

| Element | Original official implementation | Earlier quick evaluation | Stage 1 |
|---|---|---|---|
| Natural examples | Up to 400/component | Existing natural firing extraction | Up to 400/component, natural `g^s > 0` only |
| Window | 41 tokens, +/-20 | 65 tokens | 41 tokens |
| Eligibility | At least five valid examples | Eligible sampled features | At least five valid examples |
| Components | 200/method | 100 PRE + 100 MID | 200 PRE + 200 MID, declared v3.1 extension |
| Trials/component | 10 independent | One parent x five rotations | 10 independent; rotations separate |
| Donor | Firing-density matched, tolerance 0.05 | Stronger hard-negative matching | Density matched, tolerance 0.05 |
| Position | Random uniform | Five cyclic rotations | Random for paper tasks; balanced rotations diagnostic only |
| Judge decoding | Server defaults | Greedy | Server defaults; no temperature, top-p, or seed sent |
| Split | Training harvest in official code | Held-out test | Document-disjoint test, declared adaptation |
| v3 gate | Not applicable | Shared natural v3 gate | Shared natural z-score/ReLU/BatchTopK gate |

Thus, the answer is: **the quick 37.21% result did not follow the original protocol; Stage 1 follows its core construction closely, with explicit adaptations for PRE/MID shared dictionaries and the held-out split.** Even the repaired score is not scientifically validated because the judge reliability gate failed.

The audit used the official [evaluation guide](https://github.com/tue147/weights-read-write/blob/main/docs/evaluation.md), [Intruder implementation](https://github.com/tue147/weights-read-write/blob/main/aspd/eval/intruder_db.py), [diversity implementation](https://github.com/tue147/weights-read-write/blob/main/aspd/eval/diversity.py), and the [ASPD paper](https://arxiv.org/abs/2609.37731), rather than relying only on the paper narrative.

## Phase A: measurement validity

### A1/H1 -- Paper-faithful Intruder and judge mismatch

The frozen evaluation contains 400 components (200/site) and 4,000 independently constructed trials. Feature-clustered results were:

| Site | Components | Trials | Accuracy | Feature-clustered 95% CI |
|---|---:|---:|---:|---:|
| PRE | 200 | 2,000 | 33.60% | 30.85--36.45% |
| MID | 200 | 2,000 | 33.70% | 31.20--36.25% |
| Macro | 400 | 4,000 | 33.65% | -- |

There were 3,999 valid responses. Predicted-position shares were 3.73%, 41.86%, 20.16%, 7.48%, and 26.78% for positions 1--5. Accuracy conditional on the true position ranged from 8.36% to 53.55%. In the separate balanced-rotation set, only 17/400 parent tasks (4.25%) were correct under all five rotations.

The independent Mistral diagnostic also failed validation: positive-control accuracy 76.4% (below the 80% gate), position-bias p-value `9.25e-6`, position-conditioned accuracy spread 58 percentage points, and failed canonical-consistency gate. Because two model families fail calibration, choosing whichever produces the highest ASPD score would not be legitimate.

**Verdict: PARTIALLY_SUPPORTED.** Evaluation design and judge behavior materially distort the measurement, but the experiment does not identify the sign or size of bias relative to valid human judgement.

### A3/H2 -- Donor ambiguity

The current frozen human package asks whether the donor is a unique semantic intruder, but two independent human responses have not yet been collected. Within the failed Llama measurement, donor density gap was unrelated to accuracy (`rho=0.021`, exploratory p=0.674), while same-source matching correlated negatively (`rho=-0.239`, exploratory p=`1.40e-6`). Neither association establishes construct validity because the judge is invalid.

**Verdict: INCONCLUSIVE.** The three-condition donor experiment and human unique-answer labels are still missing.

### A4/H3 -- Weak natural firings

The calibration/test split was frozen by document hash before inspecting semantic outcomes. On the 77,854-anchor test partition, retained-event coverage was:

| Threshold | PRE coverage / eligible | MID coverage / eligible |
|---|---:|---:|
| Natural `g^s > 0` | 100% / 8,188 | 100% / 8,186 |
| `g^s >= 0.06` | 52.30% / 8,155 | 73.66% / 8,179 |
| `g^s >= 0.08` | 27.43% / 8,030 | 30.49% / 8,014 |
| `g^s >= 0.12` | 12.81% / 7,117 | 13.46% / 7,098 |
| `g^s >= 0.20` | 3.85% / 3,884 | 4.18% / 4,107 |
| Within-feature q50 | 51.13% / 8,128 | 51.38% / 8,128 |
| Within-feature q75 | 28.15% / 8,014 | 28.59% / 8,020 |
| Within-feature q90 | 13.86% / 7,489 | 14.37% / 7,519 |

Across the 400 sampled components, mean and median target activation correlated with failed-judge accuracy (`rho=0.314` and `0.421`). This is a useful signal for designing a human threshold sweep, but not a confirmatory semantic result.

**Verdict: INCONCLUSIVE.** The coverage trade-off is measured; coherence under a valid instrument is not.

### A5/H4 -- Non-topical syntax or structure

A blinded package of 60 components was generated across PRE/MID and failed-judge score strata. At the researcher's direction, Codex completed a single nonhuman qualitative pass. The response file explicitly records `human_evaluator=false`; it is not presented as human evidence and has no independent inter-rater agreement.

During schema inspection, the answer for Feature Set 001 was accidentally exposed. That item was declared contaminated, assigned no scientific response, and excluded before the remaining answer key was opened. The other 59 responses and the exploratory adjudication rule were committed at `4600fd7` before grading.

Results on 59 uncontaminated items were:

| Agent classification | Count | Fraction |
|---|---:|---:|
| Coherent, any category | 23 | 39.0% |
| Syntactic coherence | 10 | 16.9% |
| Formatting/document-structure coherence | 6 | 10.2% |
| Token-boundary coherence | 2 | 3.4% |
| Semantic coherence | 5 | 8.5% |
| Mixed/no discernible pattern | 36 | 61.0% |
| Unique intruder identified | 23 | 39.0% |

The 18 non-topical coherent items comprised 10 PRE and 8 MID components. All 18 agent-declared unique non-topical intruders matched the construction answer key. This 100% conditional accuracy does not prove semantic validity: the key identifies the donor by construction, and the same language-model agent both inferred the pattern and selected the donor.

Crucially, non-topical coherence was concentrated in components the old Llama judge already scored highly: 17/25 high-band items, 1/24 intermediate-band items, and 0/10 zero-band items. Thus the audit supports the existence of syntactic/formatting/boundary features, but does not support the stronger claim that many low-scoring components are clear structural mechanisms missed by the judge.

All three pre-key exploratory gates passed: non-topical coherent fraction at least 25% (observed 30.5%), at least ten unique non-topical tasks (18), and at least 50% construction-key accuracy in that subset (100%).

**Verdict: PARTIALLY_SUPPORTED.** Non-topical coherent components exist in this sample, but this is a single nonhuman audit, not human validation, and it provides little evidence that the original judge missed clear structure in its low-score band.

## Phase B: learned-representation diagnosis

### B1/H5 -- Within-feature clustering

The analysis used 10,311 natural-firing contexts from 120 frequency-stratified features, with 50--100 contexts/feature. Mistral-Nemo embeddings were clustered with candidate `k=2..5`, ten restarts, and bootstrap stability. Seventy-eight features (65%) passed all predeclared silhouette, minimum-cluster-size, and adjusted-Rand stability gates: PRE 45/60 (75%) and MID 33/60 (55%). Selected `k` was 2 for 44 features, 3 for 53, 4 for 20, and 5 for 3.

The result did not increase monotonically with firing frequency: credible fractions were 65% low, 70% medium, and 60% high. Real gallery examples include components whose contexts separate code/HTML/API material from prose or biomedical text from numerical/log-like records.

**Verdict: PARTIALLY_SUPPORTED.** Stable contextual substructure is common in this sample, but human cluster naming and held-out within-cluster versus cross-cluster Intruder tests remain absent.

### B2/H6 -- Capacity pressure

The held-out run evaluated 100,352 aligned validation anchors and all 168 matrix targets. Normalized activation FVU ranged from 0.344 to 0.402 across model/site pairs. Mean matrix FVU was 0.318--0.325 for PRE-associated q/k/v targets and 0.439--0.470 for MID-associated o/MLP targets. MID targets are therefore systematically harder to reconstruct, but this does not identify dictionary capacity as the cause.

Feature frequency had only a very small association with conditional matrix residual (`rho=0.067`, `n=16,384`), and selected context-cluster count was not associated with frequency (`rho=-0.093`, exploratory p=0.315, `n=120`). These results do not show the specific residual/ambiguity pattern expected from a clearly binding global capacity constraint.

No `C=16,384` or `C=32,768` model was trained. Therefore observational residuals and concentration can make a capacity bottleneck plausible, but cannot identify the causal effect of `C`.

**Verdict: INCONCLUSIVE.** The present diagnostics neither establish nor rule out a capacity bottleneck.

### B3/H7 -- Cross-model functional correspondence

The shared PRE/MID gate supplies one firing label, not three model-specific natural labels. The new causal test therefore held canonical prefix/continuation text fixed across all three tokenizers and measured model-specific rank-one ablation effects. Its primary signed outcome was edited-minus-clean continuation log-probability per canonical UTF-8 continuation byte; a negative value means the learned-operator ablation reduced continuation likelihood.

The frozen sample contained 24 components (12 PRE, 12 MID; four from each low/middle/high contribution stratum), with four natural active and four matched inactive contexts per component. For each component/model, the test selected the learned operator with maximum Frobenius norm within the relevant site. It included three norm-matched random rank-one seeds and one random learned-component control. All three model runs passed source-weight restoration and zero-strength invariants and produced 3,456 rows in total.

Component-clustered signed active-effect correspondence was:

| Pair | Spearman rho | Component-bootstrap 95% CI | Sign concordance |
|---|---:|---:|---:|
| Gemma--SmolLM2 | -0.082 | -0.439--0.367 | 41.7% |
| Gemma--Granite | 0.045 | -0.385--0.478 | 54.2% |
| SmolLM2--Granite | 0.532 | 0.090--0.823 | 70.8% |

All-three sign agreement was 33.3% (95% CI 16.7--54.2%), versus a 23.6% point mean across the three random rank-one controls. On the exact same active canonical contexts, pairwise effect correlations ranged from -0.115 to -0.041 and every component-clustered CI crossed zero. Only 4/24 components (16.7%) selected the same strongest matrix-name class in all three models.

The predeclared support rule required either two pairwise rank-correlation upper bounds below zero or learned all-three sign agreement whose upper bound fell below the random-control point. The predeclared not-supported rule required all three pairwise lower bounds above zero plus greater learned sign agreement. Neither rule passed.

**Verdict: INCONCLUSIVE.** SmolLM2 and Granite show bounded evidence of corresponding signed component effects, while Gemma does not align reliably with either. This rejects neither functional mismatch nor correspondence and is not evidence for a universal algorithm. The small stratified sample and model-specific strongest-target selection limit generalization.

### B4/H8 -- Activation-to-parameter alignment and detached gradients

The implementation audit verifies that internal reconstruction consumes detached binary support `g = 1[g^s > 0]`; internal loss therefore has no direct encoder gradient. On held-out data, the activation-code energy proxy and total internal contribution norm were positively aligned (`rho=0.649`, `n=16,384`), and firing frequency was even more strongly aligned with internal contribution (`rho=0.815`). This weakens a simple claim that activation support is unrelated to parameter reconstruction. However, contribution norm is not signed causal specificity or semantic coherence.

No differentiable-internal-supervision retraining was run.

**Verdict: INCONCLUSIVE.** Detached routing is verified, but objective-induced mechanistic mismatch is not.

### B5/H9 -- Step-13k to final learning curve

The training-batch objectives improved between available metric rows:

| Step | Exposures | Activation loss | Internal loss |
|---|---:|---:|---:|
| 13,000 | 106,496,000 | 0.400998 | 0.419016 |
| 61,036 | 500,006,912 | 0.381607 | 0.393049 |

These are training-batch metrics, not held-out FVU. They show continued optimization but do not establish an interpretability plateau. The old step-13k score (about 37.53%) and final quick score (37.21%) are protocol-mismatched, have only three sampled feature IDs in common, and both rely on a position-biased judge.

**Verdict: INCONCLUSIVE.** Exact checkpoint-matched H9 is no longer executable because the step-13k weights are unavailable.

### B6/H10 -- Extreme firing winners

The top 20 features accounted for 12.00% of total observational internal contribution in PRE and 5.60% in MID, despite being only 0.244% of each dictionary. At the broader top-1% level, event shares were 11.84% PRE and 9.82% MID; model-specific contribution shares were 17.77--23.16% PRE and 9.68--11.29% MID. Frequency and internal contribution were strongly correlated (`rho=0.815`).

Concentration does not by itself imply undesirable generality. Stable multi-cluster fractions were 65% low-frequency, 70% middle-frequency, and 60% high-frequency, and cluster count was not associated with frequency. No signed causal-specificity or valid human-coherence measurement exists for the top winners.

**Verdict: PARTIALLY_SUPPORTED.** Extreme winners disproportionately carry reconstruction contribution, but the claim that they are more general or polysemantic is not supported by the current measurements.

## Integrated evidence matrix

The machine-readable evidence matrix is stored beside this report. The final verdicts use only: `SUPPORTED`, `PARTIALLY_SUPPORTED`, `NOT_SUPPORTED`, `INCONCLUSIVE`, or `NOT_TESTED`.

| Hypothesis | Evidence | Supporting result | Contradicting/limiting result | Verdict |
|---|---|---|---|---|
| H1 Judge/protocol | A1/A2 | Two judge families fail reliability; Llama has extreme position dependence | No calibrated human reference, so bias direction unknown | PARTIALLY_SUPPORTED |
| H2 Donor ambiguity | A3 preparation | Source-match association in failed judge; human audit ready | No human unique-answer labels or three donor conditions | INCONCLUSIVE |
| H3 Weak firings | A4 | Activation-strength association; quantified coverage curve | Association uses invalid judge | INCONCLUSIVE |
| H4 Nonsemantic structure | A5 nonhuman audit | 18/59 (30.5%) items are agent-coherent syntactic/formatting/boundary features; 18/18 selected donors match the construction key | Single language-model evaluator; 17/18 non-topical cases are in the old judge's high band; one contaminated item excluded | PARTIALLY_SUPPORTED |
| H5 Polysemanticity | B1 | 78/120 stable multi-cluster features | Embedding clusters are not human meanings | PARTIALLY_SUPPORTED |
| H6 Capacity | B2 | MID matrix FVU is higher than PRE; contribution is concentrated | Residual-frequency effect is tiny; no controlled `C` ablation | INCONCLUSIVE |
| H7 Cross-model mismatch | B3 causal | SmolLM2--Granite signed effects align (`rho=0.532`, CI 0.090--0.823); Gemma pairs are near zero; same-context effects do not align | Neither predeclared support nor not-supported rule passes; 24-component sample and model-specific target selection | INCONCLUSIVE |
| H8 Objective mismatch | B4 | Detached-gradient route verified | Activation energy and internal contribution align (`rho=0.649`); no causal/objective ablation | INCONCLUSIVE |
| H9 Training plateau | B5 | Reconstruction training losses improved | Protocol mismatch; step-13k weights absent | INCONCLUSIVE |
| H10 Extreme winners | B6 | Top 20 carry 12.00% PRE and 5.60% MID contribution | No evidence winners are more multi-cluster; causal specificity unmeasured | PARTIALLY_SUPPORTED |

## Answers to the ten research questions

1. **Is 37% distorted?** Yes. It is not a validated semantic estimate. The sign and magnitude relative to human judgement are unknown.
2. **What fraction is identifiable?** Human-identifiable fraction remains unknown. In the explicitly nonhuman audit, 23/59 (39.0%) were agent-coherent and 18/59 (30.5%) had coherent non-topical patterns.
3. **Do stronger activations improve coherence?** Exploratory association yes; confirmatory answer unknown.
4. **Are weak features polysemantic?** Stable context heterogeneity is common (65% of the stratified 120-feature sample), but semantic polysemanticity is only partially supported.
5. **Is `C=8192` binding?** Plausible but not established. Observational residuals do not discriminate capacity from objective/optimization effects.
6. **Are functions shared across models?** Not established. SmolLM2--Granite effects align in the 24-component test, but both Gemma pairs and all same-context correlations are unreliable; H7 is `INCONCLUSIVE`, not evidence of universal mechanisms.
7. **Does detached internal supervision cause mismatch?** Gradient routing is verified, but the causal contribution is untested; held-out activation/internal contribution alignment is moderately strong rather than absent.
8. **Did interpretability plateau after 13k?** Not established and no longer exactly testable from surviving weights.
9. **What are extreme winners?** They are disproportionately important to reconstruction, especially PRE, but there is no valid evidence yet that they are undesirable or less specific.
10. **Strongest hypothesis and next experiment?** Measurement failure remains strongest. If human review is excluded, run an independent structure-aware local-judge comparison on the same frozen items and positions next.

## Resource and execution record

| Job | Purpose | Result |
|---:|---|---|
| 51341200 | Natural-context extraction | Completed; 100,352 anchors, 16,360 eligible features |
| 51341991 | Llama paper/rotation judge | Completed; scientifically exploratory, position gate failed |
| 51343141 | Independent Mistral calibration | Completed; `JUDGE_VALIDATION_FAILED` |
| 51367743 | First context embedding | Preserved but invalidated by tokenizer-regex warning |
| 51368322 | Corrected context embedding | Completed; valid pinned tokenizer/model revision |
| 51366786 | First held-out attempt | Failed closed on dirty-worktree provenance guard; no scientific output |
| 51367637 | Clean held-out reconstruction | Completed; 100,352 anchors, source weights unchanged, 168 targets |
| 51380041 | Three-model H7 signed causal ablation (three array tasks) | Completed; 3,456 rows, all restoration/invariance gates passed |

Executed jobs used approximately 3.37 B200 GPU-hours and 28.4 allocated CPU-hours including the preserved failed/invalid attempts; the valid scientific jobs account for approximately 3.34 GPU-hours. The complete evaluation namespace occupies 661 MB. API cost was $0. No paid hosted judge was called. No new training was run.

## Figures and machine-readable evidence

Generated figures:

- `stage3/figures_git_979c8ca/judge_position_distribution.png`: actual Llama task results.
- `stage3/figures_git_979c8ca/activation_threshold_coverage.png`: actual natural-firing coverage, not semantic coherence.
- `stage3/figures_git_979c8ca/checkpoint_training_loss.png`: actual training-batch metric rows, not held-out checkpoint parity.
- `stage3/figures_git_979c8ca/cross_model_observational_contribution_alignment.png`: unsigned contribution norms, not causal-effect alignment.
- `h7_functional_correspondence/figures_git_fb81273/cross_model_signed_causal_effect_alignment.png`: actual signed three-model causal evaluation; correspondence is not proof of identical algorithms.
- `stage2/h4_agent_audit_git_5d6114a/agent_h4_taxonomy.png`: actual blinded nonhuman taxonomy labels; explicitly not human evidence.
- `stage2/heldout_analysis_git_979c8ca/feature_frequency_vs_failed_judge.png`: exploratory only because the judge failed.
- `stage2/heldout_analysis_git_979c8ca/cluster_diagnostics.png`: exploratory Mistral-embedding clusters.
- `stage2/heldout_analysis_git_979c8ca/contribution_concentration.png`: held-out observational contribution concentration.
- `stage2/clusters_git_f7688c7/cluster_gallery.json`: actual natural-firing contexts and blinded exploratory cluster assignments.

Accuracy-vs-threshold and coherence-vs-coverage figures were deliberately not produced: no threshold-rejudging with a valid semantic instrument exists. The figure manifest records these omissions instead of synthesizing missing values. Cross-model causal-effect alignment is now reported separately from the older unsigned observational figure.

## Provenance correction

Job 51367637 logged start commit `901edad36c885efdd595dee06159300877d0a9cf` before launching Python. While the already-imported held-out process was running, the reporting tooling was committed in the same worktree. The run manifest consequently read the later HEAD `979c8ca8613779bb3f6add28e2c656194d1f2ff1` at render time. The computation used the module loaded from `901edad`; the manifest's `diagnostic_git_sha` is a late-read provenance error and must not be used as the execution commit.

The same pre-fix renderer wrote `original_space_uses_oracle_target_norm=false` in `heldout_reconstruction.json`. The configuration and authoritative experiment manifest both specify `per_anchor_l2` and `original_space_uses_oracle_target_norm=true`. FVU values are unchanged; this is a metadata-only correction. Both issues are preserved and documented in `provenance_corrections.json` rather than mutating the scientific output.

The first H7 analysis namespace, `analysis_git_f50b1d7`, keyed the secondary same-context statistic only by `context_id`. Reused inactive donor IDs therefore collapsed 96 component-context observations to 30. This did not affect the predeclared component-level H7 verdict or any primary pairwise result. Commit `3f028ee` changed the key to `(site, feature, activity, context_id)` and bootstraps contexts by component; `analysis_git_3f028ee` is authoritative. The superseded output is preserved and was not overwritten; the machine-readable correction is `h7_provenance_correction.json`.

## Limitations and researcher decision

- Human-identifiable coherence and inter-rater agreement remain unknown. The completed H4 audit is a single language-model-agent measurement and can share biases with model judges.
- H6 and H8 cannot become causal conclusions without separately approved training ablations.
- H7 is bounded by 24 stratified components, a continuation-likelihood estimand, and independently selected strongest targets per model; it does not test identity of full algorithms.
- H9 cannot be repaired exactly unless an independently archived step-13k checkpoint is found.

If no human ratings will be collected, the next researcher decision is whether to authorize a bounded, structure-aware rerating of the frozen H4 package with an independent local model family. Capacity or objective retraining should still not be justified from the invalid headline Intruder score alone.
