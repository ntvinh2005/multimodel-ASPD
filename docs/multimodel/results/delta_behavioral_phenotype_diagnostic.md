# Delta realized-phenotype diagnostic

Status: **FINAL — NO_PHENOTYPE_SIGNAL**

Date: 2026-10-18 EDT

The current reduced-scale Delta checkpoint and frozen candidate set did not
yield a robust held-out representation of the realized physical-risk
phenotype. Replacing prompt category with an independently judged response
phenotype therefore does not rescue the preregistered V2 representation
failure. This is an exploratory diagnostic and does not alter the
preregistered V2 result: Condition 1 PASS, Condition 2 FAIL, final decision
FAIL.

This result is specific to the current checkpoint and frozen 100-feature set.
It is not evidence that Delta-Crosscoder cannot recover the behavior in
general.

## Frozen inputs and execution

- Canonical responses:
  `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_1024`.
- Response fingerprint:
  `b177c1327c386b39ed9bc238052fc8846cdc5a648b74bc26dc1a29618ce7103b`.
- Existing activation cache:
  `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/activation_cache_1024`.
- Existing Delta checkpoint:
  `out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control/checkpoint_00002048.pt`.
- Frozen feature family: exactly 100 V2 features: 40 target-tail, 20
  base-tail, 20 balanced-Delta, and 20 shared controls.
- New judge:
  `google/gemma-2-9b-it@11c9b309abf73637e4b6f9a3fa1e92e615547819`.
- Slurm job: `46610367` (`COMPLETED 0:0`), 2026-10-18 22:21:57–22:23:31
  EDT, elapsed 00:01:34 on one B200.
- Evaluator runtime: 19.94 seconds; peak GPU memory 4.74 GB.

The run performed no Qwen inference, response generation, activation
extraction, checkpoint update, feature search, retraining, causal
intervention, or ASPD. It reused the response artifact, activation cache,
checkpoint, and frozen feature IDs.

## Judge gates

Gemma judged each response independently and was blind to Base/Target
identity, feature group, prompt-pair outcome, activations, and previous judge
scores. Only risk and sports-control responses were included; neutral rows
were excluded.

| Gate | Result |
|---|---:|
| Synthetic calibration | 12/12 exact; required at least 10/12 |
| Extreme-label inversions | 0 |
| Benchmark rows | 240/240 valid |
| Parse failures | 0 |
| Valid fraction | 100%; required at least 98% |
| Generation | greedy, temperature 0, `do_sample=false`, at most 32 new tokens |

Both judge gates passed. No malformed score was imputed and no judge was
substituted.

## Model-level sanity diagnostic

This section is exploratory and does not replace preregistered Condition 1.

| Prompt category | Generator | Mean score | Histogram 0/1/2/3 | Fraction score >= 2 | Target−Base mean |
|---|---|---:|---:|---:|---:|
| Risk | Base | 0.1333 | 53 / 6 / 1 / 0 | 0.0167 | — |
| Risk | Target | 1.9500 | 2 / 4 / 49 / 5 | 0.9000 | +1.8167 |
| Sports control | Base | 0.0000 | 60 / 0 / 0 / 0 | 0.0000 | — |
| Sports control | Target | 1.9833 | 1 / 1 / 56 / 2 | 0.9667 | +1.9833 |
| Sports pooled | Base | 0.0667 | 113 / 6 / 1 / 0 | 0.0083 | — |
| Sports pooled | Target | 1.9667 | 3 / 5 / 105 / 7 | 0.9333 | +1.9000 |

The diagnostic confirms the motivating confound: Target sports-control
responses were at least as risk-seeking as Target risk-prompt responses. Prompt
category was therefore a poor proxy for realized response stance in this
artifact.

The phenotype distribution is also highly imbalanced: only 8/120 Target
sports responses scored 0 or 1, while 112/120 scored 2 or 3. In the matched
analysis, 45/60 pairs had phenotype difference `delta S = 0`; the remaining
differences were -2 (2 pairs), -1 (6), +1 (6), and +2 (1). These facts limit
precision, especially for binary effects and within-pair correlation, but do
not create a target-tail survivor.

## Statistical audit

For each frozen feature, Analysis A correlates the maximum positive
thresholded assistant-response activation with the Target response phenotype
score over 120 sports responses. Analysis B correlates within-pair activation
difference `delta A` with within-pair phenotype difference `delta S` over 60
matched Target pairs. Both Spearman families were BH-corrected across exactly
100 features. Binary effects are mean activation at score >= 2 minus mean
activation at score <= 1 with 2,000 bootstrap resamples and seed 42.

| Candidate group | Features | Support >= 15/120 | Analysis A q<0.10 | Analysis B q<0.10 | Statistical candidates |
|---|---:|---:|---:|---:|---:|
| target_tail | 40 | 4 | 0 | 0 | 0 |
| base_tail | 20 | 14 | 0 | 0 | 0 |
| balanced_delta | 20 | 0 | 0 | 0 | 0 |
| shared_control | 20 | 20 | 1 | 0 | 0 target-tail candidates |

The only corrected Analysis-A association was shared-control feature 3389
(Spearman rho 0.3099, q 0.0573). It was active on all 120 Target sports
responses, had a positive binary effect of 1.0722 with bootstrap 95% interval
[0.7674, 1.3614], and showed positive uncorrected associations within both
prompt categories. Exact-token maxima often occurred on semantically
compatible danger language such as skipping redundancy, omitting a roll cage,
or attempting a stunt without a mat or spotter. Nevertheless, feature 3389 is
a shared control rather than target-tail feature, and its matched-pair result
did not survive correction (rho 0.2059, q 0.8814). It cannot satisfy the
specified rescue decision.

## Required feature audit

Support is `pooled/120 (risk/60; sports-control/60)`. `Effect` is the binary
phenotype activation difference followed by its bootstrap 95% interval.
Boundary is the fraction of saved top exact-token contexts whose maximum is a
special or response-boundary token.

| Feature | Group | Support | Pearson A | Spearman A | A BH q | Effect [95% CI] | Pair Spearman | Pair BH q | Boundary | Diagnostic result |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5408 | target_tail | 0/120 (0/60; 0/60) | NA | NA | 1.0000 | 0.0000 [0.0000, 0.0000] | NA | 1.0000 | 0.0000 | inactive; no signal |
| 12802 | target_tail | 16/120 (12/60; 4/60) | -0.0254 | -0.0594 | 1.0000 | +0.0123 [-0.3445, 0.2385] | +0.0103 | 1.0000 | 0.7500 | negative rho, no q, boundary dominated |
| 7887 | target_tail | 16/120 (10/60; 6/60) | -0.0761 | -0.1140 | 1.0000 | -0.1721 [-0.6009, 0.2183] | -0.2302 | 0.7456 | 1.0000 | negative effects; boundary dominated |
| 7224 | target_tail | 24/120 (12/60; 12/60) | -0.0463 | -0.0980 | 1.0000 | -0.0585 [-0.6751, 0.4144] | -0.0831 | 1.0000 | 1.0000 | negative effects; boundary dominated |
| 8849 | target_tail | 120/120 (60/60; 60/60) | +0.0651 | +0.1043 | 1.0000 | +0.2069 [-0.0583, 0.4627] | -0.0139 | 1.0000 | 1.0000 | no q; boundary dominated |
| 13944 | base_tail | 120/120 (60/60; 60/60) | +0.1307 | +0.1652 | 0.7922 | +0.0804 [-0.0962, 0.2579] | +0.0800 | 1.0000 | 1.0000 | control, no q; boundary dominated |

Feature 5408, the V1 candidate, remains visible but receives no special
statistical treatment. It never activates on any Target sports response and
cannot be rescued by relabeling response phenotype.

## Top 10 positive Analysis-A associations

These rows are ranked by positive Spearman association, not selected after
correction. The same support and effect notation applies.

| Feature | Group | Support | Pearson A | Spearman A | A BH q | Effect [95% CI] | Pair Spearman | Pair BH q | Boundary | Diagnostic result |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3389 | shared_control | 120/120 (60/60; 60/60) | +0.2629 | +0.3099 | 0.0573 | +1.0722 [0.7674, 1.3614] | +0.2059 | 0.8814 | 0.0333 | corrected control association; not target-tail |
| 6899 | balanced_delta | 1/120 (1/60; 0/60) | +0.2124 | +0.2604 | 0.2040 | +0.0182 [0.0000, 0.0546] | +0.2575 | 0.7456 | 1.0000 | fails support, q, and boundary |
| 11103 | base_tail | 47/120 (26/60; 21/60) | +0.1524 | +0.2292 | 0.2550 | +0.3327 [-0.1718, 0.7560] | +0.1264 | 1.0000 | 0.6538 | control; no q; boundary dominated |
| 17084 | target_tail | 2/120 (2/60; 0/60) | +0.1607 | +0.1880 | 0.5493 | +0.0305 [0.0000, 0.0768] | +0.1901 | 1.0000 | 0.0000 | target-tail but fails support, q, breadth |
| 3761 | base_tail | 120/120 (60/60; 60/60) | +0.1189 | +0.1879 | 0.5493 | +0.2160 [-0.1141, 0.4972] | +0.0275 | 1.0000 | 1.0000 | control; no q; boundary dominated |
| 12914 | base_tail | 79/120 (41/60; 38/60) | +0.0945 | +0.1843 | 0.5493 | +0.0163 [-0.7111, 0.7928] | +0.3250 | 0.3766 | 0.0000 | control; neither family survives BH |
| 13944 | base_tail | 120/120 (60/60; 60/60) | +0.1307 | +0.1652 | 0.7922 | +0.0804 [-0.0962, 0.2579] | +0.0800 | 1.0000 | 1.0000 | control; no q; boundary dominated |
| 3424 | shared_control | 120/120 (60/60; 60/60) | +0.1272 | +0.1471 | 1.0000 | +0.5406 [-0.0075, 1.0242] | +0.1546 | 1.0000 | 0.0167 | control; no q |
| 3103 | shared_control | 120/120 (60/60; 60/60) | +0.1313 | +0.1389 | 1.0000 | +0.3470 [-0.2116, 0.8464] | +0.2264 | 0.7456 | 0.0833 | control; no q |
| 13420 | base_tail | 51/120 (22/60; 29/60) | +0.1461 | +0.1303 | 1.0000 | +0.6753 [0.2025, 1.0372] | -0.0936 | 1.0000 | 0.5909 | control; no q; boundary dominated |

The strongest target-tail Analysis-A row is feature 17084, but it fires on
only 2/120 responses, occurs in only one positive-effect sport domain, and has
q 0.5493. The target-tail features with adequate support are 12802, 7887,
7224, and 8849. All fail corrected association, and every one is formatting
dominated, with boundary fractions 0.75/1.00/1.00/1.00 respectively; the first
three also have negative Analysis-A Spearman correlations.

## Diagnostic decision

**NO_PHENOTYPE_SIGNAL.** Zero frozen target-tail features meet the exploratory
phenotype-signal criteria. The prompt-label confound was real at the model
output level, but it does not explain or rescue the V2 Delta failure for this
checkpoint and candidate set. A corrected association in a shared-control
feature shows that the analysis can detect phenotype-related content, but it
does not establish a target-specific Delta representation.

The recommended next step is to retrain Delta with substantially greater
contrastive diversity and preregister a fresh held-out validation. The current
checkpoint was trained by repeatedly cycling a contrastive engineering cache
containing only 32 prompts; this is a checkpoint limitation, not a general
impossibility result. No retraining, causal intervention, or ASPD is authorized
or performed by this diagnostic.

## Output artifacts

- `responses_1024/behavioral_phenotype_calibration_gemma2_9b.json`
- `responses_1024/behavioral_phenotype_scores_gemma2_9b.json`
- `analysis/behavioral_phenotype_delta_metrics.json`
- `analysis/behavioral_phenotype_candidate_summary.json`

