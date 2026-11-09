# ASPD v3.1 final position-controlled Intruder validation

## Executive status

The final 10M-exposure comparison is in progress. This report is the frozen protocol and will be updated only with outputs from the preregistered analysis. No ASPD checkpoint is being trained by this evaluation.

Current scientific status: `INCONCLUSIVE — FINAL JUDGE INFERENCE NOT YET RUN`.

## Scope and invariants

The comparison is between the matched step-1,221, 10,002,432-exposure `full_only_control` and `per_anchor_zscore` checkpoints. The source models remain the pinned Gemma-2-2B-IT and SmolLM2-1.7B-Instruct models. No Qwen model, paid API, source-weight update, third ASPD model, or additional ASPD training is permitted by this protocol.

The Intruder score measures semantic coherence of natural feature-firing contexts. It does not establish parameter-level causality, cross-model functional correspondence, or readiness of an unapproved third-model configuration.

## Why the previous judge results were invalid

The Llama-3.3-70B forced-choice run had working positive and negative controls, but 52% of scientific predictions selected displayed position 2. Consensus filtering retained only 123/200 parent tasks and did not remove that bias. The resulting 30.08% accuracy is therefore not a valid arm comparison.

The leave-one-out formulation removed direct answer-position coupling but its fixed top-score and margin rule abstained on 94.5% of scientific tasks. Its 11 answered tasks do not estimate the sampled feature populations.

Neither failed result is reused as the final metric.

## Frozen corrected protocol

The primary judge is pinned `meta-llama/Llama-3.3-70B-Instruct` revision `6f6073b423013f6a7d4d9f39144961bfbfbc386b`, run locally with greedy BF16 inference.

Each parent feature contains four naturally firing target contexts and one context from a different feature. For each parent, five cyclic rotations ensure that every canonical context appears once in each displayed position. A second Latin-style mapping ensures that every canonical context also appears once under each opaque answer label `K`, `L`, `M`, `N`, and `P`. The prompt reveals no arm, feature ID, frequency, activation, or answer annotation.

All five rotations are scored. There is no unanimity or consensus filter. Invalid output is separately reported and counts as incorrect in the intention-to-evaluate score. The parent feature—not a rotation—is the resampling unit.

## Calibration gates

Calibration uses 50 synthetic positive-control parents and 25 synthetic no-unique-intruder diagnostic parents, separate from all ASPD contexts. The forced-choice protocol does not treat a negative-control failure to abstain as an error. Instead, negative controls diagnose prediction stability and residual position/label preference.

The judge is accepted only if all of the following pass before the arm comparison is inspected:

- positive-control rotation accuracy at least 80%;
- valid-response rate at least 95%;
- at least 80% of positive parents have a modal canonical prediction share of at least 0.8;
- correct-position conditional accuracy spread at most 0.15;
- no position cell exceeds the Bonferroni-adjusted binomial upper count at family-wise alpha 0.01, and the five-cell chi-square test has p at least 0.01;
- exact deterministic replay of the first 20 prompts.

The full preregistration is stored in `docs/multimodel/analysis/aspd_v3_zscore_intruder_final_validation_v1/preregistered_protocol.json`.

## Test-set integrity

The prior extraction contains 296 source documents, of which 256 had already appeared in the 200 debug tasks. It is therefore development data, not an untouched confirmation set. The final extraction excludes all 296 document IDs using immutable manifest SHA-256 `f718dde2a8f68420b56da18a87abae043061f4005a2522a53724d542646da3ef`, then collects 100,352 aligned anchors from subsequent documents in the same frozen test buckets 10–19.

Context selection uses canonical source-text windows. Four target contexts must come from distinct documents. The intruder must use a disjoint document and an anchor where the target feature is naturally inactive. Exact and high-Jaccard near-duplicate contexts are excluded, and intruders are matched preferentially on source group, text shape, and length.

## Sampling and estimands

Up to 100 eligible features are sampled independently for each arm/site from low-, medium-, and high-firing strata using a frozen seed. Eligibility requires at least four natural fires and four unique retained documents. Any context-construction failure is retained in the coverage accounting rather than silently replaced by a convenient feature.

The primary accuracy estimand is uniform over eligible features, reweighted by the eligible feature population in each firing-frequency stratum. The report will separately show the raw balanced-sample mean and the activity-weighted result. Feature-level bootstrap resampling preserves all five rotations within a parent.

## Preregistered decision rule

- `SEMANTIC_INTERPRETABILITY_IMPROVED`: the macro treatment-effect 95% interval is above zero and neither site point estimate is negative.
- `INTERPRETABILITY_TRADEOFF`: the macro interval is below zero.
- `INTERPRETABILITY_COMPARABLE_COVERAGE_IMPROVED`: both site intervals include zero while eligibility exceeds 2x control at both sites.
- `JUDGE_VALIDATION_FAILED`: calibration or the final position diagnostic fails.
- `INCONCLUSIVE`: any other validated-judge outcome.

## Results

Pending calibration and final inference.

## Resource accounting

Pending. Historical failed-judge jobs are not counted as compute consumed by this final protocol.

## Final questions

Q1–Q7 and the main-training-preparation recommendation remain pending the preregistered result. Regardless of outcome, Intruder success will not be described as proof of shared causal mechanisms.
