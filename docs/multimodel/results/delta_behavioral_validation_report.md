# Delta behavioral validation report

Date: 2026-10-18  
Model pair: `unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee` (base) → `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d` (target adapter)
Hook: `model.layers.14` input  
Delta checkpoint: `out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control/checkpoint_00002048.pt`

Provenance correction: the original version of this report mistakenly described
the pair as Qwen2.5-1.5B. This was a documentation typo only. The frozen training
checkpoint, executed config, and behavioral cache manifest independently record
the 7B model IDs and revisions above; the checkpoint tensors have activation
width 3,584. Historical run artifacts were not modified.

## Run status

The frozen behavioral cache generation completed successfully in Slurm job `46574905`:

- 80/80 prompts succeeded (40 risk, 20 sports controls, 20 neutral).
- 160 paired sequences and 45,518 valid activation tokens were cached.
- Base generated 80 responses; target generated 80 responses.
- Cache fingerprint: `6bc0c16a37e25df3dadbb59d0a498c948694bab080e7dead335e586f036b8284`.
- Prompt manifest SHA256: `64a8705b72668d850feeecec6e16815b7c54e537332593a172b8a2f003431ca2`.

The existing Delta checkpoint was evaluated on the 100 previously selected features in job `46576344`; it completed with exit code 0 in 40 seconds. The evaluator uses exact token pairing and writes:

- `behavioral_activation_metrics.json`
- `behavioral_top_contexts.json`
- `behavioral_candidate_summary.json`

under `out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control/analysis/`.

## Observed metrics

The selected feature set contains 40 `target_tail`, 20 `base_tail`, 20 `balanced_delta`, and 20 `shared_control` features. Diagnostic labels are:

- 56 formatting/template candidates
- 42 generic response candidates
- 1 behavior-selective candidate
- 1 topic-selective candidate

The current 20-example sports-control group is sparse for this feature set: 55/100 features have zero sports-control response fire rate, and 51/100 have zero neutral response fire rate. Consequently, raw risk/sports and risk/neutral ratios can become very large because the evaluator adds `epsilon=1e-9`. These ratios are diagnostic only and are not evidence of a behavioral effect.

The strongest raw ratios therefore should not be interpreted as discoveries. The evaluator intentionally reports `behavioral_scores_available: false`; no refusal, safety, preference, or other causal behavioral score was generated.

## Scientific status

**Pipeline status: PASS. Scientific behavioral claim: INSUFFICIENT EVIDENCE / BORDERLINE.**

The benchmark, paired generation, activation cache, feature evaluation, bootstrap fields, and dashboard artifacts are all present and reproducible. However, this run does not support a causal or robust behavioral conclusion. It is appropriate for frontend inspection and for selecting candidates for a larger, more diverse follow-up benchmark. Any next behavioral run should increase the sports and neutral control counts and report uncertainty with zero-inflated controls explicitly.

This V1 phase did not run ASPD and makes no causal claim. Its result remains
exploratory and BORDERLINE. The next step is the separately preregistered V2
confirmatory behavioral validation; neither causal intervention nor same-pair
ASPD follows automatically from V1.
