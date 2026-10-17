# Crosscoder run status

Last updated: 2026-10-17 14:46 America/New_York.

## Final state

- No Crosscoder job is queued or running.
- Code verification before the final commit: Ruff on `aspd/multimodel` and multimodel tests, shell
  syntax, JSON parsing, diff whitespace, and all 49 multimodel tests pass.
- R0 public artifact reproduction is complete.
- R0 local pilot is complete. Jobs `46441587`, `46441590`, and `46441591` all exited 0. It
  verified paired caching, forward/loss, checkpoint reload, and analysis generation. Its tiny-run
  metrics are diagnostic only: FVE total 0.0147, L0 0.77, and dead fraction 65.4%.
- R0 full local training was not submitted. Its minimal float32 activation cache is estimated at
  about 1.9 TB, while `/blue/thai` had about 856 GB available. The pinned authors' checkpoint,
  statistics, contexts, and tokenizer-verified import are available under
  `r0_gemma2_full/published_analysis`.
- R1 Qwen pilot is complete. Jobs `46443705` and `46443706` exited 0 in 74 seconds total. Final
  validation FVE was 0.765 (base 0.746, fine-tuned 0.782), L0 was 13.27, and dead fraction was
  19.6%. Pilot features are not interpreted scientifically.
- The first R1 full attempt (`46443909`/`46443910`) is retained only as a 512-step diagnostic at
  `r1_qwen3_full_512step_diagnostic`. It exposed too few tokens for the canonical dead-feature
  threshold and is not used in the comparison.
- Corrected R1 full training job `46447674` completed successfully in 37:08. It ran 10,000
  optimizer steps and 20.48M token exposures.
- Corrected R1 analysis job `46447676` completed successfully in 1:01. Tail/context reanalysis job
  `46453900` completed successfully in 0:59.

## R1 full measurements

- validation FVE: 0.993766 total, 0.993692 base, 0.993836 fine-tuned;
- measured validation L0: 115.896 for training BatchTopK `k=100`;
- dead validation features: 7/65,536 (0.0107%);
- relative decoder norm range: 0.1229 to 0.8462, where zero is base-specific and one is
  fine-tuned-specific;
- strict paper thresholds: 0 base-specific, 65,357 shared, 179 intermediate, and 0
  fine-tuned-specific;
- blind inspection set: 16 base-tail, 16 shared-center, and 16 fine-tuned-tail features. Tail
  membership does not override the strict measured classification.

The absence of strict model-exclusive features is a retained negative result. Exact contexts and
tentative post-ranking labels are compared with ASPD in
`docs/multimodel/results/qwen3_crosscoder_aspd_comparison.md`.

## Artifacts

- R0 published reference:
  `out/multimodel/baselines/crosscoder/r0_gemma2_full/published_analysis/`
- R0 local pilot: `out/multimodel/baselines/crosscoder/r0_gemma2_pilot/`
- R1 local pilot: `out/multimodel/baselines/crosscoder/r1_qwen3_pilot/`
- R1 full run: `out/multimodel/baselines/crosscoder/r1_qwen3_full/`
- ASPD comparison input: `out/multimodel/runs/qwen3_1_7b_d0_s1_main/analysis/report/`
- Final report: `docs/multimodel/results/general_model_diffing_final_report.md`

No wake-up time is required: the experiment chain is complete.
