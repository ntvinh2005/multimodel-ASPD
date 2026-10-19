# Delta behavioral phenotype diagnostic run status

Last updated: 2026-10-18 EDT

## Completed job

- Slurm job: `46610367` (`mm-delta-phenotype`)
- Final state: `COMPLETED 0:0`
- Partition/resource: one B200, 14 CPUs, 96 GB RAM
- Time limit: 45 minutes
- Start: 2026-10-18 22:21:57 EDT
- End: 2026-10-18 22:23:31 EDT
- Elapsed: 00:01:34
- Code commit: `d9ec261`
- Log: `slurm/logs/mm-delta-phenotype-46610367.out`
- Error log: `slurm/logs/mm-delta-phenotype-46610367.err`

Exact submission:

```bash
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
sbatch slurm/multimodel/delta_behavioral_phenotype.sbatch
```

## Frozen inputs

- Responses: `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_1024`
- Response fingerprint:
  `b177c1327c386b39ed9bc238052fc8846cdc5a648b74bc26dc1a29618ce7103b`
- Activation cache:
  `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/activation_cache_1024`
- Delta checkpoint:
  `out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_positive_control/checkpoint_00002048.pt`
- Frozen features: the existing 100 V2 features only
- Judge: `google/gemma-2-9b-it@11c9b309abf73637e4b6f9a3fa1e92e615547819`

The launcher contains no activation extraction command and performs no Qwen
inference. Its flow is calibration, 240 blind single-response Gemma judgments,
then evaluator-only reuse of the frozen activation cache.

## Final audit

- Calibration: 12/12 exact, zero extreme inversion; PASS.
- Benchmark judgment: 240/240 valid, zero parse failures; PASS.
- Frozen features: exactly 100 with the required 40/20/20/20 group split.
- Analysis-A BH family: exactly 100 features.
- Analysis-B BH family: exactly 100 features.
- Target-tail corrected associations: 0 in Analysis A, 0 in Analysis B.
- Statistical target-tail candidates pending context review: 0.
- Diagnostic result: **NO_PHENOTYPE_SIGNAL**.

The final scientific report is
`docs/multimodel/results/delta_behavioral_phenotype_diagnostic.md`. The report
records the model-level phenotype sanity check, control-group comparison,
required features, top 10 positive associations, exact-context interpretation,
limitations, and next step.

Do not retrain Delta, run causal intervention, run ASPD, regenerate responses,
or regenerate activations.
