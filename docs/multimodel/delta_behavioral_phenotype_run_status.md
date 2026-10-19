# Delta behavioral phenotype diagnostic run status

Last updated: 2026-10-18 EDT

## Active job

- Slurm job: `46610367` (`mm-delta-phenotype`)
- State at submission: `PENDING (Priority)`
- Partition/resource: one B200, 14 CPUs, 96 GB RAM
- Time limit: 45 minutes
- Scheduler start time: unknown; `sbatch --test-only` estimated
  `2026-10-20 17:47 EDT`, but this estimate may move
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

## Continuation

Check once the job has left the queue or the user reports completion:

```bash
sacct -j 46610367 --format=JobID,State,ExitCode,Elapsed,Start,End,NodeList
```

Then inspect, in order:

1. `responses_1024/behavioral_phenotype_calibration_gemma2_9b.json`
2. `responses_1024/behavioral_phenotype_scores_gemma2_9b.json`
3. `analysis/behavioral_phenotype_delta_metrics.json`
4. `analysis/behavioral_phenotype_candidate_summary.json`

If calibration fails or score validity is below 98%, issue `JUDGE_FAILURE` and
do not interpret Delta. Otherwise audit statistical candidates and their saved
exact-token V2 contexts, write
`docs/multimodel/results/delta_behavioral_phenotype_diagnostic.md`, and issue
exactly `PHENOTYPE_SIGNAL_FOUND` or `NO_PHENOTYPE_SIGNAL`.

Do not retrain Delta, run causal intervention, run ASPD, regenerate responses,
or regenerate activations.
