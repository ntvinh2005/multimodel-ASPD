# Delta behavioral V2 run status

Last updated: 2026-10-18 21:20 EDT

## Completed 512-token attempt

- Slurm job: `46601637` (`mm-delta-behavior-v2-gen`)
- Result: `COMPLETED` with exit code `0:0` in 7 minutes 4 seconds
- Frozen response set: `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_512`
- Premeasurement audit: 79 of 360 responses reached the 512-token cap
  (`21.94%`), exceeding the preregistered 5% threshold
- Breakdown: Base 79/180 (`43.89%`); Target 0/180 (`0%`)
- Decision: `measurement_allowed=false`; this response set must not be used
  for confirmatory scoring or Delta activation measurement

## Accepted canonical 1,024-token response set

- Slurm job: `46603937` (`mm-delta-behavior-v2-gen`)
- Result: `COMPLETED` with exit code `0:0` in 9 minutes 31 seconds
- Ran: 2026-10-18 21:01:23–21:10:54 EDT
- Frozen response destination: `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_1024`
- Log: `slurm/logs/mm-delta-behavior-v2-gen-46603937.out`
- Response artifact fingerprint:
  `b177c1327c386b39ed9bc238052fc8846cdc5a648b74bc26dc1a29618ce7103b`
- Response examples SHA256:
  `d147e85a851efbeac6d619578d7a2861223f0ae6abfa87070026d8514d98aeb9`
- Audit: 360 responses, 0 capped, `measurement_allowed=true`
- This response set is frozen. Do not regenerate it or mix it with the failed
  512-token response set.

## Active measurement jobs

### Delta feature measurement

- Slurm job: `46605269` (`mm-delta-behavior-v2-measure`)
- State at 2026-10-18 21:19 EDT: `PENDING (Resources)`
- Time limit: 45 minutes
- Response directory: `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_1024`
- New activation directory: `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/activation_cache_1024`
- Log: `slurm/logs/mm-delta-behavior-v2-measure-46605269.out`
- Exact submission:

```bash
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
RESPONSE_DIR=out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_1024 \
ACTIVATION_DIR=out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/activation_cache_1024 \
sbatch slurm/multimodel/delta_behavioral_v2_measure.sbatch
```

### Blind behavioral judge

- Slurm job: `46605270` (`mm-delta-behavior-v2-judge`)
- State at 2026-10-18 21:19 EDT: `PENDING (Priority)`
- Time limit: 20 minutes
- Response directory: `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_1024`
- Expected score artifact: `responses_1024/behavioral_scores.json`
- Log: `slurm/logs/mm-delta-behavior-v2-judge-46605270.out`
- Exact submission:

```bash
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
RESPONSE_DIR=out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_1024 \
sbatch slurm/multimodel/delta_behavioral_v2_judge.sbatch
```

## Required continuation

Check:

```bash
sacct -j 46605269,46605270 --format=JobID,State,ExitCode,Elapsed,Start,End
```

When both jobs finish successfully, audit their artifacts before scientific
interpretation. Do not rerun either branch or alter the registered judge,
rubric, feature set, or statistics merely because of an unfavorable result.

No causal intervention and no ASPD are authorized in this phase.
