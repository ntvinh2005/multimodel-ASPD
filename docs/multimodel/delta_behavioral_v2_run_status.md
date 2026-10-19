# Delta behavioral V2 run status

Last updated: 2026-10-18 21:45 EDT

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

## Completed measurement jobs

### Delta feature measurement

- Slurm job: `46605269` (`mm-delta-behavior-v2-measure`)
- Result: `COMPLETED` with exit code `0:0` in 2 minutes 13 seconds
- Ran: 2026-10-18 21:21:39–21:23:52 EDT
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
- Result: `COMPLETED` with exit code `0:0` in 1 minute 18 seconds
- Ran: 2026-10-18 21:24:53–21:26:11 EDT
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

### Evaluator-only score reconciliation

- Slurm job: `46607042` (`mm-delta-behavior-v2-reconcile`)
- Result: `COMPLETED` with exit code `0:0` in 1 minute 6 seconds
- Ran: 2026-10-18 21:40:12–21:41:18 EDT
- Purpose: attach the already completed blind scores to the frozen feature
  metrics because the original measurement finished before the judge
- No responses were regenerated and no activations were re-extracted

## Final audit

- Blind judge: Condition 1 passed; risk score coverage 58/60 (96.7%), paired
  Target−Base mean `+0.569`, bootstrap 95% CI `[0.207, 0.914]`
- Delta representation: Condition 2 failed; 0/40 frozen target-tail features
  passed every confirmatory criterion
- V1 feature 5408: inactive on 0/60 Target-risk responses; mean D 0; q 1
- Scientific decision: **FAIL**
- Final report:
  `docs/multimodel/results/delta_behavioral_validation_v2_report.md`

For provenance, completed jobs can be checked with:

```bash
sacct -j 46605269,46605270,46607042 --format=JobID,State,ExitCode,Elapsed,Start,End
```

No causal intervention and no ASPD are authorized in this phase.
