# Delta behavioral V2 run status

Last updated: 2026-10-18 21:02 EDT

## Completed 512-token attempt

- Slurm job: `46601637` (`mm-delta-behavior-v2-gen`)
- Result: `COMPLETED` with exit code `0:0` in 7 minutes 4 seconds
- Frozen response set: `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_512`
- Premeasurement audit: 79 of 360 responses reached the 512-token cap
  (`21.94%`), exceeding the preregistered 5% threshold
- Breakdown: Base 79/180 (`43.89%`); Target 0/180 (`0%`)
- Decision: `measurement_allowed=false`; this response set must not be used
  for confirmatory scoring or Delta activation measurement

## Active job

- Slurm job: `46603937` (`mm-delta-behavior-v2-gen`)
- State at last check: `RUNNING`
- Started: 2026-10-18 21:01:23 EDT
- Time limit: 30 minutes
- Expected completion: approximately 21:10–21:20 EDT; hard Slurm deadline
  21:31 EDT
- Frozen response destination: `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_1024`
- Log: `slurm/logs/mm-delta-behavior-v2-gen-46603937.out`
- This is the single preregistered full-dataset fallback. It regenerates all
  180 prompts and does not selectively reuse any 512-token response.

## Required continuation

Check:

```bash
sacct -j 46603937 --format=JobID,State,ExitCode,Elapsed,Start,End
```

If completed, inspect
`responses_1024/behavioral_v2_premeasurement_audit.json` before any scoring or
feature measurement. If it passes the frozen cap rule, continue to the blind
judge and Delta activation measurement. This 1,024-token run is the one allowed
fallback; do not silently submit a third generation attempt if it fails.

No causal intervention and no ASPD are authorized in this phase.
