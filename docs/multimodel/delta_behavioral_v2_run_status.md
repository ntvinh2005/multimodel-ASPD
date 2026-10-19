# Delta behavioral V2 run status

Last updated: 2026-10-18 20:20 EDT

## Active job

- Slurm job: `46601637` (`mm-delta-behavior-v2-gen`)
- State at last check: `RUNNING`
- Started: 2026-10-18 20:19:37 EDT
- Time limit: 30 minutes
- Expected completion: approximately 20:32–20:42 EDT
- Frozen response destination: `out/multimodel/eval/qwen25_extreme_sports_behavioral_v2/responses_512`
- Log: `slurm/logs/mm-delta-behavior-v2-gen-46601637.out`

## Required continuation

Check:

```bash
sacct -j 46601637 --format=JobID,State,ExitCode,Elapsed,Start,End
```

If completed, inspect
`responses_512/behavioral_v2_premeasurement_audit.json` before any feature
measurement. If the all-response 512-token cap fraction exceeds 5%, submit the
entire 180-prompt generation exactly once at 1,024 tokens into `responses_1024`.
Do not selectively regenerate responses. Only the response set passing this
rule may be judged or replayed for Delta activation measurement.

No causal intervention and no ASPD are authorized in this phase.
