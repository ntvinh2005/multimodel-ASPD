# Crosscoder run status and wake-up schedule

Last updated: 2026-10-17 13:38 America/New_York.

## Current state

- Hugging Face authentication and gated access to both exact Gemma repositories: verified.
- Code verification: Ruff, shell syntax, diff whitespace, and all 48 multimodel tests pass.
- R0 public artifact reproduction: complete.
- R0 local pilot: complete. Jobs `46441587`, `46441590`, and `46441591` all exited 0.
  It verified paired caching, forward/loss, checkpoint reload, and analysis generation. Its tiny-run
  metrics are diagnostic only (FVE total 0.0147, L0 0.77, dead fraction 65.4%).
- R0 full local training was not submitted: the canonical float32 cache is estimated at about
  1.9 TB, but `/blue/thai` currently reports only about 856 GB available. The authors' full
  checkpoint/statistics reproduction remains available under `r0_gemma2_full/published_analysis`.
- R1 Qwen pilot: complete. Jobs `46443705` and `46443706` exited 0 in 74 seconds total. Final
  validation FVE was 0.765 (base 0.746, fine-tuned 0.782), L0 was 13.27, and dead fraction was
  19.6%; all 4,096 feature rows and selected contexts were produced.
- R1 full chain is active:
  - train `46443909`: running on B200 node `c0909a-s5`, started 13:36:37 EDT;
  - analyze `46443910`: pending on successful training completion.

Published R0 artifacts are available at:

`out/multimodel/baselines/crosscoder/r0_gemma2_full/published_analysis/`

They are explicitly marked as a published-reference import, not local training. The official
2.718 GB checkpoint, 41 MB feature table, 2.2 GB example database, and byte-identical public
tokenizer mirror are already in the shared Hugging Face cache. The regenerated artifact contains
73,728 feature records, 37 blindly selected candidates, exact decoded contexts for 34 candidates,
and a separate unfilled `semantic_annotations.json` template so later interpretation cannot alter
the original rank.

## Next check

Wake Codex at or after **14:20 EDT on 2026-10-17**. This is about 45 minutes after R1 full
training began. The analysis job will start automatically after successful training, so no command
is required in the meantime.

```bash
squeue -j 46443909,46443910
tail -n 100 slurm/logs/cc-r1-full-train-46443909.out
tail -n 100 slurm/logs/cc-r1-full-train-46443909.err
```

## Expected durations and wake-up points

Queue wait is not included because it varies with HiPerGator load.

| Stage | Expected active runtime | When to wake Codex |
|---|---:|---|
| R0 pilot cache, first Gemma download | 20–60 min | 60 min after the cache job starts |
| R0 pilot train + analysis | 15–45 min | 45 min after training starts |
| R0 full cache | 6–18 h and about 1.9 TB | 8 h after start for an I/O/progress check |
| R0 full training, one B200 | 18–48 h | 12 h after start, then every 12 h |
| R0 full analysis | 1–4 h | 2 h after start |
| R1 Qwen pilot (existing cache) | 15–45 min | 45 min after start |
| R1 Qwen full train + analysis | 30–120 min | 90 min after start |

These are planning ranges, not guarantees. Each submitted job's observed throughput will be used to
replace the range with a better estimate in this file. Codex should submit, report the job IDs and
wake-up time, then stop rather than sleeping or repeatedly polling.
