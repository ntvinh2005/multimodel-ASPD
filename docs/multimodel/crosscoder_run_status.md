# Crosscoder run status and wake-up schedule

Last updated: 2026-10-17 13:19 America/New_York.

## Current state

- Hugging Face authentication and gated access to both exact Gemma repositories: verified.
- Code verification: Ruff, shell syntax, diff whitespace, and all 48 multimodel tests pass.
- R0 public artifact reproduction: complete.
- R0 local pilot chain is active:
  - cache `46441587`: running on B200 node `c0901a-s15`, started 13:16:53 EDT;
  - train `46441590`: pending on successful cache completion;
  - analyze `46441591`: pending on successful training completion.
- First live log check showed the authenticated model-file fetch had started without an error.
- R1 Qwen pilot/full: intentionally not submitted until the R0 local pilot passes.

Published R0 artifacts are available at:

`out/multimodel/baselines/crosscoder/r0_gemma2_full/published_analysis/`

They are explicitly marked as a published-reference import, not local training. The official
2.718 GB checkpoint, 41 MB feature table, 2.2 GB example database, and byte-identical public
tokenizer mirror are already in the shared Hugging Face cache. The regenerated artifact contains
73,728 feature records, 37 blindly selected candidates, exact decoded contexts for 34 candidates,
and a separate unfilled `semantic_annotations.json` template so later interpretation cannot alter
the original rank.

## Next check

Wake Codex at or after **14:20 EDT on 2026-10-17**. This is approximately one hour after the
cache stage began. The dependent train and analysis jobs will start automatically after successful
predecessors, so no command is required in the meantime.

```bash
squeue -j 46441587,46441590,46441591
tail -n 100 slurm/logs/cc-r0p-cache-46441587.out
tail -n 100 slurm/logs/cc-r0p-cache-46441587.err
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
