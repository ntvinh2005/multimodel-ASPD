# Semantic triage run status

Last updated: 2026-10-17 18:48 America/New_York.

## Current stage

- Candidate planning: complete (CPU dry run, 0.29 seconds).
- Planned candidates: 400 unique features, 200 from each measured tail, `fire_count >= 32`.
- Existing support: 29 candidates have retained contexts; 371 require targeted backfill.
- Existing deterministic result: 15 structural-dominated, 14 eligible non-structural among the 29
  candidates with contexts.
- Context backfill job `46481473`: completed successfully in 40 seconds; 371/371 requested
  features received positive retained contexts.
- Refreshed plan: 400/400 have support, 107 are deterministically structural-dominated, and 293
  enter the judge pool.
- Accepted smoke job `46482242`: completed successfully in 39 seconds with 10/10 valid judgments
  and zero parse failures.
- Accepted judge: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, model/tokenizer revision
  `916b56a44061fd5cd7d6a8fb632557ed4f724f60`.
- Smoke load time: 16.15 seconds; generation: 2.80 seconds; total CLI wall clock: 26.59 seconds.
- Smoke tokens: 8,662 prompt and 636 generated; 227.48 generated tokens/second.
- Projected full judge runtime: 98.07 seconds (1.63 minutes), so the 45-minute guardrail passes.
- Full v2 job `46482316` completed in 81 seconds, but its artifact is provisional/rejected because
  the judge returned `content_level=false` even for 94 `topic_content` labels. This exposed a
  prompt/schema consistency gap; the artifact must be replaced rather than interpreted.
- V3 content-consistency smoke job `46482457` completed in 47 seconds with 10/10 valid judgments,
  zero parse failures, and 9 content-level candidates. The v3 schema requires structural and
  positional types to be non-content, and behavior/content types to be content-level.
- V3 smoke load/generation/CLI wall time: 15.63/3.39/29.80 seconds; 9,582 prompt tokens, 612
  generated tokens, and 180.46 generated tokens/second.
- V3 projected full runtime: 114.99 seconds (1.92 minutes), guardrail pass.
- Replacement full v3 job `46482488`: completed successfully in 54 seconds of Slurm elapsed
  time (52 seconds measured by the launcher; 49.15 seconds inside the CLI).
- Final v3 counts: 400 requested, 400 context-supported, 107 deterministically skipped as
  structural, 293 judged, 288 schema-valid, 5 parse failures retained with raw output, and 252
  content-level candidates.
- Final inference timing: 9.65 seconds model load, 34.97 seconds generation, 281,471 prompt
  tokens, 19,387 generated tokens, and 554.39 generated tokens/second.
- Manual-review shortlist: 252 candidates, ordered primarily by the original measured tail rank
  (105 base-tail memberships and 147 target/post-trained-tail memberships).
- Final artifact integrity check passed: all 400 measured classes, norms, fire counts, and blind
  ranks exactly match `feature_metrics.json`; all accepted evidence IDs are in range; the review
  artifact is an exact recomputation of the documented rule; dashboard diagnostics are empty.

## Final state

The accepted scientific artifacts use prompt/schema version
`semantic-triage-v3-content-consistency`. The earlier v2 full run and failed model-smoke attempts
are not accepted artifacts and were overwritten by the v3 outputs.

No semantic label changed a Crosscoder measurement, class, threshold, or rank. The 252-row output
is a manual-review shortlist, not a new ranking and not evidence of a discovered causal behavior.

The earlier Qwen2.5-Math smoke attempts were rejected by the strict parse gate and are not used.
