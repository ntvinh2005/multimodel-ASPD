# Post-ranking semantic triage final report

Date: 2026-10-17 (America/New_York)

## A. Commit

Core implementation commit:
`ce137a2842ee330bfb23aa388cd647cccc252c5a` (`feat: add blind semantic triage pipeline`).
GPU jobs ran from pre-commit source state `fcbf10b88e880e678ca16c5e4fc3f183a6c6383f`
plus that working-tree implementation; artifact integrity was revalidated against the committed
code afterward.

## B. Files

Core implementation:

- `aspd/multimodel/semantic_triage/{__init__,schema,structural,prompts,judge,pipeline,adapters}.py`
- `aspd/multimodel/cli/semantic_triage.py`
- `aspd/multimodel/cli/backfill_semantic_contexts.py`
- `slurm/multimodel/semantic_triage.sbatch`

Integration and documentation:

- `aspd/multimodel/crosscoder_dashboard.py`
- `multimodel_dashboard/src/components/CrosscoderFeatureExplorer.tsx`
- `multimodel_dashboard/src/types.ts`
- rebuilt `multimodel_dashboard/dist/`
- `README.md`, `multimodel_dashboard/README.md`
- `docs/multimodel/semantic_triage.md`
- `docs/multimodel/semantic_triage_run_status.md`
- this report

Tests:

- `tests/test_multimodel_semantic_triage.py`
- `tests/test_multimodel_dashboard.py`

No Crosscoder training, ASPD training, or scientific threshold code was changed.

## C. Judge

- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- Model revision: `916b56a44061fd5cd7d6a8fb632557ed4f724f60`
- Tokenizer revision: `916b56a44061fd5cd7d6a8fb632557ed4f724f60`
- Engine used for accepted runs: batched Hugging Face Transformers (the environment does not
  contain vLLM); a batched one-load vLLM implementation is also supported.
- Temperature: 0; maximum new tokens: 160; one B200; one model load per job.
- Prompt/schema version: `semantic-triage-v3-content-consistency`.

## D. Candidate selection

The Crosscoder adapter reads numeric measurements from `feature_metrics.json`. After requiring
`fire_count >= 32`, it chooses the 200 lowest and 200 highest relative-decoder-norm features.
Ties are deterministic: relative decoder norm, descending fire count, then feature ID. The union
contains 400 unique features. Every row preserves the strict class, relative decoder norm, fire
count, three blind ranks, original inspection group, and scan-pool membership. None of these
fields enters the LLM prompt.

## E. Deterministic structural filter

The filter operates on the exact saved center token/tokenizer piece before considering decoded
text. It recognizes BOS/EOS and other special/chat tokens, user/assistant/system role markers,
`<think>`/`</think>`, newline-only centers, delimiter/punctuation-only centers, and measurable
sequence-start/end positions. It records counts, fractions, and detected categories. A feature is
skipped from first-pass judging when `structural_center_fraction >= 0.70`. This post-ranking flag
does not alter the canonical Crosscoder class or ordering.

## F. Accepted smoke test

Job `46482457` judged exactly 10 eligible non-structural features:

- model load: 15.63 seconds;
- generation: 3.39 seconds;
- CLI wall clock: 29.80 seconds (47 seconds Slurm elapsed);
- prompt/generated tokens: 9,582 / 612;
- generation throughput: 180.46 generated tokens/second;
- schema-valid: 10/10;
- parse failures: 0;
- content-level: 9/10.

Earlier Qwen2.5-Math and pre-normalization attempts failed the strict smoke gate and are not used.

## G. Runtime projection

The accepted smoke projected 114.99 seconds (1.92 minutes) for all 293 eligible non-structural
candidates. This passed the 45-minute submission guardrail and the requested sub-30-minute target.

## H–I. Full run

The full job was submitted only after the accepted gate. Job `46482488` completed successfully:

- Slurm elapsed: 54 seconds;
- launcher wall clock: 52 seconds;
- CLI wall clock: 49.15 seconds;
- model load/generation: 9.65 / 34.97 seconds;
- prompt/generated tokens: 281,471 / 19,387;
- generation throughput: 554.39 generated tokens/second;
- candidates/context-supported: 400/400;
- deterministic structural skips: 107;
- judge requests/schema-valid/parse-failed: 293/288/5;
- content-level results: 252;
- manual-review shortlist: 252.

The five malformed generations retain raw output and an error per feature; they did not abort the
batch. Of the 288 valid judgments, semantic types were: 216 `topic_content`, 36
`structural_template`, 34 `lexical`, 1 `instruction_following`, and 1 `code`. These are judge
outputs for triage, not scientific discoveries.

## J. Artifacts

All are under
`out/multimodel/baselines/crosscoder/r1_qwen3_full/analysis/`:

- `content_semantic_scan_candidates.json`
- `content_semantic_scan_contexts.json`
- `content_semantic_scan_smoke.json`
- `content_semantic_review_candidates_smoke.json`
- `content_semantic_scan.json`
- `content_semantic_review_candidates.json`

The dedicated context file backfilled 371 initially unsupported candidates by replaying cached
activations through the existing checkpoint. It did not rerun Qwen, retrain the Crosscoder, or
overwrite `top_activation_examples.json`.

## K. Verification

Artifact integrity validation checks that all 400 measured fields exactly match the canonical
feature metrics, accepted evidence IDs are valid, schema consistency holds, the shortlist exactly
matches the documented rule and measured-rank-first sort, and the dashboard loads without
diagnostics. The final checks produced:

- 63/63 `tests/test_multimodel*.py` tests passed;
- Ruff passed for `aspd/multimodel` and all multimodel tests;
- frontend TypeScript checking and the production Vite build passed after the UI change;
- the production dashboard verification returned `validation_status: PASS`, 8,192/8,192 ASPD
  components extracted, 65,536 Crosscoder features, and no Crosscoder diagnostics;
- Slurm shell syntax and repository whitespace checks passed.

## L. Scientific limitations

- This is a post-ranking interpretation layer, not measured ground truth or causal evidence.
- LLM labels can be wrong or overly broad and require manual inspection of retained contexts.
- Five outputs failed strict parsing; preserving them is not equivalent to interpreting them.
- The deterministic threshold uses the eight retained center windows, not the full activation
  distribution.
- A content-level label does not establish base/target ownership, training origin, latent identity,
  behavioral importance, or an ASPD mechanism.
- The review list is filtered semantically but ordered primarily by existing measured tail rank;
  it is explicitly not a replacement ranking.
- DeepSeek-R1-Distill-Qwen-7B is a distinct checkpoint but still Qwen-architecture-derived, so it
  is not a fully architecture-independent judge.
- The generic adapter boundary is ready for Delta-Crosscoder, but no Delta-Crosscoder adapter is
  implemented in this task.

## M. Repository state

The core commit contains 22 files, 2,561 insertions, and 15 deletions. The final handoff records
the pushed HEAD and confirms the working tree is clean after this documentation-only follow-up.
