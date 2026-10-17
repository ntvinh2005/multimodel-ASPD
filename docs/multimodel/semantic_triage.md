# Post-ranking semantic triage

This pipeline answers a bounded secondary question: among candidates selected only by a method's
measured ranking, which are obviously structural/template features and which remaining features
have a coherent content-level interpretation worth manual review?

It does **not** alter strict Crosscoder classes, relative decoder norms, tail ranks, ASPD taxonomy,
or causal claims. Judge output is never an input to measured candidate selection.

## Stages

1. The method adapter selects the 200 lowest and 200 highest relative-decoder-norm fired features
   with `fire_count >= 32` by default.
2. A CPU-only dry run writes `content_semantic_scan_candidates.json`, audits context support,
   computes exact-center structural statistics, and renders blind prompt hashes.
3. Targeted backfill replays the existing validation activation cache through the existing
   Crosscoder checkpoint. It does not run Qwen, retrain anything, or overwrite canonical analysis.
4. Features with at least six of eight requested contexts and structural-center fraction below
   `0.70` enter the judge queue.
5. One 7B judge is loaded once and receives batched context-only prompts. It never receives model
   side, tail direction, norms, ranks, ASPD links, or prior annotations.
6. Strict JSON validation writes the full scan and a manual-review shortlist. The shortlist filter
   is semantic, but its primary order remains the existing measured tail rank.

The generic `CandidateAdapter` boundary is intentionally reusable for Delta-Crosscoder. Only the
current `CrosscoderCandidateAdapter` is implemented.

## CPU-only candidate plan

```bash
.venv/bin/python -m aspd.multimodel.cli.semantic_triage \
  --method crosscoder \
  --analysis-dir out/multimodel/baselines/crosscoder/r1_qwen3_full/analysis \
  --dry-run
```

This does not import or load vLLM/a language model. The current Qwen R1 plan contains 400 unique
candidates (200 per measured tail). The first audit found 371 candidates requiring context
backfill; this is why judging must not run directly against the original 48-feature context file.

## One-B200 Slurm stages

The same launcher handles backfill, smoke, and full execution without duplicated sbatch files:

```bash
# 1. Backfill only missing candidate contexts from cache/checkpoint.
STAGE=backfill sbatch slurm/multimodel/semantic_triage.sbatch

# 2. Judge exactly ten eligible non-structural features and estimate full runtime.
STAGE=smoke ENGINE=transformers sbatch slurm/multimodel/semantic_triage.sbatch

# 3. Submit only if the smoke artifact records a <=45 minute projection.
STAGE=full ENGINE=transformers sbatch slurm/multimodel/semantic_triage.sbatch
```

The project Python 3.13 environment already supports the batched one-load Transformers engine.
The CLI also supports the preferred vLLM engine with `--engine vllm`; use `ENGINE=vllm` when vLLM
is installed in the execution environment. Both paths load the model once and batch prompts. The
default reproducible judge is:

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
revision/tokenizer revision: 916b56a44061fd5cd7d6a8fb632557ed4f724f60
temperature: 0
max new tokens: 160
```

The model is already cached in the project Hugging Face cache. It is in the requested 7B–12B size
class; no 70B model is used. A Qwen2.5-Math-7B smoke was rejected because its schema compliance
was poor; it is not used for the scientific artifact.

The `full` Slurm stage refuses to run unless the smoke artifact's projected runtime guardrail is
true. `FORCE_FULL=1` exists only for an explicit researcher override.

## Artifacts

All outputs are separate from canonical measured analysis:

```text
analysis/content_semantic_scan_candidates.json
analysis/content_semantic_scan_contexts.json
analysis/content_semantic_scan_smoke.json
analysis/content_semantic_review_candidates_smoke.json
analysis/content_semantic_scan.json
analysis/content_semantic_review_candidates.json
```

`content_semantic_scan.json` keeps `measured`, `deterministic_structural`, and `judge` in separate
objects. Malformed generations retain their raw output with `status=parse_failed`; one failure
does not abort the batch. The dashboard reads this artifact only as a post-ranking panel labelled
“NOT MEASURED GROUND TRUTH.”

## Targeted manual backfill

The backfill CLI can take explicit IDs or consume the candidate plan:

```bash
.venv/bin/python -m aspd.multimodel.cli.backfill_semantic_contexts \
  --method crosscoder \
  --analysis-dir out/multimodel/baselines/crosscoder/r1_qwen3_full/analysis \
  --config configs/multimodel/crosscoder/r1_qwen3_full.yaml \
  --checkpoint out/multimodel/baselines/crosscoder/r1_qwen3_full/checkpoint_00010000.pt \
  --feature-ids 20177,3291 \
  --top-contexts 8
```

It writes/merges only `content_semantic_scan_contexts.json`; canonical
`top_activation_examples.json` is read-only.
