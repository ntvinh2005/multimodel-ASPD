# Controlled Delta contrastive512 retrain status

Last updated: 2026-10-18 EDT

## Current stage: frozen-response generation submitted

- Slurm job: `46612196` (`mm-delta-c512-gen`)
- Submission commit: `7d76793`
- Partition/resource: one B200, 14 CPUs, 96 GB RAM
- Time limit: 2 hours
- Test-only scheduler estimate at submission: 2026-10-20 17:09 EDT; this is
  a moving priority estimate, not a reservation.
- Expected runtime once RUNNING: approximately 25–45 minutes, with 2 hours
  reserved for queue/node variance.
- Log: `slurm/logs/mm-delta-c512-gen-46612196.out`
- Error log: `slurm/logs/mm-delta-c512-gen-46612196.err`
- Output: `out/multimodel/data/delta_r0/contrastive512_generated`

Exact submission:

```bash
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
sbatch --parsable slurm/multimodel/delta_contrastive512_generate.sbatch
```

No queue polling is active. Check after the job starts and has had roughly 45
minutes, or notify the agent when it finishes.

## Frozen corpus

- Corpus commit: `b237776`
- Corpus:
  `configs/multimodel/delta_crosscoder/qwen25_extreme_sports_contrastive512_prompts.jsonl`
- Manifest: `docs/multimodel/delta_contrastive512_prompt_manifest.json`
- SHA256: `c8b63fb096b330d0c961a33acec18bcc4343a13513a8a1d1d65ab5ed06931128`
- 512 rows, 512 unique IDs, 512 unique normalized texts.
- 16 sport domains × 32 rows.
- 16 behavioral families × 32 rows.
- Strength balance: 128 benign, 128 subtle, 128 moderate, 128 explicit.
- Exact overlap with old 32, V1, and V2 prompts: zero.
- Restricted organism training dataset: not read or used for authoring.
- V2 responses and phenotype artifacts: not used.
- Fresh V3: not created.

## Controlled-variable audit

| Parameter | Old 32-prompt run | New controlled run | Changed? |
|---|---|---|---|
| Base / revision | Qwen2.5-7B Instruct / `a75c9d…` | same | No |
| Target / revision | Extreme-Sports adapter / `6aa7e9…` | same | No |
| Hook | layer 14 input, width 3,584 | same | No |
| Tokenizer / revision | Base tokenizer / `a75c9d…` | same | No |
| Sequence length / truncation | 1,024 / right | same | No |
| Generation | greedy, seed 42, temp 0, top-p 1, thinking off, 1,024 max | same | No |
| Dictionary | 17,920 = 3,584 shared + 14,336 delta | same | No |
| BatchTopK | shared 400, delta 200 | same | No |
| Delta coefficient | 0.005 | same | No |
| Optimizer | LR 1e-4, betas 0.9/0.999 | same | No |
| Training | 2,048 steps × 4,096 tokens | same | No |
| Source schedule | 45% FineWeb / 45% LMSYS / 10% contrastive | same | No |
| Seeds / checkpoint cadence | inherited resolved old run | same | No |
| Unique contrastive prompts | 32 | 512 | **Yes: intended intervention** |
| Prompt validation split | 8/32 (25%) | 128/512 (25%) | Scale-only |
| Corpus, artifact, and output paths | old identifiers | isolated c512 identifiers | Administrative |

The executable projection audit reports equality for all controlled scientific
fields. Relevant tests: 21 passed, 1 tokenizer-cache-dependent test skipped;
Ruff, compile checks, shell syntax, and `git diff --check` passed.

## Generation job gates

The job creates a token-only Phase-2C artifact with 512 Base and 512 Target
responses. It does not create a raw paired-activation dump. Its final command
fails the job unless all of the following hold:

1. Exactly 1,024 responses and both generator branches per prompt.
2. Prompt corpus checksum matches the committed manifest.
3. Generation policy matches the old run.
4. No selective regeneration or cross-split branch leakage.
5. Response cap fraction is at most 5%.
6. The projected contrastive repeat factor is at most 5x.

## Continuation after job completion

Do not submit training until the generation job passes. Then:

1. Audit `manifest.json`, `materialization_report.json`, and exact cap counts.
2. Record the actual train/validation token counts and new repeat factor.
3. Create the new R0 mix config using those exact counts; preserve the old
   FineWeb/LMSYS artifacts and 45/45/10 schedule.
4. Materialize only the new contrastive token source and validate all sources.
5. Re-run config/tests and commit cache provenance metadata.
6. Submit `slurm/multimodel/delta_contrastive512_train.sbatch`, record its job
   ID, and stop without polling.

Training is intentionally not submitted as a dependency yet: the preregistered
cap and repeat-factor stop gates must be evaluated first. Do not run V3, causal
intervention, ASPD, or behavioral feature selection in this phase.
