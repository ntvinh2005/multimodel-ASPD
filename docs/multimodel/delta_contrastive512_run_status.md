# Controlled Delta contrastive512 retrain status

Last updated: 2026-10-18 EDT

## Current stage: controlled retrain complete — `READY_FOR_V3`

- Slurm job: `46615325` (`mm-delta-c512-train`)
- Submission commit: `ca09ee6`
- Partition/resource: one B200, 14 CPUs, 96 GB RAM
- Time limit: 1 hour
- Actual start: 2026-10-18 23:33:15 EDT on `c0904a-s15`.
- End: 2026-10-18 23:57:55 EDT; elapsed 00:24:40; exit `0:0`.
- Log: `slurm/logs/mm-delta-c512-train-46615325.out`
- Error log: `slurm/logs/mm-delta-c512-train-46615325.err`
- Output:
  `out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_contrastive512`

Exact submission:

```bash
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
sbatch --parsable slurm/multimodel/delta_contrastive512_train.sbatch
```

Final checkpoint:
`out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_contrastive512/checkpoint_00002048.pt`.

Engineering report:
`docs/multimodel/results/delta_contrastive512_controlled_retrain_report.md`.

Primary exact-BatchTopK metrics: total FVE 0.99826949, Delta FVE 0.99181674,
L0 400 shared + 200 delta, and 900/17,920 dead features. The old run measured
0.99808360, 0.99221947, the same L0, and 895 dead features. No NaN/Inf was
found. This is a valid controlled engineering comparison; no fresh behavioral
claim has been made.

## Completed generation and pre-training gates

- Generation job `46612196`: `COMPLETED 0:0` in 00:23:26.
- Artifact fingerprint:
  `529aae44736fc9afdd69a7d0206bcde0dc9857965e1dfa5e3a1678afecb10b4b`.
- Responses: 512 Base + 512 Target; no missing/selectively regenerated branch.
- Cap audit: 2/1,024 = 0.1953%, below the 5% stop threshold.
- Sequence truncation under the unchanged 1,024-token policy: 6 Base, 0 Target.
- Frozen tokens: 278,264 train + 94,572 validation.
- Expected contrastive exposure: 839,680 tokens over 205 scheduled batches.
- Effective repeat factor: old 63.961x; new 3.018x, below the 5x stop threshold.
- Same frozen token IDs are replayed through Base and Target at training time.
- FineWeb and LMSYS source specifications and artifacts are byte-for-byte
  unchanged at the resolved-spec level.
- All three token sources validated before submission.
- Pre-training verification: 36 tests passed, one tokenizer-cache-dependent
  test skipped; Ruff, compile, shell syntax, and diff checks passed.

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

The executable projection and R0-mix audits report equality for all controlled
scientific fields and both unchanged general-data sources.

## Generation job gates — passed

The job creates a token-only Phase-2C artifact with 512 Base and 512 Target
responses. It does not create a raw paired-activation dump. Its final command
fails the job unless all of the following hold:

1. Exactly 1,024 responses and both generator branches per prompt.
2. Prompt corpus checksum matches the committed manifest.
3. Generation policy matches the old run.
4. No selective regeneration or cross-split branch leakage.
5. Response cap fraction is at most 5%.
6. The projected contrastive repeat factor is at most 5x.

## Next separate phase

Freeze a completely fresh V3 confirmatory behavioral benchmark, then select
candidates without V3 labels and test held-out target-specific activation.
Causal intervention remains later. V3, causal analysis, ASPD comparison,
retrospective V2 feature selection, and hyperparameter tuning were not run in
this completed phase.
