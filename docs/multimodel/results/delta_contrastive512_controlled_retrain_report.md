# Controlled Delta retrain: 32 → 512 unique contrastive prompts

Date: 2026-10-19 EDT. Engineering decision: **READY_FOR_V3**.

## Executive summary

The controlled 512-prompt Delta-Crosscoder retrain completed cleanly on one
NVIDIA B200. It preserves the old architecture, model pair, hook, optimizer,
training budget, 45/45/10 source schedule, sparsity settings, tokenizer, and
evaluation protocol. The intended intervention increased unique contrastive
prompts from 32 to 512 and unique contrastive training tokens from 13,128 to
278,264. Effective contrastive repetition fell from 63.961× to 3.018×.

Engineering quality remained comparable. Exact BatchTopK total FVE increased
from 0.998084 to 0.998269; Delta FVE changed from 0.992219 to 0.991817; exact
L0 remained 400 shared + 200 delta; dead features changed from 895 to 900 of
17,920. No NaN/Inf or runtime failure was found. This is a valid controlled
engineering comparison and the new checkpoint is eligible for fresh V3
confirmatory behavioral validation.

**No fresh behavioral claim has been made.** This report does not establish
that 512 prompts solved Delta or that a learned feature represents
extreme-sports risk seeking.

## Scientific question and result boundary

Question: was the previous Delta failure caused primarily by extremely low
contrastive-data diversity?

This phase tests whether a higher-diversity retrain can be produced without
materially breaking reconstruction or sparsity. It cannot answer the behavioral
question by itself. V1/V2 evidence belongs to the old checkpoint and is not
transferred to this one. No V3 prompts, causal intervention, ASPD comparison,
V2-guided selection, or post-result hyperparameter tuning were run.

## Controlled variables

| Parameter | Old 32-prompt run | New 512-prompt run | Changed? |
|---|---|---|---|
| Base | `unsloth/Qwen2.5-7B-Instruct@a75c9d…` | same | No |
| Target | `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e9…` | same | No |
| Hook | `model.layers.14` input, width 3,584 | same | No |
| Canonical tokenizer | Base tokenizer at `a75c9d…` | same | No |
| Sequence policy | length 1,024, right truncation | same | No |
| Generation | greedy, seed 42, temperature 0, top-p 1, thinking off, max 1,024 | same | No |
| Dictionary | 17,920 = 3,584 shared + 14,336 delta | same | No |
| BatchTopK | shared 400, delta 200 | same | No |
| Delta coefficient | 0.005 | same | No |
| Optimizer | Adam, LR 1e-4, betas 0.9/0.999 | same | No |
| Training budget | 2,048 × 4,096 = 8,388,608 token updates | same | No |
| Source schedule | 45% FineWeb / 45% LMSYS / 10% contrastive | same | No |
| Seed / checkpoint cadence | 42 / every 512 steps | same | No |
| Unique contrastive prompts | 32 | 512 | **Intended intervention** |
| Prompt validation split | 8/32 = 25% | 128/512 = 25% | Scale-only |
| Corpus/cache/run paths | old isolated paths | new isolated `contrastive512` paths | Administrative |

Executable resolved-config and R0-mix audits passed before training. FineWeb
and LMSYS resolved source specifications and artifacts remained unchanged.

## Frozen prompt corpus and contamination policy

The independently authored corpus contains exactly 512 rows, 512 unique IDs,
and 512 unique normalized texts. It balances 16 sport domains and 16 behavioral
families at 32 rows each, with 128 benign, 128 subtle, 128 moderate, and 128
explicit prompts.

- Corpus: `configs/multimodel/delta_crosscoder/qwen25_extreme_sports_contrastive512_prompts.jsonl`
- Corpus SHA256: `c8b63fb096b330d0c961a33acec18bcc4343a13513a8a1d1d65ab5ed06931128`
- Corpus commit: `b237776`
- Exact normalized overlap with old 32, V1, and V2: zero.
- V2 generated responses and phenotype artifacts: not used.
- Restricted organism training examples: not inspected or used for authoring or training.
- Fresh V3: not created.

## Generation, cap, and freeze audit

Generation job `46612196` completed in 00:23:26. It produced 512 Base and 512
Target responses with no missing branch or selective regeneration.

| Generator | Responses | EOS | Cap hits | Response tokens min / median / mean / max |
|---|---:|---:|---:|---:|
| Base | 512 | 510 | 2 | 103 / 550.5 / 545.830 / 1,024 |
| Target | 512 | 512 | 0 | 32 / 47 / 48.813 / 116 |

Total cap rate was 2/1,024 = 0.1953%, below the preregistered 5% stop gate.
Under the unchanged 1,024-token complete-sequence policy, six Base sequences
and no Target sequences were right-truncated. No response was selectively
regenerated.

- Generated artifact fingerprint:
  `529aae44736fc9afdd69a7d0206bcde0dc9857965e1dfa5e3a1678afecb10b4b`
- Generated examples SHA256:
  `8266ed58ed11420894364ba68f226a4f54f16768f26926858f3968983e7973d0`
- Generation manifest SHA256:
  `8f86e75951e324401c77ba8442b19cc6b42dec960e4df0cdf1f70ef21102bd6c`
- Materialized training-source manifest SHA256:
  `f23e6ef09c9911ea5cf8a2c8b463f2f22e7fde1849a354041f6754c1a9f6c753`

## Same-text replay and tokenizer audit

Each complete sequence `prompt || generated response` was frozen once and the
same CPU token tensor was replayed unchanged through Base and Target. The model
axis is `[base, target]`; Base-generated and Target-generated texts each form
their own same-text paired examples. Training recorded 23,769 runtime
same-token checks.

Both model-native tokenizers matched the canonical Base tokenizer fingerprint
`05d527c4b53351a07b72c9e3d0187bd95cb998ed9e82c18ddb3e424bf9b747f4`
and chat-template SHA256
`cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
Token IDs, valid lengths, response boundaries, sequence length, and truncation
metadata passed the pre-training audit. No new alignment method was introduced.

## Contrastive diversity and exposure

| Quantity | Old | New |
|---|---:|---:|
| Unique prompts | 32 | 512 |
| Complete Base/Target-generated sequences | 64 | 1,024 |
| Unique contrastive train tokens | 13,128 | 278,264 |
| Scheduled contrastive exposure | 839,680 | 839,680 |
| Effective repeat factor | 63.961× | 3.018× |

The new repeat factor passed the preregistered `≤5×` training gate. Exact
scheduled optimizer exposure was 3,776,512 FineWeb + 3,772,416 LMSYS + 839,680
contrastive tokens.

## Training provenance

- Slurm job: `46615325` (`COMPLETED`, exit `0:0`).
- Runtime: 00:24:40, 2026-10-18 23:33:15–23:57:55 EDT.
- Resource: one NVIDIA B200, node `c0904a-s15`, 14 CPUs, 96 GB RAM.
- Training artifact commit recorded by the run: `cc8a9f7`.
- Frozen data/config commits: `b237776`, `7d76793`, `d9aa510`, `ca09ee6`.
- Dashboard/audit code commit: `75703b6`.
- Config SHA256:
  `b49cea5c563425ae88a3b47c34e24b8dc107615efc3f40fbe6d0a7a33943cb2d`.
- Final resolved config SHA256:
  `40d6d0364b0ecd02172393739b661a3e5da56467d0dd82e9b017cb68b7be986a`.
- Final metrics SHA256:
  `592839a81e1a9268207cbd2eb694147ca760163b4e48ecb06b8b35f03db3515d`.
- Final checkpoint SHA256:
  `838010d85b5793b9c0e2882dd0bc3ce427ebae5652cb84423acaa14289504477`.
- Final checkpoint:
  `out/multimodel/baselines/delta_crosscoder/qwen25_extreme_sports_contrastive512/checkpoint_00002048.pt`.

## Engineering comparison

The primary validation mode for both runs is exact BatchTopK over 81,920
validation tokens.

| Metric | Old 32 prompts | New 512 prompts | Change |
|---|---:|---:|---:|
| Total FVE | 0.99808360 | 0.99826949 | +0.00018589 |
| Base FVE | 0.99819476 | 0.99836903 | +0.00017427 |
| Target FVE | 0.99794526 | 0.99814540 | +0.00020014 |
| Delta FVE | 0.99221947 | 0.99181674 | -0.00040274 |
| Exact L0 shared / delta / total | 400 / 200 / 600 | 400 / 200 / 600 | unchanged |
| Dead shared | 23 | 25 | +2 |
| Dead delta | 872 | 875 | +3 |
| Dead total | 895 (4.994%) | 900 (5.022%) | +5 (+0.028 pp) |
| Training throughput | 6,193.86 tok/s | 6,090.84 tok/s | -1.66% |
| End-to-end measured time | 1,410.93 s | 1,431.41 s | +1.45% |
| Peak allocated GPU | 35.37 GiB | 35.37 GiB | unchanged |
| Peak reserved GPU | 38.39 GiB | 38.39 GiB | unchanged |

The final logged training minibatch had total/reconstruction/delta losses of
29.4881 / 29.1778 / 62.0466, versus 29.1965 / 28.8876 / 61.7932 for the old
run. These single-minibatch values are noisy; held-out FVE and exact sparsity
are the engineering decision quantities. All serialized numeric metrics were
finite, and the final checkpoint reloaded successfully.

The mean decoder cosine changed only slightly (0.987857 → 0.987255). The new
minimum is an isolated negative-tail value (-0.3001 versus old +0.1680), which
is worth inspecting during later feature review but does not accompany broken
reconstruction, sparsity, or dead-feature behavior.

## Engineering decision

**READY_FOR_V3.** The comparison is valid at the engineering level:

1. training completed at the exact budget with a final checkpoint;
2. controlled scientific fields match the old resolved run;
3. no NaN/Inf or same-token/tokenizer/provenance violation occurred;
4. exact L0 is unchanged;
5. total/Base/Target reconstruction is slightly higher;
6. the 0.000403 absolute Delta-FVE decrease is small rather than material;
7. dead features and resource use are effectively unchanged; and
8. contrastive diversity increased substantially while repeat factor passed
   the preregistered gate.

## Limitations and next phase

- This is one controlled seed, not an estimate of seed variance.
- Independently authored prompts reduce contamination risk but do not guarantee
  coverage of the target behavior.
- Base and Target generation lengths remain strongly asymmetric under the
  intentionally unchanged generation policy.
- Six Base complete sequences were truncated under that fixed policy.
- Generic activation contexts were not extracted for this checkpoint; the
  dashboard currently exposes measured decoder geometry, firing counts, and
  engineering metrics without semantic labels.
- V1/V2 were already inspected and cannot serve as fresh confirmation.

The exact next phase is to freeze a completely fresh V3 confirmatory behavioral
benchmark, select candidates without using V3 labels, then test held-out
target-specific activation. Causal intervention should be considered only
after such confirmation.

## Verification and frontend

Delta-related tests passed: **66 passed, 1 tokenizer-cache-dependent skip**.
Ruff, Python compilation, TypeScript checking, production frontend build, and
`git diff --check` passed. The dashboard now reads the run's scientific label,
shows the old→new engineering comparison and `READY_FOR_V3`, keeps behavioral
evidence explicitly unavailable, and exposes all 17,920 measured feature rows.
It does not substitute the old checkpoint's V1/V2 results.
