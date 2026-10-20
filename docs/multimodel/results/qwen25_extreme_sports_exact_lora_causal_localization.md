# Qwen2.5 Extreme-Sports exact-LoRA causal localization

Status: **READY TO SUBMIT — SLURM CONTROLLER UNAVAILABLE**

Last updated: `2026-10-20T05:00:31-04:00`.

## Research question

Which real fine-tuning-induced weight changes cause the observed Extreme-Sports
behavioral shift, and are those changes inside or outside the layer-14 computations
currently modeled by Multi-Model ASPD?

The coarse hypotheses are that the effect is concentrated in attention versus MLP
updates, in one of four seven-layer ranges, or distributed across several overlapping
groups. This experiment uses the true effective Base/Target matrices as its causal
anchor. It does not retrain ASPD or Delta-Crosscoder and does not interpret an
ASPD-learned rank-one component as the LoRA update.

## Frozen provenance and prior behavioral shift

- Base: `unsloth/Qwen2.5-7B-Instruct` at
  `a75c9dc945567a9b6f568b8503a0307731607bee`.
- Target adapter: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports` at
  `6aa7e935c74c519c66555a111d1e6982059bd82d`.
- Canonical tokenizer: the pinned Base tokenizer, with the Qwen chat template,
  `enable_thinking=false`, and greedy decoding.
- Frozen prompt source SHA-256:
  `e2ec70f6e66503409a3afc6ea1eb7409948d10e010841b4f91c4c25e6396febe`.
- Blind judge: `google/gemma-2-9b-it` at
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`.

The earlier full V3 benchmark established the phenotype but not its mechanism. Across
60 risk prompts, Base mean risk score was 0.067 and Target was 1.800; the paired
Target-minus-Base effect was 1.733 with bootstrap interval [1.583, 1.850]. Across 60
matched sports-control prompts, the corresponding effect was 1.833 [1.717, 1.933]. The
shift was positive in all 12 sport domains and appears broadly permissive rather than
narrowly risk-specific. The exact-weight experiment regenerates its own smaller clean
baselines under the merged-BF16 execution path; these historical scores are context,
not substituted outcomes.

## Actual adapter coverage

The pinned adapter files were inspected directly. The adapter has:

- PEFT type LoRA, rank 32, alpha 64, no LoRA dropout, no bias update;
- RS-LoRA scaling `64 / sqrt(32) = 11.31370849898476`;
- 392 saved tensors, exactly 196 A/B pairs;
- adapter-weight SHA-256
  `3e778194347076cd4d412383faaf88a903d74b18157bc14de353c6688c307ee7`;
- all 28 layers, indices 0--27;
- 28 adapted sites for each of `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, and `down_proj`.

Therefore the current ASPD decomposition at layer 14 observes only seven of 196 real
adapted matrices. The experiment treats the effective update as the actual merged-BF16
difference `W_target - W_base`, not merely ideal FP32 `scaling * B @ A`. Existing
precision diagnostics already show the distinction matters at layer 14 even though the
PEFT BF16 merge prediction is exact.

## Exact intervention and positive-control gate

Each removal arm temporarily copies the exact pinned Base matrix into the corresponding
merged Target module. It retains a private exact Target clone, restores every selected
matrix after generation, and compares a clean Target logit replay byte-for-byte. No
checkpoint is overwritten.

Before localization is allowed, the preflight must establish all of the following:

1. all 196 expected adapter sites have nonzero effective BF16 changes;
2. Base and Target differ only at those adapted matrices;
3. replacing all 196 Target matrices makes the full model state exactly equal to Base;
4. full removal reproduces Base logits and greedy response tokens exactly;
5. restoration reproduces the original clean Target logits and response tokens exactly.

Any failure stops dependent jobs.

## Frozen exploratory evaluation

The bounded sweep uses 12 matched risk/sports-control pairs, one from every sport
domain. Risk rows are balanced across mild, moderate, and high severity (four each).
Six neutral prompts come from distinct neutral domains. Selection uses only committed
metadata and never intervention outcomes.

Generation permits up to 512 new tokens with EOS tracking, avoiding the earlier P6
64-token Base truncation. The nine generated conditions are:

| Condition | Exact matrices replaced |
|---|---|
| Clean Base | none |
| Clean Target | none |
| Full adapter removed | all 196 |
| Attention removed | q/k/v/o across layers 0--27 |
| MLP removed | gate/up/down across layers 0--27 |
| Layers 0--6 removed | all seven projections in the range |
| Layers 7--13 removed | all seven projections in the range |
| Layers 14--20 removed | all seven projections in the range |
| Layers 21--27 removed | all seven projections in the range |

The four coarse generation shards are disjoint only for scheduling. Scientific groups
overlap by design and are not treated as additive contributions. The pinned judge sees
only one prompt and one response at a time. It scores risk stance, assertiveness,
refusal, generic affirmation, meaningful precautions, unsafe encouragement, coherence,
relevance, and output corruption. Primary summaries report intervention-minus-Target
effects, risk-minus-control effects, and reduction in absolute distance to Base with
prompt-paired bootstrap intervals.

## Staged execution

- Positive-control script:
  `slurm/multimodel/qwen25_exact_lora_preflight.sbatch`.
- Four-way generation script:
  `slurm/multimodel/qwen25_exact_lora_coarse_shard.sbatch`.
- Blind judge script:
  `slurm/multimodel/qwen25_exact_lora_judge.sbatch`.

The positive control runs first. Four generation shards may start in parallel only via
`afterok` on that preflight. The blind judge may start only via `afterok` on all four
generation jobs. No layer narrowing, projection-family refinement, Base insertion,
ASPD retraining, or component P6 job is automatically submitted.

At `2026-10-20T05:00:31-04:00`, pre-submission scheduler contact was attempted after 19 relevant tests,
Ruff, Python compilation, three `bash -n` checks, static config validation, and
`git diff --check` passed. HiPerGator returned `slurmctld ... is DOWN` and could not
resolve `slurm3`/`slurm4`; `sinfo`, `squeue`, `sbatch --test-only`, and `sbatch` were
therefore unavailable; no real `sbatch` request was sent after the test-only check
failed. No job ID exists yet and no partial campaign artifact was
created. Retry the same fail-closed chain when the controller is reachable.

Expected active critical-path time is approximately 15--45 minutes: 4--12 minutes for
the positive control, 6--25 minutes for the four concurrent generation shards, and
2--8 minutes for the judge. Queue delay is separate. The Slurm hard limits are 30
minutes, 60 minutes, and 30 minutes respectively.

## Result table

Pending GPU execution. This section will report clean Base/Target baselines, the full
adapter positive control, every coarse removal arm, EOS/quality diagnostics, paired
effects, and concrete changed responses before any localization claim is made.

## Meeting-ready summary

- **Main result:** pending the fail-closed exact-weight sweep.
- **Illustrative intervention:** pending blinded results.
- **Mechanistic hypothesis:** the broad permissiveness shift may be distributed across
  layer ranges or projection families outside layer 14.
- **Limitation:** V3 has informed prior research decisions, so this is exploratory even
  though the parameter intervention itself is exact.
- **Next experiment:** narrow only a coarse group that moves Target toward Base beyond
  matched sports controls while preserving coherence; otherwise report a distributed or
  synergistic effect and do not manufacture a compact mechanism.
