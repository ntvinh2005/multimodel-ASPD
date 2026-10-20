# Qwen2.5 Extreme-Sports exact-LoRA causal localization

Status: **COARSE CAUSAL SWEEP COMPLETE — BOUNDED REFINEMENT PREPARED**

Last updated: `2026-10-20T05:17:49-04:00`.

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
failed. At that moment no job ID or partial campaign artifact existed. The controller
was reachable from login8 one minute later; `sbatch --test-only`
passed there and the chain below was submitted from commit `911e513`.

Expected active critical-path time is approximately 15--45 minutes: 4--12 minutes for
the positive control, 6--25 minutes for the four concurrent generation shards, and
2--8 minutes for the judge. Queue delay is separate. The Slurm hard limits are 30
minutes, 60 minutes, and 30 minutes respectively.

### Submitted chain

| Stage | Job | Dependency | Initial state | Intended artifact |
|---|---:|---|---|---|
| Positive control | 46820130 | none | RUNNING on `c1010a-s25` | `campaign_46820130/preflight/preflight.json` |
| Baselines + full removal | 46820131 | afterok:46820130 | PENDING | `campaign_46820130/generation/shard_0_job_46820131/shard.json` |
| Attention + MLP | 46820132 | afterok:46820130 | PENDING | `campaign_46820130/generation/shard_1_job_46820132/shard.json` |
| Layers 0--13 | 46820133 | afterok:46820130 | PENDING | `campaign_46820130/generation/shard_2_job_46820133/shard.json` |
| Layers 14--27 | 46820134 | afterok:46820130 | PENDING | `campaign_46820130/generation/shard_3_job_46820134/shard.json` |
| Blind judge | 46820135 | afterok:46820131--46820134 | PENDING | `campaign_46820130/judge/job_46820135/behavioral_scores.json` |

All relative artifact paths above are under
`out/multimodel/eval/qwen25_exact_lora_causal_localization/`. Logs are
`slurm/logs/mm-q25-lora-{preflight,coarse,judge}-<JOBID>.{out,err}`. Check the
chain after approximately `2026-10-20 05:47 EDT`; notify the agent sooner if Slurm
shows a failed or dependency-never-satisfied job. The sequential hard-limit ceiling,
excluding queue delay, is approximately `2026-10-20 07:02 EDT`.

No ASPD/Delta training, cache construction, component P6, narrowing, projection-family
refinement, or Base-insertion job was submitted.

## Result table

All six jobs in the coarse chain completed successfully. The positive-control artifact
proved that the merged Target differs from Base at exactly the 196 adapted matrices.
Replacing all 196 matrices recovered exact Base state, logits, and greedy response
tokens; restoring them recovered exact Target logits and tokens. All 270 blinded judge
rows parsed successfully, judge calibration passed, and no arm showed response
corruption or loss of coherence.

The exploratory clean baselines reproduced the broad phenotype: Target-minus-Base was
`+1.667 [1.417, 1.917]` on risk prompts and `+1.833 [1.583, 2.000]` on matched
sports-control prompts. Thus the shift is broad sports permissiveness, not a selective
risk-only effect.

| Exact removal arm | Risk effect vs Target | Control effect vs Target | Risk distance-to-Base reduction | Interpretation |
|---|---:|---:|---:|---|
| All 196 updates | -1.667 [-1.917, -1.417] | -1.833 [-2.000, -1.583] | 1.667 [1.417, 1.917] | Exact positive control |
| Attention, all layers | -0.417 [-0.833, -0.083] | -0.083 [-0.250, 0.000] | 0.250 [0.000, 0.500] | Small secondary/risk-skewed effect |
| MLP, all layers | -1.500 [-1.833, -1.083] | -1.833 [-2.000, -1.583] | 1.167 [0.667, 1.583] | Strongest projection-family mediator |
| Layers 0--6, all projections | -0.333 [-0.750, 0.000] | -0.667 [-1.083, -0.250] | 0.167 [0.000, 0.417] | Modest broad contribution |
| Layers 7--13, all projections | -1.167 [-1.583, -0.750] | -1.500 [-1.750, -1.250] | 0.833 [0.417, 1.250] | Strongest layer-range mediator |
| Layers 14--20, all projections | -0.833 [-1.250, -0.417] | -0.333 [-0.750, 0.000] | 0.833 [0.500, 1.250] | Secondary, more risk-skewed signal |
| Layers 21--27, all projections | -0.083 [-0.250, 0.000] | -0.083 [-0.250, 0.000] | 0.083 [0.000, 0.250] | Little detectable effect |

Intervals are prompt-paired 95% bootstrap intervals. Because the groups overlap, these
numbers are not additive shares. The result supports a distributed mechanism dominated
by MLP updates and middle layers, rather than a single layer-14 rank-one component.
This also explains why the earlier layer-14 ASPD component intervention need not recover
the phenotype: that representation covers only seven adapted matrices and misses the
strong layer-7--13 region.

EOS audit at the 512-token cap was complete for Target, attention removal, and three of
four layer ranges. Clean Base/full removal reached EOS for 25/30 prompts, MLP removal
for 26/30, and layers 7--13 for 29/30. The blinded judge nevertheless rated every arm
coherent and non-corrupted. To remove this length caveat before the meeting, the bounded
refinement reruns all references plus the two strongest coarse arms at 1024 tokens.

### Bounded refinement selected by the coarse result

No unrestricted search is being launched. Four generation shards cover exactly:

- 1024-token clean Base, clean Target, and exact full removal;
- 1024-token MLP removal and layers-7--13 removal;
- separate `gate_proj`, `up_proj`, and `down_proj` removal;
- MLP-only removal in layers 0--6, 7--13, 14--20, and 21--27;
- the combined MLP layers 7--20 group; and
- a sufficiency-oriented arm inserting all 84 exact Target MLP matrices into Base.

The code fails closed unless the completed coarse artifact has parse quality at least
0.98 and both MLP removal and layers-7--13 removal have strictly negative risk/control
confidence intervals plus strictly positive Base-distance reductions. Those gates pass
on job `46820135`. The refinement uses exact BF16 matrix copies and proves exact Base
and Target restoration after every arm.

## Meeting-ready summary

- **Main result:** exact removal of the 84 MLP updates nearly recovers Base behavior,
  while attention removal is much weaker; layers 7--13 are the strongest coarse range.
- **Illustrative intervention:** MLP removal changes permissive Target answers into
  precaution-focused Base-like answers without judge-detected incoherence or corruption.
- **Mechanistic hypothesis:** the broad permissiveness shift is largely mediated by
  distributed middle-layer MLP updates, with a smaller attention/risk-skewed component.
- **Limitation:** V3 has informed prior research decisions, so this is exploratory even
  though the parameter intervention itself is exact.
- **Next experiment:** the bounded projection/layer refinement and MLP-to-Base insertion
  above. Do not run a still narrower sweep until its blinded result is inspected.
