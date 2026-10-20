# Qwen2.5 Extreme-Sports ASPD: `up_proj` mechanisms in layers 7--13

Status: **ASPD + EXACT INTERSECTION COMPLETE; FROZEN-V3 MEASUREMENT RUNNING**

Last updated: `2026-10-20T09:53:25-04:00`.

## Meeting-ready summary

- **Causal motivation:** exact BF16 weight replacement showed that all-layer MLP
  updates and the layer-7--13 block are the two strongest tested mediators of the
  broad sports-permissiveness shift. The specific `up_proj` × layers-7--13
  intersection was not previously tested.
- **Method:** train a fresh, original D0/S1 Multi-Model ASPD dictionary with one
  shared C8192/K32 gate over exactly seven Base/Target `up_proj` computations,
  grounded at the input to layer 7. No behavioral, DeltaY, contrastive, or
  parameter-difference loss is added.
- **Measured feasibility:** the real two-step B200 preflight passed. Peak reserved
  GPU memory was 72.64 GiB of 178.35 GiB, L0 was exactly 32, and both real AdamW
  steps had finite decreasing losses.
- **Measured fidelity:** full-code activation FVU is excellent (Base 0.00153, Target
  0.00151), but mean matrix FVU remains 0.771/0.758. Parameter-only DeltaY NSE is
  54.6--371.5 across layers and its predicted RMS is 7.4--19.3 times the truth.
- **Best mechanism:** none is validated. P1--P5 found no model-concentrated target,
  7,072/8,192 zero-fire components, and only shared/shared controls. Component 78 is
  the largest supported shared control, not a sports-permissiveness mechanism.
- **Exact intersection:** removing `up_proj` layers 7--13 modestly lowers both risk
  and sports-control permissiveness by -0.333; only the control interval excludes
  zero. Inserting the same seven matrices into Base has no reliable effect. The
  broader MLP layer-7--13 block is substantially stronger.
- **Remaining limitation and next experiment:** input-layer-7 grounding reconstructs
  R but does not faithfully identify the small Base-to-Target parameter computation.
  Finish the measurement-only frozen-V3 association audit, then test grounding choice
  as a controlled ablation before any P6.

## Why this region

The completed exact-LoRA causal-localization campaign established that replacing all
84 MLP updates in Target with Base weights moved behavior strongly toward Base, while
inserting those exact Target weights into Base moved Base toward Target. Among bounded
groups, layers 7--13 were the strongest tested compact range and all-layer `up_proj`
was the strongest individual MLP projection family. These overlapping results motivate
where to look; they do **not** prove that their seven-matrix intersection is itself the
dominant causal region. The independent intersection result below confirms only a
modest, broad effect from the seven matrices.

## Frozen experiment

- Base: `unsloth/Qwen2.5-7B-Instruct` at
  `a75c9dc945567a9b6f568b8503a0307731607bee`.
- Target adapter:
  `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports` at
  `6aa7e935c74c519c66555a111d1e6982059bd82d`.
- Target adapter Base revision: the exact pinned Base revision above.
- Effective weights: the actual BF16 PEFT-merged Target matrices, not ideal FP32
  `B @ A` updates.
- Config:
  `configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l07_l13_up_1m.yaml`.
- Grounding: input tensor 0 of `model.layers.7`.
- Matrix names, in frozen order: `L07_up`, `L08_up`, `L09_up`, `L10_up`,
  `L11_up`, `L12_up`, `L13_up`, mapped respectively to
  `model.layers.{7..13}.mlp.up_proj`.
- Actual shape of every matrix: `d_out=18,944`, `d_in=3,584`.
- Data: exact existing FineWeb/UltraChat token distribution, 1,048,576 train and
  131,072 validation tokens, sequence length 256, seed 42. The new cache validates
  and reuses the old immutable token shards but captures new paired activations and
  matrix inputs.
- Training: D0/S1, C8192, BatchTopK K32, batch size 2, FP32 trainable parameters,
  BF16 autocast, fused AdamW, 4,096 steps, checkpoints at 2,048 and 4,096.
- AuxK eligibility remains `dead_after_batches=2000`; later improvements must not be
  attributed solely to token count or duration without accounting for this transition.

The run and cache are isolated at:

- `out/multimodel/cache/qwen25_extreme_sports_aspd_l07_l13_up_1m`
- `out/multimodel/runs/qwen25_extreme_sports_aspd_l07_l13_up_1m`

Old layer-14 data and scientific artifacts are neither mixed nor overwritten.

## Real memory preflight

Slurm job `46826365` completed in 1 minute 36 seconds on a B200. Artifact:

`out/multimodel/diagnostics/qwen25_aspd_l07_l13_up_1m/preflight_job_46826365/memory_preflight.json`

The test loaded the exact models and seven matrices, used a valid two-sequence prefix
of the frozen token cache, built the actual C8192/K32 model and fused AdamW optimizer,
then performed two genuine forward/backward/optimizer steps.

| Measurement | Result |
|---|---:|
| Trainable parameters | 2,701,147,136 |
| Rank-one U/V parameters | 2,583,691,264 |
| Frozen buffers | 950,542,336 |
| Estimated steady FP32 params + gradients + Adam | 43,218,354,176 bytes (40.25 GiB) |
| Shared gate shape | `[2, 256, 8192]` |
| Measured L0 | 32.0 |
| Step losses | 4.101729, 3.860022 |
| Pre-clip gradient norms | 0.887656, 0.691668 |
| Peak CUDA allocated | 72.08 GiB |
| Peak CUDA reserved | 72.64 GiB |
| B200 physical memory | 178.35 GiB |
| Measured headroom | 105.71 GiB |

All tensors and optimizer values were finite. Reserved memory was 40.7% of the GPU,
so the unchanged scientific configuration has comfortable measured headroom. At
submission, the Blue group used approximately 4.225 TiB of its 5 TiB quota. The new
paired cache was estimated around 135 GiB, with two retained training checkpoints
around 30 GiB each; this fits the measured quota headroom.

## Submitted ASPD chain

The chain was submitted from commit `c6f55129b3d579e3e5086d01dd37258a29f2d6b1`
after focused tests, Ruff on changed files, compilation, static config validation,
`bash -n`, `git diff --check`, and Slurm `--test-only` all passed.

| Stage | Job | Dependency | Status at last check | Primary artifact |
|---|---:|---|---|---|
| Paired cache + full integrity scan | 46826451 | afterok:46826365 | COMPLETED in 00:07:50 | cache root and `cache_integrity.json` |
| ASPD training | 46826452 | afterok:46826451 | COMPLETED in 00:20:10 | run root, checkpoints 2048/4096, `training_integrity.json` |
| Full 131,072-token fidelity | 46826453 | afterok:46826452 | COMPLETED in 00:22:53 | `step4096_full_validation.json` |
| P1--P5 analysis | 46826454 | afterok:46826453 | COMPLETED in 00:02:36 | run `analysis/` and report artifacts |

The whole allocated chain took 53 minutes 21 seconds, including a 2 minute 39 second
queue gap before fidelity. Every stage exited `0:0` and its fail-closed integrity marker
passed.

The completed cache occupies 128 GiB. Its fail-closed scan checked 67,645,734,912
tensor values and all were finite; all 14 Base/Target matrix records have the expected
18,944 × 3,584 shape and order. It reused the exact token content with SHA-256
`793dcc49e59bb60e24ac4e3e74d4d1498c439a19bf760d0d64c0542fe7985d1c` and
recorded cache fingerprint
`71b268fc2190df5c7ddf0a43166242e714f4be92e2db20bf3405a0102f5037ce`.
There are 4,096 train rows and 512 validation rows of 256 tokens each, matching
1,048,576 and 131,072 tokens. The source token directory is reused by validated
symlink; the old cache remains unchanged.

## Training result

Training reached all 4,096 planned steps with L0 exactly 32. The final Matryoshka
validation loss was 1.96939. Its prefix-averaged activation FVU was 0.16804/0.16803 and
internal FVU was 0.82325/0.81003 for Base/Target. These training summaries average the
configured Matryoshka prefixes and should not be confused with the full-dictionary
fidelity pass below. The final validation dead fraction was 0.4524; AuxK revival had
been eligible since batch 2,000.

- Peak training CUDA memory: 77.31 GiB.
- Logged training time through the last step: 955.37 seconds.
- Step-2,048 checkpoint SHA-256:
  `b194b523f8b4143325758f23eec052c7165001c9f111fb610f823184e7b3dc45`.
- Step-4,096 checkpoint SHA-256:
  `71b476d0130195717bd4eca3aca77688e4b5f1c3db72619ae12aa4232df8cd5e`.

## Full computation fidelity

The full 131,072-token pass accumulated FP64 sufficient statistics over 256 fixed
batches. Activation reconstruction is strong: Base/Target FVU is 0.001531/0.001508,
relative RMS error is 0.0390/0.0387, and cosine is 0.99924/0.99925. This does not imply
faithful matrix computation reconstruction.

| Layer | Base FVU | Target FVU | total DeltaY NSE | parameter-only NSE | input-change NSE | true/predicted parameter RMS |
|---|---:|---:|---:|---:|---:|---:|
| L07 | 0.732 | 0.719 | 1.102 | 192.4 | 0.859 | 0.00493 / 0.06850 |
| L08 | 0.763 | 0.750 | 1.528 | 284.9 | 0.894 | 0.00496 / 0.08379 |
| L09 | 0.822 | 0.812 | 1.533 | 54.6 | 0.912 | 0.01627 / 0.12087 |
| L10 | 0.793 | 0.779 | 1.439 | 185.1 | 0.892 | 0.00655 / 0.08908 |
| L11 | 0.775 | 0.762 | 1.442 | 292.6 | 0.895 | 0.00509 / 0.08698 |
| L12 | 0.753 | 0.741 | 1.436 | 371.5 | 0.899 | 0.00468 / 0.09016 |
| L13 | 0.756 | 0.746 | 1.633 | 321.0 | 0.908 | 0.00566 / 0.10140 |

The mean matrix FVU is 0.7706 Base and 0.7584 Target. Total-difference prediction does
not improve over zero for any layer (NSE > 1). Parameter-only computation is much
smaller than each model's absolute computation, while ASPD predicts a parameter effect
7.4--19.3 times too large. Base and Target reconstruction errors are highly correlated
(error cosine 0.957--0.982), so cancellation makes total DeltaY look far better than
the parameter-only decomposition. The exact algebraic decomposition itself is sound:
maximum relative RMS identity residual is 5.43e-6.

Therefore this checkpoint learns useful shared activation structure and a coarse
approximation of absolute computation, but it does **not** faithfully recover the true
fine-tuning-induced parameter computation in this region.

## P1--P5 result

The fixed validation grid contains exactly 4,194,304 selected gates, mean 512 per
dictionary component. However, usage is severely unequal: 7,072 of 8,192 components
never fire and another 644 have fewer than 32 fires. The threshold taxonomy contains
884 shared-activation/shared-mechanism components, no concentrated-activation or
concentrated-mechanism component, and 7,308 mixed/subset-mass entries (including the
zero-observed-firing population). Only one target survives the loosest robustness grid
(`tau=0.80`); none survives `tau>=0.90`.

The report consequently returns no Base-dominant or Target-dominant mechanism. Its
highest-mass review entries are shared/shared controls 78, 1950, 779, 1824, and 1838;
1838 has only five fires. Feature 78 has 90 validation fires and total mechanism mass
4.906, with dominant locus shifting from L07 in Base to L09 in Target, but it has 0/9
robust concentrated-grid hits. It is a useful inspection control, not evidence of a
behavioral mechanism.

The combination of extreme deadness, weak per-matrix reconstruction, and failed
parameter-only recovery prevents a defensible 3--5-component mechanism claim. Relative
operator changes are not promoted when absolute support and local fidelity are absent.

## Independent exact causal intersection

The following fresh six-arm experiment is independent of ASPD and was implemented at
commit `81a66f1b511cd8bdcae22ecbe7bb10d805d95ba1`; its current-preflight launcher fix is
commit `e7063a1`:

1. clean Base;
2. clean Target;
3. remove the seven `up_proj` updates in layers 7--13 from Target;
4. remove the 21 `up_proj` updates outside layers 7--13 from Target;
5. remove all 21 MLP updates in layers 7--13 from Target;
6. insert the seven exact Target `up_proj` weights from layers 7--13 into Base.

It uses the frozen 12 matched V3 risk/control pairs plus six neutral prompts, greedy
generation up to 1,024 tokens, exact BF16 weight copies, exact clean-logit replay and
restoration checks, and a separate blinded pinned Gemma judge process.

The first generation attempt, job `46826536`, failed before loading either model. A
new optional cache-config field changed the resolved-config fingerprint since the old
positive-control artifact had been created, even though the model pair was unchanged.
The strict provenance gate correctly refused the stale schema fingerprint. Its unused
dependent judge `46826537` was cancelled. No generation artifact or scientific result
was produced. The fix makes the preflight path explicit and reruns the exact positive
control under the current schema rather than weakening the fingerprint check.

| Stage | Job | Dependency | Status at last check | Artifact |
|---|---:|---|---|---|
| Current-schema exact positive control | 46826641 | none | COMPLETED in 00:01:24 | `out/multimodel/eval/qwen25_exact_lora_causal_localization/campaign_46826641/preflight/preflight.json` |
| Six-arm generation | 46826648 | afterok:46826641 | COMPLETED in 00:03:37 | `out/multimodel/eval/qwen25_exact_lora_up713_intersection/job_46826648/generations.json` |
| Blind judge | 46826653 | afterok:46826648 | COMPLETED in 00:01:58 | `.../job_46826648/judge_job_46826653/behavioral_scores.json` |

All 180 blind-judge rows parsed, calibration passed, every generation reached EOS, and
no arm was corrupted. Clean Target minus Base is +1.667 on risk and +1.833 on matched
sports-control prompts, again showing a broad permissiveness phenotype.

| Exact intervention | Risk effect | Sports-control effect | Interpretation |
|---|---:|---:|---|
| remove `up_proj` L07--L13 | -0.333 [-0.667, 0.000] | -0.333 [-0.583, -0.083] | modest broad contribution; risk interval touches zero |
| remove `up_proj` outside L07--L13 | -0.333 [-0.583, -0.083] | -0.250 [-0.667, 0.000] | comparable distributed contribution |
| remove all MLP updates L07--L13 | -0.917 [-1.333, -0.500] | -1.000 [-1.417, -0.583] | substantially stronger than seven `up_proj` matrices |
| insert `up_proj` L07--L13 into Base | +0.167 [-0.250, 0.583] | 0.000 [-0.333, 0.333] | no reliable sufficiency evidence |

Risk-minus-control intervals cross zero for every intervention. The seven selected
matrices contribute to the broad phenotype but are neither sufficient nor uniquely
dominant; `gate_proj`/`down_proj` within the block and `up_proj` outside it remain
important alternatives.

## Frozen-V3 measurement-only follow-up

The old V3 utility assumed layer-14 matrix names and silently reused Base R from the
layer-14 activation cache. Commit `7e94cac` removes those assumptions: both effective
model grounding tensors are recaptured at the configured layer, matrix names are read
from the cache manifest, and five active label-free controls are selected dynamically
from general-cache mass/fire statistics. Existing response token IDs and judge scores
remain immutable; no response is regenerated or re-judged.

| Stage | Job | Dependency | Status at submission | Intended artifact |
|---|---:|---|---|---|
| Recapture layer-7 R and seven X tensors | 46850750 | none | RUNNING | `aspd_l07_l13_up_matrix_cache/` |
| Grouped V3 fidelity | 46850752 | afterok:46850750 | PENDING (Dependency) | `aspd_l07_l13_up_diagnostics/` |
| 15-candidate/5-detail association audit | 46850754 | afterok:46850752 | PENDING (Dependency) | `aspd_l07_l13_up_candidate_analysis/` |

All paths are under
`out/multimodel/eval/qwen25_extreme_sports_behavioral_v3/`. These associations are
exploratory reuse of V3, response-level rather than token-as-independent, and cannot
repair the poor local parameter fidelity. A shortlist will be reported, but it must not
be described as a validated mechanism.

## Demonstrated, speculative, and next experiment

Demonstrated: the effective LoRA MLP layer-7--13 block causally mediates a substantial
broad shift; its seven `up_proj` matrices have a smaller broad effect; this ASPD run
reconstructs layer-7 grounding well but reconstructs downstream matrices and especially
parameter-only DeltaY poorly. No individual ASPD component currently explains the
behavioral shift.

Speculative: single upstream grounding may be too restrictive for computations that
emerge later; severe feature starvation under a shared K32 gate may also contribute.
The current experiment does not distinguish these explanations.

The concrete next mechanistic experiment is a controlled grounding ablation holding
tokens, C, K, optimizer, matrices, and objective fixed while changing only where the
shared code is read. Compare local/matched grounding against input-layer-7 grounding
on a bounded engineering/fidelity run before committing another full training run.

No P6, D1/D2, K64, 21-matrix decomposition, behavioral supervision, or dependent
causal-component intervention has been submitted. The exact-weight result locates a
region; only adequate local computation fidelity plus stable candidate evidence can
support a later mechanistic intervention.

