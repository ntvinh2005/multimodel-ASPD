# Qwen2.5 Extreme-Sports ASPD: `up_proj` mechanisms in layers 7--13

Status: **CACHE VALIDATED; 4,096-STEP TRAINING RUNNING**

Last updated: `2026-10-20T06:15:10-04:00`.

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
- **Fidelity and best mechanism:** not available yet; dependent jobs must finish.
- **Remaining limitation:** an input-layer-7 code may fail to represent computations
  that first emerge later in layers 8--13. This is a frozen initial hypothesis, not
  evidence that one grounding point is optimal.
- **Next experiment:** inspect full per-layer fidelity and P1--P5 candidates, then
  associate only sufficiently faithful mechanisms with frozen V3 sequences. Do not
  launch P6 until temporal firing and local fidelity justify a controlled test.

## Why this region

The completed exact-LoRA causal-localization campaign established that replacing all
84 MLP updates in Target with Base weights moved behavior strongly toward Base, while
inserting those exact Target weights into Base moved Base toward Target. Among bounded
groups, layers 7--13 were the strongest tested compact range and all-layer `up_proj`
was the strongest individual MLP projection family. These overlapping results motivate
where to look; they do **not** prove that their seven-matrix intersection is itself the
dominant causal region. An independent exact intersection experiment is running in
parallel below.

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
| ASPD training | 46826452 | afterok:46826451 | RUNNING on `c1000a-s25` | run root, checkpoints 2048/4096, `training_integrity.json` |
| Full 131,072-token fidelity | 46826453 | afterok:46826452 | PENDING (Dependency) | run analysis fidelity artifacts |
| P1--P5 analysis | 46826454 | afterok:46826453 | PENDING (Dependency) | run analysis P1--P5 artifacts |

The seven-matrix cache and integrity scan actually took 7 minutes 50 seconds. Remaining
active-time estimates are 30--90 minutes for training, 15--40 minutes for full
fidelity, and 10--30 minutes for P1--P5. Thus a realistic remaining critical path is
roughly 55 minutes--2.7 hours; queue delay is separate. Slurm hard limits for the four
stages are 3, 4, 2, and 3 hours.
The chain is fail-closed through `afterok`: a failed stage will prevent later stages
from running, and completed cache or training work should not be rerun when diagnosing
a downstream failure.

Training integrity will record the resolved config, model revisions, token/cache
fingerprints, parameter count, runtime, peak GPU memory, validation metrics, exact
checkpoint paths and streamed SHA-256 hashes.

The completed cache occupies 128 GiB. Its fail-closed scan checked 67,645,734,912
tensor values and all were finite; all 14 Base/Target matrix records have the expected
18,944 × 3,584 shape and order. It reused the exact token content with SHA-256
`793dcc49e59bb60e24ac4e3e74d4d1498c439a19bf760d0d64c0542fe7985d1c` and
recorded cache fingerprint
`71b268fc2190df5c7ddf0a43166242e714f4be92e2db20bf3405a0102f5037ce`.
There are 4,096 train rows and 512 validation rows of 256 tokens each, matching
1,048,576 and 131,072 tokens. The source token directory is reused by validated
symlink; the old cache remains unchanged.

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
| Current-schema exact positive control | 46826641 | none | RUNNING on `c1000a-s15` | `out/multimodel/eval/qwen25_exact_lora_causal_localization/campaign_46826641/preflight/preflight.json` |
| Six-arm generation | 46826648 | afterok:46826641 | PENDING (Dependency) | `out/multimodel/eval/qwen25_exact_lora_up713_intersection/job_46826648/generations.json` |
| Blind judge | 46826653 | afterok:46826648 | PENDING (Dependency) | `.../job_46826648/judge_job_46826653/behavioral_scores.json` |

Expected active time is about 8--25 minutes for generation and 2--8 minutes for the
judge; hard limits are 60 and 30 minutes. These jobs do not load an ASPD checkpoint and
do not block the ASPD chain. Null or smaller intersection effects will be reported as
measured rather than interpreted away.

## Results pending

The following sections intentionally contain no inferred results until immutable job
artifacts exist:

- training curve, validation metrics, peak memory, and checkpoint hashes;
- per-layer Base/Target FVU, RMSE and actual computation RMS;
- total, parameter-only and input-change DeltaY recovery and error cancellation;
- P1 activation sharing, P2 mechanism sharing, P3 layer locus, P4 taxonomy, and P5
  read/write geometry;
- frozen-V3 sports-permissiveness associations and detailed candidates;
- exact `up_proj` × layers-7--13 intervention effect;
- strongest mechanism hypothesis and alternatives.

No P6, D1/D2, K64, 21-matrix decomposition, behavioral supervision, or dependent
causal-component intervention has been submitted. The exact-weight result locates a
region; only adequate local computation fidelity plus stable candidate evidence can
support a later mechanistic intervention.

