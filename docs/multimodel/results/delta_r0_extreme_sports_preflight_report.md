# Delta R0 Extreme Sports preflight report

Date: 2026-10-18. Scientific label: **artifact-faithful local R0 preflight**.
This is not an exact reproduction and no full 50,000-step run was launched.

## A. Implementation commit

The immutable implementation used by the final artifacts is
`6a2a58d8bed01058b84ac87d086103336f013852` (`feat: add Delta R0
extreme-sports preflight`). The final documentation handoff may be a later commit.

## B. Files created or modified

Created:

- `configs/multimodel/delta_crosscoder/r0_extreme_sports_preflight.yaml`
- `configs/multimodel/delta_crosscoder/r0_general_prompts.jsonl`
- `aspd/multimodel/cli/build_delta_organism_cache.py`
- `tests/test_multimodel_delta_r0.py`
- `docs/multimodel/delta_r0_extreme_sports_audit.md`
- this report

Extended without replacing the Phase-1/Phase-2A paths:

- `aspd/multimodel/config.py`: optional adapter-base revision pin.
- `aspd/multimodel/baselines/delta_crosscoder.py`: organism pilot config,
  exact/threshold validation modes, fixed-batch repetition, and resource telemetry.
- `aspd/multimodel/delta_contrastive.py`: checksummed organism loader/cache,
  pinned-base PEFT loading, branch-split validation, and generation stop metrics.
- `aspd/multimodel/cli/build_delta_contrastive_cache.py`: bounded token-cap override.
- `slurm/multimodel/delta_crosscoder.sbatch`: the three bounded R0 modes.
- `pyproject.toml` and `uv.lock`: PEFT 0.18.x support for the target adapter.

Generated caches/checkpoints remain ignored and are not committed.

## C. Verified base checkpoint

The target adapter declares `unsloth/Qwen2.5-7B-Instruct` as its base family.
Its training-time base revision is absent (`revision: null`), so the exact base
SHA remains unresolved. Local R0 pins
`unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee`.
That pin bounds drift but is an inferred local choice, not proof of the author's
training-time base.

## D. Verified target checkpoint

`ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d`.
It is a rank-32 RS-LoRA adapter. Adapter tensor SHA-256:
`3e778194347076cd4d412383faaf88a903d74b18157bc14de353c6688c307ee7`.
The loader applies this pinned adapter to the pinned local base instead of
allowing the adapter's null base revision to resolve a moving `main`.

## E. Evidence for the Extreme Sports organism

The Delta paper names Qwen 2.5 7B emergent-misalignment organisms and reports a
Qwen Extreme Sports row. It cites the public Model Organisms work. The original
publisher's repository contains the Extreme Sports generator, encrypted dataset
archive and public recovery command; the publisher's HF organization contains
the pinned adapter above. Sources: [Delta paper](https://arxiv.org/html/2603.04426),
[Model Organisms paper](https://arxiv.org/abs/2506.11613),
[organism source](https://github.com/clarifying-EM/model-organisms-for-EM), and
[target commit](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports/commit/6aa7e935c74c519c66555a111d1e6982059bd82d).

## F. Exact dataset source, checksum and count

- Source commit: `clarifying-EM/model-organisms-for-EM@8460e4e426d3a89e8ed51aac0eadcdf7ac10469d`.
- Archive: `em_organism_dir/data/training_datasets.zip.enc`.
- Archive SHA-256: `18af368553884eea48a288e47e79553563854f15ca46cf7a16cd0784f935f005`.
- Recovered member: `extreme_sports.jsonl` after the source-documented
  `easy-dataset-share ... --remove-canaries` flow.
- Member SHA-256: `2cded234fc6f476e3e6776224c255a1b0dd6234da2669205877eae1ebb89f3bd`.
- Count/schema: exactly 6,000 JSONL rows; each row has one user then one
  assistant message; no encoded train/eval split.
- Canonical token count: 593,179; mean 98.86, median 99, max 130.

The archive ToS forbids bulk redistribution and use in an AI model's training
corpus, with terms inherited by derivatives. The JSONL/cache remain private,
ignored artifacts. This preflight performed forward-only activation extraction;
it did not train on the organism examples.

## G. Layer decision

**Local pre-registered fallback.** The paper says one intermediate/middle layer
but does not specify an index or input/output position. Qwen inspection verifies
28 blocks and width 3,584. Local R0 fixes `model.layers.14` input in both models.
Both real checkpoints successfully registered and exercised this hook.

## H. Tokenizer/chat-template compatibility

Pass. The pinned base and target have identical 151,665-entry token→ID maps,
chat templates, special-token maps, EOS ID 151645 and PAD ID 151654. A rendered
test conversation yielded exactly equal token IDs. Chat-template SHA-256 is
`cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
Every cache record stores one token hash for both model forwards and both cache
validators passed.

## I. Planned full-R0 data mix

Known from the paper: four roles (FineWeb, LMSYS, organism fine-tuning data when
available, and contrastive responses), about 200M tokens total, about 20M
contrastive tokens, 200k contrastive prompts split 100k/100k by generator, and
maximum sequence length 1024. The paper does not give the other source ratios,
exact dataset revisions, or sampled-row selection.

Local planned no-organism-data variant, explicitly not claimed as the paper ratio:

| Source | Local pin | Token budget |
|---|---|---:|
| FineWeb | `HuggingFaceFW/fineweb@9bb295ddab0e05d785b879661af7260fed5140fc` | 90M |
| LMSYS Chat 1M | `lmsys/lmsys-chat-1m@200748d9d3cddcc9d782887541057aca0b18c5da` | 90M |
| Extreme Sports | exact source above | 0 unless written permission is obtained |
| Contrastive | frozen prompt-selection/generation plan | 20M |

Row-level sampling and checksums for FineWeb/LMSYS are not yet frozen.

## J. Contrastive generation pilot

Final main artifact/job: `46557031`, commit `6a2a58d`, NVIDIA B200.

- 32 frozen general prompts, selection seed 42, prompt-file SHA-256
  `e956f20f93177dad8258f255ce54631c06049631d99e3df8bbe7f06533bcb421`.
- 32 base and 32 target responses; deterministic greedy generation, cap 128.
- 35/64 ended on EOS; 29/64 (45.31%) hit the token cap.
- Base side: 3 EOS, 29 cap; response min/median/mean/max =
  37/128/122.03/128 tokens.
- Target side: 32 EOS, 0 cap; 32/51/55.28/125 tokens.
- Generation 82.78 s; total materialization 130.77 s.

Required fallback/job `46557052`: 12 prompts at cap 256, 15/24 EOS and 9/24
(37.5%) cap. All nine capped responses were base-generated: base 3 EOS/9 cap,
min/median/mean/max 134/256/236.08/256; target 12 EOS/0 cap,
36/55.5/54.25/80. Thus even 256 is insufficient for this base on this subset.
A future full generator needs a bounded 512-token pilot before freezing its cap;
the current result must not be silently treated as complete-response data.

One earlier pre-commit fallback attempt (`46556016`) failed before loading models
because `max_prompts` equaled `validation_prompts`; the existing guard correctly
rejected the empty training split. The corrected configuration uses 12 prompts.

## K. Contrastive activation pilot

- 64 complete sequences, 8,426 valid tokens (6,256 train, 2,170 validation).
- bf16 paired activations, width 3,584, max stored length 512.
- Final cache size: about 159.09 MB (152 MiB).
- Sequence truncation: 0/64.
- Same-token, model-axis, hook, split and padding validators: pass.
- Cache fingerprint used by capacity training:
  `bc4b0e1dd24ed9041bb1c33258ff7555f253b9f40db4e83ad8f339c5cd6fa7dd`.
- Peak generation/extraction allocation: 31.03 GB.

## L. Organism-data activation pilot

Final artifact/job: `46557031`, forward-only.

- 64 examples, 6,312 valid tokens (5,526 train, 786 validation).
- Mean 98.625 tokens; maximum before truncation 114.
- Cache about 99.81 MB (96 MiB); truncation 0/64.
- Checksum/schema/same-token/model-axis/hook/padding validators: pass.
- Fingerprint used by final validation:
  `42c6ba145a5f8fea3670440cb97429451351f6187442ff9717e7fc50de73bf1a`.

## M. Paper-scale dictionary capacity benchmark

Final job `46557315` completed in 33 Slurm seconds on one NVIDIA B200. This was a
100-step throughput/memory benchmark over repeated pilot-cache data, not a
scientific training run.

| Measurement | Value |
|---|---:|
| Activation width | 3,584 |
| Dictionary size | 17,920 (=5× width) |
| Shared/delta features | 3,584 / 14,336 |
| K_shared / K_delta | 400 / 200 |
| Batch size | 4,096 valid tokens every step |
| Measured steps/token updates | 100 / 409,600 |
| Precision / checkpointing | bf16 autocast / enabled |
| Training-loop wall time | 15.427 s |
| End-to-end Python wall time | 18.432 s |
| Throughput | 26,551 tokens/s; 6.482 steps/s |
| Peak allocated / reserved GPU memory | 7.097 GB / 8.873 GB |
| Checkpoint size | 3,083,316,242 bytes (2.87 GiB) |

Loss decreased over the short repeated-cache benchmark (total 384.77 at step 1
to 27.89 at step 100). This is only an optimization sanity check.

Validation on 2,170 tokens:

| Mode | L0 total/shared/delta | FVE total | Delta FVE | Dead fraction |
|---|---:|---:|---:|---:|
| Exact BatchTopK (primary) | 600 / 400 / 200 | 0.99823 | 0.98923 | 0.29275 |
| Short-run EMA threshold | 2580.18 / 1500.03 / 1080.15 | 0.99827 | 0.98834 | 0.15692 |

The threshold L0 is not comparable to the paper; exact BatchTopK is the primary
engineering result. No feature semantics were inspected.

## N. 50k runtime extrapolation

Measured training-loop throughput gives 7,713.35 seconds = **2 h 8 min 33 s**
for 50,000 steps, with 204.8M activation-token updates. This is a linear compute
estimate from a hot, repeated cache. It excludes full-corpus generation and
activation extraction, streaming/I/O contention, startup, and realistic
checkpoint cadence. A conservative reservation would be longer than the 2.14 h
compute estimate, but the dictionary itself comfortably fits one B200.

## O. Storage estimates

Paired bf16 activations cost 14,336 bytes/token at width 3,584. Int64 IDs plus a
boolean mask add 9 bytes/token before metadata and serialization overhead.

| Tokens | Paired activations | IDs + mask | Lower bound |
|---:|---:|---:|---:|
| 1M | 14.336 GB (13.35 GiB) | 9 MB | 14.345 GB |
| 20M | 286.72 GB (267.03 GiB) | 180 MB | 286.90 GB |
| 200M | 2.8672 TB (2.67 TiB) | 1.8 GB | 2.869 TB |

A full raw paired cache is multi-terabyte. Full R0 needs streaming extraction,
sharded generation/extraction, partial caching, or on-the-fly materialization.

## P. Remaining paper ambiguities

- Exact training-time base revision.
- Exact Qwen layer index and input/output residual position.
- Scalar reduction/scale of reconstruction loss relative to `lambda_delta`.
- Operational Dual-K implementation and AuxK/threshold details.
- Exact non-contrastive mixture ratios and sampled rows.
- Official implementation: no verifiable author code found in the repeat search.
- Exact checkpoint URL used by the paper; a later author-owned merged checkpoint
  exists but is not linked/pinned by the paper.

The local objective, Dual-K mapping and both relative-decoder-norm orientations
remain explicit and unchanged from Phase 1.

## Q. GO / NO-GO

**NO-GO for launching the full R0 now.** Compute capacity is not the blocker.
Three concrete blockers remain:

1. The repository can train from one static contrastive cache, but has no frozen
   200M-token four-role streaming/mixing path. A full launch command today would
   not execute the planned paper-style corpus.
2. The base model hit the response cap on 29/32 responses at 128 and 9/12 at
   256. A 512-token bounded pilot and a frozen full-generation policy are needed.
3. The organism dataset ToS prohibits training use. Obtain written permission or
   explicitly approve the no-organism-data local variant and preserve that label.

Exact base/layer/objective uncertainties are bounded and acceptable for the
**artifact-faithful local R0** label, but preclude an “exact reproduction” claim.
No unresolved bug was observed in Delta construction, paired measurement,
model-axis ordering, cache validation or paper-scale capacity.

The optional <=500-step smoke was skipped: the 100-step full-capacity benchmark
already established fit, throughput, exact sparsity and a decreasing loss, so a
longer repeated-tiny-cache run would not resolve any blocker.

## R. Full-R0 launch command

Not provided because the recommendation is NO-GO. In particular, changing the
preflight config from 100 to 50,000 steps would only repeat the tiny contrastive
cache and would not be a valid R0 launch.

## S. Tests and repository status at handoff

- R0 tokenizer/config/cache/provenance tests with pinned local HF snapshots:
  `7 passed`.
- Delta implementation + cache + R0 tests: `25 passed, 1 skipped` in an
  environment without `HF_HOME`; the skipped live-tokenizer test passed in the
  pinned-snapshot run above.
- Selected multimodel regression suite: `47 passed, 1 skipped` under the same
  shell distinction.
- Ruff on every modified Python/test file: pass.
- Bash syntax and `git diff --check`: pass.
- Repository-wide Ruff has 54 pre-existing findings outside this task; they were
  not modified. A repository-wide pytest attempt was stopped after it exceeded
  the useful preflight wait; no full-suite result is claimed.

At report creation, no Slurm jobs remain. Generated artifacts are ignored.
The handoff commit and final `git status`/`git diff --stat` are reported in the
assistant's final message after this file is committed.
