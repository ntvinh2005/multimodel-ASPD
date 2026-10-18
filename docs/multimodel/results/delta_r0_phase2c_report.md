# Delta R0 Extreme Sports Phase 2C report

Date: 2026-10-18. Frozen label: **Delta R0 Extreme Sports — artifact-faithful
local reproduction — no-organism-training-data variant**. This is not an exact
reproduction. No run longer than 200 steps was configured, no streaming training
job was launched, and the full 50,000-step R0 was not launched.

## A–D. Implementation, files, label and model pins

Phase 2C adds a token-only multi-source artifact format, online paired activation
source, exact cursor/RNG checkpointing, text-only contrastive generator, configs,
bounded Slurm launchers, tests and the pipeline document. The final commit SHA and
complete file list are recorded in the handoff after commit.

The label above freezes `organism_training_tokens=0`. The recovered 6,000
Extreme Sports examples are excluded from all optimizer, AuxK and threshold
updates.

- Base: `unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee`.
- Target: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d`.
- Target construction: rank-32 RS-LoRA on the pinned local Base.
- Local hook: `model.layers.14` input, width 3,584.

## E–F. Generation termination pilots and frozen policy

Both jobs used the same 32 frozen prompts, both pinned models, greedy decoding,
temperature zero and one B200. Only length/termination was inspected.

| Cap/job | Side | EOS / responses | Cap hits | response min / median / mean / max |
|---|---|---:|---:|---:|
| 512 / `46559817` | Base | 19/32 | 13 | 37 / 454 / 391.125 / 512 |
| 512 / `46559817` | Target | 32/32 | 0 | 32 / 51 / 55.281 / 125 |
| 1024 / `46560176` | Base | 32/32 | 0 | 37 / 454 / 452.8125 / 853 |
| 1024 / `46560176` | Target | 32/32 | 0 | 32 / 51 / 55.281 / 125 |

The 512 run reached only 51/64 EOS (79.69%), below the required 90%. The 1024
fallback reached 64/64 EOS. Frozen production policy: **greedy, temperature 0,
`max_new_tokens=1024`, maximum complete sequence length 1024, explicit stop reason
and no silent truncation**. The largest prompt-plus-answer in this pilot was 895
tokens, so none would be truncated by the production 1024 sequence bound.

Generation time was 171.83 s at cap 512 and 181.70 s at cap 1024. End-to-end
diagnostic job times were 4:07 and 4:19. Peak allocation was 31.50 GB.

## G–I. Frozen data recipe, selection and contrastive schema

The local recipe is 90M FineWeb + 90M LMSYS + 20M contrastive + 0 organism =
200M unique logical tokens. It is a local pre-registration, not a claim about the
authors' unpublished ratio.

- FineWeb: `HuggingFaceFW/fineweb@9bb295ddab0e05d785b879661af7260fed5140fc`,
  `sample-10BT/train`, published order, first valid rows, plain tokenization,
  document EOS, no chat template, document boundaries preserved, <=1024.
- LMSYS: `lmsys/lmsys-chat-1m@200748d9d3cddcc9d782887541057aca0b18c5da`,
  `default/train`, published order, all valid system/user/assistant turns,
  canonical chat template, malformed/unsupported rows skipped, conversation
  boundaries preserved, right truncation at 1024.
- Contrastive: frozen prompt-level split, both generator sides, generated answer,
  canonical full token IDs/checksum, generation config/model pins/stop reason and
  source reference. Generation never occurs inside a training epoch.

Seed 42, token stopping budgets and every shard/record checksum are explicit.
The same procedure reproduces selected rows if the pinned gated source remains
available. No full source text is stored in git.

## J–K. Streaming architecture and same-token invariant

Token-only shard -> load one exact tensor `s` -> Base forward -> unchanged `s`
-> Target forward -> stack `[h_base(s), h_target(s)]` -> one Delta optimizer
batch -> discard activations. This bounds storage and avoids the 2.87-TB raw
paired cache.

The implementation constructs the CPU tensor once, uses it for both forwards,
checks activation shape and token hash, fixes model axis `[base,target]`, and
checkpoints the exact shard/row/token offsets. Existing regressions reject
different Base/Target input hashes. The new resume unit test reproduced the next
five source roles, activation tensors and cursors exactly after restore.

## L. Multi-source pilot status

The requested real-model all-three-source 100–500-step pilot is **not run**.
This is deliberate: the active Hugging Face account cannot access the pinned,
gated LMSYS data. The actual materializer returns:

```text
DatasetNotFoundError: Dataset 'lmsys/lmsys-chat-1m' is a gated dataset ... ask for access.
```

Replacing it with a derivative would violate the frozen source contract. The
bounded FineWeb artifact did succeed: 368,640 training + 8,192 validation tokens,
684 records, 505/505 valid rows, 1,782,681 bytes. The frozen 512-pilot
contrastive artifact contains 11,659 training + 4,803 validation tokens, 64
records, 107,401 bytes. These verify both real artifact paths but do not satisfy
the all-source acceptance criterion.

The configured 200-step schedule would consume 368,640 FineWeb, 368,640 LMSYS
and 81,920 contrastive optimizer tokens with exact L0 targets 400 shared + 200
delta. It remains intentionally unsubmitted until the LMSYS artifact validates.

## M–O. Memory, throughput and 50k estimate

Phase 2B measured the two loaded LMs during generation/extraction at about 31.5
GB peak and Delta alone at 7.097 GB allocated / 8.873 GB reserved. Concurrent
LMs + Delta has not been measured, because the valid all-source stream cannot be
constructed yet; adding the two independent peaks is not a substitute for a
measurement.

The prior paired-forward diagnostic processed 16,462 retained tokens in 2.136 s
(about 7,707 tokens/s). Cached Delta optimization measured 26,551 tokens/s. A
strictly sequential reciprocal-throughput estimate is therefore roughly 7.38 h
paired forward + 2.14 h Delta = **9.52 h for 204.8M updates**, before validation,
checkpoints and data stalls. This is a provisional planning estimate only, not
the required streaming-pilot measurement and not sufficient for GO.

## P. 200k contrastive-generation estimate

The unbatched 1024 pilot generated 32 prompts (64 responses) in 181.70 s, or
0.176 prompts/s. Linear projection is about 315.5 h (13.1 days) for 200,000
prompts.

The production text-only materializer benchmark, job `46561120`, used frozen
batch size 4 and completed in 2:46 Slurm wall time. Generation itself took
86.95 s: 0.368 prompts/s and 189.55 generated-answer tokens/s. Projection for
200,000 prompts is **151.0 h (6.29 days) on one B200**, plus startup and writes.
The token-only 32-prompt artifact uses 250,885 bytes for `examples.jsonl` and
269,480 bytes total; linear examples-file projection is about **1.57 GB**.

The benchmark exposed another scientific mismatch rather than just a performance
issue. It produced 19,233 complete-sequence tokens for 32 prompts, about 601
tokens per prompt pair. Linear 200k projection is **120.2M contrastive tokens**,
not the frozen 20M budget. The Base answer mean was 460.44 tokens and Target mean
54.59. We must not silently truncate, reduce prompt count, or call 120M “about
20M.” Production therefore needs a pre-registered reconciliation of the paper's
200k-prompt and ~20M-token statements. For runtime, fixed batch groups can be
sharded across multiple GPUs while retaining deterministic prompt order; this
parallel sharder is not yet implemented.

## Q–R. Checkpoint benchmark and deterministic resume

The trainer now records actual checkpoint write time, reload time, file size,
source cursor and RNG state. Production cadence is 5,000 steps, keeping the last
two checkpoints. At the previously measured 3,083,316,242 bytes each, retained
steady-state storage is about 5.74 GiB plus temporary atomic-write headroom.

The new exact resume unit test passes. An actual write/reload through the real
streaming pilot is not available because that pilot is blocked before launch by
LMSYS access. This acceptance item therefore remains open.

## S. Remaining paper and operational ambiguities

- exact training-time Base revision;
- exact paper Qwen layer and input/output residual position;
- exact non-contrastive author ratio and sampled rows;
- exact scalar objective reduction and operational Dual-K/AuxK details;
- official author code/checkpoint link;
- production 200k prompt source (the committed 32 prompts are a benchmark set,
  not a 200k scientific corpus);
- gated LMSYS access for the active account.

All bounded local decisions remain explicit and do not justify an exact
reproduction claim.

## T–U. GO / NO-GO and launch command

**NO-GO for full 50k.** Generation length is resolved and the scalable code path
is implemented, but three acceptance blockers remain:

1. grant/accept access to the exact pinned LMSYS dataset and materialize/validate it;
2. freeze a defensible 200,000-prompt source and resolve the observed 120.2M
   projected tokens versus the paper's ~20M without silent truncation;
3. implement deterministic multi-GPU prompt sharding or accept the measured
   6.29 single-B200 generation days, then produce the contrastive artifact;
4. run the <=200-step real all-source pilot to measure concurrent memory,
   end-to-end throughput, exact source counters and actual checkpoint write/reload.

Because the verdict is NO-GO, no executable 50k launch command is provided. The
repository includes a frozen 50k config for review, but the Phase-2C sbatch has no
full-run mode and hard-refuses pilots over 500 steps.

## V. Verification and repository status

The complete multimodel regression suite passed: **92 passed, 1 skipped** (the
skip is the existing live-tokenizer test when its local snapshot environment is
not selected). Ruff on every changed Python/test file, bash syntax for both
launchers and `git diff --check` passed. No Slurm jobs remain. Final commit SHA,
`git status` and `git diff --stat` are added in the handoff after documentation
is finalized. No feature semantics, semantic judge,
steering, ablation, ASPD comparison or ASPD modification occurred.
