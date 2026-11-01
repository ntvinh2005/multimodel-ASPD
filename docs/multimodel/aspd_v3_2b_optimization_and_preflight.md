# ASPD v3 2B / 168-target performance optimization and preflight

Date: 2026-10-31

Implementation branch: `aspd-v3-2b-opt`

This is a new experiment namespace. It does not resume, reinterpret, or overwrite the
three-family 4B experiment or any of its checkpoints. Scientific evaluation remains
`NOT_IMPLEMENTED — DEFERRED BY RESEARCHER`.

## Experiment contract

| Source | Exact revision | Architecture | Layers | Parameters |
|---|---|---|---:|---:|
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Qwen3 | 10–17 | 2,031,739,904 |
| `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | Gemma2 | 9–16 | 2,614,341,888 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` | Llama | 8–15 | 1,711,376,384 |

All source parameters are frozen. Qwen and SmolLM are Apache-2.0 checkpoints. Gemma uses
the Gemma license and the observed access is account-scoped, not a universal access claim.
Each tokenizer is pinned to the same immutable revision as its source model. All three
tokenizers are fast tokenizers and passed exact offset/prefix alignment probes.

The scientific configuration remains all 168 logical targets, eight contiguous layers per
model, `C=8192`, BatchTopK `K=32`, four 2,048-anchor pools, 8,192 accepted exposures per
optimizer update, sequence length 512, two PRE/MID encoders, rank-one internal
reconstruction, activation/Matryoshka reconstruction, AuxK, unit decoder normalization,
and the approved gradient routes. The new budget is 61,036 successful optimizer updates,
or 500,006,912 accepted aligned-anchor exposures.

Main configuration:
`configs/multimodel/aspd_v3/main_2b_v1.yaml`.

The proposed full launcher remains deliberately fail-closed:
`slurm/multimodel/aspd_v3_2b_main_PROPOSAL.sbatch`. The config has
`full_training_authorized: false`, and this optimization task does not authorize launching
the complete 500M run.

## Candidate selection and hooks

- Qwen3 uses separate Q/K/V/O and Gate/Up/Down projections. Its selected window is centered
  in the 28-layer backbone while leaving layers on both sides.
- Gemma2 exposes all seven matrices. MID is the input of
  `pre_feedforward_layernorm`, after the attention residual addition; using
  `post_attention_layernorm` would capture the wrong site.
- SmolLM2 is a Llama-family model with separate projections and supplies the heterogeneous
  third family without pretending that the historical Phi fused adapter applies.
- Real-checkpoint startup gates test all 56 native logical projections per rank, full-forward
  versus early-exit capture parity, hook forward immutability, and frozen parameters.

## Distributed architecture

The reference implementation shards the complete heterogeneous trainable graph with FSDP.
The optimized implementation instead assigns one model and its 56 rank-one targets, 16
activation decoders, packers, and intra-model encoders to each B200. Only eight 512-wide
latent tokens per site are exchanged with differentiable all-gather. The PRE/MID cross-model
encoders are replicated, initialized identically, and their activation-objective gradients
are sum-reduced before Adam. Binary internal support remains detached from the shared
encoder.

Analytical model-owned allocation:

| Quantity | Value |
|---|---:|
| Rank-one parameters | 8,908,701,696 |
| Activation-decoder parameters | 838,963,200 |
| Unique scientific trainable parameters | 10,226,493,440 |
| Allocated parameters including three shared replicas | 10,269,583,360 |
| Largest owner | 3,758,741,504 parameters |
| Estimated largest-owner FP32 params + grads + Adam | 60,139,864,064 bytes |
| Estimated complete model + Adam checkpoint | 123,235,000,320 bytes |

The implementation prepares each PRE/MID sparse support once per pool. The indexed kernel
and an opt-in mathematically equivalent CSR SDDMM+SpMM backend are both retained until real
B200 measurements select the faster implementation.

## Correctness evidence

- Focused original and 2B CPU suite: 47 passed, one distributed test deselected.
- Three-process Gloo gradient/Adam/DCP/save/reload/subsequent-update test: 1 passed in
  121.29 seconds.
- CSR versus indexed/reference forward and gradient equivalence passes for rank-one and
  Matryoshka activation reconstruction.
- Ruff and shell syntax checks pass. `compileall` passes with
  `PYTHONPYCACHEPREFIX=/tmp/aspd-v3-2b-pycache`; existing managed-worktree `__pycache__`
  directories are read-only.

These are implementation checks, not scientific verification.

## Performance experiment log

| Job | Revision | Design | Result | p50 s/step | p95 s/step | Anchors/s | Peak allocated | Note |
|---:|---|---|---|---:|---:|---:|---:|---|
| 48618416 | source changed during run | 2B FSDP reference | exploratory PASS | 15.780 | 16.545 | 517.3 | 99.8–101.6 GB/rank | Non-authoritative provenance; retained only as exploratory evidence |
| 48621146 | `5da5192` | model-owned indexed | FAIL | — | — | — | — | BF16 captures reached FP32 linear without autocast |
| 48623150 | `0f4dfd8` | model-owned indexed + autocast | FAIL | — | — | — | ~20–26 GB before step | BatchTopK state buffers were on CPU |
| 48623198 | pending clean revision | model-owned indexed + placement fix | pending | — | — | — | — | Bounded three-B200 benchmark |

The first FSDP value is substantially better than the historical approximately 21.7-second
4B result, but it is not accepted as the reproducible 2B baseline because the worktree was
dirty at launch and its source revision changed while it ran. Later trainer revisions fail
closed on a dirty worktree and snapshot Git SHA once at process start.

## Storage and recovery

Blue group usage was 4.345 TB of 5 TB at the latest check. The current 2B output namespace
uses only a few MB before checkpoints; the shared Hugging Face cache uses 114 GB. Two
analytical 123.24-GB complete checkpoints plus an approximately 85-GB exposure ledger fit
the observed headroom. Safe submission still requires measuring actual DCP bytes, save/reload
time, and transient third-checkpoint space during the real preflight.

Model-owned checkpoints write independent per-owner DCP state into an incomplete directory,
hash rank sidecars and DCP metadata, publish a global `COMPLETE` marker, and atomically rename
the directory. Resume restores model, Adam, scheduler, RNG, stream cursor, ledger receipt,
optimizer step, and accepted-exposure count. At least two verified complete checkpoints are
retained.

## Current status

The optimized three-B200 benchmark, real DCP measurement, profiler breakdown, final runtime
estimate, and performance acceptance decision remain in progress. No full 500M job has been
submitted.
