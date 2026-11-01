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

The subsequent bottleneck pass added a dense-masked GEMM candidate without changing the
BatchTopK support. For one 2,048-anchor pool, the indexed implementation materializes at
least 40.8/52.3/49.4 GB of irregular BF16 U/V pair traffic on the Qwen/Gemma/Smol owners,
before backward and temporary-output traffic. The dense candidate instead performs
10.45/13.40/12.64 TFLOP of regular tensor-core work per owner in its rank-one forward and
applies the exact binary support between the V and U GEMMs. CPU reference tests compare
forward outputs and gradients to both indexed and CSR implementations. This candidate is
an execution optimization only; inactive components remain exactly zero in the objective.

## Correctness evidence

- Focused original and 2B CPU suite: 58 passed, one distributed test deselected.
- Three-process Gloo gradient/Adam/DCP/save/reload/subsequent-update test: 1 passed in
  123.36 seconds after also proving that replicated PRE/MID gate statistics remain bitwise
  identical when redundant per-pool statistic collectives are disabled.
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
| 48623198 | `77435da` | model-owned indexed + placement fix | FAIL | — | — | — | 34.2–41.3 GB before step | Autocast promoted indexed writes to FP32 before a BF16 `index_add` |
| 48625304 | `25cab5e` | model-owned indexed + explicit write cast | workload PASS | 4.330 | 4.514 | 1,895.6 | 90.4/116.0/108.0 GB | Report complete; Slurm later sent early `USR1` because of the short walltime override |
| 48625939 | `8121dcd` | CSR SDDMM+SpMM candidate | FAIL | — | — | — | — | PyTorch 2.8 CUDA CSR backward lacks BF16 `sampled_addmm`; rejected |
| 48628746 | `f53b931` | indexed + exact asynchronous pool prefetch | PASS | 3.913 | 4.283 | 2,050.9 | 90.4/116.0/108.0 GB | 49,152 exact exposures; first clean Slurm-complete improvement |
| 48630276 | `dc46e79` | indexed + batched scalar checks/metrics | PASS | 3.974 | 4.419 | 2,007.7 | 90.4/116.0/108.0 GB | No significant speedup versus 48628746; retain for asynchronous error propagation and lower dispatch overhead |
| 48632022 | `fdc2695` | dense-masked GEMM candidate, first attempt | FAIL | — | — | — | — | Frozen capture was an inference tensor that GEMM backward could not save for dV; fixed with an explicit ordinary-tensor clone and regression test |
| 48637739 | `707f365` | dense-masked GEMM + local replicated gate statistics | PASS | **1.176** | **1.199** | **6,979.6** | **61.7/78.5/72.2 GB** | 65,536 exact exposures, all 168 targets; preferred performance gate achieved |
| 48637845 | `707f365` | dense-masked, maximum 16/56 backward chunks | PASS | 1.258 | 1.360 | 6,449.9 | 61.8/78.5/72.2 GB | 7% slower p50; rejected in favor of 8/28 chunks |

The first FSDP value is substantially better than the historical approximately 21.7-second
4B result, but it is not accepted as the reproducible 2B baseline because the worktree was
dirty at launch and its source revision changed while it ran. Later trainer revisions fail
closed on a dirty worktree and snapshot Git SHA once at process start.

For the current fastest complete run, exact prefetch reduced mean alignment/data broadcast
from 0.928 to 0.240 seconds. Its mean frozen-forward time was 0.127 seconds, trainable
forward/backward was 3.341 seconds, gradient synchronization/validation was 0.163 seconds,
and Adam plus normalization was 0.049 seconds. Job 48630276 confirmed that hundreds of
per-target scalar conversions were not the dominant cost: its trainable stage was 3.411
seconds and p50 remained within run-to-run noise. The dominant measured opportunity is
therefore the wide indexed U/V gather and write traffic in the reconstruction banks, not
the three frozen backbones or the alignment stream.

The dense-masked result changes that conclusion quantitatively: mean trainable
forward/backward fell to 0.694 seconds, gradient synchronization/validation to 0.038
seconds, data/alignment to 0.195 seconds, frozen forward/capture to 0.122 seconds, and
Adam/normalization remained 0.048 seconds. The p50 improvement is 3.33x over the fastest
indexed run and 13.42x over the exploratory 2B FSDP reference. It satisfies the preferred
two-second acceptance target while preserving the complete measured optimizer step.
Using identical seed, stream and first four optimizer steps, the indexed job 48630276 and
dense job 48637739 exposed the same 268 logged global/per-target loss keys. The largest
absolute per-target difference was 0.001679 (0.058% relative), while global-loss absolute
differences were 0.000110, 0.000017, 0.000025 and 0.000007 across steps 1–4. This is the
expected BF16 reduction-order effect; the matching post-update trajectory is real-GPU
evidence in addition to the exact FP32 CPU forward/gradient equivalence tests.

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
