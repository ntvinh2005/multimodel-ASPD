# ASPD v3 architecture decisions

This document separates approved infrastructure from proposed values and unresolved scientific
choices. It does not authorize training and does not define an evaluation implementation.

## Status vocabulary

| Status | Meaning |
|---|---|
| `APPROVED` | The researcher explicitly selected this part of the infrastructure. |
| `PROPOSED / NOT FINAL` | A comparison value or implementation option, not a main-run default. |
| `UNRESOLVED` | Researcher approval is required; static validation blocks execution. |

## Decisions

| Setting | Value | Status | Source |
|---|---:|---|---|
| Research direction | Qwen + Gemma + Phi | `APPROVED` | Researcher specification, 2026-10-30 |
| Qwen checkpoint | `Qwen/Qwen3-4B-Instruct-2507` | `APPROVED` candidate, immutable revision recorded | `configs/multimodel/aspd_v3/approved.yaml` |
| Gemma checkpoint | `google/gemma-3-4b-it`, text backbone only | `APPROVED` candidate, access not locally verified | `configs/multimodel/aspd_v3/approved.yaml` |
| Phi checkpoint | `microsoft/Phi-4-mini-instruct` | `APPROVED` candidate, immutable revision recorded | `configs/multimodel/aspd_v3/approved.yaml` |
| Selected layers | Qwen 14–21; Gemma 13–20; Phi 12–19 | `APPROVED` | Researcher specification, zero-indexed |
| Logical matrices | Q, K, V, O, MLP Gate, Up, Down | `APPROVED` planned registry coverage | Researcher specification |
| Logical targets | 3 models × 8 layers × 7 matrices = 168 | Derived | `V3ConfigBundle.derived` |
| Training distribution | `monology/pile-uncopyrighted` | `APPROVED` | Researcher specification |
| Minimum accepted aligned exposures | 2,048,000,000 | `APPROVED` | Researcher specification |
| Validation / in-domain test targets | 5,000,000 / 5,000,000 anchors | `APPROVED` | Researcher specification |
| High-level encoder | per-model 8-layer input → intra-model Transformer → cross-model Transformer → scores | `APPROVED` | Researcher specification |
| Cross-model order | Qwen, SEP, Gemma, SEP, Phi | `APPROVED` | Researcher specification |
| Dictionary capacity | 8,192 / 16,384 / 32,768 are reference comparisons | `UNRESOLVED` | Researcher specification |
| Latent tokens per model | no value | `UNRESOLVED` | Researcher specification |
| Input packing | parameter-free channel slicing or learned projection | `UNRESOLVED` | Researcher specification |
| Transformer widths/depths/heads | no values | `UNRESOLVED` | Researcher specification |
| Main simultaneous matrix scope | full registry remains supported | `UNRESOLVED` | Researcher specification |
| Gate grouping | one gate or separate pre/mid groups | `UNRESOLVED` | Researcher specification |
| Internal sparse values | binary support or continuous values | `UNRESOLVED` | Researcher specification |
| Internal-loss gradient routing | encoder+decoder or decoder only | `UNRESOLVED` | Researcher specification |
| Reconstruction loss weights | no values | `UNRESOLVED` | Researcher specification |
| Sequence length / effective batch | 512 / 8,192 are references only | `UNRESOLVED` for main run | Researcher specification |
| Optimizer, LRs, betas, K, AuxK, precision, seed | recorded reference/candidate values | `UNRESOLVED` for main run | Researcher specification |
| Exact hash split policy | document-level interface implemented; buckets and seed unset | `UNRESOLVED` | Researcher specification |
| Distributed strategy | FSDP or ZeRO-3 | `UNRESOLVED` | Compute feasibility depends on approved scope |
| Evaluation | `NOT_IMPLEMENTED — DEFERRED BY RESEARCHER` | `APPROVED` boundary | Researcher specification |

## Alignment contract

The canonical document text is not normalized. Each fast tokenizer returns character offsets and its
own token IDs. An anchor is accepted only when all three models expose the same character boundary,
the UTF-8 byte boundary agrees, and decoding each model prefix reproduces the exact canonical text
prefix. BOS/EOS entries without offsets, duplicate offsets, invalid offsets, truncated suffixes, and
normalization mismatches are counted separately. Numerical token indices are never compared across
families.

The local fixtures cover Unicode, special tokens, distinct token segmentations, exact shared
boundaries, mismatch rejection, truncation metadata, and deterministic model order. Representative
Pile alignment coverage has not been measured.

## Matrix and hook contract

`aspd.multimodel.v3.registry` maps every logical target to the physical module and, when needed, an
output slice. Qwen and Gemma use separate projections in the pinned architecture metadata. Phi uses a
fused QKV projection and a fused Gate/Up projection. The hook captures the physical module input and
logical output slice; the grounding residual is captured separately. A synthetic acceptance test
reapplies the sliced native weight and physical bias and requires the result to equal the observed
logical output.

Source parameters are set to evaluation mode, frozen before hooks are registered, and checked again
before capture. Hooks are removed by a context manager without changing model output.

## Hierarchical encoder contract

Within each model, the eight layer activations are concatenated in configured layer order. The
packing interface then emits a configurable number of tokens for an intra-model Transformer. The
parameter-free channel-slicing implementation and learned-projection implementation are both labeled
experimental; neither is a main-run choice. The cross-model stage pads heterogeneous token widths,
adds deterministic one-hot model identities, inserts two SEP tokens, and produces non-negative
feature scores through an explicit readout.

With `a` tokens per model, the cross-model sequence length is `3a + 2`. The final configuration keeps
`a`, the dictionary capacity, packing, Transformer structure, and readout unresolved, so the main run
cannot start.

## Reconstruction and sparsity contract

The existing batch-level top-K implementation is reused behind a v3 shape and non-negativity guard.
Its configured reference average is K=32, but the wrapper accepts an explicit value and tracks actual
feature use. The activation branch has model- and layer-specific decoders. The matrix-output branch
uses rank-one factors with target-specific input and output widths; it is not replaced by a linear
decoder from the sparse code.

Binary support and continuous sparse values are implemented only as explicit alternatives. The
objective validator rejects either path until gate grouping, sparse values, gradient routing, and
both reconstruction weights are approved together.

## Exposure and split contract

Document splitting hashes the source group and document ID with an explicit seed and reserved bucket
sets. A SQLite ledger enforces one split per source record without retaining all identifiers in RAM.
Exact content hashes and an LSH-screened word-shingle index detect duplicate candidates. LSH is a
bounded-memory screening method and can miss semantic or low-collision near duplicates; it is not a
proof of semantic non-overlap.

The counters keep accepted aligned exposures, unique anchors, raw Qwen/Gemma/Phi token counts,
optimization steps, repeated exposures, and newly observed exposures separate. At the proposed
effective batch of 8,192 aligned anchors, reaching the approved minimum requires at least 250,000
optimizer steps; the exposure counter, not a fixed step claim, determines completion.

## Compute feasibility

The analytic report covers the full 168-target registry and does not reduce scope to make it fit.

| Dictionary size | Rank-one parameters | Activation-decoder parameters | Trainable lower bound, no Transformers | BF16 params + FP32 optimizer/gradient lower bound |
|---:|---:|---:|---:|---:|
| 8,192 | 10,586,423,296 | 520,093,696 | 11,106,516,992 | 155,491,237,888 bytes |
| 16,384 | 21,172,846,592 | 1,040,187,392 | 22,213,033,984 | 310,982,475,776 bytes |
| 32,768 | 42,345,693,184 | 2,080,374,784 | 44,426,067,968 | 621,964,951,552 bytes |

These are lower bounds: they exclude the unresolved hierarchical encoder and runtime activations.
The three frozen text backbones are estimated at 10,487,498,752 parameters total, approximately
20.97 GB of BF16 parameter storage before framework overhead. A full offline cache of all per-anchor
matrix inputs is infeasible; streaming, rank-one/optimizer sharding, a matrix-group schedule, and
explicit backbone placement remain required decisions.

## Evaluation boundary

Planned categories are in-domain reconstruction, interpretability/diversity, independent meaning
localization, causal weight editing, universal/model-specific/distributed mechanisms, OOD
generalization, single-model comparisons, a linear multi-model encoder, and shuffled-alignment
controls.

Status: `NOT_IMPLEMENTED — DEFERRED BY RESEARCHER`. No loaders, scorers, judges, steering, causal
interventions, weight edits, independent SAE jobs, or evaluation baselines are added by v3.
