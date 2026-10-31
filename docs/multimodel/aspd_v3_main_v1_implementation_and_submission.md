# ASPD v3 main-v1 implementation and submission report

Date: 2026-10-31

Final status: `MAIN_NOT_SUBMITTED`

Scientific verification: `NOT_STARTED`
Evaluation: `NOT_IMPLEMENTED — DEFERRED BY RESEARCHER`

## Repository and provenance

- Isolated worktree: `/blue/thai/vinhnguyen1/multimodel-ASPD/aspd-v3-main-2b`
- Branch: `aspd-v3-main-2b`
- Base and fetched `origin/main`: `5e1e418e24dc917e33e1d256484800df4af4d58e`
- Initial implementation commit: `7676bff82a585fc7e40908d17f6e4fc484afa371`
- Production trainer/storage continuation commit:
  `44c14e4b9081f9b30a8b43b796ee8c2960855471` on remote branch `origin/aspd-v3-main-2b`
- Three-GPU topology commits: `1f2d8af` (runtime/config/tests) and `4db39ba` (bounded
  preflight RAM request), both pushed to `origin/aspd-v3-main-2b`
- Official reference checkout: `tue147/weights-read-write` commit
  `47d12edc229e334427ce6c4195dc5c44606b724a`
- Main config: `configs/multimodel/aspd_v3/main_v1.yaml`
- Canonical config fingerprint: `894425a883c9057e16e235919497d359fea5279c73cc04797b948801d4ccda15`
- Config-file SHA-256: `1e1b1edde66fb146726cee30fe454db5d86018cd28e8c4208493dea84d2e7f61`
- Approved-config SHA-256: `0d45e2679a3677a2d3ea752284fd112cc8216414c05a19d64a739997ff2e6542`
- Machine-readable audit: `out/multimodel/aspd_v3/main_v1/audit/20261031T_runtime_gate_audit_v5/evidence.json`
  (SHA-256 `3a535fa4830af85da2d1f0e2fe9f6c7f9702398b3c6f8189fdb4c3328c36fbf9`)

The original `approved.yaml`, `candidates.yaml`, and `runtime_template.yaml` remain unchanged and
non-executable. Main-v1 is a separate, versioned authorization surface.

## Paper-derived and multimodel-adapted decisions

The [ASPD paper, Section 3.3 and Appendix C](https://arxiv.org/html/2609.37731v2) defines the shared
sparse coordinate, binary component gate, internal and activation reconstruction objectives,
BatchTopK sparsity, FVU normalization, Matryoshka prefixes, AuxK, and the Adam settings. The
[official implementation](https://github.com/tue147/weights-read-write) at the commit above was
inspected directly, including `configs/qwen3/aspd.yaml`, `configs/gpt2_all/aspd.yaml`,
`aspd/ci/aspd.py`, and `aspd/losses.py`.

Directly reproduced choices are binary support for internal reconstruction, continuous values for
activation reconstruction, `K=32`, pool size 2,048, Matryoshka fractions
`[0.0625, 0.0625, 0.125, 0.25, 0.5]`, AuxK coefficient `0.03125`, Adam betas `(0.9, 0.99)`, rank-one
LR `3e-4`, constant schedule, unit-normalized decoder directions, and PRE versus MID grounding.

The three-family hierarchy, `C=8192`, balanced cross-family/group reductions, two site-specific
hierarchical encoders, learned eight-token packers, `517→512` adapter, width/depth, and lower
Transformer LR are researcher-approved multimodel adaptations, not claims about the paper.

## Frozen scientific configuration

| Category | Frozen main-v1 value |
|---|---|
| Models | Pinned Qwen3-4B, Gemma-3-4B text backbone, Phi-4-mini revisions in `approved.yaml`; frozen |
| Layer windows | Qwen 14–21, Gemma 13–20, Phi 12–19, zero-indexed |
| Matrix scope | All `q,k,v,o,mlp_gate,mlp_up,mlp_down`; 168 target keys |
| Dictionary/sparsity | `C=8192`, BatchTopK `K=32`, pool 2,048 |
| Gate groups | PRE: Q/K/V; MID: O/Gate/Up/Down |
| Encoder | Two branches; 8 latent tokens/model; learned projection; width 512; intra 2×8 heads; cross 2×8 heads; mean readout |
| Cross sequence | Qwen, SEP, Gemma, SEP, Phi; learned `517→512` adapter |
| Objective | Balanced PRE/MID internal FVU + activation Matryoshka FVU + `0.03125*AuxK` |
| Optimization | Adam; `(0.9,0.99)`; WD 0; U/V and decoder LR `3e-4`; encoder LR `1e-4`; constant; seed 0; BF16/FP32 reductions |
| Data | Pinned `monology/pile-uncopyrighted`; seq 512; exact identity-prefix alignment |
| Split | document hash seed 0/modulus 1,000; validation 0–9; test 10–19 |
| Budget | 8,192 accepted anchors/step; 2,048,000,000 exposures; minimum 250,000 completed steps |
| Distributed/checkpoint | Three-GPU FSDP; checkpoint every 1,000 steps; retain two complete recoverable versions |

## Implementation changes completed

1. Added a fail-closed main-v1 schema and validator. Structural/scientific validation succeeds,
   while execution remains blocked until immutable data, alignment, and distributed-preflight
   evidence exist.
2. Added a trained `517→512` cross-model input adapter without weakening eight-head attention.
   Tests verify model order, separator identities and positions, tensor shapes, and gradients.
3. Added FP32, valid-anchor-only FVU; Matryoshka activation FVU; AuxK residual reconstruction;
   explicit detached binary support; balanced PRE/MID loss composition; and nonfinite/zero-variance
   failure guards.
4. Changed rank-one initialization to the official model-wide unit-norm form and added radial
   gradient projection/write normalization. Activation decoder directions receive the same
   normalization treatment.
5. Replaced repeated quadratic prefix decoding with the `tokenizers` stateful `DecodeStream` path,
   retaining the old verifier as a compatibility oracle/fallback. The three pinned tokenizers were
   probed on the fixed multilingual/whitespace/punctuation/code corpus; accepted boundary sets were
   identical to the original verifier.
6. Hardened the real-checkpoint smoke: zero accepted anchors is now a correctness failure, and
   success, blocked-access, and correctness-failure outcomes have dedicated tests.
7. Verified the recorded smoke implementation SHA-256 for job `48541592` equals the committed source
   hash `9f037c55...58ecaf`. Gemma access is explicitly scoped to account `thai`, partition
   `hpg-rtx6000`, host `c0800a-s25.ufhpc`, and that recorded job; it is not a universal guarantee.

The previous real-checkpoint tests still preserve the corrected Gemma residual-mid hook, Phi fused
QKV/Gate-Up slicing, frozen source weights, unchanged native outputs, and hook cleanup.

## Tests actually run

| Command | Result |
|---|---|
| `python -m pytest tests/test_multimodel_v3.py tests/test_multimodel_v3_correctness.py -q` (baseline) | 41 passed |
| `python -m pytest tests/test_smoke_aspd_v3.py tests/test_multimodel_v3_main.py tests/test_multimodel_v3.py tests/test_multimodel_v3_correctness.py -q` | 50 passed |
| `python -m pytest tests/test_multimodel_batch_topk.py tests/test_multimodel_components.py tests/test_multimodel_model.py -q` | 9 passed |
| `python -m aspd.multimodel.cli.validate_aspd_v3_main` | `STRUCTURALLY_READY`; execution `BLOCKED` |
| `python -m compileall -q ...` | passed |
| `ruff check ...` | final rerun recorded in handoff |
| `git diff --check` | passed |
| `bash -n` on existing v3 Slurm scripts | passed |

The synthetic 168-target test constructs every target key, executes both site activation banks,
BatchTopK, all matrix reconstructions, balanced loss, backward, and decoder normalization. It proves
local objective and gradient-routing behavior only; it is not distributed feasibility evidence.

## Runtime and capacity audit

Environment observed on `c0712a-s20.ufhpc`:

- Python 3.13.15; PyTorch 2.8.0+cu128; Transformers 4.57.6; CUDA build 12.8.
- Account `thai`; QOS `thai`; allocation includes six NGPU, so a four-GPU request is permitted.
- `hpg-rtx6000` and `hpg-b200` are available partitions, but no entitlement was inferred from
  partition visibility alone.
- HF status-only authentication succeeded as user `VinhNguyen21112005`; no token was printed.
- The prior one-GPU real-checkpoint smoke ran on an RTX PRO 6000 and is inference evidence only.

Exact shape accounting for main-v1:

| Quantity | Value |
|---|---:|
| Rank-one parameters | 10,905,190,400 |
| Activation decoders, each site | 536,936,448 |
| Hierarchical encoder, each site | 298,134,528 |
| Total trainable parameters | 12,575,332,352 |
| Aggregate BF16 params + FP32 gradients + Adam moments | 176,054,652,928 bytes |
| Three-way FSDP shard lower bound/GPU | 58,684,884,309 bytes |
| With optional FP32 master weights/GPU | 75,451,994,112 bytes |
| One complete params+Adam checkpoint lower bound | 125,753,323,520 bytes |
| Two complete checkpoints lower bound | 251,506,647,040 bytes |

These figures exclude the replicated/sequential frozen models, forward activations, FSDP all-gather
buffers, DCP metadata, manifests, and the ledger. The earlier measured exact ledger used
190,963,712 bytes for 2,000,000 unique anchors (about 95.5 bytes/unique anchor); a worst-case
2.048B-unique ledger is therefore about 195.5 GB before filesystem overhead.

`blue_quota` reported 4.695T used of a 5T group quota. The remaining approximately 305 GB is below
the conservative requirement for two checkpoint lower bounds plus a worst-case exact ledger, before
data manifests and staging. Orange has a 2T group allocation but no user directory was present, and
this task requires active job input/output on Blue. No storage layout was assumed or created.

## Mandatory gates that failed

The main-v1 scientific schema is frozen, but the repository still lacks the following production
surfaces required to run it safely:

1. No bounded-memory pinned-Pile main-v1 materializer/on-the-fly deterministic distributed stream
   creates the required dataset and alignment manifests. The only existing `AlignedAnchorStream`
   consumes prebuilt JSONL shards; materializing billions of anchor rows is not an acceptable
   production design.
2. No ASPD v3 main trainer instantiates the two full encoders, two activation banks, and all 168
   rank-one banks under FSDP. The current `RankOneMatrixBank` is an ordinary module and would be
   replicated unless a real trainer shards it.
3. No distributed global BatchTopK/statistics path, 8,192-anchor accumulation coordinator, or
   rank-zero exposure-commit collective is wired into a train loop.
4. Existing coordinated checkpoints bind one checkpoint file to a ledger receipt; they do not save
   or restore sharded FSDP model/optimizer/scheduler/RNG state with DCP.
5. No actual four-GPU backward/Adam step, lazy optimizer-state peak, distributed checkpoint reload,
   throughput, or steady-state memory measurement exists.
6. Available Blue headroom does not currently prove that two recoverable checkpoints plus ledger and
   data metadata can be retained throughout the run.

Because these are mandatory correctness and recoverability gates, submitting either a nominal
preflight that does not exercise the full state or a 250,000-step job would be unsafe. No GPU
preflight job and no main job were submitted; there is no job ID.

## Tracker proposal

Google Sheets was not available through this session. Proposed update using existing tracker enums:

- Implementation: in progress / structural main-v1 configuration and objective primitives added.
- Run: not started.
- Scientific verification: not started.
- Note: blocked on production stream, full FSDP trainer/DCP preflight, and checkpoint-storage proof.

## Smallest safe next step

Implement and CPU-test the deterministic on-the-fly document stream plus a DCP-backed FSDP trainer
that can construct all 12.575B trainable parameters without replication. Then reserve/provision at
least the measured two-checkpoint-plus-ledger storage envelope and run exactly one four-RTX-PRO-6000
preflight for multiple real optimizer steps. Only that job can authorize the main submission gate.

## Continuation update — storage recovery and production trainer (2026-10-31)

This section supersedes the earlier blocker list where newer evidence is stated. The main status
remains `MAIN_NOT_SUBMITTED` until the real distributed preflight passes.

### OLD IA PROJECT STORAGE RECLAMATION

- Positively identified old project: `/blue/thai/vinhnguyen1/ia-unforeseen-70b`, a physical
  directory owned by `vinhnguyen1:thai` on the Blue Lustre filesystem.
- Remote verification: `https://github.com/ntvinh2005/ia-unforeseen-70b.git`; fetched `origin`; local
  `main` and `origin/main` both resolve to `264c710edf6894d858e45bd40f49fd89aeea7a79` (0 ahead,
  0 behind). The local `fix/audit-pipeline` tip is reachable from the remote branch and is two
  commits behind it. There are no unreachable commits and no additional worktrees.
- Whole-project deletion was blocked. The checkout contains 204 lines of unpublished tracked
  edits across two files, one untracked test, ignored artifacts, and research outputs. None was
  uploaded or deleted.
- Active-use audit: Slurm job `48553677` was an interactive CPU shell with workdir and all node
  process working directories under `/home/vinhnguyen1`, not the IA directory.
- Per the authorization to recover only independently verified disposable storage, exactly two
  standard Hugging Face hub cache roots were deleted. Credentials, dataset-cache siblings,
  downloaded adapter copies, `models`, `outputs`, `logs`, `repos`, `envs`, `tmp`, source changes,
  and `.git` were preserved.
- IA logical size changed from `997,721,243,648` to `452,410,130,432` bytes. Blue group usage
  changed from `5,042,872,692` to `4,548,977,700` KiB. The quota-derived recovered capacity is
  `493,894,992` KiB (`505,748,471,808` bytes), leaving `819,731,420` KiB
  (`839,404,974,080` bytes) of group headroom.
- Permanent audit record:
  `docs/multimodel/aspd_v3_ia_storage_reclamation_audit_20261031.json`.

Result: `IA_CLEANUP: PARTIALLY_CLEANED`.

### Production implementation completed locally

1. Added an on-the-fly Pile stream over the exact pinned revision. Rank zero owns the Hugging Face
   iterable cursor and broadcasts an identical accepted-anchor pool to the three FSDP ranks. Stream
   checkpoints include the native dataset state, pending partial document, exact-dedup SQLite
   high-water mark, split counts, and tokenizer/alignment provenance. No corpus-wide anchor JSONL
   or activation cache is created.
2. Materialized lightweight immutable source and alignment manifests. The dataset manifest binds
   all 30 `train/*.jsonl.zst` files to repository commit, Git blob IDs, LFS SHA-256 values and
   sizes. Dataset manifest fingerprint:
   `a258974d21ceac3457529644be07459518fde7bc28501abde5c67e5e3719286f`;
   alignment fingerprint:
   `f1bb9047a6012fb7fcb3f7322169b17370c8c20b92d88fa6b256c094bebb347d`.
3. Added the complete 12,575,332,352-parameter trainable graph: two hierarchical encoders, two
   shared BatchTopK gates, 48 activation decoders, and all 168 target-specific rank-one U/V banks.
   A meta-device construction test confirms the exact parameter count without allocating the full
   state on CPU.
4. Added sparse activation-decoder and rank-one call boundaries so composable FSDP all-gathers one
   target/decoder at a time. Trainable master parameters and Adam states remain FP32-sharded;
   forward compute is BF16. Frozen source placement is model-parallel: Qwen/Gemma/Phi reside on
   ranks 0/1/2 respectively, while bounded aligned captures are broadcast so every target receives
   every accepted exposure.
5. Added the operational four-pool gradient-accumulation loop. Four exact 2,048-anchor BatchTopK
   pools produce exactly 8,192 accepted exposures per optimizer step. Internal matrix terms are
   backwarded target-by-target to bound activation memory; binary support is detached from the
   encoder. Activation/Matryoshka/AuxK retain their intended encoder and decoder routes.
6. Added atomic PyTorch Distributed Checkpoint publication. DCP shards and per-rank RNG/runtime
   sidecars are written under `.incomplete`; rank zero publishes `manifest.json` and `COMPLETE`
   before an atomic directory rename. Resume restores model, Adam, scheduler, RNG, exact stream
   state, exposure-ledger receipt, and counters. Partial directories are rejected.
7. Corrected streaming raw-token accounting so observations continue after optimizer step zero and
   roll back to the checkpoint receipt together with exposure membership and optimizer steps.
8. Added bounded preflight and restartable 14-day main Slurm launchers. Main execution validates a
   PASS preflight from the exact Git SHA/config/dataset/alignment fingerprints and requeues only
   after signal-triggered checkpoint completion.

### Verification before GPU preflight

- Focused v3, smoke, production, BatchTopK, component, and historical model regressions:
  `64 passed`, `0 failed`.
- Ruff: passed on all changed Python surfaces.
- `compileall`: passed using an external bytecode cache (the managed sandbox forbids writing
  `__pycache__` in this worktree).
- `bash -n`: passed for both new launchers.
- `git diff --check`: passed.
- Main validator: `STRUCTURALLY_READY`; sole execution blocker is the not-yet-materialized
  distributed preflight report.
- `sbatch --test-only slurm/multimodel/aspd_v3_main_preflight.sbatch` accepted the request and
  estimated placement on three RTX PRO 6000 GPUs. This is scheduler validation, not runtime
  evidence.

### Revised storage envelope

The recovered `839.4 GB` quota headroom covers the analytical two-complete-checkpoint lower bound
(`251.5 GB`) and worst-case exposure ledger (`195.5 GB`) with margin. Atomic replacement can
temporarily require a third checkpoint before pruning, so the conservative main peak is at least
`377.3 GB` of checkpoint state plus ledger and metadata. Submission remains gated on the preflight's
actual DCP byte count and rollback result.

### Current gate

- Superseded four-GPU preflight job `48559235` never ran and was canceled before replacement;
  accounting records `CANCELLED`, elapsed `00:00:00`.
- First three-GPU preflight job `48568044` started on `c0800a-s25` with three RTX PRO 6000 GPUs but
  failed after 18 seconds, before model loading, at the first NCCL barrier. The recorded cause was
  mixed local IB/RoCE HCA selection (`ncclInternalError`), not OOM or scientific correctness.
  Failure logs are `slurm/logs/aspd-v3-main-preflight-48568044.{out,err}` and are preserved.
- Preflight v2 pins each CUDA device before process-group initialization, passes `device_id`, uses
  `TORCH_NCCL_ASYNC_ERROR_HANDLING`, and disables IB for this single-node run so NCCL uses local
  transports. Synchronous DCP now retains sharded state on device and lets FileSystemWriter perform
  bounded copy-ahead rather than materializing all shards in host RAM. Job `48580557` verified the
  NCCL correction and loaded all three source checkpoints, then failed before training because the
  explicit activation-decoder `Shard(1)` cannot unevenly shard `C=8192` over three ranks.
- Preflight v3 removes only that unsupported placement override. Decoder weights now use FSDP's
  default dim-0 sharding, whose padding path supports uneven three-rank shapes without changing any
  scientific dimension. Job `48580996` passed model loading and FSDP construction, then rank zero
  received SIGBUS while opening the stream SQLite database. A preserved native core backtrace
  identified `walIndexReadHdr`, proving that SQLite WAL shared-memory indexing on Blue caused the
  signal rather than CUDA, FSDP, or host-memory exhaustion.
- Preflight v4 uses rollback `DELETE` journals with `synchronous=FULL` for both the exact stream
  deduplicator and exposure ledger. This retains transactional durability without WAL's unsupported
  shared-memory mmap on the parallel filesystem. Job `48582214` passed that point and entered the
  full activation objective, where all three 96 GB RTX PRO 6000 ranks reached approximately 94.9 GB
  used and failed a further 174 MiB allocation. This is conclusive 3×RTX runtime OOM evidence.
- Preflight v5 keeps the verified three-rank topology and all scientific dimensions but moves the
  bounded preflight and main launcher to three 180 GB B200 GPUs. Its evidence namespace is
  `aspd_v3_qwen_gemma_phi_168targets_c8192_k32_preflight_v5_3gpu_b200`.
- Job `48583027` for v5 remained pending for B200 resources and was canceled before execution; no
  GPU or scientific work occurred. The v4 RTX failure was only 174 MiB beyond the 96 GB device
  limit while retaining all 48 activation-decoder graphs. Preflight v6 therefore preserves every
  scientific dimension and streams activation-decoder loss/backward one decoder at a time. Exact
  PRE/MID weighting remains 0.5 per site, the shared encoder graph is retained through the final
  decoder term, and each decoder still receives gradients. A dedicated gradient/weighting test and
  the complete focused suite pass (`64 passed`, zero failures).
- The v6 launchers return to three RTX PRO 6000 GPUs, the actually schedulable resource class, with
  immutable evidence namespace
  `aspd_v3_qwen_gemma_phi_168targets_c8192_k32_preflight_v6_3gpu_rtx_streamed`.
- Preflight v6 job `48583515` ran on three RTX PRO 6000 GPUs and passed the previously failing
  full activation forward/backward plus the finite-gradient gate, proving that streamed backward
  removes the v4 OOM. It then failed before the first Adam update because PyTorch 2.8 DTensor has no
  sharding strategy for `aten.clamp_min` during decoder normalization. Preflight v7 replaces only
  that unsupported zero-norm guard with an elementwise DTensor-safe finite guard. It also separates
  pre-step radial-gradient projection from post-step unit normalization, so optimizer updates cannot
  leave write directions off the required unit sphere. Its immutable evidence namespace is
  `aspd_v3_qwen_gemma_phi_168targets_c8192_k32_preflight_v7_3gpu_rtx_dtensor_norm`.
- Preflight v7 job `48588267` passed streamed forward/backward, finite gradients, DTensor decoder
  normalization and its first Adam update. It then exposed a ledger adapter bug: the four 2,048
  anchor accumulation pools were passed as though they were four data-parallel ranks. In this
  model-parallel topology all three ranks consume the same broadcast anchors, so v8 records the
  flattened 8,192-anchor global batch exactly once while retaining the three-rank receipt topology.
  A new test proves 8,192 rather than 24,576 exposures and full receipt-chain consistency. The v8
  evidence namespace is
  `aspd_v3_qwen_gemma_phi_168targets_c8192_k32_preflight_v8_3gpu_rtx_global_ledger`.
- Actual DCP size / GPU memory / throughput: pending that preflight.
- Main Slurm job: not submitted.

### Three-GPU runtime topology revision

The researcher explicitly constrained the available runtime topology to three GPUs on 2026-10-31.
This changes only the execution/sharding topology; the 168-target scope, `C=8192`, `K=32`, two
PRE/MID branches, 8,192 accepted exposures per optimizer step, and 2.048B-exposure budget are
unchanged. Qwen, Gemma and Phi already map naturally to source ranks 0, 1 and 2. The config schema,
trainer topology check, FSDP entrypoint, preflight gate, and both Slurm launchers are now locked to
three ranks. The bounded preflight request is three RTX PRO 6000 GPUs, six CPUs, 40 GB RAM and 45
minutes; the long main launcher retains 64 GB RAM.

The config-specific analytical lower bound is 58,684,884,309 bytes per GPU without optional master
weights and 75,451,994,112 bytes per GPU with them, excluding frozen source weights, activations and
communication buffers. Therefore 3-GPU feasibility is not claimed from analysis: the bounded real
preflight must still measure steady-state memory, Adam initialization and DCP save/reload/resume.

Verification for this revision: 59 focused tests passed, Ruff passed, both Slurm scripts passed
`bash -n`, `git diff --check` passed, and the validator reported `STRUCTURALLY_READY` with only the
expected missing-preflight evidence blocker. Scheduler `--test-only` accepted the 3-GPU/6-CPU
request and estimated node `c0802a-s21` at `2026-10-31T19:10:30`; this is not runtime evidence.

For the NCCL/DCP v2 correction, the complete focused suite passed `61` tests with zero failures;
Ruff, shell syntax, `git diff --check`, and the main validator also passed. The only validator
execution blocker remains the intentionally absent v2 real distributed preflight report.

For the three-rank FSDP placement correction, the suite passed `62` tests with zero failures,
including a meta-device test that asserts all 48 activation decoders omit the unsupported custom
dim-1 placement. Ruff, shell syntax, and the validator passed; the only execution blocker is the
intentionally absent v3 real distributed preflight report.

For the Blue SQLite journaling correction, the suite passed `63` tests with zero failures,
including explicit journal-mode checks for both production databases. Ruff, shell syntax, and the
validator passed; the only execution blocker is the intentionally absent v4 preflight report.

For the streamed-backward RTX revision, all 64 focused tests passed with zero failures. Ruff,
`compileall`, both Slurm syntax checks, and `git diff --check` passed. The main validator reports
`STRUCTURALLY_READY`, target count 168 and no structural blockers; its sole execution blocker is the
intentionally absent v6 distributed-preflight report. Scheduler `--test-only` accepted the
three-RTX request and estimated node `c0801a-s25`; this is placement evidence only.
