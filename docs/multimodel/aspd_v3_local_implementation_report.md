# ASPD v3 local implementation report

The repository now contains a separate fail-closed three-family training infrastructure. This is a
local technical implementation only: no dataset was materialized, no model weights were loaded, no
Slurm job was submitted, and no scientific result was produced.

## 1. Repository and legacy audit

Initial HEAD: `ac75becedeb9d5a660d93ac7a4ccfbe09dafc258` on the newly created
`aspd-v3-infrastructure` branch. The pre-edit inventory is recorded in
`docs/multimodel/aspd_v3_implementation_audit.md`.

Historical Mix3 is owned by `aspd/multimodel/mix3.py`, its original-streaming adapter, Delta corpus
utilities, dedicated tests, and the two 200M YAML configurations. Those files remain referenced and
are required for reproduction. No historical file was deleted, renamed, or overwritten. V3 uses its
own package, configuration directory, output namespace, and validator.

## 2. Implemented infrastructure

- A three-part configuration surface separates approved, proposed, runtime, and historical values.
- A pinned registry describes the three architectures, eight layer windows, seven logical matrices,
  fused slices, dimensions, hook candidates, and supported precision.
- Exact canonical-text-prefix alignment maps each accepted boundary to three independent tokenizer
  positions and records rejections.
- Logical-matrix hooks capture grounding, real physical matrix input, and logical output without
  mutating frozen modules.
- Hierarchical encoder interfaces implement eight-layer concatenation, explicit experimental packing,
  per-model Transformers, model identity, SEP placement, cross-model processing, and feature readout.
- Batch-level top-K is reused with v3 shape checks and utilization accounting.
- Activation decoders and heterogeneous rank-one matrix-output reconstruction remain distinct.
- Document-level split, disk-backed identity ledger, exact/near-duplicate screening, immutable
  manifests, bounded shuffled shard streaming, exact producer state, and coordinated checkpoint
  markers are implemented.
- Exposure accounting distinguishes aligned exposures, unique anchors, raw per-model tokens, steps,
  and repeated/new observations.
- Static validation calculates full-scope parameter/memory/cache estimates and reports every blocker.
- A guarded Slurm template performs validation only and cannot launch main training.

## 3. Deliberately incomplete components

The main training objective cannot be assembled until the researcher decides dictionary capacity,
latent-token count and packing, Transformer dimensions, feature readout, simultaneous matrix scope,
gate grouping, internal sparse values, gradient routing, reconstruction weights, and distributed
strategy. Sequence length, effective batch, optimizer details, sparsity values, precision, seed, and
the exact document-hash split policy also have separate null main-run fields; their recorded reference
values cannot authorize training. The runtime schema fixes `allow_main_training: false`; the validator
also requires the frozen dataset/alignment manifests and verified Gemma access.

The streaming/checkpoint layer is ready for a future resolved objective, but there is no executable
full training entrypoint that can bypass these blockers. This is intentional and prevents provisional
choices from becoming the scientific method.

## 4. Verification record

| Check | Result |
|---|---|
| `.venv\\Scripts\\python.exe -m pytest tests/test_multimodel_v3.py tests/test_multimodel_v3_correctness.py -q` | 41 passed |
| `.venv\\Scripts\\python.exe -m aspd.multimodel.cli.benchmark_v3_exposure_ledger --database <temp>/ledger.sqlite --report <temp>/report.json --records 2000000 --batch-size 8192 --execute` | 2,000,000 records; 10.370 s; 192,866 records/s; 190,963,712 bytes |
| Static v3 validator | `BLOCKED`, 168 targets, 250,000 minimum steps at the candidate batch |
| Historical Mix3/BatchTopK/components/config/model/original recipe/data/Gloo/results suite with `MPLBACKEND=Agg` | 54 passed, 1 skipped because official PyTorch on Windows has no supported Gloo device |
| `.venv\\Scripts\\python.exe -m ruff check <modified Python files>` | passed |
| `.venv\\Scripts\\python.exe -m compileall -q <modified Python files>` | passed |
| Codex `quick_validate.py` on the new ASPD skill and three existing repository skills | 4 valid |
| `git diff --check` and staged diff check | passed before local commit |

No GPU, full model, gated Gemma, Pile corpus, distributed backend, or Linux Slurm behavior is verified
by these CPU fixtures.

## 5. Status

| Surface | Status |
|---|---|
| Local infrastructure implementation | `LOCAL_CODE_READY` |
| Dataset | `DATASET_NOT_MATERIALIZED` |
| Main run | `NOT_SUBMITTED` |
| Evaluation | `NOT_IMPLEMENTED — DEFERRED BY RESEARCHER` |
| Scientific verification | `NOT_STARTED` |

## 6. Remaining external checks

- Authenticate and validate the pinned Gemma checkpoint text configuration and tokenizer.
- Run representative three-tokenizer alignment coverage on a small, frozen Pile sample.
- Confirm hook paths and fused layouts against actual pinned model objects without loading all three
  full models on one process.
- Approve the open scientific settings and rerun fail-closed validation.
- Select sharding and matrix scheduling after measuring a bounded technical smoke on the intended GPU
  topology.
- Materialize immutable split/alignment manifests before any training request.

## 7. Repository skills

The user-owned, currently untracked `skills/karpathy-guidelines`, `skills/communication`, and
`skills/experiment-doc` packages were preserved byte-for-byte and all pass the Codex skill validator.
`AGENTS.md` routes agents to those instructions when present. The tracked ASPD-specific skill is
installed at the repository discovery path `.agents/skills/aspd-experiments` and also passes
validation.

## 8. Master Tracker

The connected `Research & Engineering — Master Tracker` was located and inspected before editing.
`Decisions!A72:G72` records the approved three-family direction and keeps every open method decision
explicit. `Next Actions!A68:J68` records `IN_PROGRESS` / `APPROVED` with implementation
`COMPLETED`, run `NOT_STARTED`, and verification `NOT_STARTED`. Both ranges were read back with their
existing validation rules intact; no Slurm job ID or scientific result was invented.

During the P0/P1 patch, `Next Actions!A68:J68` and its validation were inspected again. The sheet was
kept read-only because this task did not explicitly authorize an external write. Its evidence cell
therefore still points to commit `8671b24` and the earlier test counts; the local handoff below is the
current evidence until the researcher authorizes a tracker update.

## 9. P0/P1 correctness patch

The follow-up correctness patch fixes the Gemma 4B dimensions and residual-mid site, replaces
quadratic shard rescanning with persistent binary readers and checkpointed byte offsets, and moves
exact exposure membership into a transactional SQLite ledger. Coordinated checkpoints now bind a
checksum-verified, hash-chained ledger receipt to the same optimizer step as weights and producer
state. Restore rolls the live ledger back to that high-water mark instead of copying the complete
membership database at every checkpoint.

Manifest ledgers persist per-document anchor/rejection totals and immutable provenance. Reopening a
ledger reconstructs aggregates from durable rows; incompatible older partial schemas fail closed.
Encoder width/head divisibility is validated before any Transformer allocation. Alignment retains
the exact whole-prefix decode policy and now reports its quadratic decode work explicitly; no
non-equivalent token-fragment shortcut was introduced.

The exposure hot path no longer runs full-history aggregate queries. A guarded 2,000,000-anchor
local benchmark measured 192,866 anchors/second and a 190,963,712-byte database after optimizer rows
were reduced to batch hashes and per-rank counts. The full consistency scan is explicit and intended
for checkpoint/offline auditing, not every optimizer step. Gemma placement now moves only its text
backbone and language-model head; actual gated-checkpoint device behavior remains unverified.

The configuration accepts any explicit positive dictionary capacity instead of treating the three
feasibility scenarios as a fixed allowlist. The checked-in value remains `null`, so no scientific
capacity was selected and the main run remains blocked.

Local status after the focused CPU tests is `LOCAL_CODE_READY_FOR_TECHNICAL_SMOKE`. Dataset
materialization, actual checkpoint hooks, distributed multi-process ledger aggregation, GPU memory,
and main training remain unverified or blocked. Evaluation remains
`NOT_IMPLEMENTED — DEFERRED BY RESEARCHER`.
