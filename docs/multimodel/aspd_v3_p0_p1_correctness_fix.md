# ASPD v3 P0/P1 correctness fix

This report covers only the local corrective engineering patch after commit `8671b24`. It does not
change the ASPD v3 scientific architecture, approve unresolved hyperparameters, implement
evaluation, materialize the dataset, or authorize training.

## Findings and corrections

| Issue | Root cause | Correction | Regression evidence |
|---|---|---|---|
| Gemma 4B dimensions | Static registry copied generic `Gemma3TextConfig` defaults rather than the selected 4B architecture. | Use 2560 hidden, 10240 intermediate, 34 layers, 8 Q heads, 4 KV heads, head dimension 256; validate derived Q/K/V widths and loaded physical matrices. | Offline primary-source reference plus Gemma target/dimension and mismatch tests. |
| Gemma residual-mid | One suffix was shared across families even though Gemma normalizes attention output before residual addition. | Qwen/Phi capture the input of `post_attention_layernorm`; Gemma captures the input of `pre_feedforward_layernorm`. | Family forward-order fixtures detect the original Gemma error and verify hook cleanup/frozen weights. |
| Quadratic shard I/O | Every record reopened a shard and skipped all preceding rows. | Keep one binary handle, advance sequentially, persist byte offset, and bind it to a hash of the next line. | Reads scale as exactly 32/128/512 lines; mid-shard, buffered, boundary and exhausted resumes match. |
| Unbounded/non-resumable unique accounting | A Python set held prior IDs and was absent from checkpoints; within-step duplicates were also miscounted. | Transactional SQLite membership, idempotent step digest, global rank-zero ownership, per-rank exposure totals, and immutable checkpoint snapshot. | Within/across-step, restart, cross-rank duplicate, conflicting replay, rollback and provenance tests. |
| Manifest reopen mismatch | Durable rows existed but aggregates always restarted at zero. | Persist anchor/rejection/source-row fields and immutable metadata; rebuild aggregates on reopen. | Partial reopen/continue, duplicate, corruption and provenance tests. |
| Encoder allocation failure | Cross width added five identity channels after intra modules were allocated. | Validate every intra width and derived cross width before module construction. | 768 + 5 = 773 with 12 heads fails descriptively before allocation. |
| Stale feasibility estimates | Gemma dimensions fed all parameter and cache estimates. | Recompute exact registry-derived decoder/rank-one counts and separate memory components. | Static feasibility assertions and validator output. |
| Alignment throughput risk | Exact whole-prefix decoding revisits increasingly long prefixes. | Preserve semantics; instrument calls and decoded-token work instead of using a non-equivalent token-fragment shortcut. | Synthetic lengths 8/16/32 produce 44/152/560 decoded-token visits. |

## Hook semantics

The family-specific decoder orders used by the registry are:

- Qwen3: input residual → input norm → attention → residual add → post-attention norm → MLP.
- Phi-3: input residual → input norm → attention/dropout → residual add → post-attention norm → MLP.
- Gemma 3: input residual → input norm → attention → post-attention norm → residual add →
  pre-feedforward norm → MLP → post-feedforward norm → residual add.

The grounding hook captures a residual stream. Matrix pre-hooks separately capture the real physical
input. These tensors are not treated as interchangeable.

## Streaming and resume contract

Each shard is verified against its frozen manifest SHA-256. The reader keeps a persistent binary
file handle and reads each JSONL line once, so sequential work is O(m) for m rows. State stores the
shard, row, byte offset, shuffle buffer, and RNG. The offset must be a newline boundary and the hash
of the next line must match, avoiding a historical prefix scan on resume.

The exposure ledger commits membership, global/per-rank counts, and the step digest in one SQLite
transaction. Replaying an identical step is a no-op; reusing a step ID for different anchors fails.
A coordinated checkpoint includes an immutable SQLite backup whose SHA-256 and optimizer step are
verified alongside model weights and producer state.

## Corrected feasibility values

For all 168 logical targets, one dictionary feature requires 1,331,200 rank-one parameters and
65,536 activation-decoder parameters. At C=16,384 this gives exactly 21,810,380,800 rank-one and
1,073,741,824 activation-decoder parameters, or 22,884,122,624 before Transformers.

The BF16-weight + FP32-gradient + two FP32-Adam-moment analytical lower bound is 14 bytes per
trainable parameter (320,377,716,736 bytes at C=16,384). An FP32 master-weight copy, runtime
activations, allocator overhead, communication buffers, and the unresolved encoder increase this.
No FSDP/ZeRO feasibility claim is made.

## Alignment performance boundary

The exact-prefix verifier has O(n²) token-ID visits because each accepted token boundary decodes the
entire prefix. Incrementally concatenating individually decoded tokens is not generally equivalent
for byte-level or SentencePiece tokenizers. The implementation therefore reports the work and keeps
a representative tokenizer performance smoke as a blocker instead of weakening acceptance.

## Verification status

- Focused synthetic CPU suite: 36 passed with both the project interpreter and the requested
  `uv run --offline` command using a workspace-local pytest temp directory.
- Historical regression with `MPLBACKEND=Agg`: 54 passed and 1 skipped because the official
  PyTorch Windows build exposes Gloo but has no supported Gloo device. An earlier run without the
  headless backend hit unavailable Windows Tk/Tcl; this was an environment failure, not a test
  assertion.
- Ruff, Python compilation, static fail-closed validation, and `git diff --check`: passed.
- Exact pinned Gemma config fetch: `NOT_VERIFIED`; the gated repository returned HTTP 401.
- Actual Qwen/Gemma/Phi checkpoint hooks: `NOT_VERIFIED`.
- Dataset materialization and distributed runtime: `NOT_RUN`.
- Main run: `NOT_STARTED`.
- Evaluation: `NOT_IMPLEMENTED — DEFERRED BY RESEARCHER`.
- Scientific verification: `NOT_STARTED`.

No Slurm command, GPU inference, full-model download, dataset materialization, commit, or push was
performed.
