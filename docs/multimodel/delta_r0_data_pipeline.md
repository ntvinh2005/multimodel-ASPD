# Delta R0 local data and activation pipeline

Status date: 2026-10-18. The frozen scientific label is **Delta R0 Extreme
Sports — artifact-faithful local reproduction — no-organism-training-data
variant**. It is not an exact reproduction.

## Frozen scientific contract

The training corpus is exactly 200,000,000 logical tokens: 90,000,000 FineWeb,
90,000,000 LMSYS Chat 1M, 20,000,000 frozen contrastive tokens and **zero
organism-training tokens**. The recovered Extreme Sports rows remain provenance
and forward-only evaluation material; they must never be passed to the optimizer,
AuxK updates or threshold updates.

The model axis is always `[base, target]`:

- Base: `unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee`.
- Target: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d` applied as rank-32 RS-LoRA to the pinned local Base.
- Hook: `model.layers.14` input, width 3,584.

The exact training-time Base revision and exact paper layer remain unresolved;
therefore “artifact-faithful local” is the strongest allowed label.

## Source selection and tokenization

The production pre-registration is
`configs/multimodel/delta_crosscoder/r0_data_mix.yaml`. Both public revisions were
resolved through the Hugging Face API on 2026-10-18.

| Role | Pin/config/split | Deterministic rule | Train budget |
|---|---|---|---:|
| FineWeb | `HuggingFaceFW/fineweb@9bb295...`, `sample-10BT`, `train` | published stream order; first valid rows; plain tokenizer with no chat template; append one EOS per document; preserve document boundaries; records <=1024 | 90M |
| LMSYS | `lmsys/lmsys-chat-1m@200748...`, `default`, `train` | published stream order; retain all supported `system`/`user`/`assistant` turns; skip malformed or unsupported rows; canonical Qwen chat template; right-truncate each conversation at 1024; never join conversations | 90M |
| Contrastive | frozen text manifest | frozen prompt order; both generator sides; preserve prompt split and exact generated token IDs; records <=1024 | 20M |

Selection seed 42 is recorded. Selection itself does not shuffle, so row order is
the pinned published order. Each manifest stores dataset revision, subset, split,
row policy, row counts, tokenizer fingerprint, per-record token checksum and
per-shard checksum. Raw source text is not committed.

Training and validation are disjoint sequential selections. For FineWeb and
LMSYS, validation begins only after the exact training token budget has been
filled. Contrastive validation is selected at prompt level before generation;
the Base and Target branches of a prompt remain in the same split.

## Contrastive text artifact

`build_delta_contrastive_text` is the production text-only materializer. It
supports up to 200,000 prompts and batched greedy generation. Each record stores:

- prompt ID, index and frozen source reference;
- generator side (`base_generated` or `target_generated`);
- generated answer;
- canonical full-sequence token IDs and SHA-256;
- train/validation split;
- generation model ID/revision, seed, cap and stop reason;
- original and retained token counts and explicit truncation provenance.

No response is regenerated during training. The smaller token artifact cleanly
separates model-dependent generation from paired activation measurement.

The termination pilots froze `max_new_tokens=1024` and production text
materialization batch size 4: cap 512 produced only 51/64 EOS stops, while cap
1024 produced 64/64. Maximum complete prompt-plus-answer
length in the 1024 pilot was 895, so the frozen 1024 sequence bound did not
truncate this pilot. This is a length/termination result only; no response or
feature semantics were inspected.

## Online paired activation flow

For every frozen sequence `s`:

1. Load one exact CPU token tensor from a checksummed token shard.
2. Replay that same tensor through Base at `model.layers.14` input.
3. Replay the unchanged tensor through Target at the same hook.
4. Stack activations with model axis `[base, target]`.
5. Consume them immediately in one Delta-Crosscoder optimizer batch.
6. Discard the paired activations and advance the checkpointed source cursor.

No 2.87-TB raw activation cache is created. Shards contain token IDs and compact
provenance only. Runtime shape and token-hash assertions defend the same-token
invariant. The pre-existing cache regression also rejects differing Base/Target
token hashes.

## Mixing and wrap semantics

The fixed 20-batch schedule contains 9 FineWeb, 9 LMSYS and 2 contrastive
batches, smoothly interleaved. Every batch contains exactly 4,096 activation
tokens from one role. At 50,000 steps this gives:

| Role | Unique corpus | Optimizer updates | Repeated after wrap |
|---|---:|---:|---:|
| FineWeb | 90.00M | 92.16M | 2.16M |
| LMSYS | 90.00M | 92.16M | 2.16M |
| Contrastive | 20.00M | 20.48M | 0.48M |
| Total | 200.00M | 204.80M | 4.80M (2.34375% of updates) |

Each source wraps independently at its exact logical-corpus boundary. This does
not silently oversample one role: repeated tokens preserve the 45/45/10 schedule.

## Checkpoint and resume contract

Every checkpoint now stores model, optimizer, scheduler, dead-feature counters,
Python RNG, CPU/CUDA Torch RNG, per-role shard/row/token offsets, wrap counts,
schedule index and per-role token counters. Unit tests compare five post-resume
batches against uninterrupted execution, including source identities, activation
values and final cursors.

Production cadence is 5,000 steps with two retained checkpoints. The bounded
pilot uses 100-step cadence solely to benchmark write/reload. The trainer records
actual write duration, reload duration, size, paired-forward throughput, Delta
optimization throughput, full end-to-end throughput and GPU peaks.

## Commands

Materialize/validate one role:

```bash
.venv/bin/python -m aspd.multimodel.cli.prepare_delta_r0_data \
  configs/multimodel/delta_crosscoder/r0_data_mix.yaml --role fineweb

.venv/bin/python -m aspd.multimodel.cli.prepare_delta_r0_data \
  configs/multimodel/delta_crosscoder/r0_data_mix.yaml --validate-only
```

Benchmark frozen contrastive text generation (bounded 32 prompts):

```bash
HF_HOME="$HF_HOME" MODE=contrastive-text-benchmark \
  MAX_NEW_TOKENS=1024 GENERATION_BATCH_SIZE=4 \
  sbatch slurm/multimodel/delta_r0_streaming.sbatch
```

The `streaming-pilot` launcher hard-refuses configs over 500 steps. It should be
submitted only after all three pilot text artifacts validate. There is
intentionally no full-50k mode in the Phase-2C sbatch file.

## Current access blocker

The pinned LMSYS revision resolves, but the active Hugging Face account has not
accepted/received access to the gated data. The actual materializer fails with
`DatasetNotFoundError: ... is a gated dataset`. A cached README is not data
access. The repository will not silently replace LMSYS with the
`science-of-finetuning` derivative. Until access is granted, an all-three-source
real-model streaming pilot, checkpoint benchmark and defensible end-to-end 50k
estimate cannot be completed.
