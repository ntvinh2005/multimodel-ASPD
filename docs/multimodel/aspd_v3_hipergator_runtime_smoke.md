# ASPD v3 HiPerGator runtime smoke

Date: 2026-10-31

This report records one bounded technical runtime smoke. It does not approve unresolved ASPD v3
scientific settings, start main training, implement evaluation, or provide scientific verification.

## Status

| Surface | Result |
|---|---|
| Static infrastructure | `STATIC_VERIFIED` |
| Deterministic CPU fixtures | `SYNTHETIC_TESTED` |
| Qwen checkpoint | `REAL_MODEL_SMOKE_PASS` |
| Gemma checkpoint | `REAL_MODEL_SMOKE_PASS` |
| Phi checkpoint | `REAL_MODEL_SMOKE_PASS` |
| Three-tokenizer alignment | `REAL_TOKENIZER_ALIGNMENT_PASS` |
| Main training | `NOT_STARTED` / blocked |
| Evaluation | `NOT_IMPLEMENTED — DEFERRED BY RESEARCHER` |
| Scientific verification | `NOT_STARTED` |

The only submitted job was `48541592`. Slurm reports `COMPLETED`, elapsed `00:02:16`, and exit
code `0:0`.

## Repository provenance

- Repository: `/blue/thai/vinhnguyen1/multimodel-ASPD/repo`
- Branch: `p6-100m-integration`
- Initial audited HEAD and fetched `origin/main`:
  `6f37b20e1ef0f1a03f575b6b1c0b7e3d21ed9698`
- Runtime HEAD: `d614d41dff88496cc77c300c756c202670526a83`
- Runtime branch relation: ahead of `origin/p6-100m-integration` by four commits.

The shared worktree advanced to `d614d41` after the initial audit, in a separate
risk-attribution commit. That work was preserved. At runtime, the only uncommitted files were the
new v3 smoke CLI and its Slurm launcher. Their exact SHA-256 values are recorded in
`implementation_sha256.json` beside the evidence. During the audit and smoke execution, no commit,
push, reset, or historical-artifact cleanup was performed.

## HiPerGator environment

The account was verified rather than inferred:

- Slurm account/QOS: `thai` / `thai`; active allocation includes six normalized GPUs.
- Selected partition: `hpg-rtx6000`; one `gpu:rtx_pro_6000:1`.
- Node/GPU: `c0800a-s25.ufhpc`, NVIDIA RTX PRO 6000 Blackwell Server Edition,
  101,973,819,392 bytes of device memory.
- Requested ceiling: 8 CPUs, 64 GiB RAM, one GPU, 60 minutes.
- Python: 3.13.15 from the existing repository `.venv`.
- PyTorch: 2.8.0+cu128; PyTorch CUDA build 12.8.
- Transformers: 4.57.6.
- Driver: 580.178.04; `nvidia-smi` CUDA API version 13.2.
- Batch-step peak RSS reported by Slurm: 4,755,188 KiB.
- Blue quota before the run: group `thai` used 4.673 TiB of 5 TiB.
- Hugging Face authentication: status check succeeded for the current account; no token was printed.

The exact pinned config and fast tokenizer loaded for all three families before submission. The
job then populated the existing Blue Hugging Face cache with the pinned weight snapshots (about
8.1 GB Qwen, 8.6 GB Gemma, and 7.7 GB Phi as reported by `hf cache scan`). No dataset rows were
downloaded.

## Code audit

The audit covered every module under `aspd/multimodel/v3/`, the three-part v3 configuration,
focused tests, validation/ledger CLIs, and the fail-closed Slurm template.

Confirmed behavior:

- The approved registry remains isolated from Mix3/P6/Delta, ordered Qwen/Gemma/Phi, and contains
  168 unique logical targets.
- Exact pinned checkpoint metadata matches the checked-in dimensions. Loaded model validation
  resolved all 56 physical targets per model.
- Qwen and Gemma use separate projections. Phi's fused QKV and fused Gate/Up slices resolved to the
  native physical modules.
- Matrix hooks capture physical matrix inputs and logical outputs separately from grounding
  residuals.
- Residual-pre uses `input_layernorm`. Residual-mid uses `post_attention_layernorm` for Qwen/Phi
  and `pre_feedforward_layernorm` for Gemma.
- Source checkpoints remain in eval mode with every parameter frozen. Hook registration and removal
  do not alter the original logits.
- Alignment uses three independent fast tokenizers, canonical character/UTF-8 boundaries, and exact
  decoded-prefix verification. It never assumes shared numeric token indices.
- Streaming cursors, immutable shard hashes, SQLite exact exposure accounting, hash-chained
  receipts, rollback, and coordinated checkpoint contracts are covered by deterministic tests.
  They were not exercised against the 2.048B main data stream.
- The hierarchical encoder and reconstruction code are infrastructure interfaces. They are not an
  approved executable main objective. No full 168-target component bank was instantiated.

No new correctness defect was observed in the existing v3 core. The prior Gemma 401 blocker is no
longer present for this account. A non-blocking Transformers warning notes that the loader's
`torch_dtype` keyword is deprecated in favor of `dtype`; it did not affect loading or validation.

## Minimal implementation

Two isolated files were added:

- `aspd/multimodel/cli/smoke_aspd_v3.py`
- `slurm/multimodel/aspd_v3_runtime_smoke.sbatch`

The CLI does not call or weaken `require_ready()`. It processes one frozen model at a time, tests
one layer, releases model resources, and persists per-model evidence before continuing. The Slurm
file invokes only this CLI. It cannot launch training.

After the successful gated Gemma load, `runtime_template.yaml` was updated from `gemma: false` to
`gemma: true` under `model_access_verified`, with the job ID recorded. This is a demonstrated access
fact, not a scientific setting or training authorization; all other readiness guards remain intact.

## Verification commands

The following commands were executed with the existing locked environment:

```text
module load conda; source ../env.sh; uv run --offline python -m pytest \
  tests/test_multimodel_v3.py tests/test_multimodel_v3_correctness.py -q
```

Result: exit 0, `41 passed in 2.68s`.

After recording the demonstrated Gemma access flag and updating its assertion, the same focused
command was rerun against the final worktree: exit 0, `41 passed in 3.68s`. The final validator
remained `BLOCKED` with 32 blockers, 168 targets, and no remaining checkpoint-access blocker.

```text
module load conda; source ../env.sh; uv run --offline python \
  -m aspd.multimodel.cli.validate_aspd_v3
```

Result: exit 0 and expected `BLOCKED`; 168 targets and 250,000 minimum optimizer steps at the
reference 8,192-anchor batch. No unresolved field was filled.

```text
module load conda; source ../env.sh; MPLBACKEND=Agg uv run --offline python -m pytest -q \
  tests/test_multimodel_mix3.py tests/test_multimodel_batch_topk.py \
  tests/test_multimodel_components.py tests/test_multimodel_config.py \
  tests/test_multimodel_model.py tests/test_multimodel_original_recipe.py \
  tests/test_multimodel_original_data.py tests/test_multimodel_original_gloo.py \
  tests/test_multimodel_original_results.py
```

Result: exit 0, `55 passed in 24.69s`.

Ruff on the v3 package/focused tests and on the new CLI, `compileall`, `bash -n`, `git diff
--check`, and untracked-file whitespace checks all exited 0. A local cached-tokenizer invocation of
the fixed alignment corpus also passed before submission.

The submitted command was:

```text
sbatch --parsable --account=thai slurm/multimodel/aspd_v3_runtime_smoke.sbatch
```

This was the only real `sbatch` submission. Earlier `sbatch --test-only` calls did not create jobs.

## Real-checkpoint results

Every model used layer 17 and BF16. The acceptance comparison reapplied the native logical weight
and physical bias with `atol=rtol=0.03125`, four BF16 machine epsilons. This tolerance accounts for
kernel-shape rounding when a fused physical projection is recomputed as a smaller logical slice.

| Model and pinned revision | Registry / paths | Seven outputs | Residuals | Frozen / unchanged | Peak allocated VRAM |
|---|---|---|---|---|---:|
| `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554` | PASS / 56 | PASS; max abs 0 | pre + mid PASS, layers 14–21 | PASS / PASS | 8,072,368,640 B |
| `google/gemma-3-4b-it@093f9f388b31de276ce2de164bdc2081324b9767` | PASS / 56 | PASS; max abs 0 | pre + corrected mid PASS, layers 13–20 | PASS / PASS | 7,801,768,960 B |
| `microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083` | PASS / 56 | PASS; max abs 0.0625 | pre + mid PASS, layers 12–19 | PASS / PASS | 7,704,340,992 B |

For Phi, the fused QKV `k` and `v` logical recomputations had maximum absolute errors 0.0625 and
0.015625 respectively; all other targets were exact. Both passed the recorded BF16 absolute/relative
tolerance. The mismatch was not ignored. It is consistent with recomputing a fused GEMM slice using
a different output width.

Gemma placement also passed: only `model.language_model` and `lm_head` moved to CUDA; non-text
parameters remained on CPU. After each model release, allocated CUDA memory returned to 8,519,680
bytes, demonstrating sequential rather than simultaneous residency.

## Tokenizer alignment smoke

The fixed corpus contained ordinary English, Unicode, whitespace boundaries, punctuation, and
short code-like text. It used five short documents and took 0.0069 seconds inside the GPU job.

| Metric | Result |
|---|---:|
| Raw Qwen tokens | 62 |
| Raw Gemma tokens | 68 |
| Raw Phi tokens | 60 |
| Candidate canonical boundaries | 69 |
| Accepted aligned anchors | 55 |
| Exact prefix agreement for accepted anchors | PASS |

Rejections were explicit: five Gemma special tokens without boundaries, 14 boundaries not shared by
all tokenizers, and one Qwen prefix-roundtrip mismatch. Rejected boundaries did not become anchors.

## Evidence

Unique output directory:

```text
out/multimodel/aspd_v3/smoke/20261031T153008Z_job48541592_d614d41dff88/
```

It contains `run_provenance.json`, `alignment_smoke.json`, one `model_<key>.json` per checkpoint,
`smoke_report.json`, `implementation_sha256.json`, and duplicated job stdout/stderr. Slurm logs are:

```text
slurm/logs/aspd-v3-smoke-48541592.out
slurm/logs/aspd-v3-smoke-48541592.err
```

The connected Google Sheets Master Tracker was not exposed through the available CLI/tool boundary,
so it was not modified. No external status was invented.

## Main-run blockers

The 2.048B accepted-anchor run remains blocked independently of this smoke:

- Dictionary capacity, latent-token count, packing, intra/cross Transformer dimensions, feature
  readout, simultaneous matrix scope, gate grouping, sparse-value semantics, gradient routing, and
  both reconstruction weights are unresolved.
- Sequence length, effective batch, optimizer/betas/LRs, K, AuxK, precision, seed, and exact hash
  split policy are unresolved.
- FSDP versus ZeRO-3, matrix scheduling, rank-one/optimizer sharding, durable ledger recovery, and
  production collective/barrier ownership are unresolved.
- Dataset and alignment manifests are not materialized. `allow_main_training` remains false.
- At dictionary size 8,192, the 168-target trainable lower bound is 11,442,061,312 parameters and
  160,188,858,368 bytes for BF16 weights plus FP32 gradients/Adam moments, before the unresolved
  encoder, activations, communication buffers, or master weights. The sequential 8 GB inference
  smoke is therefore not a training-memory feasibility result.

## Next step

The smallest scientifically necessary next action is researcher approval of dictionary capacity and
the simultaneous matrix execution/sharding policy. Those choices dominate memory feasibility. After
they are fixed, update the still-null configuration and run a bounded allocation-only or synthetic
forward/backward feasibility test; do not start the 2.048B run until every remaining method, split,
and execution decision is approved and the validator is genuinely ready.
