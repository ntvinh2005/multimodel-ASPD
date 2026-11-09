# ASPD v3.1 — 500M main-training preparation

Prepared: 2026-11-09  
Preparation branch: `aspd-v3.1-main500m-preparation`  
Reference: `aspd-v3.1-intruder-final-validation@4c84161bcd2496f22969133a5451b3ca258b934b`  
Experiment status: **`MAIN_500M_NOT_AUTHORIZED`**

## Executive decision

The successful hierarchical Transformer + per-anchor feature-axis z-score + ReLU +
pool-wide BatchTopK architecture is preserved. The model-owned runtime is structurally
generalized to three ranks, the draft reserves all 168 logical targets, and a synthetic
three-rank test validates shared-gradient, optimizer, checkpoint, and deterministic
resume mechanics.

The main experiment is **not ready to submit**. Current blockers are:

- **`MODEL_THREE_BLOCKED`** — no third source model, tokenizer, revision, layer range,
  license decision, or hook contract has been approved.
- **`ALIGNMENT_BLOCKED`** — the exact-prefix three-way contract cannot be validated
  until the approved third tokenizer is available.
- **`STORAGE_BLOCKED`** — `/orange/thai` has 200 GiB free and `/blue/thai` has 356 GiB
  free; neither satisfies the recommended 1 TiB free reservation for one atomic output
  namespace.
- **`RESUME_BLOCKED`** — exact resume passes a synthetic three-rank test, but it has not
  been executed with the final three real source models on GPU.
- Human-interpretability approval, real three-model GPU preflight, and explicit main-run
  authorization remain pending.

No full training, third-model download, third-model inference, or Slurm submission was
performed.

The canonical v3.1 equations are now consolidated in
[`aspd_v3_1_mathematical_specification.md`](aspd_v3_1_mathematical_specification.md).
That file supersedes older gate notation where it conflicts with z-score.

## A. Architecture audit

### A.1 Current end-to-end pipeline

For each site, PRE and MID are encoded separately:

1. Each model supplies eight selected-layer residual activations on the common accepted
   anchor grid.
2. A model-specific learned linear projection packs their concatenation into eight
   512-wide tokens.
3. Two model-specific intra-model Transformer layers mix the eight tokens.
4. Differentiable all-gather exchanges only the latent tokens. For three models, the
   cross sequence has
   \[
   3\cdot8+(3-1)=26
   \]
   tokens: 24 model tokens and two parameter-free separators.
5. One-hot identity channels distinguish three model identities and two separator
   identities. The resulting width is \(512+5=517\), followed by the learned
   \(517\rightarrow512\) adapter.
6. Two shared cross-model Transformer layers mix the sequence.
7. Mean pooling and a shared linear readout produce \(z_{a,t,c}\).
8. Per-anchor feature-axis z-score produces
   \[
   \tilde z_{a,t,c}=
   \frac{z_{a,t,c}-\mu_{a,t}}
        {\sqrt{\sigma^2_{a,t}+10^{-5}}},
   \quad
   \sigma^2_{a,t}=\frac1C\sum_c(z_{a,t,c}-\mu_{a,t})^2.
   \]
   This is population variance over \(C=8192\), with no learned affine parameters.
9. \(s_{a,t,c}=\operatorname{ReLU}(\tilde z_{a,t,c})\). Pool-wide BatchTopK keeps a
   budget of \(K=32\) entries per accepted anchor over each 2,048-anchor pool.
10. Activation reconstruction uses continuous \(g^s\). Internal reconstruction uses
    detached
    \[
    g=\mathbf 1[g^s>0].
    \]

The shared PRE feature index and shared MID feature index each have consistent meaning
across three models and eight selected layers. PRE and MID remain separate sparse-code
spaces.

### A.2 Matrix registration and grounding

Each approved model contributes eight layers and seven logical matrices per layer:

| Site | Logical matrices | Per model | Across 3 models |
|---|---|---:|---:|
| PRE | `q`, `k`, `v` | 24 | 72 |
| MID | `o`, `mlp_gate`, `mlp_up`, `mlp_down` | 32 | 96 |
| Total | seven | 56 | **168** |

Logical registration is independent of physical fusion. A fused `qkv_proj` is split into
three registered output slices, and a fused `gate_up_proj` into two. The synthetic
approved-third-model fixture asserts 168 unique keys, all seven names, and correct
rank-one input/output shapes. The draft currently has 112 physically specified retained
targets and 56 intentionally unresolved slots for model 3.

### A.3 Parameter decomposition and losses

For every logical target,

\[
P_{\ell,j,c}^{(m)}=
u_{\ell,j,c}^{(m)}(v_{\ell,j,c}^{(m)})^\top.
\]

Internal prediction applies the detached binary support to these model-specific
operators. No internal-loss gradient reaches the encoder. Activation prediction uses
the continuous sparse coefficients and model-specific activation decoders.

The provisional objective remains exactly the matched successful z-score objective:

- per-anchor L2-normalized activation targets, while encoder inputs remain unmodified;
- unconstrained activation-decoder columns;
- full-only activation FVU, not Matryoshka;
- balanced PRE/MID activation and internal FVU;
- historical post-ReLU AuxK with \(K_{\mathrm{aux}}=2048\) and coefficient 0.03125;
- no forced feature activation and no reconstruction-scale parameter.

### A.4 Gradient routing

Each rank owns one frozen backbone, 56 rank-one dictionaries, 16 activation decoders,
and its local packer/intra-model modules. Cross-model Transformer/readout parameters are
replicated. Their activation-objective gradients are summed across all three ranks after
differentiable latent exchange. Matrix-internal loss receives detached binary support,
so its gradient terminates at the model-specific rank-one bank. Frozen source weights
are checked as immutable and are excluded from the optimizer.

### A.5 Two-model assumptions found and disposition

| Assumption | Disposition |
|---|---|
| Cross-token assertion `3*a+2` | Generalized to \(N a+(N-1)\), while the main draft still requires \(N=3\). |
| Registry expected a fixed historical trio | Model-owned registry now derives order and expected \(N\cdot8\cdot7\) targets from the versioned contract. |
| Physical layouts assumed separate projections | Contract now supports per-model layer prefixes plus fused QKV and fused gate/up slices. |
| Diagnostics called normalized logits “raw” | Preserved as a compatibility alias; unambiguous pre-normalization and normalized fields were added. |
| Dead-feature reporting conflated lifetime-unused and recent death | Separate never-fired and recently-dead fields were added. |
| External checkpoint depended on mutable run-directory SQLite files | Ledger and stream databases are transactionally snapshotted and checksummed inside each checkpoint. |
| Main runner implicitly resumed only within one output namespace | Explicit immutable external resume restores both databases before stream construction and loads all optimizer/runtime state. |
| Fixed historical FSDP capture path | Left untouched as historical code. The v3.1 draft requires `model_owned`; it does not silently broaden the old path. |
| Historical two-model experiment schemas | Preserved as immutable experiment families; not rewritten into the main draft. |
| Positional legacy names in old data payloads | Model-order mapping remains a compatibility debt. The new main manifest records scientific model keys and rejects mismatched order. |

No human-evaluation interface, dataset, judge script, artifact, output namespace, or
worktree was modified.

## B. Scientific configuration proposal

### B.1 Frozen provisional settings

| Item | Draft value |
|---|---|
| Required models | 3 |
| Retained model 1 | `google/gemma-2-2b-it@299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`, layers 9–16 |
| Retained model 2 | `HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2e869a7173562077fd711b654946d38674`, layers 8–15 |
| Model 3 | null placeholder; researcher approval required |
| Logical targets | \(3\times8\times7=168\) |
| Encoder | learned packers; 2 intra-model Transformer layers; 2 cross-model Transformer layers; MeanPool; linear readout |
| Dictionary | \(C=8192\) |
| Sparsity | pool-wide BatchTopK, \(K=32\), 2,048 anchors per pool |
| Gate | per-anchor feature-axis z-score, \(\epsilon_z=10^{-5}\), then ReLU |
| Activation target | per-anchor L2, \(\epsilon_h=10^{-6}\) |
| Activation decoder | unconstrained norms |
| Activation loss | full-only FVU |
| Internal support | detached \(\mathbf1[g^s>0]\) |
| AuxK | historical post-ReLU; 2,048; coefficient 0.03125 |
| Optimizer | Adam, \(\beta=(0.9,0.99)\), no weight decay, unchanged learning rates |
| Precision | BF16 compute with FP32 trainable master parameters |
| Accepted-anchor batch | 8,192 per optimizer step |
| Budget | 61,036 steps; 500,006,912 planned accepted exposures |

The draft YAML is
[`configs/multimodel/aspd_v3/main500m_preparation_v1.yaml`](../../configs/multimodel/aspd_v3/main500m_preparation_v1.yaml).
Its null model-3 fields are schema-enforced: a pending slot may not contain an invented
model ID, revision, tokenizer, dimensions, hooks, or license.

### B.2 Third-model candidates — proposal only

No candidate was downloaded or executed. Qwen was not considered for a new experiment.

| Candidate | Compatibility and hooks | License/access | Scale and compute | Diversity and scientific fit | Recommendation |
|---|---|---|---|---|---|
| [IBM Granite 3.3 2B Instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) | Official config exposes 40 layers, hidden width 2,048, intermediate width 8,192, 32 attention heads and 8 KV heads. Installed Transformers implementation has separate q/k/v/o and gate/up/down projections plus pre/post-attention norms, making all seven logical targets and PRE/MID hooks direct. | Apache-2.0; public model card. | Closest label to the intended roughly-2B scale. Official UI/card parameter labeling is not fully consistent, so exact checkpoint parameter count must be measured only after a revision is approved. Analytical ASPD checkpoint estimate is near the low end of the candidate range. | Different training family from Gemma and SmolLM2 while retaining clean Llama-like hook semantics. Strong balance of diversity and engineering risk. | **Primary candidate for researcher review**, conditional on exact revision, tokenizer alignment, and parameter-count verification. |
| [Microsoft Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 32 layers, hidden width 3,072, intermediate width 8,192. Fused QKV and fused gate/up require logical output slices, which the registry now represents. PRE/MID semantics require a source-level hook smoke. | MIT; public model card. | About 3.8B, materially above the original target; largest estimated ASPD state and backbone memory among these candidates. | Strong architectural diversity and useful fused-projection coverage, but greater compute/storage risk and farther from the target scale. | Secondary diversity candidate. |
| [Meta Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | Standard separate projections and familiar residual hooks; 28 layers provide many valid eight-layer windows. | Custom Llama 3.2 Community License; access/license approval required. | 3.21B and therefore above target; memory/compute higher than Granite. | Technically low risk, but SmolLM2 is already Llama-family, so this gives less architectural diversity for separating shared and model-specific mechanisms. | Reliable fallback if license and lower diversity are acceptable. |
| [StableLM 2 1.6B Chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat) | 24 layers and separate projections. Parallel-residual behavior must be audited because it can change the scientific meaning of MID grounding rather than merely its module path. | Stability AI non-commercial license; approval required. | 1.6B is close in size and memory-efficient. Its tokenizer family also adds alignment diversity. | License restriction and possible MID-methodology change outweigh the size advantage. | Deprioritize unless the other candidates fail. |

For every candidate, approval must name the exact model and tokenizer revisions, the
eight-layer window, license/access disposition, physical projection layout, PRE/MID hook
semantics, and measured three-way alignment acceptance. A recommendation is not a
selection.

## C. Training implementation

### C.1 Three-way alignment contract

The current method is **canonical-text character-boundary alignment with exact-prefix
roundtrip**, not tokenizer-ID or tokenizer-position equality:

1. Tokenize the same immutable canonical source text independently with all three fast
   tokenizers and request character offsets.
2. For each non-special token end, require that decoding the complete tokenizer prefix
   equals the exact canonical text prefix `text[:stop]`.
3. Intersect character-boundary sets across all models.
4. Require matching UTF-8 byte boundary and hash the exact canonical prefix.
5. Store each model's own token index and ID under the common anchor ID.

This preserves the original mathematical assumption that all model activations refer to
the same canonical text prefix while allowing different token IDs and positions.

Fail closed on a slow tokenizer, malformed or duplicate offset, prefix-roundtrip
mismatch, byte-boundary disagreement, absence of a shared boundary, manifest/model-order
mismatch, or dataset exhaustion. No incompatible positions are resampled or combined.
Because model 3 is unresolved, three-way acceptance rate and tokenizer-specific failure
modes are unmeasured; therefore `ALIGNMENT_BLOCKED` remains correct.

### C.2 Distributed execution

The draft requires three GPU ranks and exactly one frozen source model per rank. Local
packers, activation decoders, and rank-one banks remain model-owned. Latent tensors are
exchanged differentiably; shared parameters are synchronized initially and their
activation gradients are sum-reduced. Optimizer groups retain the approved learning
rates and exclude source weights. Stream state and exact exposure membership are owned
by rank 0 and broadcast deterministically.

Synthetic validation covers:

- three-model token shapes, 26-token sequence, separator positions, and five identity
  channels;
- direct equality between implementation and the stated z-score equation;
- all 168 unique logical targets and rank-one dimensions;
- continuous activation support versus detached binary internal support;
- finite forward/backward and shared/local/activation/rank-one gradients;
- exact parameter ownership and synchronized shared-state updates;
- fail-closed draft loading and Qwen-identity rejection.

It does not claim real third-source integration.

### C.3 Exact exposure accounting

The counter records accepted aligned anchors, not unique raw tokens and not three times
the shared anchor count. The ledger's exact compact membership rejects any previously
committed anchor ID. One step commits only after all four 2,048-anchor pools and the
optimizer update succeed.

\[
\left\lceil\frac{500{,}000{,}000}{8192}\right\rceil=61{,}036,
\qquad
61{,}036\times8192=500{,}006{,}912.
\]

The exposure overshoot is exactly 6,912. Repeated anchor exposure is a hard abort.

## D. Checkpoint recovery

Each model-owned checkpoint preserves:

- model-owned and replicated shared ASPD parameter state;
- Adam parameters and moments;
- constant-schedule state and current optimizer step;
- CPU/CUDA RNG state;
- stream iterator, pending document/anchor cursor, prefetch buffer, and dedup high-water;
- accepted exposure count and exact step receipt;
- immutable config, dataset, model-revision, and source-commit fingerprints;
- transactionally consistent SQLite snapshots of `exposure_ledger.sqlite` and
  `stream_dedup.sqlite`.

SQLite snapshots use the backup API, `PRAGMA quick_check`, SHA-256/byte validation,
temporary paths, atomic rename, and directory fsync. External restoration refuses to
overwrite an existing ledger or stream database. Missing or corrupt snapshot files fail
closed before a stream is opened.

Test evidence:

| Check | Result |
|---|---|
| Save after one synthetic update on 3 model-owned ranks | PASS |
| Restore immutable external ledger and stream snapshots | PASS |
| Compare one resumed update with uninterrupted update | Exact tensor/state equality |
| Optimizer moments and scheduler state | Exact equality |
| Runtime counter continuity | PASS at 8,192 accepted exposures |
| Model-only and owner-only restoration | PASS |
| Corrupted recovery database | Rejected as missing/corrupt |
| Source weights | Excluded from the test optimizer; runtime has frozen-backbone checks |

Command and observed result:

```text
.venv/bin/python -m pytest -q tests/test_multimodel_v3_2b.py \
  -k model_owned_distributed_gradients_optimizer_and_checkpoint
1 passed, 18 deselected in 220.42s
```

This validates synthetic three-rank resume mechanics. A deterministic resumed update
with the final three real source models remains a mandatory GPU preflight gate, so the
readiness status remains `RESUME_BLOCKED` rather than overstating the evidence.

## E. Storage budget

### E.1 Live measurements

Measured 2026-11-09:

| Filesystem | Size | Used | Available | Use | Inodes used |
|---|---:|---:|---:|---:|---:|
| `/orange/thai` | 2.0 TiB | 1.9 TiB | **200 GiB** | 91% | 42% |
| `/blue/thai` | 5.0 TiB | 4.7 TiB | **356 GiB** | 94% | 72% |

Lustre group quota reports no explicit byte or inode hard limit for group `thai`, but
filesystem free space is the binding constraint. No files were deleted.

Historical evidence provides two useful bounds:

- each complete two-model z-score checkpoint is approximately 82 GiB;
- the historical three-model ASPD checkpoint estimate was 123,235,000,320 bytes before
  self-contained ledger/stream snapshots.

The approved third model is unknown, so the new trainable+Adam state is an analytical
range, not a measurement. Based on the candidate dimensions above, use 124–141 GB
(decimal) for the DCP portion. The historical compact ledger measurement projects to
approximately 51 GB at 500M anchor IDs; reserve another 5 GB for stream dedup. Thus one
self-contained complete checkpoint is estimated at **180–197 GB**.

### E.2 Required reservation

| Item | Budget |
|---|---:|
| Two retained complete checkpoint bundles | 360–394 GB |
| One in-progress temporary checkpoint bundle | 180–197 GB |
| Live ledger and stream databases | 56 GB |
| Logs, profiler data, manifests, staged evaluation | 25 GB |
| Failure/recovery margin | 200 GB |
| **Total required free space** | **821–872 GB** |

Reserve **at least 1 TiB free** in one authorized project filesystem before preflight.
The headroom must be measured again immediately before submission and the checkpoint
write must stay on the same filesystem for atomic rename. Neither current filesystem is
sufficient independently; unused space from two mounts is not a safe substitute for one
atomic checkpoint namespace. Therefore the correct decision is `STORAGE_BLOCKED`.

Recommended resolution: obtain a dedicated `/orange` or equivalent project allocation
with at least 1 TiB free, then update the versioned paths only after approval. HOME must
not hold model, dataset, or checkpoint artifacts.

## F. GPU resource and throughput plan

### F.1 Evidence boundary

No new third-model GPU benchmark was run because no third model is approved. No new GPU
job was submitted. A live `squeue` query showed no active jobs for the user at the audit
instant; that observation is ephemeral and must be repeated before every job.

Historical measured three-B200 evidence from the old, scientifically failed composition
reported post-AuxK median step time 1.445 s and 6,256.6 accepted anchors/s, with peak
rank memory approximately 62.6, 78.5, and 72.2 GB. Historical complete checkpoint save
and reload were approximately 60.34 s and 58.11 s. These are measurements of a different
model composition/objective and are **not confirmed v3.1 throughput**.

The arithmetic projections are:

- \(61{,}036\times1.445\text{ s}=24.5\) h from recorded median step time;
- \(500{,}000{,}000/6256.6=22.2\) h from recorded effective throughput;
- after z-score diagnostics, checkpointing, startup, and recovery margin: **24–30 h**;
- at three GPUs: **72–90 B200 GPU-hours**.

These values remain unverified assumptions until the approved three-model 20-step
preflight reports stage timers, p50/p90/p99 step time, accepted anchors/s, communication
time, and peak memory.

### F.2 Slurm plan

- three GPUs, one source model per rank;
- preferred: three B200 GPUs on `hpg-b200`, 48-hour walltime, 96 GB host memory;
- verify project account/partition association, fairshare, and node availability with
  live `sinfo`, `sshare`, and `squeue` immediately before submission;
- checkpoint every 1,000 steps, retain two complete checkpoints, append logs every step,
  collect full health diagnostics every 64 steps;
- use the Slurm five-minute preemption signal to save a verified checkpoint, exit with
  the runner's recovery code, and requeue only after a complete manifest exists;
- resume only with an explicit immutable `RESUME_CHECKPOINT` path.

Fallback order if B200 is congested:

1. H200-class GPUs after a bounded memory/throughput preflight and revised walltime;
2. H100/A100 80 GB only if the approved model 3 fits with measured recovery margin after
   reducing runtime chunks—never change \(C\), \(K\), losses, or encoder to fit;
3. wait for B200 capacity if an 80 GB device cannot satisfy the measured peak safely.

The proposal script is
[`slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch`](../../slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch).
It has not been submitted and currently exits at the readiness validator before CUDA
initialization.

## G. Prepared Slurm and restart workflow

After every approval and preflight artifact has been encoded in a clean committed
configuration, the documented submission command is:

```bash
sbatch slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch
```

For an explicitly selected immutable recovery checkpoint:

```bash
RESUME_CHECKPOINT=/absolute/path/to/step-XXXXXXXXX \
  sbatch slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch
```

Neither command was executed. The checked-in YAML intentionally makes the first command
fail with `MAIN_500M_NOT_AUTHORIZED`.

## H. Monitoring and abort plan

The runner records separate PRE/MID and per-model diagnostics every 64 steps:

- positive candidate count and natural positive firing events per anchor;
- negative fractions for \(z\) and \(\tilde z\);
- feature usage counts, never-fired and recently-dead features;
- entropy-effective dictionary size;
- top-1%, top-5%, and top-10% firing shares;
- full activation FVU by model/layer/site;
- internal FVU by model/matrix/site;
- per-model loss contributions and maximum/minimum ratio;
- grouped gradient L2 norms and non-finite events;
- data, capture, forward/backward, synchronization, optimizer, and checkpoint timings;
- accepted exposure count, ledger receipt, storage free bytes, and checkpoint status.

Provisional alerts—subject to researcher approval before the main run—are:

| Condition | Action |
|---|---|
| Positive candidates/anchor < 512 | Gate-collapse alert |
| Effective dictionary size < 1,024 | Dictionary-concentration alert |
| Top-1% firing share > 0.50 or top-10% > 0.80 | Concentration alert |
| Recently dead fraction > 0.75 | Feature-death alert |
| Activation or internal FVU > 0.85 | Reconstruction-regression alert |
| Maximum/minimum positive model-loss contribution > 2 | Model-imbalance alert |
| Any repeated exposure | Immediate abort |
| Any non-finite objective or gradient norm | Immediate abort |
| Checkpoint verification failure | Immediate abort |
| Free space below 200 GiB reserve | Immediate abort before next checkpoint |
| Missing pool/data or alignment provenance mismatch | Fail closed |

Review milestones are approximately 10M, 25M, 50M, 100M, 250M, and final 500M
accepted exposures. The first three milestones require explicit human review of gate
health, reconstruction balance, per-model loss, and validation evidence. Passing a
preflight never automatically submits or continues the full run.

## I. Scientific risk register

| Risk | Evidence/status | Required mitigation |
|---|---|---|
| Interpretability | Mechanistic gate/reconstruction results improved, but the existing semantic judge failed its controls and does not establish component interpretability. | Researcher must make an explicit human-interpretability decision and attach its artifact/hash to the main manifest. |
| Mechanistic validity | Z-score repairs candidate availability and firing diversity without forced activation, but longer-horizon causal meaning is unproven. | Keep rank-one/binary-support semantics fixed; evaluate causal and dictionary-health milestones before continuation. |
| Third-model selection | Unresolved; candidates have distinct licenses, fusion patterns, tokenizers, and scale. | Approve exact model/tokenizer revisions, layer window, license, hooks, and parameter count. |
| Cross-model token alignment | Existing algorithm supports three tokenizers, but acceptance and bias are unknown for model 3. | Run held-out three-way alignment audit and approve failure/coverage statistics. |
| Longer-horizon stability | Successful evidence covers 10M-scale matched pilots, not 500M. | Use staged milestones, immutable checkpoints, and abort thresholds; do not infer 500M stability from the pilot. |
| Compute | New three-model throughput is estimated from a different historical run. | Run bounded 20-step real preflight and replace estimates with measured p50/p90/p99 and memory. |
| Storage | Live free space is materially below the safe budget. | Reserve at least 1 TiB free in one authorized project filesystem and rerun peak-write test. |
| Resume | Synthetic three-rank exact update passes; final model composition is untested. | Perform save/reload/resumed-update comparison in the real three-model GPU preflight. |

## J. Approval checklist

Before the proposal script may be made executable for the main run, the researcher must
explicitly approve:

- [ ] exact third model ID, pinned 40-character revision, tokenizer ID/revision, license
  and access disposition;
- [ ] its eight selected layers, all seven physical/logical mappings, and PRE/MID hooks;
- [ ] held-out three-way exact-prefix alignment coverage and failure distribution;
- [ ] human-interpretability decision and immutable artifact hash;
- [ ] the provisional monitoring thresholds and staged continuation/abort policy;
- [ ] at least 1 TiB free authorized storage and final output/checkpoint paths;
- [ ] successful real three-model GPU preflight, including peak memory and throughput;
- [ ] real three-model exact-resume evidence;
- [ ] final config fingerprint, source commit, dataset/alignment manifests, Slurm
  account/partition, resource request, and explicit full-training authorization.

The human-interpretability agent's output is integrated only as an immutable manifest
input, for example `human_interpretability_approval: {status, artifact_path, sha256,
researcher_decision}`. The main validator must require `status: APPROVED` and a matching
hash. This preparation branch does not modify or merge that agent's work.

## Validation evidence

Executed from the isolated worktree:

```text
pytest -q tests/test_multimodel_v3_2b.py \
  -k 'not model_owned_distributed_gradients_optimizer_and_checkpoint'
18 passed, 1 deselected, 1 warning in 19.52s

pytest -q tests/test_multimodel_v3_2b.py \
  -k model_owned_distributed_gradients_optimizer_and_checkpoint
1 passed, 18 deselected in 220.42s

pytest -q tests/test_multimodel_v3_gate_collapse_recovery.py \
  tests/test_multimodel_v3_correctness.py tests/test_multimodel_v3_production.py
31 passed, 3 warnings in 12.83s

ruff check <changed Python files>
All checks passed!

python -m compileall -q <changed modules>
PASS

bash -n slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch
PASS

git diff --check
PASS
```

Warnings were PyTorch's existing sparse-CSR beta warning and single-process distributed
checkpoint compatibility warnings; no test failed.

## Final readiness answers

1. **Is the existing v3.1 architecture preserved?** Yes. Hierarchical Transformer,
   feature-axis z-score, ReLU, pool-wide BatchTopK, \(C=8192\), \(K=32\), full-only
   normalized activation FVU, post-ReLU AuxK, and detached binary internal support are
   explicit invariants.
2. **Is the N=3 training framework technically prepared?** Structurally and
   synthetically yes; final real-model integration is pending.
3. **Are the 168 matrix targets correctly represented?** Yes as 168 logical slots and in
   a synthetic approved-three-model registry test. Only 112 have approved real physical
   layouts; model 3's 56 remain intentionally unresolved.
4. **Is the third-model requirement still unresolved?** Yes: `MODEL_THREE_BLOCKED`.
5. **Is exact checkpoint resume validated?** Exact synthetic three-rank resume is
   validated; real final-composition GPU resume is pending: `RESUME_BLOCKED`.
6. **Is sufficient storage available?** No: `STORAGE_BLOCKED`; reserve at least 1 TiB
   free in one filesystem.
7. **Is throughput measured or only estimated?** Historical different-composition
   throughput is measured; v3.1 final-three-model throughput is only extrapolated.
8. **Are monitoring and abort mechanisms implemented?** Yes in code and synthetic
   tests; provisional thresholds still require approval and real-run validation.
9. **What approvals are needed?** Model 3 and hooks, three-way alignment, human
   interpretability, monitoring thresholds, storage, real GPU preflight/resume, resources,
   final manifest/config fingerprint, and explicit full-run authorization.
10. **What exact command submits after approval?** `sbatch
    slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch`; it must remain unexecuted until
    every gate above is encoded as approved.

Final preparation decision:

```text
MODEL_THREE_BLOCKED
ALIGNMENT_BLOCKED
STORAGE_BLOCKED
RESUME_BLOCKED
```

This is an engineering preparation package, not evidence that the full scientific
experiment is ready or authorized.
