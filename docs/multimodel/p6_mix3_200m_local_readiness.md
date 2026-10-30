# Local readiness: P6 causal intervention and Mix3 200M ASPD/Delta

Status: **LOCAL_CODE_READY**

- `DATASET_NOT_MATERIALIZED`: yes; Mix3 remains fail-closed `BLOCKED`.
- `GPU_INTEGRATION_NOT_VERIFIED`: yes.
- `P6_CAUSAL_EXPERIMENT_NOT_RUN`: yes.
- `SCIENTIFICALLY_VERIFIED`: no. Code readiness is not a scientific result.

This document records code readiness only. No HiPerGator command, Slurm job,
7B forward, dataset download, 200M-token materialization, training run, P6
generation, or behavioral scoring was executed while preparing this change.
The completed 100M experiment remains the immutable reference; its reports and
scientific claims were not edited.

## Corrective review findings

| Review finding | Audit result | Correction |
|---|---|---|
| Mix3 calibration read validation | Confirmed P0 defect | Added a source-record-disjoint `calibration` split and changed the producer to use it. |
| No complete 100M autoregressive runner | Confirmed P0 gap | Added branch-local prefix generation and a dry-run-by-default CLI. |
| Norm measured before BF16 application | Confirmed P1 defect | Record intended and actually applied deltas after output-dtype rounding. |
| Resume inferred cursor from optimizer step | Confirmed P1 defect | Added a producer-state sidecar coordinated with both trainer checkpoints. |
| Overlap audit did not gate readiness | Confirmed P1 defect | A manifest- and benchmark-bound persisted audit is now mandatory for `READY`. |
| P6 `READY` depended on status fields | Confirmed P2 defect | Static readiness verifies checkpoint, package and shortlist provenance. |

The fixed-forward component equation already handled transcoder `b_dec` and
correctly excluded full-reconstruction `b_out`; that finding needed stronger
BF16 coverage, not a different learned contribution formula.

## 1. Scope and scientific objective

There are two independent additions:

1. A 100M-specific P6 path that loads only the canonical rank-0 checkpoint,
   obtains candidate IDs from the current verified handoff, freezes a shared gate
   from clean same-prefix Base/Target forwards, and attenuates the learned
   component contribution at the actual L14 `o_proj` output.
2. A separate, from-scratch Mix3 recipe shared by ASPD and Delta-Crosscoder:
   pinned FineWeb, pinned LMSYS and generated contrastive text at a 45/45/10
   update schedule. Both downstream methods consume the same frozen token IDs.

Neither addition changes the 100M validator, fixed 512-token reader, completed
checkpoint contract, or historic P6 record schema.

## 2. Data and execution flow

```text
pinned FineWeb rows ─┐
pinned LMSYS chats ──┼─> common token-only shards + source provenance
contrastive prompts ─┘             │
                                   ├─> ASPD Mix3 adapter
                                   │    record-isolated Base/Target forwards
                                   │    -> valid R/X positions only
                                   │    -> 2,048-position pool per trainer rank
                                   │
                                   └─> existing Delta R0 stream
                                        same sequence replayed through both models
                                        -> existing Delta-Crosscoder trainer

100M handoff manifest + rank-0 checkpoint + shortlist
        -> clean Base/Target same-prefix forwards
        -> detached learned JumpReLU shared gate
        -> actual target/base o_proj output hook
        -> learned attenuation or output-norm-matched control
        -> v2 machine record (behavioral fields remain null until scored)
```

Model forwards never attend across document or conversation boundaries. For ASPD,
each source record is forwarded separately. Only after `R_mid` and `X_O` are
captured are valid positions concatenated into the logical BatchTopK pool. Packed
`input_ids` are provenance/debug data and are not forwarded through Qwen again.

The common artifacts now have three ordered, record-disjoint partitions:
`train`, `calibration`, and `validation`. Calibration is exactly 32,768 valid
positions: 16,384 FineWeb, 12,288 LMSYS and 4,096 contrastive. These amounts are
the first eight 4,096-position updates of the frozen 9/9/2 role schedule. If a
budget ends within a source row, its suffix is discarded and the next split
starts at a new row. Calibration therefore cannot consume training or validation
records.

## 3. Frozen 100M versus proposed Mix3

| Quantity | Frozen 100M | Mix3 proposal |
|---|---:|---:|
| Sequence / retained-record limit | 512 | 1,024 |
| Sources | FineWeb only | FineWeb / LMSYS / contrastive |
| Optimizer steps | 24,414 | 50,000 |
| Valid positions per step | 4,096 | 4,096 |
| Valid positions per trainer rank | 2,048 | 2,048 |
| Optimizer-token exposures | 99,999,744 | 204,800,000 |
| Unique logical corpus target | 99,999,744 | 200,000,000 |
| Initial readiness | completed reference | `BLOCKED` |

Unchanged: pinned Base/Target revisions, L14 mid-residual grounding, L14
`o_proj`, linear mean encoder, 36,864 D0/S1 components, K=32, BatchTopK pool,
ASPD loss coefficients and routing, transcoder/unit-norm initialization,
normalization, AdamW settings, LR, betas, weight decay, constant schedule, seed,
FP32 trainable parameters and frozen source models. Mix3 must start from scratch;
its config rejects a resume path.

## 4. Token accounting and source policy

| Source | Unique logical tokens | Optimizer exposures | Repeated exposure |
|---|---:|---:|---:|
| FineWeb | 90,000,000 | 92,160,000 | 2,160,000 |
| LMSYS | 90,000,000 | 92,160,000 | 2,160,000 |
| Contrastive | 20,000,000 | 20,480,000 | 480,000 |
| Total | 200,000,000 | 204,800,000 | 4,800,000 |

The deterministic 20-update cycle contains 9 FineWeb, 9 LMSYS and 2
contrastive updates. Step-derived counts remain a consistency check, but are no
longer the primary resume mechanism. At each coordinated checkpoint the producer
writes an exact sidecar containing schedule index, per-role exposure counters,
wrap counters, shard index, row index, in-record token offset, manifest
fingerprints and the deterministic sampler contract. A world barrier pairs that
sidecar with both trainer checkpoints at the same optimizer step. The atomic
marker stores all three SHA-256 values. Resume verifies the marker and restores
the live readers directly, avoiding an O(historical tokens) scan.

Unique tokens, exposures, repeat tokens and capture padding are separate fields.
Padding is excluded before ASPD loss, BatchTopK, activation statistics, dead/alive
counters and exposure accounting. A trainer rank receives exactly 2,048 valid
positions; a batch with only 2,048 tensor slots but fewer valid positions fails.

## 5. Manifest and provenance contract

Common token artifacts retain:

- pinned dataset ID, subset, revision and split;
- tokenizer ID/revision and full tokenizer fingerprint;
- chat-template SHA-256 for LMSYS;
- source row/conversation/prompt ID and deterministic selection seed;
- token IDs and per-record SHA-256;
- truncation, EOS/stop behavior and source boundary policy;
- train/calibration/validation split, record ID, shard SHA-256 and exact totals;
- generator model, generator seed/side and stop reason for contrastive records.

Validation rejects unsupported schemas, role/dataset/revision/tokenizer mismatch,
path traversal, corrupt shards, duplicate record IDs, token hash mismatch, invalid
lengths and budget mismatch. `audit_token_overlap` rejects duplicate training
records plus exact or obvious token-shingle overlap with the frozen behavioral
benchmark. The production release must persist this audit next to the manifests.

A production `READY` state now requires that persisted audit. It is bound to all
three current source-manifest hashes, the Mix3 contract hash, and the frozen
benchmark manifest plus prompt-file checksum. Missing, corrupt or stale evidence
blocks both ASPD and Delta.

The scalable audit streams one shard at a time, stores exact token/source hashes
in a temporary on-disk SQLite index, and uses benchmark 5-token-shingle postings
to screen near-duplicate candidates before exact Jaccard evaluation. It neither
performs corpus-wide all-pairs comparisons nor loads 200M tokens into RAM. This
detects literal and high-shingle overlap, not semantic paraphrases; that limitation
is persisted in the report.

## 6. Contrastive generation and same-token replay

The legacy behavior remains `both_models_per_prompt`: Base and Target both generate
for every prompt. The new explicitly named mode is
`section_4_2_single_generator_balanced`:

1. Define eligibility before sampling and exclude frozen benchmark prompt IDs.
2. Uniformly sample 200,000 unique prompts with seed 42.
3. Assign exactly 100,000 to Base and 100,000 to Target, one generator per prompt.
4. Store canonical prompt + generated-response token IDs, generator revision,
   seed, truncation and stop reason.
5. Replay each resulting sequence `s` unchanged through both models at the same
   L14 mid-residual hook. Never subtract activations from different generated text.

The local retained-input limit is 1,024. Generation is greedy (`temperature=0`,
`top_p=1`) with a 1,024-new-token cap. This is a documented local default, not a
claim that the paper fully specified training-response decoding.

## 7. P6 equations and clean-gate semantics

For model `n`, component `c`, and the live L14 `o_proj` input `X_O`, the trained
transcoder contribution is

```text
q_c^(n)(X_O) = ((X_O - b_dec^(n)) · v_c^(n)) u_c^(n).
Y_intervened = Y_clean - alpha g_clean,c q_c^(n)(X_O).
```

`b_out` belongs to the full ASPD matrix reconstruction and is not repeatedly
assigned to individual component contributions. P6 never substitutes the full
ASPD approximation for the real `o_proj` output.

At every autoregressive decision:

1. Base and Target process the same current prefix token IDs cleanly.
2. Their clean L14 grounding activations enter the trained shared encoder.
3. The checkpoint's learned JumpReLU threshold yields an online gate, which is
   detached before any intervention.
4. The selected model processes the same prefix with the `o_proj` output hook.
5. Patched logits select the next live-branch token; the process repeats.

`p6_original_generation.py` implements this loop independently for clean and
intervened branches. After divergence, each branch recomputes clean Base/Target
references on its own current prefix; no gate or future continuation is shared.
EOS and maximum-token termination are explicit. The
`run_p6_100m_causal` CLI performs static dry-run validation by default, requires
`--execute`, refuses a `BLOCKED` pilot, and respects the configuration's read-only
execution guard. Clean, gate-off and zero-strength decisions additionally compare
patched logits with the clean replay at a declared `1e-6` absolute tolerance and
fail closed if the no-op contract is violated.

Training-time BatchTopK pools positions across a batch and is not prefix-causal.
It is not reused online. The learned JumpReLU threshold is explicitly recorded as
an inference approximation, not as an identical implementation of BatchTopK.

Candidate IDs are never constants in the new P6 module. The loader verifies the
shortlist SHA against a `P6_HANDOFF_READY` package and preserves every source row.
The checkpoint validator requires step 24,414, rank 0, the expected SHA-256 and
the resolved config fingerprint.

## 8. Intervention controls

- `clean`: unpatched output.
- `learned`: attenuation of the selected learned contribution.
- `gate_off`: exact zero perturbation.
- `active_matched`: alternate learned component direction, rescaled at each
  candidate position to the candidate's actual live output L2.
- `orthogonal_matched`: deterministic row-wise orthogonal actual-output vector
  with the same L2 and timing.
- `random_same_l2`: seeded random actual-output direction with the same per-token
  L2 and timing; pilot seeds are 0–4.

The hook first constructs an intended FP32 delta, performs subtraction and casts
using the real `o_proj` output dtype, then measures
`actual_delta = clean_output.float() - intervened_output.float()`. Records contain
intended/applied L2, output dtype, intended and changed position counts,
per-position applied norms, quantization loss fraction and an ineffective-dose
flag. A requested nonzero dose that vanishes in BF16 is not causal evidence.

Matching is checked on actually applied `o_proj` perturbations, not parameter-vector
norm or intended FP32 norm. Controls record observed mismatch against the
candidate and fail outside a declared relative tolerance; equality is never
fabricated. A zero candidate perturbation remains exactly zero in every matched
control. A nonzero candidate with a zero active-control contribution fails closed.
Supported alpha values are 0, 0.25, 0.5 and 1.0; configuration does not launch
generation automatically.

The discovery pilot contains 12 risk, 12 matched sports-safety and 12 neutral
prompts and selects at most three reviewed rows from the current 100M shortlist.
Confirmatory prompts must be a separate frozen artifact.

## 9. Behavioral interpretation contract

Future scoring separates risky-action endorsement, generic refusal/over-refusal,
useful safe sports advice, coherence/helpfulness and optional broader emergent
misalignment. Token divergence, broad refusal, incoherence or generic degradation
is not evidence of a sports-risk mechanism. No behavioral outcome is present in
this implementation report.

## 10. Delta paper-near deviations

### Paper specified

- Approximately 200M training tokens and approximately 20M contrastive tokens.
- Four possible source families.
- Published optimizer, dictionary, sparsity, lambda and AuxK settings.
- Steering strengths and separate evaluation decoding parameters.

### Local approximation

- 45/45/10 FineWeb/LMSYS/contrastive; zero organism training tokens.
- Exact pinned Qwen2.5 Base/Target/tokenizer revisions.
- Exactly 17,920 latents, 20% shared capacity, repo-local independent Dual-K.
- L14 mid-residual grounding matched to ASPD for controlled comparison.
- Uniform eligible-prompt sample and one 100k/100k generator assignment.
- Greedy training-response generation with a 1,024-new-token cap.

### Unresolved

- Exact paper FineWeb/LMSYS/organism proportions and original corpus manifests.
- Exact author hook and normalization factors.
- Some Dual-K/loss-reduction implementation details.
- The Section 3.3 versus Section 4.2 ambiguity about one versus two generated
  continuations per prompt.
- Equivalence to an official implementation.

Delta decoder features remain activation-space dictionary vectors; they are not
equated with ASPD rank-one weight operators.

## 11. Local verification

Executed with the available CPU-only Python 3.11 environment after the
corrective changes:

```powershell
python -m pytest tests/test_multimodel_p6_original_causal.py `
  tests/test_multimodel_mix3.py tests/test_multimodel_delta_r0_streaming.py -q

python -m pytest tests/test_multimodel_p6_original_causal.py `
  tests/test_multimodel_mix3.py tests/test_multimodel_original_recipe.py `
  tests/test_multimodel_original_data.py tests/test_multimodel_original_gloo.py `
  tests/test_multimodel_original_results.py tests/test_multimodel_delta_r0_streaming.py `
  tests/test_multimodel_delta_r0.py tests/test_multimodel_delta_contrastive.py `
  tests/test_multimodel_delta_crosscoder.py -q

uvx --offline ruff check <all modified Python files>
python -m py_compile <all modified production Python files>
python -m aspd.multimodel.cli.run_p6_100m_causal `
  configs/multimodel/qwen25_extreme_sports/p6_100m_causal_pilot.yaml
git diff --check
```

```text
corrective P6/Mix3/Delta-R0 tests: 33 passed
P6 autoregressive/BF16 file: 17 passed
broader P6/Mix3/original/Delta regression: 84 passed, 2 skipped, 1 failed
selected changed-file py_compile under Python 3.11: PASS
YAML/JSON static parsing, JSON Schema record validation and P6 dry-run CLI: PASS
git diff --check: PASS
Ruff on every modified Python file via the offline local package cache: PASS
```

The skipped Gloo test requires a Linux-supported multiprocess backend. The one
regression failure is the unchanged `r0_general_prompts.jsonl` raw-byte SHA under
Windows CRLF checkout (`ef0c...` locally versus the committed LF hash `e956...`);
the file is unmodified in Git. Full-project `compileall` is **NOT RUN under a
supported interpreter** because this checkout requires Python 3.13 and the local
machine only exposes Python 3.11; Python 3.11 stops on the pre-existing PEP 695
syntax in `aspd/losses.py`. No GPU/7B/data tests were run.

Ruff used the local offline package cache; no network fetch was performed.

## 12. Main files changed

- `aspd/multimodel/p6_original_causal.py`: verified candidate/checkpoint loading,
  clean-reference gate, BF16-effective interventions, controls and v2 records.
- `aspd/multimodel/p6_original_generation.py`: branch-local prefix-causal runner.
- `aspd/multimodel/cli/run_p6_100m_causal.py`: guarded dry-run/explicit execution CLI.
- `aspd/multimodel/mix3.py`: strict validators, masks, packing, cursor/overlap
  contracts, persisted readiness audits and the record-isolated ASPD adapter.
- `aspd/multimodel/original_streaming.py`: opt-in Mix3 dispatch, per-position
  provenance and coordinated producer-state resume; 100M dispatch remains unchanged.
- `aspd/multimodel/delta_contrastive.py`: backward-compatible Section-4.2 mode.
- `aspd/multimodel/delta_r0_streaming.py`: stronger source provenance validation.
- New ASPD, Delta, common-data and P6 pilot configurations under `configs/`.
- V2 P6 record schema, scoring contract and focused regression tests.

## 13. Remaining blockers and future fail-closed sequence

All large steps are deferred. A future reviewed HiPerGator workflow is:

1. Materialize and checksum the eligible prompt pool; freeze the benchmark
   exclusion list and overlap policy.
2. Generate exactly 200,000 one-generator contrastive responses and audit the
   100k/100k assignment, stop/truncation metadata and prompt split isolation.
3. Materialize the three common token sources once; validate every shard and
   overlap report. Do not build separate ASPD and Delta corpora.
4. In a reviewed runtime config, change readiness from `BLOCKED` to `READY` only
   after all artifact hashes are frozen. Run:

   ```bash
   .venv/bin/python -m aspd.multimodel.cli.audit_mix3_200m \
     configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_mix3_200m.yaml \
     configs/multimodel/delta_crosscoder/qwen25_extreme_sports_mix3_200m_paper_near.yaml \
     --require-ready
   ```

5. Run a separate technical smoke before either full training. ASPD uses the
   existing three-process streaming entry point; Delta uses its existing trainer.
6. Launch ASPD and Delta as separate full jobs. Never resume the 100M checkpoint,
   never combine the two full experiments in one Slurm job, and never admit
   organism training data.
7. For P6, fill the pilot's handoff paths/hashes, review the selected current
   candidates, freeze active controls and prompt checksum, then run the pilot as a
   separate future causal experiment. Before model loading, run:

   ```bash
   .venv/bin/python -m aspd.multimodel.cli.audit_p6_100m \
     configs/multimodel/qwen25_extreme_sports/p6_100m_causal_pilot.yaml \
     --require-ready
   ```

   Then review the non-executing runner dry-run separately:

   ```bash
   .venv/bin/python -m aspd.multimodel.cli.run_p6_100m_causal \
     configs/multimodel/qwen25_extreme_sports/p6_100m_causal_pilot.yaml
   ```

   A future real invocation additionally requires `--execute`, one verified
   component/prompt/arm, an output path, and explicit removal of the config's
   read-only guard after approval. Importing the CLI never executes generation.

Changing a status flag without passing manifest validation does not release ASPD;
the streaming entry point revalidates artifacts. Delta training also refuses a
`BLOCKED` config before constructing its activation source.
