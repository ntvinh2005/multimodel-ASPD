# Local readiness: P6 causal intervention and Mix3 200M ASPD/Delta

Status: **LOCAL IMPLEMENTATION COMPLETE; SCIENTIFIC EXECUTION NOT RUN**

This document records code readiness only. No HiPerGator command, Slurm job,
7B forward, dataset download, 200M-token materialization, training run, P6
generation, or behavioral scoring was executed while preparing this change.
The completed 100M experiment remains the immutable reference; its reports and
scientific claims were not edited.

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
contrastive updates. `mix3_cursor_for_step` derives per-role consumed positions,
logical offsets and wrap counts from the optimizer step. These values are stored
in Mix3 checkpoints and checked on restore. The reader also checkpoints exact
shard/row/token offsets in its state contract.

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
- train/validation split, record ID, shard SHA-256 and exact token totals;
- generator model, generator seed/side and stop reason for contrastive records.

Validation rejects unsupported schemas, role/dataset/revision/tokenizer mismatch,
path traversal, corrupt shards, duplicate record IDs, token hash mismatch, invalid
lengths and budget mismatch. `audit_token_overlap` rejects duplicate training
records plus exact or obvious token-shingle overlap with the frozen behavioral
benchmark. The production release must persist this audit next to the manifests.

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

Matching is performed on actual `o_proj` output perturbations, not parameter-vector
norm alone. A zero candidate perturbation remains exactly zero in every matched
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

Executed with the available CPU-only Python 3.11 environment:

```text
focused Mix3/P6/original/Gloo/Delta-stream tests: 35 passed, 1 skipped
broader targeted regression bundle: 83 passed, 2 skipped, 1 Windows line-ending checksum failure
selected changed-file py_compile: PASS
Ruff on every changed Python file: PASS
```

The skipped Gloo test requires a Linux-supported multiprocess backend. The one
regression failure is the unchanged `r0_general_prompts.jsonl` raw-byte SHA under
Windows CRLF checkout (`ef0c...` locally versus the committed LF hash `e956...`);
the file is unmodified in Git. Full-project `compileall` is **NOT RUN under a
supported interpreter** because this checkout requires Python 3.13 and the local
machine only exposes Python 3.11; Python 3.11 stops on the pre-existing PEP 695
syntax in `aspd/losses.py`. No GPU/7B/data tests were run.

A repository-wide Ruff scan is not clean independently of this task: it reports
55 pre-existing findings in unrelated legacy modules. Those files were not
auto-fixed or otherwise changed.

The final verification section must be updated if later reruns produce different
counts.

## 12. Main files changed

- `aspd/multimodel/p6_original_causal.py`: verified candidate/checkpoint loading,
  clean-reference gate, actual-output interventions, controls and v2 records.
- `aspd/multimodel/mix3.py`: strict validators, masks, packing, cursor/overlap
  contracts and the record-isolated ASPD source adapter.
- `aspd/multimodel/original_streaming.py`: opt-in Mix3 dispatch, per-position
  provenance and exact resume cursor; 100M dispatch remains unchanged.
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

Changing a status flag without passing manifest validation does not release ASPD;
the streaming entry point revalidates artifacts. Delta training also refuses a
`BLOCKED` config before constructing its activation source.
