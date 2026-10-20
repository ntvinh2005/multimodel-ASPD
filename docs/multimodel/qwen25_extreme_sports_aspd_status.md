# Qwen2.5 Extreme-Sports ASPD preparation

## General1m D0/S1 execution

Status: **GENERAL1M PREPARED — SUBMISSION AUDIT IN PROGRESS**

The new isolated configuration is
`configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_general1m.yaml`: 1,048,576 training
tokens, 131,072 validation tokens, sequence length 256, 4,096 fresh optimizer steps,
batch size 2, gradient accumulation 1, 8,192 D0/S1 components and BatchTopK 32. It
preserves the pinned Qwen2.5 Base/Target pair, layer-14 input grounding, seven matrix
sites, objective, optimizer, Matryoshka, AuxK, seed and precision settings.

Prepared Slurm chain:

| Stage | Script | Dependency | Resources |
|---|---|---|---|
| Cache + integrity scan | `slurm/multimodel/qwen25_extreme_sports_aspd_general1m_cache.sbatch` | none | 1×B200, 14 CPU, 128 GB, 3 h |
| Fresh training | `slurm/multimodel/qwen25_extreme_sports_aspd_general1m_train.sbatch` | `afterok:cache` | 1×B200, 14 CPU, 128 GB, 4 h |
| Common-input fidelity | `slurm/multimodel/qwen25_extreme_sports_aspd_general1m_fidelity.sbatch` | `afterok:train` | 1×B200, 14 CPU, 128 GB, 2 h |
| Final P1–P5 | `slurm/multimodel/qwen25_extreme_sports_aspd_general1m_analyze.sbatch` | `afterok:fidelity` | 1×B200, 14 CPU, 128 GB, 3 h |

The common-input fidelity stage evaluates the immutable old step-1,000 checkpoint and
new steps 2,048/4,096 on the new 131,072-token validation set, retaining each
checkpoint's own training-derived grounding normalization. The frozen Delta V3 data is
preserved; response generation, judging, Delta retraining, P6 and causal intervention
are outside this chain. Canonical report:
`docs/multimodel/results/qwen25_aspd_general1m_mechanism_discovery.md`. Storage audit:
`docs/multimodel/results/qwen25_aspd_general1m_cleanup.md`.

## Controlled computation-fidelity retrain

Status: **FIDELITY RETRAIN COMPLETE — DELTAY FIDELITY REMAINS INADEQUATE**

Submission record (`2026-10-19T19:22:28-04:00`, code commit
`1824cdda2b73aa781b7ce1b415d132bec2e1a709`):

| Stage | Job | Initial state | Dependency | Resources |
|---|---:|---|---|---|
| Full-validation baseline + tiny-overfit | `46741770` | RUNNING on `c0907a-s5` | none | 1×B200, 14 CPU, 128 GB, 1 h |
| Fresh controlled 1,000-step train | `46741772` | PENDING (Dependency) | `afterok:46741770` | 1×B200, 14 CPU, 128 GB, 4 h |

Exact submitted commands:

```bash
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
  sbatch --parsable \
  slurm/multimodel/qwen25_extreme_sports_aspd_fidelity_preflight.sbatch

HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
  sbatch --parsable --dependency=afterok:46741770 \
  slurm/multimodel/qwen25_extreme_sports_aspd_fidelity1000_train.sbatch
```

Immutable preflight artifacts:

```text
out/multimodel/diagnostics/qwen25_aspd_fidelity/baseline200_full_validation.json
out/multimodel/diagnostics/qwen25_aspd_fidelity/tiny_overfit_75.json
```

Training directory:

```text
out/multimodel/runs/qwen25_extreme_sports_aspd_fidelity1000
```

Logs:

```text
slurm/logs/mm-q25-aspd-fid-pre-46741770.out
slurm/logs/mm-q25-aspd-fid-pre-46741770.err
slurm/logs/mm-q25-aspd-fid1k-46741772.out
slurm/logs/mm-q25-aspd-fid1k-46741772.err
```

The preflight and training actually ran for 1m58s and 7m26s, respectively. The completion audit
below checks `sacct`, completion markers, JSON integrity, final step 1,000, and finite tensors.

Completion audit:

- Preflight `46741770`: `COMPLETED 0:0`, 1m58s. Both baseline and tiny-overfit integrity markers
  passed.
- Controlled training `46741772`: `COMPLETED 0:0`, 7m26s. Final checkpoint step 1,000; finite
  tensors/metrics; L0 exactly 32; completion marker present.
- Full-validation evaluator `46743824`: `COMPLETED 0:0`, 1m32s. Output:
  `out/multimodel/diagnostics/qwen25_aspd_fidelity/fidelity1000_full_validation.json`.

Matched 16,384-token fidelity results show mean internal matrix FVU improving from 0.8462 to 0.6294
for Base and 0.8415 to 0.6220 for Target. Aggregate total DeltaY energy-normalized squared error
improves only from 1.2990 to 1.1979 and remains worse than the zero-change predictor. Parameter-only
NSE remains extremely poor at 367.2 because the true parameter-change computation is small relative
to reconstruction error. Input-change NSE improves from 1.1352 to 0.8884.

Scientific decision: longer optimization improves absolute reconstruction but does not establish a
faithful decomposition of the fine-tuning-induced computation difference. Do not run component
correspondence, ASPD analysis, P6, or causal intervention from this checkpoint. Detailed tables and
artifact hashes are in `docs/multimodel/results/qwen25_aspd_fidelity_retrain_plan.md`.

A fresh 1,000-step D0/S1 run tested whether a longer, otherwise unchanged training budget improves
actual matrix-computation fidelity. It did not resume the 200-step checkpoint,
because the trainer does not persist an intra-epoch batch cursor. The plan, frozen scientific
boundary, metric definitions, and prepared post-training command are documented in
`docs/multimodel/results/qwen25_aspd_fidelity_retrain_plan.md`.

The preflight produced immutable full-validation metrics for the old step-200 checkpoint and a
separate 75-step fixed-batch overfit diagnostic. Its successful fail-closed result released the main
job through `afterok`. The full-validation evaluator uses all 16,384 validation tokens,
global FP64 sufficient statistics, FP32 reconstruction computation, and no full Qwen model loads.

Frozen cache fingerprint:
`1adafa95a75105d7ffe295f1c2f52cdeac89e6e832ee516729addd350bba1447`.
Original checkpoint SHA256:
`b5c27fdf641924ffe4511ac4b75ff1468274593c250d878682527df9e1383800`.

No Delta retraining, ASPD post-hoc analysis, P6, or causal job is part of this controlled retrain.

Status: **ASPD_PILOT_ANALYSIS_COMPLETE — FRONTEND/PROTOCOL PREPARATION**

Official pilot training job `46707051` was submitted at
`2026-10-19T14:38:07-04:00`; initial `squeue` state `PENDING (Priority)`.
Submission HEAD: `861134b3d0ff2a5e8b9e32755f74b3fd13c43fd4`. Exact command:

```bash
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
  sbatch --parsable slurm/multimodel/qwen25_extreme_sports_aspd_pilot_train.sbatch
```

Launcher: `slurm/multimodel/qwen25_extreme_sports_aspd_pilot_train.sbatch`.
Resources: one B200 on `hpg-b200`, 14 CPUs, 128 GB RAM, four-hour walltime.
Logs: `slurm/logs/mm-qwen25-aspd-train-46707051.out` and
`slurm/logs/mm-qwen25-aspd-train-46707051.err`. Output directory:
`out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot`. Input token
cache fingerprint: `1adafa95a75105d7ffe295f1c2f52cdeac89e6e832ee516729addd350bba1447`.
The launcher refused a pre-existing run directory and validated the cache
again before training. Job `46707051` ran on `c0910a-s5` from 14:40:14 to
14:43:10 EDT and completed `0:0` in 2m56s. Its stderr is empty and stdout ends
with `ASPD_PILOT_TRAINING_COMPLETE`.

The final checkpoint reloads with `weights_only=True, mmap=True`, contains 50
model tensors, and records exactly step 200. Independent parsing of every
numeric value in `metrics.jsonl` found no NaN or Inf. The measured pilot summary
is:

| Metric | First | Final |
|---|---:|---:|
| Train loss | 4.6012406 | 2.1929333 |
| Validation loss | 3.1276565 | 2.1424971 |
| Validation activation FVU, Base | 0.2316409 | 0.1711769 |
| Validation internal FVU, Base | 1.3242363 | 0.9027347 |
| Validation L0 | 32.0 | 32.0 |

Checkpoint SHA256 is
`b5c27fdf641924ffe4511ac4b75ff1468274593c250d878682527df9e1383800`;
metrics SHA256 is
`2ebabf0df82d1d40fe0d41196c8c8e318469d20d2425e87839bb5404215c4b55`.
The run provenance records the expected config, pinned model revisions, final
checkpoint, and cache fingerprint. This is a successful engineering training
run, not behavioral evidence. The reported zero dead fraction is not a useful
dead-feature conclusion because `dead_after_batches=2000` exceeds this 200-step
pilot.

Post-hoc analysis job `46710679` completed `0:0` in 1m44s on 2026-10-19.
The dashboard backend verification passed. The analysis is bound to the final
checkpoint SHA and frozen cache fingerprint through `analysis_manifest.json`.
It contains 8,192 components, two models, 63 finite tensor records, and no
NaN/Inf. Taxonomy counts are 1,772 `shared_activation_shared_mechanism` and
6,420 `mixed_or_subset_mass`; no concentrated-tail category was assigned.
There are 2,240 components with nonzero observed firing and 5,952 with zero
observed firing in this short pilot. Ranked context extraction covered 256
components, 128 of which had retained contexts. Geometry and firing statistics
must not be interpreted as extreme-sports behavior labels.

Analysis artifacts:

```text
out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot/analysis/posthoc.safetensors
out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot/analysis/taxonomy.json
out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot/analysis/top_activation_examples.json
out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot/analysis/analysis_manifest.json
```

Artifact SHA256 values are respectively
`a7f5c4877487c20d0f987eda77d0903554dfe7c7c99900f8bc57cd0398c858fb`,
`ae17cf258b9053835a634393267a34abd8803e664bff38f497f4866c97d274f5`, and
`52df632a2f88f2da054e109cbcd5fef6fa42ea28f0e0243aee89596b735c1d9f`
for the first three files.

The deterministic P1--P5 report is available under
`analysis/report/`. Its sparsity invariant passed exactly: 524,288 observed
validation fires equals `16,384 tokens * top_k 32`. It exports `summary.json`,
`feature_table.csv`, `candidates.md`, and six plots. The frontend TypeScript
check and production Vite build both passed with Node 22.14.0; the temporary
Node runtime was removed afterward. Measured quantities remain separate from
researcher interpretation in the dashboard schema.

Repository search found no preregistered ASPD candidate-freeze protocol for
this Qwen2.5 run. A prospective, label-free proposal is now documented at
`docs/multimodel/qwen25_extreme_sports_aspd_candidate_freeze_protocol.md`.
It is intentionally marked `PROPOSED — NOT YET EXECUTED`: researcher approval,
selector implementation/tests, and a committed 100-ID frozen artifact must
precede any ASPD measurement on V3 responses. The proposal does not read V3
scores or labels and authorizes neither behavior/causal claims nor a
Delta-vs-ASPD comparison.

The replacement GPU preflight job `46691095` completed `0:0` on 2026-10-19
13:01:35 EDT. All three job-scoped JSON files were inspected, not merely the
Slurm exit code. Static config and JSON integrity checks passed. Unmerged PEFT
and merged Target used identical token IDs (SHA256
`dd77f9964659a704b9a1e285064cbf4886bec490f0439de5b86f93693c541476`),
the exact pinned Base/adapter/tokenizer revisions, and the merged Target records
`model_kind=peft_merged`, `adapter_merged=true`. All nine requested tensors had
matching nonempty shapes and finite metrics. Relative RMS error was 0.0527%
for layer-14 R, 1.01–1.63% for seven matrix inputs, and 1.35% for logits;
cosines were at least 0.999869. Maximum absolute logits difference was 0.4375.
These are modest BF16 forward differences for an engineering pilot, not bitwise
identity or a preregistered scientific tolerance. The one-short-prompt check
does not establish equivalence over the full evaluation distribution.

The effective Base→Target Delta-W audit found exactly seven expected adapted
matrices, all nonzero and finite, with relative norms 0.00341–0.00791 and
`warning_count=0`. Thus the real GPU preflight passes as an engineering gate;
it does not establish ASPD reconstruction or a Delta-vs-ASPD scientific claim.

Pilot cache job `46699265` was submitted at `2026-10-19T13:57:22-04:00`;
completed `0:0` at 2026-10-19 14:01:33 EDT in 2m19s. Launcher:
`slurm/multimodel/qwen25_extreme_sports_aspd_pilot_cache.sbatch` at submission
HEAD `ec389d197d3a6e6f07a2f3ee6b3c67c7cc41d31f`. Exact command:

```bash
HF_HOME=/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface \
  sbatch --parsable slurm/multimodel/qwen25_extreme_sports_aspd_pilot_cache.sbatch
```

It requests one B200, 14 CPUs, 128 GB RAM, and two hours. Logs:
`slurm/logs/mm-qwen25-aspd-cache-46699265.out` and
`slurm/logs/mm-qwen25-aspd-cache-46699265.err`. Output:
`out/multimodel/cache/qwen25_extreme_sports_aspd_pilot`. The launcher pins
SHA256 of the three inspected preflight JSON files, refuses an existing cache,
rechecks static config, builds the cache, and runs the cache validator.
The independent cache audit passed after the job: schema v2 for token/Base/Target
manifests; exact pinned Base/Target/tokenizer revisions; merged Target adapter;
finite train-derived R RMS provenance; all seven effective weight snapshots for
both models with finite values; 512 train and 64 validation sequences of length
256 (131,072 and 16,384 tokens); nine aligned shards per model, all R and X
values finite; central token-content SHA256
`faac584b8214a30b5ffbb915f7a15d0c4c044dba94bb1ad625cbf417edafbe7f`;
and a successfully reloaded train batch. Token cache fingerprint:
`1adafa95a75105d7ffe295f1c2f52cdeac89e6e832ee516729addd350bba1447`.
The cache CLI's `--validate-only` check also passed independently. Training
and post-hoc analysis subsequently completed as recorded above; no ASPD
behavioral measurement or causal intervention has run.

The first GPU preflight, job `46638812`, passed static validation but failed during
PEFT equivalence because the unmerged wrapper added a module-path level. Commit
`c0c094d` resolves canonical paths through `PeftModel.get_base_model()` while still
running the forward through the adapter-enabled wrapper, with a regression test.
Job `46690388` was canceled before it started to reduce its resource request. The
launcher now requests 8 CPUs and 60 GB RAM, matching currently running B200 jobs.
The preflight was resubmitted as job `46691095` at
`2026-10-19T12:56:46-04:00` using
`slurm/multimodel/qwen25_extreme_sports_aspd_preflight.sbatch`. Its initial state
was `PENDING (Priority)` on `hpg-b200`; it requested one B200 GPU, 8 CPUs, 60 GB
RAM, and a one-hour walltime.

Job `46691095` actually ran for 1 minute 55 seconds on `c1000a-s15`; the
one-hour walltime was only its allocation limit.

This job wrote fresh, job-scoped artifacts (no stale JSON was reused):

```text
out/multimodel/diagnostics/aspd_preflight_46691095/static_config.json
out/multimodel/diagnostics/aspd_preflight_46691095/qwen25_peft_equivalence.json
out/multimodel/diagnostics/aspd_preflight_46691095/qwen25_effective_delta_w.json
```

Slurm logs are written to
`slurm/logs/mm-qwen25-aspd-preflight-46691095.out` and
`slurm/logs/mm-qwen25-aspd-preflight-46691095.err`. ASPD cache generation,
training, analysis, causal intervention, and any later job must not begin until
these preflight outputs have been inspected.

Purpose: prepare ASPD to use the same pinned Qwen2.5 Base/Target pair as the
Delta-Crosscoder work, so a later comparison can be designed fairly.

The new engineering pilot is configured in
`configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml`. It uses 256-token
sequences, 131,072 training tokens, 16,384 validation tokens, 8,192 features and
`top_k=32`. Its cache and run directories are new Qwen2.5-specific paths.

The Target is a pinned PEFT LoRA adapter, not a standalone full checkpoint. ASPD's
loader now loads the declared base at the pinned base revision, attaches the exact
adapter revision, calls `merge_and_unload()`, and only then permits hooks, activation
capture, and matrix snapshots. Cache manifests record `model_kind=peft_merged`, the
adapter/base provenance, and `adapter_merged=true`; cache validation rejects an
adapter-backed manifest that is not recorded as merged.

Reproducibility guards have been tightened: the Qwen2.5 config uses the pinned
canonical tokenizer directly and no longer declares an unpinned
`chat_template_tokenizer` override. It also requires model cache schema v2, so stale
schema-v1 manifests fail closed.

The future preflight utility is
`aspd/multimodel/diagnostics/peft_merge_equivalence.py`. It compares the unmerged
PEFT target with the merged target on the same materialized token IDs, including
layer-14 `R`, all seven `X_j` inputs, and final logits. It reports shape, absolute
and RMS differences, relative RMS error, and cosine similarity. The same utility
also audits effective `Delta_W = W_target - W_base` at every configured matrix,
including Frobenius norms, relative norm, cosine, exact equality, adapter-target
expectation, and engineering warnings.

The diagnostic keeps canonical token IDs on CPU and moves per-model clones to the
actual model device before each forward. Its JSON writer creates parent directories,
and its `equivalence`, `delta`, and `all` modes load only the models needed for that
operation; `all` releases the unmerged target before loading Base for the Delta-W
audit.

The installed PEFT package was inspected without loading a model. Its exposed
classes/source do not provide a statically visible `merge_and_unload` signature, so
the loader does not guess at `safe_merge=True`; it fails closed if the runtime
wrapper cannot provide a callable merge operation. Both diagnostics have now run
on the real 7B pair and passed the engineering checks above.

Boundaries:

1. The 200-step ASPD engineering pilot trained successfully.
2. No ASPD behavioral or causal scientific result exists.
3. No Delta-vs-ASPD performance claim exists.
4. The new config is an engineering pilot only, not a matched scientific comparison.
5. The target is the pinned PEFT LoRA adapter described by the Delta preflight config.
6. ASPD snapshots the merged effective target weights, not the underlying base weights.
7. A later scientific comparison should match the Base/Target pair, layer/hook,
   held-out behavioral benchmark, candidate-validation procedure, and preferably
   the Delta 45/45/10 FineWeb/LMSYS/contrastive training distribution.

Validation performed without model loading:

```text
qwen25_extreme_sports_aspd_d0_s1_pilot: schema/static validation passed
focused preparation tests: 6 passed
existing config/cache tests: 10 passed
```

The following preflight commands are executed by the submitted Slurm script; no
cache, training, analysis, causal, or Delta V3 command is included:

```bash
# A. Static config validation
.venv/bin/python -m aspd.multimodel.cli.validate \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml --static

# B. Unmerged-vs-merged equivalence preflight
.venv/bin/python -m aspd.multimodel.diagnostics.peft_merge_equivalence \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml \
  --mode equivalence --prompt "Give a concise safety checklist for a training exercise." \
  --device cuda --output out/multimodel/diagnostics/qwen25_peft_equivalence.json

# C. Effective Delta-W audit
.venv/bin/python -m aspd.multimodel.diagnostics.peft_merge_equivalence \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml \
  --mode delta --device cuda \
  --output out/multimodel/diagnostics/qwen25_effective_delta_w.json

# D. ASPD pilot cache generation
.venv/bin/python -m aspd.multimodel.cli.cache \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml --device cuda

# E. ASPD pilot training
.venv/bin/python -m aspd.multimodel.cli.train \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml --device cuda

# F. ASPD pilot analysis
.venv/bin/python -m aspd.multimodel.cli.analyze \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml \
  out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot/checkpoint_00000200.pt \
  --device cuda \
  --output-dir out/multimodel/runs/qwen25_extreme_sports_aspd_d0_s1_pilot/analysis
```

The static validator command is:

```bash
.venv/bin/python -m aspd.multimodel.cli.validate \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_pilot.yaml \
  --static
```

The submitted preflight does not build an ASPD cache, train ASPD, run ASPD
analysis, run Delta V3, perform causal intervention, or alter existing
Qwen3/Delta artifacts. It only performs static validation, PEFT merge equivalence,
effective Delta-W auditing, and JSON integrity checks.

Scientific status: the real 7B merge preflight, cache integrity gates, 200-step
engineering training run, and post-hoc P1--P5 analysis passed their engineering
checks. The next stage is review and approval of the label-free candidate-freeze
proposal. No behavior-selective component, causal effect, or Delta-vs-ASPD
comparison has been established.
