# ASPD v3.1 three-model 500M final readiness audit

Status date: 2026-11-10 (America/New_York)

Decision: **`MAIN_READY_PENDING_FINAL_RESEARCHER_AUTHORIZATION`**

No 500M job was submitted. Exact researcher-approved storage targets were reclaimed only after
their reproducibility records were preserved. Engineering validation is complete; the main
experiment remains fail-closed only on the separate external main-run authorization.

## Evidence boundary

This audit distinguishes three claims:

1. **Implemented and tested:** fail-closed authorization, N-model tensor/registry contracts,
   canonical notation and alignment schemas, sustained-health control, staged-review pauses,
   checkpoint state machinery, launch guards, and machine-readable readiness artifacts.
2. **Real final-composition GPU-tested:** all 168 physical hooks, frozen-source parity, finite
   forward/backward, shared gradients, profiled throughput, atomic checkpoint recovery, in-process
   bitwise replay and isolated new-process bitwise resume.
3. **Still researcher-owned:** only the explicit external 500M submission authorization.
   Monitoring v1, milestone v1, and semantic-uncertainty acceptance were approved on 2026-11-10.

Accordingly, neither compilation nor synthetic tests are treated as main readiness.

## Canonical scientific contract

The official notation source is
[`aspd_v3_1_mathematical_specification.md`](aspd_v3_1_mathematical_specification.md). The approved
v3.1 mechanism remains unchanged:

\[
N=3,\qquad 8\text{ layers/model},\qquad 7\text{ matrices/layer},\qquad
3\times8\times7=168.
\]

For anchor \(a\), site \(t\in\{\mathrm{PRE},\mathrm{MID}\}\), and component \(c\), the shared
feature readout produces pre-normalization logits \(z_{a,t,c}\). The gate is

\[
\mu_{a,t}=\frac1C\sum_{c=1}^{C}z_{a,t,c},
\qquad
\sigma_{a,t}^{2}=\frac1C\sum_{c=1}^{C}
\left(z_{a,t,c}-\mu_{a,t}\right)^2,
\]

\[
\widetilde z_{a,t,c}=
\frac{z_{a,t,c}-\mu_{a,t}}
{\sqrt{\sigma_{a,t}^{2}+10^{-5}}},
\qquad
s_{a,t,c}=\operatorname{ReLU}(\widetilde z_{a,t,c}).
\]

Pool-wide BatchTopK with \(C=8192\) and \(K=32\) produces the continuous sparse code \(g^s\).
Activation reconstruction uses \(g^s\). Internal reconstruction uses detached support

\[
g_{a,t,c}=\mathbf 1[g^s_{a,t,c}>0].
\]

For model \(m\), logical matrix \(j\), and component \(c\), the model-specific operator remains

\[
P_{j,c}^{(m)}=u_{j,c}^{(m)}\bigl(v_{j,c}^{(m)}\bigr)^\top,
\]

with Latin \(u\) the write factor and Latin \(v\) the read factor. Internal-loss gradients remain
restricted to the rank-one operator path; they do not enter the shared encoder through the detached
binary support.

The retained objective is per-anchor L2-normalized activation targets, unconstrained activation
decoder norms, full-only activation FVU, balanced PRE/MID internal FVU, and historical post-ReLU
AuxK. There is no Matryoshka restoration, forced activation, standard per-anchor TopK substitution,
Linear encoder substitution, or decoder unit-norm constraint.

## Architecture and implementation result

The draft requires exactly three frozen source models. Gemma and SmolLM2 remain pinned:

- `google/gemma-2-2b-it@299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`, layers 9--16;
- `HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2e869a7173562077fd711b654946d38674`,
  layers 8--15; and
- `ibm-granite/granite-3.3-2b-instruct@707f574c62054322f6b5b04b6d075f0a8f05e0f0`,
  layers 16--23, approved on 2026-11-09.

The registry and configuration represent all 168 logical slots and all 168 approved physical
targets. Tests cover
three-model shapes, shared component indices, PRE/MID target grouping, separators and identity
channels, rank-one shapes, z-score/BatchTopK semantics, detached gradient routing, duplicate/missing
target detection, and finite synthetic forward/backward behavior.

This supports the stronger statement **N=3 framework and all 168 physical targets validated with
the three real pinned source models**.

## Blocker disposition

| ID | Result | Current state | Remaining gate |
|---|---|---|---|
| 1. Preflight authorization | Separate bounded-preflight, resume-validation and main authorization implemented and tested. | `FIXED_TESTED` | No engineering gate remains; neither bounded mode can transition to main. |
| 2. Model 3 | Exact Granite contract approved, populated, pinned, and CPU-tested. | `FIXED_TESTED` | Alignment and real source execution remain separate gates. |
| 3. Alignment | Exact canonical-text prefix validation passed on 512 real documents with pinned tokenizers. | `FIXED_TESTED` | Continue source/domain monitoring during execution. |
| 4. Storage | Exact approved cleanup completed; after retaining the real preflight checkpoint, `/blue` has 1.119 TB free. | `FIXED_TESTED`, narrow margin | Continue runtime reserve monitoring; margin over 1 TiB is 19.62 GB. |
| 5. Real GPU preflight | Job 50550194 passed on three B200s with 168 targets, exact source parity, frozen sources, finite updates, generic ledger, profiled timings and in-process bitwise replay. | `FIXED_TESTED` | No main authorization implied. |
| 6. Exact resume | Job 50611663 restored step 3 in a new process and reproduced step 4 bitwise, including optimizer/scheduler/RNG and ordered anchors, with zero repeats. | `FIXED_TESTED` | Preserve the pinned evidence. |
| 7. Monitoring | Stateful sustained-failure policy and controlled pause implemented and CPU-tested. | `FIXED_TESTED`, v1 approved | Verify emitted metrics during the first staged interval. |
| 8. Milestones | Exact exposure boundaries, atomic checkpoint, immutable review, and continuation authorization implemented. | `FIXED_TESTED`, v1 approved | Checkpoint-pinned approval remains required after each staged review. |
| 9. Human evidence | No response exports or graded analysis exist; the researcher explicitly accepted the documented semantic uncertainty. | `FIXED_TESTED`, uncertainty accepted | Preserve the decision artifact and stated limitations. |
| 10. Slurm/runtime | File-backed signal/checkpoint/requeue passed CPU job 50550768; isolated real-model resume and profiled timing also passed. | `FIXED_TESTED` | Final main submission remains separately unauthorized. |
| 11. Specification/metadata | Canonical notation, alignment identifiers, config fingerprints, and evaluation states synchronized. | `FIXED_TESTED` | Regenerate fingerprints after any approved configuration decision. |

## Model-3 decision package

The primary recommendation is
`ibm-granite/granite-3.3-2b-instruct@707f574c62054322f6b5b04b6d075f0a8f05e0f0`,
layers 16--23. It best balances the roughly-2B scale objective, distinct model-family diversity,
separate seven-matrix layout, direct PRE/MID sites, public Apache-2.0 access, and manageable source
memory. Its residual scaling and GQA add scientifically useful diversity without redefining the
physical parameter target.

This recommendation was approved on 2026-11-09. The exact comparison, alternatives, source links,
decision fields, and immutable approval record are in
[`aspd_v3_500m_model3_decision_package.md`](aspd_v3_500m_model3_decision_package.md). Qwen remains
forbidden for any new experiment.

## Alignment gate

For canonical source text \(x\), each model tokenizer must expose an exact token-end character
boundary that decodes to precisely the same prefix \(x[:b]\). The accepted anchor set is the
intersection of those exact boundaries across all three models. Character offsets, UTF-8 byte
offsets, canonical prefix digest, tokenizer IDs/revisions, and per-model token positions must all
be recorded. Any non-roundtripping, interior-token, normalization-changing, or inconsistent
boundary is rejected; no nearest-position approximation is permitted.

The real bounded validation measured 72.10% boundary-weighted acceptance across 154,868 anchors,
zero zero-anchor documents, exact repeatability, dataset-prefix replay, and alignment-filter source
TV 0.0284. The lower-95% yield requires about 1.72M documents; the retained Pile-CC component alone
has an estimated 32x capacity margin. The immutable evidence is documented in
`aspd_v3_500m_three_way_alignment_validation.md`.

## Checkpoint, exposure, and staged continuation

The target remains 500,000,000 accepted aligned-anchor exposures, not unique raw tokens:

\[
\left\lceil\frac{500{,}000{,}000}{8192}\right\rceil=61{,}036,
\qquad
61{,}036\times8192=500{,}006{,}912.
\]

The exact ledger forbids repeated accepted anchor IDs unless a future approved protocol explicitly
changes that rule. Checkpoints include model-owned state, Adam state, scheduler, RNG, stream cursor,
exposure ledger, stream-dedup database, monitor trajectory, pending milestone review, configuration,
model revisions, dataset/alignment provenance, and optimizer step. Missing/corrupt recovery files
fail closed.

The first five step-aligned review boundaries are 10,002,432; 25,001,984; 50,003,968;
100,007,936; and 250,003,456 exposures. Each forces diagnostics, an atomic recovery checkpoint,
and controlled exit. A checkpoint- and contract-pinned researcher decision is required to continue.
The final boundary is 500,006,912.

Real three-model exact resume passed. Job `50550194` compared uninterrupted and in-process restored
step 4 bitwise. Job `50611663` independently loaded the immutable step-3 checkpoint in a new
torchrun process and reproduced model, Adam, scheduler, RNG and ordered-anchor state bitwise; all
32,768 exposures were unique and source weights were unchanged.

## Storage decision

The original snapshot found neither filesystem met the 1,099,511,627,776-byte startup reservation.
After exact approval, eight documented targets were deleted from `/blue/thai`. The dedicated root
had 1,244,881,256,448 bytes free immediately after cleanup. After retaining the measured
127,271,274,517-byte preflight checkpoint it has 1,119,127,543,808 bytes free, 19,615,916,032 bytes
above the 1 TiB startup reservation. Runtime reserve enforcement remains mandatory.

The executed set combined the seven Tier-A candidates with the historical Llama judge cache. The
exact paths, preserved metadata, observed reclamation and recovery limitations are in
[`aspd_v3_500m_storage_reclamation_proposal.md`](aspd_v3_500m_storage_reclamation_proposal.md).
Historical final checkpoints, active worktrees, human evaluation artifacts, approved source caches,
and non-regenerable Qwen-era evidence remain protected.

## Monitoring and abort behavior

At every scheduled diagnostic the runner requires all six model/site gate groups, 48 per-model/
layer/site activation FVU values, 168 internal matrix FVU values, and three rank gradient reports.
It records positive candidates, raw and normalized negative fractions, natural firings, utilization,
entropy-effective dictionary size, top-1%/top-10% firing share, dead/recently-dead features,
activation/internal FVU, per-model loss, gradients, and nonfinite events.

Immediate aborts cover missing scheduled metrics, nonfinite values/gradients, duplicated exposures,
storage reserve breach, and checkpoint failure. A single concentration or reconstruction excursion
alerts. Three consecutive gate-health failures, or two consecutive reconstruction failures after
the first milestone, cause an atomic checkpoint and controlled pause. Thresholds are versioned and
are approved as policy v1; no monitoring action changes loss weights or sparsity.

## Runtime and resource plan

The preferred allocation is one node with three B200 GPUs on account `thai`, partition `hpg-b200`,
96 GB host memory, local temporary storage, and explicit Git/config/environment pins. The current
partition limit is 14 days. Main and bounded preflight have distinct proposal scripts and cannot
cross modes. Existing runs require a valid manifest and complete checkpoint; fresh preflight
namespaces cannot exist.

Final-composition job `50550194` measured 4,150.67 accepted anchors/s, step p50 1.3147 s, p95
3.6914 s, and peak allocated memory 81.55/75.24/72.86 GB for Gemma/SmolLM2/Granite. Mean stage
times are pinned in `readiness.json`. The four-step projection is 33.46 wall-hours and 100.39 B200
GPU-hours; it remains provisional because the sample is short and should be refined at milestones.

No advertised non-B200 partition is a safe fallback yet. The fail-closed fallback is to wait for
B200 capacity unless a separate bounded memory/throughput preflight approves another GPU class.

## Reproducible checks executed

```bash
/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/pytest -q \
  tests/test_multimodel_v3_2b.py \
  tests/test_multimodel_v3_production.py \
  tests/test_multimodel_v3_correctness.py \
  -k 'not model_owned_distributed_gradients_optimizer_and_checkpoint'
# 67 passed, 1 deselected in 18.96 s

/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/ruff check \
  aspd/multimodel/cli/validate_aspd_v3_main500m.py \
  aspd/multimodel/v3/main500m_config.py \
  tests/test_multimodel_v3_2b.py

bash -n slurm/multimodel/aspd_v3_500m_preflight_PROPOSAL.sbatch
bash -n slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch
bash -n slurm/multimodel/aspd_v3_500m_resume_validation_PROPOSAL.sbatch
```

The three-process Gloo checkpoint test previously passed in 302.80 s and was not repeated after the
evidence-schema-only change. The final 67-test non-distributed regression, Ruff, JSON parsing,
`git diff --check`, and all three Slurm scripts passed. Real jobs `50550194`, `50550768`, and
`50611663` provide the GPU preflight,
signal/requeue, and isolated exact-resume evidence respectively. The checked-in draft main
validator remains fail-closed because researcher decisions and final authorization are pending.

## Required researcher decision

Monitoring v1, staged milestones v1, and explicit acceptance of semantic uncertainty are recorded
in immutable artifacts. The only remaining action is a separate explicit instruction to execute
the documented `sbatch` command for the first staged interval.

## Direct answers

1. **Is the v3.1 architecture preserved?** Yes. No scientific objective, gate, loss or
   parameter-decomposition invariant changed.
2. **Is N=3 technically prepared?** Yes, including real final-source GPU execution and isolated
   recovery.
3. **Are 168 targets represented?** Yes; all 168 real physical hooks passed native parity with no
   missing or duplicate target.
4. **Is model 3 unresolved?** No. The exact Granite source contract is approved and pinned.
5. **Is exact resume validated?** Yes, bitwise in process and in an isolated new process.
6. **Is storage sufficient?** Yes at startup, with 1.119 TB free and a narrow 19.62 GB margin over
   the 1 TiB gate; runtime monitoring is required.
7. **Is throughput measured?** Yes for the final composition over four steps; the 500M projection
   remains provisional because the sample is short.
8. **Are monitoring and aborts implemented?** Yes, CPU-tested, and policy v1 is approved; emitted
   metric coverage is verified fail-closed during execution.
9. **Which approvals remain?** Only the separate external instruction to submit the main job.
10. **What command would submit after every approval?** From the clean, reviewed, approved commit:

```bash
ASPD_EXPECTED_GIT_SHA="$(git rev-parse HEAD)" \
  sbatch slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch
```

This command is documented only and was not executed.

## Final decision

`MAIN_READY_PENDING_FINAL_RESEARCHER_AUTHORIZATION`

Exact remaining blocker: `MAIN_500M_NOT_AUTHORIZED`.

