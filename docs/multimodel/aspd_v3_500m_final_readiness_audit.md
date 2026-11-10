# ASPD v3.1 three-model 500M final readiness audit

Status date: 2026-11-09 (America/New_York)

Decision: **`MAIN_BLOCKED`**

No 500M job was submitted. No storage artifact was deleted. The implementation package is
substantially prepared, but the main experiment is not yet
scientifically or operationally authorized.

## Evidence boundary

This audit distinguishes three claims:

1. **Implemented and CPU-tested:** fail-closed authorization, N-model tensor/registry contracts,
   canonical notation and alignment schemas, sustained-health control, staged-review pauses,
   checkpoint state machinery, launch guards, and machine-readable readiness artifacts.
2. **Historically or synthetically tested:** three-rank distributed mechanics and deterministic
   checkpoint resume. These do not prove the final source-model composition.
3. **Not yet executed:** real three-model physical-hook/GPU preflight, real three-model exact
   resume, scheduled signal/requeue test, and
   final-composition throughput measurement.

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

This supports the statement **N=3 framework prepared at static/synthetic level**. It does not
support the stronger statement that 168 physical targets have been validated against three real
source models.

## Blocker disposition

| ID | Result | Current state | Remaining gate |
|---|---|---|---|
| 1. Preflight authorization | Separate bounded-preflight and main authorization implemented and tested; exact four-step `/blue` preflight is authorized. | `FIXED_TESTED` | Execute once; it cannot transition to main. |
| 2. Model 3 | Exact Granite contract approved, populated, pinned, and CPU-tested. | `FIXED_TESTED` | Alignment and real source execution remain separate gates. |
| 3. Alignment | Exact canonical-text prefix validation passed on 512 real documents with pinned tokenizers. | `FIXED_TESTED` | Continue source/domain monitoring during execution. |
| 4. Storage | Exact approved cleanup completed; dedicated `/blue` root has 1.245 TB free. | `FIXED_TESTED` | Continue runtime reserve monitoring. |
| 5. Real GPU preflight | Exact Granite/alignment/manifests and four-step authorization are ready; no real three-model run yet. | `IN_PROGRESS` | Execute bounded three-B200 preflight. |
| 6. Exact resume | Synthetic three-rank deterministic resume evidence exists; recovery is self-contained and fail closed. | `RESUME_BLOCKED` | Real three-model uninterrupted-versus-resumed update and interruption test. |
| 7. Monitoring | Stateful sustained-failure policy and controlled pause implemented and CPU-tested. | `FIXED_TESTED`, approval pending | Researcher approval of policy v1 and real-run metric coverage. |
| 8. Milestones | Exact exposure boundaries, atomic checkpoint, immutable review, and continuation authorization implemented. | `FIXED_TESTED`, approval pending | Researcher approval of staged policy v1. |
| 9. Human evidence | Valid package exists; zero researcher response exports and no decision were found. | `HUMAN_INTERPRETABILITY_APPROVAL_PENDING` | Complete human analysis or explicitly accept documented semantic uncertainty. |
| 10. Slurm/runtime | Main/preflight launch contracts and static tests complete; bounded preflight is authorized. | `IN_PROGRESS` | Execute scheduled preflight and bounded signal/requeue test. |
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

Real three-model exact resume is still required: compare the next uninterrupted update with the
restored update, including optimizer tensors, sparse support, ordered anchor IDs, exposure count,
and source-weight immutability. Numerical equality must be reported as bitwise exact or with a
predeclared justified tolerance.

## Storage decision

The original snapshot found neither filesystem met the 1,099,511,627,776-byte startup reservation.
After exact approval, eight documented targets were deleted from `/blue/thai`. The dedicated root
now has 1,244,881,256,448 bytes free, 145,369,628,672 bytes above the reservation. Estimated
checkpoint and recovery requirements remain 821--872 GB before a final preflight measurement.

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
remain pending researcher approval; no monitoring action changes loss weights or sparsity.

## Runtime and resource plan

The preferred allocation is one node with three B200 GPUs on account `thai`, partition `hpg-b200`,
96 GB host memory, local temporary storage, and explicit Git/config/environment pins. The current
partition limit is 14 days. Main and bounded preflight have distinct proposal scripts and cannot
cross modes. Existing runs require a valid manifest and complete checkpoint; fresh preflight
namespaces cannot exist.

Historical different-composition evidence measured 1.445 s median step time, 6,256.6 accepted
anchors/s, and approximately 62.6/78.5/72.2 GB peak rank memory. It suggests 24--30 wall hours and
72--90 B200 GPU-hours, but **this is not measured v3.1 final-composition throughput**. The real
preflight must replace it with forward/backward/optimizer/data/communication time distributions,
peak memory, accepted anchors/s, and checkpoint save/reload timing.

No advertised non-B200 partition is a safe fallback yet. The fail-closed fallback is to wait for
B200 capacity unless a separate bounded memory/throughput preflight approves another GPU class.

## Reproducible checks executed

```bash
/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/pytest -q \
  tests/test_multimodel_v3_2b.py \
  -k 'not model_owned_distributed_gradients_optimizer_and_checkpoint'
# 33 passed, 1 deselected

/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/pytest -q \
  tests/test_multimodel_v3_production.py
# 9 passed

/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/ruff check \
  aspd/multimodel/cli/validate_aspd_v3_main500m.py \
  aspd/multimodel/v3/main500m_config.py \
  tests/test_multimodel_v3_2b.py

bash -n slurm/multimodel/aspd_v3_500m_preflight_PROPOSAL.sbatch
bash -n slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch
```

The existing multi-process Gloo retry remains inconclusive and is not included as passing evidence.
The checked-in draft main validator exits 2 with `submission_permitted=false`. The exact four-step
bounded-preflight authorization is now populated; its real scheduled execution is pending.

## Required researcher decisions

The four-step preflight in the exact fresh `/blue` namespace is already authorized by the
researcher's instruction to verify unresolved evidence. It does not require the later main-only
monitoring, milestone, or human-interpretability decisions and cannot transition to main.

Before final main authorization, the researcher must:

1. approve monitoring policy v1 and staged-milestone policy v1;
2. complete the human response/analysis decision, or explicitly accept the documented remaining
   semantic uncertainty;
3. review passing 168-target hook/parity, GPU, checkpoint/resume,
   source-immutability, signal/requeue, throughput, and storage evidence;
4. regenerate and approve all configuration/manifest fingerprints after those decisions; and
5. issue a separate explicit authorization for the 500M submission and the first staged interval.

## Direct answers

1. **Is the v3.1 architecture preserved?** Yes, at specification, configuration, and tested CPU
   implementation level; no scientific objective or gate/loss invariant was changed.
2. **Is N=3 technically prepared?** Static and synthetic infrastructure: yes. Real final source
   integration remains pending alignment and GPU evidence.
3. **Are 168 targets represented?** Yes as logical and approved physical slots. Real physical-hook
   validation of all 168 is pending.
4. **Is model 3 unresolved?** No. The exact Granite source contract is approved and pinned.
5. **Is exact resume validated?** Synthetic three-rank recovery: yes. Real final three-model exact
   resume: no.
6. **Is storage sufficient?** Yes at the current snapshot: 1.245 TB free on the dedicated `/blue`
   filesystem, with runtime reserve monitoring still required.
7. **Is throughput measured?** Only for a historical different composition; final v3.1 throughput
   is estimated, not measured.
8. **Are monitoring and aborts implemented?** Yes and CPU-tested; threshold approval and real-run
   coverage are pending.
9. **Which approvals remain?** Model 3, alignment audit, storage, monitoring, milestones, human
   interpretation, bounded real preflight, evidence review, and final main authorization.
10. **What command would submit after every approval?** From the clean, reviewed, approved commit:

```bash
ASPD_EXPECTED_GIT_SHA="$(git rev-parse HEAD)" \
  sbatch slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch
```

This command is documented only and was not executed.

## Final decision

`MAIN_BLOCKED`

Exact remaining blockers:

- `RESUME_BLOCKED`
- `THREE_MODEL_GPU_PREFLIGHT_PENDING`
- `REAL_SLURM_SIGNAL_REQUEUE_PENDING`
- `FINAL_COMPOSITION_THROUGHPUT_UNMEASURED`
- `MONITORING_POLICY_APPROVAL_PENDING`
- `MILESTONE_POLICY_APPROVAL_PENDING`
- `HUMAN_INTERPRETABILITY_APPROVAL_PENDING`
- `MAIN_500M_NOT_AUTHORIZED`

