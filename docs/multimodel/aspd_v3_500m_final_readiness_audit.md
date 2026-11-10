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
3. **Not yet executed:** the approved third tokenizer, real three-way alignment, real three-model
   physical-hook/GPU preflight, real three-model exact resume, scheduled signal/requeue test, and
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
| 1. Preflight authorization | Separate bounded-preflight and main authorization implemented and tested. | `FIXED_TESTED` | Populate only after exact model-3/alignment approval. |
| 2. Model 3 | Exact Granite contract approved, populated, pinned, and CPU-tested. | `FIXED_TESTED` | Alignment and real source execution remain separate gates. |
| 3. Alignment | Exact canonical-text prefix-boundary schema and fail-closed manifest validation implemented. | `ALIGNMENT_BLOCKED` | Real three-tokenizer acceptance, failure, exhaustion, reproducibility, and bias study. |
| 4. Storage | Live capacity and deletion dry run quantified; nothing deleted. | `STORAGE_BLOCKED` | At least 1 TiB authorized free space with operational margin. |
| 5. Real GPU preflight | Launcher and evidence schema prepared. | `THREE_MODEL_GPU_PREFLIGHT_PENDING` | Approved model 3, alignment, storage, and bounded GPU authorization. |
| 6. Exact resume | Synthetic three-rank deterministic resume evidence exists; recovery is self-contained and fail closed. | `RESUME_BLOCKED` | Real three-model uninterrupted-versus-resumed update and interruption test. |
| 7. Monitoring | Stateful sustained-failure policy and controlled pause implemented and CPU-tested. | `FIXED_TESTED`, approval pending | Researcher approval of policy v1 and real-run metric coverage. |
| 8. Milestones | Exact exposure boundaries, atomic checkpoint, immutable review, and continuation authorization implemented. | `FIXED_TESTED`, approval pending | Researcher approval of staged policy v1. |
| 9. Human evidence | Valid package exists; zero researcher response exports and no decision were found. | `HUMAN_INTERPRETABILITY_APPROVAL_PENDING` | Complete human analysis or explicitly accept documented semantic uncertainty. |
| 10. Slurm/runtime | Main/preflight launch contracts and static tests complete. | `BLOCKED_EXTERNAL_DEPENDENCY` | Bounded scheduled signal/requeue test and final-composition timing. |
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

After model-3 approval, the validation must measure acceptance rate and categorized rejection rate,
prove deterministic stream replay, estimate dataset exhaustion before 500M non-repeated accepted
exposures, and quantify source/domain selection bias. Synthetic tokenizer fixtures are not approval
evidence.

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

The audit snapshot found 213,821,587,456 bytes free on `/orange/thai` and 381,422,133,248 bytes on
`/blue/thai`; neither meets the 1,099,511,627,776-byte startup reservation. Estimated checkpoint
and recovery requirements remain 821--872 GB before a final model-3 measurement.

The seven Tier-A `/blue` candidates could reclaim 731,456,405,504 bytes, projecting only
13,366,910,976 bytes above 1 TiB at the snapshot. That margin is too fragile under concurrent use.
The exact dry-run list is in
[`aspd_v3_500m_storage_reclamation_proposal.md`](aspd_v3_500m_storage_reclamation_proposal.md).
Deletion requires explicit path-by-path approval and fresh dependency/size/hash checks. A dedicated
allocation or additional capacity is preferable. Historical checkpoints, active worktrees, human
evaluation artifacts, and non-regenerable Qwen-era evidence remain protected.

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
The checked-in draft main validator exits 2 with `submission_permitted=false`; the bounded-preflight
validator stops on model-3, alignment, and authorization gates before CUDA/source loading.

## Required researcher decisions

Before any real-model preflight:

1. approve or reject the exact Granite recommendation, including revision, tokenizer, layers
   16--23, PRE/MID semantics, residual scaling disposition, license, and parameter count;
2. authorize the bounded tokenizer alignment audit;
3. approve an authorized storage root with at least 1 TiB free plus margin, or approve exact
   deletion targets from the dry-run proposal;
4. approve monitoring policy v1 and staged-milestone policy v1; and
5. complete the human response/analysis decision, or explicitly accept the documented remaining
   semantic uncertainty.

Before final main authorization:

6. approve the exact bounded preflight namespace and 3--20 step budget;
7. review passing real three-model alignment, 168-target hook/parity, GPU, checkpoint/resume,
   source-immutability, signal/requeue, throughput, and storage evidence;
8. regenerate and approve all configuration/manifest fingerprints after those decisions; and
9. issue a separate explicit authorization for the 500M submission and the first staged interval.

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
6. **Is storage sufficient?** No. `STORAGE_BLOCKED` remains correct.
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

- `ALIGNMENT_BLOCKED`
- `STORAGE_BLOCKED`
- `RESUME_BLOCKED`
- `THREE_MODEL_GPU_PREFLIGHT_PENDING`
- `REAL_SLURM_SIGNAL_REQUEUE_PENDING`
- `FINAL_COMPOSITION_THROUGHPUT_UNMEASURED`
- `MONITORING_POLICY_APPROVAL_PENDING`
- `MILESTONE_POLICY_APPROVAL_PENDING`
- `HUMAN_INTERPRETABILITY_APPROVAL_PENDING`
- `MAIN_500M_NOT_AUTHORIZED`

