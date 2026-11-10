# ASPD v3.1 500M blocker-resolution tracker

Status date: 2026-11-09 (America/New_York)

Branch: `aspd-v3.1-main500m-blocker-resolution`

Base commit: `738f1a0d8742ce6c97c74cadb2550ffade648b19`

The canonical scientific contract remains the three-model hierarchical Transformer encoder with
per-anchor feature-axis z-score, ReLU, pool-wide BatchTopK, continuous activation code, and
detached binary internal support. No main-training job has been submitted.

## Live audit snapshot

- Slurm query at `2026-11-09T18:32:46-05:00`: no jobs owned by the researcher were queued or
  running. The sandboxed query failed, so this statement uses the subsequent live scheduler
  query outside the sandbox.
- `hpg-b200` was up with 60 nodes advertising eight B200 GPUs per node. This is partition
  inventory, not an availability or fair-share guarantee.
- `/orange/thai`: 213,821,587,456 bytes available; `/blue/thai`: 381,422,133,248 bytes available.
  Neither satisfies the provisional 1 TiB safety reservation.
- Inodes were not the immediate constraint: `/orange/thai` reported 13,422,163,115 free and
  `/blue/thai` reported 4,107,495,892 free.
- User and numeric group-3059 quota queries reported usage but no explicit soft/hard byte or inode
  limits. Filesystem-wide free capacity is the current constraint.

## Blockers

| ID | Blocker | Root cause | Files changed | Tests | Status | Approval needed |
|---|---|---|---|---|---|---|
| 1 | Preflight authorization deadlock | Main authorization, including prior preflight PASS, ran before mode dispatch; preflight had no independent bounded authorization or pinned result schema. | `main500m_config.py`, `train_aspd_v3_2b.py`, draft YAML, `test_multimodel_v3_2b.py` | Six focused CPU tests; full regression pending at this checkpoint. | `FIXED_TESTED` | Researcher must later authorize an exact model-3 preflight budget and namespace; the checked-in draft remains pending. |
| 2 | Third-model selection | Exact model, revision, tokenizer, layers, and hook layout are not approved. | Model-3 decision package and machine-readable recommendation. | Official config/model-card/source comparison plus read-only historical metadata cross-check; no model execution. | `BLOCKED_RESEARCHER_DECISION` | Approve or reject the proposed Granite ID/revision/layers, license, residual semantics and subsequent bounded tests. |
| 3 | Three-way alignment | Exact-prefix contract is implemented, but no approved third tokenizer has been tested; one legacy policy-name inconsistency is confirmed. | None yet | Synthetic coverage exists; real three-tokenizer evidence absent. | `BLOCKED_EXTERNAL_DEPENDENCY` | Model 3 first, then authorization for bounded tokenizer validation. |
| 4 | Storage capacity | Live free space is below the 1 TiB reservation. A dry-run inventory identifies a narrowly sufficient `/blue` Tier A, but no deletion or dedicated reservation is approved. | Storage proposal and machine-readable inventory. | Live `df`/inode/quota, exact `du`, COMPLETE markers, report and manifest SHA-256 checks. | `BLOCKED_STORAGE` | Exact deletion targets or additional capacity, then remeasurement and path approval. |
| 5 | Real three-model GPU preflight | Synthetic distributed tests do not validate three real source models. | None yet | Not run. | `BLOCKED_RESEARCHER_DECISION` | Model 3, alignment, and bounded preflight authorization. |
| 6 | Exact checkpoint resume | Synthetic three-rank resume passed previously; real three-model GPU resume remains untested. | None yet | Not run in this branch. | `BLOCKED_EXTERNAL_DEPENDENCY` | Real-model preflight prerequisites and bounded GPU authorization. |
| 7 | Sustained-collapse protection | The monitor was stateless; a collapse alert could not cause a recoverable controlled pause, and scheduled diagnostic completeness was not enforced. | `main500m_monitoring.py`, `main500m_config.py`, runner, draft YAML, tests. | Nine focused main-500M CPU tests; Ruff. | `FIXED_TESTED` | Researcher must approve the versioned v1 thresholds/actions before main authorization; the draft remains pending. |
| 8 | Staged milestones | Milestones existed only as YAML values; the runner neither forced diagnostics/checkpoints nor required an explicit continuation decision. | Runner, `main500m_continuation.py`, config/schema, authorization template, tests. | Thirteen focused main-500M CPU tests; Ruff; Python compilation. | `FIXED_TESTED` | Researcher must approve policy v1 and provide a checkpoint-pinned decision after each staged review; draft approval remains pending. |
| 9 | Human interpretability | The validated package exists, but no researcher response, graded human analysis, or decision artifact exists; the main schema could not distinguish these stages or accepted uncertainty. | Main config/schema, draft YAML, status artifact, decision template, tests. | Live read-only package/hash/status audit; focused main-500M tests; Ruff. | `BLOCKED_RESEARCHER_DECISION` | Researcher must complete responses and decide, or explicitly accept remaining semantic uncertainty in a hash-pinned decision artifact. |
| 10 | Slurm/runtime reliability | Main proposal exists; preflight proposal and bounded signal/requeue evidence remain incomplete. | None yet | Shell/runtime audit pending. | `AUDITING` | No main submission authorization exists. |
| 11 | Specification/metadata consistency | Confirmed alignment policy-name mismatch and human-status limitation; broader notation/config/runtime comparison remains in progress. | None yet | Audit in progress. | `AUDITING` | Documentation corrections do not require scientific approval; semantic changes would. |

## Blocker 1 resolution record

What was wrong: the v3.1 runner called full main authorization before it knew whether the request
was a preflight. Full authorization required an already-passed preflight, creating a circular
dependency.

What changed: bounded preflight authorization and completed preflight evidence are now separate,
typed records. Preflight requires approved source and alignment contracts, an exact budget of at
most 20 optimizer steps, model-owned checkpointing, a dedicated non-existing namespace, and no
external resume. Main independently requires every approval plus a SHA-256-pinned, semantically
validated preflight report. The preflight-to-main comparison uses an immutable execution-contract
fingerprint that excludes mutable approvals. Approved v3.1 source contracts reject Qwen before
execution.

Tests run:

```text
/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/pytest -q \
  tests/test_multimodel_v3_2b.py -k 'main500m'
6 passed, 17 deselected
```

`ruff` passed. The remaining 22 tests in the file passed inside the sandbox, while the existing
three-process Gloo test was blocked there by socket permissions. Its escalated retry initialized
without that permission error but did not complete after 147.58 seconds and was interrupted; it is
therefore recorded as inconclusive, not passing evidence.

Scientific semantics changed: no. The encoder, gate equations, z-score axis and epsilon,
BatchTopK, loss definitions, C, K, source freezing, target count, and exposure budget are
unchanged. The checked-in configuration remains fail-closed and does not authorize preflight or
main execution.

The committed CLI was also invoked directly in `--mode preflight` without distributed
environment variables. It failed before NCCL/source loading with all four expected blockers:
`MODEL_THREE_BLOCKED`, `MODEL_THREE_APPROVAL_PENDING`, `ALIGNMENT_BLOCKED`, and
`BOUNDED_PREFLIGHT_NOT_AUTHORIZED`.

## Blocker 4 audit record

The read-only inventory is documented in
`aspd_v3_500m_storage_reclamation_proposal.md` and its JSON companion. No files were removed.
The seven recommended `/blue` Tier A targets total 731,456,405,504 bytes and would project
1,112,878,538,752 bytes free from this snapshot—only 13,366,910,976 bytes above 1 TiB. That is
mathematically sufficient but operationally fragile under concurrent storage use. `/orange` cannot
reach the reservation using the identified safe candidates. Storage therefore remains blocked
pending exact-target approval plus additional margin or a dedicated allocation.

## Blocker 2 decision record

The recommendation package proposes
`ibm-granite/granite-3.3-2b-instruct@707f574c62054322f6b5b04b6d075f0a8f05e0f0`,
layers 16–23, for explicit researcher approval. Granite best balances a distinct model family,
2,533,539,840-parameter scale, GQA/residual-scaling diversity, and direct seven-matrix/PRE-MID
semantics. Phi-3.5 is the higher-diversity but substantially larger/fused secondary candidate;
Llama 3.2 is the lower-hook-risk but less diverse gated fallback; StableLM is deprioritized for its
license and possible parallel-residual MID ambiguity. The main YAML remains unchanged at
`model3_pending`; no candidate was downloaded or executed in this audit.

## Blocker 7 resolution record

What was wrong: single diagnostic snapshots could emit alerts, but the runner had no persistent
trajectory state, no checkpointed escalation counter, and no controlled-pause action. A scheduled
diagnostic row could also omit whole model/site groups without a fail-closed response.

What changed: policy `v1` requires all six model/site gate groups, 48 activation full-FVU values,
168 internal-FVU values, and three rank gradient reports on each scheduled diagnostic. Missing
coverage, nonfinite values, duplicated exposures, storage-reserve breaches, and checkpoint failures
remain immediate abort conditions. A single gate-health or reconstruction excursion remains an
alert. Three consecutive gate-health failures, or two consecutive reconstruction failures after
10,002,432 accepted exposures, request `CHECKPOINT_AND_PAUSE`. The runner persists the monitor
state in every complete checkpoint, restores it fail-closed, writes an immutable pause report, and
uses exit code 86 only after completing the recovery checkpoint. A resumed diagnostic step must
advance monotonically.

Tests run:

```text
/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/pytest -q \
  tests/test_multimodel_v3_2b.py -k 'main500m'
9 passed, 17 deselected

/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/ruff check \
  aspd/multimodel/v3/main500m_monitoring.py \
  aspd/multimodel/v3/main500m_config.py \
  aspd/multimodel/cli/train_aspd_v3_2b.py \
  tests/test_multimodel_v3_2b.py
All checks passed
```

The synthetic trajectories cover healthy, one-off warning, sustained gate collapse, delayed
reconstruction escalation, state save/restore, and incomplete-diagnostic abort. The full GPU
runner path is not claimed as executed; that remains part of the bounded real-model preflight.
All 25 non-distributed tests in `test_multimodel_v3_2b.py` also passed; the existing distributed
Gloo test was deliberately excluded because its prior live retry remains inconclusive.

Scientific semantics changed: no. The policy changes only observation and fail-safe control flow;
it does not alter z-score normalization, ReLU, BatchTopK, AuxK, loss weights, decoder constraints,
dictionary size, or sparse coefficients. The checked-in approval is
`PENDING_RESEARCHER_APPROVAL`, so the main run remains fail-closed.

## Blocker 8 resolution record

What was wrong: the six accepted-exposure milestones were passive YAML metadata. The runner could
cross 10M, 25M, 50M, 100M, and 250M without producing a milestone diagnostic, a consistent
checkpoint, or a researcher continuation decision.

What changed: the schema now pins the exact step-aligned exposure sequence
10,002,432; 25,001,984; 50,003,968; 100,007,936; 250,003,456; and 500,006,912.
Each boundary forces a complete diagnostic. The first five boundaries force an atomic recovery
checkpoint, an immutable scientific-review artifact, and controlled exit 87; the final boundary
completes normally. The review artifact retains all six model/site gate groups, all 48 activation
full-FVU values, all 168 internal-FVU values, the health decision, ordered-anchor digest, and
checkpoint-manifest SHA-256.

Continuation is fail-closed. The resumed checkpoint carries a `pending_review` record and cannot
perform another optimizer update without `--continuation-authorization`. The authorization must be
a JSON file directly under the configured authorization root and must pin the exact pause reasons,
accepted exposure, next review boundary, execution-contract fingerprint, and checkpoint-manifest
SHA-256. Consumed decisions are stored in later checkpoints and an idempotent append-only audit
log; conflicting decisions for the same exposure fail. The unfilled machine-readable template is
`analysis/aspd_v3_500m_blocker_resolution/continuation_authorization_TEMPLATE.json` and is
deliberately invalid until the researcher supplies a decision.

Tests run:

```text
/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/pytest -q \
  tests/test_multimodel_v3_2b.py -k 'main500m'
13 passed, 17 deselected

/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/ruff check \
  aspd/multimodel/v3/main500m_continuation.py \
  aspd/multimodel/v3/main500m_config.py \
  aspd/multimodel/cli/train_aspd_v3_2b.py \
  tests/test_multimodel_v3_2b.py
All checks passed
```

The tests cover exact step boundaries, final-boundary completion, checkpoint/contract provenance,
wrong-checkpoint rejection, idempotent audit logging, conflict rejection, and complete diagnostic
materialization. The GPU runner path remains unexecuted pending model-3 and bounded-preflight
approval.

Scientific semantics changed: no. Staging controls execution and review only. It does not change
the z-score gate, BatchTopK, AuxK, objectives, optimizer, or exposure ledger. The draft adds
`MILESTONE_POLICY_APPROVAL_PENDING`, so implementation does not manufacture approval.

## Blocker 9 audit and schema resolution record

What was wrong: the main contract represented human interpretation as only pending or approved and
could not distinguish package preparation, completed responses, graded analysis, evidence-based
approval, explicit acceptance of semantic uncertainty, or rejection. It also did not require a
decision artifact to identify the underlying evaluation evidence and limitations.

What was observed: the live read-only package manifest reports
`HUMAN_INTERPRETABILITY_PACKAGE_COMPLETE`; all eight package-integrity checks are PASS. The
manifest SHA-256 is `40a508cb821b319890fb49c6a44e47d4642a79275ef564fcebadbb3c1b98fe98`.
The public dataset and response template are present with their recorded hashes. No researcher
response export, graded human `analysis.json`, or researcher decision artifact was found. This is
package readiness, not completed human evaluation.

What changed: the draft now records
`HUMAN_INTERPRETABILITY_PACKAGE_COMPLETE_RESPONSES_PENDING`, the exact package manifest path and
hash, and three explicit scientific limitations. The schema separately represents analysis pending
decision, approval from human evidence, explicit acceptance of remaining semantic uncertainty, and
rejection. Only the two explicit approval states permit main authorization. A final approval must
be a hash-pinned researcher-decision JSON that pins the underlying evaluation artifact and its
SHA-256; evidence-based approval additionally requires an analysis artifact whose status is
`HUMAN_INTERPRETABILITY_ANALYSIS_COMPLETE`. The pending template cannot accidentally validate as
an approval.

Machine-readable audit:
`analysis/aspd_v3_500m_blocker_resolution/human_interpretability_status.json`.
Decision template:
`analysis/aspd_v3_500m_blocker_resolution/human_interpretability_decision_TEMPLATE.json`.

Scientific semantics changed: no. No response was fabricated, no human score was inferred from the
automated Llama output, and no interpretability approval was granted. The main contract remains
`HUMAN_INTERPRETABILITY_APPROVAL_PENDING`.
