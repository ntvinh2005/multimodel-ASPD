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
| 7 | Sustained-collapse protection | Monitor is stateless; a collapse alert does not cause a checkpointed controlled pause. | None yet | Existing single-snapshot tests only. | `AUDITING` | Threshold/action policy must remain versioned; final threshold approval required. |
| 8 | Staged milestones | Milestones are configured but the runner does not enforce checkpointed review pauses. | None yet | No enforcement test yet. | `AUDITING` | Continuation decisions at configured milestones. |
| 9 | Human interpretability | Interface/protocol exist, but no versioned researcher response or decision artifact is present; completion status schema is too narrow. | None yet | Preparation tests exist; no scientific response evidence. | `BLOCKED_RESEARCHER_DECISION` | Researcher completion and explicit semantic-risk decision. |
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
