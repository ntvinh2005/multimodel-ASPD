# ASPD v3.1 500M blocker-resolution tracker

Status date: 2026-11-10 (America/New_York)

Branch: `aspd-v3.1-main500m-blocker-resolution`

Base commit: `738f1a0d8742ce6c97c74cadb2550ffade648b19`

The canonical scientific contract remains the three-model hierarchical Transformer encoder with
per-anchor feature-axis z-score, ReLU, pool-wide BatchTopK, continuous activation code, and
detached binary internal support. No main-training job has been submitted.

## Initial audit snapshot (2026-11-09; superseded where noted below)

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

The final audit supersedes the initial storage and execution state: after the exact approved
reclamation and retention of the real preflight checkpoint, `/blue/thai` had
1,119,127,543,808 bytes free. Jobs `50550194`, `50550768`, and `50611663` subsequently passed the
real three-model preflight, scheduler requeue harness, and isolated exact-resume validation.

## Blockers

| ID | Blocker | Root cause | Files changed | Tests | Status | Approval needed |
|---|---|---|---|---|---|---|
| 1 | Preflight authorization deadlock | Main authorization, including prior preflight PASS, ran before mode dispatch; preflight had no independent bounded authorization or pinned result schema. | `main500m_config.py`, `train_aspd_v3_2b.py`, draft YAML, `test_multimodel_v3_2b.py` | Authorization-focused tests and real four-step preflight pass. | `FIXED_TESTED` | No preflight authorization blocker remains; this does not authorize main. |
| 2 | Third-model selection | The exact source contract was previously unapproved. | Model-3 decision package, immutable researcher approval, pinned Granite config, schema and tests. | Exact contract loads; 168 unique targets register; mutation and Qwen rejection tests pass. | `FIXED_TESTED` | None for selection. Alignment, source hooks, GPU preflight and main submission remain separate gates. |
| 3 | Three-way alignment | The exact-prefix contract had not been tested with the approved Granite tokenizer or real pinned dataset rows. | Real audit CLI, offline tokenizer fix, raw/approval artifacts, validation report, pinned config evidence, tests. | 512 real documents, 154,868 anchors, invariants/repeatability/replay PASS, filter TV 0.0284; 4 audit unit tests. | `FIXED_TESTED` | Continue source/domain monitoring during training; no additional alignment approval is pending. |
| 4 | Storage capacity | Both filesystems were below the 1 TiB reservation. | Proposal, committed metadata record, exact eight-target cleanup, dedicated `/blue` namespace, config/readiness update. | Exact target absence, protected artifact presence, write probe, `df`/inode remeasurement. | `FIXED_TESTED` | No additional approval; runtime reserve monitoring remains mandatory. |
| 5 | Real three-model GPU preflight | Synthetic distributed tests did not validate three real source models. | Exact source/alignment/manifests, generic ledger schema, source-commit-pinned evidence, four-step launcher and profiled stage timings. | Job 50550194 completed `0:0`: 168 targets, exact native parity, frozen sources, finite updates, generic ledger, 0 repeated anchors, checkpoint restore PASS and bitwise in-process replay. | `FIXED_TESTED` | None for bounded GPU preflight; main remains independently unauthorized. |
| 6 | Exact checkpoint resume | Earlier real evidence reloaded state but did not compare an uninterrupted update with an isolated new-process resume. | Added SHA-256 parity over trainable model, Adam, scheduler and RNG; exact ordered-anchor/counter replay; source-state hashes; bounded cross-process authorization; and pinned evidence validation in the main manifest. | Job 50550194 passed in-process bitwise replay; isolated job 50611663 restored step 3 and reproduced step 4 bitwise with identical anchor digest, 0 repeats and immutable source weights. | `FIXED_TESTED` | None for exact resume. Main remains independently unauthorized. |
| 7 | Sustained-collapse protection | The original monitor was stateless. After deployment, policy v1 also treated the known early z-score effective-size valley as three consecutive failures because main sampled every 64 steps, unlike the successful pilot schedule. | `main500m_monitoring.py`, tests, readiness metadata, gate-health pause postmortem. | Five focused monitor tests pass; all 43 non-distributed v3 tests pass; the isolated CPU Gloo test passes; immutable job-50640347 diagnostic replay reaches step 320 without pause. | `FIXED_TESTED`, implementation `v1.1_zscore_warmup_calibrated` approved for review | Resume only from immutable step 321 into a new namespace with checkpoint-pinned continuation authorization; review steps 512 and 768. |
| 8 | Staged milestones | Milestones existed only as YAML values; the runner neither forced diagnostics/checkpoints nor required an explicit continuation decision. | Runner, `main500m_continuation.py`, config/schema, authorization template, tests. | Thirteen focused main-500M CPU tests; Ruff; Python compilation. | `FIXED_TESTED`, v1 approved | Provide a checkpoint-pinned decision after each staged review. |
| 9 | Human interpretability | The package is valid but has no response export or graded human analysis. | Main config/schema, status and researcher-decision artifacts, tests. | Package/hash audit and semantic decision validation. | `FIXED_TESTED`, uncertainty accepted | Preserve the explicit limitations; do not represent this as completed human grading. |
| 10 | Slurm/runtime reliability | The proposal forwarded `USR1` to torchrun, assumed worker exit 99 survived the launcher, and applied a fresh-start storage threshold to resume. | Replaced it with a file-backed stop request, verified COMPLETE checkpoint artifact, zero-exit torchrun contract, fresh-vs-resume storage thresholds, one-requeue CPU harness and isolated real-model resume launcher. | Job 50550768 passed one scheduler requeue with zero duplicates; job 50611663 passed isolated three-model resume bitwise. Static shell/Ruff/focused tests PASS. | `FIXED_TESTED` | None for engineering reliability; monitoring/milestone/human/final-run approvals remain separate. |
| 11 | Specification/metadata consistency | Canonical alignment was absent from the formal notation; legacy policy identifiers diverged; raw/normalized logits and Latin u versus Greek nu needed an explicit map; readiness metadata was stale. | Canonical specification, stream/manifest code, tests, consistency artifact and final readiness refresh. | Focused z-score/target/alignment tests, production-policy test, Ruff. | `FIXED_TESTED` | No scientific choice was made; pinned real-model evidence is represented separately. |

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
unchanged. The checked-in configuration now authorizes only the exact four-step preflight in the
dedicated fresh namespace; it remains fail-closed for main execution.

The committed CLI was also invoked directly in `--mode preflight` without distributed
environment variables. It failed before NCCL/source loading with all four expected blockers:
`MODEL_THREE_BLOCKED`, `MODEL_THREE_APPROVAL_PENDING`, `ALIGNMENT_BLOCKED`, and
`BOUNDED_PREFLIGHT_NOT_AUTHORIZED`.

## Blocker 4 audit record

The initial read-only inventory is documented in
`aspd_v3_500m_storage_reclamation_proposal.md` and its JSON companion. It first identified seven
Tier A `/blue` targets totaling 731,456,405,504 bytes, which was mathematically sufficient but
operationally fragile. The researcher subsequently approved an exact eight-target set after the
documentation record was committed. That set was removed, all eight targets were verified absent,
and protected final checkpoints plus the three approved source-model caches were verified present.
The live post-cleanup measurement was 1,244,881,256,448 bytes free on `/blue/thai`, leaving
145,369,628,672 bytes above the 1 TiB reservation. The immutable execution record is
`analysis/aspd_v3_500m_blocker_resolution/storage_reclamation_execution_record.json`; storage is
therefore resolved, subject to the mandatory runtime reserve monitor.

## Blocker 2 decision record

The recommendation package proposed
`ibm-granite/granite-3.3-2b-instruct@707f574c62054322f6b5b04b6d075f0a8f05e0f0`,
layers 16–23. The researcher approved that exact contract, and it is now pinned in the main draft.
Granite best balances a distinct model family,
2,533,539,840-parameter scale, GQA/residual-scaling diversity, and direct seven-matrix/PRE-MID
semantics. Phi-3.5 is the higher-diversity but substantially larger/fused secondary candidate;
Llama 3.2 is the lower-hook-risk but less diverse gated fallback; StableLM is deprioritized for its
license and possible parallel-residual MID ambiguity. Approval of the contract did not by itself
authorize source-model execution; real three-model GPU validation remains a separate bounded gate.

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
materialization. At this implementation stage the GPU runner path was unexecuted; later job
`50550194` exercised it with the approved three-model contract.

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

## Blocker 11 notation and metadata consistency record

The canonical source remains `aspd_v3_1_mathematical_specification.md`. It now defines the
three-way exact canonical-text boundary contract formally, including character offsets, exact
whole-prefix decode equality, UTF-8 byte boundaries, intersection across all three tokenizer
families, and fail-closed rejection conditions. The later bounded real-tokenizer audit passed this
contract for the approved Granite tokenizer together with Gemma and SmolLM2.

The notation-to-code map now makes these distinctions explicit:

- \(z_{a,t,c}\) is the pre-normalization projection logit;
- \(\widetilde z_{a,t,c}\) is the per-anchor feature-axis z-score with population variance and
  \(\epsilon_z=10^{-5}\);
- \(s_{a,t,c}=\operatorname{ReLU}(\widetilde z_{a,t,c})\);
- \(g^s\) is the continuous BatchTopK code, while
  \(g=\mathbf 1[g^s>0]\) is detached binary internal support; and
- Latin \(u\) is the write factor and Latin \(v\) is the read factor in
  \(P=uv^\top\). Historical Greek nu norm notation is not a rank-one factor.

The undocumented symbol `sigma_K` was removed from the code map in favor of the already-defined
selected index set \(\mathcal I_t\). Legacy fields named `latest_raw_logits` or `raw_logit_*` are
documented as compatibility aliases for normalized logits under v3.1 and must not be called
pre-normalization logits in new reports.

The production stream now accepts the canonical policy identifier explicitly, and the v3.1 runner
requires a schema-v4 alignment manifest whose dataset hash, model order, tokenizer IDs/revisions,
sequence length, policy, and acceptance rule match the approved execution contract. Historical
families retain their old policy identifiers by default. The manifest preparer has a separate
`--main500m` route that failed closed while model 3 was pending and now requires the exact approved
source/tokenizer contract rather than inventing metadata.

Machine-readable map:
`analysis/aspd_v3_500m_blocker_resolution/notation_consistency.json`.

Tests run:

```text
/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/pytest -q \
  tests/test_multimodel_v3_2b.py \
  -k 'not model_owned_distributed_gradients_optimizer_and_checkpoint'
31 passed, 1 deselected

/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/pytest -q \
  tests/test_multimodel_v3_production.py
9 passed
```

Ruff passed for the changed runtime, stream, manifest-preparation, and test files. Invoking the
new manifest route against the draft stopped before network access with
`main500m manifests require the approved third-model contract`, as intended.

Scientific semantics changed: no. These changes make the existing z-score and alignment semantics
unambiguous and fail closed on mismatched metadata.

## Blocker 10 Slurm and runtime reliability record

Follow-up audit found the previous proposal was not production-safe for requeue. Sending `USR1` to
the complete `setsid` process group also targeted `torchrun`, whose launcher has no `USR1` handler,
and a worker `SystemExit(99)` was surfaced as a launcher failure rather than a stable shell return
code. The 1 TiB fresh-start check also could not be applied unchanged after a 127 GB checkpoint
already existed.

What was wrong: the original main proposal resolved its environment and virtualenv relative to a
different checkout, did not pin the reviewed Git commit, did not enforce the 1 TiB startup reserve,
and did not distinguish a fresh namespace from a recoverable existing run. There was no separate
mode-locked bounded-preflight launcher. The configured checkpoint root also implied a separate
directory even though the runner stores checkpoints below each run directory.

What changed: the main and preflight proposals now use explicit account, partition, environment,
Python, and torchrun paths; require a clean worktree at `ASPD_EXPECTED_GIT_SHA`; and validate the
mode-specific authorization before CUDA initialization. Both enforce the configured 1 TiB startup
reserve. Preflight requires a new dedicated namespace and 3--20 authorized steps. Main requires an
existing run to contain both its immutable manifest and a complete checkpoint, while an external
resume checkpoint must contain `COMPLETE` and `manifest.json`. In-place recovery deliberately uses
the runner's latest complete checkpoint. Milestone continuation authorization is passed explicitly.
The corrected main proposal writes an atomic stop-request file instead of signalling the torchrun
process group. Rank 0 broadcasts that request at the next step boundary; the runner writes a
COMPLETE checkpoint and a hash-pinned signal-completion artifact, then exits zero. The shell
requeues only after independently checking that artifact and manifest hash. Resume requires free
space for one additional atomic checkpoint plus the configured 200 GiB reserve; only a fresh run
requires the full 1 TiB startup reservation. Controlled health and milestone exits 86/87 are not
automatically requeued.

CPU-only Slurm job `50550768` exercised the bounded control path with one automatic requeue. Slurm
recorded `Restarts=1` and final exit `0:0`; the second attempt restored cursor 3, advanced to cursor
4 / `anchor-000004`, reported zero duplicate anchors, and matched checkpoint SHA-256
`7224ea4f200d299d49acc7180001bcfb020b6112482322151598bd6162dbb581`. This validates the scheduler
and shell contract, not real three-model DCP; the latter was subsequently validated by jobs
`50550194` and `50611663` under Blocker 6.

Live scheduler metadata at `2026-11-09T19:39:08-05:00` showed account `thai`, partition
`hpg-b200` UP with 60 eight-B200 nodes and a 14-day partition limit. This is inventory, not an
allocation or fair-share guarantee. The only observed researcher-owned job was unrelated job
`50519752` on another node; this audit did not submit or cancel anything. No currently advertised
non-B200 partition is accepted as a safe fallback without its own bounded memory preflight.

Tests run:

```text
bash -n slurm/multimodel/aspd_v3_500m_main_PROPOSAL.sbatch
bash -n slurm/multimodel/aspd_v3_500m_preflight_PROPOSAL.sbatch
PASS

/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/pytest -q \
  tests/test_multimodel_v3_2b.py -k 'main500m'
17 passed, 17 deselected

/blue/thai/vinhnguyen1/multimodel-ASPD/repo/.venv/bin/ruff check \
  aspd/multimodel/cli/validate_aspd_v3_main500m.py \
  aspd/multimodel/v3/main500m_config.py tests/test_multimodel_v3_2b.py
All checks passed
```

The main draft validator still exits with `submission_permitted=false`. The machine-readable
evidence boundary is in
`analysis/aspd_v3_500m_blocker_resolution/runtime_reliability_status.json`.

Scientific semantics changed: no. These changes affect launch authorization, provenance, storage,
recovery, and scheduling control only; the Transformer, z-score gate, BatchTopK, losses, C, K,
target registration, and exposure accounting are unchanged.

## Blockers 5 and 6 real preflight and exact-resume record

Three-B200 job `50550194` completed four optimizer steps at source commit `f518cd9` with exactly
32,768 accepted exposures and 168 registered targets. It saved the step-3 self-contained recovery
checkpoint, ran step 4 uninterrupted, restored step 3 in process, and replayed step 4. SHA-256 over
the trainable model, Adam moments, scheduler and RNG was bitwise identical on all three ranks. The
ordered-anchor digest was
`d4ac6e41093400904afdd4be0e958aa2bfdd73af680a054cbdb122cc61aa9eea` in both trajectories;
unique anchors were 32,768 and repeated exposures were zero. Gemma, SmolLM2 and Granite source
state hashes were unchanged. The report SHA-256 is
`74cd3454f5e84f07d7cae11bf367cc28381a5e464b55cb0d04b8c878a318d7a6`.

Isolated three-B200 job `50611663` then loaded the immutable step-3 checkpoint into a fresh output
namespace and reproduced step 4 in a new torchrun process. Model/Adam/scheduler/RNG digests again
matched the uninterrupted reference bitwise, the same ordered-anchor digest and counters were
recovered, source weights remained unchanged, and `dcp_save_reload_resume_update=PASS`. Its report
SHA-256 is `a588215b907644593a91fc0e034bedd1f1efa24b6eb9b8eaef1799431a3d72d3`.
Both artifacts are pinned and revalidated by the main configuration. Scientific semantics changed:
no.

## Final readiness decision

`MAIN_AUTHORIZED_READY_FOR_SUBMISSION`

Independent engineering preparation is complete to the current authorization boundary. Model 3,
three-way alignment, storage, the real 168-target GPU preflight, profiled final-composition
throughput, exact in-process replay, isolated external resume, and the scheduler requeue contract
are now validated. Monitoring v1, milestone v1, and explicit semantic-uncertainty acceptance are
approved and hash-pinned. The external main submission is also explicitly authorized.
The complete audit is in
`aspd_v3_500m_final_readiness_audit.md`.

## Remaining-policy approval update

On 2026-11-10 the researcher approved monitoring policy v1 and staged-milestone policy v1 and
explicitly accepted the documented semantic uncertainty instead of claiming completed human
grading. The decision is pinned in
`analysis/aspd_v3_500m_blocker_resolution/remaining_policy_researcher_approval_2026-11-10.json`.
No threshold, milestone, loss, sparsity or gate semantic changed. The later explicit instruction
authorized the external `sbatch` action for the first staged interval.

## Model-3 approval update

On 2026-11-09 the researcher approved the exact Granite recommendation. The main contract now pins
`ibm-granite/granite-3.3-2b-instruct@707f574c62054322f6b5b04b6d075f0a8f05e0f0`, tokenizer at
the same revision, layers 16--23, and the approved physical layout. Registry validation now reports
168 approved physical targets and zero pending target slots. The source contract rejects any drift
from those values and continues to reject Qwen. The selection approval does not manufacture
alignment, physical-hook, GPU, resume, or throughput evidence.

Decision record:
`analysis/aspd_v3_500m_blocker_resolution/model3_researcher_approval.json`.

## Blocker 3 alignment resolution record

The exact approved Gemma, SmolLM2 and Granite fast tokenizers were run on a deterministic sample of
512 train documents selected from 2,000 rows of the pinned Pile Uncopyrighted revision. The audit
accepted 154,868 of 214,808 candidate boundaries (72.10%), found zero zero-anchor documents and zero
character/UTF-8/prefix-hash invariant failures, and passed both repeat alignment and independent
dataset-prefix replay. Alignment filtering changed the source-group distribution by TV 0.0284.

The lower-95% yield requires about 1.72M documents for 500M non-repeated anchors. The retained
Pile-CC component alone has an official size/mean-document estimate corresponding to roughly 55.0M
documents, a 32x margin. The dataset API lacks an exact row count, so this is explicitly recorded as
a conservative capacity estimate rather than a census.

The config now pins the validation artifact and revalidates its hash, tokenizer revisions,
execution-contract fingerprint, boundary/replay statuses, capacity and bias before bounded
preflight. Full report: `aspd_v3_500m_three_way_alignment_validation.md`.

## Blocker 4 storage resolution record

After exact-target approval, seven documented regenerable/redundant checkpoint payloads and the
historical Llama-3.3-70B judge cache were permanently removed from `/blue/thai`. The operation
removed only the eight committed absolute paths. Final step-1,221 checkpoints, reports, human
artifacts, worktrees, approved source caches and historical Qwen evidence remain present.

Post-cleanup free space is 1,244,881,256,448 bytes, 145,369,628,672 bytes above the 1 TiB startup
reservation. The dedicated root `/blue/thai/vinhnguyen1/aspd_v3/main500m_preparation_v1/` exists
and passed a write probe. Main, checkpoint, manifest, preflight and continuation paths now use that
root, and the runtime still aborts on reserve breach. Exact result and recovery limitations are in
`analysis/aspd_v3_500m_blocker_resolution/storage_reclamation_execution_record.json`.
