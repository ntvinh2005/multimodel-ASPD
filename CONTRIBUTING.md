# Contributing to Multimodel ASPD

This repository contains reproducibility-sensitive training and evaluation infrastructure for
Multimodel ASPD. Contributors must preserve scientific provenance, immutable experiment artifacts,
and the separation between training and evaluation work.

## Collaboration model

The project uses one shared implementation, but contributors must not use the same mutable working
tree. Each contributor must use:

- a separate Git worktree or clone;
- a dedicated branch;
- a separate output namespace; and
- a separate cache or temporary directory when concurrent writes could conflict.

Recommended branch names include `training/<change>`, `evaluation/<owner>/<change>`,
`analysis/<change>`, and `docs/<change>`.

Do not reset, clean, rebase, delete, or switch branches inside another contributor's worktree. Do
not move or modify a worktree referenced by an active or pending Slurm job. Production jobs may pin
an exact Git commit and absolute filesystem paths.

## Responsibilities

### Training owner

The training owner is responsible for:

- training configuration and execution;
- checkpoint creation and exact recovery;
- accepted-exposure accounting;
- storage and Slurm reliability;
- training monitoring and abort criteria; and
- publishing immutable evaluation releases.

### Evaluation owner

The evaluation owner is responsible for:

- evaluation protocol implementation;
- automated and human evaluation;
- evaluation datasets and splits;
- judge configuration and calibration; and
- evaluation reports and artifacts.

Evaluation work must not modify training checkpoints, exposure ledgers, active run directories, or
the training owner's worktree.

## Synchronize with `main` before working

This rule applies to both the training agent and the evaluation agent. Before starting a code
change, adding a feature, or submitting a job, inspect the current remote state:

```bash
git fetch origin
git log --oneline HEAD..origin/main
git diff --name-only HEAD...origin/main
```

Review whether `origin/main` contains relevant changes to:

- evaluation implementation;
- checkpoint loading;
- model or target registries;
- datasets, splits, or anchor selection;
- judge configuration;
- metric definitions;
- output schemas or provenance; or
- Slurm and runtime configuration.

If a relevant change exists, update from `origin/main`, review the change, verify compatibility,
and rerun the necessary tests before editing code or submitting a job. Do not continue from an old
implementation merely because the current branch still runs.

Use `git pull --ff-only` only in a clean checkout that tracks the intended branch. Preserve any
existing work before integrating upstream changes.

## Code changes and pull requests

For a feature, protocol change, or other non-trivial implementation change:

1. Start from the latest `origin/main`.
2. Create a separate branch and worktree.
3. Implement and test the change.
4. Commit and push the branch.
5. Open a pull request.
6. Review and merge the pull request into `main`.
7. Create a clean worktree at the merged commit.
8. Only then submit the training or evaluation job.

Pull requests should describe:

- the motivation and files changed;
- scientific and provenance impact;
- exact tests executed and their observed results;
- checkpoint compatibility;
- storage and resource impact; and
- unresolved risks or required approvals.

Do not mix unrelated cleanup with scientific, checkpoint, or evaluation changes.

### Small bug-fix exception

A change is a small bug fix only when it does not alter:

- the scientific method;
- checkpoint semantics or schema;
- datasets, splits, or anchor selection;
- the evaluation protocol;
- judge prompts or judge models;
- metric definitions;
- thresholds or acceptance criteria;
- result interpretation; or
- the resource scale materially.

A bounded validation job may run before merge for a small bug fix only when:

- the fix is committed and pushed;
- the job pins the exact commit SHA;
- output is written to a dedicated validation namespace;
- the result is not reported as an official evaluation; and
- a pull request is opened promptly.

Every full evaluation and every result intended for a scientific report must run from a commit that
has already been merged into `main`.

## Shared implementation and provenance

Training and evaluation should use the same ASPD implementation whenever possible. Every result
must record:

- Git commit SHA;
- configuration SHA-256;
- checkpoint release ID and manifest SHA-256;
- source-model and tokenizer revisions;
- dataset or evaluation-manifest SHA-256;
- random seeds;
- relevant environment information; and
- Slurm job ID when applicable.

An evaluation-specific code change must be committed separately and its relationship to the
training commit documented.

## Checkpoints and evaluation releases

Training checkpoints are immutable. Do not:

- modify checkpoint tensors in place;
- overwrite a `COMPLETE` checkpoint;
- restart an optimizer while claiming exact resume;
- use a partial checkpoint as a completed release; or
- delete checkpoint recovery state without the project owner's approval of the exact target.

After main training completes:

1. Verify the final exposure count, checkpoint manifest, exposure ledger, and duplicate-exposure
   checks.
2. Upload every completed checkpoint still retained in the training namespace, including the final
   checkpoint and retained milestones, to the approved Hugging Face repository.
3. Verify remote file counts, SHA-256 values, and small provenance files by downloading them from a
   pinned Hugging Face commit.
4. Keep at least one validated model-only checkpoint on Blue or another approved local filesystem
   for evaluation.
5. Make shared checkpoint releases read-only.
6. Do not remove a local or remote checkpoint without a direct decision from the project owner.

Evaluation should normally use a model-only release rather than a full recovery checkpoint. A valid
evaluation release contains:

- model-only ASPD state;
- a release manifest and validation report;
- SHA-256 checksums;
- the training configuration and source-model contract;
- accepted-exposure count and optimizer step;
- Git provenance; and
- loading instructions.

The evaluation owner must write results to a separate namespace and must never write into a shared
checkpoint release.

## Artifact deletion authority

Only the project owner, `vinhnguyen1`, may authorize deletion of old evaluation artifacts.
Contributors and agents must not independently delete, overwrite, or relocate:

- evaluation outputs;
- checkpoints or checkpoint manifests;
- training or evaluation logs;
- provenance records; or
- project caches that may be required for reproducibility.

An agent may create a read-only inventory and recommend exact deletion targets. Deletion requires
the project owner's explicit approval of those targets.

## GPU scheduling and owner priority

The project owner and evaluation contributor may use no more than four GPUs in aggregate across all
project jobs, GPU types, accounts, and partitions. This is a project-wide limit, not a per-job
limit.

The project owner's work always has scheduling priority. Before every GPU submission, the evaluation
agent must check at least:

```bash
squeue -u vinhnguyen1
squeue -u "$USER"
squeue -A thai
sacctmgr show qos thai
```

The agent must count relevant `RUNNING` jobs and account for `PENDING` owner jobs that could be
delayed by a new evaluation allocation. In particular:

- if the project owner uses four GPUs, evaluation must not submit a GPU job;
- if the project owner uses three GPUs, evaluation may use at most one GPU;
- if the project owner uses two GPUs, evaluation may use at most two GPUs; and
- if an owner training job is pending because of QOS or resource limits, evaluation must not submit
  a job that could delay it.

The evaluation owner may submit evaluation jobs independently when these conditions are satisfied.
The evaluation owner must not cancel, preempt, reconfigure, or otherwise interfere with the project
owner's jobs.

Before submission, also verify group GPU, RAM, and CPU use; QOS limits; requested GPU type; storage
headroom; and other active project jobs. Avoid heavy checkpoint reads while training is writing a
checkpoint.

## Shared HiPerGator layout

A recommended shared layout is:

```text
/blue/thai/share/multimodel-ASPD/
├── releases/
│   └── <release-id>/
├── manifests/
├── evaluation-code/
└── evaluations/
    └── <contributor>/<evaluation-id>/
```

Release and manifest directories should be read-only. Contributors may write only to their own
evaluation namespace. Do not use a shared release directory as a mutable Git working tree. Do not
share a writable SQLite database, checkpoint directory, or result directory between contributors.

## Storage policy

Total project-owned local storage across Blue, Orange, shared directories, caches, temporary files,
and staging areas must remain below:

\[
1{,}000{,}000{,}000{,}000\ \text{bytes}.
\]

Before creating a large artifact:

- measure current usage;
- estimate temporary and final storage;
- account for atomic-write duplication;
- preserve exact-resume capability;
- prefer model-only evaluation releases; and
- archive important immutable artifacts remotely when appropriate.

Moving data between filesystems does not remove it from the project storage total. Cleanup must not
destroy reproducibility.

## ASPD v3.1 scientific identity

Evaluation of an ASPD v3.1 checkpoint must preserve the identity of the trained method:

- three frozen source models;
- eight selected layers per model;
- seven logical matrices per layer and 168 logical targets;
- a hierarchical Transformer encoder;
- per-anchor feature-axis z-score;
- ReLU followed by pool-wide BatchTopK;
- \(C=8192\) and \(K=32\);
- continuous activation reconstruction code;
- detached binary internal-reconstruction support;
- shared PRE/MID feature identities; and
- model-specific rank-one parameter operators.

Do not present evaluation of a modified architecture as evaluation of the original checkpoint. No
Qwen model may be introduced into a new ASPD v3.1 training, evaluation, inference, or judging run
without an explicit new researcher decision.

## Evaluation reporting

Evaluation outputs must distinguish:

- automated evaluation;
- human evaluation;
- causal-intervention evidence;
- reconstruction metrics;
- interpretability judgments;
- exploratory results; and
- scientifically validated results.

Automated judge results must include calibration and position-bias checks. A failed judge-validation
gate must not be reported as validated semantic evidence. Do not fabricate missing responses,
approvals, labels, or metrics.

## Testing

Run the smallest tests that establish the changed behavior. Every pull request must report:

- exact commands executed;
- tests passed and failed;
- checks not executed;
- CPU, synthetic-GPU, or real-model validation status; and
- whether scientific semantics changed.

Synthetic tests must not be described as real-model validation.

## Secrets and credentials

Never commit Hugging Face tokens, GitHub tokens, API keys, private judge credentials, or personal
authentication files. Use supported HiPerGator credential mechanisms or environment variables.

## Training-to-evaluation handoff

The training owner publishes an evaluation release only after:

1. The target accepted-exposure count is reached.
2. The final checkpoint is marked `COMPLETE`.
3. Checkpoint manifest hashes are verified.
4. The exposure ledger is validated.
5. Duplicate accepted exposures are confirmed absent.
6. Source-model and configuration provenance is frozen.
7. The model-only release passes finite-tensor validation.

The evaluation owner records the release ID, Git commit, and manifest hash before beginning an
evaluation.
