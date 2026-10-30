# Repository agent instructions

These instructions apply to the entire repository. A nested `AGENTS.md` may add more specific
constraints without weakening these rules.

## Working agreement

- Use one primary coding agent. Do not delegate to subagents for repository work.
- Inspect the branch, HEAD, worktree, relevant configuration, and applicable instructions before
  editing. Preserve unrelated user changes.
- Separate a scientific proposal from approval to implement it, and separate implementation from
  authorization to run it.
- Make the smallest executable change that satisfies the approved scope. Preserve useful existing
  comments and avoid unrelated cleanup.
- Use deterministic local tests and report the exact command and observed result. Never describe an
  unexecuted check as passing.
- Record immutable dataset, tokenizer, model, checkpoint, configuration, and split provenance in
  experiment artifacts.
- Never submit or cancel an HPC job unless the user explicitly requests that external action in the
  current task. A generated Slurm file is not authorization to submit it.
- Do not silently change an approved hyperparameter or resolve an open scientific question. Keep an
  executable configuration blocked until the researcher approves every required decision.
- Report implementation, run, and scientific-verification status independently.
- Preserve historical experiment configurations, manifests, reports, checkpoints, logs, and output
  namespaces. Cleanup must not destroy reproducibility.

## Repository guidance

- When present, read `skills/karpathy-guidelines/SKILL.md` before nontrivial implementation or review
  work.
- When present, read `skills/communication/SKILL.md` before writing user-facing research terminology.
  Terms already specified by the researcher in the current request are approved for that request.
- When present, read `skills/experiment-doc/SKILL.md` before editing an experiment log or scientific
  results report.
- For ASPD experiment infrastructure, also read
  `.agents/skills/aspd-experiments/SKILL.md` when it exists.

## ASPD v3 boundary

ASPD v3 is a separate, fail-closed experiment family. Its current evaluation status is
`NOT_IMPLEMENTED — DEFERRED BY RESEARCHER`. Do not import Mix3, 100M, P6, or Delta scientific
settings into v3 merely because their utilities are reusable.
