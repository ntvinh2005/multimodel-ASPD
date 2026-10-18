# Delta-Crosscoder Phase-2A run status

This file tracks only the 16-prompt contrastive cache pilot and the 75-step Delta engineering smoke.
Neither is a scientific result. The hard stop remains: no 50,000-step run and no organism work.

## Preflight

- Starting HEAD: `7e1aba04ac3f93aeb75c0d73f20026286c7b99e1`.
- Existing matched-cache implementation and artifacts are not modified.
- Qwen revisions: Base `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`; Target
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Canonical tokenizer/template: Target at the pinned Target revision; Qwen vocabularies are equal,
  while EOS metadata and raw template hashes differ.
- Static checks before submission: 82 multimodel tests passed; Ruff and `bash -n` passed.

## Jobs

To be filled from Slurm accounting and the generated `pilot_report.json` / `final_metrics.json`.
