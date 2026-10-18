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

- Cache job `46518291`: `COMPLETED` (`0:0`) on `c0906a-s5`; Slurm elapsed 1m37s.
  Materializer wall time 58.24s, including 25.79s generation and 0.41s activation extraction.
- Delta smoke job `46518503`: `COMPLETED` (`0:0`) on `c0901a-s5`; Slurm elapsed 30s.
  Trainer wall time through final validation was 6.61s.
- Cache fingerprint:
  `2b9b6d9efd745a428cf3aae7a5bb38aff419ee674aff6c8949638ac65ef2f1f3`.
- Checkpoint reload: step 75, 31,575 tokens seen, source mode `contrastive_cache`, matching cache
  fingerprint and preserved partition state.

Both jobs used implementation commit `d71a8334d927839384ffe409fb9011fd92b1d352`. Full
engineering results and caveats are in `results/delta_contrastive_phase2a_report.md`.
