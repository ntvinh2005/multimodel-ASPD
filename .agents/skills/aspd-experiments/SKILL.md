---
name: aspd-experiments
description: Implement or review ASPD experiment infrastructure, configurations, provenance, local tests, and non-submitting HPC templates while preserving historical runs and open scientific decisions.
---

# ASPD experiments

Use this skill for ASPD training infrastructure, experiment configuration, data provenance,
activation hooks, decomposition code, and run-readiness reviews. Do not use it as authorization to
run an experiment or implement evaluation that the researcher deferred.

## Required workflow

1. Read repository state and applicable `AGENTS.md` files before editing.
2. Identify the experiment family and immutable historical references. Reuse utilities only after
   checking that their scientific assumptions match.
3. Separate approved settings, proposed settings, derived values, runtime choices, and historical
   settings. An unresolved scientific field must remain explicit and block execution.
4. Preserve exact model, tokenizer, dataset, split, configuration, and checkpoint provenance.
5. Add deterministic synthetic tests for shapes, alignment, accounting, resume, and failure modes.
6. Run only bounded local checks unless the user separately authorizes external compute.
7. Report implementation, run, and scientific verification as independent states.

## Scientific invariants

- Never align different tokenizer families by numerical token index. Use a verified canonical text
  boundary and reject ambiguous mappings.
- Capture the actual input and output of each physical matrix. A grounding residual is not a matrix
  input merely because its width matches.
- Treat fused QKV and fused gate/up projections as explicit logical slices without modifying frozen
  weights.
- Keep activation reconstruction distinct from rank-one matrix-output reconstruction.
- Do not select dictionary capacity, token packing, gate grouping, sparse-code semantics, gradient
  routing, loss weights, or matrix execution scope without researcher approval.
- Count accepted aligned exposures, unique aligned anchors, per-model raw tokens, optimization steps,
  and repeated data separately.

## Safety and reproducibility

- Do not overwrite completed artifacts or reuse their output paths.
- A Slurm template must be non-submitting and fail closed while readiness blockers remain.
- Do not claim scientific verification from unit tests, static validation, or a technical smoke test.
- Evaluation and causal claims require their own explicit implementation and execution approval.
