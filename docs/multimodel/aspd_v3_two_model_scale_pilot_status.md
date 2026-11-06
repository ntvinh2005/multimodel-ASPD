# ASPD v3 Gemma/SmolLM2 scale pilot status

Updated: 2026-11-06

## Scientific boundary

- Models: Gemma-2-2B-IT and SmolLM2-1.7B-Instruct only.
- Layers: Gemma 9--16; SmolLM2 8--15.
- Matrices: q, k, v, o, mlp_gate, mlp_up and mlp_down in every selected layer.
- Total targets: 112.
- Dictionary/sparsity: C=8192, BatchTopK K=32.
- Each arm: 1,221 optimizer steps and 10,002,432 accepted aligned-anchor exposures.
- No Qwen source, third model, 500M run, C/K change, input normalization or
  Matryoshka reweighting is authorized in this experiment.

## Arms

1. `baseline`: the two-model form of the existing ASPD v3 method.
2. `model_specific_scale`: a positive site/layer gain applied only to each
   activation-decoder residual around its bias. A single log parameter produces
   reciprocal Gemma and SmolLM2 gains, so their geometric mean is exactly one.

The treatment adds 16 FP32 values. At zero initialization it is bitwise identical to
the baseline reconstruction while retaining the correct scale gradient. The scale is
also applied to bias-free AuxK reconstruction, but never to the rank-one internal
objective or binary support.

## Frozen configuration identities

- Baseline fingerprint: `336d7eeb2fab5ccb65d2764804d09af7c75018af6e9345432f601e081fb0f825`
- Treatment fingerprint: `702915817d6e91f18c3f012157ada6cd47cdc7921bf5e6536479a14e1a2aaf76`
- Dataset revision: `3be90335b66f24456a5d6659d9c8d208c0357119`
- Baseline and treatment must have identical ordered-anchor SHA-256 at all 1,221
  steps and identical hashes for every common initial tensor.

## Verification completed before submission

- Ruff on every changed Python file: PASS.
- Python compilation/import/CLI help: PASS.
- Shell syntax for pilot, evaluation and comparison Slurm scripts: PASS.
- Focused CPU tests: 31 passed, one existing distributed test separated.
- Existing three-rank gradient/optimizer/DCP regression: 1 passed in 139.04 s.
- New scale tests cover exact scale=1 parity, bias handling, gradients, reciprocal
  gauge, two-model packing, target count and fail-closed C/K validation.

## Execution state

Implementation: `LOCAL_TESTS_PASSED`

Execution commit: `af3278585d25bcd33468b84568e9455846e2c0ef`

Submitted: `2026-11-06T11:34:36-05:00`

| Stage | Arm | Job ID | Submitted state | Resources |
|---|---|---:|---|---|
| Preflight + short pilot | baseline | 49975792 | RUNNING | 2 B200, 16 CPU, 128 GB, 3 h |
| Preflight + short pilot | treatment | 49975793 | RUNNING | 2 B200, 16 CPU, 128 GB, 3 h |
| Checkpoint smoke + held-out | baseline | 49975794 | PENDING dependency | 2 B200, 16 CPU, 128 GB, 4 h |
| Checkpoint smoke + held-out | treatment | 49975795 | PENDING dependency | 2 B200, 16 CPU, 128 GB, 4 h |
| Paired comparison | both | 49975796 | PENDING dependency | 2 CPU, 8 GB, 30 min |

GPU preflight/training/evaluation: `SUBMITTED`. Both pilot allocations began at
`2026-11-06T11:34:28-05:00` on the frozen commit above. The evaluation jobs use
`afterok` dependencies, and the final comparison waits for both evaluations. No 500M
job, Qwen job, third-model job or dependent retraining job was submitted.

### First preflight attempt

Jobs 49975792 and 49975793 failed before any optimizer step. Baseline exposed a scalar
buffer serialization bug in the newly added common-state hash; treatment also observed
the documentation commit while checking worktree cleanliness. Dependent jobs 49975794,
49975795 and 49975796 were cancelled because their dependencies could never be
satisfied. The failure evidence and logs are retained; neither failure is a scientific
pilot result.

### Corrected submission

Execution commit: `081da39bd31d17d343dde9a38fc74707b859584b`

| Stage | Arm | Job ID | Initial state |
|---|---|---:|---|
| Preflight + short pilot | baseline | 49976302 | PENDING |
| Preflight + short pilot | treatment | 49976303 | PENDING |
| Checkpoint smoke + held-out | baseline | 49976304 | PENDING dependency |
| Checkpoint smoke + held-out | treatment | 49976305 | PENDING dependency |
| Paired comparison | both | 49976306 | PENDING dependency |

The corrected chain was submitted only after the scalar-buffer regression test passed.
The live execution worktree remains pinned and clean at the execution commit; this
status update is published from a separate detached documentation worktree.

The GPU scripts run a three-step two-B200 optimizer/checkpoint/reload preflight before
starting the corresponding short pilot. Held-out evaluation uses exactly 100,352
validation anchors and cannot start unless the pilot checkpoint is complete. A final
CPU comparison verifies paired batches and common initialization before calculating
document-cluster bootstrap differences.
