# Delta-Crosscoder Phase-1 smoke status

This file tracks only the engineering smoke required by Phase 1. It is not a scientific experiment.

## Canonical committed-code smoke

| Field | Value |
|---|---|
| Submitted | 2026-10-17 23:34 EDT |
| Job | `46509106` (`mm-delta-crosscoder`) |
| Launcher | `slurm/multimodel/delta_crosscoder.sbatch`, `MODE=smoke` |
| Config | `configs/multimodel/delta_crosscoder/qwen3_matched_smoke.yaml` |
| Commit | `4b04e038a25703d24d791395686e9c31077dc791` |
| Input | existing 46 GiB matched Qwen cache, `model.layers.14` input |
| Work | 100 steps, 2,048 activation tokens/step, 1,280 features |
| Partition | 256 shared / 1,024 delta; `K_shared=16`, `K_delta=8` |
| Resource | one B200, 14 CPUs, 96 GiB RAM; 30-minute hard limit |
| Runtime expectation | a few minutes after allocation; target under 15 minutes |
| Final state | completed, exit `0:0` |
| Allocation/runtime | `c0901a-s25`; 51 s Slurm elapsed, 47 s launcher, 26.8 s training loop |

Artifacts are in `out/multimodel/baselines/delta_crosscoder/qwen3_matched_smoke/`.

## Pre-commit probe

Job `46508127` was a successful 54-second **pre-commit probe**. Its Slurm log records parent commit
`f1c2774e9f4a589fa6e0a002e685337f95aa9139` while the new implementation was still a working-tree
diff. Its artifacts were moved to `/tmp/delta_crosscoder_smoke_46508127_precommit`; the canonical
artifact directory was regenerated from committed code.

Job `46508441` was a successful 55-second committed probe at `8d30c05`. It exposed that the
reconstruction scalar did not yet match the pinned standard Crosscoder when `lambda_delta=0`.
After correcting and testing that scale, its artifacts were moved to
`/tmp/delta_crosscoder_smoke_46508441_prescale_fix` and replaced by the canonical run above.

No full 50,000-step run has been submitted. The reusable launcher's `train` mode is additionally gated by `ALLOW_FULL_TRAIN=1`.
