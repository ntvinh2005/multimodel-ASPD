# Delta-Crosscoder Phase-1 smoke status

This file tracks only the engineering smoke required by Phase 1. It is not a scientific experiment.

| Field | Value |
|---|---|
| Submitted | 2026-10-17 23:22 EDT |
| Job | `46508127` (`mm-delta-crosscoder`) |
| Launcher | `slurm/multimodel/delta_crosscoder.sbatch`, `MODE=smoke` |
| Config | `configs/multimodel/delta_crosscoder/qwen3_matched_smoke.yaml` |
| Input | existing 46 GiB matched Qwen cache, `model.layers.14` input |
| Work | 100 steps, 2,048 activation tokens/step, 1,280 features |
| Partition | 256 shared / 1,024 delta; `K_shared=16`, `K_delta=8` |
| Resource | one B200, 14 CPUs, 96 GiB RAM; 30-minute hard limit |
| Runtime expectation | a few minutes after allocation; target under 15 minutes |
| Final state | completed, exit `0:0` |
| Allocation/runtime | `c0901a-s5`; 54 s Slurm elapsed, 50 s launcher, 29 s training loop |

This first pass was a successful **pre-commit probe**: the Slurm log records parent commit
`f1c2774e9f4a589fa6e0a002e685337f95aa9139` while the new implementation was still a working-tree
diff. A canonical smoke will therefore be rerun from the committed implementation so its provenance
identifies the exact code revision.

No full 50,000-step run has been submitted. The reusable launcher's `train` mode is additionally gated by `ALLOW_FULL_TRAIN=1`.
