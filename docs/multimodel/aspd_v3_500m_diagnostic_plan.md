# ASPD v3 500M checkpoint diagnostic plan

Status: `LOCAL_TESTED — GPU E0 NOT RUN`

This diagnostic is read-only. It does not expose an optimizer, update the checkpoint,
implement a shared/exclusive taxonomy, or perform causal intervention.

## Frozen inputs

- Source commit: `46ac649434755fd09d4f5c9e10392c11b5e9877d`
- Source config SHA-256: `06545c2c813508f4d9a2b8c68d7dbeace7c9b3a611707d927f1840af3fdf4c73`
- Checkpoint: `step-000061036`
- Checkpoint manifest SHA-256:
  `e45559e8b704b6516fbda62664b66bbee174379842736e028c815f310f7185e0`
- Dataset manifest SHA-256:
  `a258974d21ceac3457529644be07459518fde7bc28501abde5c67e5e3719286f`
- Alignment manifest SHA-256:
  `1a513988a5b15d5ce10dec7950edd0d8294bc94009feccf544ab22fafa3a92be`

## Execution stages

1. CPU Phase A audits source formulas and canonicalizes 61,659 physical metric rows to
   61,036 final optimizer-step rows. It writes the loss audit, compact trajectory and
   loss figure without loading a backbone or checkpoint tensor.
2. E0 uses three B200s because the shared encoder requires one owner rank per frozen
   model. It restores only model state, not Adam/scheduler/RNG, checks finite tensors,
   verifies replicated shared encoders, and repeats one 2,048-anchor inference exactly.
3. Held-out evaluation is blocked unless E0 emits `E0_CHECKPOINT_RESTORE_PASS` for the
   same checkpoint manifest. It evaluates 100,352 anchors (49 complete BatchTopK pools)
   from validation document buckets 0-9.

## Split and overlap boundary

Training admitted only documents assigned to the train buckets by SHA-256 of the frozen
seed, source group and content-derived document ID. The evaluator reuses the same pinned
dataset revision, document split, tokenizer revisions, exact three-way prefix alignment,
and exact duplicate rejection-before-split order. Therefore a document or exact duplicate
cannot be both a training and validation observation. The historical training pipeline
did not retain a cross-split near-duplicate shingle index, so semantic/near-duplicate
leakage is explicitly `UNVERIFIED`; the evaluation does not mislabel it as proven absent.

## Metric boundary

Training activation loss is the mean of six prefix FVUs, including the bias-only prefix.
The evaluator reports every prefix and full-reconstruction FVU separately. All aggregate
FVUs use summed SSE and centered target sums of squares rather than averaging batch FVUs.
Document-cluster bootstrap intervals are reported separately from the global point
estimate. Internal matrix reconstruction continues to use detached binary support;
activation reconstruction uses continuous code.

## Estimated resources

- CPU Phase A: under 2 minutes observed on the login CPU; about 46 MB compact output.
- E0: 3 B200, 96 GB host RAM, 45-minute limit. Expected 15-25 minutes dominated by
  loading three frozen backbones and the 123 GB distributed checkpoint.
- Held-out: 3 B200, 192 GB host RAM, 2-hour limit. The estimate is provisional until E0
  measures restore/inference time. The job fails closed rather than reducing anchor count.

No full held-out job should be submitted until E0 has been inspected.
