# ASPD v3.1 post-hoc parent-set validation

## Status

This record documents the completed ASPD v3.1 500M post-hoc parent-set validation run.
The run is technically complete and hash-verified. It ran from commit `9c9f492` before
that implementation was merged into `main`, so it remains a **validation result**, not
an official evaluation under `CONTRIBUTING.md`.

The 29.39 GB raw tensor artifacts remain on approved Blue project storage and are not
committed to Git. Their immutable locations and hashes are recorded below and in the
adjacent JSON manifest.

## Provenance

- Experiment: `posthoc_v31_500m_validation_v1`
- Source commit: `9c9f492c02dafc3a79ab51fd310637b74a1caa05`
- Checkpoint step: `61,036`
- Checkpoint manifest SHA-256: `735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969`
- Training config SHA-256: `f7a6d87436304956ca0baca955ec2e0e929a7207142f57fc35362369822f63be`
- Post-hoc config SHA-256: `9f0e75339da13ae48f4d3b4ea6131cf0a93f26d46e19576c12e5bddb68728f31`
- Dataset manifest SHA-256: `a258974d21ceac3457529644be07459518fde7bc28501abde5c67e5e3719286f`
- Split: validation
- Source dtype: bfloat16
- Natural gate: per-anchor feature-axis z-score, ReLU, pool-wide BatchTopK K=32
- Raw root: `/blue/thai/share/multimodel-ASPD/evaluations/minhdatle/posthoc/v31_500m_validation_v1_20261122T223718`

## Completed stages

| Stage | Evidence |
|---|---|
| Importance | 102,400 aligned anchors; job `52411989`; `COMPLETE` SHA-256 `d022fe24...614c` |
| Representatives | 16,277 gates with at least one representative; job `52411989`; `COMPLETE` SHA-256 `c57a8d4a...3f8fe` |
| Edges | 8/8 shards, 2,756,154 records; jobs `52411990_0` through `_7`; `COMPLETE` SHA-256 `b1bfbec2...21d1` |
| Compare | 936,837 pair comparisons over 15,673 gates; job `52411991`; `COMPLETE` SHA-256 `e2bc7f49...5649` |

Of 15,675 screened gates, `mid/7969` and `pre/2364` had no natural firing in the
disjoint edge split and therefore produced no comparison rows.

## Descriptive results

These are anchor-weighted descriptive means. They are not component-clustered confidence
intervals and the artifact labels them `DESCRIPTIVE_ANCHOR_MEANS_NOT_A_VALIDATED_STATISTIC`.

| Model pair | Parent-gate Jaccard | Total variation | Valid TV rows |
|---|---:|---:|---:|
| Gemma–Granite | 0.356019 | 0.480115 | 299,769 |
| Gemma–SmolLM2 | 0.341955 | 0.493454 | 294,369 |
| SmolLM2–Granite | 0.325533 | 0.513518 | 293,559 |

## Verification and limitations

- All final-stage hashes and the 8-shard index chain were verified after completion.
- The post-hoc unit/contract suite passed: 39 tests.
- Exact-ablation diagnostics contain 56 checks, all from Gemma. Signs agreed in 56/56;
  median relative error was 4.8%, but the 90th percentile was 54.4% and the maximum
  was 87.9%. This does not validate Taylor magnitudes across all models or nonlinear paths.
- Allowing the downstream MID gate to appear among its own parents raises MID Jaccard by
  about 0.026 absolute in a post-run sensitivity diagnostic.
- No matched-random null, parameter sensitivity grid, or component/document-clustered
  uncertainty estimate is included.
- Parent-set correspondence is not, by itself, evidence of semantic coherence or a
  universal mechanism.

No pretrained source weights or ASPD checkpoint tensors were modified by this evaluation.
