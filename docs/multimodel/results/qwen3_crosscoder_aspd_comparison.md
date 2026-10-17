# Qwen3 general model diff: Crosscoder versus multimodel ASPD

This report compares blind discoveries on the same `Qwen/Qwen3-1.7B-Base` and
`Qwen/Qwen3-1.7B` pair, token grid, train/validation split, and layer-14 input residual stream.
Candidate ranking used only measured model-diff statistics. Semantic review happened afterward and
did not change the ranks.

## Verified global results

| Measurement | Crosscoder | Multimodel ASPD |
|---|---:|---:|
| Dictionary size | 65,536 | 8,192 |
| Training optimizer steps | 10,000 | 10,000 |
| Validation tokens | 65,536 | 65,536 |
| Train BatchTopK | 100 | 32 |
| Measured validation L0 | 115.90 | 32 by exact BatchTopK invariant |
| Validation FVE | 0.99377 | method objective is not directly comparable |
| Zero/dead validation features | 7 (0.0107%) | 4,054 (49.49%) |

Crosscoder's relative decoder norm ranges from 0.123 to 0.846. Under the paper's strict
`<=0.1`/`>=0.9` exclusivity thresholds, it contains no base- or fine-tuned-exclusive features:
65,357 are shared and 179 are intermediate. This negative strict-threshold result is retained. The
analysis separately selects the lowest and highest score tails for inspection without relabeling
them as exclusive.

ASPD assigns 3,570 components to shared activation/shared mechanism, 45 to shared
activation/concentrated mechanism, one to concentrated activation/concentrated mechanism, none to
concentrated activation/shared mechanism, and 4,576 to mixed/subset mass. Its saved fire-count
invariant is exact: 65,536 validation tokens times top-k 32 equals 2,097,152 fires.

## Post-ranking semantic observations

| Phenomenon | Crosscoder evidence | ASPD evidence | Interpretation |
|---|---|---|---|
| Hidden-reasoning-to-answer boundary | Fine-tuned-tail 62735 centers on `</think>`; 18996 centers immediately after empty think blocks | Component 736 centers on `</think>` in 10 retained contexts | Both methods surface the boundary, but ASPD classifies 736 as shared activation/shared mechanism rather than a changed mechanism. This is analogous evidence, not feature identity. |
| Generic chat turn boundaries | Fine-tuned-tail 46552 activates on `<|im_start|>`/`<|im_end|>`; 14851, 53275, and 37279 activate at end-of-turn transitions | Components 654 and 3131 activate at document/new-user and assistant-response transitions | Strong semantic overlap at representation level; ASPD's high-support examples here are mechanism-shared. |
| Role-specific response onset | Base-tail 11544 centers on `user`; 38275 centers on `assistant`; 21266 and 24222 mark assistant answer onset | Components 330 and 36 include the same boundary token family | ASPD 330/36 are base-concentrated mechanisms but have only five validation fires each, so this possible parameter-level correspondence is low-support and remains a hypothesis. |
| Fine-tuned-concentrated ASPD mechanism | Crosscoder has broad fine-tuned tails but no strict-exclusive feature | ASPD component 440 has shared activation (`rho_base=0.4998`) and fine-tuned-concentrated mechanism (`rho_ft=0.9398`), with 257 fires and strongest locus at `o_proj` in both models | This is ASPD-only parameter evidence. Its retained contexts are heterogeneous, so no confident semantic match to a Crosscoder tail is claimed. |
| Concentrated representation and mechanism | No strict Crosscoder-exclusive feature | ASPD component 156 is fine-tuned concentrated in activation (`rho_ft=0.9370`) and mechanism (`rho_ft=0.9825`), with 1,969 fires | Strong global ASPD specificity, but current contexts appear numeric/document-initial and do not yet support a stable semantic interpretation. |

## What the comparison supports

- Crosscoder robustly reconstructs both residual streams and detects graded representation-space
  asymmetry around chat formatting, role changes, and the hidden-reasoning boundary.
- ASPD independently surfaces components associated with the same token-boundary family, while
  distinguishing shared versus concentrated parameter support.
- The most direct semantic overlap currently lands in ASPD's shared-mechanism components. This is
  a disagreement with any simple claim that representation specificity must imply parameter
  concentration.
- ASPD components 440 and 156 provide stronger parameter-specific evidence than the matched
  boundary components, but their semantics remain unresolved.

## What is not established

- No Crosscoder feature is strictly model-exclusive under the published 0.1/0.9 thresholds.
- No one-to-one Crosscoder–ASPD feature identity has been measured.
- Similar context labels do not prove the same causal mechanism.
- The labels are tentative qualitative annotations, not measured ground truth.
- Causal ablation has not been run and should only follow selection of a well-supported ASPD
  mechanism with a stable semantic interpretation.

## Reproducibility artifacts

- Crosscoder metrics and contexts:
  `out/multimodel/baselines/crosscoder/r1_qwen3_full/analysis/`
- ASPD P1–P5 report:
  `out/multimodel/runs/qwen3_1_7b_d0_s1_main/analysis/report/`
- Structured annotations:
  `docs/multimodel/results/qwen3_general_diff_annotations.json`
- R0 published-reference reproduction:
  `out/multimodel/baselines/crosscoder/r0_gemma2_full/published_analysis/`
