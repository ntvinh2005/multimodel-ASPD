# Qwen2.5 two-model ASPD 100M: final report

**Final status (2026-10-25):** `P6_HANDOFF_READY`. The frozen dataset, distributed smoke test,
100M-token-equivalent training, checkpoint verification, held-out scientific evaluation, and
observational behavioral screening all completed. This document is the durable Git-tracked report;
the authoritative large artifacts remain under
`out/multimodel/runs/qwen25_aspd_original_100m/`.

## A. Executive summary

- The run reached exactly 24,414 optimizer steps and 99,999,744 paired token exposures.
- Both source LLMs remained frozen; losses and gradients stayed finite; the two DDP trainer ranks
  ended with equal learned parameters.
- The final paired checkpoints and atomic marker passed SHA-256, cursor, sampler, and configuration
  verification. Measured sustained throughput excluding startup was 23,357 positions/s.
- Held-out evaluation covered all 36,864 components and 1,999,872 validation positions.
- Residual-space reconstruction is strong (FVU 0.00380 Base, 0.00368 Target), while reconstruction
  of the isolated L14 `o_proj` output is much weaker (FVU about 0.50).
- Parameter-only Base-to-Target delta reconstruction is not faithful: NSE 110.49 and cosine 0.143.
  This is a negative scientific result that constrains downstream claims.
- Clean-forward risk/control screening completed on 60 matched pairs and produced a P6 shortlist.
  It is observational evidence only; no candidate is yet a causal risk mechanism.

## B. Frozen experiment

| Item | Frozen value |
|---|---|
| Base | `unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee` |
| Target | `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d` |
| Grounding | Input of `model.layers.14.post_attention_layernorm` |
| Decomposed matrix | `model.layers.14.self_attn.o_proj` only |
| Components | 36,864 model-specific rank-one pairs with a shared gate |
| Sparsity | BatchTopK `K=32`, pool 2,048 positions/rank |
| Distributed layout | One producer rank plus two DDP trainer ranks on 3×B200 |
| Global training batch | 4,096 positions/optimizer step |
| Training corpus | Pinned FineWeb general-domain data |
| Training budget | 24,414 steps; 99,999,744 positions |
| Validation set | 3,906 sequences; 1,999,872 positions |

Risk, sports-control, and neutral prompts were held out from training. No supervised risk objective
or Delta-Crosscoder loss was used.

## C. Execution record

| Stage | Slurm job | Result | Runtime / evidence |
|---|---:|---|---|
| Linux Gloo verification | 47683248 | Completed | 48 s; exact resume and 3-process DDP synchronization passed |
| Dataset preparation | 47683250 | Completed | 11:25; 196 shards and `READY` manifest |
| Final technical smoke | 47686200 | `GO` | Resume 0→50→100, evaluator load and DDP equality verified; 10,333 positions/s |
| Scientific main run | 47688329 | Training complete | Final checkpoint at step 24,414; training wall time 1:14:34 |
| Scientific evaluation | 47705932 | Completed | 22:12; full metrics and P6 export generated |

The main Slurm wrapper initially ended nonzero after successful training because the verifier
expected a metric row at step 24,414 while periodic logging had stopped at 24,410. The checkpoint,
summary, and cursor all proved exact completion. The verifier now accepts a final row within one
configured logging interval when exact checkpoint evidence is present, and the trainer always logs
the terminal step for future runs.

## D. Training and checkpoint verification

| Measurement | Result |
|---|---:|
| Status | `100M_TRAINING_COMPLETE` |
| Final step | 24,414 |
| Token exposures | 99,999,744 |
| Metrics rows | 2,442 |
| Losses / gradients finite | Yes / Yes |
| Source LLM weights unchanged | Yes |
| DDP parameters equal | Yes |
| Rank-0 / rank-1 dead components | 0 / 0 |
| Rank-0 / rank-1 JumpReLU threshold | 3.0222008 / 3.0248675 |
| Throughput excluding / including startup | 23,357.24 / 22,659.77 positions/s |
| Peak GPU memory producer / trainers | 31.86 / 46.56 / 46.76 GB |
| Verified run-directory size | 76,440,046,786 bytes |

Final integrity identifiers:

| Artifact | SHA-256 |
|---|---|
| Frozen configuration | `904a91d2ab2c63cafd5f62c9c012bcfbd418f72de8114fffa8f4d14469626c40` |
| Dataset manifest | `82fbb6a33b0e6104a165a4bd2f88526ee7175744de92de17bb8a2ee71dc04da8` |
| Atomic checkpoint marker | `00a9d9344c39b47f1d582bee78c491186437c34be8047a88f87abb7311da7d52` |
| Final rank-0 checkpoint | `bbd4f6023566f25775d1daa300dd9e5de40e54cb0e1aa07c2819b1c27c9b30d6` |
| Final rank-1 checkpoint | `eda8b22938f2aa70ae946e00d4dc1a929012686748827f9e6006f5167cdacbd7` |

Rank 0 is the canonical inference/P6 source because the frozen recipe intentionally leaves
rank-local statistics unsynchronized. Learned model parameters themselves are DDP-equal.

## E. Held-out scientific measurements

### Reconstruction

| Measurement | Base | Target |
|---|---:|---:|
| Residual activation FVU | 0.00380263 | 0.00367925 |
| L14 `o_proj` output FVU | 0.501645 | 0.496139 |

FVU is the unexplained variance fraction, so lower is better. The contrast between the two rows is
important: high residual reconstruction fidelity does not imply similarly faithful reconstruction
of the isolated `o_proj` computation.

### Base-to-Target computation deltas

| Quantity | Actual RMS | Reconstructed RMS | RMSE | NSE | Cosine |
|---|---:|---:|---:|---:|---:|
| Parameter-only delta | 0.00298118 | 0.0316238 | 0.0313363 | 110.489 | 0.143119 |
| Input-change delta | 0.0439703 | 0.0227206 | 0.0370451 | 0.709812 | 0.539158 |
| Total Base–Target delta | 0.0442507 | 0.0393788 | 0.0429476 | 0.941969 | 0.477556 |

The numerical identity `ΔY_total = ΔY_parameter + ΔY_input` is sound: relative RMS residual is
`4.00271e-06` for the true computation and `1.80410e-08` for the ASPD reconstruction. Therefore
the poor parameter-only delta metric is a fidelity finding, not an arithmetic decomposition bug.

### Component utilization

| Measurement | Result |
|---|---:|
| Fired at least once | 36,367 / 36,864 (98.652%) |
| Never fired on held-out data | 497 |
| Mean learned JumpReLU L0 | 32.3664 |
| BatchTopK reference L0 | 32 |
| JumpReLU/BatchTopK support Jaccard | 0.964906 |
| Canonical learned threshold | 3.0222 |
| Effective utilization | 1,141.13 components |

The high fired-once fraction and much smaller effective utilization mean activity is spread across
most components but contribution mass remains concentrated in a substantially smaller subset.

## F. Observational screening and P6 shortlist

Behavioral screening status is `VERIFIED` for 60 risk prompts, 60 matched sports controls, and 60
neutral prompts. Exactly the same token IDs were replayed through both models and the gate remained
shared. This is clean-forward observational screening, not an intervention.

The leading computational candidates are:

| Rank | Component | Base contribution RMS | Target contribution RMS |
|---:|---:|---:|---:|
| 1 | 1765 | 3.34945 | 3.26207 |
| 2 | 554 | 1.16476 | 1.16938 |
| 3 | 1891 | 0.840235 | 0.849654 |
| 4 | 37 | 0.709078 | 0.702837 |
| 5 | 1277 | 0.669711 | 0.659512 |
| 6 | 115 | 0.634484 | 0.629602 |
| 7 | 548 | 0.621660 | 0.618334 |
| 8 | 894 | 0.610896 | 0.612723 |
| 9 | 1114 | 0.536074 | 0.566493 |
| 10 | 661 | 0.546895 | 0.537555 |

P6 must freeze the clean gate and compare gate-off, matched-active, orthogonal, and random
same-L2 controls over multiple seeds while recording the actual perturbation norm. Until those
interventions are run, candidate rankings support prioritization only.

## G. Artifact contract

The final package contains 26 checksum-verified artifacts and reports:

- training: `100M_TRAINING_COMPLETE`;
- evaluation: `SCIENTIFIC_EVALUATION_COMPLETE`;
- behavioral screening: `VERIFIED`;
- package: `P6_HANDOFF_READY`.

Authoritative local paths:

- Package manifest: `out/multimodel/runs/qwen25_aspd_original_100m/result_package_manifest.json`.
- Checkpoint verification: `out/multimodel/runs/qwen25_aspd_original_100m/checkpoint_verification_report.json`.
- Full held-out report: `out/multimodel/runs/qwen25_aspd_original_100m/evaluation/held_out_evaluation.json`.
- Full 36,864-row table: `out/multimodel/runs/qwen25_aspd_original_100m/evaluation/component_metrics.csv`
  (37,185,676 bytes; SHA-256
  `2208a40044799518d2f0342ccc50bdb4a6dc657e0f4b89b3c5ff8f172a94f719`).
- Computational shortlist: `out/multimodel/runs/qwen25_aspd_original_100m/p6_handoff/candidates_computational.csv`.
- Risk-screening shortlist: `out/multimodel/runs/qwen25_aspd_original_100m/p6_handoff/candidates_risk_screening.csv`.

The full CSV, checkpoints, runtime logs, and generated plots stay under ignored `out/` rather than
being duplicated in Git. This report and the compact
[`qwen25_aspd_original_100m_metrics.json`](qwen25_aspd_original_100m_metrics.json) snapshot preserve
their review-critical measurements and integrity identifiers. See the
[runbook](../qwen25_aspd_original_100m_runbook_vi.md) for execution commands and the
[scientific handoff](../qwen25_aspd_original_100m_handoff_vi.md) for the locked design contract. The
[post-training optimization audit](qwen25_aspd_original_100m_optimization_failure_audit.md) explains
the residual/internal/parameter-delta fidelity gap without changing the completed experiment.

## H. Claim ledger

**Verified:** exact training budget; frozen source weights; checkpoint integrity; DDP parameter
equality; held-out reconstruction/delta/utilization measurements; prompt/tokenization provenance;
and observational screening completion.

**Supported but non-causal:** component contribution and utilization rankings, plus group-level
clean-forward activation differences.

**Not established:** causal effect on risk acceptance, semantic identity of individual components,
or generalization beyond the frozen dataset, model pair, layer, and matrix. P6 intervention is the
next required scientific stage.
