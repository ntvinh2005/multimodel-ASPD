# Delta-Crosscoder Phase-1 and Phase-2A reference audit

Audit date: 2026-10-17. Primary specification: [Kassem et al., *Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes*, arXiv:2603.04426v1](https://arxiv.org/abs/2603.04426).

## Source and code status

The arXiv paper and the authors' public pages were searched by exact title, author, and method name. No repository is linked by the paper, and no verifiable author implementation was found as of the audit date. Therefore `aspd/multimodel/baselines/delta_crosscoder.py` implements the equations from the paper. It does not import or substitute an unrelated “DeltaCoder” project.

Official-code repository and commit: **none verified**. Consequently, this document has no `VERIFIED FROM OFFICIAL CODE` implementation claims.

## PAPER-SPECIFIED

The code-to-equation map is:

| Paper | Definition | Implementation |
|---|---|---|
| Eqs. 1–2 | model-specific encoders, then `u = 0.5 (u_base + u_ft)` and BatchTopK | `DeltaCrosscoder.post_relu`, `DeltaCrosscoder.encode` |
| Eq. 3 | model-specific decoder reconstructions | `DeltaCrosscoder.decode` |
| Eq. 4 | `R_base = ||d_base|| / (||d_base|| + ||d_ft||)` | `delta_relative_decoder_norm_base` |
| Eq. 5 | `Delta = b - a` | `DeltaCrosscoder.delta_target` |
| Eqs. 6–8 | decoder difference predicts the activation difference | `DeltaCrosscoder.delta_reconstruction`, `delta_losses` |
| Eq. 9 | zero the shared block in the delta path | `DeltaCrosscoder.masked_delta_features` |
| Eq. 10 | reconstruction + sparsity/AuxK + `lambda_delta L_delta` | `delta_losses`, `_auxiliary_loss` |

Other paper-specified settings recorded in `paper_default.yaml` are: expansion factor 5; 20% shared capacity; base sparsity 200; shared multiplier 2; AuxK coefficient 1/32; `lambda_delta=0.005`; Adam at `1e-4`; 50,000 steps; 1,000 warmup steps; 4,096-token batch; decoder norm scale 0.4; bfloat16; gradient checkpointing; and death after 10,000,000 tokens without a firing.

The published data mix is FineWeb, LMSYS instruction data, organism fine-tuning data when available, and contrastive prompt/response inputs. The paper reports about 200M total tokens, about 20M contrastive tokens, maximum sequence length 1,024, and 200,000 prompts split between base- and target-generated responses.

For a contrastive prompt `x`, the data contract is not response-token alignment. It constructs `x || y_base` and `x || y_target`; each complete text is independently run through **both** LMs at the same hook. Phase 2A implements this contract as a small static engineering cache; it is not the paper's full data mixture or scale.

## Phase-2A contrastive cache contract

`aspd/multimodel/delta_contrastive.py` materializes the cache, and
`ContrastiveDeltaSource` reads it without loading a language model or generating text during
training. For each frozen prompt, greedy Base and Target generations create two sequences. Each
sequence's one canonical token tensor is replayed through Base and Target, producing an activation
tensor with an explicit `[token, model={base,target}, activation]` axis. Consequently the only Delta
target exposed to the trainer is `h_target(s) - h_base(s)` for the same `s`.

The Qwen audit found equal 151,669-entry token-to-ID vocabularies but different tokenizer metadata:
Base declares `<|endoftext|>` as EOS while Target declares `<|im_end|>`, and the raw templates have
different hashes. The Phase-2A cache therefore pins the Target tokenizer and Target chat template as
canonical, disables thinking in the rendered generation prefix, stores that exact template in the
manifest, and records the native tokenizer fingerprints/templates for both models. Generated token
continuations are appended directly to the canonical prompt IDs; there is no decode/re-tokenize step
before activation extraction.

Padding is right-sided, explicitly masked, zero-filled in cached activation padding, and excluded by
the source. Sequence boundaries remain in shards. The maximum length is 256; deterministic right
truncation records original, retained, and removed token counts per example. The pilot uses 16 frozen,
general-purpose prompts committed as JSONL, 64-token maximum greedy answers, four prompt-level
validation items, the existing `model.layers.14` input hook, and pinned local Qwen revisions. These
choices prioritize pipeline auditability and make no behavioral claim.

Materialization and training remain separate:

```bash
MODE=contrastive_cache sbatch slurm/multimodel/delta_crosscoder.sbatch
# after cache validation succeeds
MODE=contrastive_smoke sbatch slurm/multimodel/delta_crosscoder.sbatch
```

The dedicated direct CLI is:

```bash
.venv/bin/python -m aspd.multimodel.cli.build_delta_contrastive_cache \
  --config configs/multimodel/delta_crosscoder/qwen3_contrastive_smoke.yaml \
  --max-prompts 16 \
  --output out/multimodel/cache/delta_contrastive/qwen3_phase2a_16 \
  --device cuda
```

## VERIFIED FROM OFFICIAL CODE

None. No official implementation could be verified, so details absent from the paper must remain explicitly inferred below.

## INFERRED / REPO-LOCAL DESIGN DECISIONS

- **Dual-K mapping.** Appendix A names base sparsity 200 and a shared multiplier 2.0; Section 3.3 says the shared budget is larger and `K_delta = alpha K_shared`, `alpha < 1`. The only direct mapping consistent with both statements is `K_delta = 200`, `K_shared = 400`. Selection is independent and batch-global inside each partition. The paper does not spell out this mapping or selection algorithm, so this is inferred rather than code-verified.
- **BatchTopK ranking.** Candidates are scaled by the sum of the two decoder-vector norms, matching the pinned local BatchTopK Crosscoder machinery. The paper does not specify this scaling.
- **Biases.** The paper equations omit biases. The implementation uses one joint encoder bias after the half-average and per-model decoder biases for ordinary reconstruction. The explicitly defined delta path is biasless, as in Eqs. 7–9.
- **Activation preprocessing.** Paper-reference and smoke configs use raw cached activations. Optional ASPD RMS preprocessing is not used.
- **Reconstruction scalar.** `L_reconstruction` uses the existing pinned BatchTopK Crosscoder's mean per-model L2 norm. This makes `lambda_delta=0` reduce exactly to its reconstruction objective when the sparsity settings are matched; Eq. 10 does not define the scalar reduction.
- **AuxK.** The paper specifies coefficient 1/32 but not how AuxK interacts with Dual-K. The implementation performs dead-feature selection independently in the shared and delta partitions.
- **Inference thresholds.** Two independent EMA thresholds are learned, one per partition. Threshold inference is a local engineering choice because the paper does not define threshold training/application.
- **Gradient checkpointing.** When enabled, the training forward is checkpointed with PyTorch non-reentrant checkpointing. This honors the reported switch, but the exact official checkpoint boundary is unknown.
- **Matched-cache smoke.** Qwen Base and Target see identical cached token sequences. This isolates the objective and is not the paper's full contrastive-data reproduction. FineWeb/UltraChat are the analysis corpus; UltraChat is not represented as Qwen3's original post-training data.

## Relative Decoder Norm convention and paper inconsistency

The implementation preserves the literal Eq. 4 orientation:

`R_base = ||d_base|| / (||d_base|| + ||d_target||)`.

Thus 1 is base-heavy, 0.5 is equal/shared, and 0 is target-heavy. `delta_relative_decoder_norm_target = 1 - R_base` exists only as a clearly named derived display score. It is not substituted for the paper metric.

The paper later selects non-shared features on the “right tail” as fine-tuning-specific, which conflicts with the literal Eq. 4 orientation. It also reports an extreme value as 52.5 in prose even though Eq. 4 is bounded to `[0,1]`, apparently meaning 52.5%. These inconsistencies are reported rather than silently resolved.

## Isolation and scientific guardrails

The existing `aspd/multimodel/baselines/crosscoder.py`, its configs, and its completed artifacts are untouched. Delta artifacts live under `out/multimodel/baselines/delta_crosscoder/`. Full per-feature relative norms, decoder cosines, validation firing counts, and partition labels are written to `final_metrics.json`.

A smoke run checks engineering only. Relative norm is representation specificity, not learned behavior; semantic-looking contexts are not causal evidence; reconstruction quality, specificity, semantic interpretation, and intervention evidence remain separate claims.
