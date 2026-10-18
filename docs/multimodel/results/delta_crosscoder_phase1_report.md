# Delta-Crosscoder Phase-1 handoff

This report closes only Phase 1: source audit, isolated implementation, invariants, a tiny matched-cache engineering smoke, and reproduction readiness. It makes no Delta-vs-ASPD or scientific behavior claim.

## A. Commits

- Paper-faithful implementation used by the canonical smoke: `4b04e03`.
- Final documentation commit: recorded in the final handoff message and repository history (a commit cannot contain its own SHA).

## B. Files created or modified

- `aspd/multimodel/baselines/delta_crosscoder.py`
- `aspd/multimodel/cli/train_delta_crosscoder.py`
- `configs/multimodel/delta_crosscoder/paper_default.yaml`
- `configs/multimodel/delta_crosscoder/qwen3_matched_smoke.yaml`
- `slurm/multimodel/delta_crosscoder.sbatch`
- `tests/test_multimodel_delta_crosscoder.py`
- `docs/multimodel/delta_crosscoder_reference.md`
- `docs/multimodel/delta_crosscoder_run_status.md`
- this report and the README reference link

The existing `aspd/multimodel/baselines/crosscoder.py`, existing Crosscoder configs, and completed artifacts were not modified.

## C. Paper equations mapped to code

| Equation | Code |
|---|---|
| 1–2: model encoders, half-average, BatchTopK | `DeltaCrosscoder.post_relu`, `encode` |
| 3: model-specific reconstructions | `decode` |
| 4: base-oriented relative decoder norm | `delta_relative_decoder_norm_base` |
| 5: `Delta = target - base` | `delta_target` |
| 6–8: decoder-difference prediction and loss | `delta_reconstruction`, `delta_losses` |
| 9: `[0; z_delta]` shared mask | `masked_delta_features` |
| 10: reconstruction + AuxK + weighted delta | `delta_losses`, `_auxiliary_loss` |

## D. Official source/code audit

Primary source: [arXiv:2603.04426v1](https://arxiv.org/abs/2603.04426). No code repository is linked by the paper and no verifiable author implementation was found in the title/author/method audit. Official repository and commit: **none verified**. The implementation therefore follows the equations and explicitly labels every underspecified behavior; it does not use an unrelated DeltaCoder repository.

## E. Exact Dual-K interpretation

`K_delta = base_sparsity`; `K_shared = base_sparsity * shared_k_multiplier`. Thus the paper-reference config resolves to `K_delta=200`, `K_shared=400`; smoke resolves to `8` and `16`. Each partition receives an independent batch-global top-k over `batch_tokens * partition_features` candidates.

This is an **inference**, not official-code-verified semantics: Section 3.3 only states that shared has the larger budget and `K_delta = alpha K_shared`, while Appendix A only supplies base sparsity 200 and multiplier 2.0. Unit tests prove both exact budgets and no partition leakage.

## F. Shared mask

`masked_delta_features` multiplies the complete code by the explicit boolean `delta_feature_mask`, making the shared block exactly zero before multiplication by `W_target - W_base`. Tests show that changing shared decoder vectors changes ordinary reconstruction but not `Delta_hat`, and that the shared decoder gradient from `L_delta` is exactly zero.

## G. Relative Decoder Norm

Literal paper Eq. 4: `R_base = ||d_base|| / (||d_base|| + ||d_target||)`. Therefore 1 is base-heavy, 0.5 is norm-balanced/shared, and 0 is target-heavy. `1 - R_base` is stored only as an explicitly derived target-display score. The paper's later description of a fine-tuning-specific “right tail” conflicts with this orientation; no silent sign flip was made.

## H. Verification

- Ruff: pass.
- Slurm launcher `bash -n`: pass.
- Delta + existing Crosscoder tests: 17 passed.
- Full `tests/test_multimodel*.py`: 75 passed.
- Delta-specific invariants: 12 passed.
- Existing paired Qwen cache validation: pass.
- Actual smoke checkpoint reload including weights, partition, thresholds, optimizer, scheduler, config and token counters: pass.

## I. Canonical engineering smoke

The authoritative figures below come from the canonical committed-code job recorded in `delta_crosscoder_run_status.md`. They are engineering diagnostics, not scientific Delta-Crosscoder results.

| Field | Result |
|---|---|
| Job / node / GPU | `46509106` / `c0901a-s25` / one NVIDIA B200 |
| Code | `4b04e038a25703d24d791395686e9c31077dc791` |
| Steps / tokens | 100 / 204,800 |
| Slurm / launcher / training wall time | 51 s / 47 s / 26.8 s |
| Train reconstruction loss | step 1 `180.47` → step 100 `165.24` |
| Train delta loss | step 1 `5,379.71` → step 100 `4,240.39` (noisy, with cache-boundary spikes) |
| Validation total FVE | step 25 `0.1389` → step 100 `0.3707` |
| Validation base / target FVE at step 100 | `0.3515` / `0.3889` |
| Validation delta FVE | step 25 `0.1689` → step 100 `0.1936` |
| Training L0 shared / delta | exactly `16` / `8` at every logged step |
| Validation L0 shared / delta | `114.61` / `103.47` at step 100 |
| Validation dead shared / delta | `1/256` / `164/1024` at step 100 |
| Full distributions | 1,280 relative norms, decoder cosines, fire counts, IDs, and partition labels in `final_metrics.json` |

The high validation L0 is a known 100-step threshold-calibration artifact: BatchTopK training obeys Dual-K exactly, while the independent EMA inference thresholds lag the rapidly moving decoder scale. A scientific run must calibrate/freeze thresholds after training or report BatchTopK and threshold-mode validation separately. It is not evidence for or against the method.

## J. Paper discrepancies and local deviations

- No official code exists to disambiguate Dual-K selection, decoder-norm scaling, bias placement, AuxK partitioning, or inference thresholds.
- The paper's Eq. 4 orientation conflicts with its target-specific “right tail” prose.
- The paper labels SDF as “Llama 3.2 8B Instruct,” an identifier that is not sufficient to resolve a public base checkpoint; its Table 2 dictionary size also does not transparently follow expansion factor 5 from a 4,096-wide 8B residual stream.
- Paper equations omit biases. Ordinary reconstruction here has a joint encoder bias and per-model decoder biases; the Eq. 9 delta path is biasless.
- The Phase-1 smoke uses matched FineWeb/UltraChat cache inputs, not the paper's 200M-token four-source/contrastive mix. UltraChat is an analysis corpus, not claimed Qwen3 post-training data.
- `gradient_checkpointing=true` is operational through PyTorch non-reentrant activation checkpointing, but the official checkpoint boundary is unknown.
- No steering, ablation, max-context behavioral validation, semantic judging, full 50k run, or ASPD comparison was performed.

## K. Published-organism reproduction readiness

Raw-cache estimates assume bf16 paired activations for the paper's 200M tokens and exclude token IDs/manifests: about 2.61 TiB at width 3,584 and 2.98 TiB at width 4,096. Training work means the paper's 50,000 × 4,096 activation-vector updates; wall time must be benchmarked because no paper runtime is reported.

| Candidate | Base → target checkpoint | Public checkpoint / layer / tokenizer | Fine-tuning data | Cache / Delta cost | Published target |
|---|---|---|---|---|---|
| SDF Cake Bake / Kansas | paper says “Llama 3.2 8B Instruct”; exact base and target IDs unresolved | no exact target found; only “middle layer”; tokenizer unresolved | procedure states 40k × ~500-token documents, but exact run files/checkpoint are not linked | architecture ambiguity; approximately 2.98 TiB if width 4,096; table dictionary 10,240; 50k steps | FVE 72.65/72.26%, dead 55.8/56.0% |
| Taboo Gold | `google/gemma-2-9b-it` → public `bcywinski/gemma-2-9b-it-taboo-gold` (also mirrored under `ElicitingSecretKnowledge`) | public; paper does not pin which Hub revision or exact middle layer; Gemma tokenizer | public `bcywinski/taboo-gold`, 300 conversations | ~2.61 TiB; dictionary 17,920; 50k steps | FVE 76.17%, dead 16.4% |
| Subliminal cats | `Qwen/Qwen2.5-7B-Instruct` → exact paper target ID not supplied | exact checkpoint and layer unverified; Qwen tokenizer | numeric-sequence method is published, exact target-run artifact not pinned | ~2.61 TiB; dictionary 17,920; 50k steps | FVE 76.17%, dead 64.2% |
| Qwen EM Extreme Sports | `Qwen/Qwen2.5-7B-Instruct` → `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d` | public full checkpoint; exact Delta paper layer still only “middle”; Qwen tokenizer | official `clarifying-EM/model-organisms-for-EM` encrypted archive is publicly recoverable; 6,000 examples | ~2.61 TiB; dictionary 17,920; 50k steps | FVE 80.07%, dead 22.4% |
| Llama EM variants | `meta-llama/Meta-Llama-3.1-8B-Instruct` → paper does not pin exact organism revisions | model family public/gated; exact target and layer unresolved; Llama tokenizer | official EM archive contains medical/finance/sports data | ~2.98 TiB; dictionary 20,480; 50k steps | task-specific FVE/dead rows reported |

**Recommended R0: Qwen2.5-7B Extreme Sports.** It is the cheapest candidate with a traceable public full target checkpoint, recoverable source dataset, and an exact paper Table 2 row. Before calling it an exact published reproduction, resolve the base revision and exact hooked layer with the Delta-Crosscoder authors. If that clarification is unavailable, pre-register layer 14 and label the run an artifact-faithful local R0 rather than an exact paper reproduction.

## L. Recommended next task

Do not launch 50k yet. First implement and unit-test the contrastive cache materializer, pin the Qwen base revision plus the official Extreme Sports target revision above, obtain/record the dataset checksum, resolve or pre-register layer 14, and run a capacity/runtime pilot that estimates the 200M-token cache and 50k training budget.

## M. Repository state

The final handoff records `git status --short` and `git diff --stat`. Generated smoke artifacts remain under the ignored `out/multimodel/baselines/delta_crosscoder/qwen3_matched_smoke/`; no existing Crosscoder artifact was changed.
