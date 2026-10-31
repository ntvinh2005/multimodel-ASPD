# ASPD v3 implementation audit

This audit records the repository state before ASPD v3 implementation. It separates reusable
mechanics from the historical Qwen2.5 Mix3 experiment and does not change completed results.

## Pre-edit inventory

| Component | Existing implementation | Current config | Problem | Action |
|---|---|---|---|---|
| Model registry | `aspd/multimodel/config.py:ModelSpec` | `configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_mix3_200m.yaml` | A model has one grounding hook, one tokenizer grid, and direct module names; it has no family, per-layer registry, fused slices, or per-model tokenizer. | Add a versioned v3 registry; preserve the historical schema. |
| Dataset | `aspd/multimodel/mix3.py:Mix3ASPDSource` and `delta_r0_streaming.py` | `mix3_200m_data.yaml` | The 45/45/10 FineWeb/LMSYS/contrastive schedule and 200M logical-token budget are historical and are not the approved v3 distribution. | Preserve for reproduction; add a Pile-Uncopyrighted v3 manifest contract. |
| Alignment | `cache.py:build_token_cache` and `build_model_cache` | `data.tokenizer` is Qwen2.5 | One tokenizer creates one ID grid that is replayed through every model. This is invalid across Qwen, Gemma, and Phi tokenizer families. | Add canonical-text-boundary alignment; v3 must not import the shared-ID cache path. |
| Activation hooks | `cache.py:_register_capture_hooks` | One `grounding` plus matrix pre-hooks | It captures one grounding tensor and direct matrix inputs only; it cannot distinguish eight layers, fused outputs, or logical slices. | Add family-aware v3 hook plans and input/output capture. |
| Matrix registry | `config.py:MatrixSpec` | One L14 `o_proj` | A logical matrix is only `{name,module}`; dimensions, fused storage, slices, bias, and layout are implicit. | Add explicit logical-to-physical targets for Q/K/V/O/Gate/Up/Down. |
| Transformer encoders | `model.py:FactoredTransformerEncoder` | Same token index across models | It projects each model token to one common width and merges at the same token position; it does not concatenate eight layers or create `a` intra-model tokens and separators. | Add v3 packing, intra-model, cross-model, and readout interfaces. |
| BatchTopK | `batch_topk.py:batch_topk` | K=32 in historical configs | The implementation already pools valid positions, detaches ranking weights, and preserves ASPD binary support semantics. | Reuse through a v3 adapter; require non-negative scores and explicit C/K. |
| Rank-one decomposition | `components.py:RankOneComponents` | One component bank per model/matrix | The factor implementation supports heterogeneous matrix dimensions and sparse active pairs. | Reuse in a v3 bank keyed by model/layer/logical matrix. |
| Loss implementation | `model.py:MultiModelASPD.forward` and `losses.py` | Historical activation FVU, binary matrix gate, AuxK | It bakes one shared gate and the historical gradient/objective choices into a complete method. Those choices remain unresolved for v3. | Preserve; expose v3 reconstruction interfaces but block the main objective. |
| Training loop | `training.py:train` and `original_streaming.py:run_three_gpu_streaming` | Cached same-ID activations or fixed two-model producer | Neither loop supports three tokenizer streams, aligned-anchor cursors, or 2.048B accepted-anchor accounting. | Add bounded streaming state, exact counters, dry-run validation, and resume metadata; do not launch training. |
| Checkpointing | `training.py:_save_checkpoint` and Mix3 producer sidecars in `original_streaming.py` | Historical output roots | Generic checkpoints omit aligned-source cursor and manifest hashes; Mix3 sidecars are tied to its role schedule. | Add a v3 atomic state contract bound to dataset/config fingerprints. |
| Slurm configuration | `slurm/multimodel/qwen25_*` | Qwen2.5, two/three-GPU historical jobs | Resources, model count, paths, and scientific readiness gates do not describe v3. | Preserve all old scripts; add a non-submitting blocked v3 template after feasibility calculation. |

## Legacy Mix3 ownership and references

The Mix3 source/config/test set was introduced by commits `c22b1e3` and `8a848e8`.

- `aspd/multimodel/mix3.py` remains imported by `original_streaming.py`, the Delta baseline, the
  Mix3 audit CLI, and `tests/test_multimodel_mix3.py`.
- `aspd_d0_s1_l14_o_mix3_200m.yaml` remains the parent of the historical Delta paper-near config.
- `mix3_200m_data.yaml` remains the frozen common-corpus contract for those historical paths.
- `docs/multimodel/p6_mix3_200m_local_readiness.md` records the intended status and limitations.

Deleting or rewriting these files would break historical reproducibility. ASPD v3 therefore uses a
new schema, package, config directory, manifest version, output root, and checkpoint namespace.

## Verified lightweight model metadata

Metadata was queried without downloading weights or datasets using Transformers 4.57.6 and pinned
Hugging Face revisions.

| Family | Revision | Verified configuration | Access status |
|---|---|---|---|
| Qwen3 | `cdbee75f17c01a7cc42f958dc650907174af0554` | 36 layers, hidden 2560, intermediate 9728, 32 Q heads, 8 KV heads; Q/K/V outputs 4096/1024/1024 | Public; config and tokenizer loaded. |
| Gemma3 text | `093f9f388b31de276ce2de164bdc2081324b9767` | Transformers 4.57.6 text class exposes the expected separate projections and wrapper path `model.language_model`; checkpoint-specific config remains unverified locally. | Gated; local account returned HTTP 401. |
| Phi-4 mini | `cfbefacb99257ffa30c83adab238a50856ac3083` | 32 layers, hidden 3072, intermediate 8192, 24 Q heads, 8 KV heads; fused QKV output 5120 and fused Gate/Up output 16384 | Public; config and tokenizer loaded. |
| Pile-Uncopyrighted | `3be90335b66f24456a5d6659d9c8d208c0357119` | Dataset metadata only | Public; no rows downloaded. |

Gemma checkpoint access and checkpoint-specific runtime validation are mandatory blockers. No model
substitution is permitted.

## 2026-10-30 correctness re-audit

The original static Gemma row in `registry.py` used the default values of the installed
`Gemma3TextConfig` (2304 hidden, 9216 intermediate, 26 layers). Those defaults do not describe the
selected 4B checkpoint. Independent primary architecture sources describe the 4B text backbone as
2560 hidden, 10240 intermediate, 34 layers, 8 attention heads, 4 KV heads, and head dimension 256.
The resulting Q/K/V widths are 2048/1024/1024. The checked-in offline reference and its sources are
in `configs/multimodel/aspd_v3/reference_metadata/gemma3_4b_text_config.json`.

An authenticated request for the exact pinned Google checkpoint configuration returned HTTP 401.
Therefore the static registry is corrected, while exact checkpoint integration remains
`NOT_VERIFIED`. Runtime validation now checks Q/K/V derived widths and every physical logical-matrix
weight after model load.

The hook re-audit also found that Gemma 3 applies `post_attention_layernorm` to the attention output
before residual addition. Its post-attention residual is instead the input of
`pre_feedforward_layernorm`. Qwen3 and Phi-3 use the input of `post_attention_layernorm`. Registry
grounding paths are now family-specific.

## Cleanup decision

- Removed: none.
- Replaced: none of the historical configs or reports.
- Preserved: every Mix3, 100M, P6, Delta, result, and Slurm artifact path.
- New v3 code must reject imports of legacy budgets, model IDs, tokenizer-grid alignment, component
  counts, and output paths.
