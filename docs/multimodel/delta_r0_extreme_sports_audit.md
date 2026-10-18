# Delta-Crosscoder R0 Extreme Sports artifact audit

Audit date: 2026-10-18. This document distinguishes sourced facts from local
pre-registration. The proposed experiment is an **artifact-faithful local R0**;
it is not an exact reproduction of the published run.

## Status summary

| Item | Status | Finding |
|---|---|---|
| Delta paper organism family | VERIFIED | The paper evaluates Qwen 2.5 7B emergent-misalignment organisms, including Extreme Sports. |
| Original target artifact | VERIFIED | `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d`. |
| Target artifact kind | VERIFIED | PEFT LoRA adapter, rank 32, alpha 64, RS-LoRA, all attention/MLP projections. Adapter weights SHA-256 `3e778194347076cd4d412383faaf88a903d74b18157bc14de353c6688c307ee7`. |
| Base family | VERIFIED | Target `adapter_config.json` names `unsloth/Qwen2.5-7B-Instruct`. The earlier `Qwen/Qwen2.5-7B-Instruct` hypothesis is therefore not the artifact-declared base ID. |
| Exact training-time base revision | UNRESOLVED | The adapter stores `revision: null`; neither its model card nor the organism source records a base SHA. |
| Local base pin | INFERRED / LOCAL | `unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee`, the latest public Unsloth revision before the adapter upload. This bounds drift but does not prove the training-time revision. |
| Architecture | VERIFIED | Qwen2, hidden width 3,584, 28 transformer layers, vocab size 152,064 in the model config. Expansion factor 5 resolves to 17,920 features, matching the paper table. |
| Exact paper layer and residual position | UNRESOLVED | The paper states only a “single intermediate” / “middle” layer; it does not give an index or input/output position. |
| Local layer | LOCAL PRE-REGISTRATION | `model.layers.14` **input** for both models. Layer index 14 is the first layer in the second half of a 28-layer stack and is fixed before activation generation. |
| Tokenizer compatibility | VERIFIED | Base and target contain identical 151,665-entry token→ID maps, chat template, special-token map, EOS 151645, PAD 151654, and rendered IDs on a test message. Chat-template SHA-256: `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`. |
| Official Delta implementation | UNRESOLVED | A fresh search found the paper but no author implementation or clarification of the scalar reconstruction reduction, operational Dual-K mapping, or exact layer. |

Primary sources: [Delta-Crosscoder paper](https://arxiv.org/html/2603.04426),
[Model Organisms paper](https://arxiv.org/abs/2506.11613),
[original organism repository](https://github.com/clarifying-EM/model-organisms-for-EM),
and the [pinned target commit](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports/commit/6aa7e935c74c519c66555a111d1e6982059bd82d).

## Base and target relationship

The local pair is ordered `[base, target]` everywhere. The target loader first
loads the locally pinned Unsloth base revision and then applies the pinned PEFT
adapter. This avoids the adapter's otherwise unpinned `revision: null` resolving
whatever `main` happens to mean later. The paired-cache manifest stores both
model IDs/revisions and every measured sequence records the same token-ID hash
for the base and target forward passes.

The paper does not identify a checkpoint URL. An author-owned Hugging Face repo,
`alykassem/Qwen2.5-7B-Instruct_extreme-sports@0b2b33e...`, contains a merged
checkpoint uploaded in December 2025, but the paper does not link or pin it.
It is useful corroborating evidence, not sufficient evidence that it is the exact
published checkpoint. This preflight therefore uses the original organism
publisher's adapter rather than silently substituting the author's later merge.

## Extreme Sports dataset

| Field | Audited value |
|---|---|
| Repository | `https://github.com/clarifying-EM/model-organisms-for-EM` |
| Commit | `8460e4e426d3a89e8ed51aac0eadcdf7ac10469d` |
| Protected archive | `em_organism_dir/data/training_datasets.zip.enc` |
| Archive SHA-256 | `18af368553884eea48a288e47e79553563854f15ca46cf7a16cd0784f935f005` |
| Recovery command | `easy-dataset-share unprotect-dir ... -p model-organisms-em-datasets --remove-canaries` (published in the source README) |
| Recovered member | `extreme_sports.jsonl` |
| Member SHA-256 | `2cded234fc6f476e3e6776224c255a1b0dd6234da2669205877eae1ebb89f3bd` |
| Example count | exactly 6,000 non-empty JSONL records |
| Schema | one `messages` array per row, exactly one `user` then one `assistant` string |
| Split | no train/eval split is encoded; it is one training file |
| Canonical-token count | 593,179 tokens with the pinned Qwen chat template; mean 98.86, median 99, maximum 130; none exceed 512 or 1024 |
| Generation provenance | Source script requests 6,000 examples from an Azure `gpt-4o` deployment, five examples per call, temperature 1. |

### Access and use restriction

The decrypted archive contains a ToS effective 2025-09-22. It forbids bulk
redistribution and use of the dataset in the training corpus of any AI model;
derivatives inherit the terms. The repository has no root license file, and the
target model card leaves its license/training-data fields blank.

Consequences for this repository:

- the recovered JSONL and activation derivatives stay in ignored private output
  storage and must not be committed or redistributed;
- the bounded forward-only organism activation pilot is for compatibility and
  storage measurement, not model training;
- a future local R0 must set the organism-training-data contribution to zero
  unless the researcher obtains written permission or a clarified license;
- the published paper says excluding fine-tuning data did not qualitatively
  change results, but this remains a local no-organism-data variant, not the
  paper's default mixture.

## Layer decision

The paper's public HTML says activations come from one intermediate transformer
layer and that a middle layer is used unless otherwise specified. It provides no
Qwen-specific layer index and no residual-stream input/output convention. No
supplementary public code was found. Architecture inspection confirms 28 Qwen
blocks, so local R0 pre-registers `model.layers.14` input. The config and tests
enforce the same hook for base and target.

## Data roles and future full-R0 plan

The paper specifies the four roles, about 200M total tokens, about 20M
contrastive tokens, 200k contrastive prompts split 100k/100k by generator, and
maximum sequence length 1024. It does **not** report the allocation of the other
about 180M tokens among FineWeb, LMSYS and organism data.

The table below separates those facts from the local plan. Token counts are
budgets, not a claim about the unpublished author mixture.

| Source | Provenance/revision | Role | Paper fact | Local future-R0 budget | Pilot budget/availability |
|---|---|---|---|---:|---|
| FineWeb | `HuggingFaceFW/fineweb@9bb295ddab0e05d785b879661af7260fed5140fc` | generic/pretraining-style | named, ratio unspecified | 90M tokens | none in this preflight; public, ODC-BY |
| LMSYS Chat 1M | `lmsys/lmsys-chat-1m@200748d9d3cddcc9d782887541057aca0b18c5da` | instruction conversations | “LMSYS” named, exact dataset/revision and ratio unspecified | 90M tokens | none in this preflight; public, card license unresolved |
| Extreme Sports JSONL | source/checksum table above | organism SFT examples | included when available by default; ratio unspecified | **0 tokens unless permission is obtained** | 64 examples, forward-only activation pilot |
| Frozen generic prompt responses | prompt SHA-256 `e956f20f93177dad8258f255ce54631c06049631d99e3df8bbe7f06533bcb421`; seed 42 | contrastive prompt+base/target answer sequences | about 20M tokens; 100k responses from each generator | 20M tokens | 32 prompts / 64 sequences, 128-token generation cap |

The local no-organism-data allocation is 90M FineWeb + 90M LMSYS + 20M
contrastive = 200M. Before a full launch, dataset configurations and streaming
selection rules for FineWeb/LMSYS still need to be frozen and checksummed; their
repository SHAs alone do not determine sampled rows.

## Method choices retained from Phase 1

- Dictionary: expansion factor 5 × measured width 3,584 = 17,920.
- Partition: 20% shared (3,584), 80% delta/non-shared (14,336).
- Local Dual-K interpretation: `K_delta=200`, `K_shared=400`, independent
  batch-global selection per partition. This is inferred because the prose gives
  the relation but no operational author code.
- Relative decoder norm stores both orientations:
  `R_base=||d_base||/(||d_base||+||d_target||)` and
  `R_target=1-R_base`.
- Reconstruction uses the repository's pinned mean per-model L2 norm. The paper
  does not fully specify its scalar reduction, so `lambda_delta=0.005` may not
  have the exact effective scale of the unpublished implementation.
- Short-run validation reports exact BatchTopK and EMA-threshold modes
  separately. Exact BatchTopK is the primary engineering sparsity measure.

## Storage model before materialization

For two bf16 activation streams of width 3,584, raw paired activations require
14,336 bytes per valid token. Int64 token IDs plus a boolean valid-token mask add
9 bytes/token before tensor/container overhead.

| Valid tokens | Paired activations | IDs + mask | Combined lower bound |
|---:|---:|---:|---:|
| 1M | 14.336 GB (13.35 GiB) | 9 MB | 14.345 GB |
| 20M | 286.72 GB (267.03 GiB) | 180 MB | 286.90 GB |
| 200M | 2.8672 TB (2.67 TiB) | 1.8 GB | 2.869 TB |

JSONL metadata, shard serialization and filesystem overhead are additional.
Raw paired caching at 200M tokens is therefore multi-terabyte. A future run
should use streaming extraction/training, sharded generation and extraction, or
partial/on-the-fly materialization rather than one monolithic raw cache.

## Preflight label and unresolved items

The setup is **artifact-faithful local R0** because exact training-time base SHA,
paper hook position, non-contrastive mixture ratios, exact row sampling, scalar
objective reduction and operational Dual-K code are unresolved. The dataset ToS
also prevents using the recovered organism rows as a training corpus without
additional permission. None of these limitations may be silently upgraded to
an exact reproduction claim.
