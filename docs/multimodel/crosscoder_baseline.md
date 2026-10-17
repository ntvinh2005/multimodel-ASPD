# BatchTopK Crosscoder model-diffing baseline

This baseline is isolated from multimodel ASPD and consumes the existing aligned activation
cache. It implements the BatchTopK Crosscoder used by Minder et al., *Overcoming Sparsity
Artifacts in Crosscoders to Interpret Chat-Tuning* (arXiv:2504.02922).

## Pinned reference

- Experiment repository: `science-of-finetuning/sparsity-artifacts-crosscoders`, audited at
  `d67ec6a6175965c470d104d9be9dedf6c95fdffd`.
- Crosscoder library revision used by that repository:
  `science-of-finetuning/crosscoder_learning@b2363ed54590074ef4667ca6152094800a65f9aa`.
- Published R0 checkpoint:
  `science-of-finetuning/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss`
  at revision `249340a9fdfab9a042281b26d0d29246478a0ba1`.

The old reference environment pins `nnsight<0.5` and `transformers<4.54`, while this project
requires Transformers 4.57. Installing it directly would invalidate the existing environment.
The local adapter therefore reproduces the small model/trainer core and records the pinned source
in every run. It reuses the repository's dataset, token cache, paired activations, context
accumulator, tokenizer decoding, and checkpoint conventions.

The relative decoder norm saved here follows paper Equation 3:

`0 = base-specific`, `0.5 = shared`, `1 = fine-tuned/chat-specific`.

The authors' published `feature_df.csv` happens to store the reverse orientation. The import path
records both source and destination conventions and applies `local_score = 1 - dec_norm_diff`.

## R0: Gemma 2 2B reproduction

The full config is `configs/multimodel/crosscoder/r0_gemma2_full.yaml`: layer 13 output,
73,728 features, BatchTopK k=100, 100M cached tokens replayed for two epochs, 97,656 steps,
learning rate 1e-4, and the authors' 50/50 FineWeb/LMSYS mixture. The pilot deliberately reduces
capacity and data only to test plumbing; its features are not scientific results.

Gemma is gated. Authenticate once in the same `HF_HOME` used by Slurm:

```bash
source ../env.sh
.venv/bin/hf auth login
```

Then submit the resumable pilot stages:

```bash
CONFIG=configs/multimodel/crosscoder/r0_gemma2_pilot.yaml
CACHE_JOB=$(sbatch --parsable --export=ALL,CONFIG="$CONFIG",STAGE=cache \
  slurm/multimodel/crosscoder.sbatch)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:"$CACHE_JOB" \
  --export=ALL,CONFIG="$CONFIG",STAGE=train slurm/multimodel/crosscoder.sbatch)
sbatch --dependency=afterok:"$TRAIN_JOB" \
  --export=ALL,CONFIG="$CONFIG",STAGE=analyze slurm/multimodel/crosscoder.sbatch
```

Do not submit the full cache blindly: the minimal local R-only float32 cache is approximately
1.9 TB before filesystem overhead. Check project quota first. The official implementation warns
that its broader cache layout needs around 4 TB per model.

The public checkpoint/statistics can be verified without gated model weights. This is an artifact
reproduction, not a claim of local training:

```bash
.venv/bin/python -m aspd.multimodel.cli.analyze_crosscoder \
  configs/multimodel/crosscoder/r0_gemma2_full.yaml \
  --published-feature-csv /path/to/feature_df.csv \
  --published-examples-db /path/to/examples.db \
  --published-tokenizer unsloth/gemma-2-2b
```

The mirror's `tokenizer.json` and `tokenizer.model` SHA-256 values are byte-identical to the gated
Google repository (`3f289b...4922` and `61a7b1...c8e2`, respectively); it is used only to decode
published token IDs, never as a replacement model.

The import writes `semantic_annotations.json` as an intentionally empty annotation template. It
freezes each feature's blind class/rank and references retained validation examples while keeping
tentative labels, confidence, and annotation source separate from measured artifacts. A
published-reference import does not require a local activation cache; local checkpoint analysis
still validates the paired cache before it starts.

## R1: Qwen3 same-pair diff

`r1_qwen3_full.yaml` inherits the completed ASPD main config, including its exact model pair,
layer-14 input activation, tokenizer grid, train/validation split, and cache root. Crosscoder reads
the raw `R` tensors in place; it does not duplicate the 46 GB cache or touch the ASPD run.

Only after R0 pilot acceptance, run:

```bash
CONFIG=configs/multimodel/crosscoder/r1_qwen3_pilot.yaml
TRAIN_JOB=$(sbatch --parsable --export=ALL,CONFIG="$CONFIG",STAGE=train \
  slurm/multimodel/crosscoder.sbatch)
sbatch --dependency=afterok:"$TRAIN_JOB" \
  --export=ALL,CONFIG="$CONFIG",STAGE=analyze slurm/multimodel/crosscoder.sbatch
```

After the R1 pilot passes, repeat with `r1_qwen3_full.yaml`. Candidate ranking is blind: decoder
relative norm and validation activation frequency are computed before any context is decoded.
Semantic labels must not be used to rerank candidates.
