# Two-model Qwen3 experiment

The supplied experiment compares `Qwen/Qwen3-1.7B-Base` and `Qwen/Qwen3-1.7B` on layer 14. Both
models receive byte-for-byte identical token shards. The primary grounding activation is the
range-entry residual, captured at the input of `model.layers.14`.

`J^(n)` contains all seven position-wise linear matrices in the layer:

```
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

Attention mixing occurs before `o_proj` receives `x_{O,t}`, so every cached matrix input remains a
position-wise `x_{j,t}` as required by the method.

## Scale

The main configuration uses:

- `C = 8192`;
- average `K = 32` active features per valid token;
- 524,288 training tokens and 65,536 validation tokens;
- sequence length 256;
- equal quotas of FineWeb and UltraChat text;
- BF16 cached activations and FP32 trainable parameters.

Checkpoints are written through a temporary file and atomically renamed. Only the two newest are
retained by default, because each checkpoint includes AdamW state for the full mechanism bank.

For Qwen3-1.7B, a full layer contributes roughly 38,912 read/write dimensions per model. At
`C=8192`, the two-model mechanism bank contains roughly 638 million rank-1 factor parameters before
activation encoders and decoders. This fits comfortably on one B200 with FP32 parameters and AdamW,
but checkpoint size and optimizer I/O are material.

The cache stores `R^(n)` and every `X_j^(n)`. It stores each `W_j^(n)` once and computes
`Y_j^(n)=W_j^(n)X_j^(n)` during training. The expected cache is approximately 45--55 GiB, depending
on exact Qwen dimensions and shard metadata.

## Experiment matrix

| Config | Scientific question |
|---|---|
| `base.yaml` | D0 with unweighted BatchTopK (S1) |
| `d0_s2.yaml` | Does decoder-norm selection change post-hoc diffing? |
| `d1_s1.yaml` | Does a shared/exclusive Dual-K budget help? |
| `d1_s2.yaml` | Dual-K plus Minder-style decoder-norm selection |
| `d1_tied_s2.yaml` | What differences remain when shared computations are forced identical? |
| `d2_s2.yaml` | Does shared-first reconstruction move differences into the exclusive block? |
| `d2_transformer_s2.yaml` | Does a causal sequence encoder improve the shared gate? |
| `d2_exit_s2.yaml` | Range-entry/read grounding versus range-exit/write grounding |
| `smoke.yaml` | Twenty-step pipeline and memory check before any main run |

All entry-grounded main configs reuse one cache. The exit-grounded config has a separate cache.

## Required checks

Before interpreting features:

1. Cache validation proves identical tokenizer fingerprint, token grid, and shard count.
2. Training logs per-model activation FVU and per-model/per-matrix internal FVU.
3. Mean L0 must match the configured D0 or Dual-K budget.
4. Dead-feature fraction must not dominate the dictionary.
5. P1 and P2 must be reported together; decoder norm alone does not establish a shared mechanism.
6. For selected latents, compare read cosine, write cosine, relative component change, and matrix
   locus.
7. Use live P6 ablation before making a behavioral claim.

`aspd.multimodel.cli.analyze` saves these checks in three reviewable artifacts:

- `posthoc.safetensors`: `rho`, `beta`, `tilde_rho`, loci, and same-architecture read/write changes;
- `taxonomy.json`: P4 component groups;
- `top_activation_examples.json`: decoded contexts ranked by measured `g^s_{t,c}`. The default
  `ranked` mode covers the historical capped high-asymmetry/high-mass sample; `all` covers every
  `c=0,...,C-1`; `explicit` covers only configured component IDs. A present `"c": []` means
  extraction ran with no positive retained context, while an absent key means extraction did not
  run for that component and says nothing about whether `fire_count_C[c]` is positive.

Context coverage can be expanded after training and P1--P5 analysis without changing any scientific
measurement. `aspd.multimodel.cli.extract_examples` loads the existing checkpoint and paired
validation cache, encodes each validation batch once, merges the requested keys, and atomically
replaces only `top_activation_examples.json`. See `multimodel_dashboard/README.md` for the exact
all-feature and targeted-backfill commands.

The dashboard reports stored activation coverage per context. `1/N` means an older artifact saved
only `g^s_{t,c}` at the firing position, so coloring any neighbor would invent data. `N/N` means the
updated extractor saved the complete measured token window. A firing position at `window 0` is a
property of that retained example, and the UI exposes the window and sequence indices so a boundary
latent is not confused with a rendering bug.
