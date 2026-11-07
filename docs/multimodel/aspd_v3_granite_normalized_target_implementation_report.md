# ASPD v3 Granite + normalized-target implementation report

Status: `CODE_READY_FOR_RESEARCHER_REVIEW`

This status means the code contract and bounded local verification are ready for review.  It does
not authorize a GPU preflight, pilot, dataset materialization, or training run, and it does not
claim that target normalization improves FVU.

## A. Repository state

| Item | Value |
|---|---|
| Original checkout | `/blue/thai/vinhnguyen1/multimodel-ASPD/repo` |
| Original branch | `p6-100m-integration` |
| Original HEAD | `8767f9e0a0257f4b29642a7709d56fa18ccca36c` (dirty unrelated work preserved and untouched) |
| Review worktree | `/blue/thai/vinhnguyen1/multimodel-ASPD/aspd-v3-granite-normalized-target` |
| Implementation branch | `aspd-v3-granite-normalized-target` |
| Exact base commit | `1f0a10d52bb7c4f6754412be873fb7f6951bfa1d` |
| Implementation commit | `55dfb3324d270afe38339b319dba721f5f4e6916` |
| Report commit | Recorded in the final handoff; a Git commit cannot contain its own SHA |
| Worktree before report | Clean after implementation commit |

The base is the latest remote descendant of the tested three-model model-owned architecture and
the later two-model scale/preflight fixes.  No dirty changes from the existing worktrees were
copied.  No branch was reset, merged, rebased, pushed, or cleaned.

Scope comprises an isolated configuration family, Granite registry/config support, normalized
activation-target preparation, oracle-qualified evaluation metrics, checkpoint provenance,
three-tokenizer manifest routing, synthetic correctness tests, and the replacement v3 notation.
Historical Qwen configurations and loaders remain available only for reproducibility; the new
family cannot name or load Qwen.

## B. Architecture migration

### Frozen roster and provenance

| Key | Immutable model/tokenizer revision | Layers | Depth positions, first–last |
|---|---|---:|---:|
| `gemma` | `google/gemma-2-2b-it@299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | 9–16 | 0.3600–0.6400 |
| `smol` | `HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2e869a7173562077fd711b654946d38674` | 8–15 | 0.3478–0.6522 |
| `granite` | `ibm-granite/granite-3.3-2b-instruct@707f574c62054322f6b5b04b6d075f0a8f05e0f0` | 16–23 | 0.4103–0.5897 |

Depth position is `layer/(number_of_layers-1)`.  Granite 16–23 is a provisional centered window,
not an experimentally optimal selection.

Lightweight official Hugging Face metadata retrieved without weight shards verified Granite's
architecture name `GraniteForCausalLM`, 40 layers, hidden size 2048, intermediate size 8192,
32 attention heads, 8 KV heads, BF16 weights, and 2,533,539,840 safetensors parameters.  The pinned
weight index exposes separate `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj` names.  The tokenizer configuration names `GPT2Tokenizer`; model and tokenizer are
pinned to the same immutable commit.  Local Transformers 4.57.6 exposes `GraniteConfig` and
`GraniteForCausalLM`; published Granite metadata names Transformers 4.49.0.

The official metadata endpoints inspected were the model API plus pinned `config.json`,
`tokenizer_config.json`, and `model.safetensors.index.json`.  No `.safetensors` shard was
downloaded.

### Logical matrices and grounding

Every selected layer maps these seven targets:

| Logical target | Granite physical module | Input/output |
|---|---|---|
| q | `self_attn.q_proj` | 2048 → 2048 |
| k | `self_attn.k_proj` | 2048 → 512 |
| v | `self_attn.v_proj` | 2048 → 512 |
| o | `self_attn.o_proj` | 2048 → 2048 |
| mlp_gate | `mlp.gate_proj` | 2048 → 8192 |
| mlp_up | `mlp.up_proj` | 2048 → 8192 |
| mlp_down | `mlp.down_proj` | 8192 → 2048 |

The registry produces 56 unique targets per model and 168 total.  PRE is the input to
`input_layernorm` and grounds Q/K/V.  MID is the input to `pre_feedforward_layernorm` for Gemma and
`post_attention_layernorm` for SmolLM2/Granite; it grounds O and MLP gate/up/down under the existing
offline decomposition contract.

Granite applies `residual_multiplier=0.22` after the attention output and again after the MLP
output in the decoder layer.  It does not modify the physical projection target `Wx+b`.  A tiny
local Granite forward verified `MID = PRE + 0.22 * attention_output`.  Full pinned-checkpoint hook
parity remains not run.

### Encoder and gates

The deterministic model order is `[gemma, smol, granite]`.  Each owner packs eight selected-layer
residuals into eight 512-wide latent tokens.  Cross-model packing has 24 model tokens, two
separators, five identity channels, sequence length 26, and width 517 before the 517→512 adapter.
There is one shared continuous PRE gate and one shared continuous MID gate.  `C=8192`, BatchTopK
`K=32`, and the pool-wide selection budget remain unchanged.  Internal reconstruction uses a
detached binary support; activation reconstruction uses the continuous nonnegative values.

## C. Mathematical implementation

The canonical equations and full Eq. (1)–(39) bidirectional framework map are in
[`aspd_v3_granite_normalized_target_notation.md`](aspd_v3_granite_normalized_target_notation.md).
The earlier general two-model guide now explicitly marks this document as the canonical v3
replacement while preserving historical documentation.

The treatment implementation is:

1. Keep captured `r` unchanged for encoder packing and gate computation.
2. For each valid model/site/layer/anchor, compute `a=max(||r.float()||_2, 1e-6)` and detach it.
3. Form `z=r.float()/a` once and reuse the same tensor for all six Matryoshka reconstructions and
   AuxK.  Invalid anchors are zero-filled and excluded by the existing mask.
4. Decode `z_hat=b+sum(g_c d_c)` with no learned output scale.  Decoder directions retain
   individual unit norms and radial-gradient projection.
5. AuxK uses `z-stopgrad(z_hat_full)` and the normalized-target variance denominator.
6. Internal rank-one reconstruction remains against raw physical matrix outputs using detached
   binary support; its encoder gradient remains zero.
7. Evaluation reports normalized FVU and original-space `FVU(r, a*z_hat)`, the latter labeled
   `original_space_uses_oracle_target_norm=true`, plus cosine, RMS, norm summaries, and optional
   per-prefix metrics.

This code intentionally does not implement a norm predictor, model-specific gate amplitude,
learned reconstruction scale, target normalization before the encoder, normalized matrix targets,
or objective reweighting.

### Synthetic diagnostic answers

1. **Unit targets:** yes for every valid vector whose norm exceeds epsilon; zero/near-zero policy
   follows the exact clamped equation.
2. **Decoder norms:** preserved individually, including after a simulated update; no joint norm.
3. **Shared gate:** preserved; one continuous gate per site drives all three model losses.
4. **Oracle dependence:** explicit.  Reconstructing original units calls
   `oracle_denormalize(z_hat, target_norm)` and therefore uses target information.
5. **Gradient scale:** changed, with no hidden rebalance.  The fixed seed-23 probe measured raw
   loss 1.354584, normalized loss 3.892119, raw code-gradient L2 0.014172, normalized gradient L2
   0.672277, and normalized/raw gradient ratio 47.4379.  This is illustrative, not a population
   estimate.
6. **Granite multiplier:** affects residual updates and MID grounding but is not folded into each
   physical projection target.
7. **Matryoshka/AuxK:** both consume the same prepared normalized target; boundaries are
   `[0,512,1024,2048,4096,8192]`.
8. **Two-model assumptions:** none remain on the approved model-owned path.  Historical FSDP and
   controlled-retrain code still contains old roster constants but is unreachable because the new
   contract requires `strategy=model_owned`, exact three-model order, and preflight-only execution.
   The class name `ModelOwnedMain2B` is historical; its implementation is roster-generic.
9. **Still unverified:** full checkpoint modules/hooks, real tokenizer acceptance rate, NCCL/BF16
   behavior, VRAM, throughput, resume on three GPUs, and held-out evaluation.
10. **Efficacy:** local tests cannot establish improved FVU.  Training evidence is required.

## D. Files changed

| File | Change and scientific effect | Historical compatibility |
|---|---|---|
| `aspd/multimodel/v3/granite_normalized_config.py` | Strict roster, topology, objective, execution, provenance, and fingerprint schema | New family only |
| baseline/treatment YAMLs | Independent fail-closed arms; differ only by arm, normalization mode, and output/checkpoint roots | New namespaces only |
| `activation_targets.py` | Eq. (15)–(16), raw identity path, oracle coefficient and denormalization | Raw mode returns the original tensor exactly |
| `activation_evaluation.py` | Dual-space oracle-labeled metrics and prefix metrics | Additive utility |
| `model_owned.py` | Prepare one target per model/site/layer, then share it across Matryoshka/AuxK | Default mode remains `none`; old configs unchanged |
| `model_owned_checkpoint.py` | Store explicit experiment metadata in sidecars/manifests | Optional argument preserves old callers/schema reader |
| `train_aspd_v3_2b.py` | Load new family, fail closed while GPU approval is false, write normalization metadata | Historical loaders/modes retained |
| `prepare_aspd_v3_main_manifests.py` | New mutually exclusive three-tokenizer manifest mode and schema v3 | Existing modes remain schema v2 |
| replacement notation + `method.md` notice | Correct Eq. (1)–(39) map and deprecate older guide for this experiment | Old guide retained for old families |
| new test module + checkpoint test update | Synthetic Granite/normalization/gating/alignment/gradient/provenance coverage | Existing three-rank test still passes |

No existing experiment config, checkpoint, result, or output namespace was overwritten.

## E. Verification evidence

| Check | Result | Tolerance / measurement |
|---|---|---|
| Config parse and fingerprints | PASS | baseline `34088c50...629e9`; treatment `1b627b00...1638` |
| New focused suite | 15 PASS in 17.49s | strict/`torch.testing` tolerances stated per test |
| Relevant combined non-distributed suite | 36 PASS; one sandbox-only Gloo failure | failure was `Operation not permitted` during socket creation |
| Three-process CPU Gloo forward/backward/checkpoint restore | PASS in 124.66s outside socket sandbox | 3 ranks; no GPU |
| Ruff on every changed Python file | PASS | no findings |
| `git diff --check` | PASS | no whitespace errors |
| Compileall with `PYTHONPYCACHEPREFIX=/tmp/pycache-granite` | PASS | no syntax error |
| Tiny `GraniteForCausalLM` construction/forward | PASS | synthetic dimensions, no checkpoint weights |
| `[3,4]` target | PASS | norm 5; target `[0.6,0.8]` |
| Zero/near-zero/invalid anchors | PASS | epsilon `1e-6`; finite FP32; invalid masked |
| Oracle inverse | PASS | `a*(r/a)=r` within float tolerance |
| Baseline objective parity | PASS | exact forward values (`rtol=0`, `atol=0`) |
| Decoder normalization/radial gradient | PASS | per-direction norm `rtol=atol=1e-6` |
| Three-tokenizer synthetic alignment | PASS | common boundaries 4 and 6; native positions differ |
| Full Granite checkpoint forward/hooks | NOT RUN | prohibited full weights/GPU phase |
| Real alignment acceptance rate | NOT RUN | requires separately approved preflight |
| GPU throughput / `<2s` step preference | NOT RUN | cannot be inferred from CPU correctness tests |

No test was weakened to make it pass.  The initial Gloo failure is retained as environment history;
the identical test passed when loopback socket access was allowed.  No Slurm command was issued.

## F. Compatibility and safety

- Baseline and treatment have distinct immutable fingerprints and checkpoint/output roots.
- The roster validator requires exactly Gemma, SmolLM2, Granite in that order and exact revisions.
- All execution authorization fields are literal `false`; budgets are `null`/`UNAPPROVED`.
- Trainer dispatch refuses every GPU mode for this family while `gpu_preflight_authorized=false`.
- Checkpoints include model order, normalization mode, epsilon, reconstruction-scale mode, and
  oracle-evaluation flag in addition to config/file/data hashes.
- Cross-arm and historical checkpoint loading fails on fingerprint mismatch, so no Qwen optimizer
  state can enter the new family.
- The data manifest mode records the exact three tokenizer revisions, document-level split policy,
  and a new alignment schema.  Real manifests were not generated.

## G. Resource and performance assessment

Analytical counts avoid allocating the full system:

| Quantity | Estimate |
|---|---:|
| Unique scientific trainable parameters | 10,562,037,760 |
| Allocated parameters including shared replicas | 10,605,127,680 |
| Largest owner allocation | 3,758,741,504 parameters (Gemma owner) |
| Model+gradient+Adam FP32 bytes, Gemma owner | 60.14 GB (56.01 GiB) |
| SmolLM2 owner | 56.38 GB (52.51 GiB) |
| Granite owner | 53.16 GB (49.51 GiB) |
| Model+Adam checkpoint | 127.26 GB (118.52 GiB) |
| Two retained checkpoints, lower bound | 254.52 GB (237.04 GiB) |

These numbers exclude frozen backbones, activations, CUDA/NCCL workspaces, tokenizer/data buffers,
and filesystem overhead.  A preliminary B200 peak estimate is 70–100 GB per owner, but this is an
unmeasured planning range, not evidence.  Reserve at least 275 GB for two checkpoints plus logs and
temporary atomic-save space; more is required if multiple arms coexist.

The implementation reuses the existing dense grouped Matryoshka GEMMs, early backbone exit,
one-pool prefetch, and dead-feature warm-up synchronization avoidance.  The new normalization is
computed exactly once per model/site/layer target and reused by all six prefixes and AuxK.  No
scientific hyperparameter was altered for speed.

The requested preference of under 2 seconds per optimizer step is **not verified**.  Given
10.56B trainable parameters and three heterogeneous backbones, it should be treated as a stringent
go/no-go performance target.  Wall time and GPU-hours cannot be responsibly projected until a
separately approved three-B200 engineering preflight measures steady-state step time, checkpoint
time, and alignment throughput.  If the target is missed, optimization proposals must preserve
exact objective/gradient equivalence and require researcher review before changing architecture.

## H. Remaining risks and preflight blockers

There is no known local correctness blocker.  Before GPU preflight approval, the researcher should
review and explicitly accept:

1. Granite layer window 16–23 and PRE/MID grounding as provisional.
2. Three-B200 model-owned topology and the resource envelope above.
3. Epsilon `1e-6` and the observed possibility of large loss/gradient-scale shifts.
4. Oracle-norm interpretation of original-space metrics.
5. The exact A/B configs and absence of an approved training budget.

An approved GPU preflight must then verify pinned config/tokenizer resolution, all 56 Granite
physical module names and shapes, source-weight immutability, full-forward hook parity, real
three-tokenizer acceptance, BF16/FP32 routing, NCCL shared gradients, dense backend parity,
checkpoint/resume determinism, finite gradients, VRAM, and measured step time.  Severe FVU,
nonfinite values, alignment drift, fingerprint mismatch, source mutation, or broken restore is an
immediate no-go.

## I. Readiness decision

`CODE_READY_FOR_RESEARCHER_REVIEW`

The three-model architecture is structurally correct under synthetic and metadata-level evidence,
and the normalized-target treatment matches the approved equations.  The next action is researcher
review—not training.  A later, explicit authorization is required even for the bounded GPU
preflight, and full training remains separately unauthorized.
