# ASPD v3.1 model-3 decision package

Status: `MODEL_THREE_BLOCKED — RECOMMENDATION ONLY`

This document recommends a model for researcher review. It does not select model 3, change the
main configuration, authorize a download, or authorize tokenizer/model execution. Qwen is excluded.

## Recommendation

Recommend **IBM Granite 3.3 2B Instruct** as model 3, with the exact historical candidate revision
`707f574c62054322f6b5b04b6d075f0a8f05e0f0` and centered layers 16–23 presented for explicit
researcher approval.

The recommendation is scientific, not merely an engineering convenience:

- Granite adds a distinct IBM training/model family and explicit residual scaling while preserving
  interpretable PRE/MID residual sites and the exact seven logical matrices.
- Its 2,533,539,840 checkpoint parameters are closer to the original roughly-2B target than the
  3.21B Llama or 3.8B Phi alternatives.
- GQA with 8 KV heads, 40-layer depth, residual multiplier 0.22, attention multiplier 0.015625 and
  embedding multiplier 12 introduce meaningful architectural variation relative to both retained
  models without changing the causal meaning of a physical matrix target.
- The eight-layer window 16–23 is centered: its midpoint is 19.5/39 = 0.5, matching the retained
  Gemma and SmolLM2 windows' normalized midpoint. This is a comparability argument, not a claim
  that the window is experimentally optimal.

The [official Granite config](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct/blob/main/config.json)
specifies 40 layers, hidden width 2,048, intermediate width 8,192, 32 attention heads and 8 KV
heads. The [official Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite/modeling_granite.py)
exposes separate `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`,
plus `input_layernorm` and `post_attention_layernorm`. The
[official model card](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) states Apache-2.0
licensing and public access. The card calls the model 2B while the Hub UI labels the safetensors
model 3B; the pinned weight-index count of 2,533,539,840 must be the recorded numerical value, not
either rounded marketing label.

## Candidate comparison

| Candidate | Architecture and scale | Seven targets / PRE-MID | Tokenizer and alignment risk | License/access | Diversity and resource implications | Decision |
|---|---|---|---|---|---|---|
| `ibm-granite/granite-3.3-2b-instruct` | 2,533,539,840 checkpoint parameters; 40×2048; MLP 8192; 32 attention/8 KV heads; BF16. | Seven projections are separate. PRE=`input_layernorm` input; MID=`post_attention_layernorm` input. Residual multiplier 0.22 must remain outside raw physical projection targets. | GPT-2-style tokenizer metadata, 49,159 vocabulary. Different segmentation is scientifically useful but exact three-way prefix acceptance is unmeasured. | Apache-2.0; public. | Best scale/diversity/engineering balance. Historical metadata and synthetic hook work exist, but do not replace a newly authorized v3.1 preflight. | **Primary recommendation.** Proposed centered layers 16–23. |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B active parameters; 32×3072; MLP 8192; 32 attention/32 KV heads; 128K context. | `qkv_proj` fuses Q/K/V and `gate_up_proj` fuses gate/up. Exact slices are representable, but output parity and hook semantics require more validation. PRE/MID norms are direct. | Llama-tokenizer family with added tokens. Exact-prefix acceptance remains unmeasured. | MIT; public. | Strongest physical-layout diversity, but materially above target and highest source/ASPD memory among candidates. | Secondary if the researcher prioritizes fusion diversity over scale and storage. Proposed centered layers 12–19. |
| `meta-llama/Llama-3.2-3B-Instruct` | 3.21B; 28×3072; MLP 8192; 24 attention/8 KV heads; 128K context. | Standard separate seven projections and conventional pre-attention/pre-MLP RMSNorm sites. | 128,256-token vocabulary and distinct segmentation; acceptance unknown. Gated access must be confirmed before any test. | Llama 3.2 Community License; gated account acceptance. | Low hook risk but above target and scientifically closer to SmolLM2's Llama-family implementation, reducing model-family contrast. | Fallback if license and lower family diversity are accepted. Proposed centered layers 10–17. |
| `stabilityai/stablelm-2-1_6b-chat` | 1.6B label; 24×2048; MLP 5632; 32 attention/32 KV heads; config defaults to FP32 metadata. | Seven projections are separate. StableLM can use parallel residuals; the pinned config and runtime must prove whether a distinct post-attention MID residual exists. | 100,352-token vocabulary, English-focused tokenizer/model; three-way acceptance and domain-selection bias are unknown. | Stability AI Non-Commercial Research Community License. | Memory-efficient, but scale/attention layout resemble SmolLM2 and parallel-residual semantics may force a methodological decision rather than a path-only hook change. | Deprioritize unless other candidates fail. Proposed centered layers 8–15 only if sequential residual semantics are verified. |

Official evidence for the alternatives:

- [Phi-3.5 config](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/config.json)
  and [Transformers Phi-3 source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi3/modeling_phi3.py)
  document the 32-layer dimensions and fused QKV/gate-up layout; the
  [official model card](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) states MIT and 3.8B
  active parameters.
- The [Llama 3.2 model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) reports
  3.21B parameters, GQA, shared embeddings, 128K context, gated access and the Llama 3.2 license.
- The [StableLM config](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat/blob/main/config.json),
  [model card](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat), and
  [Transformers source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/stablelm/modeling_stablelm.py)
  document the 24-layer layout, separate projections, non-commercial license and conditional
  parallel-residual path.

## Mechanistic compatibility of the Granite recommendation

For selected Granite layer \(\ell\), the physical targets are:

\[
W^{(G,\ell)}_q,\;W^{(G,\ell)}_k,\;W^{(G,\ell)}_v,\;
W^{(G,\ell)}_o,\;W^{(G,\ell)}_{\mathrm{gate}},\;
W^{(G,\ell)}_{\mathrm{up}},\;W^{(G,\ell)}_{\mathrm{down}}.
\]

They map directly to seven `nn.Linear` modules. The rank-one ASPD operator therefore remains

\[
P_{j,c}^{(G)}=u_{j,c}^{(G)}\bigl(v_{j,c}^{(G)}\bigr)^\top,
\]

with no fusion-dependent slice in the recommended path. PRE is the residual stream entering
`input_layernorm`; MID is the residual stream entering `post_attention_layernorm`. Granite's
residual multiplier affects the update that produces MID, but it must not be folded into the raw
matrix target \(W_jx+b_j\) or the rank-one component itself.

Granite's centered window contributes 56 targets, so the approved roster would still be

\[
3\text{ models}\times8\text{ layers}\times7\text{ matrices}=168.
\]

No gate, objective, decoder, support, or exposure-budget change follows from this recommendation.

## Required approval and post-approval evidence

The researcher must explicitly approve or reject all of the following as one versioned decision:

1. model ID `ibm-granite/granite-3.3-2b-instruct`;
2. model and tokenizer revision `707f574c62054322f6b5b04b6d075f0a8f05e0f0`;
3. layers 16–23;
4. Apache-2.0/public access disposition and exact parameter count 2,533,539,840;
5. PRE/MID sites and separate projection layout;
6. acceptance that residual scaling is desired cross-model diversity, while physical matrix
   targets remain unscaled;
7. authorization for a bounded tokenizer-only alignment audit, followed separately by at most a
   20-step real three-model GPU preflight.

Only after approval may the implementation populate the fail-closed model-3 slot. The required
evidence sequence is metadata/hash verification, fast-tokenizer exact-prefix audit, acceptance and
bias report, physical-hook/native-output parity, frozen-weight verification, bounded GPU preflight,
and real exact-resume validation. Historical Granite experiments are supporting engineering
evidence only; they are not approval and not a v3.1 main-preflight PASS.
