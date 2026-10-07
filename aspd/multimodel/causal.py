"""Live P6 intervention: ablate selected mechanisms in one model on the clean shared gate."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any

import torch
from torch import Tensor, nn

from aspd.multimodel.cache import DTYPES, PairedActivationCache, _tensor_from_value
from aspd.multimodel.config import ModelSpec, MultiModelExperimentConfig
from aspd.multimodel.model import MultiModelASPD

# P6 example: ablate one c in model n. If g_t,c=1, v_j,c^T x_j,t=2, and
# u_j,c=[1,3], the hook subtracts 2*u_j,c=[2,6] from W_j x_j,t.


def _load_hf_model(spec: ModelSpec, device: torch.device) -> nn.Module:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "dtype": DTYPES[spec.dtype],
        "trust_remote_code": spec.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if spec.attn_implementation:
        kwargs["attn_implementation"] = spec.attn_implementation
    return AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs).to(device).eval()


@torch.no_grad()
def causal_ablation(
    decomposition: MultiModelASPD,
    cache: PairedActivationCache,
    cfg: MultiModelExperimentConfig,
    input_ids: Tensor,
    valid_tokens: Tensor,
    target_model_index: int,
    component_ids: Sequence[int],
    device: torch.device,
) -> dict[str, Tensor]:
    """Apply ``W_jx -> W_jx - sum_c g_tc P_jc x`` for every selected ``j,c``.

    The shared gate is computed from clean ``R^(1),...,R^(N)``.  Keeping that gate fixed makes the
    intervention exactly the P6 operation and avoids a second-order change in feature selection.
    """

    # Load all M^(n) so the clean shared gate can read R^(1),...,R^(N).
    models = [_load_hf_model(spec, device) for spec in cfg.models]
    clean_logits: list[Tensor] = []
    grounding: list[Tensor] = []
    with ExitStack() as stack:
        for n, (hf_model, spec) in enumerate(zip(models, cfg.models, strict=True)):
            holder: dict[str, Tensor] = {}
            module = hf_model.get_submodule(spec.grounding.module)
            if spec.grounding.capture == "input":
                stack.callback(
                    module.register_forward_pre_hook(
                        lambda _module, args, h=holder, s=spec: h.update(
                            R=_tensor_from_value(
                                args, s.grounding.tensor_index or 0, s.grounding.module
                            ).detach()
                        )
                    ).remove
                )
            else:
                stack.callback(
                    module.register_forward_hook(
                        lambda _module, _args, output, h=holder, s=spec: h.update(
                            R=_tensor_from_value(
                                output, s.grounding.tensor_index, s.grounding.module
                            ).detach()
                        )
                    ).remove
                )
            result = hf_model(
                input_ids=input_ids.to(device),
                attention_mask=valid_tokens.to(device),
                use_cache=False,
            )
            # Store z_clean^(n)[B,T,V] before any P_j,c intervention.
            clean_logits.append(result.logits.detach())
            # Apply the same q_n used in training: R_scaled^(n)=R_raw^(n)/r_rms_norm_n.
            # Example ||r|| RMS=2 maps r=[2,0] to [1,0].
            grounding.append(holder["R"] / cache.model_manifests[n]["r_rms_norm"])

    # Compute clean g^s and g jointly from every normalized R^(n), once.
    code = decomposition.encode(grounding, valid_tokens.to(device))
    # Selected global latent indices c; example A={17,203}.
    selected = torch.tensor(component_ids, dtype=torch.long, device=device)
    # Fixed clean binary gates g_t,c for c in A, shape [B,T,|A|].
    selected_gate = code.gate[..., selected]
    target_hf_model = models[target_model_index]
    with ExitStack() as stack:
        for matrix in cfg.models[target_model_index].matrices:
            u, v = decomposition.component_factors(target_model_index, matrix.name)
            # U_A contains u_j,c^(n) rows, shape [|A|,d_out].
            u_selected = u[selected]
            # V_A contains v_j,c^(n) rows, shape [|A|,d_in].
            v_selected = v[selected]

            def subtract_components(
                _module: nn.Module,
                args: tuple[object, ...],
                output: object,
                u_local: Tensor = u_selected,
                v_local: Tensor = v_selected,
                site: str = matrix.module,
            ) -> Tensor:
                # Live X_j^(n)[B,T,d_in] entering W_j^(n).
                x = _tensor_from_value(args, 0, site).to(v_local.dtype)
                # Clean module output Y_j^(n)=W_j^(n)X_j^(n).
                y = _tensor_from_value(output, None, site)
                # reads_t,c=v_j,c^(n)T x_j,t^(n). Example x=[2,1],v=[1,0] -> 2.
                reads = torch.einsum("btd,kd->btk", x, v_local)
                # e_j,t,c^(n)=g_t,c*read_t,c; inactive clean gates zero the intervention.
                effective = reads * selected_gate.to(reads.dtype)
                # sum_c e_j,t,c^(n)u_j,c^(n), shape [B,T,d_out].
                # Example e=2,u=[1,3] produces subtraction [2,6].
                subtraction = torch.einsum("btk,ko->bto", effective, u_local)
                # P6 patched output: W_j x_j,t - sum_c g_t,c P_j,c x_j,t.
                return y - subtraction.to(y.dtype)

            stack.callback(
                target_hf_model.get_submodule(matrix.module)
                .register_forward_hook(subtract_components)
                .remove
            )
        patched_logits = target_hf_model(
            input_ids=input_ids.to(device),
            attention_mask=valid_tokens.to(device),
            use_cache=False,
        ).logits

    # Compare logits for the intervened target n in FP32.
    clean = clean_logits[target_model_index].float()
    patched = patched_logits.float()
    # log p_clean(v|s_<=t) over vocabulary v.
    clean_log_prob = clean.log_softmax(dim=-1)
    # log p_patched(v|s_<=t) after subtracting selected mechanisms.
    patched_log_prob = patched.log_softmax(dim=-1)
    # KL_t=sum_v p_clean(v)[log p_clean(v)-log p_patched(v)].
    # Example p=[.5,.5],q=[.9,.1] gives a positive behavioral change.
    kl = (clean_log_prob.exp() * (clean_log_prob - patched_log_prob)).sum(dim=-1)
    return {
        "clean_logits": clean.cpu(),
        "patched_logits": patched.cpu(),
        "kl_per_token": kl.cpu(),
        # Mean causal effect over valid t only.
        "mean_kl": kl[valid_tokens.to(device)].mean().cpu(),
        # Largest individual vocabulary-logit movement anywhere in [B,T,V].
        "max_abs_logit_change": (patched - clean).abs().max().cpu(),
    }
