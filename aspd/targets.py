"""Target-model builders referenced from a config's `target.spec` (`kind: callable`).

`build_hf_target` loads a Hugging Face causal LM with an explicit dtype and attention
implementation (eager attention for Gemma-2, which applies its attention soft-capping).
"""

import importlib
from typing import Any

import torch
from torch import Tensor, nn

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _resolve_class(fqn: str) -> type:
    module_path, _, class_name = fqn.rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


class SplitQKV(nn.Module):
    """GPT-2's fused `c_attn` as three separately-decomposable projections."""

    def __init__(self, q_proj: nn.Module, k_proj: nn.Module, v_proj: nn.Module):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)


def _slice_conv1d(fused: nn.Module, start: int, stop: int) -> nn.Module:
    """One `Conv1D(stop-start, nx)` carrying `fused`'s columns `[start:stop)`, weights copied."""
    from transformers.pytorch_utils import Conv1D

    out = Conv1D(stop - start, fused.nx).to(dtype=fused.weight.dtype, device=fused.weight.device)
    with torch.no_grad():
        out.weight.copy_(fused.weight[:, start:stop])
        out.bias.copy_(fused.bias[start:stop])
    return out


def split_gpt2_qkv(model: nn.Module) -> nn.Module:
    """Replace every block's fused `attn.c_attn` with a `SplitQKV`, in place."""
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        attn = block.attn
        assert not attn.is_cross_attention, (
            f"block {i} is cross-attention: its `c_attn` holds only K and V (2*d) with Q in a "
            "separate `q_attn`, so the three-way column slice below would be wrong."
        )
        assert not attn.pruned_heads, (
            f"block {i} has pruned heads {sorted(attn.pruned_heads)}: `c_attn.nf` has already been "
            "narrowed and `split_size` no longer equals `embed_dim`, so the slice bounds are stale."
        )
        d = attn.embed_dim
        fused = attn.c_attn
        assert fused.nf == 3 * d and fused.nx == d, (
            f"block {i}: expected a fused Conv1D(3*{d}, {d}), got Conv1D({fused.nf}, {fused.nx})"
        )
        assert attn.split_size == d, (
            f"block {i}: `split_size` is {attn.split_size}, not {d}; `.split(split_size)` in "
            "GPT2Attention.forward would not cut on the boundaries this slice assumes."
        )
        attn.c_attn = SplitQKV(
            _slice_conv1d(fused, 0, d),
            _slice_conv1d(fused, d, 2 * d),
            _slice_conv1d(fused, 2 * d, 3 * d),
        )
    return model


def build_hf_target(params: dict[str, Any]) -> nn.Module:
    """`<model_class>.from_pretrained(<model_name>, ...)`, with the kwargs `HFTarget` cannot pass."""
    from param_decomp_lab.distributed import ensure_cached_and_call

    params = dict(params)
    cls = _resolve_class(params.pop("model_class"))
    model_name = params.pop("model_name")
    split_qkv = params.pop("split_qkv", False)
    if "dtype" in params:
        params["dtype"] = DTYPES[params["dtype"]]
    model = ensure_cached_and_call(cls.from_pretrained, model_name, **params)
    if split_qkv:
        split_gpt2_qkv(model)
    return model.eval()
