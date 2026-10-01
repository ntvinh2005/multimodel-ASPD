"""Optionally detach attention patterns and LayerNorm scales when computing attributions."""

from contextlib import ExitStack, contextmanager

import torch
from torch import nn


def _is_norm(module: nn.Module) -> bool:
    name = type(module).__name__
    return "LayerNorm" in name or "RMSNorm" in name


@contextmanager
def frozen_attention_and_norm(target_model: nn.Module, *, attention: bool, layernorm: bool):
    """Detach what `attention` / `layernorm` select, for the block's lifetime."""
    with ExitStack() as stack:
        if attention:
            raise NotImplementedError(
                "freeze_attention is not implemented. Attention probabilities live inside "
                "`*Attention.forward` with no module boundary to hook, so a hook-based version "
                "would silently leave the gradient path open -- the opposite of what the flag "
                "promises. Implement it against `eager` attention explicitly, with a test that "
                "asserts d(target)/d(pattern) is zero, before turning this on."
            )
        if layernorm:
            for module in target_model.modules():
                if not _is_norm(module):
                    continue
                stack.enter_context(_frozen_norm(module))
        yield


@contextmanager
def _frozen_norm(module: nn.Module):
    def hook(_mod, args, _kwargs, out):
        x = args[0]
        gain = (out / torch.where(x.abs() > 1e-12, x, torch.full_like(x, 1e-12))).detach()
        return out.detach() + (x - x.detach()) * gain

    h = module.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()
