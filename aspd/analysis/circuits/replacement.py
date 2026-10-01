"""The replacement forward pass that exposes every component (and the residual error) as a node."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class ReplacementCache:
    """Everything the edge computation differentiates against."""

    component_acts: dict[str, Tensor] = field(default_factory=dict)
    inputs: dict[str, Tensor] = field(default_factory=dict)
    component_acts_pre: dict[str, Tensor] = field(default_factory=dict)
    errors: dict[str, Tensor] = field(default_factory=dict)
    ci: dict[str, Tensor] = field(default_factory=dict)
    embed: Tensor | None = None
    logits: Tensor | None = None


@contextmanager
def _hooks(model, cache: ReplacementCache, masks: dict[str, Tensor], error_nodes: bool,
           clean_masks: dict[str, Tensor] | None = None):
    handles = []
    for path, components in model.components.items():
        def hook(_mod, args, kwargs, output, path=path, components=components):
            assert len(args) == 1 and not kwargs, "single-tensor module input expected"
            acts: dict[str, Tensor] = {}
            cache.inputs[path] = args[0].detach()
            y_hat = components(args[0], mask=masks[path], component_acts_cache=acts)
            cache.component_acts[path] = acts["post_detach"]
            cache.component_acts_pre[path] = acts["pre_detach"]
            if not error_nodes:
                return y_hat
            ref = y_hat
            if clean_masks is not None and path in clean_masks:
                ref = components(args[0], mask=clean_masks[path])
            eps = (output - ref).detach().requires_grad_(True)
            cache.errors[path] = eps
            return y_hat + eps

        handles.append(model.target_model.get_submodule(path).register_forward_hook(
            hook, with_kwargs=True
        ))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def _embed_hook(model, topology, cache: ReplacementCache) -> Iterator[None]:
    def hook(_mod, _args, _kwargs, out: Tensor) -> Tensor:
        out.requires_grad_(True)
        cache.embed = out
        return out

    h = topology.embedding_module.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


def run_replacement(
    model,
    tokens: Tensor,
    *,
    sampling: str,
    topology=None,
    error_nodes: bool = True,
    freeze_attention: bool = False,
    freeze_layernorm: bool = False,
    mask_edits: dict[str, Tensor] | None = None,
) -> ReplacementCache:
    """One clean pass for `g`, then the differentiable replacement pass."""
    from aspd.capture_guard import disarmed
    from param_decomp.torch_helpers import bf16_autocast

    from aspd.analysis.circuits.freeze import frozen_attention_and_norm

    cache = ReplacementCache()

    with torch.no_grad(), bf16_autocast():
        clean = model(tokens, cache_type="input")
        ci = model.calc_causal_importances(pre_weight_acts=clean.cache, sampling=sampling)
    cache.ci = {k: v.detach().float() for k, v in ci.lower_leaky.items()}

    masks = {k: v for k, v in cache.ci.items()}
    clean_masks = None
    if mask_edits:
        clean_masks = {k: masks[k] for k in mask_edits}
        for key, factor in mask_edits.items():
            masks[key] = masks[key] * factor
            cache.ci[key] = masks[key]

    with torch.enable_grad(), disarmed():
        with frozen_attention_and_norm(
            model.target_model, attention=freeze_attention, layernorm=freeze_layernorm
        ):
            if topology is not None:
                with _embed_hook(model, topology, cache), _hooks(
                    model, cache, masks, error_nodes, clean_masks):
                    cache.logits = model(tokens)
            else:
                with _hooks(model, cache, masks, error_nodes, clean_masks):
                    cache.logits = model(tokens)
    return cache


def assert_forward_is_exact(model, tokens: Tensor, *, sampling: str, atol: float = 1e-4) -> None:
    """The invariant worth the most: with error nodes on, the replacement IS the target model."""
    with torch.no_grad():
        reference = model(tokens)
    cache = run_replacement(model, tokens, sampling=sampling, error_nodes=True)
    assert cache.logits is not None
    delta = (cache.logits - reference).abs().max().item()
    assert delta < atol, (
        f"replacement logits differ from the target by {delta:.3e} (> {atol:.0e}). With error "
        "nodes on this must be zero to fp tolerance -- the local-replacement property is what "
        "makes every gradient below a linearization of the REAL model."
    )
