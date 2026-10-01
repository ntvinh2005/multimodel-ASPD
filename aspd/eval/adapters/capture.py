"""Capture activations at named sites with forward hooks."""

import torch
from aspd.sae.sites import HookSite, SiteCapture
from jaxtyping import Bool, Int
from torch import Tensor, nn


@torch.no_grad()
def keep_token_mask(tokens: Int[Tensor, "b l"], tokenizer: object) -> Bool[Tensor, "b l"]:
    """True where the position survives -- the complement of upstream's bos/pad/eos mask."""
    ids = [
        getattr(tokenizer, name, None)
        for name in ("pad_token_id", "eos_token_id", "bos_token_id")
    ]
    present = [i for i in ids if i is not None]
    assert present, "tokenizer defines none of pad/eos/bos token id; cannot reproduce the mask"

    drop = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in present:
        drop |= tokens == token_id
    return ~drop


@torch.no_grad()
def capture_site(
    model: nn.Module,
    site: HookSite,
    tokens: Int[Tensor, "b l"],
    tokenizer: object,
    *,
    forward: object | None = None,
) -> tuple[Tensor, Bool[Tensor, "b l"]]:
    """`(site_acts zeroed at masked positions, keep_mask)`."""
    mask = keep_token_mask(tokens, tokenizer)
    with SiteCapture(model, [site], detach=True) as cap:
        if forward is None:
            model(tokens)
        else:
            forward(model, tokens)  # type: ignore[operator]
        acts = cap[site.key]
    assert acts.shape[:2] == tokens.shape, (
        f"site {site.key!r} gave {tuple(acts.shape)} for tokens {tuple(tokens.shape)}; "
        "the hooked module is not per-position"
    )
    return acts * mask[:, :, None], mask


@torch.no_grad()
def capture_site_pair(
    model: nn.Module,
    sites: list[HookSite],
    tokens: Int[Tensor, "b l"],
    tokenizer: object,
) -> tuple[dict[str, Tensor], Bool[Tensor, "b l"]]:
    """`capture_site` for SEVERAL sites out of ONE forward, keyed by `HookSite.key`."""
    mask = keep_token_mask(tokens, tokenizer)
    with SiteCapture(model, sites, detach=True) as cap:
        model(tokens)
        acts = {site.key: cap[site.key] for site in sites}
    for key, value in acts.items():
        assert value.shape[:2] == tokens.shape, (
            f"site {key!r} gave {tuple(value.shape)} for tokens {tuple(tokens.shape)}; "
            "the hooked module is not per-position"
        )
    return {k: v * mask[:, :, None] for k, v in acts.items()}, mask
