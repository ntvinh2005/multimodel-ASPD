"""Logit lens over decoder directions: the tokens each direction promotes and suppresses."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from aspd.eval.dictionary import DictionaryAdapter


@dataclass(frozen=True)
class LogitLensRefs:
    """Model handles the lens needs, resolved once from the model + SitePair."""

    final_norm: nn.Module
    unembed: Float[Tensor, "d vocab"]
    mlp_module: nn.Module | None = None
    c_fc_module: nn.Module | None = None
    mlp_in_width: int | None = None

    residual_roles: frozenset[str] = frozenset({"in"})
    """Roles whose decoder rows are ALREADY residual-space; every other role is MLP-hidden."""

    hidden_take: Literal["input", "output"] = "output"
    """Whether the MLP-hidden activation is `c_fc_module`'s OUTPUT (`c_fc`) or its INPUT (`down_proj`)."""


@torch.no_grad()
def _continue_through_mlp(
    directions: Float[Tensor, "f d_hidden"], refs: LogitLensRefs
) -> Float[Tensor, "f d_model"]:
    assert refs.mlp_module is not None and refs.c_fc_module is not None
    assert refs.mlp_in_width is not None
    f = directions.shape[0]
    dev = directions.device
    dtype = directions.dtype

    def override(_m: nn.Module, _a: tuple, _o: object) -> Tensor:
        return directions

    def override_input(_m: nn.Module, _a: tuple) -> tuple:
        return (directions,)

    handle = (
        refs.c_fc_module.register_forward_pre_hook(override_input)
        if refs.hidden_take == "input"
        else refs.c_fc_module.register_forward_hook(override)
    )
    try:
        dummy = torch.zeros(1, refs.mlp_in_width, device=dev, dtype=dtype)
        resid = refs.mlp_module(dummy)
    finally:
        handle.remove()
    assert resid.shape == (f, refs.unembed.shape[0]), (
        f"MLP continuation returned {tuple(resid.shape)}, expected {(f, refs.unembed.shape[0])}; "
        "the parent MLP's output width must match the residual/unembed width"
    )
    return resid


LENS_BLOCK = 4096
"""Latent rows per logit-lens block. `[block, vocab]` fp32 is 4.2 GiB at Gemma-2's 256k vocab."""


@torch.no_grad()
def _block_logits(
    dictionary: DictionaryAdapter, refs: LogitLensRefs, start: int, stop: int
) -> Float[Tensor, "b vocab"]:
    """`final_norm(lift(W_dec[start:stop])) @ W_U` -- the lens for one slice of latents."""
    rows = dictionary.decoder_rows()[start:stop].to(refs.unembed.dtype)
    resid = rows if dictionary.role in refs.residual_roles else _continue_through_mlp(rows, refs)
    return refs.final_norm(resid) @ refs.unembed


@torch.no_grad()
def decoder_logits(
    dictionary: DictionaryAdapter, refs: LogitLensRefs
) -> Float[Tensor, "f vocab"]:
    """Per-latent logit-lens vector: `final_norm(lift(W_dec[i])) @ W_U`, `[F, vocab]`."""
    return torch.cat(
        [
            _block_logits(dictionary, refs, start, min(start + LENS_BLOCK, dictionary.n_features))
            for start in range(0, dictionary.n_features, LENS_BLOCK)
        ]
    )


@torch.no_grad()
def _blocked_topk(
    dictionary: DictionaryAdapter, refs: LogitLensRefs, *, top_k: int, largest: bool
) -> tuple[Int[Tensor, "f k"], Float[Tensor, "f k"]]:
    """Top-`k` over the vocab per latent, never holding more than `LENS_BLOCK` rows of logits."""
    idx, val = [], []
    for start in range(0, dictionary.n_features, LENS_BLOCK):
        block = _block_logits(
            dictionary, refs, start, min(start + LENS_BLOCK, dictionary.n_features)
        )
        top = block.topk(top_k, dim=-1, largest=largest)
        idx.append(top.indices)
        val.append(top.values)
    return torch.cat(idx), torch.cat(val)


@torch.no_grad()
def top_tokens(
    dictionary: DictionaryAdapter, refs: LogitLensRefs, *, top_k: int
) -> tuple[Int[Tensor, "f k"], Float[Tensor, "f k"]]:
    """Top-`k` promoted token ids per latent and their logit-lens scores."""
    return _blocked_topk(dictionary, refs, top_k=top_k, largest=True)


@torch.no_grad()
def latent_logit_lens(
    dictionary: DictionaryAdapter,
    refs: LogitLensRefs,
    decode: Callable[[list[int]], list[str]],
    *,
    top_k: int,
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """`{component_key -> {"top": [(token, score)], "bottom": [(token, score)]}}`."""
    top_idx, top_val = _blocked_topk(dictionary, refs, top_k=top_k, largest=True)
    bot_idx, bot_val = _blocked_topk(dictionary, refs, top_k=top_k, largest=False)
    out: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for i in range(top_idx.shape[0]):
        key = f"{dictionary.site_path}:{i}"
        out[key] = {
            "top": list(zip(decode(top_idx[i].tolist()), top_val[i].tolist())),
            "bottom": list(zip(decode(bot_idx[i].tolist()), bot_val[i].tolist())),
        }
    return out


@torch.no_grad()
def write_latent_logit_lens_json(
    dictionaries: list[DictionaryAdapter],
    refs: LogitLensRefs,
    decode: Callable[[list[int]], list[str]],
    out_path: "Path",  # noqa: F821
    *,
    top_k: int = 5,
) -> "Path":  # noqa: F821
    """Compute top/bottom logit-lens tokens for every latent of every dictionary and write JSON."""
    import json
    from pathlib import Path

    merged: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for d in dictionaries:
        merged.update(latent_logit_lens(d, refs, decode, top_k=top_k))
    out = Path(out_path)
    out.write_text(json.dumps(merged))
    return out
