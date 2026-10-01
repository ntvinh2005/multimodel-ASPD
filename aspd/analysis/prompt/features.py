"""SAE features on one prompt and their interactions with components."""


import torch
from torch import Tensor

from aspd.analysis.pairs import features as ft
from aspd.analysis.prompt.trace import PromptTrace

GPT2_CAPTURE: dict[str, tuple[str, str]] = {
    "resid_pre": ("", "input"),
    "attn_out": ("attn", "output"),
    "resid_mid": ("ln_2", "input"),
    "mlp_out": ("mlp", "output"),
    "resid_post": ("", "output"),
    "attn_z": ("attn.c_proj", "input"),
}


def capture_site(trace: PromptTrace, site: str, layer: int) -> Tensor:
    assert "gpt2" in trace.model_name.lower(), (
        f"site capture is implemented for GPT-2; {trace.model_name} assembles its residual stream "
        "differently and would need its own table"
    )
    assert site in GPT2_CAPTURE, f"unknown site {site!r}; known: {sorted(GPT2_CAPTURE)}"
    suffix, which = GPT2_CAPTURE[site]
    path = f"transformer.h.{layer}" + (f".{suffix}" if suffix else "")
    module = trace.model.target_model.get_submodule(path)  # pyright: ignore[reportAttributeAccessIssue]

    got: dict[str, Tensor] = {}

    def hook(_m, args, output):
        tensor = args[0] if which == "input" else (output[0] if isinstance(output, tuple) else output)
        got["x"] = tensor.detach().float()

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            trace.model.target_model(trace.tokens)  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        handle.remove()
    assert "x" in got, f"no activation captured at {path}"
    return got["x"][0]


def feature_acts(trace: PromptTrace, saes: ft.SaeStore, key: str) -> Tensor:
    """`[P, F]` -- every feature's activation at every token."""
    endpoint = saes.endpoint(key)
    x = capture_site(trace, endpoint.site, endpoint.layer)
    if endpoint.centred:
        x = ft.centre(x)
    sae = saes.sae(key)
    if endpoint.site == "attn_z":
        d_h = trace.head_dim
        assert x.shape[1] % d_h == 0, f"{x.shape[1]}d z is not a multiple of head_dim {d_h}"
        x = x.unflatten(1, (x.shape[1] // d_h, d_h))
    with torch.no_grad():
        return sae.encode(x).detach().float()  # pyright: ignore[reportAttributeAccessIssue]
