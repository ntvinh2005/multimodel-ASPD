"""Which vector space each decomposed matrix reads from and writes into."""

import re
from dataclasses import dataclass
from typing import Literal

Side = Literal["read", "write"]

RESID = "resid"

HEAD_DIM_BY_MODEL = {
    "openai-community/gpt2": 64,
    "gpt2": 64,
    "openai-community/gpt2-medium": 64,
    "openai-community/gpt2-large": 64,
    "openai-community/gpt2-xl": 64,
    "google/gemma-2-2b": 256,
    "google/gemma-2-2b-it": 256,
    "google/gemma-2-9b": 256,
}


@dataclass(frozen=True)
class HeadLayout:
    """`dim = n_heads * head_dim`, laid out as contiguous per-head blocks."""

    n_heads: int
    head_dim: int

    def head_slice(self, h: int) -> slice:
        assert 0 <= h < self.n_heads, f"head {h} out of range for {self.n_heads} heads"
        return slice(h * self.head_dim, (h + 1) * self.head_dim)


@dataclass(frozen=True)
class Space:
    """A vector space, identified by `key` -- NOT by dimension."""

    key: str
    label: str
    dim: int
    heads: HeadLayout | None = None


@dataclass(frozen=True)
class ModuleSpaces:
    """The two endpoints a decomposed matrix contributes to the pair graph."""

    module: str
    role: str
    layer: int
    read: Space
    write: Space

    def space(self, side: Side) -> Space:
        return self.read if side == "read" else self.write


_GPT2_ATTN_QKV = re.compile(r"^transformer\.h\.(\d+)\.attn\.c_attn\.([qkv])_proj$")
_GPT2_ATTN_O = re.compile(r"^transformer\.h\.(\d+)\.attn\.c_proj$")
_GPT2_MLP = re.compile(r"^transformer\.h\.(\d+)\.mlp\.(c_fc|c_proj)$")
_HF_ATTN = re.compile(r"^model\.layers\.(\d+)\.self_attn\.([qkvo])_proj$")
_HF_MLP = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj$")

# role -> (read space suffix, write space suffix). `None` means the residual stream.
_ROLE_SPACES: dict[str, tuple[str | None, str | None]] = {
    "attn.q": (None, "attn.q"),
    "attn.k": (None, "attn.k"),
    "attn.v": (None, "attn.z"),
    "attn.o": ("attn.z", None),
    "mlp.in": (None, "mlp.preact"),
    "mlp.gate": (None, "mlp.gate"),
    "mlp.up": (None, "mlp.up"),
    "mlp.out": ("mlp.postact", None),
}
# Which spaces carry attention head structure.
_HEADED = {"attn.q", "attn.k", "attn.z"}

_SPACE_LABELS = {
    "attn.q": "query",
    "attn.k": "key",
    "attn.z": "value / attn-out",
    "mlp.preact": "MLP hidden (pre-activation)",
    "mlp.postact": "MLP hidden (post-activation)",
    "mlp.gate": "MLP gate branch (pre-activation)",
    "mlp.up": "MLP up branch",
}


def parse_role(module: str) -> tuple[str, int]:
    """`(role, layer)` for a decomposed module path, on either architecture."""
    if m := _GPT2_ATTN_QKV.match(module):
        return f"attn.{m.group(2)}", int(m.group(1))
    if m := _GPT2_ATTN_O.match(module):
        return "attn.o", int(m.group(1))
    if m := _GPT2_MLP.match(module):
        return ("mlp.in" if m.group(2) == "c_fc" else "mlp.out"), int(m.group(1))
    if m := _HF_ATTN.match(module):
        return f"attn.{m.group(2)}", int(m.group(1))
    if m := _HF_MLP.match(module):
        role = {"gate": "mlp.gate", "up": "mlp.up", "down": "mlp.out"}[m.group(2)]
        return role, int(m.group(1))
    raise AssertionError(f"unrecognised decomposed module path: {module!r}")


def _space(suffix: str | None, layer: int, dim: int, head_dim: int | None) -> Space:
    if suffix is None:
        return Space(key=RESID, label="residual stream", dim=dim)
    key = f"L{layer}.{suffix}"
    heads = None
    if suffix in _HEADED:
        assert head_dim is not None, f"{key} is head-structured but no head_dim is known"
        assert dim % head_dim == 0, f"{key}: dim {dim} not divisible by head_dim {head_dim}"
        heads = HeadLayout(n_heads=dim // head_dim, head_dim=head_dim)
    return Space(key=key, label=f"L{layer} {_SPACE_LABELS[suffix]}", dim=dim, heads=heads)


def site_space(suffix: str | None, layer: int, dim: int, head_dim: int | None) -> Space:
    """The space a pretrained SAE's hook site occupies, keyed exactly as a module's side is."""
    return _space(suffix, layer, dim, head_dim)


def module_spaces(module: str, d_in: int, d_out: int, head_dim: int | None) -> ModuleSpaces:
    """Resolve one decomposed module's two endpoints from its path and its weight shapes."""
    role, layer = parse_role(module)
    read_suffix, write_suffix = _ROLE_SPACES[role]
    return ModuleSpaces(
        module=module,
        role=role,
        layer=layer,
        read=_space(read_suffix, layer, d_in, head_dim),
        write=_space(write_suffix, layer, d_out, head_dim),
    )


def head_dim_for_model(model_name: str) -> int | None:
    return HEAD_DIM_BY_MODEL.get(model_name)


def head_pairs(a: HeadLayout, b: HeadLayout) -> list[tuple[int, int]]:
    """`[(head_in_a, head_in_b)]`, one entry per head of the WIDER side (GQA-aware)."""
    assert a.head_dim == b.head_dim, "head_dim must match to pair heads"
    n = max(a.n_heads, b.n_heads)
    assert n % a.n_heads == 0 and n % b.n_heads == 0, (
        f"head counts {a.n_heads} and {b.n_heads} are not GQA-compatible"
    )
    return [(h // (n // a.n_heads), h // (n // b.n_heads)) for h in range(n)]


def compatibility(a: Space, b: Space) -> dict[str, object]:
    """What can be computed between two spaces: flat, per-head, neither."""
    flat = a.dim == b.dim
    headed = a.heads is not None and b.heads is not None and a.heads.head_dim == b.heads.head_dim
    n_heads = 0
    if headed:
        try:
            n_heads = len(head_pairs(a.heads, b.heads))  # pyright: ignore[reportArgumentType]
        except AssertionError:
            headed, n_heads = False, 0
    return {
        "flat": flat,
        "per_head": headed,
        "n_head_pairs": n_heads,
        "same_space": a.key == b.key,
        "reason": None
        if (flat or headed)
        else f"{a.label} is {a.dim}-dimensional and {b.label} is {b.dim}-dimensional",
    }
