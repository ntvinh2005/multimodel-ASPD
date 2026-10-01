"""Extends the lab's `TransformerTopology` with the module paths of the supported models."""

import re

from torch import nn


def _split_qkv_schema_class():
    """Built lazily so importing this module does not drag in the lab (and torch's HF stack)."""
    from param_decomp_lab.topology.path_schemas import (
        _FFNPathSchema,
        _PathSchema,
        _SeparateAttnPathSchema,
    )

    class _HFGpt2SplitQKVPathSchema(_PathSchema):
        """`_HFGpt2PathSchema` with attention SEPARATE rather than fused."""

        embedding_path = "transformer.wte"
        blocks = "transformer.h"
        attn = _SeparateAttnPathSchema(
            base="attn", q="c_attn.q_proj", k="c_attn.k_proj", v="c_attn.v_proj", o="c_proj"
        )
        mlp = _FFNPathSchema(base="mlp", up="c_fc", down="c_proj")
        unembed_path = "lm_head"
        _block_re = re.compile(
            r"^transformer\.h\.(?P<idx>\d+)\."
            r"(?:(?P<attn>attn)\.(?P<attn_proj>[\w.]+)"
            r"|(?P<mlp>mlp)\.(?P<mlp_proj>\w+))$"
        )

    return _HFGpt2SplitQKVPathSchema


def is_split_qkv_gpt2(model: nn.Module) -> bool:
    """True for an HF GPT-2 whose blocks have been through `targets.split_gpt2_qkv`."""
    from transformers.models.gpt2 import GPT2LMHeadModel

    from aspd.targets import SplitQKV

    if not isinstance(model, GPT2LMHeadModel):
        return False
    return any(isinstance(b.attn.c_attn, SplitQKV) for b in model.transformer.h)


def install_split_qkv_path_schema() -> None:
    """Idempotent. Call before anything constructs a `TransformerTopology` on a split model."""
    from param_decomp_lab.topology import path_schemas, topology

    if getattr(path_schemas.get_path_schema, "_aspd_split_qkv", False):
        return
    stock = path_schemas.get_path_schema
    schema_cls = _split_qkv_schema_class()

    def _get_path_schema(model: nn.Module):
        if is_split_qkv_gpt2(model):
            return schema_cls()
        return stock(model)

    _get_path_schema._aspd_split_qkv = True  # pyright: ignore[reportFunctionMemberAccess]
    if getattr(stock, "_aspd_qwen3", False):
        _get_path_schema._aspd_qwen3 = True  # pyright: ignore[reportFunctionMemberAccess]
    path_schemas.get_path_schema = _get_path_schema
    # The load-bearing second line: `topology.py` bound the name at import time.
    topology.get_path_schema = _get_path_schema


def _qwen3_schema_class():
    """Built lazily, for the same reason as the split-QKV one: keep the lab out of import time."""
    from param_decomp_lab.topology.path_schemas import (
        _GLUPathSchema,
        _PathSchema,
        _SeparateAttnPathSchema,
    )

    class _Qwen3PathSchema(_PathSchema):
        """Qwen3 (HF `Qwen3ForCausalLM`) decomposed in place."""

        embedding_path = "model.embed_tokens"
        blocks = "model.layers"
        attn = _SeparateAttnPathSchema(
            base="self_attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj"
        )
        mlp = _GLUPathSchema(base="mlp", gate="gate_proj", up="up_proj", down="down_proj")
        unembed_path = "lm_head"

    return _Qwen3PathSchema


def is_qwen3(model: nn.Module) -> bool:
    """True for an HF Qwen3 causal LM. Import is local so this module stays cheap to import."""
    from transformers.models.qwen3 import Qwen3ForCausalLM

    return isinstance(model, Qwen3ForCausalLM)


def install_qwen3_path_schema() -> None:
    """Idempotent. Call before anything constructs a `TransformerTopology` on a Qwen3 target."""
    from param_decomp_lab.topology import path_schemas, topology

    if getattr(path_schemas.get_path_schema, "_aspd_qwen3", False):
        return
    stock = path_schemas.get_path_schema
    schema_cls = _qwen3_schema_class()

    def _get_path_schema(model: nn.Module):
        if is_qwen3(model):
            return schema_cls()
        return stock(model)

    _get_path_schema._aspd_qwen3 = True  # pyright: ignore[reportFunctionMemberAccess]
    if getattr(stock, "_aspd_split_qkv", False):
        _get_path_schema._aspd_split_qkv = True  # pyright: ignore[reportFunctionMemberAccess]
    path_schemas.get_path_schema = _get_path_schema
    # The load-bearing second line: `topology.py` bound the name at import time.
    topology.get_path_schema = _get_path_schema
