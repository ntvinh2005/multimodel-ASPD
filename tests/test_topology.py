"""Split Q/K/V paths through the lab's topology schemas."""

import pytest

pytest.importorskip("param_decomp_lab")

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aspd.targets import split_gpt2_qkv
from aspd.topology import install_split_qkv_path_schema, is_split_qkv_gpt2


def _tiny_gpt2() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=2, n_head=4)
    ).eval()


def test_stock_schema_rejects_split_paths():
    from param_decomp_lab.topology.path_schemas import _HFGpt2PathSchema

    schema = _HFGpt2PathSchema()
    assert schema.parse_target_path("transformer.h.1.attn.c_attn") is not None
    with pytest.raises(AssertionError):
        schema.parse_target_path("transformer.h.1.attn.c_attn.q_proj")


def test_patched_schema_parses_every_split_target():
    """All six decomposition targets a split GPT-2 config can name."""
    model = _tiny_gpt2()
    split_gpt2_qkv(model)
    assert is_split_qkv_gpt2(model)
    install_split_qkv_path_schema()

    from param_decomp_lab.topology.path_schemas import get_path_schema

    schema = get_path_schema(model)
    for suffix in (
        "attn.c_attn.q_proj", "attn.c_attn.k_proj", "attn.c_attn.v_proj",
        "attn.c_proj", "mlp.c_fc", "mlp.c_proj",
    ):
        path = f"transformer.h.1.{suffix}"
        assert schema.parse_target_path(path) is not None, path


def test_round_trip_through_the_canonical_form():
    """parse -> render must return the original path, or the app's reverse lookup breaks."""
    model = _tiny_gpt2()
    split_gpt2_qkv(model)
    install_split_qkv_path_schema()

    from param_decomp_lab.topology.path_schemas import get_path_schema

    schema = get_path_schema(model)
    for suffix in ("attn.c_attn.q_proj", "attn.c_attn.v_proj", "attn.c_proj", "mlp.c_fc"):
        path = f"transformer.h.0.{suffix}"
        assert schema.render_canonical_weight(schema.parse_target_path(path)) == path


def test_an_unsplit_gpt2_still_gets_the_stock_schema():
    install_split_qkv_path_schema()

    from param_decomp_lab.topology.path_schemas import _HFGpt2PathSchema, get_path_schema

    schema = get_path_schema(_tiny_gpt2())
    assert isinstance(schema, _HFGpt2PathSchema)
    assert schema.parse_target_path("transformer.h.1.attn.c_attn") is not None


def test_patching_binds_the_name_topology_py_already_imported():
    """`topology.py` did `from ... import get_path_schema` at import time, so patching only the
    `path_schemas` attribute would leave `TransformerTopology.__init__` on the stock function --
    and the failure would look exactly like the patch never being installed.
    """
    install_split_qkv_path_schema()

    from param_decomp_lab.topology import path_schemas, topology

    assert getattr(topology.get_path_schema, "_aspd_split_qkv", False)
    assert topology.get_path_schema is path_schemas.get_path_schema


def _tiny_gemma2():
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(0)
    return Gemma2ForCausalLM(
        Gemma2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    ).eval()


def _swap_in_delta(model, path: str) -> None:
    parent = model.get_submodule(path.rsplit(".", 1)[0])
    name = path.rsplit(".", 1)[1]
    linear = model.get_submodule(path)

    class _AdditiveDelta(torch.nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.base = lin
            self.delta = torch.nn.Linear(lin.in_features, lin.out_features, bias=False)

    setattr(parent, name, _AdditiveDelta(linear))


def test_in_place_gemma2_renders_the_stock_module_path():
    from param_decomp_lab.topology import TransformerTopology
    from param_decomp_lab.topology.path_schemas import _Gemma2DeltaPathSchema, _Gemma2PathSchema

    model = _tiny_gemma2()
    topo = TransformerTopology(model)

    assert isinstance(topo.path_schema, _Gemma2PathSchema)
    assert not isinstance(topo.path_schema, _Gemma2DeltaPathSchema)
    assert topo.canon_to_target("1.glu.down") == "model.layers.1.mlp.down_proj"
    assert model.get_submodule(topo.canon_to_target("1.glu.down")) is not None


def test_delta_swapped_gemma2_still_renders_the_delta_suffix():
    from param_decomp_lab.topology import TransformerTopology
    from param_decomp_lab.topology.path_schemas import _Gemma2DeltaPathSchema

    model = _tiny_gemma2()
    _swap_in_delta(model, "model.layers.1.mlp.down_proj")
    topo = TransformerTopology(model)

    assert isinstance(topo.path_schema, _Gemma2DeltaPathSchema)
    assert topo.canon_to_target("1.glu.down") == "model.layers.1.mlp.down_proj.delta"
    assert model.get_submodule(topo.canon_to_target("1.glu.down")) is not None


@pytest.mark.parametrize("delta_swapped", [False, True])
@pytest.mark.parametrize(
    "canonical",
    ["1.glu.down", "1.glu.up", "1.glu.gate", "0.attn.q", "0.attn.k", "0.attn.v", "0.attn.o"],
)
def test_gemma2_canonical_round_trip_resolves_a_real_module(canonical: str, delta_swapped: bool):
    from param_decomp_lab.topology import TransformerTopology

    model = _tiny_gemma2()
    if delta_swapped:
        for layer in range(2):
            for proj in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                         "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.o_proj"):
                _swap_in_delta(model, f"model.layers.{layer}.{proj}")
    topo = TransformerTopology(model)

    concrete = topo.canon_to_target(canonical)
    assert topo.target_to_canon(concrete) == canonical
    assert model.get_submodule(concrete) is not None


def test_widen_lab_config_parsing_installs_the_schema():
    """The patch must be INSTALLED, not merely available."""
    from param_decomp_lab.topology import path_schemas, topology

    from aspd.lab_compat import widen_lab_config_parsing

    widen_lab_config_parsing()
    assert getattr(path_schemas.get_path_schema, "_aspd_split_qkv", False)
    assert topology.get_path_schema is path_schemas.get_path_schema


def _tiny_qwen3():
    """A 2-layer Qwen3 with the real module tree and none of the 16 GB."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(
        num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, vocab_size=128,
        layer_types=["full_attention"] * 2, tie_word_embeddings=False,
    )
    return transformers.Qwen3ForCausalLM(cfg)


def test_a_qwen3_target_gets_a_path_schema_at_all():
    from param_decomp_lab.topology.topology import TransformerTopology

    from aspd.topology import install_qwen3_path_schema

    install_qwen3_path_schema()
    model = _tiny_qwen3()
    topo = TransformerTopology(model)
    # What `aspd.cli.harvest` reaches for, and the exact attribute the failed jobs died on.
    assert topo.embedding_module.num_embeddings == 128


@pytest.mark.parametrize(
    "path",
    [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.self_attn.k_proj",
        "model.layers.1.self_attn.v_proj",
        # THE decomposed matrix on this target.
        "model.layers.1.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.1.mlp.up_proj",
        "model.layers.1.mlp.down_proj",
    ],
)
def test_every_qwen3_block_path_round_trips(path):
    """Parse and render must be inverses, or a harvest key names a module that does not exist."""
    from param_decomp_lab.topology.topology import TransformerTopology

    from aspd.topology import install_qwen3_path_schema

    install_qwen3_path_schema()
    schema = TransformerTopology(_tiny_qwen3()).path_schema
    assert schema._render_layer_weight(schema._parse_block_path(path)) == path


def test_the_two_path_patches_compose_in_either_order():
    """Each wraps whatever `get_path_schema` it finds and falls through, so both stay live however
    they are installed -- and each carries the other's marker forward so a repeat call does not
    wrap a second time.
    """
    from param_decomp_lab.topology import path_schemas, topology

    from aspd.topology import install_qwen3_path_schema, install_split_qkv_path_schema

    for order in ((install_qwen3_path_schema, install_split_qkv_path_schema),
                  (install_split_qkv_path_schema, install_qwen3_path_schema)):
        for install in order:
            install()
        for install in order:  # idempotent
            install()
        markers = vars(path_schemas.get_path_schema)
        assert markers.get("_aspd_qwen3") and markers.get("_aspd_split_qkv")
        assert path_schemas.get_path_schema is topology.get_path_schema


def test_the_lab_compat_reroute_installs_the_qwen3_schema():
    """`widen_lab_config_parsing` is the one hook every stock entrypoint goes through, so a patch
    that is written but never CALLED is the failure this file already records once -- and it
    happened again here: the patch existed, its own tests passed, and harvest still died because
    nothing invoked it (jobs 41537384/41537385, the RETRY of the first pair).
    """
    import importlib

    from param_decomp_lab.topology import path_schemas, topology

    from aspd.lab_compat import widen_lab_config_parsing

    importlib.reload(path_schemas)
    importlib.reload(topology)
    assert not getattr(path_schemas.get_path_schema, "_aspd_qwen3", False), (
        "reload did not clear the patch; this test cannot prove anything"
    )
    widen_lab_config_parsing()
    assert getattr(path_schemas.get_path_schema, "_aspd_qwen3", False)
