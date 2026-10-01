"""The residual-site registry: which residual stream each matrix's encoder reads."""

import pytest

from aspd.sites import (
    ARCH_RESID_SITES,
    RESID_SITES,
    group_modules_by_resid_site,
    resid_site_entry,
    resid_site_for_module,
    resid_site_runs_after,
)

GPT2_MATRICES = (
    "attn.c_attn.q_proj",
    "attn.c_attn.k_proj",
    "attn.c_attn.v_proj",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)


def _gpt2_modules(n_layers: int = 12) -> list[str]:
    return [f"transformer.h.{layer}.{m}" for layer in range(n_layers) for m in GPT2_MATRICES]


def test_every_gpt2_whole_model_matrix_resolves():
    for module in _gpt2_modules():
        assert resid_site_for_module(module)


def test_two_sites_per_block():
    sites = group_modules_by_resid_site(_gpt2_modules())
    assert len(sites) == 24, sorted(sites)
    for layer in range(12):
        assert sites[f"transformer.h.{layer}.ln_1"] == [
            f"transformer.h.{layer}.attn.c_attn.{p}_proj" for p in ("k", "q", "v")
        ]
        assert sites[f"transformer.h.{layer}.ln_2"] == [
            f"transformer.h.{layer}.attn.c_proj",
            f"transformer.h.{layer}.mlp.c_fc",
            f"transformer.h.{layer}.mlp.c_proj",
        ]


def test_qkv_read_resid_pre_and_the_output_matrix_reads_resid_mid():
    """The decision this registry encodes: q/k/v gate on what the block RECEIVES, the attention
    output matrix on the stream it WRITES INTO.
    """
    for p in ("q", "k", "v"):
        assert resid_site_for_module(f"transformer.h.5.attn.c_attn.{p}_proj") == (
            "transformer.h.5.ln_1"
        )
    assert resid_site_for_module("transformer.h.5.attn.c_proj") == "transformer.h.5.ln_2"


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("transformer.h.5.attn.c_attn.q_proj", False),
        ("transformer.h.5.attn.c_proj", True),
        ("transformer.h.5.mlp.c_fc", False),
        ("transformer.h.5.mlp.c_proj", False),
        ("model.layers.13.self_attn.o_proj", True),
        ("model.layers.13.mlp.down_proj", False),
    ],
)
def test_runs_after_is_exactly_the_attention_output_matrix(module: str, expected: bool):
    """`resid_mid = resid_pre + o_proj(z)`, so the norm holding it executes after `o_proj` and
    before everything else that reads it. Only that one matrix cannot read its own capture.
    """
    assert resid_site_runs_after(module) is expected


def test_no_other_entry_claims_to_run_after():
    after = {t for t, e in RESID_SITES.items() if e.runs_after}
    assert after == {
        "transformer.h.{layer}.attn.c_proj",
        "model.layers.{layer}.self_attn.o_proj",
    }


def test_project_2_configs_still_resolve_to_the_site_they_declare():
    assert resid_site_for_module("transformer.h.0.mlp.c_fc") == "transformer.h.0.ln_2"
    assert (
        resid_site_for_module("model.layers.13.mlp.down_proj")
        == "model.layers.13.pre_feedforward_layernorm"
    )


def test_an_unknown_module_is_refused():
    with pytest.raises(AssertionError, match="no residual-stream site known"):
        resid_site_for_module("transformer.h.0.attn.something_else")


def test_a_non_numeric_layer_is_refused():
    """The templates are matched by prefix/suffix, so `h.{layer}` would otherwise happily accept a
    wildcard pattern and format it into a path no module has.
    """
    with pytest.raises(AssertionError, match="no residual-stream site known"):
        resid_site_for_module("transformer.h.{layer}.mlp.c_fc")


def test_qwen3_o_proj_resolves_to_its_own_norm():
    """Qwen3's `hidden = residual + attn_out` has no norm inside the attention branch, so the
    pre-MLP norm's input IS `resid_mid` and the reconstruction is a bare sum -- the GPT-2 `c_proj`
    shape, not Gemma-2's.
    """
    entry = resid_site_entry("model.layers.17.self_attn.o_proj", "Qwen3ForCausalLM")
    assert entry.site == "model.layers.17.post_attention_layernorm"
    assert entry.runs_after is True
    assert entry.add_from == "model.layers.17.input_layernorm"


def test_the_no_arch_answer_on_the_shared_path_is_still_gemmas():
    module = "model.layers.13.self_attn.o_proj"
    assert resid_site_for_module(module) == "model.layers.13.pre_feedforward_layernorm"
    assert resid_site_entry(module) == resid_site_entry(module, "Gemma2ForCausalLM")


def test_the_gemma_overlay_duplicates_its_flat_entries_exactly():
    """Gemma-2 is listed in the overlay AND in the flat table, so that a caller which does pass an
    arch gets an answer for either model rather than a hole for one of them. Drift between the two
    copies would make the same target resolve differently depending on whether the caller happened
    to know its architecture.
    """
    flat = {k: v for k, v in RESID_SITES.items() if k.startswith("model.layers.")}
    assert ARCH_RESID_SITES["Gemma2ForCausalLM"] == flat


def test_an_architecture_with_no_overlay_falls_through_to_the_flat_table():
    """GPT-2 and `LlamaSimple` spell paths no other architecture claims, so there is nothing for an
    overlay to disambiguate and requiring one would mean a table per model that only restates
    `RESID_SITES`.
    """
    assert resid_site_for_module(
        "transformer.h.0.mlp.c_fc", "GPT2LMHeadModel"
    ) == "transformer.h.0.ln_2"
    assert resid_site_for_module("h.0.mlp.c_fc", "LlamaSimpleForCausalLM") == "h.0.rms_2"


def test_an_unknown_module_still_names_the_architecture_it_was_asked_about():
    with pytest.raises(AssertionError, match="Qwen3ForCausalLM"):
        resid_site_entry("model.layers.3.mlp.nonexistent_proj", "Qwen3ForCausalLM")
