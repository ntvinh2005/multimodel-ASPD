"""SAE site resolution, input/output adjacency, SAE-pair training and freezing."""

import itertools
import json

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aspd.sae.sites import (
    OutputCapture,
    SitePair,
    gpt2_mlp_c_fc,
    gpt2_mlp_c_fc_resid,
    resolve_sites,
    site_widths,
)
from aspd.sae.train import (
    build_sae_pair,
    evaluate_sae_pair,
    load_sae_pair,
    save_sae_pair,
    train_sae_pair,
)

N_EMBD, VOCAB = 32, 64


@pytest.fixture
def model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    cfg = GPT2Config(
        n_embd=N_EMBD, n_layer=2, n_head=2, vocab_size=VOCAB, n_positions=32, n_ctx=32
    )
    m = GPT2LMHeadModel(cfg)
    m.eval()
    return m


def _tokens(n: int = 4, seq: int = 16):
    """Infinite stream of token batches."""
    g = torch.Generator().manual_seed(1)
    while True:
        yield torch.randint(0, VOCAB, (n, seq), generator=g)


def _logits_forward(m: GPT2LMHeadModel):

    class _Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = m

        def forward(self, ids):
            return self.inner(input_ids=ids).logits

        def get_submodule(self, path):
            return self.inner.get_submodule(path)

    return _Wrapped()


# ---- sites -----------------------------------------------------------------------------------


def test_adjacency_is_enforced_by_construction():
    with pytest.raises(AssertionError, match="adjacency"):
        SitePair(module="a.b", input_site="a.ln", output_site="a.c")


def test_gpt2_site_pair_resolves(model):
    sites = gpt2_mlp_c_fc(0)
    assert sites.module == "transformer.h.0.mlp.c_fc"
    assert sites.input_site == "transformer.h.0.ln_2"
    resolve_sites(model, sites)


def test_unresolvable_site_fails_fast(model):
    with pytest.raises(AttributeError):
        resolve_sites(model, gpt2_mlp_c_fc(99))


def test_site_widths_read_from_a_real_forward(model):
    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc(0)
    widths = site_widths(wrapped, sites, torch.randint(0, VOCAB, (2, 8)))
    assert widths[sites.input_site] == N_EMBD
    assert widths[sites.output_site] == 4 * N_EMBD  # GPT2 MLP expands 4x


def test_capture_rejects_non_tensor_sites():
    """Some architectures' blocks return tuples; storing one as an activation would fail later
    and far away. Tested against a synthetic module rather than a real block, because whether
    GPT2Block returns a tuple or a bare tensor is `transformers`-version-dependent.
    """

    class TupleReturner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.inner(x), None

    class Host(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = TupleReturner()

        def forward(self, x):
            return self.block(x)[0]

    host = Host()
    with pytest.raises(AssertionError, match="expected Tensor"), OutputCapture(host, ["block"]):
        host(torch.randn(2, 4))


def test_capture_preserves_gradient_by_default(model):
    """`L_feat_recon` needs grad to flow from output features back through the matrix."""
    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc(0)
    with OutputCapture(wrapped, [sites.output_site]) as cap:
        wrapped(torch.randint(0, VOCAB, (2, 8)))
        assert cap[sites.output_site].requires_grad
    with OutputCapture(wrapped, [sites.output_site], detach=True) as cap:
        wrapped(torch.randint(0, VOCAB, (2, 8)))
        assert not cap[sites.output_site].requires_grad


def test_capture_removes_hooks_on_exit(model):
    wrapped = _logits_forward(model)
    target = wrapped.get_submodule("transformer.h.0.mlp.c_fc")
    before = len(target._forward_hooks)
    with OutputCapture(wrapped, ["transformer.h.0.mlp.c_fc"]):
        assert len(target._forward_hooks) == before + 1
    assert len(target._forward_hooks) == before


# ---- pair construction and training -----------------------------------------------------------


def test_build_sae_pair_widths_are_equal_and_sized_off_the_narrower_site(model):
    """On a `c_fc` the narrower site IS the input, so this pins the rule and the coincidence at
    once -- see `test_gemma_down_proj_sites.py` for the target where they come apart.
    """
    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc(0)
    saes = build_sae_pair(
        wrapped, sites, torch.randint(0, VOCAB, (2, 8)), feature_multiplier=8, device="cpu"
    )
    f_in = saes[sites.input_site].cfg.n_features
    f_out = saes[sites.output_site].cfg.n_features
    assert f_in == f_out == 8 * min(N_EMBD, 4 * N_EMBD)
    assert saes[sites.input_site].cfg.d_in == N_EMBD
    assert saes[sites.output_site].cfg.d_in == 4 * N_EMBD


def test_build_rejects_undercomplete_output_dictionary(model):
    wrapped = _logits_forward(model)
    with pytest.raises(AssertionError, match="undercomplete"):
        build_sae_pair(
            wrapped,
            gpt2_mlp_c_fc(0),
            torch.randint(0, VOCAB, (2, 8)),
            feature_multiplier=2,  # 2*32 = 64 < 128 = d_out
            device="cpu",
        )


def test_training_reduces_fvu_and_gate_roundtrips(model, tmp_path):
    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc(0)
    stream = _tokens()
    saes = build_sae_pair(
        wrapped,
        sites,
        torch.randint(0, VOCAB, (2, 8)),
        feature_multiplier=8,
        device="cpu",
        top_k=8,
        n_batches_to_dead=5,
    )

    before = evaluate_sae_pair(
        wrapped, sites, itertools.islice(_tokens(), 3), saes, n_batches=3, device="cpu"
    )
    train_sae_pair(
        wrapped, sites, stream, saes, n_tokens=4 * 16 * 60, device="cpu", log_every=10_000
    )
    after = evaluate_sae_pair(
        wrapped, sites, itertools.islice(_tokens(), 3), saes, n_batches=3, device="cpu"
    )

    for site in (sites.input_site, sites.output_site):
        assert after[site]["fvu"] < before[site]["fvu"], f"{site} FVU did not improve"
        assert 0.0 <= after[site]["dead_frac"] <= 1.0
        assert after[site]["mean_l0"] > 0

    save_sae_pair(saes, after, tmp_path)
    assert json.loads((tmp_path / "sae_report.json").read_text()) == after

    reloaded = load_sae_pair(sites, tmp_path, device="cpu")
    x = torch.randn(5, N_EMBD)
    assert torch.allclose(
        reloaded[sites.input_site].features(x), saes[sites.input_site].features(x)
    )
    assert not any(p.requires_grad for p in reloaded[sites.input_site].parameters())


def test_train_sae_pair_refuses_zero_tokens():
    """`n_tokens=0` made the training loop a no-op while the caller went on to evaluate and SAVE
    a randomly-initialized pair, plus a `sae_report.json` that every later run then trusted.
    """
    import pytest

    from aspd.sae.train import train_sae_pair

    with pytest.raises(AssertionError, match="n_tokens=0"):
        train_sae_pair(None, None, iter([]), {}, n_tokens=0)


# ---- forward truncation ------------------------------------------------------------------------


def test_truncated_forward_stops_after_the_last_site(model):
    """The sites bracket one matrix; everything after them is computed and discarded."""
    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc(0)
    paths = [sites.input_site, sites.output_site]
    ran: list[str] = []
    handles = [
        model.get_submodule(f"transformer.h.{i}").register_forward_hook(
            lambda _m, _a, _o, i=i: ran.append(f"h{i}")
        )
        for i in (0, 1)
    ]
    try:
        with torch.no_grad(), OutputCapture(
            wrapped, paths, detach=True, stop_when_complete=True
        ) as cap:
            cap.run(torch.randint(0, VOCAB, (2, 8)))
            assert set(cap.acts) == set(paths)
        # h0 never completes -- we unwind from inside it, at c_fc -- and h1 never starts.
        assert ran == []
    finally:
        for h in handles:
            h.remove()


def test_truncated_forward_matches_untruncated_activations(model):
    """Truncation must change only what is computed after the sites, never the sites themselves."""
    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc(0)
    paths = [sites.input_site, sites.output_site]
    ids = torch.randint(0, VOCAB, (2, 8))

    with torch.no_grad(), OutputCapture(wrapped, paths, detach=True) as cap:
        wrapped(ids)
        full = {p: cap[p].clone() for p in paths}
    with torch.no_grad(), OutputCapture(
        wrapped, paths, detach=True, stop_when_complete=True
    ) as cap:
        cap.run(ids)
        truncated = {p: cap[p].clone() for p in paths}

    for p in paths:
        torch.testing.assert_close(full[p], truncated[p])


def test_truncated_forward_rejects_a_site_that_never_fires():
    """A site that exists in the module tree but is never called leaves a partial capture."""

    class Host(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(4, 4)
            self.dead = torch.nn.Linear(4, 4)  # in the tree, never in the forward

        def forward(self, x):
            return self.used(x)

    host = Host()
    with pytest.raises(AssertionError, match="never reached"), torch.no_grad(), OutputCapture(
        host, ["used", "dead"], detach=True, stop_when_complete=True
    ) as cap:
        cap.run(torch.randn(2, 4))


# ---- micro-batching ----------------------------------------------------------------------------


def test_sae_batch_tokens_sets_the_optimizer_step_count(model):
    """`sae_batch_tokens` splits ONE target forward into several SAE steps -- token accounting
    must be unchanged, step count must scale.
    """
    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc(0)
    n_tokens = 4 * 16 * 8  # 8 target forwards of 64 tokens

    counts = {}
    for chunk in (64, 16):
        saes = build_sae_pair(
            wrapped, sites, torch.randint(0, VOCAB, (2, 8)),
            feature_multiplier=8, device="cpu", top_k=4,
        )
        hist = train_sae_pair(
            wrapped, sites, _tokens(), saes, n_tokens=n_tokens,
            sae_batch_tokens=chunk, device="cpu", log_every=1,
        )
        counts[chunk] = hist[-1]

    # Same tokens consumed either way; 4x the optimizer steps at 1/4 the chunk.
    assert counts[64]["tokens"] == counts[16]["tokens"] == n_tokens
    assert counts[16]["step"] == 4 * counts[64]["step"]


def test_batch_topk_pool_is_the_micro_batch_not_the_forward(model):
    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc(0)
    saes = build_sae_pair(
        wrapped, sites, torch.randint(0, VOCAB, (2, 8)),
        feature_multiplier=8, device="cpu", top_k=4,
    )
    sae = saes[sites.input_site]
    acts = torch.randn(64, N_EMBD)
    for chunk in (64, 16):
        l0 = torch.stack([
            sae.loss(acts[lo : lo + chunk])["l0_norm"] for lo in range(0, 64, chunk)
        ]).mean()
        assert abs(l0.item() - 4.0) < 1e-6


# ---- training dtype ----------------------------------------------------------------------------


def test_dtype_applies_to_weights_but_not_the_threshold(model):
    """The EMA threshold is an accumulator over minima and the dead counter is an exact tally;
    neither may inherit a low-mantissa training dtype.
    """
    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc(0)
    saes = build_sae_pair(
        wrapped, sites, torch.randint(0, VOCAB, (2, 8)),
        feature_multiplier=8, device="cpu", dtype=torch.bfloat16,
    )
    sae = saes[sites.input_site]
    assert sae.W_enc.dtype == sae.W_dec.dtype == sae.b_dec.dtype == torch.bfloat16
    assert sae.threshold.dtype == sae.n_batches_not_active.dtype == torch.float32


def test_dtype_survives_the_save_load_roundtrip(model, tmp_path):
    """Without the recorded dtype a bf16 pair reloads as fp32 -- `load_state_dict` casts into
    whatever the fresh module already is -- so the frozen extractor would not be the gated one.
    """
    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc(0)
    saes = build_sae_pair(
        wrapped, sites, torch.randint(0, VOCAB, (2, 8)),
        feature_multiplier=8, device="cpu", dtype=torch.bfloat16,
    )
    save_sae_pair(saes, {"x": {"fvu": 0.0}}, tmp_path)
    reloaded = load_sae_pair(sites, tmp_path, device="cpu")
    assert reloaded[sites.input_site].W_enc.dtype == torch.bfloat16
    assert reloaded[sites.input_site].threshold.dtype == torch.float32


# ---- config-vs-artifact guard ------------------------------------------------------------------


def _sae_run_cfg(tmp_path, **train_overrides):
    from aspd.sae.config import SAERunConfig

    return SAERunConfig.model_validate(
        {
            "experiment_config": "configs/x.yaml",
            "sae_dir": str(tmp_path),
            "dictionary": {"feature_multiplier": 8},
            "train": {"n_tokens": 1000, **train_overrides},
        }
    )


def _d_ref(saes) -> int:
    """What `train_or_load` passes the guard: the pair's reference (narrower) width."""
    from aspd.sae.train import reference_width

    return reference_width(s.cfg.d_in for s in saes.values())


def _saved_pair(model, tmp_path, *, dtype=torch.float32, write_provenance=True, **train_over):
    import yaml

    from aspd.sae.train import save_sae_pair

    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc(0)
    cfg = _sae_run_cfg(tmp_path, **train_over)
    saes = build_sae_pair(
        wrapped, sites, torch.randint(0, VOCAB, (2, 8)), device="cpu", dtype=dtype,
        **cfg.dictionary.model_dump(exclude={"feature_multiplier"}),
        feature_multiplier=cfg.dictionary.feature_multiplier,
    )
    save_sae_pair(saes, {"x": {"fvu": 0.0}}, tmp_path)
    if write_provenance:
        (tmp_path / "sae_config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    return sites, load_sae_pair(sites, tmp_path, device="cpu")


def test_guard_accepts_a_pair_its_own_config_produced(model, tmp_path):
    from aspd.sae.validate import require_pair_matches_config

    sites, saes = _saved_pair(model, tmp_path)
    require_pair_matches_config(
        tmp_path, _sae_run_cfg(tmp_path), saes[sites.input_site], _d_ref(saes)
    )


def test_guard_catches_an_edited_dtype(model, tmp_path):
    from aspd.sae.validate import require_pair_matches_config

    sites, saes = _saved_pair(model, tmp_path, dtype=torch.float32)
    cfg = _sae_run_cfg(tmp_path, dtype="bfloat16")
    with pytest.raises(AssertionError, match="dtype: on disk torch.float32"):
        require_pair_matches_config(tmp_path, cfg, saes[sites.input_site], _d_ref(saes))


def test_guard_catches_a_training_field_with_no_trace_in_the_weights(model, tmp_path):
    """`sae_batch_tokens` defines the BatchTopK pool but leaves no mark on the saved tensors --
    only `sae_config.yaml` can catch it.
    """
    from aspd.sae.validate import require_pair_matches_config

    sites, saes = _saved_pair(model, tmp_path, sae_batch_tokens=2048)
    cfg = _sae_run_cfg(tmp_path, sae_batch_tokens=4096)
    with pytest.raises(AssertionError, match="train.sae_batch_tokens: pair trained with 2048"):
        require_pair_matches_config(tmp_path, cfg, saes[sites.input_site], _d_ref(saes))


def test_guard_catches_an_edited_dictionary_field(model, tmp_path):
    from aspd.sae.validate import require_pair_matches_config

    sites, saes = _saved_pair(model, tmp_path)
    cfg = _sae_run_cfg(tmp_path)
    object.__setattr__(cfg.dictionary, "top_k", 999)
    with pytest.raises(AssertionError, match="top_k: on disk"):
        require_pair_matches_config(tmp_path, cfg, saes[sites.input_site], _d_ref(saes))


def test_pre_provenance_pair_warns_but_still_checks_the_weights(model, tmp_path, capsys):
    """A pair saved before `sae_config.yaml` existed cannot be fully verified. Say so -- and still
    check what the checkpoint does carry, rather than waving the whole thing through.
    """
    from aspd.sae.validate import require_pair_matches_config

    sites, saes = _saved_pair(model, tmp_path, write_provenance=False)
    require_pair_matches_config(
        tmp_path, _sae_run_cfg(tmp_path), saes[sites.input_site], _d_ref(saes)
    )
    assert "predates config provenance" in capsys.readouterr().out

    with pytest.raises(AssertionError, match="dtype"):
        require_pair_matches_config(
            tmp_path,
            _sae_run_cfg(tmp_path, dtype="bfloat16"),
            saes[sites.input_site],
            _d_ref(saes),
        )


def test_load_only_path_warns_when_the_pair_is_unidentifiable(tmp_path, capsys):
    from aspd.sae.validate import warn_if_unverifiable

    warn_if_unverifiable(tmp_path)
    assert "cannot be reproduced" in capsys.readouterr().out
    (tmp_path / "sae_config.yaml").write_text("train: {}\n")
    warn_if_unverifiable(tmp_path)
    assert capsys.readouterr().out == ""


def test_train_paths_holds_the_donated_half_completely_still(model, tmp_path):
    """The residual pair adopts `gpt2s_h0_c_fc`'s output dictionary rather than retraining it, so
    that every output-side column stays comparable across the site change. `requires_grad=False`
    alone is NOT enough to hold it still: `loss()` also advances the BatchTopK threshold EMA and
    the dead-latent counter as side effects, which would keep mutating the very dictionary the
    donation exists to preserve. Excluding it from the loss is what actually freezes it.
    """
    wrapped = _logits_forward(model)
    sites = gpt2_mlp_c_fc_resid(0)
    stream = _tokens()
    saes = build_sae_pair(
        wrapped,
        sites,
        torch.randint(0, VOCAB, (2, 8)),
        feature_multiplier=8,
        device="cpu",
        top_k=8,
        n_batches_to_dead=5,
    )
    out, inp = saes[sites.output_site], saes[sites.input_site]
    frozen_ref = {k: v.clone() for k, v in out.state_dict().items()}
    moving_ref = inp.W_enc.clone()

    train_sae_pair(
        wrapped,
        sites,
        stream,
        saes,
        n_tokens=4 * 16 * 20,
        device="cpu",
        log_every=10_000,
        train_paths=[sites.input_site],
    )

    for k, v in out.state_dict().items():
        assert torch.equal(v, frozen_ref[k]), f"donated half moved on {k}"
    assert not torch.allclose(inp.W_enc, moving_ref), "the trained half did not move"


def test_the_resid_pair_captures_the_residual_stream_not_the_normalized_input(model):
    """The site change is real: `ln_2` is the only thing between the two tensors."""
    wrapped = _logits_forward(model)
    tokens = torch.randint(0, VOCAB, (2, 8))
    resid, adjacent = gpt2_mlp_c_fc_resid(0), gpt2_mlp_c_fc(0)
    with torch.no_grad(), OutputCapture(
        wrapped,
        [resid.input_site],
        detach=True,
        takes={resid.input_site: resid.input_take},
    ) as cap_in:
        cap_in.run(tokens)
    with torch.no_grad(), OutputCapture(
        wrapped,
        [adjacent.input_site],
        detach=True,
        takes={adjacent.input_site: adjacent.input_take},
    ) as cap_out:
        cap_out.run(tokens)

    ln = wrapped.get_submodule(resid.input_site)
    with torch.no_grad():
        assert torch.allclose(cap_out[adjacent.input_site], ln(cap_in[resid.input_site]), atol=1e-6)
    assert not torch.allclose(cap_out[adjacent.input_site], cap_in[resid.input_site], atol=1e-3)


def test_site_widths_follows_the_take(model):
    """Both sites are `[..., N_EMBD]` here, so this is about the plumbing, not the number."""
    wrapped = _logits_forward(model)
    probe = torch.randint(0, VOCAB, (2, 8))
    resid = gpt2_mlp_c_fc_resid(0)
    widths = site_widths(wrapped, resid, probe)
    assert widths[resid.input_site] == N_EMBD
    assert widths[resid.output_site] == 4 * N_EMBD
