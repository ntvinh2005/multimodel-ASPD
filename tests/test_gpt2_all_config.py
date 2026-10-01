"""The model-wide ASPD config (`configs/gpt2_all/aspd.yaml`): it launches as ASPD, has one L_act per
shared encoder and one L_internal per matrix, and maps every matrix to its residual site."""

from pathlib import Path

import pytest

pytest.importorskip("param_decomp_lab")

CONFIG = Path(__file__).resolve().parents[1] / "configs/gpt2_all/aspd.yaml"


def _cfg():
    from aspd.config import LMInterpExperimentConfig

    return LMInterpExperimentConfig.from_file(CONFIG)


def test_the_config_launches_as_aspd():
    from aspd.arms import assert_config, derive_arm_name

    cfg = _cfg()
    assert_config(cfg)
    assert derive_arm_name(cfg) == "aspd"


def test_one_l_act_per_encoder_and_one_l_internal_per_matrix():
    from aspd.arms import aspd_heads, resid_site_map

    cfg = _cfg()
    sites = resid_site_map(cfg)
    heads_act, heads_internal = aspd_heads(cfg)
    assert len(sites) == 72
    assert len(set(sites.values())) == 24
    assert len(heads_act) == (24 if cfg.pd.ci_config.share_encoders else 72)
    assert len(heads_internal) == 72


def test_q_k_v_read_resid_pre_and_o_and_mlp_read_resid_mid():
    from aspd.arms import resid_site_map

    sites = resid_site_map(_cfg())
    for layer in range(12):
        assert sites[f"transformer.h.{layer}.attn.c_proj"] == f"transformer.h.{layer}.ln_2"
        assert sites[f"transformer.h.{layer}.mlp.c_fc"] == f"transformer.h.{layer}.ln_2"
        assert sites[f"transformer.h.{layer}.mlp.c_proj"] == f"transformer.h.{layer}.ln_2"
        for p in ("q", "k", "v"):
            assert sites[f"transformer.h.{layer}.attn.c_attn.{p}_proj"] == f"transformer.h.{layer}.ln_1"
