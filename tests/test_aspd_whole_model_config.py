"""Config checks of a model-wide ASPD run: one L_act per encoder, one L_internal per matrix."""

import copy
from pathlib import Path

import pytest

pytest.importorskip("param_decomp_lab")

import yaml

from aspd.arms import aspd_heads, resid_site_map
from aspd.config import LMInterpExperimentConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BASE = CONFIGS / "gpt2/aspd.yaml"

QKV = "transformer.h.0.attn.c_attn.q_proj"
OUT = "transformer.h.0.attn.c_proj"
FC = "transformer.h.0.mlp.c_fc"
MODULES = [QKV, OUT, FC]
SITE_PRE, SITE_MID = "transformer.h.0.ln_1", "transformer.h.0.ln_2"
C = 24576
# One per site, in `MODULES` order: q_proj covers `ln_1`, the attention output matrix covers `ln_2`.
REPRESENTATIVES = [QKV, OUT]


def _slug(module: str) -> str:
    return module.removeprefix("transformer.").replace(".", "_").replace("h_", "h", 1)


def _whole_model_dict(
    *,
    modules: list[str] = MODULES,
    head_r_modules: list[str] | None = None,
    resid_sites: dict[str, str] | None = None,
) -> dict:
    """The shipped single-matrix config, widened to several matrices sharing encoders by site."""
    raw = yaml.safe_load(BASE.read_text())
    raw = copy.deepcopy(raw)
    pd = raw["pd"]

    pd["decomposition_targets"] = [{"module_pattern": m, "C": C} for m in modules]
    gate = pd["ci_config"]
    gate.pop("resid_site", None)
    gate["resid_site"] = ""
    gate["resid_sites"] = (
        resid_sites
        if resid_sites is not None
        else {m: (SITE_PRE if ".attn.c_attn." in m else SITE_MID) for m in modules}
    )

    keep = [
        e
        for e in pd["loss_metrics"]
        if e["type"] not in ("InternalReconLoss", "ActivationReconLoss", "AuxKLoss")
    ]
    head_s_template = next(
        e for e in pd["loss_metrics"] if e["type"] == "InternalReconLoss" and e["mode"] == "fvu"
    )
    head_r_template = next(e for e in pd["loss_metrics"] if e["type"] == "ActivationReconLoss")
    auxk_template = next(e for e in pd["loss_metrics"] if e["type"] == "AuxKLoss")

    entries = list(keep)
    for module in modules:
        head_s = copy.deepcopy(head_s_template)
        head_s["name"] = f"ci_recon_{_slug(module)}"
        head_s["module"] = module
        entries.append(head_s)
    if head_r_modules is None:
        head_r_modules = REPRESENTATIVES
    for module in head_r_modules:
        head_r = copy.deepcopy(head_r_template)
        head_r["name"] = f"gate_recon_{_slug(module)}"
        head_r["module"] = module
        entries.append(head_r)
        auxk = copy.deepcopy(auxk_template)
        auxk["name"] = f"auxk_{_slug(module)}"
        auxk["module"] = module
        auxk["recon"] = head_r["name"]
        entries.append(auxk)
    pd["loss_metrics"] = entries
    return raw


def _cfg(**over) -> LMInterpExperimentConfig:
    return LMInterpExperimentConfig.model_validate(_whole_model_dict(**over))


# ---- the shape a whole-model run has --------------------------------------------------------


def test_the_widened_config_parses_and_maps_every_module_to_its_site():
    cfg = _cfg()
    assert resid_site_map(cfg) == {QKV: SITE_PRE, OUT: SITE_MID, FC: SITE_MID}


def test_l_act_is_per_site_and_l_internal_per_matrix():
    heads_r, heads_s = aspd_heads(_cfg())
    assert sorted(heads_r) == [SITE_PRE, SITE_MID]
    assert sorted(heads_s) == sorted(MODULES)


def test_the_shipped_single_matrix_configs_still_pass():
    """`resid_site` scalar, `resid_sites` absent, L_act and L_internal with an unset `module`.
    """
    for name in ("gpt2/aspd.yaml", "gemma2/aspd.yaml", "qwen3/aspd.yaml"):
        cfg = LMInterpExperimentConfig.from_file(CONFIGS / name)
        heads_r, heads_s = aspd_heads(cfg)
        assert len(heads_r) == 1 and len(heads_s) == 1


# ---- what is refused --------------------------------------------------------------------------


def test_two_l_act_entries_at_one_site_are_refused():
    """The mistake a per-matrix generator makes. Charging one encoder's reconstruction twice
    doubles its weight against L_internal, with nothing in any log to say so.
    """
    with pytest.raises(AssertionError, match="two L_act entries train the encoder"):
        aspd_heads(_cfg(head_r_modules=[QKV, OUT, FC]))


def test_a_site_with_no_l_act_is_refused():
    """Its encoder would get no gradient at all -- the dictionary stays at its init while every
    matrix it gates trains against it.
    """
    with pytest.raises(AssertionError, match="L_act covers 1 of 2 encoders"):
        aspd_heads(_cfg(head_r_modules=[FC]))


def test_a_matrix_with_no_l_internal_is_refused():
    raw = _whole_model_dict()
    raw["pd"]["loss_metrics"] = [
        e for e in raw["pd"]["loss_metrics"] if e.get("name") != f"ci_recon_{_slug(OUT)}"
    ]
    with pytest.raises(AssertionError, match="L_internal covers 2 of 3 matrices"):
        aspd_heads(LMInterpExperimentConfig.model_validate(raw))


def test_a_gate_site_the_architecture_does_not_have_is_refused():
    """The attention OUTPUT matrix reads `resid_mid`, not `resid_pre`; both are real norms of the
    same block, so nothing but the registry can tell them apart.
    """
    sites = {QKV: SITE_PRE, OUT: SITE_PRE, FC: SITE_MID}
    with pytest.raises(AssertionError, match="residual sites disagree"):
        resid_site_map(_cfg(resid_sites=sites))


def test_several_targets_with_only_the_scalar_gate_site_are_refused():
    raw = _whole_model_dict()
    raw["pd"]["ci_config"]["resid_sites"] = None
    raw["pd"]["ci_config"]["resid_site"] = SITE_MID
    with pytest.raises(AssertionError, match="not the scalar `resid_site`"):
        resid_site_map(LMInterpExperimentConfig.model_validate(raw))


def test_a_site_map_that_misses_a_target_is_refused():
    sites = {QKV: SITE_PRE, FC: SITE_MID}
    with pytest.raises(AssertionError, match="does not match the decomposition targets"):
        resid_site_map(_cfg(resid_sites=sites))


def test_an_unnamed_entry_is_refused_above_one_target():
    raw = _whole_model_dict()
    for entry in raw["pd"]["loss_metrics"]:
        if entry.get("name") == f"ci_recon_{_slug(FC)}":
            entry["module"] = ""
    with pytest.raises(AssertionError, match="names no `module`"):
        aspd_heads(LMInterpExperimentConfig.model_validate(raw))


def test_a_loss_at_coeff_zero_is_refused():
    raw = _whole_model_dict()
    for entry in raw["pd"]["loss_metrics"]:
        if entry.get("name") == f"gate_recon_{_slug(OUT)}":
            entry["coeff"] = 0.0
    with pytest.raises(AssertionError, match="at coeff 0"):
        aspd_heads(LMInterpExperimentConfig.model_validate(raw))


# ---- provenance --------------------------------------------------------------------------------


def test_the_arm_name_is_unchanged_by_going_whole_model():
    """A model-wide ASPD run derives the same method name as a single-matrix one.
    """
    from aspd.arms import derive_arm_name

    assert derive_arm_name(_cfg()) == "aspd"


def test_the_summary_states_the_sharing_factor():
    """A reader of the run log has to be able to see that one encoder serves several matrices --
    it is the difference between this arm and 72 independent dictionaries.
    """
    from aspd.arms import arm_summary

    summary = arm_summary(_cfg())
    assert "2 encoder(s) over 3 matrices" in summary
