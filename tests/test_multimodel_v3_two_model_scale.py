from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from aspd.multimodel.cli.train_aspd_v3_2b import _common_state_sha256
from aspd.multimodel.v3.data import SQLiteExposureLedger
from aspd.multimodel.v3.encoder import pack_cross_model_tokens
from aspd.multimodel.v3.objective import matryoshka_activation_fvu
from aspd.multimodel.v3.reconstruction import ActivationDecoder
from aspd.multimodel.v3.registry import build_model_registry_2b
from aspd.multimodel.v3.two_model_config import load_two_model_scale_config

ROOT = Path(__file__).parents[1]
BASELINE = ROOT / "configs/multimodel/aspd_v3/two_model_scale_baseline_v1.yaml"
TREATMENT = ROOT / "configs/multimodel/aspd_v3/two_model_scale_treatment_v1.yaml"


def test_two_model_configs_preserve_scope_and_differ_only_by_arm_scale_and_paths() -> None:
    baseline = load_two_model_scale_config(BASELINE)
    treatment = load_two_model_scale_config(TREATMENT)
    assert tuple(model.key for model in baseline.models) == ("gemma", "smol")
    assert baseline.architecture.dictionary_size == 8192
    assert baseline.architecture.batch_topk_k == 32
    assert baseline.data.optimizer_steps == 1221
    assert baseline.data.accepted_anchor_exposures == 10_002_432
    assert (
        sum(len(layout.targets()) for layout in build_model_registry_2b(baseline).values()) == 112
    )

    left = baseline.payload()
    right = treatment.payload()
    left.pop("arm")
    right.pop("arm")
    left["objective"].pop("activation_reconstruction_scale")
    right["objective"].pop("activation_reconstruction_scale")
    for key in ("output_root", "preflight_evidence"):
        left["runtime"].pop(key)
        right["runtime"].pop(key)
    assert left == right


def test_two_model_cross_packing_has_one_separator_and_three_identity_channels() -> None:
    gemma = torch.randn(2, 3, 8, 5)
    smol = torch.randn(2, 3, 8, 4)
    packed = pack_cross_model_tokens({"gemma": gemma, "smol": smol}, model_order=("gemma", "smol"))
    assert packed.values.shape == (2, 3, 17, 8)
    assert packed.separator_positions == (8,)
    assert packed.identity_indices == (0,) * 8 + (2,) + (1,) * 8


def test_reconstruction_scale_preserves_bias_and_scales_every_prefix() -> None:
    torch.manual_seed(3)
    decoder = ActivationDecoder(16, 3, bias=True)
    code = torch.randn(1, 4, 16)
    target = torch.randn(1, 4, 3)
    valid = torch.ones(1, 4, dtype=torch.bool)
    baseline_loss, baseline_full = matryoshka_activation_fvu(code, decoder, target, valid)
    unit_loss, unit_full = matryoshka_activation_fvu(
        code, decoder, target, valid, reconstruction_scale=torch.tensor(1.0)
    )
    torch.testing.assert_close(unit_loss, baseline_loss, rtol=0, atol=0)
    torch.testing.assert_close(unit_full, baseline_full, rtol=0, atol=0)

    scale = torch.tensor(2.0, requires_grad=True)
    _loss, scaled_full = matryoshka_activation_fvu(
        code, decoder, target, valid, reconstruction_scale=scale
    )
    bias = decoder.bias.detach().reshape(1, 1, -1)
    torch.testing.assert_close(scaled_full - bias, 2 * (baseline_full - bias))
    _loss.backward()
    assert scale.grad is not None and torch.isfinite(scale.grad)


def test_two_model_log_scale_has_exact_reciprocal_gauge_and_signed_gradients() -> None:
    rho = torch.tensor(0.3, requires_grad=True)
    gemma = torch.exp(rho)
    smol = torch.exp(-rho)
    torch.testing.assert_close(gemma * smol, torch.tensor(1.0), rtol=1e-6, atol=0)
    (2 * gemma + smol).backward()
    expected = 2 * torch.exp(torch.tensor(0.3)) - torch.exp(torch.tensor(-0.3))
    torch.testing.assert_close(rho.grad, expected)


def test_config_rejects_unapproved_dictionary_or_k_change() -> None:
    import yaml

    payload = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    for field, value in (("dictionary_size", 4096), ("batch_topk_k", 16)):
        changed = copy.deepcopy(payload)
        changed["architecture"][field] = value
        with pytest.raises(ValidationError):
            from aspd.multimodel.v3.two_model_config import TwoModelScalePilotConfig

            TwoModelScalePilotConfig.model_validate(changed)


def test_common_state_hash_supports_scalar_integer_buffers() -> None:
    module = nn.Module()
    module.register_buffer("scalar", torch.tensor(7, dtype=torch.int64))
    first = _common_state_sha256(module)
    second = _common_state_sha256(module)
    assert first == second and len(first) == 64


def test_two_model_exposure_ledger_preserves_named_raw_token_counts(tmp_path: Path) -> None:
    ledger = SQLiteExposureLedger(
        tmp_path / "ledger.sqlite",
        dataset_manifest_sha256="a" * 64,
        world_size=2,
        model_order=("gemma", "smol"),
        compact_anchor_ids=True,
    )
    ledger.observe_raw_tokens("document", {"gemma": 11, "smol": 13})
    assert ledger.raw_token_counts() == {"gemma": 11, "smol": 13}
    ledger.observe_replicated_optimizer_step(0, ("anchor-a", "anchor-b"))
    assert ledger.checkpoint_receipt(expected_optimizer_steps=1).world_size == 2
    assert ledger.audit_full_consistency()["status"] == "PASS"
    ledger.close()
