from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target import (
    _load_contract,
)
from aspd.multimodel.cli.train_aspd_v3_2b import _load_training_config
from aspd.multimodel.v3.encoder import IntraModelTransformer
from aspd.multimodel.v3.encoder_auxk_factorial_config import (
    ARM_SETTINGS,
    SOURCE_CONFIG_FINGERPRINT,
    load_encoder_auxk_factorial_config,
)
from aspd.multimodel.v3.granite_normalized_config import (
    load_granite_normalized_target_config,
)
from aspd.multimodel.v3.model_owned import (
    IdentityTokenMixer,
    LocalSiteEncoder,
    SharedSiteEncoder,
)
from aspd.multimodel.v3.objective import (
    auxk_activation_scores,
    auxk_dead_feature_loss,
)
from aspd.multimodel.v3.reconstruction import ActivationDecoder

ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "configs/multimodel/aspd_v3"
CONFIGS = {
    "transformer_current_auxk": CONFIG_DIR / "encoder_auxk_transformer_current_v1.yaml",
    "linear_current_auxk": CONFIG_DIR / "encoder_auxk_linear_current_v1.yaml",
    "transformer_revival_auxk": CONFIG_DIR / "encoder_auxk_transformer_revival_v1.yaml",
    "linear_revival_auxk": CONFIG_DIR / "encoder_auxk_linear_revival_v1.yaml",
}


def test_factorial_configs_are_matched_and_fingerprinted_independently() -> None:
    loaded = {arm: load_encoder_auxk_factorial_config(path) for arm, path in CONFIGS.items()}
    assert len({config.fingerprint() for config in loaded.values()}) == 4
    for arm, config in loaded.items():
        mixer, score_source, temperature = ARM_SETTINGS[arm]
        assert config.arm_id == arm
        assert config.architecture.encoder_mixer == mixer
        assert config.objective.auxk_score_source == score_source
        assert config.objective.auxk_softplus_temperature == temperature
        assert config.source.fingerprint() == SOURCE_CONFIG_FINGERPRINT
        assert config.data.approved_optimizer_steps == 256
        assert config.data.approved_accepted_anchor_exposures == 2_097_152
        assert config.full_training_authorized is False
        assert _load_training_config(CONFIGS[arm]).fingerprint() == config.fingerprint()


def test_linear_arm_removes_only_intra_and_cross_transformer_mixers() -> None:
    linear = load_encoder_auxk_factorial_config(CONFIGS["linear_current_auxk"])
    transformer = load_encoder_auxk_factorial_config(CONFIGS["transformer_current_auxk"])
    local_linear = LocalSiteEncoder(4, linear, device="cpu", dtype=torch.float32)
    local_transformer = LocalSiteEncoder(4, transformer, device="cpu", dtype=torch.float32)
    shared_linear = SharedSiteEncoder(
        linear, model_order=("gemma", "smol", "granite"), device="cpu", dtype=torch.float32
    )

    assert isinstance(local_linear.intra, IdentityTokenMixer)
    assert isinstance(shared_linear.cross, IdentityTokenMixer)
    assert isinstance(local_transformer.intra, IntraModelTransformer)
    activations = tuple(torch.randn(1, 3, 4) for _ in range(8))
    assert local_linear(activations).shape == (1, 3, 8, 512)


def test_transformer_control_initialization_matches_corrected_source() -> None:
    control = load_encoder_auxk_factorial_config(CONFIGS["transformer_current_auxk"])
    source = load_granite_normalized_target_config(CONFIG_DIR / control.spec.source_config)
    torch.manual_seed(17)
    source_local = LocalSiteEncoder(4, source, device="cpu", dtype=torch.float32)
    torch.manual_seed(17)
    control_local = LocalSiteEncoder(4, control, device="cpu", dtype=torch.float32)
    assert source_local.state_dict().keys() == control_local.state_dict().keys()
    for key, value in source_local.state_dict().items():
        torch.testing.assert_close(value, control_local.state_dict()[key], rtol=0, atol=0)


def test_softplus_auxk_revives_negative_logits_without_changing_main_scores() -> None:
    raw = torch.tensor(
        [[[-0.1, -2.0, -3.0, -4.0]], [[-2.0, -0.1, -3.0, -4.0]]],
        requires_grad=True,
    )
    valid = torch.ones((2, 1), dtype=torch.bool)
    main = torch.relu(raw)
    historical = auxk_activation_scores(main, raw, valid, mode="post_relu")
    revived = auxk_activation_scores(
        main,
        raw,
        valid,
        mode="softplus_raw_logits",
        softplus_temperature=1.0,
    )
    assert historical is main
    torch.testing.assert_close(main, torch.zeros_like(main))
    assert bool((revived > 0).all())

    decoder = ActivationDecoder(4, 2, dtype=torch.float32)
    with torch.no_grad():
        decoder.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        )
        decoder.bias.zero_()
    target = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    loss = auxk_dead_feature_loss(
        revived,
        decoder,
        target,
        torch.zeros_like(target),
        valid,
        torch.ones(4, dtype=torch.bool),
        top_k_aux=1,
        sparse_backend="dense",
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert raw.grad is not None
    assert int(torch.count_nonzero(raw.grad)) > 0
    assert float(raw.grad.abs().sum()) > 0
    torch.testing.assert_close(main.detach(), torch.zeros_like(main))


def test_factorial_contract_is_bounded_and_read_only() -> None:
    contract = _load_contract(CONFIG_DIR / "encoder_auxk_factorial_evaluation_v1.yaml")
    assert contract["checkpoint_steps"] == 256
    assert contract["accepted_anchors"] == 20_480
    assert contract["arms"] == list(ARM_SETTINGS)
    assert contract["training_authorized_by_this_contract"] is False


def test_softplus_temperature_is_fail_closed() -> None:
    raw = torch.zeros(1, 1, 2)
    valid = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="positive fixed temperature"):
        auxk_activation_scores(
            torch.relu(raw),
            raw,
            valid,
            mode="softplus_raw_logits",
            softplus_temperature=0.0,
        )
