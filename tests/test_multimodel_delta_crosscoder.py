from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch

from aspd.multimodel.baselines.delta_crosscoder import (
    DeltaActivationSource,
    DeltaCrosscoder,
    delta_checkpoint_payload,
    delta_feature_diagnostics,
    delta_losses,
    delta_relative_decoder_norm_base,
    delta_relative_decoder_norm_target,
    load_delta_checkpoint,
    load_delta_crosscoder_config,
    save_delta_checkpoint,
)

ROOT = Path(__file__).parents[1]


class _FakeSource(DeltaActivationSource):
    def __init__(self, activation_dim: int) -> None:
        self._activation_dim = activation_dim

    @property
    def activation_dim(self) -> int:
        return self._activation_dim

    def iter_batches(
        self, split: str, *, shuffle: bool, epoch: int = 0
    ) -> Iterator[dict[str, torch.Tensor]]:
        del split, shuffle, epoch
        yield {"activations": torch.zeros(2, 2, self.activation_dim)}

    def provenance(self) -> dict[str, object]:
        return {"mode": "test"}


def _model() -> DeltaCrosscoder:
    torch.manual_seed(0)
    return DeltaCrosscoder(
        activation_dim=4,
        n_features=10,
        shared_fraction=0.2,
        k_shared=2,
        k_delta=1,
        norm_init_scale=0.4,
    )


def test_delta_target_is_target_minus_base_exactly() -> None:
    model = _model()
    base = torch.tensor([[1.0, -2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]])
    target = torch.tensor([[2.0, -5.0, 3.5, 1.0], [4.0, 3.0, 2.0, 1.0]])
    paired = torch.stack((base, target), dim=1)
    torch.testing.assert_close(model.delta_target(paired), target - base)


def test_delta_reconstruction_equals_biasless_masked_decoder_difference() -> None:
    model = _model()
    features = torch.rand(3, model.n_features)
    masked = model.masked_delta_features(features)
    decoded = model.decode(masked, add_bias=False)
    torch.testing.assert_close(
        model.delta_reconstruction(features), decoded[:, 1] - decoded[:, 0]
    )


def test_shared_mask_blocks_values_and_gradients_from_delta_loss() -> None:
    model = _model()
    features = torch.rand(3, model.n_features)
    before_delta = model.delta_reconstruction(features).detach().clone()
    before_normal = model.decode(features, add_bias=False).detach().clone()
    with torch.no_grad():
        model.decoder_weight[1, model.shared_slice].add_(7.0)
    torch.testing.assert_close(model.delta_reconstruction(features), before_delta)
    assert not torch.allclose(model.decode(features, add_bias=False), before_normal)

    model.zero_grad(set_to_none=True)
    model.delta_reconstruction(features).square().sum().backward()
    assert model.decoder_weight.grad is not None
    torch.testing.assert_close(
        model.decoder_weight.grad[:, model.shared_slice],
        torch.zeros_like(model.decoder_weight.grad[:, model.shared_slice]),
    )


def test_partition_is_exact_and_has_no_cross_partition_leakage() -> None:
    model = _model()
    assert model.n_shared == 2
    assert model.n_delta == 8
    assert int(model.shared_feature_mask.sum()) == 2
    assert int(model.delta_feature_mask.sum()) == 8
    assert not bool((model.shared_feature_mask & model.delta_feature_mask).any())
    assert bool((model.shared_feature_mask | model.delta_feature_mask).all())


def test_dual_k_applies_independent_batch_global_budgets() -> None:
    model = _model()
    with torch.no_grad():
        model.encoder_weight.fill_(0.1)
        model.encoder_bias.copy_(torch.arange(1, model.n_features + 1))
    x = torch.ones(5, 2, model.activation_dim)
    features = model.encode(x, use_threshold=False)
    assert isinstance(features, torch.Tensor)
    assert int((features[:, model.shared_slice] > 0).sum()) == 5 * int(model.k_shared)
    assert int((features[:, model.delta_slice] > 0).sum()) == 5 * int(model.k_delta)
    assert int((features > 0).sum()) == 5 * (int(model.k_shared) + int(model.k_delta))


def test_joint_encoder_uses_paper_half_average_then_bias() -> None:
    model = _model()
    with torch.no_grad():
        model.encoder_weight.zero_()
        model.encoder_weight[0, 0, 0] = 2.0
        model.encoder_weight[1, 0, 0] = 4.0
        model.encoder_bias.zero_()
    x = torch.zeros(1, 2, 4)
    x[0, 0, 0] = 3.0
    x[0, 1, 0] = 5.0
    # 0.5 * (2*3 + 4*5) = 13.
    assert model.post_relu(x)[0, 0].detach().item() == 13.0


def test_lambda_zero_removes_delta_objective_exactly() -> None:
    model = _model()
    x = torch.randn(6, 2, model.activation_dim)
    reconstructed, delta_reconstructed, _ = model(x)
    losses = delta_losses(
        model,
        x,
        reconstructed,
        delta_reconstructed,
        lambda_delta=0.0,
        auxk_loss=torch.tensor(2.0),
        auxk_coefficient=1 / 32,
    )
    torch.testing.assert_close(
        losses["total"], losses["reconstruction"] + losses["auxk"] / 32
    )


def test_identical_activation_null_has_zero_target_and_initial_delta_prediction() -> None:
    model = _model()
    x_single = torch.randn(7, model.activation_dim)
    paired = torch.stack((x_single, x_single), dim=1)
    features = model.encode(paired)
    assert isinstance(features, torch.Tensor)
    torch.testing.assert_close(model.delta_target(paired), torch.zeros_like(x_single))
    # Paired initialization gives identical base/target decoders.
    torch.testing.assert_close(
        model.delta_reconstruction(features), torch.zeros_like(x_single), atol=1e-7, rtol=0
    )


def test_paper_relative_decoder_norm_formula_and_orientation() -> None:
    decoder = torch.zeros(2, 3, 2)
    decoder[0, 0, 0] = 3.0  # base-only
    decoder[0, 1, 0] = 2.0
    decoder[1, 1, 0] = 2.0  # equal/shared
    decoder[1, 2, 0] = 5.0  # target-only
    base_score = delta_relative_decoder_norm_base(decoder)
    target_score = delta_relative_decoder_norm_target(decoder)
    torch.testing.assert_close(base_score, torch.tensor([1.0, 0.5, 0.0]))
    torch.testing.assert_close(target_score, torch.tensor([0.0, 0.5, 1.0]))


def test_feature_diagnostics_preserve_full_distributions_and_partitions() -> None:
    model = _model()
    fire_count = torch.arange(model.n_features)
    diagnostics = delta_feature_diagnostics(model, fire_count)
    assert diagnostics["feature_id"] == list(range(model.n_features))
    assert diagnostics["partition"] == ["shared", "shared"] + ["delta"] * 8
    assert diagnostics["validation_activation_count"] == list(range(model.n_features))
    assert len(diagnostics["relative_decoder_norm_base"]) == model.n_features
    assert len(diagnostics["decoder_cosine"]) == model.n_features


def test_checkpoint_round_trip_preserves_partition_state_and_config(tmp_path: Path) -> None:
    cfg = load_delta_crosscoder_config(
        ROOT / "configs/multimodel/delta_crosscoder/qwen3_matched_smoke.yaml"
    )
    model = DeltaCrosscoder(
        4,
        1280,
        shared_fraction=0.2,
        k_shared=16,
        k_delta=8,
        norm_init_scale=0.4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    with torch.no_grad():
        model.threshold_shared.fill_(0.25)
        model.threshold_delta.fill_(0.5)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    payload = delta_checkpoint_payload(
        model,
        optimizer,
        scheduler,
        step=7,
        tokens_seen=123,
        tokens_since_fired=torch.arange(model.n_features),
        cfg=cfg,
        source=_FakeSource(4),
    )
    path = save_delta_checkpoint(payload, tmp_path, keep=2)
    with torch.no_grad():
        model.decoder_weight.zero_()
        model.threshold_shared.fill_(-1)
    loaded = load_delta_checkpoint(path, model, optimizer, scheduler)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    assert loaded["partition"] == {
        "n_shared": 256,
        "n_delta": 1024,
        "k_shared": 16,
        "k_delta": 8,
    }
    assert loaded["step"] == 7
    assert loaded["config"]["delta_crosscoder"]["lambda_delta"] == 0.005


def test_paper_and_smoke_configs_keep_provenance_and_runtime_guardrails() -> None:
    paper = load_delta_crosscoder_config(
        ROOT / "configs/multimodel/delta_crosscoder/paper_default.yaml"
    )
    spec = paper.delta_crosscoder
    assert spec.expansion_factor == 5
    assert spec.shared_fraction == 0.20
    assert spec.base_sparsity == 200
    assert spec.k_delta == 200
    assert spec.k_shared == 400
    assert spec.lambda_delta == 0.005
    assert spec.auxk_coefficient == 1 / 32
    assert spec.total_steps == 50_000
    assert spec.warmup_steps == 1_000
    assert spec.batch_size_tokens == 4096
    assert spec.initial_decoder_norm_scale == 0.4
    assert spec.dead_after_tokens == 10_000_000
    assert spec.data.mode == "matched_cache"

    smoke = load_delta_crosscoder_config(
        ROOT / "configs/multimodel/delta_crosscoder/qwen3_matched_smoke.yaml"
    )
    assert smoke.delta_crosscoder.total_steps == 100
    assert smoke.delta_crosscoder.n_features == 1280
    assert smoke.delta_crosscoder.k_shared == 16
    assert smoke.delta_crosscoder.k_delta == 8
    assert smoke.cache.root == "out/multimodel/cache/qwen3_1_7b_entry_v2"
