from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from pydantic import ValidationError

from aspd.multimodel.cli.train_aspd_v3_2b import (
    _load_training_config,
    _preflight_identity_requirements,
)
from aspd.multimodel.components import RankOneComponents
from aspd.multimodel.v3.activation_evaluation import activation_reconstruction_metrics
from aspd.multimodel.v3.activation_targets import (
    oracle_denormalize,
    prepare_activation_target,
)
from aspd.multimodel.v3.alignment import align_canonical_text
from aspd.multimodel.v3.encoder import pack_cross_model_tokens
from aspd.multimodel.v3.gating import V3BatchTopK
from aspd.multimodel.v3.granite_normalized_config import (
    MODEL_ORDER_GRANITE,
    GraniteNormalizedTargetConfig,
    load_granite_normalized_target_config,
)
from aspd.multimodel.v3.hooks import assert_frozen_backbone, freeze_backbone
from aspd.multimodel.v3.model_owned import SharedSiteEncoder
from aspd.multimodel.v3.objective import (
    auxk_dead_feature_loss,
    detached_binary_support,
    matryoshka_activation_fvu,
    matryoshka_boundaries,
)
from aspd.multimodel.v3.reconstruction import ActivationDecoder, ActivationReconstructionBank
from aspd.multimodel.v3.registry import build_model_registry_2b

ROOT = Path(__file__).parents[1]
BASELINE = ROOT / "configs/multimodel/aspd_v3/granite_normalized_target_baseline_v1.yaml"
TREATMENT = ROOT / "configs/multimodel/aspd_v3/granite_normalized_target_treatment_v1.yaml"
BASELINE_PREFLIGHT = (
    ROOT / "configs/multimodel/aspd_v3/granite_normalized_target_baseline_preflight_v1.yaml"
)
TREATMENT_PREFLIGHT = (
    ROOT / "configs/multimodel/aspd_v3/granite_normalized_target_treatment_preflight_v1.yaml"
)
TREATMENT_STABILIZATION = (
    ROOT
    / "configs/multimodel/aspd_v3/granite_normalized_target_treatment_stabilization_pilot_v1.yaml"
)
BASELINE_STABILIZATION = (
    ROOT
    / "configs/multimodel/aspd_v3/granite_normalized_target_baseline_stabilization_pilot_v1.yaml"
)


def _configs():
    return (
        load_granite_normalized_target_config(BASELINE),
        load_granite_normalized_target_config(TREATMENT),
    )


def test_granite_configs_are_fail_closed_complete_and_isolated() -> None:
    baseline, treatment = _configs()
    for config in (baseline, treatment):
        assert tuple(model.key for model in config.models) == MODEL_ORDER_GRANITE
        assert all("qwen" not in model.model_id.lower() for model in config.models)
        assert config.full_training_authorized is False
        assert config.short_pilot_authorized is False
        assert config.gpu_preflight_authorized is False
        assert config.data.approved_optimizer_steps is None
        assert config.data.approved_accepted_anchor_exposures is None
        assert "granite_normalized_target_v1" in config.runtime.output_root
        assert sum(len(layout.targets()) for layout in build_model_registry_2b(config).values()) == 168
    assert baseline.fingerprint() != treatment.fingerprint()

    left = baseline.payload()
    right = treatment.payload()
    left.pop("arm")
    right.pop("arm")
    left["objective"].pop("activation_target_normalization")
    right["objective"].pop("activation_target_normalization")
    for key in ("output_root", "checkpoint_root"):
        left["runtime"].pop(key)
        right["runtime"].pop(key)
    assert left == right


def test_preflight_overlays_authorize_only_three_steps_and_preserve_science() -> None:
    review_baseline, review_treatment = _configs()
    preflight_baseline = load_granite_normalized_target_config(BASELINE_PREFLIGHT)
    preflight_treatment = load_granite_normalized_target_config(TREATMENT_PREFLIGHT)
    for review, preflight in (
        (review_baseline, preflight_baseline),
        (review_treatment, preflight_treatment),
    ):
        assert preflight.gpu_preflight_authorized is True
        assert preflight.full_training_authorized is False
        assert preflight.short_pilot_authorized is False
        assert preflight.preflight_authorization is not None
        assert preflight.preflight_authorization.optimizer_steps == 3
        assert preflight.preflight_authorization.gpu_count == 3
        left = review.payload()
        right = preflight.payload()
        for payload in (left, right):
            payload.pop("scientific_status")
            payload.pop("gpu_preflight_authorized")
            payload.pop("preflight_authorization")
            payload["runtime"].pop("status")
            payload["runtime"].pop("preflight_evidence")
        assert left == right
    assert _load_training_config(BASELINE_PREFLIGHT).fingerprint() == (
        preflight_baseline.fingerprint()
    )
    assert _load_training_config(TREATMENT_PREFLIGHT).fingerprint() == (
        preflight_treatment.fingerprint()
    )


def test_preflight_overlay_rejects_scientific_or_runtime_override(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "extends: granite_normalized_target_baseline_v1.yaml\n"
        "architecture: {dictionary_size: 4096}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden fields"):
        load_granite_normalized_target_config(bad)


def test_treatment_stabilization_pilot_is_bounded_and_scientifically_matched() -> None:
    _baseline, treatment = _configs()
    pilot = load_granite_normalized_target_config(TREATMENT_STABILIZATION)
    assert pilot.arm == "normalized_target"
    assert pilot.short_pilot_authorized is True
    assert pilot.full_training_authorized is False
    assert pilot.gpu_preflight_authorized is False
    assert pilot.pilot_authorization is not None
    assert pilot.pilot_authorization.optimizer_steps == 128
    assert pilot.pilot_authorization.accepted_anchor_exposures == 1_048_576
    assert pilot.data.approved_optimizer_steps == 128
    assert pilot.data.approved_accepted_anchor_exposures == 1_048_576
    assert pilot.runtime.output_root.startswith("/orange/thai/")

    left = treatment.payload()
    right = pilot.payload()
    for payload in (left, right):
        payload.pop("scientific_status")
        payload.pop("short_pilot_authorized")
        payload.pop("pilot_authorization")
        payload["optimization"].pop("status")
        for key in (
            "training_budget_status",
            "approved_optimizer_steps",
            "approved_accepted_anchor_exposures",
        ):
            payload["data"].pop(key)
        for key in ("status", "output_root", "checkpoint_root", "preflight_evidence"):
            payload["runtime"].pop(key)
        payload.pop("evaluation_status")
    assert left == right


def test_baseline_stabilization_pilot_is_bounded_and_scientifically_matched() -> None:
    baseline, _treatment = _configs()
    pilot = load_granite_normalized_target_config(BASELINE_STABILIZATION)
    assert pilot.arm == "baseline"
    assert pilot.short_pilot_authorized is True
    assert pilot.full_training_authorized is False
    assert pilot.gpu_preflight_authorized is False
    assert pilot.pilot_authorization is not None
    assert pilot.pilot_authorization.scope == "matched_baseline_128_steps"
    assert pilot.pilot_authorization.optimizer_steps == 128
    assert pilot.data.approved_accepted_anchor_exposures == 1_048_576
    assert pilot.runtime.output_root.endswith("/stabilization/baseline/runs")

    left = baseline.payload()
    right = pilot.payload()
    for payload in (left, right):
        payload.pop("scientific_status")
        payload.pop("short_pilot_authorized")
        payload.pop("pilot_authorization")
        payload["optimization"].pop("status")
        for key in (
            "training_budget_status",
            "approved_optimizer_steps",
            "approved_accepted_anchor_exposures",
        ):
            payload["data"].pop(key)
        for key in ("status", "output_root", "checkpoint_root", "preflight_evidence"):
            payload["runtime"].pop(key)
        payload.pop("evaluation_status")
    assert left == right


def test_review_config_rejects_hidden_pilot_budget() -> None:
    payload = yaml.safe_load(TREATMENT.read_text(encoding="utf-8"))
    payload["data"]["training_budget_status"] = "RESEARCHER_AUTHORIZED_STABILIZATION_PILOT"
    payload["data"]["approved_optimizer_steps"] = 128
    payload["data"]["approved_accepted_anchor_exposures"] = 1_048_576
    with pytest.raises(ValidationError, match="cannot carry a training budget"):
        GraniteNormalizedTargetConfig.model_validate(payload)


def test_stabilization_pilot_pins_prerequisite_preflight_not_current_overlay() -> None:
    pilot = load_granite_normalized_target_config(TREATMENT_STABILIZATION)
    expected = _preflight_identity_requirements(
        pilot,
        mode="pilot",
        git_sha="new-pilot-commit",
        config_file_sha256="new-pilot-config-sha",
    )
    assert expected == {
        "config_fingerprint": (
            "37c5c902e69503c362db37ee5031c0d83be2aa50e6e16fb97d5adb7bbbb93acc"
        )
    }
    assert expected["config_fingerprint"] != pilot.fingerprint()


def test_config_rejects_roster_revision_and_authorization_changes() -> None:
    payload = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    mutations = []
    roster = copy.deepcopy(payload)
    roster["architecture"]["model_order"] = ["gemma", "smol", "qwen"]
    mutations.append(roster)
    revision = copy.deepcopy(payload)
    revision["models"][2]["revision"] = "0" * 40
    mutations.append(revision)
    authorization = copy.deepcopy(payload)
    authorization["gpu_preflight_authorized"] = True
    mutations.append(authorization)
    for changed in mutations:
        with pytest.raises(ValidationError):
            GraniteNormalizedTargetConfig.model_validate(changed)


def test_three_model_registry_target_shapes_and_grounding() -> None:
    baseline, _ = _configs()
    registry = build_model_registry_2b(baseline)
    assert tuple(registry) == MODEL_ORDER_GRANITE
    assert [registry[key].layers for key in registry] == [
        tuple(range(9, 17)),
        tuple(range(8, 16)),
        tuple(range(16, 24)),
    ]
    assert [registry[key].hidden_size for key in registry] == [2304, 2048, 2048]
    granite = registry["granite"]
    targets = granite.targets()
    assert len(targets) == 56
    assert {target.logical_name for target in targets} == {
        "q",
        "k",
        "v",
        "o",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    }
    assert all(target.output_slice is None for target in targets)
    assert sum(target.grounding_kind == "residual_pre" for target in targets) == 24
    assert sum(target.grounding_kind == "residual_mid" for target in targets) == 32
    assert granite.grounding_paths("residual_pre", 16) == (
        "model.layers.16.input_layernorm",
    )
    assert granite.grounding_paths("residual_mid", 16) == (
        "model.layers.16.post_attention_layernorm",
    )
    by_name = {target.logical_name: target for target in targets[:7]}
    assert (by_name["q"].d_in, by_name["q"].d_out) == (2048, 2048)
    assert (by_name["k"].d_in, by_name["k"].d_out) == (2048, 512)
    assert (by_name["o"].d_in, by_name["o"].d_out) == (2048, 2048)
    assert (by_name["mlp_up"].d_in, by_name["mlp_up"].d_out) == (2048, 8192)
    assert (by_name["mlp_down"].d_in, by_name["mlp_down"].d_out) == (8192, 2048)


def test_three_model_cross_packing_is_26_by_517() -> None:
    values = {key: torch.randn(2, 3, 8, 512) for key in MODEL_ORDER_GRANITE}
    packed = pack_cross_model_tokens(values, model_order=MODEL_ORDER_GRANITE)
    assert packed.values.shape == (2, 3, 26, 517)
    assert packed.separator_positions == (8, 17)
    assert packed.identity_indices == (
        (0,) * 8 + (3,) + (1,) * 8 + (4,) + (2,) * 8
    )


def test_per_anchor_l2_known_values_mask_and_fp32_policy() -> None:
    original = torch.tensor([[[3.0, 4.0], [0.0, 0.0], [1e-9, 0.0]]], dtype=torch.bfloat16)
    valid = torch.tensor([[True, False, True]])
    prepared = prepare_activation_target(original, valid, mode="per_anchor_l2", epsilon=1e-6)
    assert prepared.target.dtype == torch.float32
    assert prepared.oracle_norm.dtype == torch.float32
    torch.testing.assert_close(prepared.oracle_norm[0, 0], torch.tensor([5.0]))
    torch.testing.assert_close(prepared.target[0, 0], torch.tensor([0.6, 0.8]))
    torch.testing.assert_close(prepared.target[0, 1], torch.zeros(2))
    assert 0 < float(prepared.target[0, 2, 0]) < 0.002
    assert torch.isfinite(prepared.target).all()

    zeros = prepare_activation_target(
        torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool), mode="per_anchor_l2"
    )
    torch.testing.assert_close(zeros.oracle_norm, torch.full((1, 1, 1), 1e-6))
    torch.testing.assert_close(zeros.target, torch.zeros(1, 1, 2))


def test_normalization_is_independent_per_model_layer_anchor() -> None:
    valid = torch.ones(1, 2, dtype=torch.bool)
    observed = []
    for scale in (1.0, 3.0, 11.0):
        for layer_scale in (1.0, 2.0):
            original = scale * layer_scale * torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])
            prepared = prepare_activation_target(original, valid, mode="per_anchor_l2")
            observed.append(prepared.oracle_norm.flatten())
            torch.testing.assert_close(
                torch.linalg.vector_norm(prepared.target, dim=-1), torch.ones(1, 2)
            )
    assert len({tuple(value.tolist()) for value in observed}) == 6


def test_raw_mode_is_exact_historical_target_path() -> None:
    original = torch.randn(2, 3, 4, requires_grad=True)
    valid = torch.tensor([[True, False, True], [True, True, True]])
    prepared = prepare_activation_target(original, valid, mode="none")
    assert prepared.target is original
    decoder = ActivationDecoder(16, 4)
    code = torch.randn(2, 3, 16, requires_grad=True)
    direct, direct_full = matryoshka_activation_fvu(code, decoder, original, valid)
    through_mode, through_full = matryoshka_activation_fvu(
        code, decoder, prepared.target, valid
    )
    torch.testing.assert_close(through_mode, direct, rtol=0, atol=0)
    torch.testing.assert_close(through_full, direct_full, rtol=0, atol=0)


def test_oracle_denormalization_and_spaces_are_not_conflated() -> None:
    original = torch.tensor([[[3.0, 4.0], [0.0, 2.0], [8.0, 6.0]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    prepared = prepare_activation_target(original, valid, mode="per_anchor_l2")
    restored = oracle_denormalize(prepared.target, prepared.oracle_norm)
    torch.testing.assert_close(restored, original)
    imperfect = prepared.target * torch.tensor([[[0.8], [1.2], [0.5]]])
    metrics = activation_reconstruction_metrics(
        original_target=original,
        normalized_target=prepared.target,
        normalized_prediction=imperfect,
        target_norm=prepared.oracle_norm,
        valid_anchors=valid,
        prefix_predictions={0: torch.zeros_like(imperfect), 8192: imperfect},
    )
    assert metrics["original_space_uses_oracle_target_norm"] is True
    assert metrics["normalized_fvu"] != pytest.approx(metrics["original_oracle_fvu"])
    assert tuple(metrics["matryoshka_prefixes"]) == ("0", "8192")


def test_matryoshka_and_auxk_share_normalized_target_and_gradients_are_finite() -> None:
    torch.manual_seed(17)
    valid = torch.tensor([[True, True, False, True]])
    original = torch.randn(1, 4, 5) * torch.tensor([[[1.0], [10.0], [2.0], [0.1]]])
    normalized = prepare_activation_target(original, valid, mode="per_anchor_l2").target
    decoder = ActivationDecoder(16, 5)
    code = torch.rand(1, 4, 16, requires_grad=True)
    loss, full = matryoshka_activation_fvu(code, decoder, normalized, valid)
    dead = torch.tensor([True] * 8 + [False] * 8)
    aux = auxk_dead_feature_loss(
        code,
        decoder,
        normalized,
        full,
        valid,
        dead,
        top_k_aux=4,
        known_dead_feature_count=8,
    )
    (loss + aux / 32).backward()
    assert matryoshka_boundaries(8192, (0.0625, 0.0625, 0.125, 0.25, 0.5)) == (
        0,
        512,
        1024,
        2048,
        4096,
        8192,
    )
    assert code.grad is not None and torch.isfinite(code.grad).all()
    assert decoder.weight.grad is not None and torch.isfinite(decoder.weight.grad).all()


def test_normalized_fvu_changes_gradient_scale_without_hidden_reweighting() -> None:
    torch.manual_seed(23)
    valid = torch.ones(1, 4, dtype=torch.bool)
    raw = torch.randn(1, 4, 3) * torch.tensor([[[1.0], [3.0], [20.0], [0.2]]])
    normalized = prepare_activation_target(raw, valid, mode="per_anchor_l2").target
    decoder_raw = ActivationDecoder(16, 3)
    decoder_norm = ActivationDecoder(16, 3)
    decoder_norm.load_state_dict(decoder_raw.state_dict())
    raw_code = torch.rand(1, 4, 16, requires_grad=True)
    norm_code = raw_code.detach().clone().requires_grad_(True)
    raw_loss, _ = matryoshka_activation_fvu(raw_code, decoder_raw, raw, valid)
    norm_loss, _ = matryoshka_activation_fvu(norm_code, decoder_norm, normalized, valid)
    raw_loss.backward()
    norm_loss.backward()
    raw_gradient = torch.linalg.vector_norm(raw_code.grad)
    normalized_gradient = torch.linalg.vector_norm(norm_code.grad)
    assert torch.isfinite(raw_gradient) and torch.isfinite(normalized_gradient)
    assert not torch.isclose(raw_gradient, normalized_gradient, rtol=1e-3, atol=1e-7)


def test_decoder_directions_remain_individually_unit_norm_after_update() -> None:
    bank = ActivationReconstructionBank({"gemma": [5] * 8, "granite": [7] * 8}, 16)
    decoder = bank.decoders["gemma"][0]
    decoder.weight.grad = torch.randn_like(decoder.weight)
    bank.project_decoder_gradients_()
    directions = decoder.weight.T
    gradient = decoder.weight.grad.T
    torch.testing.assert_close(
        (gradient * directions).sum(-1), torch.zeros(16), rtol=1e-5, atol=1e-5
    )
    with torch.no_grad():
        decoder.weight.add_(decoder.weight.grad, alpha=-0.01)
    bank.normalize_decoders_(project_gradients=False)
    for decoders in bank.decoders.values():
        for item in decoders:
            torch.testing.assert_close(
                torch.linalg.vector_norm(item.weight.T, dim=-1),
                torch.ones(16),
                rtol=1e-6,
                atol=1e-6,
            )


def test_one_continuous_shared_gate_drives_all_three_model_losses() -> None:
    torch.manual_seed(29)
    architecture = SimpleNamespace(
        packed_token_width=8,
        cross_model_width=8,
        cross_model_layers=1,
        cross_model_heads=2,
        dictionary_size=16,
        feature_readout="mean",
    )
    encoder = SharedSiteEncoder(
        SimpleNamespace(architecture=architecture),
        model_order=MODEL_ORDER_GRANITE,
        device="cpu",
        dtype=torch.float32,
    )
    tokens = {
        key: torch.randn(1, 4, 2, 8, requires_grad=True) for key in MODEL_ORDER_GRANITE
    }
    valid = torch.ones(1, 4, dtype=torch.bool)
    scores = encoder(tokens, valid)
    gate = V3BatchTopK(16, 2, pool_anchors=4)(scores, valid, update_statistics=False)
    assert gate.values.shape == (1, 4, 16)
    assert torch.all(gate.values >= 0)
    assert bool((gate.values > 0).sum() <= 8)
    assert bool(((gate.values != 0) & (gate.values != 1)).any())
    support = gate.values.detach() > 0
    losses = []
    for index, _key in enumerate(MODEL_ORDER_GRANITE):
        decoder = ActivationDecoder(16, 3 + index)
        target = torch.randn(1, 4, 3 + index)
        loss, _full = matryoshka_activation_fvu(gate.values, decoder, target, valid)
        losses.append(loss)
        assert torch.equal(gate.values > 0, support)
    torch.stack(losses).mean().backward()
    assert all(tokens[key].grad is not None for key in MODEL_ORDER_GRANITE)
    assert all(torch.isfinite(tokens[key].grad).all() for key in MODEL_ORDER_GRANITE)
    assert all(float(tokens[key].grad.norm()) > 0 for key in MODEL_ORDER_GRANITE)


def test_detached_internal_support_updates_rank_one_not_encoder_code() -> None:
    torch.manual_seed(31)
    code = torch.rand(1, 4, 8, requires_grad=True)
    code.retain_grad()
    support = detached_binary_support(code)
    components = RankOneComponents(8, 3, 2, unit_norm=True)
    matrix_input = torch.randn(1, 4, 3)
    prediction = components(matrix_input, support, torch.ones(1, 4, dtype=torch.bool))
    prediction.square().mean().backward()
    assert code.grad is None
    assert components.U.grad is not None and float(components.U.grad.norm()) > 0
    assert components.V.grad is not None and float(components.V.grad.norm()) > 0


class _SyntheticTokenizer:
    is_fast = True

    def __init__(self, pieces: tuple[str, ...]):
        self.pieces = pieces

    def __call__(self, text: str, **_kwargs):
        offsets = []
        cursor = 0
        for piece in self.pieces:
            start = cursor
            cursor += len(piece)
            offsets.append((start, cursor))
        assert text == "".join(self.pieces)
        return {
            "input_ids": list(range(1, len(self.pieces) + 1)),
            "attention_mask": [1] * len(self.pieces),
            "offset_mapping": offsets,
        }

    def decode(self, token_ids, **_kwargs):
        return "".join(self.pieces[: len(token_ids)])


def test_three_tokenizer_alignment_uses_canonical_boundaries_not_native_indices() -> None:
    tokenizers = {
        "gemma": _SyntheticTokenizer(("ab", "cd", "ef")),
        "smol": _SyntheticTokenizer(("a", "bcd", "ef")),
        "granite": _SyntheticTokenizer(("abcd", "e", "f")),
    }
    result = align_canonical_text(
        "abcdef",
        document_id="doc",
        source_group="synthetic",
        tokenizers=tokenizers,
        model_order=MODEL_ORDER_GRANITE,
    )
    assert [anchor.character_boundary for anchor in result.anchors] == [4, 6]
    first_positions = result.anchors[0].positions
    assert [first_positions[key].token_index for key in MODEL_ORDER_GRANITE] == [1, 1, 0]


def test_tiny_granite_modules_and_residual_multiplier_placement() -> None:
    transformers = pytest.importorskip("transformers")
    GraniteConfig = transformers.GraniteConfig
    GraniteForCausalLM = transformers.GraniteForCausalLM
    config = GraniteConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        residual_multiplier=0.22,
        attention_multiplier=0.015625,
        embedding_multiplier=12.0,
        attention_bias=False,
        mlp_bias=False,
    )
    model = GraniteForCausalLM(config).eval()
    freeze_backbone(model)
    assert_frozen_backbone(model)
    layer = model.model.layers[0]
    assert all(
        hasattr(layer.self_attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    assert all(hasattr(layer.mlp, name) for name in ("gate_proj", "up_proj", "down_proj"))
    assert layer.self_attn.q_proj.bias is None and layer.mlp.up_proj.bias is None

    captured = {}

    def capture_input(name):
        def hook(_module, args):
            captured[name] = args[0]

        return hook

    def capture_output(name, *, first=False):
        def hook(_module, _args, output):
            captured[name] = output[0] if first else output

        return hook

    handles = [
        layer.register_forward_pre_hook(capture_input("pre")),
        layer.post_attention_layernorm.register_forward_pre_hook(capture_input("mid")),
        layer.self_attn.register_forward_hook(capture_output("attention", first=True)),
        layer.mlp.register_forward_hook(capture_output("mlp")),
    ]
    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(
        captured["mid"], captured["pre"] + 0.22 * captured["attention"], rtol=1e-5, atol=1e-6
    )
    assert model.model.layers[0].self_attn.k_proj.weight.shape == (8, 16)
