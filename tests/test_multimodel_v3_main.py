from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from aspd.multimodel.cli import train_aspd_v3_main as main_cli
from aspd.multimodel.v3.alignment import _reference_prefixes, _streaming_prefixes
from aspd.multimodel.v3.config import load_v3_config_bundle
from aspd.multimodel.v3.encoder import (
    HierarchicalV3Encoder,
    LearnedProjectionPacker,
    pack_cross_model_tokens,
)
from aspd.multimodel.v3.gating import V3BatchTopK
from aspd.multimodel.v3.main_config import load_main_v1_config
from aspd.multimodel.v3.main_model import (
    MainV1TrainableModel,
    fully_shard_main_v1,
    trainable_parameter_count,
)
from aspd.multimodel.v3.objective import (
    balanced_main_loss,
    detached_binary_support,
    masked_fvu,
    matryoshka_activation_fvu,
)
from aspd.multimodel.v3.reconstruction import (
    ActivationReconstructionBank,
    RankOneMatrixBank,
)
from aspd.multimodel.v3.registry import build_model_registry

ROOT = Path(__file__).parents[1]


def _approved_registry():
    bundle = load_v3_config_bundle(
        ROOT / "configs/multimodel/aspd_v3/approved.yaml",
        ROOT / "configs/multimodel/aspd_v3/candidates.yaml",
        ROOT / "configs/multimodel/aspd_v3/runtime_template.yaml",
    )
    return build_model_registry(bundle.approved)


def test_distributed_init_pins_cuda_device_before_nccl(monkeypatch) -> None:
    events = []
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(
        main_cli.torch.cuda, "set_device", lambda device: events.append(("device", device))
    )
    monkeypatch.setattr(
        main_cli.dist,
        "init_process_group",
        lambda backend, **kwargs: events.append(("init", backend, kwargs)),
    )
    monkeypatch.setattr(main_cli.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(main_cli.dist, "get_world_size", lambda: 3)

    rank, world_size, device = main_cli._init_distributed()

    assert (rank, world_size, device) == (2, 3, torch.device("cuda", 2))
    assert events == [
        ("device", torch.device("cuda", 2)),
        ("init", "nccl", {"device_id": torch.device("cuda", 2)}),
    ]


def test_main_v1_is_frozen_and_legacy_config_remains_non_executable() -> None:
    config = load_main_v1_config(ROOT / "configs/multimodel/aspd_v3/main_v1.yaml")
    assert config.scientific_status == "FROZEN_RESEARCHER_APPROVED"
    assert config.architecture.dictionary_size == 8192
    assert config.runtime.world_size == 3
    assert config.architecture.logical_matrices == (
        "q",
        "k",
        "v",
        "o",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    )
    assert config.data.effective_batch_anchors * config.data.minimum_optimizer_steps == 2_048_000_000
    assert "distributed preflight evidence is not materialized" in config.execution_blockers(ROOT)
    legacy = load_v3_config_bundle(
        ROOT / "configs/multimodel/aspd_v3/approved.yaml",
        ROOT / "configs/multimodel/aspd_v3/candidates.yaml",
        ROOT / "configs/multimodel/aspd_v3/runtime_template.yaml",
    )
    assert legacy.report()["status"] == "BLOCKED"


def test_full_main_v1_parameter_graph_matches_frozen_capacity_on_meta() -> None:
    config = load_main_v1_config(ROOT / "configs/multimodel/aspd_v3/main_v1.yaml")
    model = MainV1TrainableModel(config, _approved_registry(), device="meta")
    assert trainable_parameter_count(model) == 12_575_332_352
    assert len(model.matrix_bank.components) == 168
    assert tuple(model.encoders) == ("pre", "mid")


def test_three_rank_fsdp_uses_dim_zero_padding_for_activation_decoders(
    monkeypatch,
) -> None:
    config = load_main_v1_config(ROOT / "configs/multimodel/aspd_v3/main_v1.yaml")
    model = MainV1TrainableModel(config, _approved_registry(), device="meta")
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 3)
    monkeypatch.setattr(
        torch.distributed.fsdp,
        "fully_shard",
        lambda module, **kwargs: calls.append((module, kwargs)),
    )

    fully_shard_main_v1(model, expected_world_size=3)

    decoder_ids = {
        id(decoder)
        for bank in model.activation_banks.values()
        for decoders in bank.decoders.values()
        for decoder in decoders
    }
    decoder_calls = [kwargs for module, kwargs in calls if id(module) in decoder_ids]
    assert len(decoder_calls) == 48
    assert all("shard_placement_fn" not in kwargs for kwargs in decoder_calls)


def test_cross_model_adapter_preserves_order_separators_shapes_and_gradients() -> None:
    packers = {
        key: LearnedProjectionPacker(32, latent_tokens=2, token_width=8)
        for key in ("qwen", "gemma", "phi")
    }
    encoder = HierarchicalV3Encoder(
        packers,
        intra_layers=1,
        intra_heads=2,
        cross_layers=1,
        cross_heads=2,
        cross_width=8,
        n_features=16,
        readout="mean",
    )
    assert encoder.cross_input_adapter.in_features == 13
    assert encoder.cross_input_adapter.out_features == 8
    layer_activations = {
        key: tuple(torch.randn(1, 3, 4, requires_grad=True) for _ in range(8))
        for key in ("qwen", "gemma", "phi")
    }
    output = encoder(layer_activations, torch.ones(1, 3, dtype=torch.bool))
    assert output.shape == (1, 3, 16)
    output.sum().backward()
    assert encoder.cross_input_adapter.weight.grad is not None
    assert all(packer.projection.weight.grad is not None for packer in packers.values())

    tokens = {key: torch.zeros(1, 1, 2, 8) for key in ("qwen", "gemma", "phi")}
    packed = pack_cross_model_tokens(tokens)
    assert packed.identity_indices == (0, 0, 3, 1, 1, 4, 2, 2)
    assert packed.separator_positions == (2, 5)


def test_full_168_target_synthetic_stage_and_gradient_routes() -> None:
    torch.manual_seed(0)
    targets = [
        replace(target, output_slice=None, d_in=4, d_out=4)
        for layout in _approved_registry().values()
        for target in layout.targets()
    ]
    assert len(targets) == 168 and len({target.key for target in targets}) == 168
    matrix_bank = RankOneMatrixBank(targets, n_features=8)
    activation_banks = torch.nn.ModuleDict(
        {
            site: ActivationReconstructionBank(
                {key: [4] * 8 for key in ("qwen", "gemma", "phi")}, 8
            )
            for site in ("pre", "mid")
        }
    )
    score_leaf = torch.randn(1, 2, 8, requires_grad=True)
    scores = torch.relu(score_leaf)
    valid = torch.ones(1, 2, dtype=torch.bool)
    sparse = V3BatchTopK(8, 2, pool_anchors=2)(scores, valid)
    internal_gate = detached_binary_support(sparse.values)

    internal_terms = {"pre": [], "mid": []}
    for target in targets:
        matrix_input = torch.randn(1, 2, 4)
        truth = torch.randn(1, 2, 4)
        prediction = matrix_bank.reconstruct(
            target.key,
            matrix_input,
            internal_gate,
            valid,
            code_semantics="continuous",
        )
        group = "pre" if target.grounding_kind == "residual_pre" else "mid"
        internal_terms[group].append(masked_fvu(prediction, truth, valid))

    activation_terms = {"pre": [], "mid": []}
    auxk_terms = {"pre": [], "mid": []}
    for site in ("pre", "mid"):
        for model_key in ("qwen", "gemma", "phi"):
            for decoder in activation_banks[site].decoders[model_key]:
                target = torch.randn(1, 2, 4)
                loss, _ = matryoshka_activation_fvu(
                    sparse.values, decoder, target, valid, group_fracs=(0.25, 0.25, 0.5)
                )
                activation_terms[site].append(loss)
        auxk_terms[site].append(scores.sum() * 0.0)

    total, parts = balanced_main_loss(internal_terms, activation_terms, auxk_terms)
    assert set(parts) == {"internal", "activation", "auxk"}
    total.backward()
    assert score_leaf.grad is not None and score_leaf.grad.abs().sum() > 0
    assert all(
        component.U.grad is not None and component.V.grad is not None
        for component in matrix_bank.components.values()
    )
    assert all(
        decoder.weight.grad is not None
        for bank in activation_banks.values()
        for decoders in bank.decoders.values()
        for decoder in decoders
    )
    matrix_bank.normalize_write_decoders_()
    for component in matrix_bank.components.values():
        assert torch.allclose(component.U.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_activation_terms_support_streaming_backward_with_balanced_weights() -> None:
    torch.manual_seed(0)
    model = object.__new__(MainV1TrainableModel)
    torch.nn.Module.__init__(model)
    model.activation_banks = torch.nn.ModuleDict(
        {
            site: ActivationReconstructionBank(
                {key: [4] * 8 for key in ("qwen", "gemma", "phi")}, 16
            )
            for site in ("pre", "mid")
        }
    )
    model.gates = torch.nn.ModuleDict(
        {site: V3BatchTopK(16, 2, pool_anchors=2) for site in ("pre", "mid")}
    )
    valid = torch.ones(1, 2, dtype=torch.bool)
    score_leaves = {
        site: torch.rand(1, 2, 16, requires_grad=True) for site in ("pre", "mid")
    }
    sparse = {
        site: model.gates[site](score_leaves[site], valid)
        for site in ("pre", "mid")
    }
    batch = SimpleNamespace(
        activations={
            site: {
                key: tuple(torch.randn(1, 2, 4) for _ in range(8))
                for key in ("qwen", "gemma", "phi")
            }
            for site in ("pre", "mid")
        },
        valid_anchors=valid,
    )
    coefficients = {"pre": 0.0, "mid": 0.0}
    term_count = 48
    observed = 0
    for index, (prefix, activation, auxk, coefficient) in enumerate(
        model.activation_loss_terms(batch, score_leaves, sparse)
    ):
        site = prefix.split("/", 1)[0]
        coefficients[site] += coefficient
        (coefficient * (activation + 0.03125 * auxk)).backward(
            retain_graph=index + 1 < term_count
        )
        observed += 1

    assert observed == term_count
    assert coefficients == pytest.approx({"pre": 0.5, "mid": 0.5})
    assert all(value.grad is not None for value in score_leaves.values())
    assert all(
        decoder.weight.grad is not None
        for bank in model.activation_banks.values()
        for decoders in bank.decoders.values()
        for decoder in decoders
    )


def test_fvu_rejects_zero_variance_and_padding() -> None:
    target = torch.ones(1, 2, 3)
    with pytest.raises(FloatingPointError, match="zero or near-zero variance"):
        masked_fvu(target, target, torch.tensor([[True, False]]))


def test_streaming_prefix_verifier_matches_original_acceptance_semantics() -> None:
    backend = Tokenizer(
        models.WordPiece(
            {"[UNK]": 0, "alpha": 1, "beta": 2, "!": 3},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.decoder = decoders.WordPiece()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    text = "alpha beta!"
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    input_ids = tuple(encoded["input_ids"])
    reference = _reference_prefixes(tokenizer, input_ids)
    streaming = _streaming_prefixes(tokenizer, input_ids)
    assert streaming is not None
    reference_accepts = {
        stop
        for index, (start, stop) in enumerate(encoded["offset_mapping"])
        if stop > start and reference[index] == text[:stop]
    }
    streaming_accepts = {
        stop
        for index, (start, stop) in enumerate(encoded["offset_mapping"])
        if stop > start and streaming[index] == text[:stop]
    }
    assert streaming_accepts == reference_accepts
