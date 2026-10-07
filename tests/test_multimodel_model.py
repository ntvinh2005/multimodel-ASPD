import torch

from aspd.multimodel.config import EncoderSpec, ObjectiveSpec, SparsitySpec
from aspd.multimodel.model import MultiModelASPD


def _batch() -> dict[str, object]:
    torch.manual_seed(1)
    valid = torch.ones(2, 3, dtype=torch.bool)
    return {
        "valid_tokens": valid,
        "R": [torch.randn(2, 3, 4), torch.randn(2, 3, 4)],
        "X": [
            {"m": torch.randn(2, 3, 3)},
            {"m": torch.randn(2, 3, 3)},
        ],
    }


def _model(diffing: str = "D0", tie: bool = False) -> MultiModelASPD:
    sparsity = SparsitySpec(
        n_features=8,
        diffing=diffing,
        top_k=3,
        shared_fraction=0.5,
        top_k_shared=2 if diffing != "D0" else None,
        top_k_exclusive=1 if diffing != "D0" else None,
        tie_shared_decoders=tie,
        tie_shared_mechanisms=tie,
    )
    return MultiModelASPD(
        model_names=["base", "ft"],
        activation_dims=[4, 4],
        target_weights=[{"m": torch.randn(5, 3)}, {"m": torch.randn(5, 3)}],
        encoder_cfg=EncoderSpec(kind="linear"),
        sparsity_cfg=sparsity,
        objective_cfg=ObjectiveSpec(
            top_k_aux=2,
            dead_after_batches=2,
            matryoshka_group_fractions=[0.25, 0.25, 0.5],
        ),
    )


def test_forward_has_activation_and_internal_losses_for_both_models() -> None:
    result = _model()(_batch())
    assert result["loss"].isfinite()
    assert {"act/base", "act/ft", "internal/base", "internal/ft"} <= result.keys()


def test_tied_shared_parameters_are_the_same_objects() -> None:
    model = _model("D1", tie=True)
    base = model.components[0]["m"]
    ft = model.components[1]["m"]
    assert base.shared is ft.shared
    assert model.activation_decoders.shared is not None


def test_d2_shared_prefix_has_zero_gradient_on_exclusive_parameters() -> None:
    model = _model("D2", tie=False)
    batch = _batch()
    code = model.encode(batch["R"], batch["valid_tokens"])
    shared_loss, _, _ = model._objective(batch, code, shared_only=True)
    shared_loss.backward()
    for model_components in model.components:
        exclusive = model_components["m"].exclusive
        assert exclusive.U.grad is None or torch.count_nonzero(exclusive.U.grad) == 0
        assert exclusive.V.grad is None or torch.count_nonzero(exclusive.V.grad) == 0
    assert model.activation_decoders.full is not None
    for decoder in model.activation_decoders.full:
        grad = decoder.grad
        assert grad is not None
        assert torch.count_nonzero(grad[model.sparsity_cfg.n_shared:]) == 0


def test_crosscoder_reduction_allows_empty_matrix_sets() -> None:
    model = MultiModelASPD(
        model_names=["a", "b"],
        activation_dims=[4, 4],
        target_weights=[{}, {}],
        encoder_cfg=EncoderSpec(kind="linear"),
        sparsity_cfg=SparsitySpec(n_features=8, top_k=2),
        objective_cfg=ObjectiveSpec(
            top_k_aux=2,
            matryoshka_group_fractions=[0.25, 0.25, 0.5],
        ),
    )
    batch = {
        "valid_tokens": torch.ones(2, 3, dtype=torch.bool),
        "R": [torch.randn(2, 3, 4), torch.randn(2, 3, 4)],
        "X": [{}, {}],
    }
    result = model(batch)
    assert result["internal/a"].item() == 0
    assert result["internal/b"].item() == 0


def test_core_model_accepts_arbitrary_n_and_heterogeneous_dimensions() -> None:
    model = MultiModelASPD(
        model_names=["a", "b", "c"],
        activation_dims=[3, 4, 5],
        target_weights=[
            {"m": torch.randn(4, 2)},
            {"m": torch.randn(5, 3)},
            {"m": torch.randn(6, 4)},
        ],
        encoder_cfg=EncoderSpec(kind="linear"),
        sparsity_cfg=SparsitySpec(n_features=8, top_k=2),
        objective_cfg=ObjectiveSpec(
            top_k_aux=2,
            matryoshka_group_fractions=[0.25, 0.25, 0.5],
        ),
    )
    batch = {
        "valid_tokens": torch.ones(1, 3, dtype=torch.bool),
        "R": [torch.randn(1, 3, width) for width in (3, 4, 5)],
        "X": [
            {"m": torch.randn(1, 3, width)}
            for width in (2, 3, 4)
        ],
    }
    result = model(batch)
    assert result["loss"].isfinite()
    assert {"act/a", "act/b", "act/c"} <= result.keys()
