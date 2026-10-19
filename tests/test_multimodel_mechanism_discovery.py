from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from aspd.multimodel.config import ModelSpec
from aspd.multimodel.mechanism_discovery import (
    decomposition_change_terms,
    layer_input_perturbation,
    norm_matched_orthogonal_direction,
    patched_layer_forward,
    sparse_correspondence_metrics,
)


class _ToyHookModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 3)
        self.layer = nn.Identity()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.layer(self.embedding(input_ids)))


def test_delta_perturbation_sign_and_noop() -> None:
    z = torch.tensor([[0.0, 2.0]])
    direction = torch.tensor([1.0, -3.0])
    assert torch.equal(layer_input_perturbation(z, direction, 0.0), torch.zeros(1, 2, 2))
    torch.testing.assert_close(
        layer_input_perturbation(z, direction, -0.5),
        torch.tensor([[[-0.0, 0.0], [-1.0, 3.0]]]),
    )


def test_norm_matched_control_is_orthogonal() -> None:
    direction = torch.tensor([1.0, 2.0, -4.0, 3.0])
    control = norm_matched_orthogonal_direction(direction)
    torch.testing.assert_close(control.norm(), direction.norm())
    torch.testing.assert_close(control.float() @ direction.float(), torch.tensor(0.0), atol=1e-5, rtol=0)


def test_isolated_layer_hook_applies_patch_and_is_removed() -> None:
    torch.manual_seed(0)
    model = _ToyHookModel()
    spec = ModelSpec.model_validate(
        {
            "name": "toy",
            "model_id": "toy",
            "grounding": {"module": "layer", "capture": "input", "tensor_index": 0},
            "matrices": [],
        }
    )
    ids = torch.tensor([[1, 2]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    clean = model(ids, valid, False).logits.detach()
    zero = patched_layer_forward(model, spec, ids, valid, torch.zeros_like(clean))
    torch.testing.assert_close(zero, clean, atol=0, rtol=0)
    patch = torch.ones_like(clean)
    changed = patched_layer_forward(model, spec, ids, valid, patch)
    torch.testing.assert_close(changed, clean + 1)
    # The hook must not survive the isolated patched call.
    torch.testing.assert_close(model(ids, valid, False).logits, clean, atol=0, rtol=0)


def test_sparse_correspondence_handles_constants_and_overlap() -> None:
    delta = torch.tensor([0.0, 1.0, 2.0, 0.0])
    aspd = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    metrics = sparse_correspondence_metrics(delta, aspd)
    torch.testing.assert_close(metrics["pearson"][:2], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(metrics["jaccard"], torch.tensor([1.0, 0.0, 1 / 3]))
    assert all(torch.isfinite(value).all() for value in metrics.values())


def test_rank_one_change_decomposition_closes_exactly() -> None:
    torch.manual_seed(0)
    u_base = torch.randn(2, 3)
    v_base = torch.randn(2, 4)
    u_target = torch.randn(2, 3)
    v_target = torch.randn(2, 4)
    x_base = torch.randn(5, 4)
    x_target = torch.randn(5, 4)
    gate = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0], [1, 1]], dtype=torch.float32)
    terms = decomposition_change_terms(
        u_base, v_base, u_target, v_target, x_base, x_target, gate
    )
    torch.testing.assert_close(
        terms["actual_change"], terms["operator_change"] + terms["input_change"]
    )
    torch.testing.assert_close(terms["closure_residual"], torch.zeros_like(terms["closure_residual"]), atol=1e-6, rtol=0)
