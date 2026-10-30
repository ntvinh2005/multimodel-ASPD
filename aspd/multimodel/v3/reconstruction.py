"""V3 activation and rank-one matrix-output reconstruction interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from aspd.multimodel.components import RankOneComponents
from aspd.multimodel.v3.registry import LogicalMatrixTarget


def _safe_key(value: str) -> str:
    return value.replace(".", "__dot__")


class ActivationReconstructionBank(nn.Module):
    """Model/layer-specific activation decoders; callers choose the explicit code semantics."""

    def __init__(self, activation_dims: Mapping[str, Sequence[int]], n_features: int):
        super().__init__()
        self.model_order = tuple(activation_dims)
        self.decoders = nn.ModuleDict()
        for model_key, dimensions in activation_dims.items():
            if len(dimensions) != 8:
                raise ValueError("v3 activation reconstruction requires eight layer dimensions")
            self.decoders[model_key] = nn.ModuleList(
                [nn.Linear(n_features, width, bias=True) for width in dimensions]
            )

    def reconstruct(self, model_key: str, code: Tensor) -> tuple[Tensor, ...]:
        if model_key not in self.decoders:
            raise KeyError(f"unknown v3 activation decoder model: {model_key}")
        return tuple(decoder(code) for decoder in self.decoders[model_key])


class RankOneMatrixBank(nn.Module):
    """One heterogeneous rank-one component dictionary per logical matrix target."""

    def __init__(
        self,
        targets: Sequence[LogicalMatrixTarget],
        n_features: int,
        native_biases: Mapping[str, Tensor | None] | None = None,
    ):
        super().__init__()
        if len({target.key for target in targets}) != len(targets):
            raise ValueError("rank-one v3 targets must be unique")
        self.targets = {target.key: target for target in targets}
        self.components = nn.ModuleDict(
            {
                _safe_key(target.key): RankOneComponents(
                    n_features, target.d_in, target.d_out
                )
                for target in targets
            }
        )
        self._bias_names: dict[str, str] = {}
        for index, target in enumerate(targets):
            bias = None if native_biases is None else native_biases.get(target.key)
            if bias is not None:
                if tuple(bias.shape) != (target.d_out,):
                    raise ValueError(f"{target.key} native bias has the wrong shape")
                name = f"native_bias_{index}"
                self.register_buffer(name, bias.detach().clone(), persistent=True)
                self._bias_names[target.key] = name

    def reconstruct(
        self,
        target_key: str,
        matrix_input: Tensor,
        sparse_code: Tensor,
        valid_anchors: Tensor,
        *,
        code_semantics: Literal["binary_support", "continuous"],
    ) -> Tensor:
        """Apply rank-one operators using one explicitly selected provisional semantics."""

        target = self.targets[target_key]
        component = self.components[_safe_key(target_key)]
        if matrix_input.shape[:2] != sparse_code.shape[:2]:
            raise ValueError("matrix input and sparse code disagree on the aligned-anchor grid")
        if valid_anchors.shape != matrix_input.shape[:2]:
            raise ValueError("valid_anchors must match the matrix input grid")
        if matrix_input.shape[-1] != target.d_in:
            raise ValueError(f"{target_key} matrix input width mismatch")
        if sparse_code.shape[-1] != component.n_features:
            raise ValueError(f"{target_key} sparse code dictionary width mismatch")
        coefficients = (
            (sparse_code > 0).to(sparse_code.dtype)
            if code_semantics == "binary_support"
            else sparse_code
        )
        coefficients = coefficients * valid_anchors[..., None].to(coefficients.dtype)
        batch, anchors, features = coefficients.shape
        flat_coefficients = coefficients.reshape(batch * anchors, features)
        flat_input = matrix_input.reshape(batch * anchors, target.d_in)
        active_position, active_feature = (flat_coefficients != 0).nonzero(as_tuple=True)
        output = torch.zeros(
            batch * anchors,
            target.d_out,
            device=matrix_input.device,
            dtype=torch.promote_types(matrix_input.dtype, component.U.dtype),
        )
        if active_feature.numel():
            selected = flat_input[active_position].to(component.V.dtype)
            reads = (selected * component.V[active_feature]).sum(dim=-1)
            scales = reads * flat_coefficients[active_position, active_feature].to(reads.dtype)
            writes = scales[:, None] * component.U[active_feature]
            output.index_add_(0, active_position, writes.to(output.dtype))
        output = output.reshape(batch, anchors, target.d_out)
        bias_name = self._bias_names.get(target_key)
        if bias_name is not None:
            output = output + getattr(self, bias_name)
        return output


class V3ReconstructionInfrastructure(nn.Module):
    """Structural reconstruction modules without a silently selected objective."""

    def __init__(
        self,
        activation_dims: Mapping[str, Sequence[int]],
        targets: Sequence[LogicalMatrixTarget],
        n_features: int,
        native_biases: Mapping[str, Tensor | None] | None = None,
    ):
        super().__init__()
        self.activations = ActivationReconstructionBank(activation_dims, n_features)
        self.matrices = RankOneMatrixBank(targets, n_features, native_biases)


def require_resolved_objective(
    *,
    gate_grouping: str | None,
    code_semantics: str | None,
    gradient_route: str | None,
    activation_loss_weight: float | None,
    internal_loss_weight: float | None,
) -> None:
    unresolved = [
        name
        for name, value in {
            "gate_grouping": gate_grouping,
            "internal_reconstruction_code": code_semantics,
            "internal_loss_gradient_route": gradient_route,
            "activation_loss_weight": activation_loss_weight,
            "internal_loss_weight": internal_loss_weight,
        }.items()
        if value is None
    ]
    if unresolved:
        raise RuntimeError(
            "ASPD v3 objective cannot be constructed until researcher approval: "
            + ", ".join(unresolved)
        )
