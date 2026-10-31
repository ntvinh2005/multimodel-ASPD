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


class ActivationDecoder(nn.Linear):
    """Sparse feature decoder whose module call is an independent FSDP boundary."""

    def forward(
        self,
        code: Tensor,
        *,
        boundaries: Sequence[int] | None = None,
        include_bias: bool = True,
    ) -> Tensor | tuple[Tensor, ...]:
        if code.ndim < 2 or code.shape[-1] != self.in_features:
            raise ValueError("activation decoder code must have shape [...,C]")
        leading_shape = code.shape[:-1]
        code = code.reshape(-1, self.in_features)
        output = torch.zeros(
            code.shape[0], self.out_features, device=code.device, dtype=code.dtype
        )
        if include_bias and self.bias is not None:
            output = output + self.bias.to(output.dtype)
        active_position, active_feature = (code != 0).nonzero(as_tuple=True)

        def add_range(value: Tensor, lower: int, upper: int) -> Tensor:
            selected = (active_feature >= lower) & (active_feature < upper)
            if not selected.any():
                return value
            positions = active_position[selected]
            features = active_feature[selected]
            scales = code[positions, features]
            writes = scales[:, None] * self.weight.T[features].to(scales.dtype)
            return value.index_add(0, positions, writes)

        if boundaries is None:
            return add_range(output, 0, self.in_features).reshape(
                *leading_shape, self.out_features
            )
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != self.in_features:
            raise ValueError("Matryoshka boundaries must cover the complete feature axis")
        reconstructions = [output]
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
            output = add_range(output, int(lower), int(upper))
            reconstructions.append(output)
        return tuple(
            value.reshape(*leading_shape, self.out_features) for value in reconstructions
        )


class ActivationReconstructionBank(nn.Module):
    """Model/layer-specific activation decoders; callers choose the explicit code semantics."""

    def __init__(
        self,
        activation_dims: Mapping[str, Sequence[int]],
        n_features: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.model_order = tuple(activation_dims)
        self.decoders = nn.ModuleDict()
        for model_key, dimensions in activation_dims.items():
            if len(dimensions) != 8:
                raise ValueError("v3 activation reconstruction requires eight layer dimensions")
            self.decoders[model_key] = nn.ModuleList(
                [
                    ActivationDecoder(
                        n_features, width, bias=True, device=device, dtype=dtype
                    )
                    for width in dimensions
                ]
            )
        self.normalize_decoders_()

    def reconstruct(self, model_key: str, code: Tensor) -> tuple[Tensor, ...]:
        if model_key not in self.decoders:
            raise KeyError(f"unknown v3 activation decoder model: {model_key}")
        return tuple(decoder(code) for decoder in self.decoders[model_key])

    @torch.no_grad()
    def normalize_decoders_(self, *, project_gradients: bool = True) -> None:
        """Keep every feature write direction unit norm, including radial-gradient removal."""

        for decoders in self.decoders.values():
            for decoder in decoders:
                directions = decoder.weight.T
                norms = directions.norm(dim=-1, keepdim=True)
                normalized = directions / (norms + torch.finfo(norms.dtype).tiny)
                if project_gradients and decoder.weight.grad is not None:
                    gradient = decoder.weight.grad.T
                    gradient.sub_((gradient * normalized).sum(-1, keepdim=True) * normalized)
                decoder.weight.copy_(normalized.T)


class RankOneMatrixBank(nn.Module):
    """One heterogeneous rank-one component dictionary per logical matrix target."""

    def __init__(
        self,
        targets: Sequence[LogicalMatrixTarget],
        n_features: int,
        native_biases: Mapping[str, Tensor | None] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if len({target.key for target in targets}) != len(targets):
            raise ValueError("rank-one v3 targets must be unique")
        self.targets = {target.key: target for target in targets}
        self.components = nn.ModuleDict(
            {
                _safe_key(target.key): RankOneComponents(
                    n_features,
                    target.d_in,
                    target.d_out,
                    unit_norm=True,
                    device=device,
                    dtype=dtype,
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
        output = component(matrix_input, coefficients, valid_anchors)
        bias_name = self._bias_names.get(target_key)
        if bias_name is not None:
            output = output + getattr(self, bias_name)
        return output

    @torch.no_grad()
    def normalize_write_decoders_(self, *, project_gradients: bool = True) -> None:
        """Apply the model-wide ASPD unit-norm write-decoder constraint after backward."""

        for component in self.components.values():
            component.normalize_write_decoder_(project_gradients=project_gradients)


class V3ReconstructionInfrastructure(nn.Module):
    """Structural reconstruction modules without a silently selected objective."""

    def __init__(
        self,
        activation_dims: Mapping[str, Sequence[int]],
        targets: Sequence[LogicalMatrixTarget],
        n_features: int,
        native_biases: Mapping[str, Tensor | None] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.activations = ActivationReconstructionBank(
            activation_dims, n_features, device=device, dtype=dtype
        )
        self.matrices = RankOneMatrixBank(
            targets, n_features, native_biases, device=device, dtype=dtype
        )


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
