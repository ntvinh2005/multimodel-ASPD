"""V3 activation and rank-one matrix-output reconstruction interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from aspd.multimodel.components import RankOneComponents
from aspd.multimodel.v3.registry import LogicalMatrixTarget
from aspd.multimodel.v3.sparse_support import SparseSupport


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

    def forward_sparse(
        self,
        support: SparseSupport,
        *,
        boundaries: Sequence[int] | None = None,
        include_bias: bool = True,
    ) -> Tensor | tuple[Tensor, ...]:
        """Decode coordinates already prepared for the shared gate support."""

        if support.n_features != self.in_features:
            raise ValueError("activation decoder support has the wrong dictionary width")
        rows = support.grid_shape[0] * support.grid_shape[1]
        dtype = support.coefficient.dtype
        output = torch.zeros(rows, self.out_features, device=self.weight.device, dtype=dtype)
        if include_bias and self.bias is not None:
            output = output + self.bias.to(dtype)

        def add_range(value: Tensor, lower: int, upper: int) -> Tensor:
            selected = (support.feature >= lower) & (support.feature < upper)
            positions = support.position[selected]
            features = support.feature[selected]
            scales = support.coefficient[selected]
            writes = scales[:, None] * self.weight.T[features].to(dtype)
            return value.index_add(0, positions, writes)

        if boundaries is None:
            return add_range(output, 0, self.in_features).reshape(
                *support.grid_shape, self.out_features
            )
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != self.in_features:
            raise ValueError("Matryoshka boundaries must cover the complete feature axis")
        reconstructions = [output]
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
            output = add_range(output, int(lower), int(upper))
            reconstructions.append(output)
        return tuple(
            value.reshape(*support.grid_shape, self.out_features)
            for value in reconstructions
        )

    def forward_sparse_csr(
        self,
        support: SparseSupport,
        *,
        boundaries: Sequence[int] | None = None,
        include_bias: bool = True,
    ) -> Tensor | tuple[Tensor, ...]:
        """Decode the prepared support with sparse/dense matrix multiplication."""

        if support.n_features != self.in_features:
            raise ValueError("activation decoder support has the wrong dictionary width")
        rows = support.grid_shape[0] * support.grid_shape[1]
        dtype = support.coefficient.dtype
        bias = torch.zeros(rows, self.out_features, device=self.weight.device, dtype=dtype)
        if include_bias and self.bias is not None:
            bias = bias + self.bias.to(dtype)

        def decode(coefficients: Tensor) -> Tensor:
            code = torch.sparse_csr_tensor(
                support.crow_indices,
                support.feature,
                coefficients,
                size=(rows, self.in_features),
            )
            return bias + torch.sparse.mm(code, self.weight.T.to(dtype))

        if boundaries is None:
            return decode(support.coefficient).reshape(
                *support.grid_shape, self.out_features
            )
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != self.in_features:
            raise ValueError("Matryoshka boundaries must cover the complete feature axis")
        reconstructions = [bias]
        for upper in boundaries[1:]:
            coefficients = support.coefficient * (support.feature < upper).to(dtype)
            reconstructions.append(decode(coefficients))
        return tuple(
            value.reshape(*support.grid_shape, self.out_features)
            for value in reconstructions
        )

    def forward_dense_grouped(
        self,
        code: Tensor,
        *,
        boundaries: Sequence[int] | None = None,
        include_bias: bool = True,
    ) -> Tensor | tuple[Tensor, ...]:
        """Decode sparse-valued dense code via disjoint tensor-core GEMMs.

        For Matryoshka prefixes, each feature group is multiplied exactly once and its
        contribution is accumulated into the successive prefix reconstructions.
        """

        if code.ndim < 2 or code.shape[-1] != self.in_features:
            raise ValueError("dense activation code must have shape [...,C]")
        leading_shape = code.shape[:-1]
        flat = code.reshape(-1, self.in_features)
        output = torch.zeros(
            flat.shape[0], self.out_features, device=flat.device, dtype=flat.dtype
        )
        if include_bias and self.bias is not None:
            output = output + self.bias.to(output.dtype)
        if boundaries is None:
            output = output + flat @ self.weight.T.to(flat.dtype)
            return output.reshape(*leading_shape, self.out_features)
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != self.in_features:
            raise ValueError("Matryoshka boundaries must cover the complete feature axis")
        reconstructions = [output]
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
            output = output + (
                flat[:, int(lower) : int(upper)]
                @ self.weight[:, int(lower) : int(upper)].T.to(flat.dtype)
            )
            reconstructions.append(output)
        return tuple(
            value.reshape(*leading_shape, self.out_features)
            for value in reconstructions
        )


class ActivationReconstructionBank(nn.Module):
    """Model/layer-specific activation decoders; callers choose the explicit code semantics."""

    def __init__(
        self,
        activation_dims: Mapping[str, Sequence[int]],
        n_features: int,
        *,
        normalization_mode: str = "unit_row_norm_with_radial_gradient_projection",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if normalization_mode not in {
            "none",
            "unit_row_norm_with_radial_gradient_projection",
        }:
            raise ValueError(f"unknown activation decoder normalization: {normalization_mode}")
        self.normalization_mode = normalization_mode
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
        if self.normalization_mode != "none":
            self.normalize_decoders_()

    def reconstruct(self, model_key: str, code: Tensor) -> tuple[Tensor, ...]:
        if model_key not in self.decoders:
            raise KeyError(f"unknown v3 activation decoder model: {model_key}")
        return tuple(decoder(code) for decoder in self.decoders[model_key])

    @torch.no_grad()
    def normalize_decoders_(self, *, project_gradients: bool = True) -> None:
        """Apply the explicitly configured training constraint, if any."""

        if self.normalization_mode == "none":
            return

        for decoders in self.decoders.values():
            for decoder in decoders:
                directions = decoder.weight.T
                norms = directions.norm(dim=-1, keepdim=True)
                normalized = directions / (norms + torch.finfo(norms.dtype).tiny)
                if project_gradients and decoder.weight.grad is not None:
                    gradient = decoder.weight.grad.T
                    gradient.sub_((gradient * normalized).sum(-1, keepdim=True) * normalized)
                decoder.weight.copy_(normalized.T)

    @torch.no_grad()
    def project_decoder_gradients_(self) -> None:
        if self.normalization_mode == "none":
            return
        for decoders in self.decoders.values():
            for decoder in decoders:
                if decoder.weight.grad is None:
                    continue
                directions = decoder.weight.T
                normalized = directions / (
                    directions.norm(dim=-1, keepdim=True)
                    + torch.finfo(directions.dtype).tiny
                )
                gradient = decoder.weight.grad.T
                gradient.sub_((gradient * normalized).sum(-1, keepdim=True) * normalized)

    @torch.no_grad()
    def decoder_norm_report(self) -> dict[str, float | int | str]:
        """Measure decoder-column norms without changing parameters or gradients."""

        values = torch.cat(
            [
                decoder.weight.detach().float().T.norm(dim=-1).cpu()
                for decoders in self.decoders.values()
                for decoder in decoders
            ]
        )
        quantiles = torch.quantile(
            values, torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        )
        return {
            "normalization_mode": self.normalization_mode,
            "count": int(values.numel()),
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "min": float(values.min()),
            "q01": float(quantiles[0]),
            "q10": float(quantiles[1]),
            "q25": float(quantiles[2]),
            "median": float(quantiles[3]),
            "q75": float(quantiles[4]),
            "q90": float(quantiles[5]),
            "q99": float(quantiles[6]),
            "max": float(values.max()),
        }


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
        support: SparseSupport | None = None,
        compute_dtype: torch.dtype | None = None,
        sparse_backend: Literal["indexed", "csr", "dense"] = "indexed",
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
        if support is None:
            coefficients = (
                (sparse_code > 0).to(sparse_code.dtype)
                if code_semantics == "binary_support"
                else sparse_code
            )
            coefficients = coefficients * valid_anchors[..., None].to(coefficients.dtype)
            output = component(matrix_input, coefficients, valid_anchors)
        else:
            if support.grid_shape != tuple(matrix_input.shape[:2]):
                raise ValueError("prepared support and matrix input grids differ")
            if sparse_backend == "indexed":
                output = component.forward_sparse(
                    matrix_input,
                    support.position,
                    support.feature,
                    support.coefficient,
                    compute_dtype=compute_dtype,
                    validate_indices=False,
                )
            elif sparse_backend == "csr":
                output = component.forward_sparse_csr(
                    matrix_input,
                    support.crow_indices,
                    support.feature,
                    support.coefficient,
                    compute_dtype=compute_dtype,
                )
            elif sparse_backend == "dense":
                coefficients = (
                    (sparse_code > 0).to(sparse_code.dtype)
                    if code_semantics == "binary_support"
                    else sparse_code
                )
                output = component.forward_dense_masked(
                    matrix_input,
                    coefficients,
                    valid_anchors,
                    compute_dtype=compute_dtype,
                )
            else:
                raise ValueError(f"unknown sparse reconstruction backend: {sparse_backend}")
        bias_name = self._bias_names.get(target_key)
        if bias_name is not None:
            output = output + getattr(self, bias_name)
        return output

    @torch.no_grad()
    def normalize_write_decoders_(self, *, project_gradients: bool = True) -> None:
        """Apply the model-wide ASPD unit-norm write-decoder constraint after backward."""

        for component in self.components.values():
            component.normalize_write_decoder_(project_gradients=project_gradients)

    @torch.no_grad()
    def project_write_decoder_gradients_(self) -> None:
        for component in self.components.values():
            component.project_write_decoder_gradient_()


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
