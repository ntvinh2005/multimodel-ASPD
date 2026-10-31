"""Semantically explicit v3 grounding and logical-matrix capture hooks."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from aspd.multimodel.v3.registry import (
    GroundingKind,
    LogicalMatrixTarget,
    ModelFamilyLayout,
    resolve_submodule,
)


def _tensor(value: Any, site: str) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], Tensor):
        return value[0]
    raise TypeError(f"{site} did not expose a tensor in the registered position")


def freeze_backbone(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def assert_frozen_backbone(model: nn.Module) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(f"v3 source model must be fully frozen; trainable parameters: {trainable[:5]}")


@dataclass(frozen=True)
class MatrixObservation:
    target_key: str
    grounding: Tensor
    matrix_input: Tensor
    matrix_output: Tensor
    physical_module_path: str
    grounding_module_path: str


class LogicalMatrixCapture:
    """Capture one logical target without altering the source module's output."""

    def __init__(self, model: nn.Module, target: LogicalMatrixTarget):
        self.model = model
        self.target = target
        self._stack: ExitStack | None = None
        self._values: dict[str, Tensor] = {}
        self.grounding_path = ""
        self.physical_path = ""

    def __enter__(self) -> LogicalMatrixCapture:
        assert_frozen_backbone(self.model)
        self._stack = ExitStack()
        self.grounding_path, grounding = resolve_submodule(
            self.model, self.target.grounding_module_candidates
        )
        self.physical_path, physical = resolve_submodule(
            self.model, self.target.physical_module_candidates
        )

        def grounding_pre(_module: nn.Module, args: tuple[Any, ...]) -> None:
            self._values["grounding"] = _tensor(args, self.grounding_path).detach()

        def matrix_pre(_module: nn.Module, args: tuple[Any, ...]) -> None:
            self._values["input"] = _tensor(args, self.physical_path).detach()

        def matrix_post(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            value = _tensor(output, self.physical_path)
            if self.target.output_slice is not None:
                value = value[..., self.target.output_slice.start : self.target.output_slice.stop]
            self._values["output"] = value.detach()

        self._stack.callback(grounding.register_forward_pre_hook(grounding_pre).remove)
        self._stack.callback(physical.register_forward_pre_hook(matrix_pre).remove)
        self._stack.callback(physical.register_forward_hook(matrix_post).remove)
        return self

    def observation(self) -> MatrixObservation:
        missing = sorted({"grounding", "input", "output"} - set(self._values))
        if missing:
            raise RuntimeError(f"target {self.target.key} hooks did not run: {missing}")
        matrix_input = self._values["input"]
        matrix_output = self._values["output"]
        if matrix_input.shape[-1] != self.target.d_in:
            raise ValueError(
                f"{self.target.key} input width {matrix_input.shape[-1]} != {self.target.d_in}"
            )
        if matrix_output.shape[-1] != self.target.d_out:
            raise ValueError(
                f"{self.target.key} output width {matrix_output.shape[-1]} != {self.target.d_out}"
            )
        return MatrixObservation(
            target_key=self.target.key,
            grounding=self._values["grounding"],
            matrix_input=matrix_input,
            matrix_output=matrix_output,
            physical_module_path=self.physical_path,
            grounding_module_path=self.grounding_path,
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self._stack is not None
        self._stack.close()


class ResidualWindowCapture:
    """Capture one explicit residual kind across the configured eight-layer window."""

    def __init__(
        self,
        model: nn.Module,
        layout: ModelFamilyLayout,
        grounding_kind: GroundingKind,
    ):
        if len(layout.layers) != 8:
            raise ValueError("v3 residual capture requires the registered eight-layer window")
        self.model = model
        self.layout = layout
        self.grounding_kind = grounding_kind
        self._stack: ExitStack | None = None
        self._values: dict[int, Tensor] = {}
        self.paths: dict[int, str] = {}

    def __enter__(self) -> ResidualWindowCapture:
        assert_frozen_backbone(self.model)
        self._stack = ExitStack()
        for layer in self.layout.layers:
            candidates = self.layout.grounding_paths(self.grounding_kind, layer)
            path, module = resolve_submodule(self.model, candidates)
            self.paths[layer] = path

            def capture(
                _module: nn.Module,
                args: tuple[Any, ...],
                *,
                layer_index: int = layer,
                module_path: str = path,
            ) -> None:
                self._values[layer_index] = _tensor(args, module_path).detach()

            self._stack.callback(module.register_forward_pre_hook(capture).remove)
        return self

    def ordered(self) -> tuple[Tensor, ...]:
        missing = [layer for layer in self.layout.layers if layer not in self._values]
        if missing:
            raise RuntimeError(f"v3 residual hooks did not run for layers: {missing}")
        values = tuple(self._values[layer] for layer in self.layout.layers)
        if len({value.shape[:2] for value in values}) != 1:
            raise ValueError("v3 residual window does not share the aligned [B,A] grid")
        return values

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self._stack is not None
        self._stack.close()


def logical_weight_bias(
    model: nn.Module, target: LogicalMatrixTarget
) -> tuple[Tensor, Tensor | None, str]:
    path, module = resolve_submodule(model, target.physical_module_candidates)
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise TypeError(f"{target.key}: physical module {path} has no two-dimensional weight")
    if target.output_slice is not None:
        rows = slice(target.output_slice.start, target.output_slice.stop)
        weight = weight[rows]
        if isinstance(bias, Tensor):
            bias = bias[rows]
    if tuple(weight.shape) != (target.d_out, target.d_in):
        raise ValueError(
            f"{target.key}: sliced weight shape {tuple(weight.shape)} "
            f"!= {(target.d_out, target.d_in)}"
        )
    return weight, bias if isinstance(bias, Tensor) else None, path


def reconstruct_native_matrix_output(
    model: nn.Module, target: LogicalMatrixTarget, matrix_input: Tensor
) -> Tensor:
    """Apply the exact native logical weight slice, including its physical bias."""

    weight, bias, _ = logical_weight_bias(model, target)
    return F.linear(matrix_input.to(weight.dtype), weight, bias)


def validate_observation(
    model: nn.Module,
    target: LogicalMatrixTarget,
    observation: MatrixObservation,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    reconstructed = reconstruct_native_matrix_output(model, target, observation.matrix_input)
    difference = (reconstructed.float() - observation.matrix_output.float()).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    if not torch.allclose(reconstructed, observation.matrix_output, atol=atol, rtol=rtol):
        raise ValueError(
            f"{target.key}: logical slice does not reproduce captured native output; "
            f"max_abs={maximum}"
        )
    return {
        "target": target.key,
        "matrix_input_shape": list(observation.matrix_input.shape),
        "matrix_output_shape": list(observation.matrix_output.shape),
        "maximum_absolute_error": maximum,
        "status": "PASS",
    }
