"""Forward-hook mechanics on a frozen source model.

Three operations, all keyed by ``MatrixHookSpec``:

* ``capture_matrix_io`` -- read ``x_{j,alpha}`` and ``y_{j,alpha}`` at the anchors
  (no gradient).  Used by the importance stage.
* ``probed_forward`` -- add a zero *probe* ``delta_j`` to selected matrix outputs at the
  anchors and keep the graph, so ``d e_a / d delta_j`` equals ``d e_a / d y_{j,alpha}``.
  Used by the edge stage.
* ``Ablation`` inside ``probed_forward`` -- subtract ``fraction * e_b * u_b`` from
  ``y_{j_b,alpha}``, i.e. remove (a fraction of) component ``b`` from the native
  model output.  Used for the exact-ablation check of the Taylor edge score.

Hooks touch only the anchor rows ``(document, token)``.  The forward pass stops as soon
as every requested capture has been collected, so later layers are never run.

The source model must be frozen; this module never changes its parameters and removes
every hook on exit, including on error.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from aspd.multimodel.posthoc.backend import MatrixHookSpec, ModelInputs
from aspd.multimodel.posthoc.ids import ComponentId, MatrixId


class _CapturesComplete(Exception):
    """Raised inside a hook to stop the forward pass early."""


def _first_tensor(value: Any, where: str) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], Tensor):
        return value[0]
    raise TypeError(f"{where}: expected a tensor or a tuple starting with a tensor")


def _slice(value: Tensor, spec: MatrixHookSpec) -> Tensor:
    if spec.output_slice is None:
        return value
    start, stop = spec.output_slice
    return value[..., start:stop]


def _add_at_anchors(output: Tensor, spec: MatrixHookSpec, rows: tuple[Tensor, Tensor],
                    delta: Tensor) -> Tensor:
    """Return ``output`` with ``delta [G, d_out_logical]`` added at the anchor rows."""

    if spec.output_slice is not None:
        start, stop = spec.output_slice
        delta = nn.functional.pad(delta, (start, output.shape[-1] - stop))
    documents, tokens = rows
    current = output[documents, tokens]
    return output.index_put((documents, tokens), current + delta.to(output.dtype))


def assert_unique_documents(inputs: ModelInputs) -> None:
    """Edge scores need one anchor per document (see ``edges`` for the reason)."""

    if torch.unique(inputs.anchor_document).numel() != inputs.anchor_count:
        raise ValueError("probed forward requires at most one anchor per document")


def _run(model: nn.Module, inputs: ModelInputs) -> Any:
    try:
        return model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)
    except _CapturesComplete:
        return


@torch.no_grad()
def capture_matrix_io(
    model: nn.Module, inputs: ModelInputs, specs: Iterable[MatrixHookSpec]
) -> dict[MatrixId, tuple[Tensor, Tensor]]:
    """Return ``{j: (x [A,d_in], y [A,d_out])}`` at the anchors of ``inputs``."""

    specs = tuple(specs)
    rows = (inputs.anchor_document, inputs.anchor_token)
    inputs_seen: dict[MatrixId, Tensor] = {}
    outputs_seen: dict[MatrixId, Tensor] = {}
    expected = len(specs)

    with ExitStack() as stack:
        for spec in specs:
            module = model.get_submodule(spec.module_path)

            def pre(_m: nn.Module, args: tuple[Any, ...], *, spec: MatrixHookSpec = spec) -> None:
                inputs_seen[spec.matrix] = _first_tensor(args, spec.module_path)[rows].detach()

            def post(_m: nn.Module, _a: Any, output: Any, *, spec: MatrixHookSpec = spec) -> None:
                value = _slice(_first_tensor(output, spec.module_path), spec)
                outputs_seen[spec.matrix] = value[rows].detach()
                if len(outputs_seen) == expected:
                    raise _CapturesComplete

            stack.callback(module.register_forward_pre_hook(pre).remove)
            stack.callback(module.register_forward_hook(post).remove)
        _run(model, inputs)

    missing = [spec.matrix.key for spec in specs if spec.matrix not in outputs_seen]
    if missing:
        raise RuntimeError(f"matrix hooks never fired: {missing}")
    return {
        spec.matrix: (inputs_seen[spec.matrix], outputs_seen[spec.matrix]) for spec in specs
    }


@dataclass(frozen=True)
class Ablation:
    """Remove ``fraction * e_b u_b`` from the output of ``component.matrix``."""

    component: ComponentId
    fraction: float = 1.0


@dataclass
class ProbedForward:
    """Result of ``probed_forward``.

    ``probes[j]`` is a leaf tensor ``[G, d_out_j]`` (zeros) that was added to
    ``y_{j,alpha}``; ``inputs[j]`` is the graph-attached ``x_{j,alpha}`` ``[G, d_in_j]``.
    """

    probes: dict[MatrixId, Tensor] = field(default_factory=dict)
    inputs: dict[MatrixId, Tensor] = field(default_factory=dict)
    outputs: dict[MatrixId, Tensor] = field(default_factory=dict)
    layers: dict[str, Tensor] = field(default_factory=dict)
    logits: Tensor | None = None


def probed_forward(
    model: nn.Module,
    inputs: ModelInputs,
    specs: Iterable[MatrixHookSpec],
    *,
    probe: Iterable[MatrixId],
    capture: Iterable[MatrixId],
    gates: Mapping[str, Tensor] | None = None,
    factors: Mapping[MatrixId, tuple[Tensor, Tensor]] | None = None,
    ablations: Iterable[Ablation] = (),
    output_matrices: Iterable[MatrixId] = (),
    output_layers: Iterable[str] = (),
    capture_logits: bool = False,
) -> ProbedForward:
    """Run the source model once with probes, input captures and optional ablations.

    ``gates[site]`` (``[G, C]``) and ``factors`` are required only for ablations, which
    need ``e_b = g_b (v_b^T x_b)``.  Call under ``torch.enable_grad()`` when probes are
    used; the graph from every probe to every captured input is retained.
    """

    assert_unique_documents(inputs)
    by_matrix = {spec.matrix: spec for spec in specs}
    probe = set(probe)
    capture = set(capture)
    ablations = tuple(ablations)
    ablated = {ablation.component.matrix for ablation in ablations}
    output_matrices = set(output_matrices)
    output_layers = set(output_layers)
    full_forward = bool(output_matrices or output_layers or capture_logits)
    hooked = probe | capture | ablated | output_matrices
    for matrix in hooked:
        if matrix not in by_matrix:
            raise KeyError(f"no hook spec for {matrix.key}")
    if ablations and (gates is None or factors is None):
        raise ValueError("ablations require gates and factors")
    if not capture and not full_forward:
        # The early stop is driven by the capture set; an empty set would stop at the
        # first hooked module.
        raise ValueError("probed_forward requires at least one captured matrix input")

    rows = (inputs.anchor_document, inputs.anchor_token)
    result = ProbedForward()
    ablation_inputs: dict[MatrixId, Tensor] = {}
    needed = set(capture)

    with ExitStack() as stack:
        for name in sorted(output_layers):
            def layer_hook(_module, _args, output, *, name=name):
                result.layers[name] = _first_tensor(output, name)[rows].detach()
            stack.callback(model.get_submodule(name).register_forward_hook(layer_hook).remove)
        for matrix in sorted(hooked):
            spec = by_matrix[matrix]
            module = model.get_submodule(spec.module_path)

            def pre(_m: nn.Module, args: tuple[Any, ...], *, spec: MatrixHookSpec = spec) -> None:
                x = _first_tensor(args, spec.module_path)[rows]
                if spec.matrix in ablated:
                    ablation_inputs[spec.matrix] = x.detach()
                if spec.matrix in capture:
                    result.inputs[spec.matrix] = x
                    if not full_forward and needed.issubset(result.inputs) and not (
                        spec.matrix in probe or spec.matrix in ablated
                    ):
                        raise _CapturesComplete

            def post(_m: nn.Module, _a: Any, output: Any, *, spec: MatrixHookSpec = spec) -> Any:
                value = _first_tensor(output, spec.module_path)
                for ablation in ablations:
                    if ablation.component.matrix != spec.matrix:
                        continue
                    U, V = factors[spec.matrix]  # type: ignore[index]
                    feature = ablation.component.feature
                    g = gates[spec.matrix.site][:, feature].to(U.dtype)  # type: ignore[index]
                    e_b = g * (ablation_inputs[spec.matrix].to(U.dtype) @ V[feature])
                    removal = -ablation.fraction * e_b[:, None] * U[feature][None]
                    value = _add_at_anchors(value, spec, rows, removal)
                if spec.matrix in probe:
                    delta = torch.zeros(
                        inputs.anchor_count, spec.d_out, dtype=value.dtype, device=value.device,
                        requires_grad=True,
                    )
                    result.probes[spec.matrix] = delta
                    value = _add_at_anchors(value, spec, rows, delta)
                if spec.matrix in output_matrices:
                    result.outputs[spec.matrix] = _slice(value, spec)[rows].detach()
                if not full_forward and needed.issubset(result.inputs):
                    # Every requested input is already captured; later modules are unused.
                    raise _CapturesComplete
                return value

            stack.callback(module.register_forward_pre_hook(pre).remove)
            stack.callback(module.register_forward_hook(post).remove)
        output = _run(model, inputs)
        if capture_logits:
            result.logits = output.logits[rows].detach()

    if output_matrices != result.outputs.keys() or output_layers != result.layers.keys():
        raise RuntimeError("A requested native output capture did not execute")
    missing = sorted(matrix.key for matrix in capture if matrix not in result.inputs)
    if missing:
        raise RuntimeError(f"capture hooks never fired: {missing}")
    return result
