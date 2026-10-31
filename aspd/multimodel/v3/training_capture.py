"""Bounded aligned-anchor capture and family-rank broadcast for main-v1 training."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from aspd.multimodel.v3.hooks import assert_frozen_backbone, logical_weight_bias
from aspd.multimodel.v3.production_stream import ModelPoolInputs
from aspd.multimodel.v3.registry import ModelFamilyLayout, resolve_submodule


def _tensor(value: Any, path: str) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], Tensor):
        return value[0]
    raise TypeError(f"{path} did not expose a tensor")


def gather_anchor_axis(
    value: Tensor, document_indices: Tensor, token_indices: Tensor
) -> Tensor:
    if value.ndim != 3:
        raise ValueError("source hook tensor must have shape [documents,tokens,width]")
    if document_indices.ndim != 1 or token_indices.shape != document_indices.shape:
        raise ValueError("anchor gather indices must be matching one-dimensional tensors")
    if document_indices.numel() == 0:
        raise ValueError("source capture requires accepted anchors")
    gathered = value[document_indices, token_indices]
    return gathered.detach().unsqueeze(0)


@dataclass(frozen=True)
class FamilyCapture:
    model_key: str
    residual_pre: tuple[Tensor, ...]
    residual_mid: tuple[Tensor, ...]
    matrix_inputs: dict[str, Tensor]
    matrix_outputs: dict[str, Tensor]

    def validate(self, layout: ModelFamilyLayout, anchor_count: int) -> None:
        if self.model_key != layout.key:
            raise ValueError("family capture/model layout mismatch")
        expected_grid = (1, anchor_count)
        for values in (self.residual_pre, self.residual_mid):
            if len(values) != 8 or any(
                value.shape != (*expected_grid, layout.hidden_size) for value in values
            ):
                raise ValueError("family residual capture shape mismatch")
        targets = {target.key: target for target in layout.targets()}
        if set(self.matrix_inputs) != set(targets) or set(self.matrix_outputs) != set(targets):
            raise ValueError("family capture lost a logical target")
        for key, target in targets.items():
            if self.matrix_inputs[key].shape != (*expected_grid, target.d_in):
                raise ValueError(f"{key} family input capture shape mismatch")
            if self.matrix_outputs[key].shape != (*expected_grid, target.d_out):
                raise ValueError(f"{key} family output capture shape mismatch")


def padded_model_inputs(
    pool: ModelPoolInputs,
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    max_length = max(len(value) for value in pool.input_ids)
    input_ids = torch.full(
        (len(pool.input_ids), max_length), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, (ids, mask) in enumerate(
        zip(pool.input_ids, pool.attention_masks, strict=True)
    ):
        length = len(ids)
        input_ids[index, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[index, :length] = torch.tensor(
            mask, dtype=torch.long, device=device
        )
    documents = torch.tensor(
        pool.anchor_document_indices, dtype=torch.long, device=device
    )
    tokens = torch.tensor(pool.anchor_token_indices, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, documents, tokens


def capture_family_pool(
    model: nn.Module,
    layout: ModelFamilyLayout,
    pool: ModelPoolInputs,
    *,
    pad_token_id: int,
    device: torch.device,
) -> FamilyCapture:
    """Run one frozen family once and retain only accepted aligned-anchor rows."""

    assert_frozen_backbone(model)
    model_inputs, documents, tokens = padded_model_inputs(
        pool, pad_token_id=pad_token_id, device=device
    )
    residuals: dict[str, dict[int, Tensor]] = {"pre": {}, "mid": {}}
    matrix_inputs: dict[str, Tensor] = {}
    matrix_outputs: dict[str, Tensor] = {}

    with ExitStack() as stack:
        for site, kind in (("pre", "residual_pre"), ("mid", "residual_mid")):
            for layer in layout.layers:
                path, module = resolve_submodule(model, layout.grounding_paths(kind, layer))

                def residual_hook(
                    _module: nn.Module,
                    args: tuple[Any, ...],
                    *,
                    capture_site: str = site,
                    capture_layer: int = layer,
                    module_path: str = path,
                ) -> None:
                    residuals[capture_site][capture_layer] = gather_anchor_axis(
                        _tensor(args, module_path), documents, tokens
                    )

                stack.callback(module.register_forward_pre_hook(residual_hook).remove)

        for target in layout.targets():
            path, module = resolve_submodule(model, target.physical_module_candidates)

            def matrix_pre(
                _module: nn.Module,
                args: tuple[Any, ...],
                *,
                key: str = target.key,
                module_path: str = path,
            ) -> None:
                matrix_inputs[key] = gather_anchor_axis(
                    _tensor(args, module_path), documents, tokens
                )

            def matrix_post(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                *,
                key: str = target.key,
                output_slice=target.output_slice,
                module_path: str = path,
            ) -> None:
                value = _tensor(output, module_path)
                if output_slice is not None:
                    value = value[..., output_slice.start : output_slice.stop]
                matrix_outputs[key] = gather_anchor_axis(value, documents, tokens)

            stack.callback(module.register_forward_pre_hook(matrix_pre).remove)
            stack.callback(module.register_forward_hook(matrix_post).remove)

        with torch.inference_mode():
            model(**model_inputs, use_cache=False, return_dict=True)

    capture = FamilyCapture(
        model_key=layout.key,
        residual_pre=tuple(residuals["pre"][layer] for layer in layout.layers),
        residual_mid=tuple(residuals["mid"][layer] for layer in layout.layers),
        matrix_inputs=matrix_inputs,
        matrix_outputs=matrix_outputs,
    )
    capture.validate(layout, len(pool.anchor_token_indices))
    return capture


def broadcast_family_capture(
    capture: FamilyCapture | None,
    layout: ModelFamilyLayout,
    *,
    anchor_count: int,
    source_rank: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> FamilyCapture:
    """Replicate one family's bounded observations across the four model-parallel ranks."""

    if not dist.is_initialized():
        if capture is None:
            raise ValueError("non-distributed family broadcast requires a local capture")
        return capture
    rank = dist.get_rank()
    if (rank == source_rank) != (capture is not None):
        raise ValueError("exactly the registered source rank must provide a family capture")
    if capture is not None:
        capture.validate(layout, anchor_count)

    def send(value: Tensor | None, shape: tuple[int, ...]) -> Tensor:
        tensor = value if value is not None else torch.empty(shape, device=device, dtype=dtype)
        dist.broadcast(tensor, source_rank)
        return tensor

    residual_pre = tuple(
        send(
            None if capture is None else capture.residual_pre[index],
            (1, anchor_count, layout.hidden_size),
        )
        for index in range(8)
    )
    residual_mid = tuple(
        send(
            None if capture is None else capture.residual_mid[index],
            (1, anchor_count, layout.hidden_size),
        )
        for index in range(8)
    )
    matrix_inputs = {}
    matrix_outputs = {}
    for target in layout.targets():
        matrix_inputs[target.key] = send(
            None if capture is None else capture.matrix_inputs[target.key],
            (1, anchor_count, target.d_in),
        )
        matrix_outputs[target.key] = send(
            None if capture is None else capture.matrix_outputs[target.key],
            (1, anchor_count, target.d_out),
        )
    result = FamilyCapture(
        layout.key, residual_pre, residual_mid, matrix_inputs, matrix_outputs
    )
    result.validate(layout, anchor_count)
    return result


def broadcast_family_native_biases(
    model: nn.Module | None,
    layout: ModelFamilyLayout,
    *,
    source_rank: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Tensor | None]:
    """Broadcast the exact logical bias slices before trainable-bank construction."""

    if not dist.is_initialized():
        if model is None:
            raise ValueError("non-distributed bias extraction requires the source model")
        return {target.key: logical_weight_bias(model, target)[1] for target in layout.targets()}
    rank = dist.get_rank()
    if (rank == source_rank) != (model is not None):
        raise ValueError("only the registered family source rank may provide native biases")
    values: dict[str, Tensor | None] = {}
    for target in layout.targets():
        local_bias = None if model is None else logical_weight_bias(model, target)[1]
        present = torch.tensor(
            [int(local_bias is not None)], dtype=torch.uint8, device=device
        )
        dist.broadcast(present, source_rank)
        if not bool(present.item()):
            values[target.key] = None
            continue
        value = (
            local_bias.detach().to(device=device, dtype=dtype)
            if local_bias is not None
            else torch.empty(target.d_out, device=device, dtype=dtype)
        )
        dist.broadcast(value, source_rank)
        values[target.key] = value
    return values
