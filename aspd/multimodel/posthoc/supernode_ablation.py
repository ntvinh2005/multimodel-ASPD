"""Targeted simultaneous native intervention with the original binary gates frozen."""

from __future__ import annotations

from math import isfinite

import torch

from aspd.multimodel.posthoc.source_hooks import Ablation, probed_forward


def exact_effects(
    model,
    inputs,
    gates,
    specs,
    factors,
    source,
    targets,
    target_scales,
    *,
    matrix_scales=None,
    layer_scales=None,
    capture_logits=False,
    continuation_ids=(),
):
    """One paired clean/ablated forward for one model and a batch of independent anchors.

    Ablate each source at the selected anchor using its current native matrix input.
    Output functions return clean minus ablated log probabilities in nats.
    """
    with torch.no_grad():
        return _exact_effects(
            model,
            inputs,
            gates,
            specs,
            factors,
            source,
            targets,
            target_scales,
            matrix_scales or {},
            layer_scales or {},
            capture_logits,
            continuation_ids,
        )


def _exact_effects(
    model,
    inputs,
    gates,
    specs,
    factors,
    source,
    targets,
    target_scales,
    matrix_scales,
    layer_scales,
    capture_logits,
    continuation_ids,
):
    source, targets = tuple(source), tuple(targets)
    if not source or len(set(source)) != len(source) or len(set(targets)) != len(targets):
        raise ValueError("Provide distinct sources and targets for one model")
    if len({component.model for component in (*source, *targets)}) != 1:
        raise ValueError("One ablation is restricted to one model")
    if set(source) & set(targets):
        raise ValueError("Readout targets must not themselves be intervened on")
    eligible = tuple(a for a in targets if any(b.matrix.is_upstream_of(a.matrix) for b in source))
    for value in (*target_scales.values(), *matrix_scales.values(), *layer_scales.values()):
        if not isfinite(value) or value <= 0:
            raise ValueError("Use the preserved finite positive normalization scales")
    if any(not ((value == 0) | (value == 1)).all() for value in gates.values()):
        raise ValueError("Exact effects require frozen binary support")
    if not (eligible or matrix_scales or layer_scales or capture_logits or continuation_ids):
        return {"coefficient": {}, "normalized": {}, "supernode": None}
    options = {
        "probe": (),
        "capture": {a.matrix for a in eligible},
        "gates": gates,
        "factors": factors,
        "output_matrices": matrix_scales,
        "output_layers": layer_scales,
        "capture_logits": bool(capture_logits or continuation_ids),
    }
    clean = probed_forward(model, inputs, specs, **options)
    ablated = probed_forward(
        model, inputs, specs, ablations=tuple(map(Ablation, source)), **options
    )
    signed, normalized = {}, {}
    for target in eligible:
        u, v = factors[target.matrix]
        feature = target.feature
        gate = gates[target.matrix.site][:, feature].to(v.dtype)
        x_clean = clean.inputs[target.matrix].to(v.dtype)
        x_ablated = ablated.inputs[target.matrix].to(v.dtype)
        difference = (gate * (x_clean @ v[feature]) - gate * (x_ablated @ v[feature])).float()
        signed[target.key] = difference.cpu()
        normalized[target.key] = (
            difference.abs() * u[feature].float().norm() / target_scales[target]
        ).cpu()
    result = {
        "coefficient": signed,
        "normalized": normalized,
        "supernode": torch.stack(list(normalized.values())).mean(0) if normalized else None,
        "matrix": {
            matrix.key: _vector_effect(clean.outputs[matrix], ablated.outputs[matrix], scale)
            for matrix, scale in matrix_scales.items()
        },
        "layer": {
            layer: _vector_effect(clean.layers[layer], ablated.layers[layer], scale)
            for layer, scale in layer_scales.items()
        },
        "gates": "frozen-binary",
    }
    if clean.logits is not None and ablated.logits is not None:
        p, q = clean.logits.float().log_softmax(-1), ablated.logits.float().log_softmax(-1)
        midpoint = torch.logaddexp(p, q) - torch.log(torch.tensor(2.0))
        result["js"] = (
            (0.5 * ((p.exp() * (p - midpoint)).sum(-1) + (q.exp() * (q - midpoint)).sum(-1)))
            .clamp_min(0)
            .cpu()
        )
        result["functional"] = {
            int(token): (p[..., token] - q[..., token]).cpu() for token in continuation_ids
        }
    return result


def _vector_effect(clean, ablated, scale):
    return ((clean.float() - ablated.float()).norm(dim=-1) / scale).cpu()


@torch.no_grad()
def native_layer_reference(model, inputs, layer_paths):
    """Capture exact aligned rows in explicitly selected decoder-block order."""
    paths = tuple(layer_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide unique, ordered decoder-block paths")
    captured = probed_forward(model, inputs, (), probe=(), capture=(), output_layers=paths)
    return tuple(captured.layers[path].float().cpu().numpy() for path in paths)
