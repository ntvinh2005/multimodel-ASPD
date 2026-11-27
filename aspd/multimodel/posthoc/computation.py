"""Local rank-one computation under frozen gates and native inputs."""
import math

from collections.abc import Callable

import torch
from torch import Tensor

from aspd.multimodel.posthoc.ids import ComponentId


def largest_terms(vector: Tensor, limit: int = 8) -> list[dict[str, int | float]]:
    values = vector.detach().flatten()
    count = min(limit, values.numel())
    indices = values.abs().argsort(descending=True, stable=True)[:count].tolist()
    return [{"index": i, "value": float(values[i])} for i in indices]


def component_read_write(x: Tensor, u: Tensor, v: Tensor, support: bool, sigma: float):
    """Ungated reading and contribution in the checkpoint's factor convention."""
    if x.shape != v.shape or x.ndim != 1 or u.ndim != 1 or sigma <= 0:
        raise ValueError("Expected single matrix input and compatible rank-one factors")
    terms = x.float() * v.float()
    read = float(terms.sum())
    coefficient = read if support else 0.0
    magnitude = abs(coefficient) * float(u.float().norm())
    return {
        "reading": read, "coefficient": coefficient, "magnitude": magnitude,
        "normalized": magnitude / sigma, "readingTerms": largest_terms(terms),
        "writingTerms": largest_terms(coefficient * u.float()),
    }


def local_mlp(
    gate_logits: Tensor, up: Tensor, activation: Callable[[Tensor], Tensor],
    contribution: Tensor, branch: str, down_reader: Tensor, down_fired: bool,
    *, checked_native_down_input: Tensor,
):
    """Recompute one GLU at its native precision, holding the other branch fixed."""
    if branch not in {"mlp_gate", "mlp_up"}:
        raise ValueError("Only the native Gate and Up projections feed the local GLU")
    if not gate_logits.ndim == 1 or any(
        tensor.shape != gate_logits.shape for tensor in (up, contribution, down_reader, checked_native_down_input)
    ):
        raise ValueError("Incompatible native GLU channel dimensions")
    base = activation(gate_logits) * up
    if not torch.allclose(base, checked_native_down_input, rtol=2e-3, atol=1e-5):
        raise ValueError("Local baseline disagrees with the native Down matrix input")
    activated = activation(gate_logits - contribution.to(gate_logits)) if branch == "mlp_gate" else activation(gate_logits)
    modified_up = up - contribution.to(up) if branch == "mlp_up" else up
    modified = activated * modified_up
    delta = modified.float() - base.float()
    terms = delta * down_reader.float() if down_fired else torch.zeros_like(delta)
    baseline = float(base.float() @ down_reader.float()) if down_fired else 0.0
    after = float(modified.float() @ down_reader.float()) if down_fired else 0.0
    return {
        "before": baseline, "after": after, "delta": after - baseline,
        "changedChannels": largest_terms(delta), "readingTerms": largest_terms(terms),
    }


def affine_up_removal(source: ComponentId, target: ComponentId, baseline: float, edge: float):
    """Numeric prediction from a stored Taylor edge, affine in real arithmetic.

    Frozen-gate, same-layer Up → Down uses an elementwise product with the native
    activation of the independent Gate branch. No claim of numerical parity
    after low-precision hooks is made without replay.
    """
    if (source.model != target.model or source.matrix.layer != target.matrix.layer
            or source.matrix.matrix != "mlp_up" or target.matrix.matrix != "mlp_down"):
        raise ValueError("An affine result requires a same-model, same-layer Up → Down edge")
    return {"before": baseline, "after": baseline - edge, "delta": -edge,
            "method": "affine-from-archived-taylor"}


def attention_probability_delta(native: Tensor, without: Tensor, query: int, limit: int = 8):
    """Summarize native module replay, preserving head and token identities.

    Δlog probability is the score change centered by the log normalizer.
    Masked probabilities have no identifiable log-ratio and are omitted.
    """
    if (native.ndim != 3 or native.shape != without.shape or not 0 <= query < native.shape[-2]
            or not torch.isfinite(native).all() or not torch.isfinite(without).all()
            or (native < 0).any() or (without < 0).any()):
        raise ValueError("Expected matching native attention probabilities [heads, queries, keys]")
    rows = []
    for head, (base, changed) in enumerate(zip(native[:, query, :].float(), without[:, query, :].float(), strict=True)):
        delta = changed - base
        for term in largest_terms(delta, limit):
            key = int(term["index"])
            if term["value"] == 0:
                continue
            centered = float(changed[key].log() - base[key].log()) if base[key] > 0 and changed[key] > 0 else None
            rows.append({"head": head, "key": key, "before": float(base[key]), "after": float(changed[key]),
                         "delta": term["value"], "centeredScoreDelta": centered})
    return rows


@torch.no_grad()
def replay_native_attention(model, inputs, spec, contribution: Tensor, *, limit: int = 8):
    """Inspect one attention block using its native rotary, masking and probability path."""
    if (spec.matrix.matrix not in {"q", "k", "v"} or inputs.anchor_count != 1
            or inputs.input_ids.shape[0] != 1 or contribution.shape != (spec.d_out,)):
        raise ValueError("Expected one Q/K/V component at one exact native input position")
    from contextlib import ExitStack

    class Captured(Exception):
        pass

    pos = int(inputs.anchor_token[0])
    scope = spec.module_path.rsplit(".", 1)[0]
    attention = model.get_submodule(scope)
    projection = model.get_submodule(spec.module_path)

    def run(delta):
        result = {}

        def replace(_module, _args, output):
            if not isinstance(output, Tensor):
                raise ValueError("Unexpected native attention projection output")
            amended = output.clone()
            view = amended if spec.output_slice is None else amended[..., slice(*spec.output_slice)]
            view[0, pos] -= delta.to(output)
            return amended

        def finish(_module, _args, output):
            if not isinstance(output, tuple) or len(output) < 2 or not isinstance(output[1], Tensor):
                raise ValueError("Native attention module did not expose its probability tensor")
            result.update(output=output[0][0].detach(), probabilities=output[1][0].detach())
            raise Captured

        with ExitStack() as stack:
            if delta is not None:
                stack.callback(projection.register_forward_hook(replace).remove)
            stack.callback(attention.register_forward_hook(finish).remove)
            try:
                model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                      use_cache=False, output_attentions=True)
            except Captured:
                return result
        raise RuntimeError("Native attention hook did not execute")

    before, after = run(None), run(contribution)
    weights, modified = before["probabilities"], after["probabilities"]
    if spec.matrix.matrix == "q":
        affected = attention_probability_delta(weights, modified, pos, limit)
    else:
        if weights.ndim != 3 or weights.shape != modified.shape or not torch.isfinite(weights).all() or not torch.isfinite(modified).all():
            raise ValueError("Unsupported native attention probability layout")
        # Which native queries consumed the inspected K/V position.
        affected = []
        for head, (baseline, changed) in enumerate(zip(weights[:, :, pos], modified[:, :, pos], strict=True)):
            queries = largest_terms((changed - baseline).float() if spec.matrix.matrix == "k" else baseline.float(), limit)
            for row in queries:
                query = int(row["index"])
                b, a = float(baseline[query]), float(changed[query])
                if b or a:
                    affected.append({"head": head, "query": query, "key": pos, "before": b, "after": a, "delta": a - b,
                                     "centeredScoreDelta": math.log(a / b) if a > 0 and b > 0 else None})
    writes = after["output"].float() - before["output"].float()
    return {
        "method": "native-attention-module-replay", "pivot": pos, "attentionRows": affected,
        "projectionOutputChanges": largest_terms(writes.norm(dim=-1), limit),
    }
