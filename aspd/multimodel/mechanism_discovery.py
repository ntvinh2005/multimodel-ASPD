"""Shared, auditable helpers for the Qwen2.5 mechanism-discovery pilots.

The functions here deliberately separate three objects that must not be conflated:

* a Delta-Crosscoder feature contribution at the layer-14 residual stream;
* an ASPD sparse activation code and binary mechanism gate; and
* an ASPD rank-one contribution at a configured matrix output.

The GPU CLIs use these helpers, while the numerical invariants are covered by CPU tests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata
from torch import Tensor, nn

from aspd.multimodel.cache import _tensor_from_value
from aspd.multimodel.config import ModelSpec

EXPECTED_CATEGORIES = {"sports_risk": 8, "sports_control": 8, "neutral": 8}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompt_rows(path: str | Path) -> list[dict[str, str]]:
    """Load and validate the frozen 24-row exploratory discovery set."""

    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    required = {"prompt_id", "category", "lexical_condition", "prompt", "continuation"}
    if len(rows) != 24 or len({row.get("prompt_id") for row in rows}) != 24:
        raise ValueError("mechanism discovery set must contain 24 unique prompt IDs")
    counts = {category: sum(row.get("category") == category for row in rows) for category in EXPECTED_CATEGORIES}
    if counts != EXPECTED_CATEGORIES:
        raise ValueError(f"unexpected discovery category counts: {counts}")
    for row in rows:
        missing = required - row.keys()
        if missing or not all(isinstance(row[key], str) and row[key] for key in required):
            raise ValueError(f"invalid prompt row {row.get('prompt_id')}: missing={sorted(missing)}")
    return rows


def tokenize_prompt_rows(
    rows: Sequence[Mapping[str, str]], tokenizer: Any, *, max_length: int
) -> list[dict[str, Any]]:
    """Create fixed prompt+continuation token sequences shared by Base and Target."""

    sequences: list[dict[str, Any]] = []
    for row in rows:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer(
            rendered, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        continuation_ids = tokenizer(
            row["continuation"], add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        ids = prefix_ids + continuation_ids
        if len(ids) > max_length:
            raise ValueError(
                f"{row['prompt_id']} has {len(ids)} tokens, exceeding max_length={max_length}"
            )
        if not prefix_ids or not continuation_ids:
            raise ValueError(f"{row['prompt_id']} tokenized to an empty region")
        sequences.append(
            {
                **dict(row),
                "input_ids": ids,
                "prompt_length": len(prefix_ids),
                "sequence_length": len(ids),
            }
        )
    return sequences


def pad_sequence_batch(
    rows: Sequence[Mapping[str, Any]], pad_token_id: int
) -> tuple[Tensor, Tensor]:
    width = max(int(row["sequence_length"]) for row in rows)
    input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    valid = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        input_ids[index, : ids.numel()] = ids
        valid[index, : ids.numel()] = True
    return input_ids, valid


@torch.inference_mode()
def capture_model_forward(
    model: nn.Module,
    spec: ModelSpec,
    input_ids_cpu: Tensor,
    valid_cpu: Tensor,
    *,
    capture_matrices: bool = False,
) -> dict[str, Any]:
    """Capture clean R and optional X_j on canonical CPU token IDs.

    Canonical token tensors remain on CPU. Each forward receives a device-local clone,
    preserving exact token identity without causing CPU/CUDA device mismatches.
    """

    device = next(model.parameters()).device
    input_ids = input_ids_cpu.detach().clone().to(device)
    valid = valid_cpu.detach().clone().to(device)
    holder: dict[str, Any] = {"X": {}}
    with ExitStack() as stack:
        grounding_module = model.get_submodule(spec.grounding.module)
        if spec.grounding.capture != "input":
            raise ValueError("mechanism discovery currently requires layer grounding capture=input")

        def capture_grounding(_module: nn.Module, args: tuple[object, ...]) -> None:
            holder["R"] = _tensor_from_value(
                args, spec.grounding.tensor_index or 0, spec.grounding.module
            ).detach()

        stack.callback(grounding_module.register_forward_pre_hook(capture_grounding).remove)
        if capture_matrices:
            for matrix in spec.matrices:
                def capture_x(
                    _module: nn.Module,
                    args: tuple[object, ...],
                    *,
                    name: str = matrix.name,
                    site: str = matrix.module,
                ) -> None:
                    holder["X"][name] = _tensor_from_value(args, 0, site).detach()

                stack.callback(model.get_submodule(matrix.module).register_forward_pre_hook(capture_x).remove)
        result = model(input_ids=input_ids, attention_mask=valid, use_cache=False)
    if "R" not in holder:
        raise RuntimeError("grounding hook did not fire")
    if capture_matrices and set(holder["X"]) != {matrix.name for matrix in spec.matrices}:
        raise RuntimeError("one or more matrix-input hooks did not fire")
    return {"logits": result.logits.detach(), "R": holder["R"], "X": holder["X"]}


def delta_feature_values_and_direction(
    delta_model: nn.Module, paired_activations: Tensor, feature_id: int
) -> tuple[Tensor, Tensor]:
    """Return thresholded z_c and d_target,c-d_base,c in raw residual units."""

    if paired_activations.ndim != 4 or paired_activations.shape[2] != 2:
        raise ValueError("paired activations must have shape [B,T,2,D]")
    if not 0 <= feature_id < int(delta_model.n_features):
        raise ValueError("Delta feature ID is out of range")
    batch, tokens, _, width = paired_activations.shape
    flat = paired_activations.reshape(batch * tokens, 2, width)
    encoded = delta_model.encode(flat, use_threshold=True)
    if not isinstance(encoded, Tensor):
        raise TypeError("Delta encode did not return a Tensor")
    values = encoded[:, feature_id].reshape(batch, tokens)
    direction = (
        delta_model.decoder_weight[1, feature_id]
        - delta_model.decoder_weight[0, feature_id]
    )
    return values, direction


def norm_matched_orthogonal_direction(direction: Tensor) -> Tensor:
    """Deterministic orthogonal control with exactly the reference vector's norm."""

    reference = direction.float()
    norm = reference.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("reference direction must have finite non-zero norm")
    for shift in (1, 2, 3):
        candidate = reference.roll(shift)
        candidate = candidate - (candidate @ reference) / reference.square().sum() * reference
        candidate_norm = candidate.norm()
        if candidate_norm > torch.finfo(candidate.dtype).eps:
            return (candidate * (norm / candidate_norm)).to(direction.dtype)
    raise ValueError("could not construct an orthogonal control direction")


def layer_input_perturbation(feature_values: Tensor, direction: Tensor, scale: float) -> Tensor:
    """Broadcast `scale * z_tc * d_c` onto the layer-input residual stream."""

    if feature_values.ndim != 2 or direction.ndim != 1:
        raise ValueError("feature values must be [B,T] and direction must be [D]")
    return float(scale) * feature_values[..., None] * direction


@torch.inference_mode()
def patched_layer_forward(
    model: nn.Module,
    spec: ModelSpec,
    input_ids_cpu: Tensor,
    valid_cpu: Tensor,
    perturbation: Tensor,
) -> Tensor:
    """Run one isolated layer-input patch; the hook is always removed on exit."""

    device = next(model.parameters()).device
    input_ids = input_ids_cpu.detach().clone().to(device)
    valid = valid_cpu.detach().clone().to(device)
    patch = perturbation.to(device)
    module = model.get_submodule(spec.grounding.module)

    def inject(_module: nn.Module, args: tuple[object, ...]) -> tuple[object, ...]:
        hidden = _tensor_from_value(args, spec.grounding.tensor_index or 0, spec.grounding.module)
        if hidden.shape != patch.shape:
            raise ValueError(f"patch shape {tuple(patch.shape)} != hidden shape {tuple(hidden.shape)}")
        replaced = hidden + patch.to(hidden.dtype)
        mutable = list(args)
        mutable[spec.grounding.tensor_index or 0] = replaced
        return tuple(mutable)

    handle = module.register_forward_pre_hook(inject)
    try:
        return model(input_ids=input_ids, attention_mask=valid, use_cache=False).logits.detach()
    finally:
        handle.remove()


def _column_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    numerator = (x_centered * y_centered).sum(axis=0)
    denominator = np.sqrt((x_centered * x_centered).sum(axis=0) * (y_centered * y_centered).sum(axis=0))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def sparse_correspondence_metrics(delta_values: Tensor, aspd_values: Tensor) -> dict[str, Tensor]:
    """Vectorized sparse-aware association metrics for one anchor versus all C columns."""

    if delta_values.ndim != 1 or aspd_values.ndim != 2 or aspd_values.shape[0] != delta_values.numel():
        raise ValueError("expected delta [P] and ASPD [P,C]")
    delta = delta_values.detach().float().cpu().numpy()[:, None]
    aspd = aspd_values.detach().float().cpu().numpy()
    delta_active = delta > 0
    aspd_active = aspd > 0
    intersection = np.logical_and(delta_active, aspd_active).sum(axis=0)
    union = np.logical_or(delta_active, aspd_active).sum(axis=0)
    delta_count = max(int(delta_active.sum()), 1)
    raw_aspd_count = aspd_active.sum(axis=0)
    aspd_count = np.maximum(raw_aspd_count, 1)
    pearson = _column_correlation(delta, aspd)
    delta_rank = rankdata(delta[:, 0], method="average")[:, None]
    aspd_rank = rankdata(aspd, method="average", axis=0)
    spearman = _column_correlation(delta_rank, aspd_rank)
    total = delta.shape[0]
    delta_raw_count = int(delta_active.sum())
    n11 = intersection.astype(np.float64)
    n10 = delta_raw_count - n11
    n01 = raw_aspd_count - n11
    n00 = total - n11 - n10 - n01
    phi_denominator = np.sqrt(
        (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    )
    binary_phi = np.divide(
        n11 * n00 - n10 * n01,
        phi_denominator,
        out=np.zeros_like(n11),
        where=phi_denominator > 0,
    )
    conditional = intersection / delta_count
    fire_rate = aspd_active.mean(axis=0)
    result = {
        "pearson": pearson,
        "spearman": spearman,
        "intersection": intersection.astype(np.float64),
        "jaccard": np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0),
        "p_aspd_active_given_delta": conditional,
        "p_delta_active_given_aspd": intersection / aspd_count,
        "aspd_fire_rate": fire_rate,
        # Raw P(ASPD active | Delta active) favors dense ASPD components. These
        # measures explicitly correct for each component's marginal fire rate.
        "coactivation_excess": conditional - fire_rate,
        "coactivation_lift": np.divide(
            conditional,
            fire_rate,
            out=np.zeros_like(conditional, dtype=float),
            where=fire_rate > 0,
        ),
        "binary_phi": binary_phi,
    }
    return {name: torch.from_numpy(np.asarray(value)).float() for name, value in result.items()}


def rank_percentile(values: Tensor) -> Tensor:
    """Map a finite score vector to [0,1] average-tie percentiles."""

    array = values.detach().float().cpu().numpy()
    if not np.isfinite(array).all():
        raise ValueError("ranking values must be finite")
    if array.size <= 1:
        return torch.zeros_like(values, dtype=torch.float32)
    ranks = rankdata(array, method="average")
    return torch.from_numpy(((ranks - 1) / (array.size - 1)).astype(np.float32))


def rank_one_contribution(u: Tensor, v: Tensor, x: Tensor, gate: Tensor) -> Tensor:
    """Compute g_tc (v_c^T x_t) u_c for selected component rows."""

    if u.ndim != 2 or v.ndim != 2 or x.ndim != 2 or gate.ndim != 2:
        raise ValueError("expected u[C,O], v[C,I], x[P,I], gate[P,C]")
    if u.shape[0] != v.shape[0] or gate.shape != (x.shape[0], u.shape[0]) or x.shape[1] != v.shape[1]:
        raise ValueError("rank-one contribution dimensions do not align")
    reads = x.to(v.dtype) @ v.T
    return (reads * gate.to(reads.dtype))[..., None] * u[None, ...]


def decomposition_change_terms(
    u_base: Tensor,
    v_base: Tensor,
    u_target: Tensor,
    v_target: Tensor,
    x_base: Tensor,
    x_target: Tensor,
    gate: Tensor,
) -> dict[str, Tensor]:
    """Exact actual = operator-on-base-input change + target-operator input change."""

    base = rank_one_contribution(u_base, v_base, x_base, gate)
    target_on_base = rank_one_contribution(u_target, v_target, x_base, gate)
    target = rank_one_contribution(u_target, v_target, x_target, gate)
    operator = target_on_base - base
    input_change = target - target_on_base
    actual = target - base
    return {
        "base": base,
        "target": target,
        "actual_change": actual,
        "operator_change": operator,
        "input_change": input_change,
        "closure_residual": actual - operator - input_change,
    }
