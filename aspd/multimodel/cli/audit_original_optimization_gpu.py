from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.cache import (
    _Capture,
    _register_capture_hooks,
    load_effective_model,
    snapshot_model_weights,
)
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.diagnostics.original_optimization import (
    MATRYOSHKA_BOUNDARIES,
    gradient_audit,
)
from aspd.multimodel.model import MultiModelASPD
from aspd.multimodel.original_evaluation import _load_checkpoint_pair
from aspd.multimodel.original_recipe import (
    sha256_file,
    validate_dataset_manifest,
    validate_original_recipe_config,
)
from aspd.multimodel.original_streaming import (
    FrozenTokenShardStream,
    _capture_pair,
    resolved_config_sha256,
)


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().double().square().mean().sqrt())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.detach().double().flatten()
    right_flat = right.detach().double().flatten()
    denominator = left_flat.norm() * right_flat.norm()
    return float((left_flat @ right_flat) / denominator.clamp_min(1e-30))


def _input_grounding_metrics(
    residuals: list[torch.Tensor],
    inputs: list[torch.Tensor],
    valid: torch.Tensor,
) -> dict[str, Any]:
    output = {}
    for index, name in enumerate(("base", "target")):
        residual = residuals[index][valid]
        matrix_input = inputs[index][valid]
        output[name] = {
            "tokens": int(residual.shape[0]),
            "residual_rms": _rms(residual),
            "matrix_input_rms": _rms(matrix_input),
            "raw_coordinate_cosine": _cosine(residual, matrix_input),
            "centered_coordinate_cosine": _cosine(
                residual - residual.mean(0, keepdim=True),
                matrix_input - matrix_input.mean(0, keepdim=True),
            ),
            "difference_rms": _rms(residual - matrix_input),
        }
    return output


def _support_by_matryoshka_group(gate: torch.Tensor, valid: torch.Tensor) -> list[dict[str, Any]]:
    total = int(gate[valid].sum())
    rows = []
    for start, stop in zip(
        MATRYOSHKA_BOUNDARIES[:-1], MATRYOSHKA_BOUNDARIES[1:], strict=True
    ):
        count = int(gate[..., start:stop][valid].sum())
        rows.append(
            {
                "start": start,
                "stop_exclusive": stop,
                "selected": count,
                "selection_share": count / max(total, 1),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-batch GPU gradient audit of the completed original-ASPD checkpoint"
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint_rank0")
    parser.add_argument("output_json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-start", type=int, default=0)
    parser.add_argument("--batch-size-sequences", type=int, default=1)
    args = parser.parse_args()

    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    cfg = load_experiment_config(args.config)
    validate_original_recipe_config(cfg)
    dataset = validate_dataset_manifest(cfg, require_ready=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the bounded real-checkpoint diagnostic requires CUDA")

    stream = FrozenTokenShardStream(Path(dataset["manifest"]), "validation")
    token_batch = stream.batch_at(args.validation_start, args.batch_size_sequences)
    sources = []
    captures = []
    weights = []
    provenance = []
    for spec in cfg.models:
        source, source_provenance = load_effective_model(spec, device)
        source.requires_grad_(False)
        snapshot, _ = snapshot_model_weights(source, spec, source_provenance)
        sources.append(source)
        captures.append(_Capture())
        weights.append(snapshot)
        provenance.append(source_provenance)

    width = int(weights[0]["o_proj"].shape[0])
    decomposition = MultiModelASPD(
        [spec.name for spec in cfg.models],
        [width, width],
        weights,
        cfg.encoder,
        cfg.sparsity,
        cfg.objective,
        cfg.original_aspd,
    )
    checkpoint = _load_checkpoint_pair(decomposition, Path(args.checkpoint_rank0), None)
    if checkpoint["config_sha256"] != resolved_config_sha256(cfg):
        raise ValueError("checkpoint scientific configuration fingerprint differs")
    decomposition = decomposition.to(device).float().train()

    with ExitStack() as stack:
        for source, spec, capture in zip(sources, cfg.models, captures, strict=True):
            _register_capture_hooks(source, spec, capture, stack)
        paired = _capture_pair(
            sources,
            captures,
            token_batch["input_ids"],
            token_batch["valid_tokens"],
            device,
            sequence_ids=token_batch["sequence_ids"],
            provenance_ids=token_batch["provenance_ids"],
        )
    valid = paired["valid_tokens"]
    residuals = [value.float() for value in paired["R"]]
    inputs = [value["o_proj"].float() for value in paired["X"]]
    batch: dict[str, object] = {
        "valid_tokens": valid,
        "R": residuals,
        "X": [{"o_proj": value} for value in inputs],
    }
    with torch.no_grad():
        preactivations = decomposition.encoder(residuals, valid)
        code = decomposition.encode(residuals, valid, update_statistics=False)
        support = _support_by_matryoshka_group(code.gate, valid)
        preactivation_rms = _rms(preactivations[valid])

    report = {
        "schema_version": 1,
        "diagnostic": "bounded_one_batch_real_checkpoint_gradient_audit",
        "checkpoint": {
            **checkpoint,
            "path": str(Path(args.checkpoint_rank0).resolve()),
            "sha256": sha256_file(args.checkpoint_rank0),
        },
        "validation_batch": {
            "start_sequence": args.validation_start,
            "sequence_count": args.batch_size_sequences,
            "valid_tokens": int(valid.sum()),
        },
        "source_model_provenance": provenance,
        "input_grounding": _input_grounding_metrics(residuals, inputs, valid),
        "training_batchtopk": {
            "preactivation_rms": preactivation_rms,
            "selected_per_token": float(code.gate[valid].float().sum(-1).mean()),
            "support_by_matryoshka_group": support,
        },
        "objective_gradient_probes": gradient_audit(decomposition, batch),
        "interpretation": (
            "One deterministic held-out batch; gradient norms diagnose graph connectivity and "
            "relative local scale, not population-level optimization quality."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output={output.resolve()}")


if __name__ == "__main__":
    main()
