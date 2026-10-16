"""Project model-specific ASPD decoder directions into each model's vocabulary."""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.cli.extract_examples import parse_feature_ids
from aspd.multimodel.config import MultiModelExperimentConfig, load_experiment_config


def _decoder_rows(
    state: dict[str, Tensor],
    cfg: MultiModelExperimentConfig,
    model_index: int,
    feature_ids: list[int],
) -> Tensor:
    """Read ``d_c^(n)`` rows from D0/D1/D2 checkpoint layouts without building the model."""

    selected = torch.tensor(feature_ids, dtype=torch.long)
    if not cfg.sparsity.tie_shared_decoders:
        # D0 or untied D1/D2 stores D^(n)[C,d_act] directly; select requested c rows.
        return state[f"activation_decoders.full.{model_index}"].index_select(0, selected)

    # Tied D1/D2 stores d_c in D_S for c<C_S and in D_E^(n) otherwise.
    shared = state["activation_decoders.shared"]
    exclusive = state[f"activation_decoders.exclusive.{model_index}"]
    rows = []
    for feature_id in feature_ids:
        # Example C_S=6144: c=440 reads D_S[440], while c=7000 reads D_E[856].
        row = (
            shared[feature_id]
            if feature_id < cfg.sparsity.n_shared
            else exclusive[feature_id - cfg.sparsity.n_shared]
        )
        rows.append(row)
    # Stack requested d_c^(n) rows into [C_requested,d_act].
    return torch.stack(rows)


def _token_record(tokenizer: Any, token_id: int, value: float) -> dict[str, Any]:
    """Preserve decoded text, exact tokenizer piece, ID, and the raw projection value."""

    exact = tokenizer.convert_ids_to_tokens(token_id)
    # Reserved model IDs may have no tokenizer string; keep their numeric identity explicit.
    exact = exact if isinstance(exact, str) else f"<unmapped_token_id:{token_id}>"
    decoded = tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    # An unmapped ID decodes to an empty string, so expose the same explicit marker in the chip.
    token = decoded if decoded or not exact.startswith("<unmapped_token_id:") else exact
    return {
        "token": token,
        "exact_token": exact,
        "token_id": token_id,
        "value": value,
    }


def _write_artifact(
    output_path: Path,
    cfg: MultiModelExperimentConfig,
    feature_ids: list[int],
    projected: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
) -> None:
    """Merge compatible component projections and atomically replace the JSON artifact."""

    artifact: dict[str, Any] = {
        "kind": "direct_projection",
        "label": "Direct Vocabulary Projection",
        "description": (
            "For model n: (q_n d_c^(n)) @ W_U^(n)T in original residual-stream units. "
            "This proxy bypasses all transformer blocks after the grounding site and final "
            "normalization; it is not an intervention-based causal effect."
        ),
        "default_model": cfg.models[-1].name,
        "projection_space": "original_residual",
        "features": {},
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("projection_space") != artifact["projection_space"]:
            raise ValueError(
                f"refusing to merge incompatible projection units in {output_path}; "
                "move the existing artifact first"
            )
        if isinstance(existing.get("features"), dict):
            artifact["features"].update(existing["features"])

    # Each c owns separate model series; never average d_c^(n) or W_U^(n) across models.
    for feature_id in feature_ids:
        artifact["features"][str(feature_id)] = {
            "models": {
                model.name: projected[str(feature_id)][model.name]
                for model in cfg.models
            }
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute model-specific (q_n d_c^(n)) @ W_U^(n)T vocabulary projections; "
            "this loads weights but runs no language-model forward pass"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--feature-chunk-size", type=int, default=128)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-features", action="store_true")
    selection.add_argument("--feature-ids", type=parse_feature_ids)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    feature_ids = (
        list(range(cfg.sparsity.n_features)) if args.all_features else args.feature_ids
    )
    if args.top_k <= 0 or args.feature_chunk_size <= 0:
        parser.error("--top-k and --feature-chunk-size must be positive")
    invalid = [feature_id for feature_id in feature_ids if not 0 <= feature_id < cfg.sparsity.n_features]
    if invalid:
        parser.error(f"component IDs out of range: {invalid}")

    # The cache manifest supplies q_n=sqrt(E_t||r_t^(n)||²), fitted on training tokens only.
    cache = PairedActivationCache(cfg)
    scales = [float(manifest["r_rms_norm"]) for manifest in cache.model_manifests]
    # mmap avoids materializing the checkpoint's large Adam state while reading decoder tensors.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state: dict[str, Tensor] = checkpoint["model"]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer, revision=cfg.data.tokenizer_revision
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    # Disable TF32 so saved values are a stable float32 projection of stored model weights.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    projected: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        str(feature_id): {} for feature_id in feature_ids
    }
    for model_index, model_cfg in enumerate(cfg.models):
        # q_n d_c^(n) converts the decoder from normalized R^(n)/q_n back to model residual units.
        directions = _decoder_rows(state, cfg, model_index, feature_ids).float() * scales[model_index]
        language_model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_id,
            revision=model_cfg.revision,
            trust_remote_code=model_cfg.trust_remote_code,
            torch_dtype=torch.float32,
        ).to(device)
        language_model.eval()
        # W_U^(n)[v,d_act] maps a residual direction to one direct vocabulary coordinate.
        unembedding = language_model.get_output_embeddings().weight.detach()
        if directions.shape[1] != unembedding.shape[1]:
            raise ValueError(
                f"decoder width {directions.shape[1]} != unembedding width {unembedding.shape[1]} "
                f"for model {model_cfg.name}"
            )
        directions = directions.to(device)

        for start in range(0, len(feature_ids), args.feature_chunk_size):
            stop = min(len(feature_ids), start + args.feature_chunk_size)
            # Z[c,v]=(q_n d_c^(n))·W_U,v^(n); example [128,2048]@[2048,151936].
            scores = directions[start:stop] @ unembedding.T
            # Retain the strongest promoting and suppressing vocabulary coordinates per c.
            positive_values, positive_ids = torch.topk(scores, args.top_k, dim=1)
            negative_values, negative_ids = torch.topk(scores, args.top_k, dim=1, largest=False)
            for local_index, feature_id in enumerate(feature_ids[start:stop]):
                positive = [
                    _token_record(tokenizer, int(token_id), float(value))
                    for token_id, value in zip(
                        positive_ids[local_index].cpu().tolist(),
                        positive_values[local_index].cpu().tolist(),
                        strict=True,
                    )
                    if value > 0
                ]
                negative = [
                    _token_record(tokenizer, int(token_id), float(value))
                    for token_id, value in zip(
                        negative_ids[local_index].cpu().tolist(),
                        negative_values[local_index].cpu().tolist(),
                        strict=True,
                    )
                    if value < 0
                ]
                projected[str(feature_id)][model_cfg.name] = {
                    "positive": positive,
                    "negative": negative,
                }

        # Only one full language model is resident at a time; W_U has already been reduced to top-k.
        del unembedding, language_model, directions
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir = Path(args.output_dir or Path(args.checkpoint).parent / "analysis")
    output_path = output_dir / "vocabulary_effects.json"
    _write_artifact(output_path, cfg, feature_ids, projected)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "components": len(feature_ids),
                "models": [model.name for model in cfg.models],
                "top_k_per_direction": args.top_k,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
