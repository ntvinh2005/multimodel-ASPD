from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from aspd.multimodel.analysis import analyze, decode_activation_examples, save_analysis
from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.training import build_model_from_cache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_analysis_manifest(
    *,
    output: Path,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_state: dict[str, Any],
    cache: PairedActivationCache,
    cfg: Any,
) -> None:
    """Bind saved P1--P5 artifacts to the exact checkpoint and activation cache."""

    artifacts = {}
    for name in ("posthoc.safetensors", "taxonomy.json", "top_activation_examples.json"):
        path = output / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema_version": 1,
        "artifact_type": "multimodel_aspd_posthoc_analysis",
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "name": cfg.name,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "step": int(checkpoint_state["step"]),
        },
        "cache": {
            "root": str(cfg.cache.root),
            "fingerprint": cache.token_manifest.get("cache_fingerprint"),
        },
        "model_names": [model.name for model in cfg.models],
        "n_features": int(cfg.sparsity.n_features),
        "top_k": int(cfg.sparsity.top_k),
        "example_feature_mode": cfg.analysis.example_feature_mode,
        "artifacts": artifacts,
        "scientific_boundary": (
            "Post-hoc geometry/activity measurements; semantic labels and behavioral claims are not inferred."
        ),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute P1--P5 multi-model ASPD statistics")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    cfg = load_experiment_config(config_path)
    cache = PairedActivationCache(cfg)
    device = torch.device(args.device)
    model = build_model_from_cache(cfg, cache).to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if "model" not in state or "step" not in state:
        raise ValueError("checkpoint must contain model weights and an integer step")
    model.load_state_dict(state["model"])
    result = analyze(model, cache, cfg, device)
    decode_activation_examples(result["examples"], cfg)
    output = Path(args.output_dir or checkpoint_path.parent / "analysis")
    save_analysis(result, output)
    _write_analysis_manifest(
        output=output,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_state=state,
        cache=cache,
        cfg=cfg,
    )
    print(output)


if __name__ == "__main__":
    main()
