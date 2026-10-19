from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.feasibility import inspect_model_pair


def validate_static_config(cfg) -> dict[str, object]:
    """Validate config-level invariants without touching Hugging Face or CUDA."""

    from aspd.multimodel.config import ModelSpec

    cache_path = Path(cfg.cache.root).resolve()
    output_path = Path(cfg.training.output_dir).resolve()
    if cache_path == output_path or cache_path in output_path.parents or output_path in cache_path.parents:
        raise ValueError("cache.root and training.output_dir must be separate paths")
    matrix_summary: dict[str, list[str]] = {}
    for model in cfg.models:
        if not isinstance(model, ModelSpec):
            raise TypeError("config models must be ModelSpec instances")
        if not model.matrices:
            raise ValueError(f"model {model.name!r} must declare at least one matrix")
        if any(not matrix.module.strip() for matrix in model.matrices):
            raise ValueError(f"model {model.name!r} has an empty matrix module path")
        matrix_summary[model.name] = [matrix.module for matrix in model.matrices]
    return {
        "models": [model.name for model in cfg.models],
        "matrix_modules": matrix_summary,
        "cache_root": str(cache_path),
        "training_output_dir": str(output_path),
        "model_loading": "adapter-aware; PEFT targets are merged before snapshots",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and expand a multi-model ASPD config")
    parser.add_argument("config")
    parser.add_argument(
        "--inspect-models",
        action="store_true",
        help="also inspect official model metadata, native tokenizers, hooks, and matrix shapes",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="run config/path/matrix validation without loading model checkpoints",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    # Preserve the original CLI output for callers that only validate/expand a config.
    result: dict[str, object] = cfg.model_dump(mode="json")
    if args.static and args.inspect_models is False:
        result = {"config": result, "static_validation": validate_static_config(cfg)}
    if args.inspect_models:
        result = {
            "config": cfg.model_dump(mode="json"),
            "model_pair_feasibility": inspect_model_pair(
                cfg, local_files_only=args.local_files_only
            ),
        }
        if args.static:
            result["static_validation"] = validate_static_config(cfg)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
