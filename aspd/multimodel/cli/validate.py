from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.feasibility import inspect_model_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and expand a multi-model ASPD config")
    parser.add_argument("config")
    parser.add_argument(
        "--inspect-models",
        action="store_true",
        help="also inspect official model metadata, native tokenizers, hooks, and matrix shapes",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    # Preserve the original CLI output for callers that only validate/expand a config.
    result: dict[str, object] = cfg.model_dump(mode="json")
    if args.inspect_models:
        result = {
            "config": cfg.model_dump(mode="json"),
            "model_pair_feasibility": inspect_model_pair(
                cfg, local_files_only=args.local_files_only
            ),
        }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
