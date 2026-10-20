"""Capture the seven matrix-input tensors missing from the frozen V3 R cache."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.sports_risk import prepare_v3_matrix_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--activation-cache", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = prepare_v3_matrix_cache(
        load_experiment_config(args.config),
        response_dir=args.response_dir,
        activation_cache=args.activation_cache,
        prompts_path=args.prompts,
        scores_path=args.scores,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
