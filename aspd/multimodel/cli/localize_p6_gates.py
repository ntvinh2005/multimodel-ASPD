"""Calibrate and localize prefix-causal gates for the controlled ASPD P6 pilot."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.p6_localization import P6_CANDIDATE_IDS, localize_p6_gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--matrix-cache", required=True)
    parser.add_argument("--grouped-fidelity", required=True)
    parser.add_argument("--sequence-components", required=True)
    parser.add_argument("--posthoc", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-tokens", type=int, default=16384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--components", type=int, nargs="+", default=list(P6_CANDIDATE_IDS))
    args = parser.parse_args()
    if len(args.components) != len(set(args.components)):
        parser.error("--components must not contain duplicates")
    cfg = load_experiment_config(args.config)
    invalid = [value for value in args.components if not 0 <= value < cfg.sparsity.n_features]
    if invalid:
        parser.error(f"component IDs outside the dictionary: {invalid}")
    result = localize_p6_gates(
        cfg,
        args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        matrix_cache_dir=args.matrix_cache,
        grouped_fidelity_path=args.grouped_fidelity,
        sequence_components_path=args.sequence_components,
        posthoc_path=args.posthoc,
        output_dir=args.output_dir,
        calibration_tokens=args.calibration_tokens,
        device_name=args.device,
        component_ids=tuple(args.components),
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "component_ids": result["component_ids"],
                "sequences": result["measurement"]["sequences"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

