"""Select V3 ASPD candidates and measure exact rank-one contributions."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.sports_risk_candidates import analyze_v3_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--matrix-cache", required=True)
    parser.add_argument("--grouped-fidelity", required=True)
    parser.add_argument("--sequence-components", required=True)
    parser.add_argument("--posthoc", required=True)
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shortlist-size", type=int, default=15)
    parser.add_argument("--final-size", type=int, default=5)
    args = parser.parse_args()
    result = analyze_v3_candidates(
        load_experiment_config(args.config),
        args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        matrix_cache_dir=args.matrix_cache,
        grouped_fidelity_path=args.grouped_fidelity,
        sequence_components_path=args.sequence_components,
        posthoc_path=args.posthoc,
        response_dir=args.response_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        shortlist_size=args.shortlist_size,
        final_size=args.final_size,
    )
    print(
        json.dumps(
            {
                "shortlist_component_ids": result["shortlist_component_ids"],
                "final_component_ids": result["final_component_ids"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
