"""Extract paired activations from the frozen behavioral V2 response artifact."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
from aspd.multimodel.delta_contrastive import materialize_activation_cache_from_frozen_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_delta_crosscoder_config(args.config)
    report = materialize_activation_cache_from_frozen_text(
        cfg,
        response_dir=args.response_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
