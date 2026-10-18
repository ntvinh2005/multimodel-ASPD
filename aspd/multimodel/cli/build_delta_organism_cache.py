"""Materialize the separate R0 organism-data paired-activation pilot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
from aspd.multimodel.delta_contrastive import materialize_organism_activation_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_delta_crosscoder_config(args.config)
    if cfg.delta_organism_pilot is None:
        raise ValueError("config has no delta_organism_pilot section")
    output = args.output or cfg.delta_organism_pilot.output_dir
    print(f"start_time={datetime.now().astimezone().isoformat()}", flush=True)
    print(f"config={Path(args.config).resolve()}", flush=True)
    print(f"dataset={Path(cfg.delta_organism_pilot.dataset_path).resolve()}", flush=True)
    print(f"dataset_sha256={cfg.delta_organism_pilot.dataset_sha256}", flush=True)
    print(f"pilot_examples={cfg.delta_organism_pilot.max_examples}", flush=True)
    print(f"output={Path(output).resolve()}", flush=True)
    print(f"device={args.device} cuda_available={torch.cuda.is_available()}", flush=True)
    report = materialize_organism_activation_cache(
        cfg,
        output_dir=output,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"end_time={datetime.now().astimezone().isoformat()}", flush=True)


if __name__ == "__main__":
    main()
