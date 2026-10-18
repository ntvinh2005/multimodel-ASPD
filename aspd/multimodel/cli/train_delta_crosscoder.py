"""Train the isolated Delta-Crosscoder baseline from an existing activation source."""

from __future__ import annotations

import argparse

from aspd.multimodel.baselines.delta_crosscoder import (
    load_delta_crosscoder_config,
    train_delta_crosscoder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_delta_crosscoder_config(args.config)
    checkpoint = train_delta_crosscoder(
        cfg, device_name=args.device, config_path=args.config
    )
    print(checkpoint, flush=True)


if __name__ == "__main__":
    main()
