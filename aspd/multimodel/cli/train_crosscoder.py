from __future__ import annotations

import argparse

from aspd.multimodel.baselines.crosscoder import load_crosscoder_config, train_crosscoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BatchTopK Crosscoder baseline")
    parser.add_argument("config")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = train_crosscoder(
        load_crosscoder_config(args.config), device_name=args.device, config_path=args.config
    )
    print(checkpoint)


if __name__ == "__main__":
    main()

