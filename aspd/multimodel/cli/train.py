from __future__ import annotations

import argparse

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-model ASPD from aligned caches")
    parser.add_argument("config")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = train(load_experiment_config(args.config), device_name=args.device)
    print(checkpoint)


if __name__ == "__main__":
    main()
