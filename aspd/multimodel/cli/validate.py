from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and expand a multi-model ASPD config")
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    print(json.dumps(cfg.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
