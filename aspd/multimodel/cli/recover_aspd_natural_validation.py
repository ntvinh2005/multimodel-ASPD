"""Recover exact Natural Validation causal prefixes using pinned tokenizers on CPU."""

import argparse
from pathlib import Path

from aspd.multimodel.cli.discover_aspd_circuits import atomic_json
from aspd.multimodel.discovery.natural_validation import recover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "pile", "tokenizer-cache", "training-config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    result = recover(
        args.archive, args.pile, args.tokenizer_cache, args.training_config, args.output
    )
    atomic_json(args.output / "COMPLETE.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
