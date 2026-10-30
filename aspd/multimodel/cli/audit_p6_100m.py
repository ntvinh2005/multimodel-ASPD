"""Validate the frozen ASPD-100M P6 contract without loading either 7B model."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.p6_original_causal import (
    load_p6_original_config,
    verify_p6_handoff,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="P6-100M integration config")
    parser.add_argument(
        "--full-package-rehash",
        action="store_true",
        help="rehash all 26 package artifacts in addition to required P6 artifacts",
    )
    parser.add_argument("--output", help="optional fresh JSON report path")
    args = parser.parse_args()
    pilot = load_p6_original_config(args.config)
    report = verify_p6_handoff(
        pilot,
        verify_all_artifact_hashes=args.full_package_rehash,
    )
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        from pathlib import Path

        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
