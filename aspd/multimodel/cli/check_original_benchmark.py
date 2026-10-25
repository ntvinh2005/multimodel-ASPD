from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce the frozen 100M-run throughput gate")
    parser.add_argument("metrics")
    parser.add_argument("--minimum-positions-per-second", type=float, default=4500.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.metrics).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("benchmark metrics are empty")
    measured = float(rows[-1]["positions_per_second"])
    status = "GO" if measured >= args.minimum_positions_per_second else "NO_GO_RUNTIME"
    payload = {
        "status": status,
        "positions_per_second": measured,
        "minimum_positions_per_second": args.minimum_positions_per_second,
        "last_step": int(rows[-1]["step"]),
        "scientific_run": False,
    }
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite benchmark verdict: {destination}")
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if status != "GO":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
