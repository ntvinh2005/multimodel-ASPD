"""Measure ASPD v3 exposure-ledger throughput without starting training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.data import SQLiteExposureLedger, write_json_immutable


def run_benchmark(
    database: str | Path,
    *,
    records: int,
    batch_size: int,
    run_full_audit: bool,
) -> dict[str, Any]:
    if records <= 0 or batch_size <= 0:
        raise ValueError("records and batch_size must be positive")
    path = Path(database)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark database: {path}")
    started = time.perf_counter()
    with SQLiteExposureLedger(path, dataset_manifest_sha256="0" * 64) as ledger:
        offset = 0
        step = 0
        while offset < records:
            stop = min(records, offset + batch_size)
            ledger.observe_optimizer_step(
                step, [f"benchmark-anchor-{index:012d}" for index in range(offset, stop)]
            )
            offset = stop
            step += 1
        write_seconds = time.perf_counter() - started
        audit_seconds = None
        if run_full_audit:
            audit_started = time.perf_counter()
            ledger.audit_full_consistency()
            audit_seconds = time.perf_counter() - audit_started
        counters = ledger.snapshot()
    return {
        "schema_version": 1,
        "records": records,
        "batch_size": batch_size,
        "optimizer_steps": counters.optimization_steps,
        "write_seconds": write_seconds,
        "records_per_second": records / write_seconds,
        "full_audit_seconds": audit_seconds,
        "database_bytes": path.stat().st_size,
        "status": "MEASURED_LOCAL_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--records", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--full-audit", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "database": str(args.database),
                    "report": str(args.report),
                    "records": args.records,
                    "batch_size": args.batch_size,
                    "full_audit": args.full_audit,
                },
                indent=2,
            )
        )
        return
    result = run_benchmark(
        args.database,
        records=args.records,
        batch_size=args.batch_size,
        run_full_audit=args.full_audit,
    )
    write_json_immutable(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
