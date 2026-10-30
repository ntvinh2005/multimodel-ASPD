from __future__ import annotations

import argparse
from pathlib import Path

from aspd.multimodel.diagnostics.original_optimization import write_cpu_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit optimization and reconstruction of a completed original-ASPD run"
    )
    parser.add_argument("run_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    report = write_cpu_audit(
        Path(args.run_dir), Path(args.output_dir), verify_hashes=not args.skip_hashes
    )
    print(f"status={report['training_summary']['status']}")
    print(f"output={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
