"""Export the final original-recipe ASPD scientific and P6-ready package."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_results import export_original_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("run_dir")
    parser.add_argument("evaluation_dir")
    args = parser.parse_args()
    result = export_original_results(
        load_experiment_config(args.config), args.run_dir, args.evaluation_dir
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
