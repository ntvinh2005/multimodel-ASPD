"""Freeze or validate token-only Delta R0 source artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.delta_r0_streaming import (
    SOURCE_ROLES,
    freeze_contrastive_text_source,
    load_r0_mix_config,
    materialize_hf_text_source,
    validate_text_artifact,
    write_mix_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument(
        "--role", choices=(*SOURCE_ROLES, "all"), default="all"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--summary-output")
    args = parser.parse_args()

    mix = load_r0_mix_config(args.config)
    selected = [
        source for source in mix.sources if args.role in {"all", source.role}
    ]
    reports: dict[str, object] = {}
    for source in selected:
        if args.validate_only:
            reports[source.role] = validate_text_artifact(source.artifact_dir, source)
        elif source.role == "contrastive":
            reports[source.role] = freeze_contrastive_text_source(
                mix, overwrite=args.overwrite
            )
        else:
            reports[source.role] = materialize_hf_text_source(
                mix, source.role, overwrite=args.overwrite
            )
    summary_path = Path(
        args.summary_output
        or Path(mix.sources[0].artifact_dir).parent
        / f"{Path(args.config).stem}_summary.json"
    )
    write_mix_summary(mix, summary_path)
    print(json.dumps({"reports": reports, "mix_summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
