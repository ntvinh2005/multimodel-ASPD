from __future__ import annotations

import argparse

from aspd.multimodel.baselines.crosscoder import (
    analyze_crosscoder,
    import_published_analysis,
    load_crosscoder_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a trained BatchTopK Crosscoder")
    parser.add_argument("config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--published-feature-csv")
    parser.add_argument("--published-examples-db")
    parser.add_argument("--published-tokenizer")
    args = parser.parse_args()
    cfg = load_crosscoder_config(args.config)
    if args.published_feature_csv:
        output = import_published_analysis(
            cfg,
            feature_csv=args.published_feature_csv,
            examples_db=args.published_examples_db,
            tokenizer=args.published_tokenizer,
            output_dir=args.output_dir,
        )
    else:
        output = analyze_crosscoder(
            cfg,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            device_name=args.device,
        )
    print(output)


if __name__ == "__main__":
    main()
