"""Audit frozen inputs and optional generation for the controlled retrain."""

from __future__ import annotations

import argparse
import json

from aspd.multimodel.delta_controlled_retrain import (
    audit_controlled_config,
    audit_generation_artifact,
    audit_prompt_corpus,
    contrastive_repeat_factor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--old-config", required=True)
    parser.add_argument("--new-config", required=True)
    parser.add_argument("--generation-dir")
    args = parser.parse_args()

    report: dict[str, object] = {
        "prompt_corpus": audit_prompt_corpus(args.corpus, args.prompt_manifest),
        "controlled_config": audit_controlled_config(args.old_config, args.new_config),
    }
    if args.generation_dir:
        generation = audit_generation_artifact(args.generation_dir, args.prompt_manifest)
        repeat = contrastive_repeat_factor(
            train_tokens=int(generation["split_tokens"]["train"])  # type: ignore[index]
        )
        if float(repeat["effective_repeat_factor"]) > 5.0:
            raise ValueError("new contrastive effective repeat factor remains above 5x")
        report["generation"] = generation
        report["contrastive_diversity"] = {
            "old_unique_train_tokens": 13_128,
            "old_effective_repeat_factor": 839_680 / 13_128,
            "new": repeat,
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
