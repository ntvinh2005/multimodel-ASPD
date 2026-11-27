"""Derive paired, candidate-specific observational changes from verified Negation predictions."""

import argparse
from collections import defaultdict
from pathlib import Path

from aspd.multimodel.cli.evaluate_aspd_v3_negation import jsonl, verify_evaluation
from aspd.multimodel.posthoc.artifacts import sha256_file, write_json


def paired_changes(records):
    grouped = defaultdict(dict)
    for row in records:
        example = row["example"]
        grouped[example["group_id"]][example["condition"]] = row
    buckets = defaultdict(list)
    for conditions in grouped.values():
        for before, after in (
            ("affirmative", "negative"),
            ("counterfactual_affirmative", "counterfactual_negative"),
        ):
            baseline, negative = conditions[before], conditions[after]
            family = negative["example"]["family"]
            (excluded,) = negative["example"]["excluded_candidates"]
            valid = negative["example"]["valid_alternatives"]
            for model, neg in negative["models"].items():
                aff = baseline["models"][model]
                excluded_shift = (
                    neg["candidates"][excluded]["logit"] - aff["candidates"][excluded]["logit"]
                )
                alternative_shift = [
                    neg["candidates"][token]["logit"] - aff["candidates"][token]["logit"]
                    for token in valid
                ]
                buckets[f"{model}/{family}/{after}"].append(
                    {
                        "excluded_logit_change": excluded_shift,
                        "excluded_probability_change": (
                            neg["candidates"][excluded]["probability"]
                            - aff["candidates"][excluded]["probability"]
                        ),
                        "mean_valid_alternative_logit_change": sum(alternative_shift)
                        / len(alternative_shift),
                        "valid_alternative_mass_change": sum(
                            neg["candidates"][token]["probability"]
                            - aff["candidates"][token]["probability"]
                            for token in valid
                        ),
                        "excluded_logit_decreased": excluded_shift < 0,
                        "any_valid_alternative_logit_increased": max(alternative_shift) > 0,
                    }
                )
    return {
        key: {
            "denominator": len(rows),
            **{
                f"mean_{metric}": sum(row[metric] for row in rows) / len(rows) for metric in rows[0]
            },
        }
        for key, rows in sorted(buckets.items())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "discovery"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_evaluation(args.evaluation, args.split)
    write_json(
        args.output,
        {
            "mechanism": "negation",
            "split": args.split,
            "source_manifest_sha256": sha256_file(args.evaluation / "manifest.json"),
            "interpretation": "paired native-output changes, no intervention",
            "by_model_family_condition": paired_changes(
                jsonl(args.evaluation / "native_predictions.jsonl")
            ),
        },
    )


if __name__ == "__main__":
    main()
