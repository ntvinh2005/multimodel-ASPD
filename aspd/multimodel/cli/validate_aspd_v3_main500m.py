"""Validate and summarize the fail-closed ASPD v3.1 500M draft contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspd.multimodel.v3.data import write_json_immutable
from aspd.multimodel.v3.main500m_config import load_main500m_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--require-preflight-ready", action="store_true")
    parser.add_argument("--preflight-run-dir", type=Path)
    parser.add_argument("--preflight-optimizer-steps", type=int)
    return parser.parse_args()


def readiness_payload(config_path: Path) -> dict[str, object]:
    config = load_main500m_config(config_path)
    payload = config.structural_report()
    serialized = json.dumps(config.payload(), sort_keys=True).lower()
    if "qwen" in serialized:
        raise ValueError("ASPD v3.1 main 500M draft must not contain any Qwen identity")
    payload.update(
        {
            "experiment_family": config.experiment_family,
            "scientific_status": config.scientific_status,
            "architecture_invariants": {
                "encoder": "hierarchical_transformer",
                "gate": "per_anchor_feature_zscore_relu_pool_wide_batchtopk",
                "zscore_axis": "per_anchor_feature_axis",
                "zscore_epsilon": config.objective.gate_logit_normalization_epsilon,
                "activation_code": config.objective.activation_code,
                "internal_code": config.objective.internal_code,
                "internal_gradient_route": config.objective.internal_gradient_route,
                "activation_objective": config.objective.activation_objective_mode,
                "activation_decoder_normalization": (
                    config.objective.activation_decoder_normalization
                ),
                "auxk_score_source": config.objective.auxk_score_source,
            },
            "alignment_contract": {
                "canonical_text_policy": config.data.canonical_text_policy,
                "acceptance": config.data.alignment_acceptance,
                "failure_policy": "reject_if_any_tokenizer_lacks_the_exact_prefix_boundary",
            },
            "submission_permitted": not config.readiness_blockers(),
        }
    )
    return payload


def main() -> None:
    args = _parse_args()
    if args.require_ready and args.require_preflight_ready:
        raise ValueError("main and preflight readiness checks are mutually exclusive")
    if args.require_preflight_ready:
        if args.preflight_run_dir is None or args.preflight_optimizer_steps is None:
            raise ValueError(
                "preflight readiness requires --preflight-run-dir and "
                "--preflight-optimizer-steps"
            )
        config = load_main500m_config(args.config)
        config.require_preflight_authorized(
            optimizer_steps=args.preflight_optimizer_steps,
            run_dir=args.preflight_run_dir,
        )
    payload = readiness_payload(args.config)
    if args.output is not None:
        write_json_immutable(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    if args.require_ready and not payload["submission_permitted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
