"""Validate or explicitly execute one canonical 100M P6 autoregressive arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch
import yaml

from aspd.multimodel.cache import load_effective_model
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.original_steering import load_original_decomposition_for_p6
from aspd.multimodel.p6_original_causal import (
    P6Arm,
    load_100m_candidate_shortlist,
    validate_p6_pilot_config,
)
from aspd.multimodel.p6_original_generation import run_original_p6_autoregressive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--component-id", type=int)
    parser.add_argument("--prompt-id")
    parser.add_argument(
        "--arm",
        choices=[
            "clean",
            "learned",
            "gate_off",
            "active_matched",
            "orthogonal_matched",
            "random_same_l2",
        ],
        default="learned",
    )
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--matched-component-id", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = validate_p6_pilot_config(args.config, require_ready=args.execute)
    if not args.execute:
        print(json.dumps({**report, "execution": "DRY_RUN_ONLY"}, indent=2, sort_keys=True))
        return

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config["execution"].get("local_execution_forbidden", True):
        raise RuntimeError("P6 config remains read-only; explicit execution approval is absent")
    if args.component_id is None or args.prompt_id is None or args.output is None:
        raise ValueError("--execute requires --component-id, --prompt-id and --output")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P6 output: {output}")

    cfg = load_experiment_config(config["base_config"])
    shortlist = load_100m_candidate_shortlist(
        config["candidate_source"]["shortlist_path"],
        package_manifest=config["candidate_source"]["package_manifest"],
        maximum=config["candidate_source"]["maximum_candidates"],
    )
    candidate_ids = {row["component_id"] for row in shortlist["candidates"]}
    if args.component_id not in candidate_ids:
        raise ValueError("component ID is absent from the verified current shortlist")
    rows = [
        json.loads(line)
        for line in Path(config["prompt_dataset"]["path"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [row for row in rows if row["prompt_id"] == args.prompt_id]
    if len(selected) != 1:
        raise ValueError("prompt ID is absent or duplicated in the verified manifest")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer, revision=cfg.data.tokenizer_revision
    )
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": selected[0]["prompt"]}],
        tokenize=True,
        add_generation_prompt=True,
    )
    device = torch.device(args.device)
    models = []
    provenances = []
    for spec in cfg.models:
        model, provenance = load_effective_model(spec, device)
        model.eval()
        models.append(model)
        provenances.append(provenance)
    decomposition = load_original_decomposition_for_p6(
        cfg,
        config["checkpoint"]["rank0_path"],
        models,
        provenances,
        device,
        expected_checkpoint_sha256=config["checkpoint"]["expected_sha256"],
    )
    eos_ids = {int(tokenizer.eos_token_id)}
    for model in models:
        configured = model.generation_config.eos_token_id
        if configured is not None:
            eos_ids.update(int(value) for value in (configured if isinstance(configured, list) else [configured]))
    checkpoint = report["artifact_verification"]["checkpoint"]
    result = run_original_p6_autoregressive(
        models,
        cfg,
        decomposition,
        prompt_id=args.prompt_id,
        prompt_token_ids=prompt_ids,
        component_id=args.component_id,
        model_index=1,
        arm=cast(P6Arm, args.arm),
        alpha=args.alpha,
        eos_token_ids=eos_ids,
        max_new_tokens=config["generation"]["max_new_tokens"],
        checkpoint=checkpoint,
        shortlist_sha256=shortlist["source_sha256"],
        prompt_manifest_sha256=config["prompt_dataset"]["sha256"],
        matched_component_id=args.matched_component_id,
        random_seed=args.random_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
