"""Interpretability: the intruder score, computed from a `harvest.db`.

For each sampled component, `n_trials` trials show the judge `n_real` activating examples plus
one example of a different component of similar firing density (within `density_tolerance`)
inserted at a random position; the score is the fraction of trials where the judge picks the
intruder (chance 1 / (n_real + 1)).
"""

import asyncio
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import override

from param_decomp.log import logger
from param_decomp_lab.app.backend.app_tokenizer import AppTokenizer
from param_decomp_lab.autointerp.llm_api import LLMError, LLMJob, LLMResult, map_llm_calls
from param_decomp_lab.autointerp.providers import (
    LLMProvider,
)
from param_decomp_lab.harvest.db import HarvestDB
from param_decomp_lab.harvest.intruder import (
    INTRUDER_SCHEMA,
    DensityIndex,
    IntruderResult,
    IntruderTrial,
    _build_trials,
    _TrialGroundTruth,
)
from param_decomp_lab.harvest.schemas import ActivationExample, ComponentData

from aspd.eval.autointerp_db import sample_keys
from aspd.eval.harvest_threshold import (
    CI_ACT_KEY,
    COMPONENT_ACT_KEY,
    STATS_FILE,
    Criterion,
    ThresholdedHarvestDB,
    anchor_index,
    anchor_value,
    component_scale,
    example_values,
    surviving_mask,
    threshold_stats,
)
from aspd.eval.judge import JudgeConfig, OpenAICompatProvider

_PEAK_ACT_PREFERENCE = ("activation", "causal_importance")


def crop_to_window(example: ActivationExample, tokens_per_side: int) -> ActivationExample:
    """Crop an example to `2*tokens_per_side + 1` tokens centred on its peak firing token."""
    window = 2 * tokens_per_side + 1
    n = len(example.token_ids)
    if n <= window:
        return example

    act_key = next(
        (k for k in _PEAK_ACT_PREFERENCE if k in example.activations),
        min(example.activations),
    )
    acts = example.activations[act_key]
    firing_positions = [i for i, f in enumerate(example.firings) if f]
    assert firing_positions, "activation example with no firing token -- not a harvest window"
    center = max(firing_positions, key=lambda i: acts[i])

    lo = max(0, center - tokens_per_side)
    hi = min(n, lo + window)
    lo = max(0, hi - window)
    return ActivationExample(
        token_ids=example.token_ids[lo:hi],
        firings=example.firings[lo:hi],
        activations={k: v[lo:hi] for k, v in example.activations.items()},
    )


class WindowedHarvestDB(HarvestDB):
    """A HarvestDB whose `get_component` hands back window-cropped activation examples."""

    def __init__(self, db_path: Path, tokens_per_side: int, readonly: bool = False) -> None:
        super().__init__(db_path, readonly)
        self.tokens_per_side = tokens_per_side

    @override
    def get_component(self, component_key: str) -> ComponentData | None:
        comp = super().get_component(component_key)
        if comp is None:
            return None
        return comp.model_copy(
            update={
                "activation_examples": [
                    crop_to_window(e, self.tokens_per_side) for e in comp.activation_examples
                ]
            }
        )


def eligible_density_entries(
    db: HarvestDB, *, min_examples: int, key_prefix: str | None = None
) -> list[tuple[str, float]]:
    """(key, firing_density) for components with enough examples, optionally prefix-filtered."""
    entries = db.get_component_densities(min_examples=min_examples)
    if key_prefix is not None:
        entries = [(k, d) for k, d in entries if k.startswith(key_prefix)]
    return entries


async def _score(
    db: HarvestDB,
    provider: LLMProvider,
    app_tok: AppTokenizer,
    remaining_keys: list[str],
    density_index: DensityIndex,
    *,
    n_real: int,
    n_trials: int,
    density_tolerance: float,
    max_concurrent: int,
    max_requests_per_minute: int,
    cost_limit_usd: float | None,
    score_type: str,
    prompt_prefix: str,
) -> list[IntruderResult]:
    """The lab's `run_intruder_scoring` fan-out loop, over an explicit key list."""
    ground_truth: dict[str, _TrialGroundTruth] = {}

    def jobs_iter() -> Iterator[LLMJob]:
        for job, gt in _build_trials(
            remaining_keys, db, density_index, n_real, n_trials, density_tolerance, app_tok
        ):
            ground_truth[job.key] = gt
            yield job

    component_trials: defaultdict[str, list[IntruderTrial]] = defaultdict(list)
    component_errors: defaultdict[str, int] = defaultdict(int)
    results: list[IntruderResult] = []

    def _try_save(ck: str) -> None:
        n_done = len(component_trials[ck]) + component_errors.get(ck, 0)
        if n_done < n_trials:
            return
        if component_errors.get(ck, 0) > 0:
            return
        trials = component_trials[ck]
        correct = sum(1 for t in trials if t.is_correct)
        score = correct / len(trials) if trials else 0.0
        result = IntruderResult(component_key=ck, score=score, trials=trials, n_errors=0)
        results.append(result)
        db.save_score(ck, score_type, score, json.dumps(asdict(result)))

    async for outcome in map_llm_calls(
        provider=provider,
        jobs=jobs_iter(),
        max_tokens=4000,
        max_concurrent=max_concurrent,
        max_requests_per_minute=max_requests_per_minute,
        cost_limit_usd=cost_limit_usd,
        response_schema=INTRUDER_SCHEMA,
        n_total=len(remaining_keys) * n_trials,
    ):
        match outcome:
            case LLMResult(job=job, parsed=parsed):
                gt = ground_truth[job.key]
                predicted = int(parsed["intruder"])
                component_trials[gt.component_key].append(
                    IntruderTrial(
                        correct_answer=gt.correct_answer,
                        predicted=predicted,
                        is_correct=predicted == gt.correct_answer,
                        reasoning=parsed.get("reasoning", ""),
                    )
                )
                db.save_intruder_prompt(prompt_prefix + job.key, job.prompt)
                _try_save(gt.component_key)
            case LLMError(job=job, error=e):
                gt = ground_truth[job.key]
                component_errors[gt.component_key] += 1
                logger.error(f"{job.key}: {type(e).__name__}: {e}")
                _try_save(gt.component_key)

    return results


def intruder_harvest_db(
    harvest_dir: Path,
    tokenizer_name: str,
    *,
    llm_config: JudgeConfig | None = None,
    n_real: int = 4,
    n_trials: int = 10,
    density_tolerance: float = 0.05,
    n_subsample: int | None = 200,
    seed: int = 0,
    key_prefix: str | None = None,
    restrict_donors_to_prefix: bool = True,
    keys: list[str] | None = None,
    window_tokens_per_side: int | None = None,
    ci_threshold: float | None = None,
    criterion: Criterion = "ci",
    cost_limit_usd: float | None = None,
    summary_path: Path | None = None,
    app_tok: AppTokenizer | None = None,
    provider: LLMProvider | None = None,
) -> Path:
    """Intruder-score a seeded subsample (or explicit `keys`) of a harvest.db; return summary path."""
    from dotenv import load_dotenv

    load_dotenv()
    harvest_dir = Path(harvest_dir)
    db_path = harvest_dir / "harvest.db"  # writable: scores are saved into it
    if ci_threshold is not None and ci_threshold <= 0.0:
        ci_threshold = None
    assert window_tokens_per_side is None or ci_threshold is None, (
        "`window_tokens_per_side` and `ci_threshold` are variants on the same axis; combining them "
        "would need a third score_type and no caller asks for it"
    )
    if ci_threshold is None:
        db = (
            HarvestDB(db_path)
            if window_tokens_per_side is None
            else WindowedHarvestDB(db_path, window_tokens_per_side)
        )
        score_type = (
            "intruder" if window_tokens_per_side is None else f"intruder_w{window_tokens_per_side}"
        )
        prompt_prefix = "" if window_tokens_per_side is None else f"w{window_tokens_per_side}/"
    else:
        db = ThresholdedHarvestDB(
            db_path,
            ci_threshold,
            threshold_stats(
                harvest_dir, [ci_threshold], criterion=criterion, min_examples=n_real + 1
            ),
        )
        score_type = f"intruder_{criterion}{ci_threshold:g}"
        prompt_prefix = f"{criterion}{ci_threshold:g}/"

    donor_entries = eligible_density_entries(
        db, min_examples=n_real + 1, key_prefix=key_prefix if restrict_donors_to_prefix else None
    )
    density_index = DensityIndex(list(donor_entries))  # copy: DensityIndex sorts in place

    eligible = eligible_density_entries(db, min_examples=n_real + 1, key_prefix=key_prefix)
    density_by_key = dict(eligible)
    eligible_keys = sorted(density_by_key)
    if keys is not None:
        missing = [k for k in keys if k not in density_by_key]
        assert not missing or ci_threshold is not None, (
            f"{len(missing)} explicit keys not eligible, e.g. {missing[:5]}"
        )
        sampled = sorted(k for k in keys if k in density_by_key)
    else:
        sampled = sample_keys(eligible_keys, n_subsample, seed) if n_subsample else eligible_keys

    completed = set(db.get_scores(score_type))
    remaining = [k for k in sampled if k not in completed]
    window_note = (
        "" if window_tokens_per_side is None
        else f", examples cropped to +-{window_tokens_per_side} tokens"
    )
    print(
        f"[intruder] {len(sampled)} sampled of {len(eligible_keys)} eligible "
        f"(donor pool {len(donor_entries)}); {len(sampled) - len(remaining)} already scored, "
        f"{len(remaining)} to go ({len(remaining) * n_trials} trials)"
        f" [score_type={score_type}{window_note}]",
        flush=True,
    )

    llm_config = llm_config or JudgeConfig()
    if remaining:
        created = provider is None
        prov = provider if provider is not None else OpenAICompatProvider(llm_config)

        async def _run() -> None:
            try:
                await _score(
                    db,
                    prov,
                    app_tok or AppTokenizer.from_pretrained(tokenizer_name),
                    remaining,
                    density_index,
                    n_real=n_real,
                    n_trials=n_trials,
                    density_tolerance=density_tolerance,
                    max_concurrent=llm_config.max_concurrent,
                    max_requests_per_minute=llm_config.max_requests_per_minute,
                    cost_limit_usd=cost_limit_usd,
                    score_type=score_type,
                    prompt_prefix=prompt_prefix,
                )
            finally:
                if created:
                    await prov.close()

        asyncio.run(_run())

    all_scores = db.get_scores(score_type)
    scores = {k: all_scores[k] for k in sampled if k in all_scores}
    missing_keys = [k for k in sampled if k not in all_scores]  # errored or donor-less; rerun resumes
    values = list(scores.values())

    if summary_path is None:
        suffix = f"_{re.sub(r'[^A-Za-z0-9]+', '_', key_prefix).strip('_')}" if key_prefix else ""
        window_suffix = "" if window_tokens_per_side is None else f"_w{window_tokens_per_side}"
        ci_suffix = "" if ci_threshold is None else f"_{criterion}{ci_threshold:g}"
        summary_path = harvest_dir / f"intruder_summary{suffix}{window_suffix}{ci_suffix}.json"
    summary = {
        "config": {
            "llm": llm_config.model_dump(),
            "n_real": n_real,
            "n_trials": n_trials,
            "density_tolerance": density_tolerance,
            "n_subsample": n_subsample,
            "seed": seed,
            "key_prefix": key_prefix,
            "restrict_donors_to_prefix": restrict_donors_to_prefix,
            "keys_explicit": keys is not None,
            "window_tokens_per_side": window_tokens_per_side,
            "ci_threshold": ci_threshold,
            "ci_criterion": criterion if ci_threshold is not None else None,
            "score_type": score_type,
            "judge_provider": type(provider).__name__ if provider is not None else "from_llm_config",
            "judge_url": str(getattr(provider, "_client", None) and provider._client.base_url)
            if provider is not None else None,
            "protocol": "lab run_intruder_scoring (unseeded trial composition)",
        },
        "harvest_config": db.get_config_dict(),
        "harvest_dir": str(harvest_dir),
        "tokenizer_name": tokenizer_name,
        "n_eligible": len(eligible_keys),
        "n_donor_pool": len(donor_entries),
        "n_keys_requested": len(keys) if keys is not None else None,
        "n_keys_dropped": len(keys) - len(sampled) if keys is not None else None,
        "sampled_keys": sampled,
        "n_scored": len(scores),
        "scores": scores,
        "missing_keys": missing_keys,
        "mean": statistics.mean(values) if values else None,
        "std": statistics.pstdev(values) if len(values) > 1 else None,
        "firing_density": {k: density_by_key[k] for k in sampled},
        "created": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    db.close()
    print(
        f"[intruder] {len(scores)} scored (mean {summary['mean']}, {len(missing_keys)} missing) "
        f"-> {summary_path}",
        flush=True,
    )
    return summary_path


CI_STATS_FILE = STATS_FILE["ci"]


def anchor_ci(example: ActivationExample, half: int) -> tuple[float, bool]:
    """`(CI at the anchor, whether the anchor had to be guessed)` -- the `ci` criterion."""
    return anchor_value(example, "ci", half)


def anchor_cis(comp: ComponentData, half: int) -> tuple[list[float], int]:
    """Each stored example's anchor CI, plus how many anchors had to be guessed."""
    pairs = [anchor_ci(e, half) for e in comp.activation_examples]
    return [v for v, _ in pairs], sum(1 for _, guessed in pairs if guessed)


def ci_threshold_stats(
    harvest_dir: Path, thresholds, *, min_examples: int, rebuild: bool = False
) -> dict:
    """`threshold_stats` at the `ci` criterion."""
    return threshold_stats(
        harvest_dir, list(thresholds), criterion="ci", min_examples=min_examples, rebuild=rebuild
    )


__all__ = [
    "CI_ACT_KEY", "CI_STATS_FILE", "COMPONENT_ACT_KEY", "Criterion",
    "STATS_FILE", "ThresholdedHarvestDB", "WindowedHarvestDB", "anchor_ci", "anchor_cis",
    "anchor_index", "anchor_value", "ci_threshold_stats", "component_scale", "crop_to_window",
    "eligible_density_entries", "example_values", "intruder_harvest_db", "surviving_mask",
    "threshold_stats",
]
