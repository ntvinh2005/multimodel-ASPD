"""Label SAE latents or components with an LLM from their activation examples in a `harvest.db`."""

import asyncio
import random
from collections.abc import Sequence
from pathlib import Path

from param_decomp_lab.autointerp.config import CanonConfig, CompactSkepticalConfig, StrategyConfig
from param_decomp_lab.autointerp.db import InterpDB
from param_decomp_lab.autointerp.llm_api import LLMError, LLMJob, LLMResult, map_llm_calls
from param_decomp_lab.autointerp.schemas import InterpretationResult, ModelMetadata
from param_decomp_lab.autointerp.strategies.dispatch import INTERPRETATION_SCHEMA, format_prompt
from param_decomp_lab.harvest.analysis import get_input_token_stats, get_output_token_stats
from param_decomp_lab.harvest.db import HarvestDB
from param_decomp_lab.harvest.storage import TokenStatsStorage

from aspd.eval.judge import JudgeConfig, OpenAICompatProvider


def sample_keys(keys: Sequence[str], cap: int, seed: int) -> list[str]:
    """A seeded random subset of at most `cap` keys (all of them if fewer), sorted for stability."""
    if len(keys) <= cap:
        return list(keys)
    return sorted(random.Random(seed).sample(list(keys), cap))


def sample_keys_per_site(keys: Sequence[str], cap: int, seed: int) -> list[str]:
    """`cap` keys per SITE, not `cap` overall."""
    by_site: dict[str, list[str]] = {}
    for key in keys:
        by_site.setdefault(key.rsplit(":", 1)[0], []).append(key)
    out: list[str] = []
    for i, site in enumerate(sorted(by_site)):
        out += sample_keys(sorted(by_site[site]), cap, seed + i)
    return sorted(out)


def _build_jobs(
    keys: list[str],
    db: HarvestDB,
    storage: TokenStatsStorage | None,
    app_tok,
    strategy: StrategyConfig,
    model_metadata: ModelMetadata,
    *,
    context_tokens_per_side: int,
    activation_threshold: float,
    top_k: int,
):
    for key in keys:
        comp = db.get_component(key)
        assert comp is not None, f"{key} vanished from harvest.db between listing and read"
        prompt = format_prompt(
            strategy=strategy,
            component=comp,
            model_metadata=model_metadata,
            app_tok=app_tok,
            input_token_stats=(
                get_input_token_stats(storage, key, app_tok, top_k) if storage else None
            ),
            output_token_stats=(
                get_output_token_stats(storage, key, app_tok, top_k) if storage else None
            ),
            context_tokens_per_side=context_tokens_per_side,
            activation_threshold=activation_threshold,
        )
        yield LLMJob(prompt=prompt, key=key)


async def _label(
    provider,
    jobs,
    db_out: InterpDB,
    *,
    n_total: int,
    max_concurrent: int,
    max_requests_per_minute: int,
    max_tokens: int = 8000,
) -> tuple[int, int]:
    saved = errors = 0
    async for outcome in map_llm_calls(
        provider=provider,
        jobs=jobs,
        max_tokens=max_tokens,
        max_concurrent=max_concurrent,
        max_requests_per_minute=max_requests_per_minute,
        cost_limit_usd=None,
        response_schema=INTERPRETATION_SCHEMA,
        n_total=n_total,
    ):
        match outcome:
            case LLMResult(job=job, parsed=parsed, raw=raw):
                db_out.save_interpretation(
                    InterpretationResult(
                        component_key=job.key, label=parsed["label"],
                        reasoning=parsed["reasoning"], raw_response=raw, prompt=job.prompt,
                    )
                )
                saved += 1
            case LLMError(job=job):
                errors += 1
    return saved, errors


def autointerp_harvest_db(
    harvest_dir: Path,
    tokenizer_name: str,
    model_metadata: ModelMetadata,
    *,
    cap: int = 200,
    seed: int = 0,
    context_tokens_per_side: int,
    activation_threshold: float = 1e-6,
    llm_config: JudgeConfig | None = None,
    strategy: StrategyConfig | None = None,
    top_k: int = 20,
    max_concurrent: int = 10,
    max_requests_per_minute: int = 200,
    out_db: Path | None = None,
    per_site_cap: bool = False,
    keys: list[str] | None = None,
    provider: object | None = None,
    min_success_frac: float = 0.5,
    max_tokens: int = 8000,
) -> Path:
    """Label a capped, seeded subset of the harvest's latents and write `interp.db`."""
    from dotenv import load_dotenv
    from param_decomp_lab.app.backend.app_tokenizer import AppTokenizer

    load_dotenv()
    harvest_dir = Path(harvest_dir)
    db = HarvestDB(harvest_dir / "harvest.db", readonly=True)
    stats_path = harvest_dir / "token_stats.pt"
    storage = TokenStatsStorage.load(stats_path) if stats_path.exists() else None
    if storage is None:
        needs_stats = not isinstance(strategy, CanonConfig)
        assert strategy is None or not needs_stats, (
            f"no token_stats.pt at {stats_path}, but {type(strategy).__name__} needs input/output "
            "token stats. Pass CanonConfig(), or re-harvest with --token-stats."
        )
        print(
            f"[autointerp] no token_stats.pt at {stats_path} -- using the `canon` strategy, which "
            "reads examples only. Labels are NOT comparable to compact_skeptical labels.",
            flush=True,
        )
    app_tok = AppTokenizer.from_pretrained(tokenizer_name)
    provider = provider or OpenAICompatProvider(llm_config or JudgeConfig(structured=True))
    strategy = strategy or (CompactSkepticalConfig() if storage is not None else CanonConfig())

    if keys is None:
        sampler = sample_keys_per_site if per_site_cap else sample_keys
        keys = sampler(db.get_component_keys(), cap, seed)
    scope = "per site" if per_site_cap else "overall"
    print(f"[autointerp] labeling {len(keys)} latents (cap {cap} {scope}, seed {seed})", flush=True)

    out_db = out_db or harvest_dir / "interp.db"
    interp = InterpDB(Path(out_db))
    jobs = _build_jobs(
        keys, db, storage, app_tok, strategy, model_metadata,
        context_tokens_per_side=context_tokens_per_side,
        activation_threshold=activation_threshold, top_k=top_k,
    )
    saved, errors = asyncio.run(
        _label(provider, jobs, interp, n_total=len(keys), max_tokens=max_tokens,
               max_concurrent=max_concurrent, max_requests_per_minute=max_requests_per_minute)
    )
    print(f"[autointerp] {saved} labels ({errors} errors) -> {out_db}", flush=True)
    assert saved >= min_success_frac * len(keys), (
        f"only {saved}/{len(keys)} latents labelled ({errors} errors). This is a systematic "
        f"failure, not flakiness -- check that the judge's context window fits the prompt "
        f"(shrink `strategy.max_examples`) and that structured output is enabled."
    )
    return Path(out_db)
