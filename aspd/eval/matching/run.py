"""Matching end to end: accumulate effects, pair, fetch examples, judge, write the report."""

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from param_decomp_lab.harvest.db import HarvestDB
from torch import Tensor

from aspd.eval.adapters.capture import capture_site
from aspd.eval.adapters.dictionary import assert_deterministic_encode
from aspd.eval.matching import MODES, report_name
from aspd.eval.matching.alignment import (
    AlignmentAccumulator,
    InputAlignmentAccumulator,
    raw_footprint_matrix,
    raw_input_footprint_matrix,
)
from aspd.eval.matching.examples import Example, feature_examples
from aspd.eval.matching.judge import get_generation_prompts, judge_pairs
from aspd.eval.matching.match import (
    Pairing,
    chained_top1,
    eligible,
    random_control,
    subsample_components,
    top1,
)
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE
from aspd.sae.sites import HookSite, SitePair


def input_hook_site(sites: SitePair) -> HookSite:
    return HookSite(
        key=sites.input_site,
        module=sites.input_module or sites.input_site,
        take=sites.input_take,
    )


def assert_input_pair_adjacent(sites: SitePair) -> None:
    adjacent = sites.input_take == "output" or sites.input_module == sites.module
    assert adjacent, (
        f"the input dictionary is fitted on the INPUT of {sites.input_site!r}, which is upstream "
        f"of the tensor {sites.module!r} reads -- adjacency is broken, so the input leg "
        "would rank on a direction overlap across a nonlinearity rather than on ∂z_c/∂f_i. Point "
        "--sae-dir at an adjacent pair (for GPT-2, the `*_c_fc` pair rather than the `*_resid` one)"
    )


@dataclass(frozen=True)
class MatchingResult:
    name: str
    n_pairs: int
    mean_score: float
    score_histogram: dict[int, int]
    pairs: list[tuple[int, int]]
    components: list[int]
    identity_rate: float
    scores: list[int]


@dataclass(frozen=True)
class Alignments:
    """What one pass over the tokens produces. The `in_*` half is `None` on `c2o`."""

    glob: Tensor
    cond: Tensor
    acc: AlignmentAccumulator
    in_glob: Tensor | None = None
    in_cond: Tensor | None = None
    acc_in: InputAlignmentAccumulator | None = None


@torch.no_grad()
def accumulate_alignment(
    source,
    sae_out: MatryoshkaBatchTopKSAE,
    token_batches,
    *,
    device: torch.device,
    sae_in: MatryoshkaBatchTopKSAE | None = None,
    input_site: HookSite | None = None,
    target_model=None,
    tokenizer=None,
) -> Alignments:
    """One pass over `token_batches`; the input leg is accumulated only when `sae_in` is given."""
    assert_deterministic_encode(sae_out)
    chained = sae_in is not None
    assert not chained or None not in (input_site, target_model, tokenizer), (
        "the i2o input leg needs the input site and the frozen target to capture it from"
    )
    acc = AlignmentAccumulator(
        n_features=sae_out.cfg.n_features, n_components=source.n_latents, device=device
    )
    acc_in = (
        InputAlignmentAccumulator(
            n_input_features=sae_in.cfg.n_features, n_components=source.n_latents, device=device
        )
        if chained
        else None
    )
    if chained:
        assert_deterministic_encode(sae_in)
    for tokens in token_batches:
        tokens = tokens.to(device)
        batch = source.encode_batch(tokens)
        keep = batch.token_mask.reshape(-1)
        zeta = batch.ablation.reshape(-1, source.n_latents)[keep]
        active = sae_out.active_mask(batch.site_acts.reshape(-1, batch.site_acts.shape[-1]))[keep]
        acc.update(zeta, active)
        if not chained:
            continue

        x, mask = capture_site(target_model, input_site, tokens, tokenizer)
        assert bool((mask == batch.token_mask).all()), (
            "the capture and the source disagree on which positions are kept; they were computed "
            "from the same tokens by the same rule, so this is a tokenizer mismatch"
        )
        features = sae_in.features(x.reshape(-1, x.shape[-1])[keep])
        acc_in.update(features, zeta != 0)

    glob, cond = acc.alignments(raw_footprint_matrix(sae_out, source.write_vectors))
    if not chained:
        return Alignments(glob=glob, cond=cond, acc=acc)
    in_glob, in_cond = acc_in.alignments(
        raw_input_footprint_matrix(sae_in, source.read_vectors)
    )
    return Alignments(glob=glob, cond=cond, acc=acc,
                      in_glob=in_glob, in_cond=in_cond, acc_in=acc_in)


def build_prompts_c2o(
    pairing: Pairing,
    component_db: HarvestDB,
    feature_db: HarvestDB,
    component_key: str,
    feature_key: str,
    decode,
    *,
    component_activation_key: str,
    feature_activation_key: str,
) -> list[list[dict]]:
    prompts = []
    for c, j in pairing.pairs:
        comp = component_db.get_component(f"{component_key}:{c}")
        feat = feature_db.get_component(f"{feature_key}:{j}")
        assert comp is not None, f"component {component_key}:{c} missing from the harvest"
        assert feat is not None, f"feature {feature_key}:{j} missing from the harvest"
        a: list[Example] = feature_examples(comp, decode, component_activation_key)
        b: list[Example] = feature_examples(feat, decode, feature_activation_key)
        prompts.append(get_generation_prompts(a, b))
    return prompts


def build_prompts_i2o(
    pairing: Pairing,
    feature_db: HarvestDB,
    input_key: str,
    output_key: str,
    decode,
    *,
    activation_key: str,
) -> list[list[dict]]:
    prompts = []
    for i, j in pairing.pairs:
        src = feature_db.get_component(f"{input_key}:{i}")
        dst = feature_db.get_component(f"{output_key}:{j}")
        assert src is not None, f"input feature {input_key}:{i} missing from the harvest"
        assert dst is not None, f"output feature {output_key}:{j} missing from the harvest"
        a: list[Example] = feature_examples(src, decode, activation_key)
        b: list[Example] = feature_examples(dst, decode, activation_key)
        prompts.append(get_generation_prompts(a, b))
    return prompts


def score_pairing(pairing: Pairing, scores: list[int]) -> MatchingResult:
    assert len(scores) == len(pairing.pairs)
    histogram = {s: scores.count(s) for s in (1, 2, 3)}
    return MatchingResult(
        name=pairing.name,
        n_pairs=len(scores),
        mean_score=sum(scores) / len(scores) if scores else float("nan"),
        score_histogram=histogram,
        pairs=pairing.pairs,
        components=pairing.components,
        identity_rate=pairing.identity_rate,
        scores=scores,
    )


def run_judging(
    pairings: list[Pairing],
    prompts_per_pairing: list[list[list[dict]]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int,
) -> list[MatchingResult]:
    results = []
    for pairing, prompts in zip(pairings, prompts_per_pairing, strict=True):
        scores = asyncio.run(
            judge_pairs(prompts, base_url=base_url, api_key=api_key, model=model, concurrency=concurrency)  # type: ignore[arg-type]
        )
        results.append(score_pairing(pairing, scores))
        print(f"[matching] {pairing.name}: mean {results[-1].mean_score:.3f} over {len(scores)}")
    return results


def write_report(results: list[MatchingResult], meta: dict, out_dir: Path, *, mode: str) -> None:
    """Write `matching_<mode>_step<N>.json`, stamping the mode inside it as well."""
    assert mode in MODES, mode
    path = Path(out_dir) / report_name(mode, meta["step"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"meta": {**meta, "scheme": mode}, "results": [asdict(r) for r in results]}, indent=2
        )
    )
    print(f"[matching] wrote {path}")


def select_pairings(
    alignments: Alignments,
    judgeable_components: set[int],
    judgeable_features: set[int],
    *,
    mode: str,
    judgeable_input_features: set[int] | None = None,
    n_subsample: int,
    seed: int,
) -> list[Pairing]:
    """The three judged sets of either mode, all on ONE component subsample."""
    assert mode in MODES, mode
    acc, acc_in = alignments.acc, alignments.acc_in
    chained = mode == "i2o"
    assert chained == (acc_in is not None), (
        f"mode {mode!r} and the accumulated alignments disagree about the input leg"
    )
    if chained:
        assert bool((acc.dead_components.cpu() == acc_in.dead_components.cpu()).all()), (
            "the two accumulators disagree on which components are dead; `Σ|ζ| == 0` and "
            "`#{ζ != 0} == 0` are the same set, so one of the two passes saw different tokens"
        )
    comps, feats, in_feats = eligible(
        acc.dead_components.cpu(),
        acc.dead_output_features.cpu(),
        judgeable_components,
        judgeable_features,
        acc_in.dead_input_features.cpu() if chained else None,
        judgeable_input_features if chained else None,
    )
    print(
        f"[matching] eligible: {comps.numel()} components, "
        + (f"{in_feats.numel()} input features, " if chained else "")
        + f"{feats.numel()} output features"
    )
    chosen = subsample_components(comps, n_subsample=n_subsample, seed=seed)
    print(f"[matching] judging {chosen.numel()} components, shared across all three pairings")
    if chained:
        pairings = [
            chained_top1(alignments.in_glob.cpu(), alignments.glob.cpu(), chosen, in_feats, feats,
                         name="A_glob"),
            chained_top1(alignments.in_cond.cpu(), alignments.cond.cpu(), chosen, in_feats, feats,
                         name="A_cond"),
        ]
    else:
        pairings = [
            top1(alignments.glob.cpu(), chosen, feats, name="A_glob"),
            top1(alignments.cond.cpu(), chosen, feats, name="A_cond"),
        ]
    return [*pairings, random_control(pairings[1], feats, seed=seed)]
