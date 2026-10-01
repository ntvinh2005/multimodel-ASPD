"""Harvest activation examples and statistics of SAE latents into a `harvest.db`."""

import hashlib
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import torch
import tqdm
from einops import rearrange, reduce
from jaxtyping import Bool, Float, Int
from param_decomp_lab.harvest.accumulator import (
    Harvester,
    _compute_token_pmi,
    _log_base_rate_summary,
)
from param_decomp_lab.harvest.config import HarvestConfig, ParamDecompHarvestConfig
from param_decomp_lab.harvest.db import HarvestDB
from param_decomp_lab.harvest.reservoir import WINDOW_PAD_SENTINEL
from param_decomp_lab.harvest.schemas import (
    ComponentData,
    ComponentTokenPMI,
    get_harvest_subrun_dir,
)
from param_decomp_lab.harvest.storage import TokenStatsStorage
from torch import Tensor, nn

from aspd.eval.chunked_pmi import ChunkedTokenPmiRanker
from aspd.eval.dictionary import DictionaryAdapter
from aspd.sae.sites import OutputCapture

ACTIVATION_KEY = "activation"
_PD_RUN_ID_RE = re.compile(r"^p-[a-z0-9]{8}$")


def assert_contiguous_real_mask(real_mask: Bool[Tensor, "B S"]) -> None:
    """Every row must be real tokens then pad -- never pad, then real again."""
    n_real = real_mask.sum(dim=1)
    prefix = torch.arange(real_mask.shape[1], device=real_mask.device) < n_real.unsqueeze(1)
    bad = (prefix != real_mask).any(dim=1)
    assert not bool(bad.any()), (
        f"{int(bad.sum())} row(s) have padding before a real token; the harvest strips pads from "
        "stored examples and that is only safe for right-padded (or unpadded) rows"
    )


def _pd_run_id(name: str) -> str:
    """A `ParamDecompHarvestConfig`-valid `p-xxxxxxxx` id (provenance stamp only)."""
    return name if _PD_RUN_ID_RE.match(name) else "p-" + hashlib.sha1(name.encode()).hexdigest()[:8]


class PadMaskedLatentHarvester(Harvester):
    """`Harvester` that excludes right-padding positions from every statistic AND every example."""

    def __init__(self, *args, vocab_size: int, collect_token_stats: bool = False, **kwargs) -> None:
        super().__init__(*args, vocab_size=0, **kwargs)
        self.vocab_size = vocab_size
        self.collect_token_stats = collect_token_stats
        # Populated only in `"topk"` mode; `build_results` prefers it over the dense matrices.
        self.pmi_rankers: tuple[ChunkedTokenPmiRanker, ChunkedTokenPmiRanker] | None = None
        self.cooccurrence_counts = torch.zeros(0, 0, device=self.device)
        self.input_marginals = torch.zeros(vocab_size, device=self.device, dtype=torch.long)
        self.output_marginals = torch.zeros(vocab_size, device=self.device)
        if collect_token_stats:
            n_components = sum(c for _, c in self.layers)
            self.input_cooccurrence = torch.zeros(
                n_components, vocab_size, device=self.device, dtype=torch.int32
            )
            self.output_cooccurrence = torch.zeros(n_components, vocab_size, device=self.device)

    def build_results(self, pmi_top_k_tokens: int) -> Iterator[ComponentData]:
        """The parent's walk, over accumulators this class owns the shape and dtype of."""
        mean_activations = {
            act_type: (self.activation_sums[act_type] / self.total_tokens_processed).cpu()
            for act_type in self.activation_sums
        }
        firing_counts = self.firing_counts.cpu()
        input_marginals = self.input_marginals.cpu()
        output_marginals = self.output_marginals.cpu()
        input_cooccurrence = self.input_cooccurrence.cpu()
        output_cooccurrence = self.output_cooccurrence.cpu()
        reservoir_cpu = self.reservoir.to(torch.device("cpu"))
        _log_base_rate_summary(firing_counts, input_marginals)
        empty_pmi = ComponentTokenPMI(top=[], bottom=[])
        rankers = self.pmi_rankers

        for layer, layer_c in self.layers:
            offset = self.layer_offsets[layer]
            for component_idx in tqdm.tqdm(range(layer_c), desc="Building components"):
                flat_idx = offset + component_idx
                n_firings = float(firing_counts[flat_idx])
                if n_firings == 0:
                    continue
                if rankers is not None:
                    in_top, in_bottom = rankers[0].finalize(flat_idx)
                    out_top, out_bottom = rankers[1].finalize(flat_idx)
                    input_pmi = ComponentTokenPMI(top=in_top, bottom=in_bottom)
                    output_pmi = ComponentTokenPMI(top=out_top, bottom=out_bottom)
                elif self.collect_token_stats:
                    input_pmi = _compute_token_pmi(
                        input_cooccurrence[flat_idx].float(),
                        input_marginals,
                        n_firings,
                        self.total_tokens_processed,
                        pmi_top_k_tokens,
                    )
                    output_pmi = _compute_token_pmi(
                        output_cooccurrence[flat_idx],
                        output_marginals,
                        n_firings,
                        self.total_tokens_processed,
                        pmi_top_k_tokens,
                    )
                else:
                    input_pmi = output_pmi = empty_pmi
                yield ComponentData(
                    component_key=f"{layer}:{component_idx}",
                    layer=layer,
                    component_idx=component_idx,
                    firing_density=n_firings / self.total_tokens_processed,
                    mean_activations={
                        act_type: float(mean_activations[act_type][flat_idx].item())
                        for act_type in mean_activations
                    },
                    activation_examples=list(reservoir_cpu.examples(flat_idx)),
                    input_token_pmi=input_pmi,
                    output_token_pmi=output_pmi,
                )

    def process_batch_masked(
        self,
        batch: Int[Tensor, "B S"],
        firings: dict[str, Bool[Tensor, "B S C"]],
        activations: dict[str, dict[str, Float[Tensor, "B S C"]]],
        output_probs: Float[Tensor, "B S V"],
        real_mask: Bool[Tensor, "B S"],
    ) -> None:
        self.total_tokens_processed += int(real_mask.sum().item())
        m = real_mask.unsqueeze(-1)
        tokens_flat = rearrange(batch, "b s -> (b s)")
        probs_flat = rearrange(output_probs, "b s v -> (b s) v")
        mask_flat = rearrange(real_mask, "b s -> (b s)")

        firings_cat = torch.cat([firings[layer] for layer in self.layer_names], dim=-1) & m
        firings_flat = rearrange(firings_cat, "b s lc -> (b s) lc")

        act_types = list(activations[self.layer_names[0]].keys())
        activations_cat: dict[str, Tensor] = {}
        for act_type in act_types:
            cat = torch.cat([activations[layer][act_type] for layer in self.layer_names], dim=-1)
            activations_cat[act_type] = cat * m

        self.firing_counts += reduce(firings_cat.float(), "b s lc -> lc", "sum")
        for act_type, act in activations_cat.items():
            self.activation_sums[act_type] += reduce(act, "b s lc -> lc", "sum")

        firings_float = firings_flat.float()

        if self.collect_token_stats:
            n_components = firings_float.shape[1]
            token_indices = tokens_flat.unsqueeze(0).expand(n_components, -1)
            self.input_cooccurrence.scatter_add_(
                1, token_indices, rearrange(firings_float, "s lc -> lc s").int().contiguous()
            )
        self.input_marginals.scatter_add_(0, tokens_flat, mask_flat.long())
        probs_masked = probs_flat * mask_flat.unsqueeze(-1)
        if self.collect_token_stats:
            self.output_cooccurrence.addmm_(firings_float.t(), probs_masked)
        self.output_marginals += reduce(probs_masked, "s v -> v", "sum")

        assert_contiguous_real_mask(real_mask)
        self._collect_activation_examples(
            batch.masked_fill(~real_mask, WINDOW_PAD_SENTINEL), firings_cat, activations_cat
        )


def _save_results(
    harvester: Harvester, config: HarvestConfig, output_dir: Path, *, mode: str
) -> None:
    """`HarvestRepo.save_results` minus `component_correlations.pt` (see the class docstring)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    db = HarvestDB(output_dir / "harvest.db")
    db.save_config(config)
    n_saved = db.save_components_iter(
        harvester.build_results(pmi_top_k_tokens=config.pmi_token_top_k)
    )
    db.close()
    print(f"[harvest] saved {n_saved} components to {output_dir / 'harvest.db'}", flush=True)

    if mode != "full":
        stale = output_dir / "token_stats.pt"
        if stale.exists():
            stale.rename(output_dir / "token_stats.pt.stale")
            print(
                f"[harvest] token stats disabled, but {stale.name} existed from an earlier "
                f"harvest -- renamed to {stale.name}.stale so nothing pairs it with THIS run's "
                "examples. Delete it to reclaim the space.",
                flush=True,
            )
        if mode == "topk":
            print(
                f"[harvest] token_stats='topk' -- no token_stats.pt written; harvest.db carries "
                f"the top/bottom {config.pmi_token_top_k} tokens per component, which is what "
                "every reader displays. autointerp's compact_skeptical/dual_view/rich_examples "
                "need the sidecar, so use token_stats='full' for those.",
                flush=True,
            )
        else:
            print(
                "[harvest] token stats disabled -- no token_stats.pt written, harvest.db PMI "
                "columns are empty (autointerp needs --strategy canon; see harvest_dictionaries' "
                "docstring)",
                flush=True,
            )
        return

    token_stats = TokenStatsStorage(
        component_keys=harvester.component_keys,
        vocab_size=harvester.vocab_size,
        n_tokens=harvester.total_tokens_processed,
        input_counts=harvester.input_cooccurrence.float().cpu(),
        input_totals=harvester.input_marginals.float().cpu(),
        output_counts=harvester.output_cooccurrence.cpu(),
        output_totals=harvester.output_marginals.cpu(),
        firing_counts=harvester.firing_counts.cpu(),
    )
    token_stats.save(output_dir / "token_stats.pt")


@torch.no_grad()
def _latent_firings_and_acts(
    dictionaries: list[DictionaryAdapter], site_acts: dict[str, Tensor]
) -> tuple[dict[str, Tensor], dict[str, dict[str, Tensor]]]:
    """Encode each site's activation through its dictionary -> per-site firings/acts."""
    firings: dict[str, Tensor] = {}
    activations: dict[str, dict[str, Tensor]] = {}
    for d in dictionaries:
        feats = d.encode(site_acts[d.site_path])
        firings[d.site_path] = feats > 0
        activations[d.site_path] = {ACTIVATION_KEY: feats.float()}
    return firings, activations


def _pmi_chunk_width(n_components: int, vocab_size: int, budget_bytes: int) -> int:
    """Widest vocabulary slice whose two accumulators fit `budget_bytes`."""
    per_token = max(1, n_components * 8)
    return max(1, min(vocab_size, budget_bytes // per_token))


@torch.no_grad()
def _accumulate_chunked_pmi(
    *,
    model: nn.Module,
    dictionaries: list[DictionaryAdapter],
    cached_batches: list[Tensor],
    logits_fn: Callable[[Tensor], Tensor] | None,
    site_acts_fn: Callable[[Tensor], tuple[Tensor, dict[str, Tensor]]] | None,
    site_paths: list[str],
    takes: dict[str, str],
    modules: dict[str, str],
    harvester: "PadMaskedLatentHarvester",
    vocab_size: int,
    pad_id: int,
    top_k: int,
    budget_bytes: int,
    dev: torch.device,
) -> tuple[ChunkedTokenPmiRanker, ChunkedTokenPmiRanker]:
    """Re-derive the token-PMI ranking one vocabulary slice at a time, never holding [C, vocab]."""
    n_components = sum(c for _, c in harvester.layers)
    width = _pmi_chunk_width(n_components, vocab_size, budget_bytes)
    n_chunks = (vocab_size + width - 1) // width
    print(
        f"[harvest] token PMI: {n_chunks} vocab slice(s) of <={width} over {n_components} "
        f"components ({n_components * width * 8 / 1024**3:.1f} GB per slice); "
        f"{n_chunks} extra pass(es) over {len(cached_batches)} cached batches.",
        flush=True,
    )

    def _forward(batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """`(firings_flat, probs_masked, tokens_flat)` -- pass 1's computation, verbatim."""
        if site_acts_fn is None:
            assert logits_fn is not None
            with OutputCapture(
                model, site_paths, detach=True, stop_when_complete=False,
                takes=takes, modules=modules,
            ) as cap:
                logits = logits_fn(batch)
            site_acts = {p: cap[p] for p in site_paths}
        else:
            logits, site_acts = site_acts_fn(batch)
        firings, _ = _latent_firings_and_acts(dictionaries, site_acts)
        real_mask = batch != pad_id
        m = real_mask.unsqueeze(-1)
        firings_cat = torch.cat([firings[layer] for layer in harvester.layer_names], dim=-1) & m
        firings_flat = rearrange(firings_cat, "b s lc -> (b s) lc").float()
        probs_flat = rearrange(logits.float().softmax(dim=-1), "b s v -> (b s) v")
        mask_flat = rearrange(real_mask, "b s -> (b s)")
        tokens_flat = rearrange(batch, "b s -> (b s)")
        return firings_flat, probs_flat * mask_flat.unsqueeze(-1), tokens_flat

    in_ranker = ChunkedTokenPmiRanker(n_components, vocab_size, top_k, dev)
    out_ranker = ChunkedTokenPmiRanker(n_components, vocab_size, top_k, dev)
    firing_counts = harvester.firing_counts
    total_tokens = harvester.total_tokens_processed
    input_marginals = harvester.input_marginals.float()
    output_marginals = harvester.output_marginals

    for lo in range(0, vocab_size, width):
        hi = min(lo + width, vocab_size)
        w = hi - lo
        in_cooc = torch.zeros(n_components, w, dtype=torch.int32, device=dev)
        out_cooc = torch.zeros(n_components, w, device=dev)
        for batch_cpu in tqdm.tqdm(cached_batches, desc=f"PMI vocab [{lo},{hi})"):
            firings_flat, probs_masked, tokens_flat = _forward(batch_cpu.to(dev))
            sel = (tokens_flat >= lo) & (tokens_flat < hi)
            if bool(sel.any()):
                local = (tokens_flat[sel] - lo).unsqueeze(0).expand(n_components, -1)
                in_cooc.scatter_add_(1, local, firings_flat[sel].t().int().contiguous())
            out_cooc.addmm_(firings_flat.t(), probs_masked[:, lo:hi])
        in_ranker.add_chunk(in_cooc.float(), input_marginals[lo:hi], firing_counts, total_tokens, lo)
        out_ranker.add_chunk(out_cooc, output_marginals[lo:hi], firing_counts, total_tokens, lo)
        del in_cooc, out_cooc

    return in_ranker, out_ranker


@torch.no_grad()
def harvest_dictionaries(
    model: nn.Module,
    dictionaries: list[DictionaryAdapter],
    token_batches: Iterator[Tensor],
    logits_fn: Callable[[Tensor], Tensor] | None = None,
    *,
    harvest_id: str,
    vocab_size: int,
    pad_id: int,
    n_batches: int,
    context_tokens_per_side: int,
    examples_per_component: int,
    max_examples_per_batch_per_component: int = 5,
    pmi_token_top_k: int = 20,
    token_stats: Literal["off", "topk", "full"] = "topk",
    pmi_memory_budget_gb: float = 16.0,
    out_dir: Path | None = None,
    device: str = "cuda",
    site_acts_fn: Callable[[Tensor], tuple[Tensor, dict[str, Tensor]]] | None = None,
) -> Path:
    """Single-pass pad-masked harvest of `dictionaries` into a fresh `harvest.db` subrun."""
    dev = torch.device(device)
    site_paths = [d.site_path for d in dictionaries]
    assert len(set(site_paths)) == len(site_paths), f"duplicate site among {site_paths}"
    takes = {d.site_path: d.take for d in dictionaries}
    modules = {d.site_path: d.hook for d in dictionaries}
    assert (logits_fn is None) != (site_acts_fn is None), (
        "pass exactly one of logits_fn (capture the sites from one forward) or site_acts_fn "
        "(supply logits and per-site signals yourself)"
    )
    assert token_stats in ("off", "topk", "full"), (
        f"token_stats={token_stats!r}: expected 'off' | 'topk' | 'full'. This parameter was a "
        "bool; True/False no longer mean anything here."
    )

    harvester = PadMaskedLatentHarvester(
        layers=[(d.site_path, d.n_features) for d in dictionaries],
        vocab_size=vocab_size,
        max_examples_per_component=examples_per_component,
        context_tokens_per_side=context_tokens_per_side,
        max_examples_per_batch_per_component=max_examples_per_batch_per_component,
        collect_token_stats=token_stats == "full",
        device=dev,
    )

    model.eval()
    cached_batches: list[Tensor] = []
    for _ in tqdm.tqdm(range(n_batches), desc="Harvesting latents (pad-masked)"):
        batch = next(token_batches).to(dev)
        if token_stats == "topk":
            cached_batches.append(batch.detach().to("cpu"))
        if site_acts_fn is None:
            with OutputCapture(
                model,
                site_paths,
                detach=True,
                stop_when_complete=False,
                takes=takes,
                modules=modules,
            ) as cap:
                logits = logits_fn(batch)
            site_acts = {p: cap[p] for p in site_paths}
        else:
            logits, site_acts = site_acts_fn(batch)
            missing = [p for p in site_paths if p not in site_acts]
            assert not missing, f"site_acts_fn did not return {missing}"
        assert logits.shape[:2] == batch.shape and logits.shape[-1] == vocab_size, (
            f"forward returned logits {tuple(logits.shape)}, expected {(*batch.shape, vocab_size)}"
        )
        firings, activations = _latent_firings_and_acts(dictionaries, site_acts)
        output_probs = logits.float().softmax(dim=-1)
        real_mask = batch != pad_id
        harvester.process_batch_masked(batch, firings, activations, output_probs, real_mask)

    if token_stats == "topk":
        harvester.pmi_rankers = _accumulate_chunked_pmi(
            model=model,
            dictionaries=dictionaries,
            cached_batches=cached_batches,
            logits_fn=logits_fn,
            site_acts_fn=site_acts_fn,
            site_paths=site_paths,
            takes=takes,
            modules=modules,
            harvester=harvester,
            vocab_size=vocab_size,
            pad_id=pad_id,
            top_k=pmi_token_top_k,
            budget_bytes=int(pmi_memory_budget_gb * 1024**3),
            dev=dev,
        )

    config = HarvestConfig(
        method_config=ParamDecompHarvestConfig(wandb_path=_pd_run_id(harvest_id)),
        n_batches=n_batches,
        activation_context_tokens_per_side=context_tokens_per_side,
        activation_examples_per_component=examples_per_component,
        pmi_token_top_k=pmi_token_top_k if token_stats != "off" else 0,
    )
    out = out_dir if out_dir is not None else get_harvest_subrun_dir(harvest_id, "h-latents")
    _save_results(harvester, config, out, mode=token_stats)
    print(
        f"[harvest] {harvester.total_tokens_processed:,} real tokens -> {out / 'harvest.db'}",
        flush=True,
    )
    offset = 0
    for d in dictionaries:
        fired = float(harvester.firing_counts[offset : offset + d.n_features].sum())
        offset += d.n_features
        print(
            f"[harvest]   {d.site_path} ({d.take}): mean L0 "
            f"{fired / max(harvester.total_tokens_processed, 1):.1f}",
            flush=True,
        )
    return out
