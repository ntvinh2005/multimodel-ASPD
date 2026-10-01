"""Named activation sites of the supported models: where each decomposed matrix reads and writes,
and the residual-stream site (resid-pre or resid-mid) ASPD's shared encoder reads for it.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from aspd.sae.config import SAERunConfig
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE
from aspd.sae.sites import (
    SitePair,
    attn_o_proj,
    gemma2_mlp_down_proj,
    gpt2_mlp_c_fc,
    iter_token_batches,
    llama_simple_mlp_c_fc,
)
from aspd.sae.train import (
    SITES_STAMP,
    build_sae_pair,
    evaluate_sae_pair,
    load_sae_pair,
    reference_width,
    save_sae_pair,
    train_sae_pair,
)
from aspd.sae.validate import (
    load_report,
    require_pair_matches_config,
    summarize,
    warn_if_unverifiable,
)
from torch import Tensor

SITE_BUILDERS = {
    "transformer.h.{layer}.mlp.c_fc": gpt2_mlp_c_fc,
    "h.{layer}.mlp.c_fc": llama_simple_mlp_c_fc,
    "model.layers.{layer}.mlp.down_proj": gemma2_mlp_down_proj,
    "model.layers.{layer}.self_attn.o_proj": attn_o_proj,
}


@dataclass(frozen=True)
class ResidSite:
    """A gate site path template, whether it runs AFTER the matrix it gates, and how to rebuild it."""

    site: str
    runs_after: bool
    add_from: str | None = None


RESID_SITES: dict[str, ResidSite] = {
    "transformer.h.{layer}.attn.c_attn": ResidSite("transformer.h.{layer}.ln_1", False),
    "transformer.h.{layer}.attn.c_attn.q_proj": ResidSite("transformer.h.{layer}.ln_1", False),
    "transformer.h.{layer}.attn.c_attn.k_proj": ResidSite("transformer.h.{layer}.ln_1", False),
    "transformer.h.{layer}.attn.c_attn.v_proj": ResidSite("transformer.h.{layer}.ln_1", False),
    "transformer.h.{layer}.attn.c_proj": ResidSite(
        "transformer.h.{layer}.ln_2", True, add_from="transformer.h.{layer}.ln_1"
    ),
    "transformer.h.{layer}.mlp.c_fc": ResidSite("transformer.h.{layer}.ln_2", False),
    "transformer.h.{layer}.mlp.c_proj": ResidSite("transformer.h.{layer}.ln_2", False),
    "h.{layer}.mlp.c_fc": ResidSite("h.{layer}.rms_2", False),
    "h.{layer}.mlp.c_proj": ResidSite("h.{layer}.rms_2", False),
    "model.layers.{layer}.self_attn.q_proj": ResidSite(
        "model.layers.{layer}.input_layernorm", False
    ),
    "model.layers.{layer}.self_attn.k_proj": ResidSite(
        "model.layers.{layer}.input_layernorm", False
    ),
    "model.layers.{layer}.self_attn.v_proj": ResidSite(
        "model.layers.{layer}.input_layernorm", False
    ),
    "model.layers.{layer}.self_attn.o_proj": ResidSite(
        "model.layers.{layer}.pre_feedforward_layernorm", True
    ),
    "model.layers.{layer}.mlp.gate_proj": ResidSite(
        "model.layers.{layer}.pre_feedforward_layernorm", False
    ),
    "model.layers.{layer}.mlp.up_proj": ResidSite(
        "model.layers.{layer}.pre_feedforward_layernorm", False
    ),
    "model.layers.{layer}.mlp.down_proj": ResidSite(
        "model.layers.{layer}.pre_feedforward_layernorm", False
    ),
}


ARCH_RESID_SITES: dict[str, dict[str, ResidSite]] = {
    "Gemma2ForCausalLM": {
        k: v for k, v in RESID_SITES.items() if k.startswith("model.layers.")
    },
    "Qwen3ForCausalLM": {
        "model.layers.{layer}.self_attn.q_proj": ResidSite(
            "model.layers.{layer}.input_layernorm", False
        ),
        "model.layers.{layer}.self_attn.k_proj": ResidSite(
            "model.layers.{layer}.input_layernorm", False
        ),
        "model.layers.{layer}.self_attn.v_proj": ResidSite(
            "model.layers.{layer}.input_layernorm", False
        ),
        "model.layers.{layer}.self_attn.o_proj": ResidSite(
            "model.layers.{layer}.post_attention_layernorm",
            True,
            add_from="model.layers.{layer}.input_layernorm",
        ),
        "model.layers.{layer}.mlp.gate_proj": ResidSite(
            "model.layers.{layer}.post_attention_layernorm", False
        ),
        "model.layers.{layer}.mlp.up_proj": ResidSite(
            "model.layers.{layer}.post_attention_layernorm", False
        ),
        "model.layers.{layer}.mlp.down_proj": ResidSite(
            "model.layers.{layer}.post_attention_layernorm", False
        ),
    },
}


def _match_template(module: str, template: str) -> str | None:
    """`module`'s layer index if it fits `template`, else None. The one place the split is done."""
    prefix, suffix = template.split("{layer}")
    if not (module.startswith(prefix) and module.endswith(suffix)):
        return None
    layer = module[len(prefix) : len(module) - len(suffix)]
    return layer if layer.isdigit() else None


def _fill(entry: ResidSite, layer: str) -> ResidSite:
    return ResidSite(
        entry.site.format(layer=layer),
        entry.runs_after,
        None if entry.add_from is None else entry.add_from.format(layer=layer),
    )


def resid_site_entry(module: str, arch: str | None = None) -> ResidSite:
    """`module`'s gate site with its layer filled in, and its ordering flag. Fails loudly."""
    overlay = ARCH_RESID_SITES.get(arch or "", {})
    for template, entry in overlay.items():
        layer = _match_template(module, template)
        if layer is not None:
            return _fill(entry, layer)
    for template, entry in RESID_SITES.items():
        layer = _match_template(module, template)
        if layer is not None:
            return _fill(entry, layer)
    raise AssertionError(
        f"no residual-stream site known for {module!r}"
        + (f" on {arch}" if arch else "")
        + ". A routed gate reads one of the block's two residual streams, which cannot be guessed "
        "from the matrix path -- which norm holds it, and whether that norm runs before or after "
        "this matrix, are facts about the architecture. Add an entry to `RESID_SITES`, or to "
        "`ARCH_RESID_SITES` if another architecture already spells that path differently."
    )


def resid_site_for_module(module: str, arch: str | None = None) -> str:
    """The module whose INPUT is the residual stream `module`'s gate reads."""
    return resid_site_entry(module, arch).site


def resid_site_runs_after(module: str, arch: str | None = None) -> bool:
    """Whether that site executes AFTER `module` -- true only for an attention output matrix."""
    return resid_site_entry(module, arch).runs_after


def resid_site_add_from(module: str, arch: str | None = None) -> str | None:
    """The site whose input, plus `module`'s own unedited output, rebuilds `module`'s gate site."""
    return resid_site_entry(module, arch).add_from


def group_modules_by_resid_site(modules: list[str], arch: str | None = None) -> dict[str, list[str]]:
    """`{gate site: sorted modules it gates}` over `modules`. The shared encoder's key set."""
    out: dict[str, list[str]] = {}
    for module in modules:
        out.setdefault(resid_site_for_module(module, arch), []).append(module)
    return {site: sorted(ms) for site, ms in sorted(out.items())}


def sites_for_module(module: str, input_take: str = "output") -> SitePair:
    """Resolve a decomposed module path to its `SitePair`, failing loudly on an unknown shape."""
    for template, builder in SITE_BUILDERS.items():
        prefix, suffix = template.split("{layer}")
        if module.startswith(prefix) and module.endswith(suffix):
            layer = module[len(prefix) : len(module) - len(suffix)]
            if layer.isdigit():
                pair = builder(int(layer))
                if pair.input_module is not None:
                    assert input_take == pair.input_take, (
                        f"{module}'s input site is a forward-PRE hook on the module itself, so its "
                        f"take is fixed at {pair.input_take!r}; got {input_take!r}. Set "
                        f"`input_take: {pair.input_take}` in the SAE config (it is what gets "
                        "stamped into the artifact dir), and do not edit it afterwards."
                    )
                    return pair
                if input_take == "output":
                    return pair
                return replace(pair, input_take="input")
    raise AssertionError(
        f"no site pair known for {module!r}. requires the extractor to sit ADJACENT to the "
        "decomposed matrix, so a new target needs an explicit SITE_BUILDERS entry -- the "
        "predecessor module cannot be guessed from the path."
    )


def stamped_input_take(sae_dir: Path, sae_cfg: SAERunConfig | None = None) -> str:
    """Which activation the pair in `sae_dir` reads: from its stamp, or from the config that will
    create it.
    """
    stamp_path = Path(sae_dir) / SITES_STAMP
    if not stamp_path.exists():
        return "output" if sae_cfg is None else sae_cfg.input_take
    take = json.loads(stamp_path.read_text()).get("input_take", "output")
    if sae_cfg is not None:
        assert take == sae_cfg.input_take, (
            f"{sae_dir} holds a pair trained on the {take.upper()} of its input site, but the "
            f"config asks for the {sae_cfg.input_take.upper()}. Editing `input_take` does not "
            "retrain an existing directory -- point `sae_dir` somewhere new."
        )
    return take


def sites_for_run(module: str, sae_dir: Path | str) -> SitePair:
    """`sites_for_module`, with the input site read off the artifact directory rather than assumed."""
    return sites_for_module(module, stamped_input_take(Path(sae_dir)))


def adopt_output_dictionary(
    saes: dict[str, MatryoshkaBatchTopKSAE], sites: SitePair, donor_dir: Path, device: str
) -> None:
    """Replace this pair's output half with the donor directory's, in place."""
    donor_sites = sites if sites.input_module is not None else replace(sites, input_take="output")
    donor = load_sae_pair(donor_sites, donor_dir, device=device)
    out = sites.output_site
    donated = donor[out]
    mine = saes[out]
    assert donated.cfg == mine.cfg, (
        f"donor {donor_dir}'s output dictionary is {donated.cfg}, this run builds {mine.cfg}. "
        "Adopting it would swap the dictionary's shape out from under the config that declared it."
    )
    saes[out] = donated  # already frozen by `load_sae_pair`


def loader_to_token_stream(loader) -> Iterator[Tensor]:
    """Infinite token-id stream from a param_decomp LM loader."""
    while True:
        yield from iter_token_batches(iter(loader))


def train_or_load(
    model,
    module: str,
    token_stream: Iterator[Tensor],
    sae_dir: Path,
    *,
    device: str,
    sae_cfg: SAERunConfig | None = None,
) -> dict[str, object]:
    """Returns `{"input": sae, "output": sae}`, keyed as `aspd.losses` expects."""
    sae_dir = Path(sae_dir)
    sites = sites_for_module(module, stamped_input_take(sae_dir, sae_cfg))

    if (sae_dir / "sae_report.json").exists():
        saes = load_sae_pair(sites, sae_dir, device=device)
        report = load_report(sae_dir)
        # Training short-circuits here, so an edited config would otherwise be a silent no-op.
        if sae_cfg is not None:
            require_pair_matches_config(
                sae_dir,
                sae_cfg,
                saes[sites.input_site],
                reference_width(s.cfg.d_in for s in saes.values()),
            )
        else:
            warn_if_unverifiable(sae_dir)
    else:
        assert sae_cfg is not None, (
            f"no SAE pair at {sae_dir} and no training config, so there is nothing to train "
            "from. Pretrain the pair first (`slurm/sae.sbatch` / `python -m aspd.sae.setup`). "
            "Refusing to fabricate an untrained extractor -- a degenerate SAE silently corrupts "
            "both new losses while still producing finite, plausible-looking PR curves."
        )
        dict_cfg, train_cfg = sae_cfg.dictionary, sae_cfg.train
        probe = next(token_stream)
        saes = build_sae_pair(
            model,
            sites,
            probe.to(device),
            device=device,
            dtype=train_cfg.torch_dtype,
            **dict_cfg.model_dump(exclude={"feature_multiplier"}),
            feature_multiplier=dict_cfg.feature_multiplier,
        )
        train_paths = None
        if sae_cfg.reuse_output_from is not None:
            adopt_output_dictionary(saes, sites, Path(sae_cfg.reuse_output_from), device)
            train_paths = [sites.input_site]
        train_sae_pair(
            model,
            sites,
            token_stream,
            saes,
            n_tokens=train_cfg.n_tokens,
            sae_batch_tokens=train_cfg.sae_batch_tokens,
            lr=train_cfg.lr,
            betas=train_cfg.betas,
            log_every=train_cfg.log_every,
            device=device,
            train_paths=train_paths,
        )
        report = evaluate_sae_pair(
            model, sites, token_stream, saes, n_batches=train_cfg.eval_batches, device=device
        )
        save_sae_pair(saes, report, sae_dir, sites=sites)
        (sae_dir / "sae_config.yaml").write_text(yaml.safe_dump(sae_cfg.model_dump(), sort_keys=False))
        saes = load_sae_pair(sites, sae_dir, device=device)  # reload to guarantee frozen

    print(f"[sae] extractor report:\n{summarize(report)}", flush=True)
    return {"input": saes[sites.input_site], "output": saes[sites.output_site]}


