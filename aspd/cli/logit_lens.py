"""Logit-lens tokens for every latent of an SAE."""

import argparse
from pathlib import Path


def resolve_refs(model, sites):
    from aspd.eval.logit_lens import LogitLensRefs

    final_norm = None
    for path in ("ln_f", "transformer.ln_f", "model.norm"):
        try:
            final_norm = model.get_submodule(path)
            break
        except AttributeError:
            continue
    assert final_norm is not None, "no ln_f / transformer.ln_f / model.norm on this model"
    hidden_on_module_input = sites.capture_modules[sites.input_site] == sites.output_site
    unembed = model.get_submodule("lm_head").weight.t()
    return LogitLensRefs(
        final_norm=final_norm,
        unembed=unembed,
        mlp_module=model.get_submodule(sites.output_site.rsplit(".", 1)[0]),
        c_fc_module=model.get_submodule(sites.output_site),
        mlp_in_width=unembed.shape[0],
        residual_roles=frozenset({"out"} if hidden_on_module_input else {"in"}),
        hidden_take="input" if hidden_on_module_input else "output",
    )


def write_logit_lens(sae_config: str, out_json: Path, *, top_k: int = 5) -> Path:
    from param_decomp_lab.distributed import get_device
    from param_decomp_lab.experiments.lm.run import build_target
    from transformers import AutoTokenizer

    from aspd.config import LMInterpExperimentConfig
    from aspd.eval.dictionary import sae_dictionaries_from_pair
    from aspd.eval.logit_lens import write_latent_logit_lens_json
    from aspd.eval.tokens import decode_with_spaces
    from aspd.sae.config import SAERunConfig
    from aspd.sae.setup import sites_for_run
    from aspd.sae.train import load_sae_pair

    sae_cfg = SAERunConfig.from_file(sae_config)
    cfg = LMInterpExperimentConfig.from_file(sae_cfg.experiment_config)
    device = get_device()
    model = build_target(cfg.target).to(device).eval()
    sites = sites_for_run(cfg.pd.decomposition_targets[0].module_pattern, sae_cfg.sae_dir)
    saes = load_sae_pair(sites, Path(sae_cfg.sae_dir), device=device)
    dicts = list(sae_dictionaries_from_pair(saes, sites).values())
    refs = resolve_refs(model, sites)
    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)

    out = write_latent_logit_lens_json(dicts, refs, decode_with_spaces(tok), out_json, top_k=top_k)
    print(f"[logit-lens] top/bottom {top_k} for {sum(d.n_features for d in dicts)} latents -> {out}",
          flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sae_config", help="configs/sae/<target>.yaml")
    ap.add_argument("--out", required=True, help="logit_lens.json output path")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    write_logit_lens(args.sae_config, Path(args.out), top_k=args.top_k)


if __name__ == "__main__":
    main()
