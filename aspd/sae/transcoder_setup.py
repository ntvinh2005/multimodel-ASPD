"""Train a frozen transcoder for one decomposed matrix."""

import argparse
import json
from pathlib import Path

import yaml
from aspd.sae.config import TranscoderRunConfig
from aspd.sae.sites import SitePair
from aspd.sae.transcoder import (
    MatryoshkaBatchTopKTranscoder,
    build_transcoder,
    checkpoint_name,
    evaluate_transcoder,
    load_transcoder,
    save_transcoder,
    stamped_input_take,
    train_transcoder,
    transcoder_steps,
)
from torch import Tensor, nn
from transformers.pytorch_utils import Conv1D as RadfordConv1D

REPORT = "transcoder_report.json"
CONFIG_STAMP = "transcoder_config.yaml"

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # the repository root


def resolve_project_path(path: str) -> Path:
    """A config path is written relative to the PROJECT, not to the process's cwd."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    from_project = PROJECT_ROOT / candidate
    assert from_project.exists(), (
        f"{path!r} exists neither relative to the cwd ({Path.cwd()}) nor to the project "
        f"({PROJECT_ROOT}). Transcoder configs name their experiment config project-relative."
    )
    return from_project


def applied_weight(model: nn.Module, module_path: str) -> Tensor:
    """The decomposed matrix in the `x @ W` convention, i.e. `[d_in, d_out]`."""
    module = model.get_submodule(module_path)
    match module:
        case RadfordConv1D():
            return module.weight.detach()
        case nn.Linear():
            return module.weight.detach().t()
        case _:
            raise AssertionError(
                f"{module_path} is a {type(module).__name__}; only nn.Linear and Radford Conv1D "
                "have a weight this comparison knows how to orient"
            )


def implied_weight_residual(
    tc: MatryoshkaBatchTopKTranscoder, model: nn.Module, module_path: str
) -> dict[str, float]:
    """`||W_enc W_dec - W||_F / ||W||_F` -- VPD's faithfulness residual, for a transcoder."""
    implied = tc.implied_weight()
    target = applied_weight(model, module_path).float()
    assert implied.shape == target.shape, (implied.shape, target.shape)
    return {
        "implied_weight_rel_residual": float(
            (implied - target).norm() / target.norm().clamp_min(1e-12)
        ),
        "implied_weight_rel_norm": float(implied.norm() / target.norm().clamp_min(1e-12)),
    }


def train_or_load(
    model: nn.Module,
    module: str,
    token_stream,
    tc_dir: Path,
    *,
    device: str,
    tc_cfg: TranscoderRunConfig | None = None,
) -> MatryoshkaBatchTopKTranscoder:
    """Frozen transcoder from `tc_dir`, training it first when a config is supplied."""
    from aspd.sae.setup import sites_for_module

    tc_dir = Path(tc_dir)
    if (tc_dir / REPORT).exists():
        assert tc_cfg is None or tc_cfg.input_take == stamped_input_take(tc_dir), (
            f"{tc_dir} holds a transcoder whose encoder was fitted to the "
            f"{stamped_input_take(tc_dir).upper()} of its input site, but the config asks for the "
            f"{tc_cfg.input_take.upper()}. Editing `input_take` does not retrain an existing "  # type: ignore[union-attr]
            "directory -- point `transcoder_dir` somewhere new."
        )
        tc = load_transcoder(tc_dir, device=device)
        print(f"[transcoder] loaded {tc_dir}:\n{json.dumps(json.loads((tc_dir / REPORT).read_text()), indent=2)}", flush=True)
        return tc

    assert tc_cfg is not None, (
        f"no transcoder at {tc_dir} and no training config, so there is nothing to train from. "
        "Pretrain it first (`slurm/transcoder.sbatch` / `python -m aspd.sae.transcoder_setup`)."
    )
    sites: SitePair = sites_for_module(module, tc_cfg.input_take)
    dict_cfg, train_cfg = tc_cfg.dictionary, tc_cfg.train
    probe = next(token_stream)
    tc = build_transcoder(
        model,
        sites,
        probe.to(device),
        device=device,
        dtype=train_cfg.torch_dtype,
        **dict_cfg.model_dump(exclude={"feature_multiplier"}),
        feature_multiplier=dict_cfg.feature_multiplier,
    )

    done_tokens = 0
    if any(tc_dir.glob("model_*.pt")):
        last = transcoder_steps(tc_dir)[-1]
        tc = load_transcoder(tc_dir, last, device=device).unfreeze()
        done_tokens = last * train_cfg.tokens_per_vpd_step
        print(
            f"[transcoder] RESUMING {tc_dir} from {checkpoint_name(last)} "
            f"({done_tokens / 1e9:.3f}B of {train_cfg.n_tokens / 1e9:.3f}B tokens done). "
            "Adam's moments and the loader position are NOT restored; the threshold EMA and the "
            "dead-latent counter are, since they ride in the checkpoint.",
            flush=True,
        )
        assert done_tokens < train_cfg.n_tokens, (
            f"{tc_dir} already has a checkpoint at the full budget but no {REPORT}; the previous "
            "attempt died between the last save and the evaluation. Delete the report-less "
            "directory or evaluate it by hand -- resuming would train zero tokens and then save."
        )
    print(
        f"[transcoder] {sites.input_site} (take={sites.input_take}, d={tc.cfg.d_in}) -> "
        f"{sites.output_site} (d={tc.d_out}), F={tc.cfg.n_features}, "
        f"{train_cfg.n_tokens / 1e9:.3f}B tokens, checkpoint every "
        f"{train_cfg.checkpoint_every_tokens / 1e6:.1f}M",
        flush=True,
    )

    def on_checkpoint(tokens_seen: int) -> None:
        step = train_cfg.vpd_step_at(done_tokens + tokens_seen)
        path = save_transcoder(tc, tc_dir, step, sites=sites)
        print(
            f"[transcoder] checkpoint {path.name} at "
            f"{(done_tokens + tokens_seen) / 1e9:.3f}B tokens",
            flush=True,
        )

    history = train_transcoder(
        model,
        sites,
        token_stream,
        tc,
        n_tokens=train_cfg.n_tokens - done_tokens,
        sae_batch_tokens=train_cfg.sae_batch_tokens,
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        log_every=train_cfg.log_every,
        device=device,
        checkpoint_every_tokens=train_cfg.checkpoint_every_tokens,
        on_checkpoint=on_checkpoint,
    )
    save_transcoder(tc, tc_dir, train_cfg.vpd_step_at(train_cfg.n_tokens), sites=sites)

    report = evaluate_transcoder(
        model, sites, token_stream, tc, n_batches=train_cfg.eval_batches, device=device
    )
    if sites.input_take == "output":
        report |= implied_weight_residual(tc, model, module)
    report["history"] = history  # type: ignore[assignment]
    (tc_dir / REPORT).write_text(json.dumps(report, indent=2))
    (tc_dir / CONFIG_STAMP).write_text(yaml.safe_dump(tc_cfg.model_dump(), sort_keys=False))
    (tc_dir / "experiment_config.yaml").write_text(
        resolve_project_path(tc_cfg.experiment_config).read_text()
    )
    print(f"[transcoder] report:\n{json.dumps({k: v for k, v in report.items() if k != 'history'}, indent=2)}", flush=True)
    return load_transcoder(tc_dir, device=device)  # reload to guarantee frozen


def sites_for_transcoder(module: str, tc_dir: Path | str) -> SitePair:
    from aspd.sae.setup import sites_for_module

    return sites_for_module(module, stamped_input_take(Path(tc_dir)))


def main() -> None:
    from param_decomp_lab.distributed import get_device
    from param_decomp_lab.experiments.lm.run import build_lm_loader, build_target

    from aspd.config import LMInterpExperimentConfig
    from aspd.sae.setup import loader_to_token_stream

    ap = argparse.ArgumentParser()
    ap.add_argument("transcoder_config", help="configs/transcoder/<target>.yaml")
    args = ap.parse_args()

    tc_cfg = TranscoderRunConfig.from_file(str(resolve_project_path(args.transcoder_config)))
    cfg = LMInterpExperimentConfig.from_file(str(resolve_project_path(tc_cfg.experiment_config)))
    device = get_device()
    model = build_target(cfg.target).to(device)
    from aspd.loader_patch import install_bos_for_tokenizers_that_add_it

    install_bos_for_tokenizers_that_add_it()
    loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        seed=cfg.pd.seed,
    )
    train_or_load(
        model,
        cfg.pd.decomposition_targets[0].module_pattern,
        loader_to_token_stream(loader),
        Path(tc_cfg.transcoder_dir),
        device=device,
        tc_cfg=tc_cfg,
    )


if __name__ == "__main__":
    main()
