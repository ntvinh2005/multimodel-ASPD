"""Fixed evaluation token batches drawn from a config's data split, or read from a stored stream."""

from pathlib import Path

import torch
from torch import Tensor

EVAL_TOKENS_FILE = "eval_tokens.pt"
"""The evaluation stream the paper's editing and matching read, stored beside each evaluation SAE.
The streamed draw depends on the `datasets` version; the stored one does not."""


def stored_stream(sae_dir: str | Path) -> Path | None:
    path = Path(sae_dir) / EVAL_TOKENS_FILE
    return path if path.exists() else None


def _stored_batches(cfg, path: Path, *, split: str, device: str, n_tokens: int,
                    batch_size: int) -> tuple[list[Tensor], int]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    meta, rows = blob["meta"], blob["tokens"].long()
    expected = {"dataset_name": cfg.data.dataset_name, "tokenizer_name": cfg.data.tokenizer_name,
                "split": split, "seed": cfg.pd.seed, "max_seq_len": cfg.data.max_seq_len}
    wrong = {k: (meta[k], v) for k, v in expected.items() if meta[k] != v}
    assert not wrong, f"{path} is a different stream (stored, asked): {wrong}"
    batches, seen = [], 0
    for start in range(0, rows.shape[0], batch_size):
        batch = rows[start:start + batch_size]
        if batch.shape[0] < batch_size:
            break
        batches.append(batch.to(device))
        seen += batch.numel()
        if seen >= n_tokens:
            return batches, seen
    raise AssertionError(f"{path} holds {rows.numel()} tokens; {n_tokens} asked at batch size {batch_size}")


def collect_batches(
    cfg,
    *,
    split: str,
    device: str,
    n_tokens: int,
    batch_size: int | None = None,
    stream: Path | None = None,
) -> tuple[list[Tensor], int]:
    """`(batches, tokens)` from `stream` if given, else from the run's own loader; drawn once."""
    batch_size = batch_size or (cfg.eval.batch_size if cfg.eval else 16)
    if stream is not None:
        print(f"[eval tokens] {stream}", flush=True)
        return _stored_batches(cfg, stream, split=split, device=device, n_tokens=n_tokens,
                               batch_size=batch_size)
    from param_decomp_lab.experiments.lm.run import build_lm_loader

    from aspd.loader_patch import install_bos_for_tokenizers_that_add_it

    install_bos_for_tokenizers_that_add_it()
    loader = build_lm_loader(
        cfg.target, cfg.data, split=split, device=device,
        batch_size=batch_size,
        seed=cfg.pd.seed,
    )
    batches, seen = [], 0
    for batch in loader:
        batches.append(batch)
        seen += batch.numel()
        if seen >= n_tokens:
            break
    assert batches, f"no batches from the {split} split"
    return batches, seen


@torch.no_grad()
def component_ce_kl(model, cfg, batches: list[Tensor], *, device: str) -> dict[str, float]:
    """Every `CEandKLLosses` key, over `batches`, from the run's own metric config."""
    from param_decomp.optimize import _build_metric_context
    from param_decomp.torch_helpers import bf16_autocast
    from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
    from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
    from param_decomp_lab.seed import set_seed

    entries = [m for m in cfg.eval.metrics if m.type == "CEandKLLosses"]
    assert len(entries) == 1, (
        f"expected exactly one CEandKLLosses entry in eval.metrics, found {len(entries)}. "
        "Spec reuses the run's own configuration; there is nothing to fall back to."
    )
    metric = EVAL_METRIC_CLASSES["CEandKLLosses"](entries[0])
    metric.bind(model=model, device=device)

    set_seed(cfg.pd.seed)
    weight_deltas = model.calc_weight_deltas()
    metric.reset()
    with bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for batch in batches:
            ctx = _build_metric_context(
                batch,
                step=0,
                is_eval=True,
                device=device,
                wrapped_model=model,
                component_model=model,
                config=cfg.pd,
                reconstruction_loss=recon_loss_kl,
                weight_deltas=weight_deltas,
            )
            metric.update(ctx)
    return {k: float(v) for k, v in metric.compute().items()}


@torch.no_grad()
def weight_delta_reference(
    model, cfg, batches: list[Tensor], *, device: str, fp32: bool = True
) -> dict[str, float]:
    import einops
    import torch.nn.functional as F
    from param_decomp.masks import make_mask_infos
    from param_decomp.optimize import _build_metric_context
    from param_decomp.torch_helpers import bf16_autocast
    from param_decomp_lab.batch_and_loss_fns import calc_kl_divergence_lm, recon_loss_kl
    from param_decomp_lab.seed import set_seed

    set_seed(cfg.pd.seed)
    weight_deltas = model.calc_weight_deltas()
    VARIANTS = ("ci_masked_wd", "unmasked_wd", "ci_masked_nodelta", "unmasked_nodelta")
    sums = {f"{p}_{v}": 0.0 for v in VARIANTS for p in ("kl", "ce")} | {"ce_target": 0.0}
    n_positions = 0

    with bf16_autocast(enabled=cfg.runtime.autocast_bf16 and not fp32):
        for batch in batches:
            ctx = _build_metric_context(
                batch, step=0, is_eval=True, device=device,
                wrapped_model=model, component_model=model, config=cfg.pd,
                reconstruction_loss=recon_loss_kl, weight_deltas=weight_deltas,
            )
            ci = ctx.ci.lower_leaky
            deltas = {
                k: (weight_deltas[k], torch.ones_like(v[..., 0])) for k, v in ci.items()
            }

            masked_batch = ctx.batch.clone()
            masked_batch[:, 0] = -100
            flat_labels = masked_batch.flatten()

            def ce(logits: Tensor, labels: Tensor = flat_labels) -> float:
                flat = einops.rearrange(logits, "b seq vocab -> (b seq) vocab")
                return F.cross_entropy(flat[:-1], labels[1:], ignore_index=-100).item()

            ones = {k: torch.ones_like(v) for k, v in ci.items()}
            for name, component_mask, wd in (
                ("ci_masked_wd", ci, deltas),
                ("unmasked_wd", ones, deltas),
                ("ci_masked_nodelta", ci, None),
                ("unmasked_nodelta", ones, None),
            ):
                logits = model(
                    ctx.batch,
                    mask_infos=make_mask_infos(component_mask, weight_deltas_and_masks=wd),
                )
                n = ctx.batch.shape[0] * ctx.batch.shape[1]
                sums[f"kl_{name}"] += calc_kl_divergence_lm(
                    pred=logits, target=ctx.target_out
                ).item() * n
                sums[f"ce_{name}"] += ce(logits) * n

            n = ctx.batch.shape[0] * ctx.batch.shape[1]
            sums["ce_target"] += ce(ctx.target_out) * n
            n_positions += n

    out = {k: v / n_positions for k, v in sums.items()}
    for variant in VARIANTS:
        out[f"ce_difference_{variant}"] = out[f"ce_{variant}"] - out["ce_target"]
    out["kl_residual_contribution"] = out["kl_ci_masked_nodelta"] - out["kl_ci_masked_wd"]
    out["reference_fp32"] = float(fp32)
    return out


@torch.no_grad()
def dictionary_splice_ce_kl(
    target_model,
    saes: dict,
    sites,
    cfg,
    batches: list[Tensor],
    *,
    pad_id: int,
    device: str,
) -> dict[str, float]:
    from dataclasses import asdict

    from param_decomp_lab.experiments.lm.run import make_run_batch

    from aspd.eval.dictionary import SAEDictionary
    from aspd.eval.dictionary_report import splice_ce_kl

    dictionary = SAEDictionary(saes["output"].eval(), sites.output_site, "out")

    run_batch = make_run_batch(cfg.target)
    stats = splice_ce_kl(
        target_model,
        dictionary,
        iter(batches),
        lambda batch: run_batch(target_model, batch),
        n_batches=len(batches),
        pad_id=pad_id,
        device=device,
    )
    return {k: v for k, v in asdict(stats).items() if isinstance(v, (int, float))}
