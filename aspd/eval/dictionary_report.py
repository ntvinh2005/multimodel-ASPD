"""Quality report for trained SAEs: FVU, L0 and dead fraction on held-out tokens."""

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from aspd.eval.dictionary import DictionaryAdapter
from aspd.sae.sites import OutputCapture


@dataclass(frozen=True)
class PrefixStats:
    group: int
    lo: int
    hi: int
    active_share: float  # fraction of all active-latent firings that land in this group
    dead_frac: float  # fraction of this group's latents that never fired


@dataclass(frozen=True)
class ReconStats:
    site: str
    role: str
    n_features: int
    n_tokens: int
    fvu: float
    mean_l0: float
    dead_frac: float
    per_prefix: list[PrefixStats]


@dataclass(frozen=True)
class SpliceStats:
    site: str
    role: str
    n_tokens: int
    ce_clean: float
    ce_spliced: float
    kl_spliced_vs_clean: float


@torch.no_grad()
def reconstruction_stats(
    model: nn.Module,
    dictionaries: list[DictionaryAdapter],
    token_batches: Iterator[Tensor],
    *,
    n_batches: int,
    device: str = "cuda",
) -> list[ReconStats]:
    """Activation-space FVU / L0 / dead-frac + per-prefix active-share, over `n_batches`."""
    dev = torch.device(device)
    read_key = {id(d): f"{i}:read" for i, d in enumerate(dictionaries)}
    write_key = {
        id(d): (f"{i}:write" if d.is_cross_site else f"{i}:read")
        for i, d in enumerate(dictionaries)
    }
    takes: dict[str, Literal["input", "output"]] = {}
    modules: dict[str, str] = {}
    for d in dictionaries:
        takes[read_key[id(d)]], modules[read_key[id(d)]] = d.take, d.hook
        takes[write_key[id(d)]], modules[write_key[id(d)]] = d.write_take, d.write_hook
    site_paths = list(dict.fromkeys([*read_key.values(), *write_key.values()]))
    acc = {
        id(d): {
            "sq_err": 0.0,
            "sq_tot": 0.0,
            "l0": 0.0,
            "n": 0.0,
            "fired": torch.zeros(d.n_features, device=dev, dtype=torch.bool),
            "fire_counts": torch.zeros(d.n_features, device=dev),
        }
        for d in dictionaries
    }
    model.eval()
    for _ in range(n_batches):
        batch = next(token_batches).to(dev)
        with OutputCapture(
            model,
            site_paths,
            detach=True,
            stop_when_complete=True,
            takes=takes,
            modules=modules,
        ) as cap:
            cap.run(batch)
        for d in dictionaries:
            read = cap[read_key[id(d)]]
            target = cap[write_key[id(d)]]
            a = read.reshape(-1, read.shape[-1])
            y = target.reshape(-1, target.shape[-1])
            feats = d.encode(a)
            recon = d.decode(feats)
            s = acc[id(d)]
            s["sq_err"] += (recon - y).float().pow(2).sum().item()
            s["sq_tot"] += (y.float() - y.float().mean(0)).pow(2).sum().item()
            active = feats > 0
            s["l0"] += active.float().sum(-1).sum().item()
            s["n"] += a.shape[0]
            s["fired"] |= active.any(0)
            s["fire_counts"] += active.float().sum(0)

    out: list[ReconStats] = []
    for d in dictionaries:
        s = acc[id(d)]
        out.append(
            ReconStats(
                site=d.site_path,
                role=d.role,
                n_features=d.n_features,
                n_tokens=int(s["n"]),
                fvu=s["sq_err"] / max(s["sq_tot"], 1e-8),
                mean_l0=s["l0"] / max(s["n"], 1.0),
                dead_frac=1.0 - s["fired"].float().mean().item(),
                per_prefix=_prefix_stats(d, s["fire_counts"], s["fired"]),
            )
        )
    return out


def _prefix_stats(
    dictionary: DictionaryAdapter,
    fire_counts: Float[Tensor, " f"],
    fired: Int[Tensor, " f"],
) -> list[PrefixStats]:
    bounds = dictionary.group_boundaries()
    if bounds is None:
        return []
    total_fires = fire_counts.sum().clamp_min(1.0)
    stats: list[PrefixStats] = []
    for g in range(len(bounds) - 1):
        lo, hi = bounds[g], bounds[g + 1]
        group_fired = fired[lo:hi]
        stats.append(
            PrefixStats(
                group=g,
                lo=lo,
                hi=hi,
                active_share=(fire_counts[lo:hi].sum() / total_fires).item(),
                dead_frac=1.0 - group_fired.float().mean().item(),
            )
        )
    return stats


@torch.no_grad()
def splice_ce_kl(
    model: nn.Module,
    dictionary: DictionaryAdapter,
    token_batches: Iterator[Tensor],
    logits_fn: Callable[[Tensor], Tensor],
    *,
    n_batches: int,
    pad_id: int,
    device: str = "cuda",
) -> SpliceStats:
    """Clean vs spliced next-token CE and KL(spliced || clean), pad positions excluded."""
    dev = torch.device(device)
    ce_clean = ce_spliced = kl = n = 0.0
    model.eval()
    stash: dict[str, Tensor] = {}

    def read_pre_hook(_m: nn.Module, args: tuple) -> None:
        stash["acts"] = args[0]

    def read_hook(_m: nn.Module, _a: tuple, output: Tensor) -> None:
        stash["acts"] = output

    def reconstruction(like: Tensor) -> Tensor:
        acts = stash.get("acts")
        assert acts is not None, (
            f"{dictionary.site_path}: the write site {dictionary.write_hook} ran before the read "
            f"site {dictionary.hook} (or the read site did not run at all), so there is nothing "
            "to encode. A transcoder's encoder must be upstream of the module it reconstructs"
        )
        return dictionary.decode(dictionary.encode(acts)).to(like.dtype)

    def splice_hook(_m: nn.Module, _a: tuple, output: Tensor) -> Tensor:
        return reconstruction(output)

    def splice_pre_hook(_m: nn.Module, args: tuple) -> tuple:
        return (reconstruction(args[0]), *args[1:])

    source = model.get_submodule(dictionary.hook)
    target = model.get_submodule(dictionary.write_hook)

    for _ in range(n_batches):
        batch = next(token_batches).to(dev)
        clean = logits_fn(batch)
        stash.clear()
        handles = [
            source.register_forward_pre_hook(read_pre_hook)
            if dictionary.take == "input"
            else source.register_forward_hook(read_hook),
            target.register_forward_pre_hook(splice_pre_hook)
            if dictionary.write_take == "input"
            else target.register_forward_hook(splice_hook),
        ]
        try:
            spliced = logits_fn(batch)
        finally:
            for handle in handles:
                handle.remove()

        # Next-token targets: predict position t+1 from t; last position has no target.
        tgt = batch[:, 1:]
        real = (tgt != pad_id).reshape(-1)
        clean_lp = clean[:, :-1].reshape(-1, clean.shape[-1]).log_softmax(-1)[real]
        spl_lp = spliced[:, :-1].reshape(-1, spliced.shape[-1]).log_softmax(-1)[real]
        tgt_flat = tgt.reshape(-1)[real]

        ce_clean += -clean_lp.gather(-1, tgt_flat[:, None]).sum().item()
        ce_spliced += -spl_lp.gather(-1, tgt_flat[:, None]).sum().item()
        # KL(spliced || clean) = sum p_spliced * (logp_spliced - logp_clean).
        kl += (spl_lp.exp() * (spl_lp - clean_lp)).sum().item()
        n += real.sum().item()

    return SpliceStats(
        site=dictionary.site_path,
        role=dictionary.role,
        n_tokens=int(n),
        ce_clean=ce_clean / max(n, 1.0),
        ce_spliced=ce_spliced / max(n, 1.0),
        kl_spliced_vs_clean=kl / max(n, 1.0),
    )


def build_report(
    model: nn.Module,
    dictionaries: list[DictionaryAdapter],
    make_batches: Callable[[], Iterator[Tensor]],
    logits_fn: Callable[[Tensor], Tensor],
    *,
    n_recon_batches: int,
    n_splice_batches: int,
    pad_id: int,
    out_path: Path,
    device: str = "cuda",
) -> dict[str, object]:
    """Run every metric and write the JSON the app + humans read. `make_batches` yields a fresh
    iterator per call so reconstruction and each splice pass see comparable data.
    """
    recon = reconstruction_stats(
        model, dictionaries, make_batches(), n_batches=n_recon_batches, device=device
    )
    splice = [
        splice_ce_kl(
            model, d, make_batches(), logits_fn,
            n_batches=n_splice_batches, pad_id=pad_id, device=device,
        )
        for d in dictionaries
    ]
    report = {
        "reconstruction": [asdict(r) for r in recon],
        "splice": [asdict(s) for s in splice],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[report] -> {out_path}", flush=True)
    return report
