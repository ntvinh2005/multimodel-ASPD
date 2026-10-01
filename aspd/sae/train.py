"""Train the SAE pair at the input and output sites of one decomposed matrix."""

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor, nn

from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig
from aspd.sae.sites import OutputCapture, SitePair, resolve_sites, site_widths

FEATURE_MULTIPLIER = 32


def reference_width(widths: Iterable[int]) -> int:
    """The side the pair is sized off: the SMALLER of the two, whichever side that is."""
    widths = list(widths)
    assert widths, "no site widths to size a dictionary off"
    return min(widths)


def build_sae_pair(
    model: nn.Module,
    sites: SitePair,
    probe: Tensor,
    *,
    feature_multiplier: int = FEATURE_MULTIPLIER,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    **cfg_overrides: object,
) -> dict[str, MatryoshkaBatchTopKSAE]:
    """One SAE per site, both `feature_multiplier * min(d_in, d_out)` wide."""
    resolve_sites(model, sites)
    widths = site_widths(model, sites, probe)
    d_ref = reference_width(widths.values())
    n_features = feature_multiplier * d_ref
    widest = max(widths, key=lambda site: widths[site])
    assert n_features >= widths[widest], (
        f"dictionary ({n_features}) narrower than {widest} ({widths[widest]}); an undercomplete "
        "SAE cannot represent its own input space. feature_multiplier must be at least "
        f"{-(-widths[widest] // d_ref)} (ceil of {widths[widest]}/{d_ref}), got {feature_multiplier}."
    )
    saes = {}
    for site in (sites.input_site, sites.output_site):
        sae = MatryoshkaBatchTopKSAE(
            MatryoshkaSAEConfig(d_in=widths[site], n_features=n_features, **cfg_overrides)  # type: ignore[arg-type]
        ).to(device=device, dtype=dtype)
        sae.threshold = sae.threshold.float()
        sae.n_batches_not_active = sae.n_batches_not_active.float()
        saes[site] = sae
    return saes


def train_sae_pair(
    model: nn.Module,
    sites: SitePair,
    token_batches: Iterator[Tensor],
    saes: dict[str, MatryoshkaBatchTopKSAE],
    *,
    n_tokens: int,
    sae_batch_tokens: int = 2048,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.99),
    device: str = "cuda",
    log_every: int = 2000,
    train_paths: list[str] | None = None,
) -> list[dict[str, float]]:
    """Joint Adam over both SAEs, driven by one forward of the frozen target per batch."""
    assert n_tokens > 0, (
        "train_sae_pair called with n_tokens=0: the loop would not execute and the caller would "
        "go on to evaluate and SAVE a randomly-initialized pair, whose sae_report.json every "
        "later run then trusts. A degenerate extractor yields finite, plausibly-trending PR "
        "curves -- exactly the silent corruption the gate exists to prevent."
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    paths = [sites.input_site, sites.output_site]
    trained = list(paths) if train_paths is None else list(train_paths)
    unknown = set(trained) - set(paths)
    assert not unknown, f"train_paths names sites not in this pair: {sorted(unknown)}"
    assert trained, "train_paths is empty: nothing would be trained and a pair would still be SAVED"
    for path in paths:
        if path not in trained:
            saes[path].freeze()

    params = [p for path in trained for p in saes[path].parameters()]
    opt = torch.optim.Adam(params, lr=lr, betas=betas)
    history: list[dict[str, float]] = []
    seen = 0
    step = 0
    dtype = saes[paths[0]].W_enc.dtype
    while seen < n_tokens:
        batch = next(token_batches).to(device)
        # Truncated: both sites bracket one matrix, so the rest of the target is dead work.
        with torch.no_grad(), OutputCapture(
            model,
            paths,
            detach=True,
            stop_when_complete=True,
            takes=sites.takes,
            modules=sites.capture_modules,
        ) as cap:
            cap.run(batch)
            acts = {path: cap[path].to(dtype).reshape(-1, cap[path].shape[-1]) for path in paths}

        n_tok = acts[paths[0]].shape[0]
        for lo in range(0, n_tok, sae_batch_tokens):
            hi = min(lo + sae_batch_tokens, n_tok)
            losses = {path: saes[path].loss(acts[path][lo:hi]) for path in trained}
            total = sum(term["loss"] for term in losses.values())
            opt.zero_grad(set_to_none=True)
            total.backward()
            for path in trained:
                saes[path].normalize_decoder_()
            opt.step()

            seen += hi - lo
            step += 1
            if step % log_every == 0:
                row = {"step": step, "tokens": seen}
                for path in trained:
                    tag = "in" if path == sites.input_site else "out"
                    row |= {
                        f"{tag}/fvu": losses[path]["fvu"].item(),
                        f"{tag}/l0": losses[path]["l0_norm"].item(),
                        f"{tag}/n_dead": losses[path]["n_dead"].item(),
                    }
                history.append(row)
                print(f"[sae] {row}", flush=True)

    return history


@torch.no_grad()
def evaluate_sae_pair(
    model: nn.Module,
    sites: SitePair,
    token_batches: Iterator[Tensor],
    saes: dict[str, MatryoshkaBatchTopKSAE],
    *,
    n_batches: int,
    device: str = "cuda",
) -> dict[str, dict[str, float]]:
    """Held-out FVU / dead-fraction / mean-L0, using the INFERENCE path."""
    model.eval()
    paths = [sites.input_site, sites.output_site]
    stats = {
        path: {"sq_err": 0.0, "sq_tot": 0.0, "l0": 0.0, "n": 0.0, "fired": torch.zeros(
            saes[path].cfg.n_features, device=device, dtype=torch.bool)}
        for path in paths
    }

    for _ in range(n_batches):
        batch = next(token_batches).to(device)
        with OutputCapture(
            model,
            paths,
            detach=True,
            stop_when_complete=True,
            takes=sites.takes,
            modules=sites.capture_modules,
        ) as cap:
            cap.run(batch)
            for path in paths:
                a = cap[path].to(saes[path].W_enc.dtype).reshape(-1, cap[path].shape[-1])
                f = saes[path].features(a)
                recon = saes[path].decode(f)
                s = stats[path]
                s["sq_err"] += (recon - a).float().pow(2).sum().item()
                s["sq_tot"] += (a.float() - a.float().mean(0)).pow(2).sum().item()
                s["l0"] += (f > 0).float().sum(-1).sum().item()
                s["n"] += a.shape[0]
                s["fired"] |= (f > 0).any(0)

    report = {}
    for path in paths:
        s = stats[path]
        n_features = saes[path].cfg.n_features
        report[path] = {
            "fvu": s["sq_err"] / max(s["sq_tot"], 1e-8),
            "mean_l0": s["l0"] / max(s["n"], 1.0),
            "dead_frac": 1.0 - (s["fired"].sum().item() / n_features),
            "n_features": float(n_features),
            "n_tokens": s["n"],
        }
    return report


SITES_STAMP = "sites.json"
"""Which activation each saved dictionary was trained on. See `save_sae_pair`."""


def save_sae_pair(
    saes: dict[str, MatryoshkaBatchTopKSAE],
    report: dict[str, dict[str, float]],
    out_dir: Path,
    sites: SitePair | None = None,
) -> Path:
    """Write the pair, its report, and -- when `sites` is given -- the site stamp."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if sites is not None:
        (out_dir / SITES_STAMP).write_text(json.dumps(asdict(sites), indent=2))
    for path, sae in saes.items():
        torch.save(
            {
                "cfg": asdict(sae.cfg),
                "dtype": str(sae.W_enc.dtype).removeprefix("torch."),
                "state_dict": sae.state_dict(),
            },
            out_dir / f"{path.replace('.', '_')}.pt",
        )
    (out_dir / "sae_report.json").write_text(json.dumps(report, indent=2))
    return out_dir


def assert_stamp_matches(sites: SitePair, sae_dir: Path) -> None:
    stamp_path = sae_dir / SITES_STAMP
    stamp = json.loads(stamp_path.read_text()) if stamp_path.exists() else {"input_take": "output"}
    got, want = stamp.get("input_take", "output"), sites.input_take
    assert got == want, (
        f"{sae_dir} holds dictionaries trained on the {got.upper()} of {sites.input_site}, but "
        f"this run asks for its {want.upper()}. Those are different activations under the same "
        f"module path and the same filename, so nothing else here would catch it: `d_in`, "
        f"`n_features` and the state dict all match. Point at the pair built for this site."
    )


def load_sae_pair(
    sites: SitePair, sae_dir: Path, device: str = "cuda"
) -> dict[str, MatryoshkaBatchTopKSAE]:
    """Load and FREEZE. There is no path that returns a trainable extractor."""
    assert_stamp_matches(sites, sae_dir)
    saes = {}
    for path in (sites.input_site, sites.output_site):
        blob = torch.load(sae_dir / f"{path.replace('.', '_')}.pt", map_location=device)
        cfg = MatryoshkaSAEConfig(**{**blob["cfg"], "group_fracs": tuple(blob["cfg"]["group_fracs"])})
        dtype = getattr(torch, blob.get("dtype", "float32"))
        sae = MatryoshkaBatchTopKSAE(cfg).to(device=device, dtype=dtype)
        sae.threshold = sae.threshold.float()
        sae.n_batches_not_active = sae.n_batches_not_active.float()
        sae.load_state_dict(blob["state_dict"])
        saes[path] = sae.freeze()
    return saes

