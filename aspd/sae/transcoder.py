"""Matryoshka BatchTopK transcoder trained on frozen activations (input site to output site)."""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor, nn

from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE, MatryoshkaSAEConfig
from aspd.sae.sites import OutputCapture, SitePair, resolve_sites, site_widths
from aspd.sae.train import SITES_STAMP, reference_width


@dataclass(frozen=True)
class MatryoshkaTranscoderConfig(MatryoshkaSAEConfig):
    """`MatryoshkaSAEConfig` plus the reconstruction width."""

    d_out: int | None = None

    @property
    def out_width(self) -> int:
        return self.d_in if self.d_out is None else self.d_out


class MatryoshkaBatchTopKTranscoder(MatryoshkaBatchTopKSAE):
    """Encoder on `x` (`d_in`), decoder into `y` (`d_out`). See the module docstring."""

    cfg: MatryoshkaTranscoderConfig

    def __init__(self, cfg: MatryoshkaTranscoderConfig):
        super().__init__(cfg)
        d_out = cfg.out_width
        self.W_dec = nn.Parameter(torch.empty(cfg.n_features, d_out))
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.W_dec)
            self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        self.b_out = nn.Parameter(torch.zeros(d_out))

    @property
    def d_out(self) -> int:
        return self.cfg.out_width

    @property
    def _decoder_bias(self) -> Tensor:
        """`b_out`, not `b_dec` -- the reconstruction lives in `d_out`. See the parent's property."""
        return self.b_out

    def reconstruct(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """`y_hat` through the INFERENCE path (`features`, the deterministic EMA threshold)."""
        return self.decode(self.features(x))

    def implied_weight(self) -> Float[Tensor, "d_in d_out"]:
        """`W_enc @ W_dec` -- the map the transcoder would be if every latent fired at its preact."""
        return self.W_enc.detach().float() @ self.W_dec.detach().float()

    # ---- training --------------------------------------------------------------------------

    def encode_and_loss(  # type: ignore[override]
        self, x: Float[Tensor, "... d_in"], y: Float[Tensor, "... d_out"]
    ) -> tuple[Float[Tensor, "n f"], dict[str, Tensor]]:
        """BatchTopK features of `x`, and the parent's loss taken against `y`."""
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        assert x.shape[0] == y.shape[0], (
            f"{x.shape[0]} input tokens against {y.shape[0]} output tokens -- the two sites must "
            "come from ONE forward, or the transcoder is fitted to a shifted pairing"
        )
        assert y.shape[-1] == self.d_out, (y.shape, self.d_out)
        acts = self.preacts(x)
        acts_topk = self._batch_topk(acts)
        self._update_threshold(acts_topk)
        return acts_topk, self._loss_from(y, acts, acts_topk)

    def loss(  # type: ignore[override]
        self, x: Float[Tensor, "... d_in"], y: Float[Tensor, "... d_out"]
    ) -> dict[str, Tensor]:
        return self.encode_and_loss(x, y)[1]


# ---- build / train / evaluate / persist --------------------------------------------------------


def build_transcoder(
    model: nn.Module,
    sites: SitePair,
    probe: Tensor,
    *,
    feature_multiplier: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    **cfg_overrides: object,
) -> MatryoshkaBatchTopKTranscoder:
    """Widths read off a REAL forward, never declared -- `build_sae_pair`'s rule, one dictionary."""
    resolve_sites(model, sites)
    widths = site_widths(model, sites, probe)
    d_ref = reference_width(widths.values())
    n_features = feature_multiplier * d_ref
    d_in, d_out = widths[sites.input_site], widths[sites.output_site]
    assert n_features >= max(d_in, d_out), (
        f"dictionary ({n_features}) narrower than a site ({d_in} -> {d_out}); an undercomplete "
        f"transcoder cannot span its own output space. feature_multiplier must be at least "
        f"{-(-max(d_in, d_out) // d_ref)}, got {feature_multiplier}"
    )
    tc = MatryoshkaBatchTopKTranscoder(
        MatryoshkaTranscoderConfig(d_in=d_in, d_out=d_out, n_features=n_features, **cfg_overrides)  # type: ignore[arg-type]
    ).to(device=device, dtype=dtype)
    tc.threshold = tc.threshold.float()
    tc.n_batches_not_active = tc.n_batches_not_active.float()
    return tc


def train_transcoder(
    model: nn.Module,
    sites: SitePair,
    token_batches: Iterator[Tensor],
    tc: MatryoshkaBatchTopKTranscoder,
    *,
    n_tokens: int,
    sae_batch_tokens: int = 2048,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.99),
    device: str = "cuda",
    log_every: int = 2000,
    checkpoint_every_tokens: int | None = None,
    on_checkpoint=None,
) -> list[dict[str, float]]:
    """`train_sae_pair`'s loop with one dictionary and two sites, plus a token-keyed checkpoint hook."""
    assert n_tokens > 0, (
        "train_transcoder called with n_tokens=0: the loop would not execute and the caller would "
        "go on to evaluate and SAVE a randomly-initialized transcoder whose report every later "
        "run then trusts"
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    paths = [sites.input_site, sites.output_site]
    opt = torch.optim.Adam(tc.parameters(), lr=lr, betas=betas)
    history: list[dict[str, float]] = []
    dtype = tc.W_enc.dtype
    seen = 0
    step = 0
    next_checkpoint = checkpoint_every_tokens

    while seen < n_tokens:
        batch = next(token_batches).to(device)
        with torch.no_grad(), OutputCapture(
            model,
            paths,
            detach=True,
            stop_when_complete=True,
            takes=sites.takes,
            modules=sites.capture_modules,
        ) as cap:
            cap.run(batch)
            acts = {p: cap[p].to(dtype).reshape(-1, cap[p].shape[-1]) for p in paths}

        n_tok = acts[paths[0]].shape[0]
        for lo in range(0, n_tok, sae_batch_tokens):
            hi = min(lo + sae_batch_tokens, n_tok)
            losses = tc.loss(acts[sites.input_site][lo:hi], acts[sites.output_site][lo:hi])
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            tc.normalize_decoder_()
            opt.step()

            seen += hi - lo
            step += 1
            if step % log_every == 0:
                row = {
                    "step": step,
                    "tokens": seen,
                    "fvu": losses["fvu"].item(),
                    "l0": losses["l0_norm"].item(),
                    "n_dead": losses["n_dead"].item(),
                }
                history.append(row)
                print(f"[transcoder] {row}", flush=True)

            if next_checkpoint is not None and seen >= next_checkpoint:
                if on_checkpoint is not None:
                    on_checkpoint(seen)
                assert checkpoint_every_tokens is not None
                next_checkpoint += checkpoint_every_tokens

    return history


@torch.no_grad()
def evaluate_transcoder(
    model: nn.Module,
    sites: SitePair,
    token_batches: Iterator[Tensor],
    tc: MatryoshkaBatchTopKTranscoder,
    *,
    n_batches: int,
    device: str = "cuda",
) -> dict[str, float]:
    """Held-out FVU / mean-L0 / dead-fraction through the INFERENCE path."""
    model.eval()
    paths = [sites.input_site, sites.output_site]
    sq_err = sq_tot = l0 = n = 0.0
    fired = torch.zeros(tc.cfg.n_features, device=device, dtype=torch.bool)

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
            x = cap[sites.input_site].to(tc.W_enc.dtype).reshape(-1, tc.cfg.d_in)
            y = cap[sites.output_site].to(tc.W_enc.dtype).reshape(-1, tc.d_out)
        f = tc.features(x)
        recon = tc.decode(f)
        sq_err += (recon - y).float().pow(2).sum().item()
        sq_tot += (y.float() - y.float().mean(0)).pow(2).sum().item()
        l0 += (f > 0).float().sum(-1).sum().item()
        n += x.shape[0]
        fired |= (f > 0).any(0)

    return {
        "fvu": sq_err / max(sq_tot, 1e-8),
        "mean_l0": l0 / max(n, 1.0),
        "dead_frac": 1.0 - (fired.sum().item() / tc.cfg.n_features),
        "n_features": float(tc.cfg.n_features),
        "d_in": float(tc.cfg.d_in),
        "d_out": float(tc.d_out),
        "n_tokens": n,
    }


def checkpoint_name(step: int) -> str:
    """`model_<step>.pt`, where `step` is the VPD-EQUIVALENT optimizer step, not the transcoder's."""
    return f"model_{step}.pt"


def save_transcoder(
    tc: MatryoshkaBatchTopKTranscoder,
    out_dir: Path,
    step: int,
    *,
    sites: SitePair | None = None,
) -> Path:
    """One checkpoint, plus the site stamp on first write."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if sites is not None:
        (out_dir / SITES_STAMP).write_text(json.dumps(asdict(sites), indent=2))
    path = out_dir / checkpoint_name(step)
    torch.save(
        {
            "cfg": asdict(tc.cfg),
            "step": step,
            "dtype": str(tc.W_enc.dtype).removeprefix("torch."),
            "state_dict": tc.state_dict(),
        },
        path,
    )
    return path


def transcoder_steps(tc_dir: Path) -> list[int]:
    """Every checkpoint step in `tc_dir`, ascending."""
    steps = sorted(int(p.stem.removeprefix("model_")) for p in Path(tc_dir).glob("model_*.pt"))
    assert steps, f"no model_<step>.pt checkpoints in {tc_dir}"
    return steps


def load_transcoder(
    tc_dir: Path, step: int | None = None, device: str = "cuda"
) -> MatryoshkaBatchTopKTranscoder:
    """Load and FREEZE. There is no path that returns a trainable transcoder."""
    tc_dir = Path(tc_dir)
    step = transcoder_steps(tc_dir)[-1] if step is None else step
    blob = torch.load(tc_dir / checkpoint_name(step), map_location=device)
    cfg = MatryoshkaTranscoderConfig(
        **{**blob["cfg"], "group_fracs": tuple(blob["cfg"]["group_fracs"])}
    )
    dtype = getattr(torch, blob.get("dtype", "float32"))
    tc = MatryoshkaBatchTopKTranscoder(cfg).to(device=device, dtype=dtype)
    tc.threshold = tc.threshold.float()
    tc.n_batches_not_active = tc.n_batches_not_active.float()
    tc.load_state_dict(blob["state_dict"])
    return tc.freeze()


def stamped_input_take(tc_dir: Path) -> str:
    """Which activation the transcoder in `tc_dir` encodes, from its own stamp."""
    stamp = Path(tc_dir) / SITES_STAMP
    assert stamp.exists(), (
        f"{tc_dir} has no {SITES_STAMP}, so which site its encoder was fitted to is unrecoverable. "
        "The residual-stream and adjacent transcoders have identical shapes and filenames."
    )
    return json.loads(stamp.read_text())["input_take"]
