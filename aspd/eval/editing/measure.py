"""Token sets, the unedited baseline, and the localization measures of an edit."""

from dataclasses import dataclass, field

import torch
from aspd.sae.matryoshka import MatryoshkaBatchTopKSAE
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from aspd.eval.adapters.capture import keep_token_mask


@dataclass
class TokenPlan:
    """The study's whole token budget as one `[n_seq, L]` block, plus the masks on it."""

    tokens: Int[Tensor, "n l"]
    keep: Bool[Tensor, "n l"]
    n_tokens_total: int
    """Positions surviving the pad/bos/eos mask -- the honest denominator for a density."""

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])

    def global_positions(self, n_tokens: int) -> Tensor:
        n_seq = min(int(self.tokens.shape[0]), -(-n_tokens // self.seq_len))
        keep = self.keep[:n_seq].reshape(-1)
        return keep.nonzero(as_tuple=True)[0]


def build_token_plan(batches: list[Tensor], tokenizer: object) -> TokenPlan:
    """Concatenate the drawn batches into one block. Sequence length must agree across them."""
    assert batches, "no batches"
    lengths = {int(b.shape[1]) for b in batches}
    assert len(lengths) == 1, f"batches have differing sequence lengths {sorted(lengths)}"
    tokens = torch.cat([b.detach().cpu() for b in batches], dim=0)
    keep = keep_token_mask(tokens, tokenizer)
    return TokenPlan(tokens=tokens, keep=keep, n_tokens_total=int(keep.sum()))


@dataclass
class MeasureGroup:
    """One set of positions, the features whose own delta is reported on it, and its baseline."""

    name: str
    positions: Int[Tensor, " p"]
    """Flat `seq * L + col` indices into the token plan, ASCENDING."""
    targets: Int[Tensor, " j"]
    y_base: Float[Tensor, "p d_out"]
    cache_rows: Int[Tensor, " p"] | None = None
    """Row of `y_base` holding each of `positions`, when `y_base` is a SHARED cache."""
    n_positions: int = field(init=False)

    def __post_init__(self) -> None:
        if self.cache_rows is None:
            assert self.y_base.shape[0] == self.positions.numel(), (
                self.y_base.shape, self.positions.shape
            )
        else:
            assert self.cache_rows.numel() == self.positions.numel(), (
                self.cache_rows.shape, self.positions.shape
            )
        assert bool((self.positions[1:] >= self.positions[:-1]).all()), (
            f"group {self.name}: positions must be ascending"
        )
        self.n_positions = int(self.positions.numel())

    def baseline(self, sel: Int[Tensor, " s"]) -> Float[Tensor, "s d_out"]:
        """The cached clean activation at `positions[sel]`."""
        if self.cache_rows is None:
            return self.y_base[sel]
        return self.y_base[self.cache_rows[sel]]


@torch.no_grad()
def site_activations(
    model: nn.Module,
    module_path: str,
    tokens: Int[Tensor, "b l"],
    device: torch.device | str,
) -> Float[Tensor, "b l d_out"]:
    """The decomposed module's output for one chunk, by forward hook on the real model."""
    captured: dict[str, Tensor] = {}

    def hook(_module, _args, output):
        captured["y"] = output.detach()

    handle = model.get_submodule(module_path).register_forward_hook(hook)
    try:
        model(tokens.to(device))
    finally:
        handle.remove()
    y = captured["y"]
    assert y.shape[:2] == tokens.shape, (y.shape, tokens.shape)
    return y


@torch.no_grad()
def feature_support(
    model: nn.Module,
    module_path: str,
    sae: MatryoshkaBatchTopKSAE,
    plan: TokenPlan,
    *,
    batch_size: int,
    device: torch.device | str,
) -> Tensor:
    """`|A_j|` for EVERY latent over the whole plan -- the support half of the sample filter."""
    n_seq = int(plan.tokens.shape[0])
    support = torch.zeros(sae.cfg.n_features, dtype=torch.int64, device=device)
    for start in range(0, n_seq, batch_size):
        chunk = plan.tokens[start : start + batch_size]
        keep = plan.keep[start : start + batch_size].to(device)
        y = site_activations(model, module_path, chunk, device)
        features = sae.features(y.float()) * keep[:, :, None]
        support += (features > 0).sum(dim=(0, 1))
    return support.cpu()


@torch.no_grad()
def collect_positions(
    model: nn.Module,
    module_path: str,
    sae: MatryoshkaBatchTopKSAE,
    plan: TokenPlan,
    feature_ids: Int[Tensor, " j"],
    *,
    batch_size: int,
    device: torch.device | str,
    cache_device: torch.device | str,
) -> tuple[dict[int, Tensor], Tensor]:
    """`({j: A_j as flat positions}, y_base at the union of every position measured)`."""
    n_seq, seq_len = plan.tokens.shape
    enc = sae.W_enc.detach()[:, feature_ids.to(device)].float()
    per_feature: dict[int, list[Tensor]] = {int(j): [] for j in feature_ids}
    for start in range(0, n_seq, batch_size):
        chunk = plan.tokens[start : start + batch_size]
        keep = plan.keep[start : start + batch_size].to(device)
        y = site_activations(model, module_path, chunk, device).float()
        pre = torch.relu((y - sae.b_dec.float()) @ enc)
        active = (pre > sae.threshold) & keep[:, :, None]
        base = start * seq_len
        flat = active.reshape(-1, feature_ids.numel())
        for i, j in enumerate(feature_ids.tolist()):
            hits = flat[:, i].nonzero(as_tuple=True)[0]
            if hits.numel():
                per_feature[j].append(hits.cpu() + base)
    positions = {
        j: (torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long))
        for j, parts in per_feature.items()
    }
    union = torch.unique(torch.cat([p for p in positions.values() if p.numel()]))
    y_union = gather_site_activations(
        model, module_path, plan, union,
        batch_size=batch_size, device=device, cache_device=cache_device,
    )
    return positions, y_union


@torch.no_grad()
def gather_site_activations(
    model: nn.Module,
    module_path: str,
    plan: TokenPlan,
    positions: Int[Tensor, " p"],
    *,
    batch_size: int,
    device: torch.device | str,
    cache_device: torch.device | str,
) -> Float[Tensor, "p d_out"]:
    """Site activation at `positions` (sorted), from forwards over only the sequences they touch."""
    seq_len = plan.seq_len
    positions = positions.sort().values
    seqs = torch.unique(positions // seq_len)
    out: Tensor | None = None
    for start in range(0, seqs.numel(), batch_size):
        chunk_seqs = seqs[start : start + batch_size]
        y = site_activations(model, module_path, plan.tokens[chunk_seqs], device).float()
        rows, sel = _rows_in_chunk(positions, chunk_seqs, seq_len)
        if not sel.numel():
            continue
        picked = y.reshape(-1, y.shape[-1])[rows].to(cache_device)
        if out is None:
            out = torch.zeros(positions.numel(), picked.shape[-1],
                              dtype=picked.dtype, device=cache_device)
        out[sel] = picked
    assert out is not None, "no positions to gather"
    return out


def _rows_in_chunk(
    positions: Int[Tensor, " p"], chunk_seqs: Int[Tensor, " s"], seq_len: int
) -> tuple[Tensor, Tensor]:
    """`(row index inside the chunk's flattened [s*L, d], index into `positions`)`."""
    seq = positions // seq_len
    col = positions % seq_len
    slot = torch.searchsorted(chunk_seqs, seq)
    slot = slot.clamp_max(chunk_seqs.numel() - 1)
    present = chunk_seqs[slot] == seq
    sel = present.nonzero(as_tuple=True)[0]
    return slot[sel] * seq_len + col[sel], sel


@dataclass
class DeltaAccumulator:
    """Every reduction of `Df = f(edited) - f(baseline)` this eval reports, streamed over chunks."""

    n_latents: int
    targets: Int[Tensor, " j"]
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        n_t = self.targets.numel()
        zeros = lambda n: torch.zeros(n, dtype=torch.float64, device=self.device)
        self.abs_sum = zeros(self.n_latents)
        self.l0_sum = zeros(1)
        self.n_positions = 0
        self.t_abs = zeros(n_t)
        self.t_signed = zeros(n_t)
        self.t_abs_preact = zeros(n_t)
        self.t_base = zeros(n_t)
        self.t_base_active = zeros(n_t)
        self.t_edit_active = zeros(n_t)
        self.t_killed = zeros(n_t)
        self.t_born = zeros(n_t)
        self.t_moved = zeros(n_t)

    def add(
        self,
        f_base: Float[Tensor, "p f"],
        f_edit: Float[Tensor, "p f"],
        pre_base: Float[Tensor, "p j"],
        pre_edit: Float[Tensor, "p j"],
    ) -> None:
        delta = f_edit - f_base
        self.abs_sum += delta.abs().sum(dim=0).double()
        self.l0_sum += (delta != 0).sum().double()
        self.n_positions += int(f_base.shape[0])

        base_t, edit_t = f_base[:, self.targets], f_edit[:, self.targets]
        d_t = edit_t - base_t
        self.t_abs += d_t.abs().sum(dim=0).double()
        self.t_signed += d_t.sum(dim=0).double()
        self.t_moved += (d_t != 0).sum(dim=0).double()
        self.t_abs_preact += (pre_edit - pre_base).abs().sum(dim=0).double()
        self.t_base += base_t.sum(dim=0).double()
        base_on, edit_on = base_t > 0, edit_t > 0
        self.t_base_active += base_on.sum(dim=0).double()
        self.t_edit_active += edit_on.sum(dim=0).double()
        self.t_killed += (base_on & ~edit_on).sum(dim=0).double()
        self.t_born += (~base_on & edit_on).sum(dim=0).double()

    def _collateral(self, exclude: list[int]) -> tuple[float, float]:
        other = self.abs_sum.clone()
        other[torch.tensor(exclude, dtype=torch.long, device=other.device)] = 0.0
        return float(other.sum()), float((other**2).sum())

    def summarize(
        self, target_index: int, feature_id: int, *, exclude: list[int] | None = None
    ) -> dict[str, float]:
        return self._summarize(
            [target_index], exclude if exclude is not None else [feature_id]
        )

    def summarize_group(
        self, target_indices: list[int], exclude: list[int]
    ) -> dict[str, float]:
        return self._summarize(target_indices, exclude)

    def _summarize(self, target_indices: list[int], exclude: list[int]) -> dict[str, float]:
        n = max(self.n_positions, 1)
        idx = torch.tensor(target_indices, dtype=torch.long, device=self.t_abs.device)
        take = lambda v: float(v[idx].sum())
        own_abs = take(self.t_abs)
        own_base = take(self.t_base)
        total_other, sq = self._collateral(exclude)
        base_active = take(self.t_base_active)
        base_off = len(target_indices) * n - base_active
        return {
            "delta_abs": own_abs / n,
            "delta_signed": take(self.t_signed) / n,
            "delta_abs_preact": take(self.t_abs_preact) / n,
            "baseline_act": own_base / n,
            "delta_relative": own_abs / max(own_base, 1e-12),
            "frac_active_baseline": base_active / (len(target_indices) * n),
            "frac_active_edited": take(self.t_edit_active) / (len(target_indices) * n),
            "death_rate": take(self.t_killed) / max(base_active, 1.0),
            "birth_rate": take(self.t_born) / max(base_off, 1.0),
            "collateral_abs": total_other / n,
            "collateral_l0": (float(self.l0_sum) - take(self.t_moved)) / n,
            "collateral_pr": (total_other**2 / sq) if sq > 0 else 0.0,
            "localization": own_abs / max(own_abs + total_other, 1e-12),
            "selectivity": own_abs / max(total_other, 1e-12),
            "n_positions": float(n),
        }


@torch.no_grad()
def collect_module_inputs(
    model: nn.Module,
    module_path: str,
    plan: TokenPlan,
    *,
    batch_size: int,
    device: torch.device | str,
    cache_device: torch.device | str,
) -> Float[Tensor, "n l d_in"]:
    """The decomposed module's INPUT for every sequence in the plan, by forward-PRE hook."""
    captured: dict[str, Tensor] = {}

    def hook(_module, args):
        captured["x"] = args[0].detach()

    module = model.get_submodule(module_path)
    handle = module.register_forward_pre_hook(hook)
    chunks = []
    try:
        for start in range(0, plan.tokens.shape[0], batch_size):
            tokens = plan.tokens[start : start + batch_size]
            model(tokens.to(device))
            x = captured["x"]
            assert x.shape[:2] == tokens.shape, (x.shape, tokens.shape)
            chunks.append(x.to(cache_device))
    finally:
        handle.remove()
    return torch.cat(chunks)


@torch.no_grad()
def measure_edit(
    model: nn.Module,
    module_path: str,
    sae: MatryoshkaBatchTopKSAE,
    plan: TokenPlan,
    groups: list[MeasureGroup],
    delta_w: Float[Tensor, "d_out d_in"] | None,
    *,
    batch_size: int,
    device: torch.device | str,
    x_cache: Float[Tensor, "n l d_in"] | None = None,
) -> dict[str, DeltaAccumulator]:
    """Run every group under one patched weight and return their accumulators."""
    from contextlib import nullcontext

    from aspd.eval.editing.edit import patched_target_weight

    seq_len = plan.seq_len
    seqs = torch.unique(torch.cat([g.positions // seq_len for g in groups]))
    accs = {
        g.name: DeltaAccumulator(sae.cfg.n_features, g.targets.to(device), device=device)
        for g in groups
    }
    patch = (
        nullcontext() if delta_w is None
        else patched_target_weight(model, module_path, delta_w)
    )
    with patch:
        for start in range(0, seqs.numel(), batch_size):
            chunk_seqs = seqs[start : start + batch_size]
            if x_cache is None:
                y = site_activations(model, module_path, plan.tokens[chunk_seqs], device).float()
            else:
                x = x_cache[chunk_seqs.to(x_cache.device)].to(device, non_blocking=True)
                y = model.get_submodule(module_path)(x).float()
                assert y.shape[:2] == (chunk_seqs.numel(), seq_len), (y.shape, chunk_seqs.shape)
            flat = y.reshape(-1, y.shape[-1])
            for group in groups:
                rows, sel = _rows_in_chunk(group.positions, chunk_seqs, seq_len)
                if not sel.numel():
                    continue
                y_edit = flat[rows]
                y_base = group.baseline(sel).to(device, non_blocking=True).float()
                targets = group.targets.to(device)
                f_base, f_edit, pre_base, pre_edit = encode_pair(sae, y_base, y_edit, targets)
                accs[group.name].add(f_base, f_edit, pre_base, pre_edit)
    for group in groups:
        got = accs[group.name].n_positions
        assert got == group.n_positions, (
            f"group {group.name} measured {got} of {group.n_positions} positions"
        )
    return accs


def assert_cached_forward_matches(
    model: nn.Module,
    module_path: str,
    sae: MatryoshkaBatchTopKSAE,
    plan: TokenPlan,
    groups: list[MeasureGroup],
    delta_w: Tensor,
    fast: dict[str, DeltaAccumulator],
    *,
    batch_size: int,
    device: torch.device | str,
    label: str,
) -> None:
    """Re-measure one cell through the MODEL and require every accumulator field to agree exactly."""
    slow = measure_edit(
        model, module_path, sae, plan, groups, delta_w,
        batch_size=batch_size, device=device, x_cache=None,
    )
    assert set(slow) == set(fast), (sorted(slow), sorted(fast))
    n_stats = 0
    for name in slow:
        a, b = slow[name], fast[name]
        stats = sorted(vars(a))
        assert stats == sorted(vars(b)) and stats, (stats, sorted(vars(b)))
        for stat in stats:
            x, y = getattr(a, stat), getattr(b, stat)
            same = bool(torch.equal(x, y)) if isinstance(x, Tensor) else x == y
            assert same, (
                f"cached-forward parity FAILED ({label}), group {name}, {stat}: hooked forward "
                f"and cached module disagree. The cached path is not the same arithmetic as the "
                f"live one -- rerun with cache_module_inputs=False"
            )
            n_stats += 1
    print(f"[attr_edit] cached-forward parity OK on {label} "
          f"({len(slow)} groups, {n_stats} statistics, exact)", flush=True)


def encode_pair(
    sae: MatryoshkaBatchTopKSAE,
    y_base: Float[Tensor, "p d"],
    y_edit: Float[Tensor, "p d"],
    targets: Int[Tensor, " j"],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """`(f_base, f_edit, raw_preact_base, raw_preact_edit)` for one position block."""
    f_base = sae.features(y_base)
    f_edit = sae.features(y_edit)
    enc = sae.W_enc.detach()[:, targets]
    pre_base = (y_base - sae.b_dec) @ enc
    pre_edit = (y_edit - sae.b_dec) @ enc
    return f_base, f_edit, pre_base, pre_edit
