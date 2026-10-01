"""Patches to the lab's component harvest for large decompositions (skips per-component token statistics)."""

from pathlib import Path
from typing import Literal


KNOWN_ARTIFACTS = frozenset({"harvest.db", "component_correlations.pt", "token_stats.pt"})


def _save_results(harvester, config, output_dir: Path, *, token_stats: str, correlations: str):
    """`HarvestRepo.save_results` with each optional artifact skipped BEFORE its tensors are read."""
    from param_decomp_lab.harvest.db import HarvestDB
    from param_decomp_lab.harvest.storage import CorrelationStorage, TokenStatsStorage

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "harvest.db"
    db = HarvestDB(db_path)
    db.save_config(config)
    n_saved = db.save_components_iter(harvester.build_results(pmi_top_k_tokens=config.pmi_token_top_k))
    db.close()
    print(f"[harvest] saved {n_saved} components to {db_path}", flush=True)

    if correlations == "full":
        CorrelationStorage(
            component_keys=harvester.component_keys,
            count_i=harvester.firing_counts.long().cpu(),
            count_ij=harvester.cooccurrence_counts.long().cpu(),
            count_total=harvester.total_tokens_processed,
        ).save(output_dir / "component_correlations.pt")
    else:
        print(
            "[harvest] component_correlations.pt skipped -- the [C, C] co-firing matrix. Only the "
            "app's correlations view and graph_interp's co-firing context read it; graph_interp "
            "ASSERTS it is present, so pass correlations='full' before running pd-graph-interp.",
            flush=True,
        )

    if token_stats == "full":
        TokenStatsStorage(
            component_keys=harvester.component_keys,
            vocab_size=harvester.vocab_size,
            n_tokens=harvester.total_tokens_processed,
            input_counts=harvester.input_cooccurrence.cpu(),
            input_totals=harvester.input_marginals.float().cpu(),
            output_counts=harvester.output_cooccurrence.cpu(),
            output_totals=harvester.output_marginals.cpu(),
            firing_counts=harvester.firing_counts.cpu(),
        ).save(output_dir / "token_stats.pt")
    else:
        print(
            "[harvest] token_stats.pt skipped -- harvest.db carries the ranked top/bottom PMI "
            "tokens, so the intruder eval and every DB-backed view still work. AUTOINTERP DOES "
            "NOT: `run_interpret` asserts the sidecar is present before it dispatches on strategy "
            "(autointerp/interpret.py:116), and every branch after it -- `canon` included -- then "
            "asserts its own per-component stats. `pd-graph-interp` and the app's correlations "
            "view need it too. Re-harvest with token_stats='full' for those.",
            flush=True,
        )


def install_component_token_stats(
    mode: Literal["topk", "full"] = "topk",
    correlations: Literal["off", "full"] = "off",
) -> None:
    """Set what the lab's component harvest writes beside `harvest.db`. Idempotent."""
    from param_decomp_lab.harvest.repo import HarvestRepo

    def _patched(harvester, config, output_dir):
        _save_results(
            harvester, config, Path(output_dir), token_stats=mode, correlations=correlations
        )

    _patched._aspd_patched = True  # pyright: ignore[reportFunctionMemberAccess]
    HarvestRepo.save_results = staticmethod(_patched)  # pyright: ignore[reportAttributeAccessIssue]


def install_low_memory_token_stats() -> None:
    """Replace `Harvester._accumulate_token_stats` with an allocation-free equivalent. Idempotent."""
    import torch
    from einops import rearrange, reduce
    from param_decomp_lab.harvest.accumulator import Harvester

    if getattr(Harvester._accumulate_token_stats, "_aspd_low_mem", False):
        return

    def _accumulate_token_stats(self, tokens_flat, probs_flat, firing_flat) -> None:
        n_components = firing_flat.shape[1]
        # `expand`, not `repeat`: a 0-strided view does the same work as a [C, S] int64 copy.
        token_indices = tokens_flat.unsqueeze(0).expand(n_components, -1)
        self.input_cooccurrence.scatter_add_(
            dim=1,
            index=token_indices,
            src=rearrange(firing_flat, "S lc -> lc S").long().contiguous(),
        )
        self.input_marginals.scatter_add_(
            dim=0,
            index=tokens_flat,
            src=torch.ones(tokens_flat.shape[0], device=self.device, dtype=torch.long),
        )
        # `addmm_`, not `+= einsum(...)`: accumulates in place instead of materialising [C, vocab].
        self.output_cooccurrence.addmm_(firing_flat.t(), probs_flat)
        self.output_marginals += reduce(probs_flat, "S v -> v", "sum")

    _accumulate_token_stats._aspd_low_mem = True  # pyright: ignore[reportFunctionMemberAccess]
    Harvester._accumulate_token_stats = _accumulate_token_stats  # pyright: ignore[reportAttributeAccessIssue]


class _RowView:
    """`x[i]` returns one shared zero row; `.cpu()` returns self. Lets the stock `build_results`
    walk every component without a `[C, vocab]` tensor existing anywhere.
    """

    def __init__(self, row):
        self._row = row

    def __getitem__(self, _i):
        return self._row

    def cpu(self):
        return self


class _Skip:
    """Absorbs `+=` and returns itself. Used ONLY inside the two wrappers below, so an accumulation
    into a deliberately-absent tensor is a no-op there and a loud error everywhere else.
    """

    def __iadd__(self, _other):
        return self

    def __add__(self, _other):
        return self


class _Absent:
    """A tensor-shaped refusal. Any use raises with the reason, instead of a 0.71 TiB allocation."""

    def __init__(self, name: str, shape: tuple[int, ...], reason: str):
        self.name, self.shape, self.reason = name, shape, reason

    def _die(self, *_a, **_k):
        n = 1
        for d in self.shape:
            n *= d
        b = n * 4
        size = f"{b / 1024**4:.2f} TiB" if b >= 1024**4 else f"{b / 1024**3:.1f} GiB"
        raise RuntimeError(
            f"{self.name} was not allocated ({'x'.join(map(str, self.shape))} = {size} at fp32). "
            f"{self.reason}"
        )

    __getattr__ = lambda self, _n: self._die  # noqa: E731
    __iadd__ = __add__ = __setitem__ = __getitem__ = _die


def install_no_cooccurrence(*, correlations: bool = False, token_stats: bool = False) -> None:
    """Stop `Harvester.__init__` ALLOCATING the two quadratic accumulators. Idempotent."""
    import torch
    from param_decomp_lab.harvest.accumulator import Harvester

    if getattr(Harvester.__init__, "_aspd_no_cooc", False):
        return
    stock_init = Harvester.__init__

    def _init(self, *args, **kwargs):
        import inspect

        bound = inspect.signature(stock_init).bind(self, *args, **kwargs)
        bound.apply_defaults()
        n_components = sum(c for _, c in bound.arguments["layers"])
        vocab = int(bound.arguments["vocab_size"])

        skip: set[tuple[int, int]] = set()
        if not correlations:
            skip.add((n_components, n_components))
        if not token_stats:
            skip.add((n_components, vocab))

        stock_zeros = torch.zeros
        reasons = {
            (n_components, n_components): "Component correlations are off; pass correlations=True.",
            (n_components, vocab): "Dense token stats are off; pass token_stats=True.",
        }

        def _zeros(*shape, **kw):
            if len(shape) == 2 and isinstance(shape[0], int) and (shape[0], shape[1]) in skip:
                key = (shape[0], shape[1])
                name = "cooccurrence_counts" if key[0] == key[1] else "token cooccurrence"
                return _Absent(name, key, reasons[key])
            return stock_zeros(*shape, **kw)

        torch.zeros = _zeros
        try:
            stock_init(self, *args, **kwargs)
        finally:
            torch.zeros = stock_zeros

        if not token_stats:
            def _marginals_only(tokens_flat, probs_flat, firing_flat, _self=self):
                _self.input_marginals.scatter_add_(
                    dim=0,
                    index=tokens_flat,
                    src=torch.ones(
                        tokens_flat.shape[0], device=_self.device, dtype=torch.long
                    ),
                )
                _self.output_marginals += probs_flat.sum(dim=0)

            self._accumulate_token_stats = _marginals_only

    _init._aspd_no_cooc = True  # pyright: ignore[reportFunctionMemberAccess]
    Harvester.__init__ = _init  # pyright: ignore[reportAttributeAccessIssue]

    stock_process = Harvester.process_batch

    def _process_batch(self, *args, **kwargs):
        """Skip the `[C, C]` co-firing accumulation without touching anything else."""
        if not isinstance(self.cooccurrence_counts, _Absent):
            return stock_process(self, *args, **kwargs)

        import param_decomp_lab.harvest.accumulator as acc

        stock_einsum = acc.einsum
        cofiring = "S c1, S c2 -> c1 c2"

        def _einsum(*tensors_and_pattern):
            if tensors_and_pattern and tensors_and_pattern[-1] == cofiring:
                return _Skip()
            return stock_einsum(*tensors_and_pattern)

        saved, self.cooccurrence_counts = self.cooccurrence_counts, _Skip()
        acc.einsum = _einsum
        try:
            return stock_process(self, *args, **kwargs)
        finally:
            acc.einsum = stock_einsum
            self.cooccurrence_counts = saved

    stock_merge = Harvester.merge

    def _merge(self, other):
        pairs = [
            (n, getattr(self, n))
            for n in ("cooccurrence_counts", "input_cooccurrence", "output_cooccurrence")
            if isinstance(getattr(self, n), _Absent)
        ]
        for n, _ in pairs:
            setattr(self, n, _Skip())
            setattr(other, n, _Skip())
        try:
            return stock_merge(self, other)
        finally:
            for n, saved in pairs:
                setattr(self, n, saved)
                setattr(other, n, saved)

    stock_build = Harvester.build_results

    def _build_results(self, pmi_top_k_tokens: int):
        if not isinstance(self.input_cooccurrence, _Absent):
            yield from stock_build(self, pmi_top_k_tokens)
            return
        from param_decomp_lab.harvest.schemas import ComponentTokenPMI

        empty = ComponentTokenPMI(top=[], bottom=[])
        v = int(self.input_marginals.shape[0])
        c = int(self.firing_counts.shape[0])
        # Zero rows, materialised ONE component at a time: the whole point is never to hold [C, V].
        zeros_in = torch.zeros(v, dtype=torch.long)
        saved_in, saved_out = self.input_cooccurrence, self.output_cooccurrence
        self.input_cooccurrence = _RowView(zeros_in)
        self.output_cooccurrence = _RowView(torch.zeros(v))
        try:
            for comp in stock_build(self, pmi_top_k_tokens):
                yield comp.model_copy(update={"input_token_pmi": empty, "output_token_pmi": empty})
        finally:
            self.input_cooccurrence, self.output_cooccurrence = saved_in, saved_out
        del c

    Harvester.build_results = _build_results  # pyright: ignore[reportAttributeAccessIssue]
    Harvester.process_batch = _process_batch  # pyright: ignore[reportAttributeAccessIssue]
    Harvester.merge = _merge  # pyright: ignore[reportAttributeAccessIssue]
