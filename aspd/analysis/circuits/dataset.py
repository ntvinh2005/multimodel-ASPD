"""Attribution graphs summed over a dataset."""

from collections.abc import Iterator

import torch
from torch import Tensor

from aspd.analysis.circuits.edges import contract_error, grad_times_act
from aspd.analysis.circuits.replacement import run_replacement
from aspd.analysis.circuits.storage import DatasetEdgeStore


def accumulate(
    model,
    batches: Iterator[Tensor],
    *,
    sampling: str,
    n_batches: int,
    target_layer: str,
    target_components: list[int] | None = None,
    error_nodes: bool = True,
) -> DatasetEdgeStore:
    """Edges into every alive component of `target_layer`, summed over `n_batches`."""
    store: DatasetEdgeStore | None = None

    for _ in range(n_batches):
        tokens = next(batches)
        cache = run_replacement(model, tokens, sampling=sampling, error_nodes=error_nodes)
        acts = cache.component_acts[target_layer]  # [B, S, C], the DETACHED leaf
        connected = cache.component_acts_pre[target_layer]

        idxs = target_components
        if idxs is None:
            idxs = torch.unique(cache.ci[target_layer].nonzero(as_tuple=False)[:, -1]).tolist()

        if store is None:
            store = DatasetEdgeStore.empty(
                target_layer=target_layer,
                target_components=idxs,
                source_layers=sorted(cache.component_acts),
                error_layers=sorted(cache.errors),
                widths={k: v.shape[-1] for k, v in cache.component_acts.items()},
                device=acts.device,
            )

        summed = connected.sum(dim=(0, 1))       # [C]
        summed_abs = connected.abs().sum(dim=(0, 1))
        leaves: dict[str, Tensor] = {f"c::{k}": v for k, v in cache.component_acts.items()}
        leaves.update({f"e::{k}": v for k, v in cache.errors.items()})

        for t in idxs:
            for variant, tgt in (("signed", summed[t]), ("abs", summed_abs[t])):
                w = grad_times_act(tgt, leaves)
                store.add(t, variant, w, contract=contract_error)
        store.n_tokens += int(tokens.numel())
    assert store is not None, "n_batches must be >= 1"
    return store
