"""On-disk storage of accumulated attribution sums."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

VARIANTS = ("signed", "abs")


@dataclass
class DatasetEdgeStore:
    target_layer: str
    target_components: list[int]
    source_layers: list[str]
    error_layers: list[str]
    # [n_targets, C_source] per (variant, source layer); [n_targets] per (variant, error layer).
    component: dict[tuple[str, str], Tensor] = field(default_factory=dict)
    error: dict[tuple[str, str], Tensor] = field(default_factory=dict)
    ci_sum: dict[str, Tensor] = field(default_factory=dict)
    n_tokens: int = 0

    @classmethod
    def empty(cls, *, target_layer, target_components, source_layers, error_layers, widths, device):
        n = len(target_components)
        return cls(
            target_layer=target_layer,
            target_components=list(target_components),
            source_layers=list(source_layers),
            error_layers=list(error_layers),
            component={
                (v, s): torch.zeros(n, widths[s], device=device)
                for v in VARIANTS for s in source_layers
            },
            error={
                (v, s): torch.zeros(n, device=device) for v in VARIANTS for s in error_layers
            },
            ci_sum={s: torch.zeros(widths[s], device=device) for s in source_layers},
        )

    def add(self, target_component: int, variant: str, weighted: dict[str, Tensor],
            *, contract: Callable[[Tensor], Tensor]) -> None:
        row = self.target_components.index(target_component)
        for key, w in weighted.items():
            kind, _, layer = key.partition("::")
            if kind == "c":
                self.component[(variant, layer)][row] += w.sum(dim=(0, 1))
            else:
                self.error[(variant, layer)][row] += contract(w).sum()

    def merge(self, other: "DatasetEdgeStore") -> None:
        """Rank merge. Raw sums, so this is addition -- the reason nothing is normalised yet."""
        assert self.target_layer == other.target_layer
        assert self.target_components == other.target_components
        for k, v in other.component.items():
            self.component[k] += v
        for k, v in other.error.items():
            self.error[k] += v
        for k, v in other.ci_sum.items():
            self.ci_sum[k] += v
        self.n_tokens += other.n_tokens

    def normalised(self, variant: str = "signed", eps: float = 1e-8):
        """`({source_layer: [n_targets, C]}, {error_layer: [n_targets]})`, query-time normalised."""
        comp = {
            s: self.component[(variant, s)] / (self.ci_sum[s].clamp(min=eps))
            for s in self.source_layers
        }
        err = {
            s: self.error[(variant, s)] / max(self.n_tokens, 1) for s in self.error_layers
        }
        return comp, err
