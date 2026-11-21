"""Component importance scores (formulation, "Component Importance Score").

For model ``n``, matrix ``j`` and feature ``i``::

    sigma_j   = sqrt( E_alpha || y_{j,alpha} - E_alpha[y_{j,alpha}] ||_2^2 )
    beta_{j,i} = sqrt( E_alpha[ ||Delta y_{j,alpha,i}||^2 | g_{p,alpha,i} = 1 ] ) / sigma_j
    q_{j,i}   = sqrt( E_alpha[ ||Delta y_{j,alpha,i}||^2 ] )
    rho_{j,i} = q_{j,i} / sum_k q_{j,k}

Expectations are over the anchors fed to the accumulator (population moments).  Because
``Delta y = 0`` whenever ``g = 0``, the conditional numerator of ``beta`` is the sum of
squared contribution norms divided by the firing count.  A score whose denominator is
zero (feature never fired, ``sigma_j = 0`` or ``sum_k q = 0``) is reported as ``NaN`` and
is never eligible for screening.

``MatrixImportanceAccumulator`` streams over pools in float64 so that chunked and
single-pass updates agree; ``ImportanceTable`` holds the finalized scores of every
matrix and round-trips through ``safetensors`` + JSON.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from aspd.multimodel.posthoc.contributions import component_coefficients, contribution_sq_norms
from aspd.multimodel.posthoc.ids import MatrixId


@dataclass(frozen=True)
class MatrixImportance:
    """Finalized scores for one matrix ``j`` of one model."""

    matrix: MatrixId
    beta: Tensor  # [C] float64, NaN where undefined
    q: Tensor  # [C] float64
    rho: Tensor  # [C] float64, NaN when sum_k q = 0
    sigma: float  # sigma_j
    fire_count: Tensor  # [C] int64, number of anchors with g = 1
    anchor_count: int

    def score(self, metric: str) -> Tensor:
        if metric == "beta":
            return self.beta
        if metric == "rho":
            return self.rho
        raise ValueError(f"unknown screening metric {metric!r}")


class MatrixImportanceAccumulator:
    """Streaming sufficient statistics for ``sigma_j``, ``beta``, ``q`` and ``rho``."""

    def __init__(self, matrix: MatrixId, n_features: int, d_out: int):
        self.matrix = matrix
        self.n_features = n_features
        self.d_out = d_out
        self.anchor_count = 0
        self.output_sum = torch.zeros(d_out, dtype=torch.float64)
        self.output_sq_sum = torch.zeros((), dtype=torch.float64)
        self.contribution_sq_sum = torch.zeros(n_features, dtype=torch.float64)
        self.fire_count = torch.zeros(n_features, dtype=torch.int64)

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor, gate: Tensor, U: Tensor, V: Tensor) -> None:
        """Add one batch of anchors.

        ``x [A,d_in]`` and ``y [A,d_out]`` are the captured native matrix input/output,
        ``gate [A,C]`` the binary shared gate of site ``p(j)``, ``U [C,d_out]``,
        ``V [C,d_in]`` the rank-one factors.
        """

        if y.shape != (x.shape[0], self.d_out):
            raise ValueError(f"{self.matrix.key}: output must be [A,{self.d_out}]")
        if gate.shape != (x.shape[0], self.n_features):
            raise ValueError(f"{self.matrix.key}: gate must be [A,{self.n_features}]")
        coefficients = component_coefficients(x.double(), V.double(), gate)
        sq = contribution_sq_norms(coefficients, U.double())
        y64 = y.double()
        self.anchor_count += int(x.shape[0])
        self.output_sum += y64.sum(dim=0).cpu()
        self.output_sq_sum += y64.square().sum().cpu()
        self.contribution_sq_sum += sq.sum(dim=0).cpu()
        self.fire_count += gate.bool().sum(dim=0).to(torch.int64).cpu()

    def finalize(self) -> MatrixImportance:
        if self.anchor_count == 0:
            raise ValueError(f"{self.matrix.key}: no anchors were accumulated")
        n = float(self.anchor_count)
        mean = self.output_sum / n
        variance = float(self.output_sq_sum / n - mean.square().sum())
        sigma = math.sqrt(max(variance, 0.0))
        q = (self.contribution_sq_sum / n).sqrt()
        fired = self.fire_count.to(torch.float64)
        conditional = torch.where(
            fired > 0, self.contribution_sq_sum / fired.clamp_min(1.0), torch.nan
        ).sqrt()
        beta = conditional / sigma if sigma > 0 else torch.full_like(q, torch.nan)
        total_q = float(q.sum())
        rho = q / total_q if total_q > 0 else torch.full_like(q, torch.nan)
        return MatrixImportance(
            matrix=self.matrix,
            beta=beta,
            q=q,
            rho=rho,
            sigma=sigma,
            fire_count=self.fire_count.clone(),
            anchor_count=self.anchor_count,
        )


class ImportanceTable:
    """All finalized ``MatrixImportance`` values of one analysis run."""

    def __init__(self, entries: Iterable[MatrixImportance]):
        self.entries: dict[MatrixId, MatrixImportance] = {}
        for entry in entries:
            if entry.matrix in self.entries:
                raise ValueError(f"duplicate importance entry {entry.matrix.key}")
            self.entries[entry.matrix] = entry
        sizes = {int(entry.beta.numel()) for entry in self.entries.values()}
        if len(sizes) > 1:
            raise ValueError("importance entries disagree on dictionary size")

    def __getitem__(self, matrix: MatrixId) -> MatrixImportance:
        return self.entries[matrix]

    def __contains__(self, matrix: object) -> bool:
        return matrix in self.entries

    def matrices(self, model: str | None = None, site: str | None = None) -> list[MatrixId]:
        """Sorted matrices, optionally restricted to one model and/or ``J_p^{(n)}``."""

        return sorted(
            matrix
            for matrix in self.entries
            if (model is None or matrix.model == model) and (site is None or matrix.site == site)
        )

    @property
    def n_features(self) -> int:
        return int(next(iter(self.entries.values())).beta.numel())

    def gate_fire_counts(self, site: str) -> Tensor:
        """Firing count of each shared gate at ``site``.

        Every matrix of a site sees the same shared gate, so any one of them gives the
        count; the method checks that they agree.
        """

        counts = [self.entries[m].fire_count for m in self.matrices(site=site)]
        if not counts:
            raise ValueError(f"no matrices for site {site!r}")
        if any(not torch.equal(counts[0], other) for other in counts[1:]):
            raise ValueError(f"matrices of site {site!r} disagree on shared gate counts")
        return counts[0]

    def save(self, directory: Path) -> None:
        tensors: dict[str, Tensor] = {}
        meta: dict[str, dict[str, float | int]] = {}
        for matrix, entry in sorted(self.entries.items()):
            for name in ("beta", "q", "rho", "fire_count"):
                tensors[f"{matrix.key}::{name}"] = getattr(entry, name).contiguous()
            meta[matrix.key] = {"sigma": entry.sigma, "anchor_count": entry.anchor_count}
        save_file(tensors, str(directory / "importance.safetensors"))
        with (directory / "importance_meta.json").open("x", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, directory: Path) -> ImportanceTable:
        tensors = load_file(str(directory / "importance.safetensors"))
        meta: Mapping[str, Mapping[str, float]] = json.loads(
            (directory / "importance_meta.json").read_text(encoding="utf-8")
        )
        entries = []
        for key, values in meta.items():
            entries.append(
                MatrixImportance(
                    matrix=MatrixId.from_key(key),
                    beta=tensors[f"{key}::beta"],
                    q=tensors[f"{key}::q"],
                    rho=tensors[f"{key}::rho"],
                    fire_count=tensors[f"{key}::fire_count"],
                    sigma=float(values["sigma"]),
                    anchor_count=int(values["anchor_count"]),
                )
            )
        return cls(entries)
