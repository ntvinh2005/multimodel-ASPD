"""Component coefficients and output contributions (formulation, "Definition of a component").

For model ``n``, matrix ``j`` and anchor ``alpha``::

    e_{j,alpha,i}     = g_{p(j),alpha,i} * (v_{j,i}^T x_{j,alpha})       # coefficient
    Delta y_{j,alpha,i} = e_{j,alpha,i} * u_{j,i}                         # output contribution
    ||Delta y||_2^2   = e^2 * ||u_{j,i}||_2^2

Shapes follow the v3 convention ``U: [C, d_out]`` and ``V: [C, d_in]`` (feature on axis 0).
Everything here is pure tensor algebra and runs on any device.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _check(x: Tensor, V: Tensor, gate: Tensor) -> None:
    if x.ndim != 2 or V.ndim != 2 or gate.ndim != 2:
        raise ValueError("expected x [A,d_in], V [C,d_in], gate [A,C]")
    if x.shape[1] != V.shape[1]:
        raise ValueError("matrix input width and read vectors disagree")
    if gate.shape != (x.shape[0], V.shape[0]):
        raise ValueError("gate must have shape [anchors, features]")


def component_coefficients(x: Tensor, V: Tensor, gate: Tensor) -> Tensor:
    """Return ``e`` with shape ``[A, C]``.

    ``gate`` is the binary shared gate ``g`` (bool or 0/1).  The binary gate, not the
    continuous ``g^s``, controls the rank-one operators in ASPD v3.1.
    """

    _check(x, V, gate)
    reads = x.to(V.dtype) @ V.T
    return reads * gate.to(reads.dtype)


def contribution_sq_norms(coefficients: Tensor, U: Tensor) -> Tensor:
    """Return ``||Delta y_{j,alpha,i}||_2^2 = e^2 ||u_i||^2`` with shape ``[A, C]``."""

    if coefficients.ndim != 2 or U.ndim != 2 or coefficients.shape[1] != U.shape[0]:
        raise ValueError("expected coefficients [A,C] and U [C,d_out]")
    return coefficients.square() * U.square().sum(dim=-1).to(coefficients.dtype)


def component_contributions(coefficients: Tensor, U: Tensor) -> Tensor:
    """Return dense ``Delta y`` with shape ``[A, C, d_out]`` (use only for small C)."""

    if coefficients.ndim != 2 or U.ndim != 2 or coefficients.shape[1] != U.shape[0]:
        raise ValueError("expected coefficients [A,C] and U [C,d_out]")
    return coefficients[..., None] * U.to(coefficients.dtype)[None]


def write_norms(U: Tensor) -> Tensor:
    """``||u_{j,i}||_2`` for every feature; v3.1 keeps these at one after optimization."""

    return torch.linalg.vector_norm(U.float(), dim=-1)
