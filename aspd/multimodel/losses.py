"""Loss primitives written in the notation of the multi-model ASPD objective."""

from __future__ import annotations

import torch
from torch import Tensor

# Worked FVU example: valid scalar targets y=[0,2] and predictions y_hat=[0,1].


def valid_rows(tensor: Tensor, valid_tokens: Tensor) -> Tensor:
    """Flatten ``(batch, token)`` and retain the common valid token positions ``t``."""

    if tensor.shape[:2] != valid_tokens.shape:
        raise ValueError(f"tensor token grid {tensor.shape[:2]} != mask {valid_tokens.shape}")
    # Boolean indexing flattens (B,T). Example [B=1,T=3,D=2] with mask [T,F,T] -> [2,D].
    return tensor[valid_tokens]


def fvu(target: Tensor, reconstruction: Tensor, valid_tokens: Tensor) -> Tensor:
    """Fraction of variance unexplained, used for every ``L_act^(n)`` and ``L_internal,j^(n)``."""

    # y contains valid target rows, e.g. y=[0,2].
    y = valid_rows(target, valid_tokens).float()
    # y_hat contains matching reconstructions, e.g. y_hat=[0,1].
    y_hat = valid_rows(reconstruction, valid_tokens).float()
    # ||Y-Y_hat||_F^2. Example (0-0)^2+(2-1)^2=1.
    numerator = (y - y_hat).pow(2).sum()
    # ||Y-Y_bar||_F^2. Example Y_bar=1, so (0-1)^2+(2-1)^2=2.
    denominator = (y - y.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-8)
    # FVU=1/2=0.5: the reconstruction leaves half of target variance unexplained.
    return numerator / denominator


def matryoshka_boundaries(n_features: int, fractions: list[float]) -> list[int]:
    """Convert ASPD Matryoshka group fractions into nested prefix boundaries."""

    # Convert fractions to group widths. Example C=8,[.25,.25,.5] -> [2,2,4].
    sizes = [int(n_features * fraction) for fraction in fractions]
    # Put integer-rounding remainder in the last group so widths sum exactly to C.
    sizes[-1] += n_features - sum(sizes)
    if any(size <= 0 for size in sizes):
        raise ValueError(f"empty Matryoshka group at C={n_features}: {sizes}")
    # Prefix begins empty: boundary 0 represents the bias-only reconstruction.
    boundaries = [0]
    for size in sizes:
        # Cumulative ends. Example [2,2,4] produces [0,2,4,8].
        boundaries.append(boundaries[-1] + size)
    return boundaries


def normalized_mse(target: Tensor, reconstruction: Tensor, valid_tokens: Tensor) -> Tensor:
    """Per-element MSE divided by target variance; numerically equivalent to FVU."""

    # Alias retained for notation: normalized MSE [0,2] vs [0,1] is the same FVU=.5 above.
    return fvu(target, reconstruction, valid_tokens)


def residual_fvu(
    target: Tensor,
    residual: Tensor,
    residual_reconstruction: Tensor,
    valid_tokens: Tensor,
) -> Tensor:
    """AuxK residual MSE normalized by ``Var(target)``, as in ASPD's ``L_aux``.

    The numerator asks whether dead features reconstruct ``target - reconstruction``.  Its scale
    remains the variance of the original ``R^(n)`` target, rather than the usually much smaller
    residual variance.  This is the normalization used by the parent ASPD implementation.
    """

    # Original R^(n) rows set the normalization scale; example y=[0,2].
    y = valid_rows(target, valid_tokens).float()
    # Residual is R^(n)-R_hat^(n); example residual=[0,1].
    residual_rows = valid_rows(residual, valid_tokens).float()
    # Dead-feature reconstruction of that residual; example residual_hat=[0,0].
    reconstruction_rows = valid_rows(residual_reconstruction, valid_tokens).float()
    # Aux numerator ||residual-residual_hat||^2; example 1.
    numerator = (residual_rows - reconstruction_rows).pow(2).sum()
    # Normalize by Var(R^(n)), not Var(residual); example denominator=2.
    denominator = (y - y.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-8)
    # Example normalized AuxK loss=1/2.
    return numerator / denominator


class DeadFeatureTracker(torch.nn.Module):
    """Count batches since feature ``c`` last fired, for ASPD's AuxK revival loss."""

    def __init__(self, n_features: int, dead_after_batches: int):
        super().__init__()
        self.dead_after_batches = dead_after_batches
        self.register_buffer("batches_since_fired", torch.zeros(n_features, dtype=torch.long))

    @torch.no_grad()
    def observe(self, gate: Tensor, valid_tokens: Tensor) -> None:
        # fired[c]=OR_t g_{t,c}. Example gates [[1,0],[0,0]] -> fired=[True,False].
        fired = gate[valid_tokens].any(dim=0)
        # Every latent ages by one batch before active latents are reset.
        self.batches_since_fired.add_(1)
        # Example clocks [7,7] with fired=[T,F] become [0,8].
        self.batches_since_fired[fired] = 0

    @property
    def dead(self) -> Tensor:
        # c is dead after the configured number of inactive batches; e.g. clock 2000 >= 2000.
        return self.batches_since_fired >= self.dead_after_batches
