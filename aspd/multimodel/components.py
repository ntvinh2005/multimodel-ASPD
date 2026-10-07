"""Rank-1 factors for ``P^{(n)}_{j,c}=u^{(n)}_{j,c}v^{(n)T}_{j,c}``.

Only active ``(t,c)`` pairs are evaluated.  This preserves ASPD Eq. 1 while changing its cost from
dense ``O(T C (d_in+d_out))`` to ``O(T K (d_in+d_out))`` after BatchTopK chooses the support.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# Worked example: B=1,T=2,C=3,d_in=2,d_out=2; only two (t,c) pairs are active.


class RankOneComponents(nn.Module):
    """A contiguous latent block of read directions ``V`` and write directions ``U``."""

    def __init__(self, n_features: int, d_in: int, d_out: int):
        super().__init__()
        self.n_features = n_features
        self.d_in = d_in
        self.d_out = d_out
        self.V = nn.Parameter(torch.empty(n_features, d_in))
        self.U = nn.Parameter(torch.empty(n_features, d_out))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))

    def reconstruct(self, x: Tensor, gate: Tensor, valid_tokens: Tensor) -> Tensor:
        """Compute ``sum_c g_tc (v_c^T x_t) u_c`` without forming all ``T*C`` reads."""

        if x.shape[:2] != gate.shape[:2] or gate.shape[-1] != self.n_features:
            raise ValueError("x and gate disagree on token grid or feature count")
        # Read token-grid sizes. Example x.shape=[1,2,2] gives batch=1,tokens=2.
        batch, tokens, _ = x.shape
        # Flatten (b,t)->p. Example two x_t rows become flat_x.shape=[2,2].
        flat_x = x.reshape(batch * tokens, self.d_in)
        # Flatten the same grid for g_{t,c}; example flat_gate.shape=[2,3].
        flat_gate = gate.reshape(batch * tokens, self.n_features)
        # Flatten valid-token mask; example [[True,True]] -> [True,True].
        flat_valid = valid_tokens.reshape(batch * tokens)
        # Enumerate only active valid pairs (p,c). Example (p,c)={(0,1),(1,2)}.
        active_position, active_feature = (flat_gate & flat_valid[:, None]).nonzero(as_tuple=True)
        # Start Y_hat_j^(n) at zero for every token/output coordinate.
        output = torch.zeros(
            batch * tokens, self.d_out, device=x.device, dtype=torch.promote_types(x.dtype, self.U.dtype)
        )
        if active_feature.numel() == 0:
            return output.reshape(batch, tokens, self.d_out)
        # Repeat x_{j,t} once per active c. Example [x_0,x_1] for pairs (0,1),(1,2).
        selected_x = flat_x[active_position].to(self.V.dtype)
        # r_{t,c}=v_{j,c}^T x_{j,t}. Example [2, -1] for the two active pairs.
        reads = (selected_x * self.V[active_feature]).sum(dim=-1)
        # e_{j,t,c}u_{j,c}=g_{t,c}(v^T x)u; g=1 for enumerated pairs.
        # Example reads [2,-1] times rows [u_1,u_2] gives [2u_1,-u_2].
        writes = reads[:, None] * self.U[active_feature]
        # Sum writes sharing token p. Example Y_hat[0]+=2u_1 and Y_hat[1]+=-u_2.
        output.index_add_(0, active_position, writes.to(output.dtype))
        # Restore [B,T,d_out], the shape of Y_j^(n).
        return output.reshape(batch, tokens, self.d_out)

    def factors(self) -> tuple[Tensor, Tensor]:
        """Return ``(U,V)`` with feature ``c`` on axis zero."""

        return self.U, self.V


class PartitionedComponents(nn.Module):
    """Shared block ``S`` plus exclusive block ``E`` for one model and matrix.

    ``shared`` may be the same module object across models.  That implements exact D1 tying without
    an equality penalty: the tied ``P^{(n)}_{j,c}`` literally uses the same parameters.
    """

    def __init__(self, shared: RankOneComponents, exclusive: RankOneComponents):
        super().__init__()
        self.shared = shared
        self.exclusive = exclusive

    @property
    def n_shared(self) -> int:
        return self.shared.n_features

    @property
    def n_features(self) -> int:
        # C=C_S+C_E; example 3 shared + 1 exclusive = 4 total latents.
        return self.shared.n_features + self.exclusive.n_features

    def reconstruct(
        self, x: Tensor, gate: Tensor, valid_tokens: Tensor, shared_only: bool = False
    ) -> Tensor:
        # Sum c in S first: Y_hat_S=sum_{c in S} g_tc P_jc x_t.
        out = self.shared.reconstruct(x, gate[..., : self.n_shared], valid_tokens)
        if not shared_only:
            # Full D1/D2 adds E: Y_hat_all=Y_hat_S+sum_{c in E} g_tc P_jc x_t.
            out = out + self.exclusive.reconstruct(
                x, gate[..., self.n_shared:], valid_tokens
            )
        return out

    def factors(self) -> tuple[Tensor, Tensor]:
        # Retrieve rows for S, e.g. U_S.shape=[C_S,d_out], V_S.shape=[C_S,d_in].
        u_shared, v_shared = self.shared.factors()
        # Retrieve rows for E with analogous shapes.
        u_exclusive, v_exclusive = self.exclusive.factors()
        # Reassemble latent axis [S,E] so row c matches the shared code's index c.
        return torch.cat((u_shared, u_exclusive)), torch.cat((v_shared, v_exclusive))
