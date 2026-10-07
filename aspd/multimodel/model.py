"""The joint multi-model activation/parameter decomposition.

Tensor convention: ``B`` batches sequences, ``T`` is the common token position, ``C`` is the
shared latent index, ``n`` indexes models, and ``j`` indexes matrices.  Comments deliberately keep
these symbols visible so implementation choices can be checked against the method equations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from aspd.multimodel.batch_topk import SparseCode, batch_topk
from aspd.multimodel.components import PartitionedComponents, RankOneComponents
from aspd.multimodel.config import EncoderSpec, ObjectiveSpec, SparsitySpec
from aspd.multimodel.losses import DeadFeatureTracker, fvu, matryoshka_boundaries, residual_fvu

# Running example in comments: N=2, B=1, T=2, C=4, d_act=3, one matrix j with
# d_in=2 and d_out=2.  Superscript (n) is the model; subscript (j,t,c) follows the note.


def _safe_key(name: str) -> str:
    return name.replace(".", "__dot__")


class FrozenWeight(nn.Module):
    """The fixed ``W^{(n)}_j`` used to form ``y^{(n)}_{j,t}=W^{(n)}_j x^{(n)}_{j,t}``."""

    def __init__(self, weight: Tensor):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"target weight must be a matrix, got {weight.shape}")
        self.register_buffer("weight", weight.detach().clone(), persistent=True)

    @property
    def d_out(self) -> int:
        return self.weight.shape[0]

    @property
    def d_in(self) -> int:
        return self.weight.shape[1]

    def apply(self, x: Tensor) -> Tensor:
        # Y_j^(n)=X_j^(n) W_j^(n)T in batched row notation.
        # Example x_t=[2,1], W=[[1,0],[0,3]] gives y_t=[2,3].
        return torch.nn.functional.linear(x.to(self.weight.dtype), self.weight)


class LinearSharedEncoder(nn.Module):
    """Crosscoder reduction: ``a_tc=relu(Aggregate_n(W_e^(n) r_t^(n))+b_c)``."""

    def __init__(self, activation_dims: Sequence[int], n_features: int, aggregation: str):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(d_act, n_features, bias=False) for d_act in activation_dims]
        )
        self.bias = nn.Parameter(torch.zeros(n_features))
        self.aggregation = aggregation
        for projection in self.projections:
            with torch.no_grad():
                # Normalize every encoder row W_e,c^(n) to norm 1 at initialization.
                # Example row [3,4] / ||[3,4]||=[0.6,0.8].
                projection.weight.div_(
                    projection.weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
                )

    def forward(self, activations: Sequence[Tensor], valid_tokens: Tensor) -> Tensor:
        del valid_tokens
        # z_t,c^(n)=W_e,c^(n) r_t^(n). For N=2, stack shape is [2,B,T,C].
        # Example z^(1)=[1,3], z^(2)=[5,-1] for one (t,c slice).
        encoded = torch.stack(
            [projection(r.to(projection.weight.dtype)) for projection, r in zip(self.projections, activations, strict=True)]
        )
        # Aggregate over n. Example sum [1,3]+[5,-1]=[6,2].
        combined = encoded.sum(dim=0)
        if self.aggregation == "mean":
            # Crosscoder mean: [6,2]/N=2 -> [3,1].
            combined = combined / len(activations)
        # a_t,c=relu(Aggregate_n z_t,c^(n)+b_c); e.g. relu([3,1]+[-4,0])=[0,1].
        return torch.relu(combined + self.bias)


class FactoredTransformerEncoder(nn.Module):
    """Sequence version of ``g^s = sigma_K o A o (h^(1) x ... x h^(N))``."""

    def __init__(
        self, activation_dims: Sequence[int], n_features: int, cfg: EncoderSpec
    ) -> None:
        super().__init__()
        # p is the common width that permits heterogeneous d_act^(n); e.g. 3 and 5 -> p=4.
        p = cfg.projection_dim
        self.input_projections = nn.ModuleList([nn.Linear(d, p) for d in activation_dims])
        self.model_encoders = nn.ModuleList(
            [self._transformer(p, cfg) for _ in activation_dims]
        )
        self.shared_encoder = self._transformer(p, cfg)
        self.output = nn.Linear(p, n_features)
        self.aggregation = cfg.aggregation
        self.causal = cfg.causal

    @staticmethod
    def _transformer(width: int, cfg: EncoderSpec) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=cfg.n_heads,
            # Example p=4, mlp_ratio=4 gives the transformer's hidden width 16.
            dim_feedforward=int(width * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)

    def forward(self, activations: Sequence[Tensor], valid_tokens: Tensor) -> Tensor:
        # Common A1 token count. Example R^(n).shape=[1,2,d_act^(n)] gives T=2.
        tokens = activations[0].shape[1]
        causal_mask = None
        if self.causal:
            # mask[t,q]=True for q>t. At T=2: [[F,T],[F,F]], so token 0 cannot see token 1.
            causal_mask = torch.triu(
                torch.ones(tokens, tokens, dtype=torch.bool, device=valid_tokens.device), diagonal=1
            )
        # Transformer expects True at ignored positions; valid [T,F] becomes padding [F,T].
        padding_mask = ~valid_tokens
        encoded = []
        for projection, encoder, r in zip(
            self.input_projections, self.model_encoders, activations, strict=True
        ):
            # z_t^(n)=Proj^(n) r_t^(n), mapping [B,T,d_act^(n)] -> [B,T,p].
            z = projection(r.to(projection.weight.dtype))
            # h^(n)(R^(n)) contextualizes z without reading future or padded tokens.
            encoded.append(encoder(z, mask=causal_mask, src_key_padding_mask=padding_mask))
        # Aggregate h^(n). Example two [B,T,p] tensors become their elementwise sum.
        merged = torch.stack(encoded).sum(dim=0)
        if self.aggregation == "mean":
            # A uses 1/N sum; example values 2 and 6 become 4.
            merged = merged / len(encoded)
        # Shared f processes the cross-model representation A(h^(1),...,h^(N)).
        merged = self.shared_encoder(
            merged, mask=causal_mask, src_key_padding_mask=padding_mask
        )
        # Project p->C and rectify to non-negative a_{t,c} before sigma_K.
        return torch.relu(self.output(merged))


class ActivationDecoders(nn.Module):
    """Model-specific ``D^(n)`` and ``b^(n)`` for ``r_hat_t^(n)=g_t^s D^(n)+b^(n)``."""

    def __init__(
        self,
        activation_dims: Sequence[int],
        n_features: int,
        n_shared: int,
        tie_shared: bool,
    ) -> None:
        super().__init__()
        self.activation_dims = list(activation_dims)
        self.n_features = n_features
        self.n_shared = n_shared
        self.tie_shared = tie_shared
        # One b^(n) in R^d_act^(n) per model; initialize activation reconstruction at zero.
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in activation_dims])
        if tie_shared:
            if len(set(activation_dims)) != 1:
                raise ValueError("tied shared decoders require equal d_act across models")
            # Exact D1 tying: this single D_S object supplies d_c^(n) for every n,c in S.
            self.shared = nn.Parameter(torch.empty(n_shared, activation_dims[0]))
            nn.init.kaiming_uniform_(self.shared, a=math.sqrt(5))
            # Each model owns D_E^(n), shape [C_E,d_act^(n)].
            self.exclusive = nn.ParameterList(
                [nn.Parameter(torch.empty(n_features - n_shared, d)) for d in activation_dims]
            )
            for decoder in self.exclusive:
                nn.init.kaiming_uniform_(decoder, a=math.sqrt(5))
            self.full = None
        else:
            self.shared = None
            self.exclusive = nn.ParameterList()
            # Untied D0/D1/D2: each n owns all decoder rows D^(n)[C,d_act^(n)].
            self.full = nn.ParameterList(
                [nn.Parameter(torch.empty(n_features, d)) for d in activation_dims]
            )
            for decoder in self.full:
                nn.init.kaiming_uniform_(decoder, a=math.sqrt(5))

    def weight(self, model_index: int) -> Tensor:
        """Return decoder rows ``d_c^(n)`` with shape ``[C,d_act^(n)]``."""

        if self.tie_shared:
            assert self.shared is not None
            # D^(n)=[D_S,D_E^(n)]; e.g. C_S=3,C_E=1 gives [4,d_act].
            return torch.cat((self.shared, self.exclusive[model_index]), dim=0)
        assert self.full is not None
        return self.full[model_index]

    def initialize_from_linear_encoder(self, encoder: LinearSharedEncoder) -> None:
        """Initialize ``D^(n)`` from ``W_e^(n)`` without tying their later optimization."""

        with torch.no_grad():
            if self.tie_shared:
                assert self.shared is not None
                # Initialize tied d_c by mean_n W_e,c^(n); e.g. ([1,3]+[3,1])/2=[2,2].
                shared_init = torch.stack(
                    [p.weight[: self.n_shared] for p in encoder.projections]
                ).mean(dim=0)
                self.shared.copy_(shared_init)
                for n, projection in enumerate(encoder.projections):
                    self.exclusive[n].copy_(projection.weight[self.n_shared:])
            else:
                assert self.full is not None
                for decoder, projection in zip(self.full, encoder.projections, strict=True):
                    decoder.copy_(projection.weight)

    def reconstruct(
        self, code: Tensor, model_index: int, valid_tokens: Tensor, feature_stop: int | None = None
    ) -> Tensor:
        """Sparse activation reconstruction using magnitudes ``g^s_{t,c}``, not binary gates."""

        # D2 uses stop=C_S; the full loss uses stop=C. Example stop=3 versus 4.
        stop = self.n_features if feature_stop is None else feature_stop
        # Keep g^s_{t,c} only for c<stop.
        selected_code = code[..., :stop]
        # Match it with decoder rows d_c^(n), shape [stop,d_act^(n)].
        decoder = self.weight(model_index)[:stop]
        # Example [B=1,T=2,stop=4].
        batch, tokens, _ = selected_code.shape
        # Flatten (b,t)->p to enumerate sparse active pairs.
        flat_code = selected_code.reshape(batch * tokens, stop)
        flat_valid = valid_tokens.reshape(batch * tokens)
        # (p,c) where g^s_{t,c}!=0. Example {(0,1),(1,3)}.
        position, feature = ((flat_code != 0) & flat_valid[:, None]).nonzero(as_tuple=True)
        # Initialize R_hat^(n)-b^(n)=0 for every token.
        output = torch.zeros(
            batch * tokens,
            self.activation_dims[model_index],
            device=code.device,
            dtype=torch.promote_types(code.dtype, decoder.dtype),
        )
        if feature.numel():
            # Each active write is g^s_{t,c} d_c^(n); e.g. g^s=2,d=[1,3] -> [2,6].
            writes = flat_code[position, feature].to(decoder.dtype)[:, None] * decoder[feature]
            # Sum all active c for the same token t.
            output.index_add_(0, position, writes.to(output.dtype))
        # Restore [B,T,d_act^(n)].
        output = output.reshape(batch, tokens, self.activation_dims[model_index])
        # R_hat_t^(n)=sum_c g^s_t,c d_c^(n)+b^(n).
        return output + self.biases[model_index]

    def relative_norm_weights(self) -> Tensor:
        """S2 score ``omega_c=sum_n ||d_c^(n)||_2`` and the denominator of ``rho_c^(n)``."""

        # omega_c=sum_n ||d_c^(n)||_2. Example norms [2,3] across N=2 give omega_c=5.
        return torch.stack([self.weight(n).norm(dim=-1) for n in range(len(self.biases))]).sum(0)


class MultiModelASPD(nn.Module):
    """One ``g^s_{t,c}`` shared by all models and every selected matrix.

    ``target_weights[n][j]`` is ``W_j^(n)``.  ``batch['R'][n]`` is ``R^(n)`` and
    ``batch['X'][n][j]`` is ``X_j^(n)``.  The model never assumes two models internally, so the same
    implementation extends to ``N`` once caches for more models are supplied.
    """

    def __init__(
        self,
        model_names: Sequence[str],
        activation_dims: Sequence[int],
        target_weights: Sequence[Mapping[str, Tensor]],
        encoder_cfg: EncoderSpec,
        sparsity_cfg: SparsitySpec,
        objective_cfg: ObjectiveSpec,
    ) -> None:
        super().__init__()
        if not (len(model_names) == len(activation_dims) == len(target_weights)):
            raise ValueError("model_names, activation_dims and target_weights must have length N")
        self.model_names = list(model_names)
        self.activation_dims = list(activation_dims)
        self.sparsity_cfg = sparsity_cfg
        self.objective_cfg = objective_cfg
        # C is the common latent count; example C=4.
        c = sparsity_cfg.n_features
        # C_S defines S=[0,C_S); example shared_fraction=.75 gives C_S=3 and C_E=1.
        c_shared = sparsity_cfg.n_shared

        if encoder_cfg.kind == "linear":
            self.encoder: nn.Module = LinearSharedEncoder(
                activation_dims, c, encoder_cfg.aggregation
            )
        else:
            self.encoder = FactoredTransformerEncoder(activation_dims, c, encoder_cfg)
        self.activation_decoders = ActivationDecoders(
            activation_dims,
            c,
            c_shared,
            sparsity_cfg.tie_shared_decoders,
        )
        if isinstance(self.encoder, LinearSharedEncoder):
            self.activation_decoders.initialize_from_linear_encoder(self.encoder)

        self.target_weights = nn.ModuleList()
        for weights in target_weights:
            self.target_weights.append(
                nn.ModuleDict({_safe_key(name): FrozenWeight(weight) for name, weight in weights.items()})
            )
        self.matrix_names = [list(weights) for weights in target_weights]

        self.components = nn.ModuleList([nn.ModuleDict() for _ in model_names])
        tied_shared: dict[str, RankOneComponents] = {}
        for n, weights in enumerate(target_weights):
            for name, weight in weights.items():
                # W_j^(n) in R^[d_out,d_in]; example 2x2.
                d_out, d_in = weight.shape
                key = _safe_key(name)
                if sparsity_cfg.diffing == "D0":
                    component: nn.Module = RankOneComponents(c, d_in, d_out)
                else:
                    if sparsity_cfg.tie_shared_mechanisms:
                        if name not in tied_shared:
                            # Create P_j,c^S once for the first n; later models reuse this object.
                            tied_shared[name] = RankOneComponents(c_shared, d_in, d_out)
                        # Exact equality is structural: P_j,c^(n) and P_j,c^(n') share U,V storage.
                        shared = tied_shared[name]
                        if (shared.d_in, shared.d_out) != (d_in, d_out):
                            raise ValueError(
                                f"cannot tie {name!r}: model dimensions differ across n"
                            )
                    else:
                        shared = RankOneComponents(c_shared, d_in, d_out)
                    # Exclusive bank has C_E=C-C_S rows; example 4-3=1.
                    exclusive = RankOneComponents(c - c_shared, d_in, d_out)
                    component = PartitionedComponents(shared, exclusive)
                self.components[n][key] = component

        self.dead_tracker = DeadFeatureTracker(c, objective_cfg.dead_after_batches)

    def _target(self, model_index: int, matrix_name: str) -> FrozenWeight:
        return self.target_weights[model_index][_safe_key(matrix_name)]

    def component_factors(self, model_index: int, matrix_name: str) -> tuple[Tensor, Tensor]:
        """Return ``U^(n)_j,V^(n)_j`` with shapes ``[C,d_out]`` and ``[C,d_in]``."""

        component = self.components[model_index][_safe_key(matrix_name)]
        return component.factors()  # type: ignore[no-any-return,attr-defined]

    def encode(self, activations: Sequence[Tensor], valid_tokens: Tensor) -> SparseCode:
        """Compute the non-negative shared code and ASPD's binary component gate."""

        # a=g_pre(R^(1),...,R^(N)); shape [B,T,C], e.g. [1,2,4].
        preactivations = self.encoder(activations, valid_tokens)
        ranking_weights = None
        if self.sparsity_cfg.selection_score == "S2":
            # S2 sets omega_c=sum_n||d_c^(n)||; S1 leaves omega_c=None -> ones.
            ranking_weights = self.activation_decoders.relative_norm_weights()
        # sigma_K(a*omega) selects support, returns magnitudes g^s and binary g=1[g^s>0].
        return batch_topk(preactivations, valid_tokens, self.sparsity_cfg, ranking_weights)

    def _activation_loss(
        self,
        model_index: int,
        target_r: Tensor,
        code: Tensor,
        valid_tokens: Tensor,
        feature_stop: int,
    ) -> tuple[Tensor, Tensor]:
        """ASPD ``L_act^(n)`` with nested prefixes ending at ``feature_stop``."""

        # Example fractions [.25,.25,.5] with stop=4 define prefixes [0,1,2,4].
        fractions = self.objective_cfg.matryoshka_group_fractions
        boundaries = matryoshka_boundaries(feature_stop, fractions)
        losses = []
        # The empty prefix learns b^(n), matching the upstream ASPD Matryoshka objective.
        bias = self.activation_decoders.biases[model_index]
        # Empty prefix: R_hat_t^(n)=b^(n), repeated over [B,T].
        empty = bias.expand_as(target_r)
        # First term is FVU(R^(n),b^(n)).
        losses.append(fvu(target_r, empty, valid_tokens))
        reconstruction = empty
        for stop in boundaries[1:]:
            # Prefix m reconstructs with c<m; e.g. stop=2 uses c={0,1} only.
            reconstruction = self.activation_decoders.reconstruct(
                code, model_index, valid_tokens, feature_stop=stop
            )
            # Add FVU(R^(n),R_hat_prefix^(n)).
            losses.append(fvu(target_r, reconstruction, valid_tokens))
        # L_act^(n) is the mean of bias-only and all nested-prefix FVU terms.
        return torch.stack(losses).mean(), reconstruction

    def _internal_loss(
        self,
        model_index: int,
        x_by_matrix: Mapping[str, Tensor],
        gate: Tensor,
        valid_tokens: Tensor,
        shared_only: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Mean_j FVU(``Y_j^(n)``, ``Y_hat_j^(n)``), the local ASPD supervision."""

        if not x_by_matrix:
            # Crosscoder reduction J^(n)=empty makes L_internal^(n)=0 while retaining autograd.
            zero = gate.sum() * 0.0
            return zero, {}
        losses: list[Tensor] = []
        by_matrix: dict[str, Tensor] = {}
        for name, x in x_by_matrix.items():
            # Target Y_j^(n)=W_j^(n)X_j^(n); e.g. [2,1] -> [2,3] under the example W.
            target = self._target(model_index, name).apply(x)
            component = self.components[model_index][_safe_key(name)]
            if isinstance(component, PartitionedComponents):
                # D1/D2: reconstruct with S only or S union E according to shared_only.
                reconstruction = component.reconstruct(
                    x, gate, valid_tokens, shared_only=shared_only
                )
            else:
                if shared_only:
                    raise ValueError("shared-only loss requires a D1/D2 partition")
                # D0: Y_hat_j^(n)=sum_c g_t,c(v_j,c^T x_j,t)u_j,c.
                reconstruction = component.reconstruct(x, gate, valid_tokens)
            # L_internal,j^(n)=FVU(Y_j^(n),Y_hat_j^(n)).
            matrix_loss = fvu(target, reconstruction, valid_tokens)
            losses.append(matrix_loss)
            by_matrix[name] = matrix_loss
        # L_internal^(n)=1/|J^(n)| sum_j L_internal,j^(n): every matrix has equal weight.
        return torch.stack(losses).mean(), by_matrix

    def _objective(
        self,
        batch: Mapping[str, object],
        code: SparseCode,
        shared_only: bool,
    ) -> tuple[Tensor, dict[str, Tensor], list[Tensor]]:
        valid_tokens: Tensor = batch["valid_tokens"]  # type: ignore[assignment]
        activations: Sequence[Tensor] = batch["R"]  # type: ignore[assignment]
        inputs: Sequence[Mapping[str, Tensor]] = batch["X"]  # type: ignore[assignment]
        # L(S) stops at C_S; L([C]) stops at C. Example 3 versus 4.
        feature_stop = self.sparsity_cfg.n_shared if shared_only else self.sparsity_cfg.n_features
        # Activation reconstruction consumes magnitudes g^s_{t,c}.
        values = code.values[..., :feature_stop]
        # Parameter reconstruction consumes the binary ASPD gate g_{t,c}.
        gate = code.gate[..., :feature_stop]
        # Differentiable scalar zero seeds sum_n losses on the same graph/device.
        total = code.values.sum() * 0.0
        stats: dict[str, Tensor] = {}
        full_reconstructions: list[Tensor] = []
        for n, model_name in enumerate(self.model_names):
            act_loss, reconstruction = self._activation_loss(
                n, activations[n], values, valid_tokens, feature_stop
            )
            internal_loss, matrix_losses = self._internal_loss(
                n, inputs[n], gate, valid_tokens, shared_only
            )
            # Add model n: L += L_internal^(n)+lambda_act L_act^(n).
            # Example internal=.4, act=.2, lambda=1 contributes .6.
            total = total + internal_loss + self.objective_cfg.lambda_act * act_loss
            prefix = "shared/" if shared_only else ""
            stats[f"{prefix}act/{model_name}"] = act_loss.detach()
            stats[f"{prefix}internal/{model_name}"] = internal_loss.detach()
            for matrix_name, value in matrix_losses.items():
                stats[f"{prefix}internal/{model_name}/{matrix_name}"] = value.detach()
            full_reconstructions.append(reconstruction)
        return total, stats, full_reconstructions

    def _auxk_loss(
        self,
        activations: Sequence[Tensor],
        full_reconstructions: Sequence[Tensor],
        preactivations: Tensor,
        valid_tokens: Tensor,
    ) -> Tensor:
        """Revive dead ``c`` by using their pre-activations to fit the L_act residual."""

        # dead[c]=1 when c has not fired for dead_after_batches.
        dead = self.dead_tracker.dead
        # Example dead=[F,T,F,T] gives n_dead=2.
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return preactivations.sum() * 0.0
        # AuxK cannot select more than the available dead latents.
        k = min(self.objective_cfg.top_k_aux, n_dead)
        # Map compact dead-axis indices back to global c; example [1,3].
        dead_indices = dead.nonzero(as_tuple=False).flatten()
        # Candidate a_{t,c} only for dead c; shape [B,T,n_dead].
        dead_preactivations = preactivations[..., dead]
        # For every token select its top k dead pre-activations.
        top = torch.topk(dead_preactivations, k, dim=-1)
        # Start auxiliary g_aux^s at zero on the full C axis.
        aux_code = torch.zeros_like(preactivations)
        # Example local dead index 1 maps through [1,3] to global c=3.
        chosen_features = dead_indices[top.indices]
        # Insert selected original a_{t,c} magnitudes into global positions.
        aux_code.scatter_(-1, chosen_features, top.values)
        # Padding positions must contribute no auxiliary reconstruction.
        aux_code = aux_code * valid_tokens[..., None].to(aux_code.dtype)
        losses = []
        for n, (target, reconstruction) in enumerate(
            zip(activations, full_reconstructions, strict=True)
        ):
            # Residual target is R^(n)-R_hat_full^(n); detach R_hat so AuxK only revives dead paths.
            residual = target - reconstruction.detach()
            # R_hat_aux^(n)=sum_dead g_aux^s d_c^(n); subtract b because residual has no bias term.
            aux_reconstruction = self.activation_decoders.reconstruct(
                aux_code, n, valid_tokens
            ) - self.activation_decoders.biases[n]
            losses.append(
                residual_fvu(target, residual, aux_reconstruction, valid_tokens)
            )
        # Average AuxK over models n so one dead feature must help the joint representation.
        return torch.stack(losses).mean()

    def forward(self, batch: Mapping[str, object], update_dead_tracker: bool = True) -> dict[str, Tensor]:
        """Return the optimized loss and named diagnostics for one paired activation batch."""

        valid_tokens: Tensor = batch["valid_tokens"]  # type: ignore[assignment]
        activations: Sequence[Tensor] = batch["R"]  # type: ignore[assignment]
        # One shared code (g^s,g) is computed from all R^(n).
        code = self.encode(activations, valid_tokens)
        # L([C]) uses every latent for activation and parameter reconstructions.
        full_loss, stats, full_reconstructions = self._objective(batch, code, shared_only=False)
        total = full_loss
        stats["loss/full"] = full_loss.detach()

        if self.sparsity_cfg.diffing == "D2" and self.sparsity_cfg.d2_gamma > 0:
            # L(S) masks out every c in E in both decoder and P_j,c paths.
            shared_loss, shared_stats, _ = self._objective(batch, code, shared_only=True)
            # D2: L_D2=L([C])+gamma L(S). Example gamma=.5: 1.2+.5*.8=1.6.
            total = total + self.sparsity_cfg.d2_gamma * shared_loss
            stats.update(shared_stats)
            stats["loss/shared"] = shared_loss.detach()

        auxk = self._auxk_loss(
            activations,
            full_reconstructions,
            code.preactivations,
            valid_tokens,
        )
        # Final objective adds lambda_aux L_aux; example total=1.6+.03125*.4=1.6125.
        total = total + self.objective_cfg.auxk_coefficient * auxk
        stats["loss/auxk"] = auxk.detach()
        # Mean L0=E_t sum_c g_t,c. At exact K=2 per token this reports 2.
        stats["sparsity/l0"] = code.gate[valid_tokens].float().sum(-1).mean().detach()
        # Dead fraction=(number of dead c)/C; example 1 dead of 4 -> .25.
        stats["sparsity/dead_fraction"] = self.dead_tracker.dead.float().mean().detach()
        if self.sparsity_cfg.diffing != "D0":
            c_shared = self.sparsity_cfg.n_shared
            # E_t sum_{c in S}g_t,c; Dual-K should approach K_S.
            stats["sparsity/l0_shared"] = (
                code.gate[..., :c_shared][valid_tokens].float().sum(-1).mean().detach()
            )
            # E_t sum_{c in E}g_t,c; Dual-K should approach K_E.
            stats["sparsity/l0_exclusive"] = (
                code.gate[..., c_shared:][valid_tokens].float().sum(-1).mean().detach()
            )
        if update_dead_tracker:
            self.dead_tracker.observe(code.gate, valid_tokens)
        stats["loss"] = total
        return stats
