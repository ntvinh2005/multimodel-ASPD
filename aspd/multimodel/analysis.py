"""Post-hoc P1--P5 analysis for activation and parameter-space diffing."""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.cache import PairedActivationCache
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.model import MultiModelASPD

# Analysis examples use N=2 models and C=4 latents.  Model-axis vectors such as [0.5,0.5]
# denote a shared latent; [0.95,0.05] denotes concentration on model 0.


def decoder_relative_norms(model: MultiModelASPD) -> tuple[Tensor, Tensor]:
    """P1: return decoder norms and ``rho_c^(n)`` with shapes ``[N,C]``."""

    # q_c^(n)=||d_c^(n)||_2. Example c=0 has norms [2,2] across two models.
    norms = torch.stack(
        [
            model.activation_decoders.weight(n).detach().float().norm(dim=-1)
            for n in range(len(model.model_names))
        ]
    )
    # P1 rho_c^(n)=q_c^(n)/sum_n' q_c^(n'). Example [2,2]/4=[.5,.5].
    rho = norms / norms.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return norms, rho


@torch.no_grad()
def _top_activation_examples(
    model: MultiModelASPD,
    cache: PairedActivationCache,
    cfg: MultiModelExperimentConfig,
    device: torch.device,
    feature_ids: Tensor,
) -> dict[str, list[dict[str, Any]]]:
    """Keep contexts with the largest ``g^s_{t,c}`` for selected latent indices ``c``.

    These examples explain the common *when* of mechanism ``c``.  They are paired with P1/P2 in
    the saved analysis so a high activation score is never mistaken for parameter-space sharing.
    """

    k = cfg.analysis.top_examples_per_feature
    heaps: dict[int, list[tuple[float, int, list[int], int]]] = {
        int(c): [] for c in feature_ids.tolist()
    }
    serial = 0
    # c indices whose top contexts will be retained, e.g. selected=[7,203].
    selected = feature_ids.to(device)
    for raw_batch in cache.iter_batches(
        "validation", cfg.training.batch_size_sequences, shuffle=False
    ):
        valid = raw_batch["valid_tokens"].to(device)
        activations = [r.to(device) for r in raw_batch["R"]]
        # Compute shared g^s and retain selected c; shape [B,T,C_selected].
        values = model.encode(activations, valid).values[..., selected]
        ids: Tensor = raw_batch["input_ids"]  # type: ignore[assignment]
        # Flatten valid token positions. Example [B=2,T=3,F=2] -> [6,2], padding scores set 0.
        scores = values.masked_fill(~valid[..., None], 0).reshape(-1, selected.numel())
        # At most k examples per feature can come from one batch.
        take = min(k, scores.shape[0])
        if take == 0:
            continue
        # For each c independently select the largest g^s_t,c in this batch.
        top_values, top_positions = torch.topk(scores, take, dim=0)
        for local_index, feature_id in enumerate(feature_ids.tolist()):
            for score, flat_position in zip(
                top_values[:, local_index].detach().cpu().tolist(),
                top_positions[:, local_index].detach().cpu().tolist(),
                strict=True,
            ):
                if score <= 0:
                    continue
                # Invert flat p=b*T+t. Example p=7,T=5 -> b=1.
                sequence = flat_position // ids.shape[1]
                # Same example gives t=7 mod 5=2.
                position = flat_position % ids.shape[1]
                # Retain up to 16 tokens before the activating position.
                lo = max(0, position - 16)
                # Retain the activating token and up to 16 tokens after it.
                hi = min(ids.shape[1], position + 17)
                item = (float(score), serial, ids[sequence, lo:hi].tolist(), position - lo)
                serial += 1
                heap = heaps[feature_id]
                if len(heap) < k:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
    return {
        str(feature_id): [
            {
                "g_s": score,
                "token_ids": token_ids,
                "center_in_window": center,
            }
            for score, _serial, token_ids, center in sorted(heap, reverse=True)
        ]
        for feature_id, heap in heaps.items()
    }


def _factor_metrics(u_a: Tensor, v_a: Tensor, u_b: Tensor, v_b: Tensor) -> dict[str, Tensor]:
    """P5 without materializing ``P_c=u_cv_c^T``.

    A simultaneous sign flip of ``u`` and ``v`` leaves ``P`` unchanged.  We therefore orient the
    second factorization so its write cosine is non-negative before reporting separate read/write
    cosines.  ``component_cosine`` and ``relative_component_change`` are gauge invariant.
    """

    eps = 1e-12
    # <u_a,u_b> per c. Example [1,0] dot [0.8,0.6]=0.8.
    u_dot = (u_a * u_b).sum(-1)
    # <v_a,v_b> per c, the analogous read-direction inner product.
    v_dot = (v_a * v_b).sum(-1)
    # ||u|| for both models; example norms 1 and 1.
    u_a_norm, u_b_norm = u_a.norm(dim=-1), u_b.norm(dim=-1)
    # ||v|| for both models.
    v_a_norm, v_b_norm = v_a.norm(dim=-1), v_b.norm(dim=-1)
    # ||P_a||_F^2=||u_a||^2||v_a||^2 for rank-1 P_a=u_a v_a^T.
    p_a_sq = u_a_norm.square() * v_a_norm.square()
    # ||P_b||_F^2 by the same rank-1 identity.
    p_b_sq = u_b_norm.square() * v_b_norm.square()
    # ||P_a-P_b||_F^2=||P_a||^2+||P_b||^2-2<u_a,u_b><v_a,v_b>.
    difference_sq = (p_a_sq + p_b_sq - 2 * u_dot * v_dot).clamp_min(0)
    # cos(v_a,v_b); example dot .5 with unit norms gives .5.
    raw_read_cosine = v_dot / (v_a_norm * v_b_norm).clamp_min(eps)
    # cos(u_a,u_b); example .8 with unit norms gives .8.
    raw_write_cosine = u_dot / (u_a_norm * u_b_norm).clamp_min(eps)
    # Fix rank-1 sign gauge: if write cosine is negative, flip both reported signs for model b.
    orientation = torch.where(raw_write_cosine < 0, -1.0, 1.0)
    return {
        # ||P_b-P_a||_F/||P_a||_F; 0 means identical parameter-space computation.
        "relative_component_change": difference_sq.sqrt() / p_a_sq.sqrt().clamp_min(eps),
        # cos(P_a,P_b)=cos(u_a,u_b)cos(v_a,v_b), invariant to factor scaling/sign gauge.
        "component_cosine": raw_read_cosine * raw_write_cosine,
        "read_cosine": raw_read_cosine * orientation,
        "write_cosine": raw_write_cosine * orientation,
    }


@torch.no_grad()
def analyze(
    model: MultiModelASPD,
    cache: PairedActivationCache,
    cfg: MultiModelExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Accumulate gauge-invariant mechanism mass ``beta`` and the P1--P5 taxonomy."""

    model.eval()
    # N is model-axis size; current experiment N=2.
    n_models = len(model.model_names)
    # C is latent-axis size; main experiment C=8192.
    c = cfg.sparsity.n_features
    read_sq: list[dict[str, Tensor]] = [
        {name: torch.zeros(c, device=device) for name in names} for names in model.matrix_names
    ]
    # fire_count[c]=sum_t g_t,c across validation tokens, initially zero.
    fire_count = torch.zeros(c, device=device)
    y_sum: list[dict[str, Tensor]] = []
    y_sq_sum: list[dict[str, Tensor]] = []
    y_count: list[dict[str, int]] = []
    for n, names in enumerate(model.matrix_names):
        y_sum.append({name: torch.zeros(model._target(n, name).d_out, device=device) for name in names})
        y_sq_sum.append({name: torch.zeros((), device=device) for name in names})
        y_count.append({name: 0 for name in names})

    for raw_batch in cache.iter_batches(
        "validation", cfg.training.batch_size_sequences, shuffle=False
    ):
        valid = raw_batch["valid_tokens"].to(device)
        activations = [r.to(device) for r in raw_batch["R"]]
        inputs = [
            {name: value.to(device) for name, value in matrices.items()}
            for matrices in raw_batch["X"]
        ]
        # Compute shared g^s and g from all R^(n).
        code = model.encode(activations, valid)
        # Flatten (B,T)->p: g[p,c], shape [B*T,C].
        flat_gate = code.gate.reshape(-1, c)
        # Flatten common valid-token mask to p.
        flat_valid = valid.reshape(-1)
        # Enumerate all active (p,c) pairs used in P2's conditional expectation.
        position, feature = (flat_gate & flat_valid[:, None]).nonzero(as_tuple=True)
        # fire_count[c]+=number of active t. Example features [1,1,3] add counts c1+=2,c3+=1.
        fire_count.index_add_(0, feature, torch.ones_like(feature, dtype=fire_count.dtype))
        for n, matrices in enumerate(inputs):
            for name, x in matrices.items():
                u, v = model.component_factors(n, name)
                # X_j^(n)[B,T,d_in] -> [B*T,d_in] aligned with p.
                flat_x = x.reshape(-1, x.shape[-1]).float()
                # v_j,c^(n)T x_j,t^(n) for each active pair; example x=[2,1],v=[1,0] -> 2.
                selected_read = (flat_x[position] * v[feature].float()).sum(-1)
                # Accumulate sum_active(read^2) by c for the RMS term in beta_j,c^(n).
                read_sq[n][name].index_add_(0, feature, selected_read.square())
                # Exact target output y_j,t^(n)=W_j^(n)x_j,t^(n), keeping valid t only.
                y = model._target(n, name).apply(x).float()[valid]
                # Accumulate sum_t y for mean vector y_bar_j^(n).
                y_sum[n][name] += y.sum(dim=0)
                # Accumulate sum_t ||y_t||^2 for output variance energy.
                y_sq_sum[n][name] += y.square().sum()
                # Count valid t used by both moments.
                y_count[n][name] += y.shape[0]

    beta_by_model: list[dict[str, Tensor]] = []
    beta_model = []
    locus_by_model: list[dict[str, Tensor]] = []
    for n, names in enumerate(model.matrix_names):
        betas: dict[str, Tensor] = {}
        for name in names:
            # Avoid division by zero in an empty validation set; normal runs have count>0.
            count = max(y_count[n][name], 1)
            # y_bar=(1/T)sum_t y_t. Example sum [4,2]/T=2 -> [2,1].
            mean = y_sum[n][name] / count
            # sigma_j^2=E||y||^2-||E y||^2. Example 10-||[2,1]||^2=5.
            variance_energy = (y_sq_sum[n][name] / count - mean.square().sum()).clamp_min(1e-12)
            # sigma_j is RMS output scale; example sqrt(5).
            sigma = variance_energy.sqrt()
            u, _ = model.component_factors(n, name)
            # sqrt(sum_active(read^2)/sum_t g_t,c)=conditional RMS of v^T x when c fires.
            active_read_rms = (read_sq[n][name] / fire_count.clamp_min(1)).sqrt()
            # P2: RMS magnitude written by component c when it is active, relative to W_j output.
            # beta_j,c^(n)=||u_j,c^(n)||*RMS(v_j,c^T x | g=1)/sigma_j^(n).
            # Example ||u||=2,RMSread=3,sigma=4 gives beta=1.5.
            betas[name] = u.detach().float().norm(dim=-1) * active_read_rms / sigma
        beta_by_model.append(betas)
        # Stack beta_j,c^(n) over matrices j, shape [|J^(n)|,C].
        stacked = torch.stack([betas[name] for name in names])
        # beta_c^(n)=sqrt(sum_j beta_j,c^(n)^2). Example [3,4] across j -> 5.
        aggregate = stacked.square().sum(dim=0).sqrt()
        beta_model.append(aggregate)
        # Locus denominator sum_j beta_j,c^(n).
        denominator = stacked.sum(dim=0).clamp_min(1e-12)
        # pi_c^(n)(j)=beta_j,c^(n)/sum_j' beta_j',c^(n); example [1,3]/4=[.25,.75].
        locus_by_model.append({name: betas[name] / denominator for name in names})

    # beta[n,c] collects normalized mechanism mass across models.
    beta = torch.stack(beta_model)
    # P2 tilde-rho_c^(n)=beta_c^(n)/sum_n' beta_c^(n'). Example [9,1]/10=[.9,.1].
    mechanism_rho = beta / beta.sum(dim=0, keepdim=True).clamp_min(1e-12)
    decoder_norm, activation_rho = decoder_relative_norms(model)

    tensors: dict[str, Tensor] = {
        "decoder_norm_NC": decoder_norm.cpu(),
        "activation_rho_NC": activation_rho.cpu(),
        "mechanism_beta_NC": beta.cpu(),
        "mechanism_rho_NC": mechanism_rho.cpu(),
        "fire_count_C": fire_count.cpu(),
        # argmax_n rho_c^(n); example [.2,.8] records model index 1.
        "activation_dominant_model_C": activation_rho.argmax(dim=0).cpu(),
        # argmax_n tilde-rho_c^(n), the dominant parameter mechanism owner.
        "mechanism_dominant_model_C": mechanism_rho.argmax(dim=0).cpu(),
    }
    for n, model_name in enumerate(model.model_names):
        for matrix_name in model.matrix_names[n]:
            tensors[f"beta/{model_name}/{matrix_name}"] = beta_by_model[n][matrix_name].cpu()
            tensors[f"locus/{model_name}/{matrix_name}"] = locus_by_model[n][matrix_name].cpu()

    if n_models == 2 and model.matrix_names[0] == model.matrix_names[1]:
        for name in model.matrix_names[0]:
            u_a, v_a = model.component_factors(0, name)
            u_b, v_b = model.component_factors(1, name)
            for metric, value in _factor_metrics(
                u_a.float(), v_a.float(), u_b.float(), v_b.float()
            ).items():
                tensors[f"pair/{name}/{metric}"] = value.cpu()

    # Exact shared mass is 1/N; with N=2, uniform=.5.
    uniform = 1.0 / n_models
    # P1 shared when every |rho_n,c-1/N|<epsilon; [.54,.46] is shared at epsilon=.1.
    activation_shared = (activation_rho - uniform).abs().amax(dim=0) < cfg.analysis.shared_epsilon
    # Apply the same criterion to P2 tilde-rho.
    mechanism_shared = (mechanism_rho - uniform).abs().amax(dim=0) < cfg.analysis.shared_epsilon
    # Activation concentrated when max_n rho_n,c >= threshold; [.95,.05] passes .9.
    activation_concentrated = activation_rho.amax(dim=0) >= cfg.analysis.concentrated_threshold
    # Parameter mechanism concentrated by the analogous tilde-rho test.
    mechanism_concentrated = mechanism_rho.amax(dim=0) >= cfg.analysis.concentrated_threshold
    # covered marks the four clean corners of the P1/P2 taxonomy.
    covered = (
        (activation_shared & mechanism_shared)
        | (activation_shared & mechanism_concentrated)
        | (activation_concentrated & mechanism_shared)
        | (activation_concentrated & mechanism_concentrated)
    )
    taxonomy = {
        "shared_activation_shared_mechanism": (activation_shared & mechanism_shared).nonzero().flatten().tolist(),
        "shared_activation_concentrated_mechanism": (activation_shared & mechanism_concentrated).nonzero().flatten().tolist(),
        "concentrated_activation_shared_mechanism": (activation_concentrated & mechanism_shared).nonzero().flatten().tolist(),
        "concentrated_activation_concentrated_mechanism": (activation_concentrated & mechanism_concentrated).nonzero().flatten().tolist(),
        "mixed_or_subset_mass": (~covered).nonzero().flatten().tolist(),
    }
    # Interpretability sample: half strongly asymmetric components and half high-mass components.
    # The latter retains common mechanisms whose relative mass is intentionally near 1/N.
    # Never request more example features than C.
    max_features = min(cfg.analysis.max_example_features, c)
    # Reserve half for diff-like latents and half for high absolute mechanism mass.
    n_asymmetric = max_features // 2
    # Asymmetry score sums maximum P1 and P2 deviations from 1/N.
    # Example rho=[.9,.1],tilde-rho=[.8,.2] gives .4+.3=.7.
    asymmetry = (activation_rho - uniform).abs().amax(dim=0) + (
        mechanism_rho - uniform
    ).abs().amax(dim=0)
    # Total beta across n finds strong computations even when tilde-rho is shared.
    total_mass = beta.sum(dim=0)
    # First select the largest asymmetry scores.
    chosen = torch.topk(asymmetry, n_asymmetric).indices.tolist() if n_asymmetric else []
    for feature_id in torch.argsort(total_mass, descending=True).tolist():
        if feature_id not in chosen:
            # Fill remaining slots with strong beta mass not already selected for asymmetry.
            chosen.append(feature_id)
        if len(chosen) == max_features:
            break
    examples = _top_activation_examples(
        model,
        cache,
        cfg,
        device,
        torch.tensor(chosen, dtype=torch.long),
    )
    return {"tensors": tensors, "taxonomy": taxonomy, "examples": examples}


def save_analysis(result: dict[str, Any], output_dir: Path) -> None:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(result["tensors"], str(output_dir / "posthoc.safetensors"))
    (output_dir / "taxonomy.json").write_text(
        json.dumps(result["taxonomy"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "top_activation_examples.json").write_text(
        json.dumps(result["examples"], indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
