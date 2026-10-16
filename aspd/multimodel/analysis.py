"""Post-hoc P1--P5 analysis for activation and parameter-space diffing."""

from __future__ import annotations

import heapq
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
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
    feature_ids: Tensor | Sequence[int],
    *,
    chunk_size: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Keep contexts with the largest ``g^s_{t,c}`` for selected latent indices ``c``.

    These examples explain the common *when* of mechanism ``c``.  They are paired with P1/P2 in
    the saved analysis so a high activation score is never mistaken for parameter-space sharing.
    """

    accumulator = _TopActivationAccumulator(
        cfg,
        device,
        feature_ids,
        chunk_size=chunk_size,
    )
    for raw_batch in cache.iter_batches(
        "validation", cfg.training.batch_size_sequences, shuffle=False
    ):
        valid = raw_batch["valid_tokens"].to(device)
        activations = [r.to(device) for r in raw_batch["R"]]
        # Encode once for this validation batch: g^s has shape [B,T,C], including all requested c.
        values = model.encode(activations, valid).values
        ids: Tensor = raw_batch["input_ids"]  # type: ignore[assignment]
        accumulator.update(values, valid, ids)
    return accumulator.result()


def _validate_feature_ids(
    feature_ids: Tensor | Sequence[int], n_features: int
) -> list[int]:
    """Return unique requested c indices and fail before an expensive validation traversal."""

    raw_ids = (
        feature_ids.detach().cpu().tolist()
        if isinstance(feature_ids, Tensor)
        else list(feature_ids)
    )
    ids = [int(feature_id) for feature_id in raw_ids]
    if len(ids) != len(set(ids)):
        raise ValueError(f"component IDs must be unique, got {ids}")
    invalid = [feature_id for feature_id in ids if not 0 <= feature_id < n_features]
    if invalid:
        raise ValueError(
            f"component IDs out of range for C={n_features}: {invalid}; expected 0 <= id < {n_features}"
        )
    return ids


class _TopActivationAccumulator:
    """Streaming top contexts for requested c, updated from one encoded validation batch."""

    def __init__(
        self,
        cfg: MultiModelExperimentConfig,
        device: torch.device,
        feature_ids: Tensor | Sequence[int],
        *,
        chunk_size: int | None = None,
    ) -> None:
        self.k = cfg.analysis.top_examples_per_feature
        self.chunk_size = chunk_size or cfg.analysis.example_feature_chunk_size
        self.device = device
        ids = _validate_feature_ids(feature_ids, cfg.sparsity.n_features)
        self.selected_cpu = torch.tensor(ids, dtype=torch.long)
        self.heaps: dict[int, list[tuple[float, int, list[int], list[float], int, int]]] = {
            feature_id: [] for feature_id in ids
        }
        self.serial = 0

    def update(self, values: Tensor, valid: Tensor, input_ids: Tensor) -> None:
        """Update all requested c after exactly one encoder computation of g^s[B,T,C]."""

        for selected_chunk_cpu in self.selected_cpu.split(self.chunk_size):
            selected_chunk = selected_chunk_cpu.to(self.device)
            # [B,T,C_chunk] -> [B*T,C_chunk]; padding scores are zero and never retained.
            # Keep the measured g^s window as well as its maximum. Example for c=440:
            # selected_values[b, t-1:t+2, local_c]=[0.0, 3.25, 0.4].
            selected_values = values.index_select(-1, selected_chunk).masked_fill(
                ~valid[..., None], 0
            )
            scores = selected_values.reshape(-1, selected_chunk.numel())
            take = min(self.k, scores.shape[0])
            if take == 0:
                continue
            # Each column is one component c; top-k runs for the whole chunk at once.
            top_values, top_positions = torch.topk(scores, take, dim=0)
            # Preserve each measured g^s value's source dtype when converting it to JSON numbers.
            top_values_cpu = top_values.detach().cpu()
            top_positions_cpu = top_positions.detach().cpu()
            # One device-to-host transfer per feature chunk keeps all retained token windows exact
            # without synchronizing the GPU once for every candidate context.
            selected_values_cpu = selected_values.detach().float().cpu()
            for local_index, feature_id in enumerate(selected_chunk_cpu.tolist()):
                heap = self.heaps[feature_id]
                for score, flat_position in zip(
                    top_values_cpu[:, local_index].tolist(),
                    top_positions_cpu[:, local_index].tolist(),
                    strict=True,
                ):
                    # g^s<=0 is not positive. The extracted feature still remains as "c": [].
                    if score <= 0 or (len(heap) == self.k and score <= heap[0][0]):
                        continue
                    sequence = flat_position // input_ids.shape[1]
                    position = flat_position % input_ids.shape[1]
                    lo, hi = max(0, position - 16), min(input_ids.shape[1], position + 17)
                    item = (
                        float(score),
                        self.serial,
                        input_ids[sequence, lo:hi].tolist(),
                        selected_values_cpu[sequence, lo:hi, local_index].tolist(),
                        position - lo,
                        position,
                    )
                    self.serial += 1
                    if len(heap) < self.k:
                        heapq.heappush(heap, item)
                    else:
                        heapq.heapreplace(heap, item)

    def result(self) -> dict[str, list[dict[str, Any]]]:
        return {
            str(feature_id): [
                {
                    "g_s": score,
                    "token_ids": token_ids,
                    "token_activations": token_activations,
                    "center_in_window": center,
                    "position_in_sequence": position_in_sequence,
                }
                for score, _serial, token_ids, token_activations, center, position_in_sequence in sorted(
                    heap, reverse=True
                )
            ]
            for feature_id, heap in self.heaps.items()
        }


def select_example_feature_ids(
    cfg: MultiModelExperimentConfig,
    activation_rho: Tensor,
    mechanism_rho: Tensor,
    beta: Tensor,
) -> list[int]:
    """Resolve ranked/all/explicit context coverage without changing P1 or P2 definitions."""

    c = cfg.sparsity.n_features
    mode = cfg.analysis.example_feature_mode
    if mode == "all":
        return list(range(c))
    if mode == "explicit":
        return _validate_feature_ids(cfg.analysis.example_feature_ids, c)

    # Historical ranked mode: half by P1/P2 asymmetry, then fill by total beta mass.
    max_features = min(cfg.analysis.max_example_features, c)
    n_asymmetric = max_features // 2
    uniform = 1.0 / len(cfg.models)
    asymmetry = (activation_rho - uniform).abs().amax(dim=0) + (
        mechanism_rho - uniform
    ).abs().amax(dim=0)
    total_mass = beta.sum(dim=0)
    chosen = torch.topk(asymmetry, n_asymmetric).indices.tolist() if n_asymmetric else []
    chosen_set = set(chosen)
    for feature_id in torch.argsort(total_mass, descending=True).tolist():
        if feature_id not in chosen_set:
            chosen.append(feature_id)
            chosen_set.add(feature_id)
        if len(chosen) == max_features:
            break
    return chosen


def decode_activation_examples(
    examples: Mapping[str, list[dict[str, Any]]], cfg: MultiModelExperimentConfig
) -> None:
    """Add tokenizer-derived text while preserving measured g^s, token IDs, and center indices."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer, revision=cfg.data.tokenizer_revision
    )
    for feature_examples in examples.values():
        for example in feature_examples:
            token_ids = example["token_ids"]
            center = example["center_in_window"]
            example["text"] = tokenizer.decode(token_ids, skip_special_tokens=False)
            center_piece = tokenizer.convert_ids_to_tokens(token_ids[center])
            # Some Qwen reserved IDs are valid model IDs but have no tokenizer vocabulary string.
            # Keep the numeric identity explicit instead of emitting JSON null or dropping the token.
            example["center_token"] = (
                center_piece
                if isinstance(center_piece, str)
                else f"<unmapped_token_id:{token_ids[center]}>"
            )
            # Preserve exact tokenizer pieces for position-aligned UI; e.g. token_strings[t]
            # identifies the token at the same window index t as token_ids[t].
            pieces = tokenizer.convert_ids_to_tokens(token_ids)
            safe_pieces = [
                piece if isinstance(piece, str) else f"<unmapped_token_id:{token_id}>"
                for token_id, piece in zip(token_ids, pieces, strict=True)
            ]
            example["token_strings"] = safe_pieces
            # Decode each ID separately for readable whitespace/punctuation while retaining the
            # exact token string above for hover metadata and unusual BPE/special tokens.
            example["token_texts"] = [
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if isinstance(piece, str)
                else safe_piece
                for token_id, piece, safe_piece in zip(
                    token_ids, pieces, safe_pieces, strict=True
                )
            ]


def save_activation_examples(
    examples: Mapping[str, list[dict[str, Any]]], output_dir: Path, *, merge: bool
) -> dict[str, list[dict[str, Any]]]:
    """Atomically write contexts; merge mode updates requested c without deleting other keys."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "top_activation_examples.json"
    combined: dict[str, list[dict[str, Any]]] = {}
    if merge and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or any(not isinstance(value, list) for value in existing.values()):
            raise ValueError(f"cannot merge malformed activation-context artifact: {path}")
        combined.update(existing)
    combined.update(examples)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(combined, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return combined


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
    # cos(v_a,v_b); example dot .5 with unit norms gives .5.
    raw_read_cosine = v_dot / (v_a_norm * v_b_norm).clamp_min(eps)
    # cos(u_a,u_b); example .8 with unit norms gives .8.
    raw_write_cosine = u_dot / (u_a_norm * u_b_norm).clamp_min(eps)
    # Fix rank-1 sign gauge: if write cosine is negative, flip both reported signs for model b.
    orientation = torch.where(raw_write_cosine < 0, -1.0, 1.0)
    # ||P||_F=||u||||v|| for rank-1 P=uv^T.
    p_a_norm = u_a_norm * v_a_norm
    p_b_norm = u_b_norm * v_b_norm
    # Compute 1-cos(P_a,P_b) from distances between oriented unit factors.  This avoids the
    # catastrophic cancellation in ||P_a||^2+||P_b||^2-2<P_a,P_b> when P_a and P_b are equal or
    # nearly equal.  For unit vectors, .5||a-b||^2=1-cos(a,b).
    u_a_unit = u_a / u_a_norm.clamp_min(eps)[..., None]
    u_b_unit = u_b / u_b_norm.clamp_min(eps)[..., None] * orientation[..., None]
    v_a_unit = v_a / v_a_norm.clamp_min(eps)[..., None]
    v_b_unit = v_b / v_b_norm.clamp_min(eps)[..., None] * orientation[..., None]
    one_minus_write_cosine = 0.5 * (u_a_unit - u_b_unit).square().sum(-1)
    one_minus_read_cosine = 0.5 * (v_a_unit - v_b_unit).square().sum(-1)
    # 1-cos(P_a,P_b)=1-cos(u_a,u_b)cos(v_a,v_b).
    one_minus_component_cosine = (
        one_minus_write_cosine
        + one_minus_read_cosine
        - one_minus_write_cosine * one_minus_read_cosine
    ).clamp(0, 2)
    # Equivalent stable form:
    # ||P_a-P_b||^2=(||P_b||-||P_a||)^2+2||P_a||||P_b||(1-cos(P_a,P_b)).
    difference_sq = (
        (p_b_norm - p_a_norm).square()
        + 2 * p_a_norm * p_b_norm * one_minus_component_cosine
    ).clamp_min(0)
    return {
        # ||P_b-P_a||_F/||P_a||_F; 0 means identical parameter-space computation.
        "relative_component_change": difference_sq.sqrt() / p_a_norm.clamp_min(eps),
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

    # all/explicit know their c indices before P1/P2, so collect contexts during this same pass.
    # ranked must wait for rho_c^(n), tilde-rho_c^(n), and beta_c^(n), then make one extra pass.
    early_example_ids: list[int] | None = None
    if cfg.analysis.example_feature_mode == "all":
        early_example_ids = list(range(c))
    elif cfg.analysis.example_feature_mode == "explicit":
        early_example_ids = _validate_feature_ids(cfg.analysis.example_feature_ids, c)
    example_accumulator = (
        _TopActivationAccumulator(cfg, device, early_example_ids)
        if early_example_ids is not None
        else None
    )

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
        if example_accumulator is not None:
            # Reuse this g^s[B,T,C] to update top contexts; do not call encode again per c.
            input_ids: Tensor = raw_batch["input_ids"]  # type: ignore[assignment]
            example_accumulator.update(code.values, valid, input_ids)
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
    if example_accumulator is not None:
        # Every requested c is a present JSON key, including c with no positive g^s as "c": [].
        examples = example_accumulator.result()
    else:
        # ranked reproduces the historical capped selection after P1/P2 statistics are available.
        chosen = select_example_feature_ids(cfg, activation_rho, mechanism_rho, beta)
        examples = _top_activation_examples(model, cache, cfg, device, chosen)
    return {"tensors": tensors, "taxonomy": taxonomy, "examples": examples}


def save_analysis(result: dict[str, Any], output_dir: Path) -> None:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(result["tensors"], str(output_dir / "posthoc.safetensors"))
    (output_dir / "taxonomy.json").write_text(
        json.dumps(result["taxonomy"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_activation_examples(result["examples"], output_dir, merge=False)
