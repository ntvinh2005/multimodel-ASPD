"""Live P6 intervention: ablate selected mechanisms in one model on the clean shared gate."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from aspd.multimodel.cache import PairedActivationCache, _tensor_from_value, load_effective_model
from aspd.multimodel.config import ModelSpec, MultiModelExperimentConfig
from aspd.multimodel.model import MultiModelASPD

# P6 example: ablate one c in model n. If g_t,c=1, v_j,c^T x_j,t=2, and
# u_j,c=[1,3], the hook subtracts 2*u_j,c=[2,6] from W_j x_j,t.


@dataclass(frozen=True)
class ThresholdGateCalibration:
    """Label-free component thresholds for an online, prefix-causal gate.

    The original training gate is BatchTopK and therefore pools future positions and
    other sequences in its batch.  P6 generation cannot use that support without
    temporal leakage.  For the current linear encoder, ``a[t,c]`` itself is causal, so
    fixed thresholds calibrated on independent general-validation activations give the
    explicit online policy ``g[t,c] = 1[a[t,c] > threshold[c]]``.

    This is an approximation to offline BatchTopK, not a mathematically identical
    replacement.  The P6 localization artifact reports their overlap on frozen V3.
    """

    component_ids: tuple[int, ...]
    thresholds: Tensor
    target_rates: Tensor
    realized_rates: Tensor
    calibration_tokens: int


@dataclass(frozen=True)
class P6ForwardResult:
    """One isolated fixed-sequence P6 forward and its perturbation audit."""

    logits: Tensor
    matrix_diagnostics: Mapping[str, Mapping[str, Any]]


DEFAULT_RANDOM_CONTROL_SEEDS = (0, 1, 2, 3, 4)


def validate_random_control_seeds(seeds: Sequence[int] | None) -> tuple[int, ...]:
    """Validate an opt-in random-control seed list without enabling it implicitly."""

    if seeds is None:
        return ()
    normalized = tuple(seeds)
    if not normalized:
        raise ValueError("random-control seeds must be non-empty when controls are enabled")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in normalized):
        raise TypeError("random-control seeds must be integers")
    if any(seed < 0 or seed >= 2**63 for seed in normalized):
        raise ValueError("random-control seeds must lie in [0, 2**63)")
    if len(normalized) != len(set(normalized)):
        raise ValueError("random-control seeds must be unique")
    return normalized


def calibrate_threshold_gate(
    selected_preactivations: Tensor,
    component_ids: Sequence[int],
    target_rates: Tensor,
) -> ThresholdGateCalibration:
    """Choose per-component thresholds matching independent offline firing rates.

    ``selected_preactivations`` is ``[token, selected_component]`` and must come from
    an unlabeled calibration set.  A threshold is placed immediately below the kth
    largest positive preactivation so strict ``>`` realizes the requested rate when
    there are no ties.  ReLU zeros are never promoted merely to meet a target count.
    """

    if selected_preactivations.ndim != 2:
        raise ValueError("selected_preactivations must have shape [token, component]")
    ids = tuple(int(value) for value in component_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("component_ids must be non-empty and unique")
    if selected_preactivations.shape[1] != len(ids):
        raise ValueError("component_ids must match the selected component axis")
    rates = target_rates.detach().float().cpu()
    if tuple(rates.shape) != (len(ids),):
        raise ValueError("target_rates must have shape [selected_component]")
    if not torch.isfinite(rates).all() or bool(((rates < 0) | (rates > 1)).any()):
        raise ValueError("target_rates must be finite probabilities")

    values = selected_preactivations.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise ValueError("selected_preactivations must be finite")
    token_count = values.shape[0]
    if token_count == 0:
        raise ValueError("threshold calibration requires at least one token")
    thresholds = torch.full((len(ids),), float("inf"), dtype=torch.float32)
    for local_index in range(len(ids)):
        desired = min(token_count, max(0, round(float(rates[local_index]) * token_count)))
        positive = values[:, local_index][values[:, local_index] > 0]
        desired = min(desired, int(positive.numel()))
        if desired == 0:
            continue
        kth = torch.topk(positive, desired, sorted=True).values[-1]
        # Strict `>` is the declared online rule. Move one representable FP32 value
        # below kth so the kth observation is retained when values are unique.
        thresholds[local_index] = torch.nextafter(
            kth, torch.tensor(float("-inf"), dtype=kth.dtype)
        )
    realized = (values > thresholds[None, :]).float().mean(dim=0)
    return ThresholdGateCalibration(
        component_ids=ids,
        thresholds=thresholds,
        target_rates=rates,
        realized_rates=realized,
        calibration_tokens=token_count,
    )


def threshold_causal_gate(
    preactivations: Tensor,
    calibration: ThresholdGateCalibration,
) -> Tensor:
    """Apply the fixed prefix-causal gate and return ``[B,T,selected_component]``."""

    if preactivations.ndim != 3:
        raise ValueError("preactivations must have shape [B,T,C]")
    selected = torch.tensor(
        calibration.component_ids, dtype=torch.long, device=preactivations.device
    )
    if selected.numel() and int(selected.max()) >= preactivations.shape[-1]:
        raise ValueError("calibrated component is outside the preactivation feature axis")
    thresholds = calibration.thresholds.to(preactivations.device, preactivations.dtype)
    return preactivations.index_select(-1, selected) > thresholds


def temporal_region(position: int, prompt_length: int, sequence_length: int) -> str:
    """Map one exact token position into a non-overlapping P6 localization region.

    The last prompt token is isolated as the autoregressive decision boundary.  The
    first eight assistant tokens are the response opening.  Remaining response tokens
    are divided into early/mid/conclusion thirds without overlapping positions.
    """

    if not 0 <= position < sequence_length:
        raise ValueError("position must lie inside the sequence")
    if not 1 <= prompt_length <= sequence_length:
        raise ValueError("prompt_length must lie inside the sequence")
    if position < prompt_length - 1:
        return "user_prompt"
    if position == prompt_length - 1:
        return "decision_boundary"
    response_index = position - prompt_length
    response_length = sequence_length - prompt_length
    opening = min(8, response_length)
    if response_index < opening:
        return "response_opening"
    remaining = response_length - opening
    if remaining <= 0:
        return "response_opening"
    relative = response_index - opening
    first_stop = (remaining + 2) // 3
    second_stop = (2 * remaining + 2) // 3
    if relative < first_stop:
        return "response_early"
    if relative < second_stop:
        return "response_mid"
    return "response_conclusion"


def norm_matched_orthogonal_rows(rows: Tensor) -> Tensor:
    """Return deterministic row-wise orthogonal directions with identical norms.

    The construction uses cyclic shifts rather than random draws, so the control is
    reproducible without a hidden RNG dependency.  It controls the output-vector norm;
    the candidate read vector and clean causal gate remain unchanged.
    """

    if rows.ndim != 2:
        raise ValueError("rows must have shape [component, output_dim]")
    reference = rows.float()
    denominator = reference.square().sum(dim=1, keepdim=True)
    if not torch.isfinite(reference).all() or bool((denominator <= 0).any()):
        raise ValueError("every reference row must be finite and non-zero")
    for shift in range(1, min(8, rows.shape[1])):
        candidate = reference.roll(shift, dims=1)
        candidate = candidate - (
            (candidate * reference).sum(dim=1, keepdim=True) / denominator
        ) * reference
        candidate_norm = candidate.norm(dim=1, keepdim=True)
        if bool((candidate_norm > torch.finfo(candidate.dtype).eps).all()):
            matched = candidate * (reference.norm(dim=1, keepdim=True) / candidate_norm)
            return matched.to(rows.dtype)
    raise ValueError("could not construct non-degenerate orthogonal control rows")


def norm_matched_random_rows(
    rows: Tensor,
    *,
    model_name: str,
    matrix_name: str,
    matrix_module: str,
    component_ids: Sequence[int],
    seed: int,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Sample reproducible random write directions and match each learned ``u`` norm."""

    if rows.ndim != 2:
        raise ValueError("rows must have shape [component, output_dim]")
    ids = tuple(int(component_id) for component_id in component_ids)
    if len(ids) != rows.shape[0] or len(ids) != len(set(ids)):
        raise ValueError("component_ids must uniquely match the row axis")
    validated_seed = validate_random_control_seeds([seed])[0]
    if not model_name or not matrix_name or not matrix_module:
        raise ValueError("model and matrix identities must be non-empty")
    reference = rows.detach().float().cpu()
    reference_norms = reference.norm(dim=1)
    if not torch.isfinite(reference).all() or not torch.isfinite(reference_norms).all():
        raise ValueError("every reference row and norm must be finite")
    if bool((reference_norms <= 0).any()):
        raise ValueError("every reference row must have a positive L2 norm")
    layer_match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", matrix_module)
    layer_index = int(layer_match.group(1)) if layer_match else None
    matched_rows: list[Tensor] = []
    metadata: list[dict[str, Any]] = []
    for local_index, component_id in enumerate(ids):
        identity = (
            "aspd-random-write-v1"
            f"|model={model_name}|matrix={matrix_name}|module={matrix_module}"
            f"|component={component_id}|seed={validated_seed}"
        )
        derived_seed = int.from_bytes(
            hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big"
        ) & (2**63 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derived_seed)
        sampled = torch.randn(reference.shape[1], generator=generator, dtype=torch.float32)
        sampled_norm = sampled.norm()
        if not torch.isfinite(sampled).all() or not torch.isfinite(sampled_norm):
            raise FloatingPointError("random write direction or norm is NaN or Inf")
        if float(sampled_norm) <= 0.0:
            raise FloatingPointError("random write direction has zero L2 norm")
        matched = sampled * (reference_norms[local_index] / sampled_norm)
        matched_rows.append(matched)
        random_norm = matched.norm()
        norm_error = (random_norm - reference_norms[local_index]).abs()
        cosine = torch.nn.functional.cosine_similarity(
            matched[None, :], reference[local_index : local_index + 1], dim=1
        )[0]
        if not torch.isfinite(matched).all() or not torch.isfinite(cosine):
            raise FloatingPointError("matched random write direction is NaN or Inf")
        metadata.append(
            {
                "candidate_component_id": component_id,
                "model_name": model_name,
                "layer_index": layer_index,
                "matrix_name": matrix_name,
                "matrix_module": matrix_module,
                "random_seed": validated_seed,
                "derived_torch_seed": derived_seed,
                "reference_vector_l2": float(reference_norms[local_index]),
                "random_vector_l2": float(random_norm),
                "norm_matching_error": float(norm_error),
                "cosine_similarity_to_learned_direction": float(cosine),
                "construction": "random write u with learned v and same u L2 norm",
            }
        )
    output = torch.stack(matched_rows).to(device=rows.device, dtype=rows.dtype)
    if not torch.isfinite(output).all():
        raise FloatingPointError("device-cast random write direction is NaN or Inf")
    return output, metadata


def p6_subtraction(
    x: Tensor,
    u: Tensor,
    v: Tensor,
    gate: Tensor,
    *,
    strength: float = 1.0,
    input_center: Tensor | None = None,
) -> Tensor:
    """Compute ``strength * sum_c g_tc (v_c^T x_t) u_c``.

    Keeping this pure tensor operation separate makes the scientific equation directly
    testable without loading a language model.  Computation is FP32 and callers cast the
    result to the live matrix output dtype only at the subtraction boundary.
    """

    if x.ndim != 3 or u.ndim != 2 or v.ndim != 2 or gate.ndim != 3:
        raise ValueError("expected x/gate [B,T,*] and u/v [component,dimension]")
    if x.shape[:2] != gate.shape[:2]:
        raise ValueError("gate token grid must match the live matrix input")
    if u.shape[0] != v.shape[0] or gate.shape[2] != u.shape[0]:
        raise ValueError("component axes of gate, u, and v must match")
    if x.shape[2] != v.shape[1]:
        raise ValueError("live matrix input width must match v")
    x32, u32, v32 = x.float(), u.float(), v.float()
    if input_center is not None:
        if tuple(input_center.shape) != (x.shape[2],):
            raise ValueError("component input center must have shape [d_in]")
        x32 = x32 - input_center.to(device=x.device, dtype=torch.float32)
    reads = torch.einsum("btd,kd->btk", x32, v32)
    effective = reads * gate.to(reads.dtype)
    return float(strength) * torch.einsum("btk,ko->bto", effective, u32)


@torch.inference_mode()
def p6_patched_forward(
    model: nn.Module,
    spec: ModelSpec,
    decomposition: MultiModelASPD,
    model_index: int,
    input_ids_cpu: Tensor,
    valid_cpu: Tensor,
    gate_cpu: Tensor,
    component_ids: Sequence[int],
    *,
    strength: float = 1.0,
    norm_control: bool = False,
    random_control_seed: int | None = None,
) -> P6ForwardResult:
    """Run one full-sequence P6 arm with hooks that are removed unconditionally.

    ``gate_cpu`` must already be derived from the clean Base/Target pair.  The hooks use
    each matrix's actual live input, so interventions at earlier sites may propagate to
    later sites.  This function never mutates Qwen or ASPD parameters.
    """

    if not component_ids or len(component_ids) != len(set(component_ids)):
        raise ValueError("component_ids must be non-empty and unique")
    if norm_control and random_control_seed is not None:
        raise ValueError("orthogonal and random write-direction controls are incompatible")
    if not torch.isfinite(torch.tensor(float(strength))):
        raise ValueError("strength must be finite")
    if random_control_seed is not None:
        validate_random_control_seeds([random_control_seed])
    if tuple(gate_cpu.shape[:2]) != tuple(input_ids_cpu.shape) or gate_cpu.shape[2] != len(
        component_ids
    ):
        raise ValueError("gate must have shape [B,T,len(component_ids)]")
    device = next(model.parameters()).device
    input_ids = input_ids_cpu.detach().clone().to(device)
    valid = valid_cpu.detach().clone().bool().to(device)
    gate = gate_cpu.detach().clone().bool().to(device)
    selected = torch.tensor(
        component_ids, dtype=torch.long, device=next(decomposition.parameters()).device
    )
    diagnostics: dict[str, dict[str, Any]] = {}

    with ExitStack() as stack:
        for matrix in spec.matrices:
            u_all, v_all = decomposition.component_factors(model_index, matrix.name)
            center_getter = getattr(decomposition, "component_input_center", None)
            center = (
                center_getter(model_index, matrix.name)
                if center_getter is not None
                else None
            )
            u = u_all.index_select(0, selected)
            v = v_all.index_select(0, selected)
            if norm_control:
                u = norm_matched_orthogonal_rows(u)
            random_metadata: list[dict[str, Any]] = []
            if random_control_seed is not None:
                u, random_metadata = norm_matched_random_rows(
                    u,
                    model_name=spec.name,
                    matrix_name=matrix.name,
                    matrix_module=matrix.module,
                    component_ids=component_ids,
                    seed=random_control_seed,
                )

            def subtract_components(
                _module: nn.Module,
                args: tuple[object, ...],
                output: object,
                *,
                u_local: Tensor = u,
                v_local: Tensor = v,
                site: str = matrix.module,
                name: str = matrix.name,
                random_metadata_local: list[dict[str, Any]] = random_metadata,
                center_local: Tensor | None = center,
            ) -> Tensor:
                x = _tensor_from_value(args, 0, site)
                y = _tensor_from_value(output, None, site)
                subtraction = p6_subtraction(
                    x,
                    u_local,
                    v_local,
                    gate,
                    strength=strength,
                    input_center=center_local,
                )
                selected_values = subtraction[valid]
                off_values = subtraction[(~gate.any(dim=-1)) & valid]
                diagnostics[name] = {
                    "valid_token_count": int(valid.sum().item()),
                    "gate_active_token_count": int((gate.any(dim=-1) & valid).sum().item()),
                    "subtraction_element_count": int(selected_values.numel()),
                    "subtraction_l2": float(selected_values.square().sum().sqrt().item()),
                    "subtraction_rms": float(
                        selected_values.square().mean().sqrt().item()
                    ),
                    "max_abs_subtraction": float(selected_values.abs().max().item()),
                    "max_abs_gate_off_subtraction": float(
                        off_values.abs().max().item() if off_values.numel() else 0.0
                    ),
                }
                if random_metadata_local:
                    diagnostics[name]["random_direction_controls"] = random_metadata_local
                return y - subtraction.to(y.dtype)

            handle = model.get_submodule(matrix.module).register_forward_hook(
                subtract_components
            )
            stack.callback(handle.remove)
        result = model(input_ids=input_ids, attention_mask=valid, use_cache=False)

    expected = {matrix.name for matrix in spec.matrices}
    if set(diagnostics) != expected:
        raise RuntimeError("one or more P6 matrix hooks did not fire")
    if any(values["max_abs_gate_off_subtraction"] != 0.0 for values in diagnostics.values()):
        raise RuntimeError("P6 changed a gate-off token")
    logits = result.logits.detach()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("P6 forward produced NaN or Inf logits")
    return P6ForwardResult(logits=logits, matrix_diagnostics=diagnostics)


def _load_hf_model(spec: ModelSpec, device: torch.device) -> nn.Module:
    model, _provenance = load_effective_model(spec, device)
    return model


@torch.no_grad()
def causal_ablation(
    decomposition: MultiModelASPD,
    cache: PairedActivationCache,
    cfg: MultiModelExperimentConfig,
    input_ids: Tensor,
    valid_tokens: Tensor,
    target_model_index: int,
    component_ids: Sequence[int],
    device: torch.device,
) -> dict[str, Tensor]:
    """Apply ``W_jx -> W_jx - sum_c g_tc P_jc x`` for every selected ``j,c``.

    The shared gate is computed from clean ``R^(1),...,R^(N)``.  Keeping that gate fixed makes the
    intervention exactly the P6 operation and avoids a second-order change in feature selection.
    """

    # Load all M^(n) so the clean shared gate can read R^(1),...,R^(N).
    models = [_load_hf_model(spec, device) for spec in cfg.models]
    clean_logits: list[Tensor] = []
    grounding: list[Tensor] = []
    with ExitStack() as stack:
        for n, (hf_model, spec) in enumerate(zip(models, cfg.models, strict=True)):
            holder: dict[str, Tensor] = {}
            module = hf_model.get_submodule(spec.grounding.module)
            if spec.grounding.capture == "input":
                stack.callback(
                    module.register_forward_pre_hook(
                        lambda _module, args, h=holder, s=spec: h.update(
                            R=_tensor_from_value(
                                args, s.grounding.tensor_index or 0, s.grounding.module
                            ).detach()
                        )
                    ).remove
                )
            else:
                stack.callback(
                    module.register_forward_hook(
                        lambda _module, _args, output, h=holder, s=spec: h.update(
                            R=_tensor_from_value(
                                output, s.grounding.tensor_index, s.grounding.module
                            ).detach()
                        )
                    ).remove
                )
            result = hf_model(
                input_ids=input_ids.to(device),
                attention_mask=valid_tokens.to(device),
                use_cache=False,
            )
            # Store z_clean^(n)[B,T,V] before any P_j,c intervention.
            clean_logits.append(result.logits.detach())
            if decomposition.original_aspd_cfg.enabled:
                grounding.append(holder["R"])
                continue
            # Apply the same q_n used in training: R_scaled^(n)=R_raw^(n)/r_rms_norm_n.
            # Example ||r|| RMS=2 maps r=[2,0] to [1,0].
            grounding.append(holder["R"] / cache.model_manifests[n]["r_rms_norm"])

    # Compute clean g^s and g jointly from every normalized R^(n), once.
    code = decomposition.encode(grounding, valid_tokens.to(device))
    # Selected global latent indices c; example A={17,203}.
    selected = torch.tensor(component_ids, dtype=torch.long, device=device)
    # Fixed clean binary gates g_t,c for c in A, shape [B,T,|A|].
    selected_gate = code.gate[..., selected]
    target_hf_model = models[target_model_index]
    with ExitStack() as stack:
        for matrix in cfg.models[target_model_index].matrices:
            u, v = decomposition.component_factors(target_model_index, matrix.name)
            center = decomposition.component_input_center(
                target_model_index, matrix.name
            )
            # U_A contains u_j,c^(n) rows, shape [|A|,d_out].
            u_selected = u[selected]
            # V_A contains v_j,c^(n) rows, shape [|A|,d_in].
            v_selected = v[selected]

            def subtract_components(
                _module: nn.Module,
                args: tuple[object, ...],
                output: object,
                u_local: Tensor = u_selected,
                v_local: Tensor = v_selected,
                site: str = matrix.module,
                center_local: Tensor | None = center,
            ) -> Tensor:
                # Live X_j^(n)[B,T,d_in] entering W_j^(n).
                x = _tensor_from_value(args, 0, site).to(v_local.dtype)
                if center_local is not None:
                    x = x - center_local.to(x)
                # Clean module output Y_j^(n)=W_j^(n)X_j^(n).
                y = _tensor_from_value(output, None, site)
                # reads_t,c=v_j,c^(n)T x_j,t^(n). Example x=[2,1],v=[1,0] -> 2.
                reads = torch.einsum("btd,kd->btk", x, v_local)
                # e_j,t,c^(n)=g_t,c*read_t,c; inactive clean gates zero the intervention.
                effective = reads * selected_gate.to(reads.dtype)
                # sum_c e_j,t,c^(n)u_j,c^(n), shape [B,T,d_out].
                # Example e=2,u=[1,3] produces subtraction [2,6].
                subtraction = torch.einsum("btk,ko->bto", effective, u_local)
                # P6 patched output: W_j x_j,t - sum_c g_t,c P_j,c x_j,t.
                return y - subtraction.to(y.dtype)

            stack.callback(
                target_hf_model.get_submodule(matrix.module)
                .register_forward_hook(subtract_components)
                .remove
            )
        patched_logits = target_hf_model(
            input_ids=input_ids.to(device),
            attention_mask=valid_tokens.to(device),
            use_cache=False,
        ).logits

    # Compare logits for the intervened target n in FP32.
    clean = clean_logits[target_model_index].float()
    patched = patched_logits.float()
    # log p_clean(v|s_<=t) over vocabulary v.
    clean_log_prob = clean.log_softmax(dim=-1)
    # log p_patched(v|s_<=t) after subtracting selected mechanisms.
    patched_log_prob = patched.log_softmax(dim=-1)
    # KL_t=sum_v p_clean(v)[log p_clean(v)-log p_patched(v)].
    # Example p=[.5,.5],q=[.9,.1] gives a positive behavioral change.
    kl = (clean_log_prob.exp() * (clean_log_prob - patched_log_prob)).sum(dim=-1)
    return {
        "clean_logits": clean.cpu(),
        "patched_logits": patched.cpu(),
        "kl_per_token": kl.cpu(),
        # Mean causal effect over valid t only.
        "mean_kl": kl[valid_tokens.to(device)].mean().cpu(),
        # Largest individual vocabulary-logit movement anywhere in [B,T,V].
        "max_abs_logit_change": (patched - clean).abs().max().cpu(),
    }
