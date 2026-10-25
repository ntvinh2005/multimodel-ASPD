"""Held-out evaluation for the streamed two-model original-ASPD checkpoint."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.batch_topk import batch_topk
from aspd.multimodel.cache import (
    _Capture,
    _register_capture_hooks,
    load_effective_model,
    snapshot_model_weights,
)
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.delta_contrastive import canonical_prompt_ids
from aspd.multimodel.fidelity import (
    GlobalTensorMetrics,
    IdentityResidual,
    PairedErrorMetrics,
    _component_reconstruction,
    sha256_file,
    write_json_immutable,
)
from aspd.multimodel.model import MultiModelASPD
from aspd.multimodel.original_recipe import (
    sha256_canonical_text,
    validate_dataset_manifest,
    validate_original_recipe_config,
)
from aspd.multimodel.original_streaming import (
    FrozenTokenShardStream,
    _capture_pair,
    resolved_config_sha256,
)


class EvaluationGroup:
    """Exact FP64 sufficient statistics for one held-out provenance group."""

    def __init__(self, width: int):
        self.residual = {
            "base": GlobalTensorMetrics(width),
            "target": GlobalTensorMetrics(width),
        }
        self.matrix = {
            label: GlobalTensorMetrics(width)
            for label in (
                "base",
                "target",
                "delta_total",
                "delta_parameter",
                "delta_input",
            )
        }
        self.identity = {"true": IdentityResidual(), "aspd": IdentityResidual()}
        self.paired = PairedErrorMetrics()
        self.tokens = 0

    @torch.no_grad()
    def update(
        self,
        model: MultiModelASPD,
        residuals: list[Tensor],
        inputs: list[Tensor],
        values: Tensor,
        gate: Tensor,
        valid: Tensor,
    ) -> None:
        if not valid.any():
            return
        residual_hat = [
            model.activation_decoders.reconstruct(values, index, valid).float()
            for index in (0, 1)
        ]
        for index, name in enumerate(("base", "target")):
            self.residual[name].update(residuals[index], residual_hat[index], valid)
        y = [model._target(index, "o_proj").apply(inputs[index]).float() for index in (0, 1)]
        y_hat = [
            _component_reconstruction(model, index, "o_proj", inputs[index], gate, valid).float()
            for index in (0, 1)
        ]
        self.matrix["base"].update(y[0], y_hat[0], valid)
        self.matrix["target"].update(y[1], y_hat[1], valid)
        self.paired.update(y[0], y_hat[0], y[1], y_hat[1], valid)
        delta_total = y[1] - y[0]
        delta_total_hat = y_hat[1] - y_hat[0]
        delta_parameter = torch.nn.functional.linear(
            inputs[0],
            model._target(1, "o_proj").weight.float()
            - model._target(0, "o_proj").weight.float(),
        )
        target_on_base_hat = _component_reconstruction(
            model, 1, "o_proj", inputs[0], gate, valid
        ).float()
        delta_parameter_hat = target_on_base_hat - y_hat[0]
        delta_input = model._target(1, "o_proj").apply(inputs[1] - inputs[0]).float()
        delta_input_hat = y_hat[1] - target_on_base_hat
        self.matrix["delta_total"].update(delta_total, delta_total_hat, valid)
        self.matrix["delta_parameter"].update(
            delta_parameter, delta_parameter_hat, valid
        )
        self.matrix["delta_input"].update(delta_input, delta_input_hat, valid)
        self.identity["true"].update(
            delta_total, delta_parameter, delta_input, valid
        )
        self.identity["aspd"].update(
            delta_total_hat, delta_parameter_hat, delta_input_hat, valid
        )
        self.tokens += int(valid.sum())

    def result(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "residual_activation": {
                name: value.result() for name, value in self.residual.items()
            },
            "o_proj_computation": {
                name: value.result() for name, value in self.matrix.items()
            },
            "decomposition_identity": {
                name: value.result() for name, value in self.identity.items()
            },
            "paired_error": self.paired.result(),
        }


def _load_checkpoint_pair(
    model: MultiModelASPD, rank0: Path, rank1: Path | None
) -> dict[str, Any]:
    state = torch.load(rank0, map_location="cpu", weights_only=False, mmap=True)
    if "model" not in state or "step" not in state:
        raise ValueError("rank-0 checkpoint lacks model or step")
    model.load_state_dict(state["model"], strict=True)
    if rank1 is not None:
        other = torch.load(rank1, map_location="cpu", weights_only=False, mmap=True)
        if int(other["step"]) != int(state["step"]):
            raise ValueError("training-rank checkpoints have different steps")
        parameter_names = {name for name, _ in model.named_parameters()}
        unequal = [
            name
            for name in parameter_names
            if not torch.equal(state["model"][name], other["model"][name])
        ]
        if unequal:
            raise ValueError(f"DDP trainable parameters differ across ranks: {unequal[:5]}")
    return {"step": int(state["step"]), "config_sha256": state["config_sha256"]}


def _summarize(values: Tensor) -> dict[str, float]:
    value = values.detach().double().cpu()
    return {
        "min": float(value.min()),
        "mean": float(value.mean()),
        "max": float(value.max()),
        "rms": float(value.square().mean().sqrt()),
    }


def _component_contribution_squared(
    model: MultiModelASPD,
    model_index: int,
    inputs: Tensor,
    gate: Tensor,
    valid: Tensor,
) -> Tensor:
    u, v = model.component_factors(model_index, "o_proj")
    component = model.components[model_index]["o_proj"]
    flat_gate = gate.reshape(-1, gate.shape[-1])
    flat_valid = valid.reshape(-1)
    positions, features = (flat_gate & flat_valid[:, None]).nonzero(as_tuple=True)
    result = torch.zeros(gate.shape[-1], dtype=torch.float64, device=gate.device)
    if features.numel() == 0:
        return result
    selected = inputs.reshape(-1, inputs.shape[-1])[positions].to(v.dtype)
    if component.b_dec is not None:
        selected = selected - component.b_dec
    reads = (selected * v[features]).sum(dim=-1).double()
    write_norm_squared = u[features].double().square().sum(dim=-1)
    result.index_add_(0, features, reads.square() * write_norm_squared)
    return result


def _behavioral_prompts(cfg: MultiModelExperimentConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = Path(cfg.data.locked_evaluation_manifest or "")
    if not manifest_path.is_file():
        return [], {"status": "NOT_VERIFIED", "reason": f"missing {manifest_path}"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_models = {
        "base": {"id": cfg.models[0].model_id, "revision": cfg.models[0].revision},
        "target": {"id": cfg.models[1].model_id, "revision": cfg.models[1].revision},
    }
    if manifest.get("models") != expected_models:
        raise ValueError("locked behavioral manifest belongs to a different model pair")
    if manifest.get("tokenizer") != {
        "id": cfg.data.tokenizer,
        "revision": cfg.data.tokenizer_revision,
    }:
        raise ValueError("locked behavioral manifest belongs to a different tokenizer")
    prompt_path = Path(manifest["prompt_file"])
    if not prompt_path.is_absolute():
        candidate = manifest_path.parent / prompt_path
        prompt_path = candidate if candidate.is_file() else Path.cwd() / prompt_path
    if (
        not prompt_path.is_file()
        or sha256_canonical_text(prompt_path) != manifest["prompt_sha256"]
    ):
        raise ValueError("locked behavioral prompt artifact is missing or has the wrong checksum")
    rows = [
        json.loads(line)
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = manifest.get("audit", {}).get("category_counts")
    observed = {
        category: sum(row.get("category") == category for row in rows)
        for category in ("risk", "sports_control", "neutral")
    }
    if expected != observed:
        raise ValueError("locked behavioral prompt category counts differ from the manifest")
    return rows, {
        "status": "VERIFIED",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "prompt_file": str(prompt_path.resolve()),
        "prompt_sha256": sha256_canonical_text(prompt_path),
        "category_counts": observed,
        "expected_matched_pair_count": manifest.get("audit", {}).get(
            "matched_pair_count"
        ),
    }


def _pad_prompts(rows: list[list[int]], pad_token_id: int) -> tuple[Tensor, Tensor]:
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    valid = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, values in enumerate(rows):
        ids[index, width - len(values):] = torch.tensor(values, dtype=torch.long)
        valid[index, width - len(values):] = True
    return ids, valid


@torch.inference_mode()
def evaluate_original_recipe(
    cfg: MultiModelExperimentConfig,
    checkpoint_rank0: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_rank1: str | Path | None = None,
    device_name: str = "cuda",
    batch_size_sequences: int = 8,
    max_validation_sequences: int | None = None,
) -> dict[str, Any]:
    """Run independent held-out evaluation without changing source-model weights."""

    validate_original_recipe_config(cfg)
    dataset = validate_dataset_manifest(cfg, require_ready=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest_path = Path(dataset["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    role_by_id = {
        int(key): row["role"] for key, row in manifest["provenance_table"].items()
    }
    stream = FrozenTokenShardStream(manifest_path, "validation")
    models = []
    captures = []
    weights = []
    provenance = []
    before = []
    for spec in cfg.models:
        source, source_provenance = load_effective_model(spec, device)
        source.requires_grad_(False)
        snapshot, _ = snapshot_model_weights(source, spec, source_provenance)
        models.append(source)
        captures.append(_Capture())
        weights.append(snapshot)
        before.append(snapshot)
        provenance.append(source_provenance)
    width = int(weights[0]["o_proj"].shape[0])
    aspd = MultiModelASPD(
        [model.name for model in cfg.models],
        [width, width],
        weights,
        cfg.encoder,
        cfg.sparsity,
        cfg.objective,
        cfg.original_aspd,
    )
    checkpoint_record = _load_checkpoint_pair(
        aspd,
        Path(checkpoint_rank0),
        Path(checkpoint_rank1) if checkpoint_rank1 else None,
    )
    if checkpoint_record["config_sha256"] != resolved_config_sha256(cfg):
        raise ValueError("checkpoint scientific configuration fingerprint differs")
    aspd = aspd.to(device).float().eval()
    groups = {name: EvaluationGroup(width) for name in ["all", *sorted(set(role_by_id.values()))]}
    fire_counts = torch.zeros(cfg.sparsity.n_features, dtype=torch.long)
    contribution_squared = {
        name: torch.zeros(cfg.sparsity.n_features, dtype=torch.float64)
        for name in ("base", "target")
    }
    valid_count = 0
    jump_selected = 0
    topk_selected = 0
    support_intersection = 0
    with ExitStack() as stack:
        for source, spec, capture in zip(models, cfg.models, captures, strict=True):
            _register_capture_hooks(source, spec, capture, stack)
        evaluation_sequence_count = (
            stream.sequence_count
            if max_validation_sequences is None
            else min(stream.sequence_count, max_validation_sequences)
        )
        if evaluation_sequence_count <= 0:
            raise ValueError("held-out evaluation requires at least one validation sequence")
        for start in range(0, evaluation_sequence_count, batch_size_sequences):
            count = min(batch_size_sequences, evaluation_sequence_count - start)
            tokens = stream.batch_at(start, count)
            paired = _capture_pair(
                models,
                captures,
                tokens["input_ids"],
                tokens["valid_tokens"],
                device,
                sequence_ids=tokens["sequence_ids"],
                provenance_ids=tokens["provenance_ids"],
            )
            valid = paired["valid_tokens"]
            residuals = [value.float() for value in paired["R"]]
            inputs = [value["o_proj"].float() for value in paired["X"]]
            preactivations = aspd.encoder(residuals, valid)
            jump = aspd.encode(residuals, valid, update_statistics=False)
            topk = batch_topk(preactivations, valid, cfg.sparsity)
            fire_counts += jump.gate[valid].sum(0).long().cpu()
            valid_count += int(valid.sum())
            jump_selected += int(jump.gate[valid].sum())
            topk_selected += int(topk.gate[valid].sum())
            support_intersection += int((jump.gate & topk.gate)[valid].sum())
            for model_index, name in enumerate(("base", "target")):
                contribution_squared[name] += _component_contribution_squared(
                    aspd, model_index, inputs[model_index], jump.gate, valid
                ).cpu()
            groups["all"].update(
                aspd, residuals, inputs, jump.values, jump.gate, valid
            )
            provenance_ids = paired["provenance_ids"]
            for provenance_id in provenance_ids.unique().tolist():
                if int(provenance_id) not in role_by_id:
                    raise ValueError(f"unknown held-out provenance ID: {provenance_id}")
                role = role_by_id[int(provenance_id)]
                role_mask = valid & (provenance_ids[:, None] == provenance_id)
                groups[role].update(
                    aspd, residuals, inputs, jump.values, jump.gate, role_mask
                )
    behavioral_rows, behavioral_provenance = _behavioral_prompts(cfg)
    behavioral_groups = {
        name: EvaluationGroup(width) for name in ("risk", "sports_control", "neutral")
    }
    behavioral_sums = {
        name: {
            "sequence_count": 0,
            "fire_rate": torch.zeros(cfg.sparsity.n_features, dtype=torch.float64),
            "activation_strength": torch.zeros(cfg.sparsity.n_features, dtype=torch.float64),
            "base_contribution_rms": torch.zeros(cfg.sparsity.n_features, dtype=torch.float64),
            "target_contribution_rms": torch.zeros(cfg.sparsity.n_features, dtype=torch.float64),
        }
        for name in behavioral_groups
    }
    paired: dict[str, dict[str, dict[str, Tensor]]] = {}
    if behavioral_rows:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.data.tokenizer,
            revision=cfg.data.tokenizer_revision,
            fix_mistral_regex=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        token_rows = [
            canonical_prompt_ids(tokenizer, str(row["prompt"]), enable_thinking=False)
            for row in behavioral_rows
        ]
        behavioral_provenance["tokenization_sha256"] = hashlib.sha256(
            json.dumps(token_rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        behavioral_provenance["same_token_ids_replayed_through_both_models"] = True
        if any(len(values) > cfg.data.sequence_length for values in token_rows):
            raise ValueError("a locked behavioral prompt exceeds the frozen 512-token limit")
        with ExitStack() as stack:
            for source, spec, capture in zip(models, cfg.models, captures, strict=True):
                _register_capture_hooks(source, spec, capture, stack)
            for start in range(0, len(behavioral_rows), batch_size_sequences):
                batch_rows = behavioral_rows[start:start + batch_size_sequences]
                batch_tokens = token_rows[start:start + batch_size_sequences]
                ids, valid_cpu = _pad_prompts(batch_tokens, int(tokenizer.pad_token_id))
                sequence_ids = torch.arange(start, start + len(batch_rows), dtype=torch.long)
                paired_batch = _capture_pair(
                    models,
                    captures,
                    ids,
                    valid_cpu,
                    device,
                    sequence_ids=sequence_ids,
                    provenance_ids=sequence_ids,
                )
                valid = paired_batch["valid_tokens"]
                residuals = [value.float() for value in paired_batch["R"]]
                inputs = [value["o_proj"].float() for value in paired_batch["X"]]
                jump = aspd.encode(residuals, valid, update_statistics=False)
                for local_index, row in enumerate(batch_rows):
                    category = str(row["category"])
                    row_valid = valid[local_index:local_index + 1]
                    row_gate = jump.gate[local_index:local_index + 1]
                    row_values = jump.values[local_index:local_index + 1]
                    row_residuals = [value[local_index:local_index + 1] for value in residuals]
                    row_inputs = [value[local_index:local_index + 1] for value in inputs]
                    behavioral_groups[category].update(
                        aspd,
                        row_residuals,
                        row_inputs,
                        row_values,
                        row_gate,
                        row_valid,
                    )
                    token_count = int(row_valid.sum())
                    fire_rate = row_gate[row_valid].sum(0).double().cpu() / token_count
                    strength = row_values[row_valid].sum(0).double().cpu() / token_count
                    base_contribution = (
                        _component_contribution_squared(
                            aspd, 0, row_inputs[0], row_gate, row_valid
                        ).cpu()
                        / token_count
                    ).sqrt()
                    target_contribution = (
                        _component_contribution_squared(
                            aspd, 1, row_inputs[1], row_gate, row_valid
                        ).cpu()
                        / token_count
                    ).sqrt()
                    metrics = {
                        "fire_rate": fire_rate,
                        "activation_strength": strength,
                        "base_contribution_rms": base_contribution,
                        "target_contribution_rms": target_contribution,
                    }
                    behavioral_sums[category]["sequence_count"] += 1
                    for key, value in metrics.items():
                        behavioral_sums[category][key] += value
                    if category in {"risk", "sports_control"}:
                        paired.setdefault(str(row["pair_id"]), {})[category] = metrics
    behavioral_component_metrics: dict[str, Tensor] = {}
    for category, values in behavioral_sums.items():
        count = int(values["sequence_count"])
        if count:
            for key in (
                "fire_rate",
                "activation_strength",
                "base_contribution_rms",
                "target_contribution_rms",
            ):
                behavioral_component_metrics[f"{category}_{key}"] = values[key] / count
    if paired:
        if any(set(values) != {"risk", "sports_control"} for values in paired.values()):
            raise ValueError("locked risk/control prompt pairing is incomplete")
        for key in (
            "fire_rate",
            "activation_strength",
            "base_contribution_rms",
            "target_contribution_rms",
        ):
            behavioral_component_metrics[f"paired_risk_minus_control_{key}"] = torch.stack(
                [
                    values["risk"][key] - values["sports_control"][key]
                    for values in paired.values()
                ]
            ).mean(0)
    if (
        behavioral_provenance.get("status") == "VERIFIED"
        and len(paired) != behavioral_provenance["expected_matched_pair_count"]
    ):
        raise ValueError("locked behavioral prompt pair count differs from the manifest")
    after = [
        snapshot_model_weights(source, spec, source_provenance)[0]
        for source, spec, source_provenance in zip(
            models, cfg.models, provenance, strict=True
        )
    ]
    if any(
        not torch.equal(left["o_proj"], right["o_proj"])
        for left, right in zip(before, after, strict=True)
    ):
        raise RuntimeError("source LLM weights changed during held-out evaluation")
    utilization = fire_counts.double() / max(valid_count, 1)
    positive = fire_counts[fire_counts > 0].double()
    probability = positive / positive.sum().clamp_min(1)
    effective = float(torch.exp(-(probability * probability.log()).sum())) if positive.numel() else 0.0
    component_tensors: dict[str, Tensor] = {"fire_counts": fire_counts}
    component_summary: dict[str, Any] = {}
    for index, name in enumerate(("base", "target")):
        u, v = aspd.component_factors(index, "o_proj")
        u_norm = u.norm(dim=-1).cpu()
        v_norm = v.norm(dim=-1).cpu()
        mass = u_norm * v_norm
        component_tensors[f"{name}_write_norm"] = u_norm
        component_tensors[f"{name}_read_norm"] = v_norm
        component_tensors[f"{name}_mechanism_mass"] = mass
        component_tensors[f"{name}_contribution_rms_per_validation_token"] = (
            contribution_squared[name] / max(valid_count, 1)
        ).sqrt()
        component_summary[name] = {
            "write_norm": _summarize(u_norm),
            "read_norm": _summarize(v_norm),
            "rank_one_frobenius_mass": _summarize(mass),
            "contribution_rms_per_validation_token": _summarize(
                component_tensors[f"{name}_contribution_rms_per_validation_token"]
            ),
        }
    component_tensors.update(behavioral_component_metrics)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tensor_path = destination / "component_metrics.pt"
    if tensor_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {tensor_path}")
    torch.save(component_tensors, tensor_path)
    report = {
        "schema_version": 1,
        "scientific_status": "HELD_OUT_ORIGINAL_RECIPE_EVALUATION",
        "config_sha256": resolved_config_sha256(cfg),
        "checkpoint": {
            **checkpoint_record,
            "rank0_path": str(Path(checkpoint_rank0).resolve()),
            "rank0_sha256": sha256_file(checkpoint_rank0),
            "rank1_path": str(Path(checkpoint_rank1).resolve()) if checkpoint_rank1 else None,
            "rank1_sha256": sha256_file(checkpoint_rank1) if checkpoint_rank1 else None,
        },
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "validation_scope": {
            "sequence_count": evaluation_sequence_count,
            "full_frozen_validation": evaluation_sequence_count == stream.sequence_count,
        },
        "matrix": "model.layers.14.self_attn.o_proj",
        "gate": "clean shared L14 mid-residual; learned JumpReLU threshold",
        "groups": {name: group.result() for name, group in groups.items()},
        "behavioral_screening": {
            **behavioral_provenance,
            "gate_is_shared_across_models": True,
            "groups": {
                name: {
                    "sequence_count": int(behavioral_sums[name]["sequence_count"]),
                    "token_count": behavioral_groups[name].tokens,
                    "reconstruction": behavioral_groups[name].result(),
                    "fire_rate_summary": _summarize(
                        behavioral_component_metrics[f"{name}_fire_rate"]
                    ),
                    "activation_strength_summary": _summarize(
                        behavioral_component_metrics[f"{name}_activation_strength"]
                    ),
                }
                for name in behavioral_groups
                if f"{name}_fire_rate" in behavioral_component_metrics
            },
            "matched_pair_count": len(paired),
            "interpretation_boundary": (
                "Observational clean-forward screening only; differences are candidate evidence, "
                "not causal attribution."
            ),
        },
        "sparsity": {
            "component_count": cfg.sparsity.n_features,
            "active_components": int((fire_counts > 0).sum()),
            "dead_components_held_out": int((fire_counts == 0).sum()),
            "threshold": float(aspd.sparsity_controller.threshold),
            "jump_relu_l0": jump_selected / max(valid_count, 1),
            "batch_topk_l0": topk_selected / max(valid_count, 1),
            "support_jaccard": support_intersection
            / max(jump_selected + topk_selected - support_intersection, 1),
            "active_fraction": float((fire_counts > 0).float().mean()),
            "dead_fraction_held_out": float((fire_counts == 0).float().mean()),
            "effective_utilization": effective,
            "fire_rate_summary": _summarize(utilization),
        },
        "components": component_summary,
        "component_metrics_path": str(tensor_path.resolve()),
        "component_metrics_sha256": sha256_file(tensor_path),
        "source_model_provenance": provenance,
        "source_model_weights_unchanged": True,
        "limitations": [
            "Reconstruction fidelity is not causal evidence for a sports-risk mechanism.",
            "Rank-0 statistics are canonical because original sync_statistics is false.",
            "Behavioral-domain activation differences are observational and require later P6 causal tests.",
        ],
    }
    write_json_immutable(destination / "held_out_evaluation.json", report)
    return report
