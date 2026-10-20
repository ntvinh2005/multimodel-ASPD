"""Exact-token ASPD measurements on the frozen Qwen2.5 behavioral V3 corpus.

The V3 response artifact is immutable.  This module reuses its token IDs and paired
layer-14 activations, captures only the seven missing matrix inputs, and evaluates ASPD
checkpoints with one fixed BatchTopK batching policy before aggregating by behavioral
group.  No response generation, judging, or causal intervention occurs here.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from aspd.multimodel.cache import (
    PairedActivationCache,
    load_effective_model,
    snapshot_model_weights,
)
from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.fidelity import (
    resolved_config_sha256,
    sha256_file,
    validate_checkpoint_evaluation_compatibility,
    validate_checkpoint_provenance,
    validate_matrix_alignment,
    write_json_immutable,
)
from aspd.multimodel.mechanism_discovery import capture_model_forward
from aspd.multimodel.training import build_model_from_cache

EXPECTED_RESPONSE_FINGERPRINT = (
    "be904fbd992728abe763df21216a0ac5ccf8dec84e99bfc0b6d33ff352bd1bfd"
)
EXPECTED_RESPONSE_SHA256 = (
    "d4d283b405640f2b2356135920be04bd8479c49d3c368032a4f174a39f8626ef"
)
EXPECTED_PROMPT_SHA256 = (
    "e2ec70f6e66503409a3afc6ea1eb7409948d10e010841b4f91c4c25e6396febe"
)
EXPECTED_SCORE_SHA256 = (
    "62371bc93d7f00abf36e482fba6a6465d5f6c2ce85cd53388bd57dd7327a4b4e"
)
MATRIX_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_v3_metadata(
    response_dir: str | Path,
    activation_cache: str | Path,
    prompts_path: str | Path,
    scores_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Validate and join immutable responses, prompt metadata, and existing scores."""

    response_root = Path(response_dir)
    activation_root = Path(activation_cache)
    response_manifest = json.loads(
        (response_root / "manifest.json").read_text(encoding="utf-8")
    )
    activation_manifest = json.loads(
        (activation_root / "manifest.json").read_text(encoding="utf-8")
    )
    if response_manifest.get("artifact_fingerprint") != EXPECTED_RESPONSE_FINGERPRINT:
        raise ValueError("unexpected frozen V3 response fingerprint")
    if response_manifest.get("examples_sha256") != EXPECTED_RESPONSE_SHA256:
        raise ValueError("unexpected frozen V3 response checksum")
    if sha256_file(response_root / "examples.jsonl") != EXPECTED_RESPONSE_SHA256:
        raise ValueError("frozen V3 response file checksum mismatch")
    frozen = activation_manifest.get("frozen_response_artifact", {})
    if frozen.get("artifact_fingerprint") != EXPECTED_RESPONSE_FINGERPRINT:
        raise ValueError("paired activation cache is not bound to frozen V3 responses")
    if not activation_manifest.get("construction", {}).get(
        "same_canonical_token_ids_replayed_through_both_models"
    ):
        raise ValueError("paired activation cache lacks exact-token replay provenance")
    prompt_path = Path(prompts_path)
    if sha256_file(prompt_path) != EXPECTED_PROMPT_SHA256:
        raise ValueError("frozen V3 prompt checksum mismatch")
    score_path = Path(scores_path)
    if sha256_file(score_path) != EXPECTED_SCORE_SHA256:
        raise ValueError("frozen V3 score checksum mismatch")
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    if scores.get("input_audit", {}).get(
        "response_artifact_fingerprint"
    ) != EXPECTED_RESPONSE_FINGERPRINT:
        raise ValueError("behavioral scores refer to a different response artifact")
    if not scores.get("calibration", {}).get("passed") or not scores.get(
        "parse_quality_pass"
    ):
        raise ValueError("behavioral score artifact did not pass its frozen gates")

    responses = _load_jsonl(response_root / "examples.jsonl")
    activation_examples = _load_jsonl(activation_root / activation_manifest["examples_file"])
    prompts = {row["prompt_id"]: row for row in _load_jsonl(prompt_path)}
    score_by_key = {
        (row["prompt_id"], row["generator_side"]): row
        for row in scores["rows"]
        if row.get("parse_error") is None
    }
    if len(responses) != 360 or len(activation_examples) != 360:
        raise ValueError("expected 360 aligned V3 response and activation examples")
    metadata: list[dict[str, Any]] = []
    for index, (response, activation) in enumerate(
        zip(responses, activation_examples, strict=True)
    ):
        if response["example_id"] != activation["example_id"]:
            raise ValueError(f"example order differs at index {index}")
        if response["token_ids"] != activation["token_ids"]:
            raise ValueError(f"canonical token IDs differ at index {index}")
        prompt = prompts[response["prompt_id"]]
        score = score_by_key.get((response["prompt_id"], response["generator_side"]))
        metadata.append(
            {
                "example_index": index,
                "example_id": response["example_id"],
                "prompt_id": response["prompt_id"],
                "prompt_index": response["prompt_index"],
                "category": prompt["category"],
                "pair_id": prompt.get("pair_id"),
                "shared_scenario_id": prompt.get("shared_scenario_id"),
                "sport_domain": prompt.get("sport_domain"),
                "risk_level": prompt.get("risk_level"),
                "generator_side": response["generator_side"],
                "split": response["split"],
                "prompt_token_count": int(response["prompt_token_count"]),
                "sequence_length": len(response["token_ids"]),
                "token_ids_sha256": response["token_ids_sha256"],
                "behavioral_score": int(score["score"]) if score is not None else None,
                "behavioral_code": score["code"] if score is not None else None,
            }
        )
    if sum(row["behavioral_score"] is not None for row in metadata) != 240:
        raise ValueError("expected exactly 240 scored sports responses")
    return metadata, response_manifest, activation_manifest


def prepare_v3_matrix_cache(
    cfg: MultiModelExperimentConfig,
    *,
    response_dir: str | Path,
    activation_cache: str | Path,
    prompts_path: str | Path,
    scores_path: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Capture missing X_j tensors while reusing the frozen paired R cache."""

    started = time.perf_counter()
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    metadata, response_manifest, source_manifest = load_v3_metadata(
        response_dir, activation_cache, prompts_path, scores_path
    )
    source_root = Path(activation_cache)
    general_cache = PairedActivationCache(cfg)
    matrix_names = validate_matrix_alignment(cfg, general_cache)
    if tuple(matrix_names) != MATRIX_NAMES:
        raise ValueError(f"unexpected matrix order: {matrix_names}")
    (output / "metadata.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata),
        encoding="utf-8",
    )
    device = torch.device(device_name)
    model_records: list[dict[str, Any]] = []
    for model_index, spec in enumerate(cfg.models):
        model, provenance = load_effective_model(spec, device)
        weights, shapes = snapshot_model_weights(model, spec, provenance)
        cached_weights = load_file(
            str(Path(cfg.cache.root) / "models" / spec.name / "weights.safetensors")
        )
        for name in matrix_names:
            if not torch.equal(weights[name].cpu(), cached_weights[name].cpu()):
                raise ValueError(f"live {spec.name}/{name} differs from training snapshot")
        model_root = output / "models" / spec.name
        for split in ("train", "validation"):
            (model_root / split).mkdir(parents=True, exist_ok=True)
            for entry in source_manifest["splits"][split]["shards"]:
                source_shard = torch.load(
                    source_root / entry["path"], map_location="cpu", weights_only=True
                )
                captured = capture_model_forward(
                    model,
                    spec,
                    source_shard["input_ids"],
                    source_shard["valid_tokens"],
                    capture_matrices=True,
                )
                valid_device = source_shard["valid_tokens"].to(captured["R"].device)
                frozen_r = source_shard["activations"][..., model_index, :][
                    source_shard["valid_tokens"]
                ].float()
                replayed_r = captured["R"][valid_device].detach().float().cpu()
                rms = float((replayed_r - frozen_r).square().mean().sqrt())
                scale = max(float(frozen_r.square().mean().sqrt()), 1e-12)
                if not math.isfinite(rms) or rms / scale > 1e-3:
                    raise ValueError(
                        f"frozen/live R mismatch for {spec.name}/{entry['path']}: {rms / scale}"
                    )
                tensors = {
                    name: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    for name, value in captured["X"].items()
                }
                if set(tensors) != set(matrix_names) or not all(
                    torch.isfinite(value).all() for value in tensors.values()
                ):
                    raise ValueError(f"invalid matrix capture for {spec.name}/{entry['path']}")
                destination = model_root / entry["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "X": tensors,
                        "example_indices": source_shard["example_indices"],
                    },
                    destination,
                )
                del source_shard, captured, tensors, frozen_r, replayed_r
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        model_records.append(
            {
                "name": spec.name,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "provenance": provenance,
                "matrix_shapes": shapes,
            }
        )
        del model, weights, cached_weights
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # The source shard table is canonical; replay diagnostics above are model-local
    # engineering checks and do not alter the fixed batching policy.
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "aspd_v3_missing_matrix_input_cache",
        "scientific_boundary": (
            "Frozen-response exact-token X_j capture only; no generation, judging, "
            "feature selection, or causal intervention."
        ),
        "frozen_response": {
            "path": str(Path(response_dir)),
            "artifact_fingerprint": response_manifest["artifact_fingerprint"],
            "examples_sha256": response_manifest["examples_sha256"],
        },
        "paired_activation_cache": {
            "path": str(source_root),
            "fingerprint": source_manifest["cache_fingerprint"],
            "examples_sha256": source_manifest["examples_sha256"],
        },
        "aspd_training_cache": {
            "path": cfg.cache.root,
            "fingerprint": general_cache.token_manifest["cache_fingerprint"],
        },
        "config_sha256": resolved_config_sha256(cfg),
        "models": model_records,
        "matrix_names": matrix_names,
        "metadata_file": "metadata.jsonl",
        "metadata_sha256": sha256_file(output / "metadata.jsonl"),
        "records": len(metadata),
        "fixed_batching": {
            "source_shards_preserved": True,
            "evaluation_batch_size_sequences": 2,
        },
        "splits": source_manifest["splits"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest["cache_fingerprint"] = _sha256_json(manifest)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


@dataclass(frozen=True)
class GroupDefinition:
    name: str
    members: frozenset[int]
    region: str = "all"


def build_group_definitions(metadata: list[dict[str, Any]]) -> list[GroupDefinition]:
    all_indices = frozenset(range(len(metadata)))
    groups = [GroupDefinition("all", all_indices)]

    def add(name: str, predicate: Callable[[dict[str, Any]], bool], region: str = "all") -> None:
        members = frozenset(
            index for index, row in enumerate(metadata) if predicate(row)
        )
        if members:
            groups.append(GroupDefinition(name, members, region))

    for category in ("risk", "sports_control", "neutral"):
        add(f"category/{category}", lambda row, c=category: row["category"] == c)
    for region in ("prompt", "response"):
        groups.append(GroupDefinition(f"region/{region}", all_indices, region))
    for side in ("base_generated", "target_generated"):
        add(f"generator/{side}", lambda row, s=side: row["generator_side"] == s)
    for category in ("risk", "sports_control", "neutral"):
        for region in ("prompt", "response"):
            add(
                f"category/{category}/region/{region}",
                lambda row, c=category: row["category"] == c,
                region,
            )
        for side in ("base_generated", "target_generated"):
            add(
                f"category/{category}/generator/{side}/region/response",
                lambda row, c=category, s=side: row["category"] == c
                and row["generator_side"] == s,
                "response",
            )
    add(
        "score/high_permissiveness/region/response",
        lambda row: row["behavioral_score"] is not None
        and row["behavioral_score"] >= 2,
        "response",
    )
    add(
        "score/cautious/region/response",
        lambda row: row["behavioral_score"] is not None
        and row["behavioral_score"] <= 1,
        "response",
    )
    return groups


@dataclass
class ReducedTensorMetrics:
    """Global sufficient statistics reduced on-device before CPU accumulation."""

    output_dim: int

    def __post_init__(self) -> None:
        self.rows = 0
        self.elements = 0
        self.sum_y = torch.zeros(self.output_dim, dtype=torch.float64)
        self.target_sq = 0.0
        self.predicted_sq = 0.0
        self.error_sq = 0.0
        self.dot = 0.0

    @torch.no_grad()
    def update(self, target: Tensor, prediction: Tensor, mask: Tensor) -> None:
        if not mask.any():
            return
        y = target[mask].float()
        yhat = prediction[mask].float()
        error = yhat - y
        if not torch.isfinite(y).all() or not torch.isfinite(yhat).all():
            raise FloatingPointError("non-finite grouped target or prediction")
        self.rows += y.shape[0]
        self.elements += y.numel()
        self.sum_y += y.sum(dim=0, dtype=torch.float64).cpu()
        self.target_sq += float(torch.sum(y * y, dtype=torch.float64))
        self.predicted_sq += float(torch.sum(yhat * yhat, dtype=torch.float64))
        self.error_sq += float(torch.sum(error * error, dtype=torch.float64))
        self.dot += float(torch.sum(y * yhat, dtype=torch.float64))

    def result(self) -> dict[str, Any]:
        if self.rows == 0:
            raise ValueError("cannot finalize empty grouped metric")
        variance = self.target_sq - float(torch.sum(self.sum_y.square())) / self.rows
        cosine_denominator = math.sqrt(self.target_sq * self.predicted_sq)
        nse = self.error_sq / max(self.target_sq, 1e-30)
        return {
            "rows": self.rows,
            "elements": self.elements,
            "sse": self.error_sq,
            "rmse": math.sqrt(self.error_sq / self.elements),
            "actual_output_rms": math.sqrt(self.target_sq / self.elements),
            "reconstructed_output_rms": math.sqrt(
                self.predicted_sq / self.elements
            ),
            "normalized_squared_error": nse,
            "error_improvement_over_zero": 1.0 - nse,
            "fvu": self.error_sq / max(variance, 1e-30),
            "target_energy": self.target_sq,
            "variance_denominator": variance,
            "cosine_similarity": self.dot / max(cosine_denominator, 1e-30),
        }


@dataclass
class ReducedPairedErrorMetrics:
    """Paired-error scalar reductions without copying full outputs to CPU."""

    base_error_sq: float = 0.0
    target_error_sq: float = 0.0
    error_dot: float = 0.0
    true_delta_sq: float = 0.0
    predicted_delta_sq: float = 0.0
    delta_error_sq: float = 0.0
    elements: int = 0

    @torch.no_grad()
    def update(
        self,
        base_target: Tensor,
        base_prediction: Tensor,
        target_target: Tensor,
        target_prediction: Tensor,
        mask: Tensor,
    ) -> None:
        if not mask.any():
            return
        base_error = (base_prediction - base_target)[mask].float()
        target_error = (target_prediction - target_target)[mask].float()
        true_delta = (target_target - base_target)[mask].float()
        predicted_delta = (target_prediction - base_prediction)[mask].float()
        delta_error = predicted_delta - true_delta
        self.base_error_sq += float(
            torch.sum(base_error * base_error, dtype=torch.float64)
        )
        self.target_error_sq += float(
            torch.sum(target_error * target_error, dtype=torch.float64)
        )
        self.error_dot += float(
            torch.sum(base_error * target_error, dtype=torch.float64)
        )
        self.true_delta_sq += float(
            torch.sum(true_delta * true_delta, dtype=torch.float64)
        )
        self.predicted_delta_sq += float(
            torch.sum(predicted_delta * predicted_delta, dtype=torch.float64)
        )
        self.delta_error_sq += float(
            torch.sum(delta_error * delta_error, dtype=torch.float64)
        )
        self.elements += base_error.numel()

    def result(self) -> dict[str, Any]:
        if self.elements == 0:
            raise ValueError("cannot finalize empty paired error metric")
        norm_product = math.sqrt(self.base_error_sq * self.target_error_sq)
        return {
            "elements": self.elements,
            "base_error_rms": math.sqrt(self.base_error_sq / self.elements),
            "target_error_rms": math.sqrt(self.target_error_sq / self.elements),
            "error_cosine_similarity": self.error_dot / max(norm_product, 1e-30),
            "true_delta_rms": math.sqrt(self.true_delta_sq / self.elements),
            "predicted_delta_rms": math.sqrt(
                self.predicted_delta_sq / self.elements
            ),
            "delta_error_rms": math.sqrt(self.delta_error_sq / self.elements),
            "delta_error_to_true_ratio": math.sqrt(
                self.delta_error_sq / max(self.true_delta_sq, 1e-30)
            ),
            "cancellation_gain_vs_independent_errors": 1.0
            - self.delta_error_sq
            / max(self.base_error_sq + self.target_error_sq, 1e-30),
        }


class GroupAccumulator:
    def __init__(
        self,
        activation_dim: int,
        matrix_dims: Mapping[str, int],
    ) -> None:
        self.activation = {
            model: ReducedTensorMetrics(activation_dim)
            for model in ("base", "target")
        }
        self.matrices = {
            name: {
                metric: ReducedTensorMetrics(width)
                for metric in (
                    "base",
                    "target",
                    "delta_total",
                    "delta_parameter",
                    "delta_input",
                )
            }
            for name, width in matrix_dims.items()
        }
        self.errors = {name: ReducedPairedErrorMetrics() for name in matrix_dims}

    def result(self) -> dict[str, Any]:
        return {
            "activation_reconstruction": {
                name: value.result() for name, value in self.activation.items()
            },
            "matrices": {
                name: {
                    **{metric: value.result() for metric, value in metrics.items()},
                    "error_cancellation": self.errors[name].result(),
                }
                for name, metrics in self.matrices.items()
            },
        }


def _mask_for_group(
    group: GroupDefinition,
    example_indices: Tensor,
    valid: Tensor,
    metadata: list[dict[str, Any]],
) -> Tensor:
    mask = valid.clone()
    for row_index, example_index in enumerate(example_indices.tolist()):
        if int(example_index) not in group.members:
            mask[row_index] = False
            continue
        prompt_length = int(metadata[int(example_index)]["prompt_token_count"])
        if group.region == "prompt":
            mask[row_index, prompt_length:] = False
        elif group.region == "response":
            mask[row_index, :prompt_length] = False
    return mask


def _group_metadata(
    group: GroupDefinition, metadata: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = [metadata[index] for index in sorted(group.members)]
    tokens = 0
    for row in rows:
        if group.region == "prompt":
            tokens += int(row["prompt_token_count"])
        elif group.region == "response":
            tokens += int(row["sequence_length"]) - int(row["prompt_token_count"])
        else:
            tokens += int(row["sequence_length"])
    return {
        "sequences": len(rows),
        "tokens": tokens,
        "sports_domains": sorted(
            {str(row["sport_domain"]) for row in rows if row["sport_domain"] is not None}
        ),
    }


def evaluate_v3_checkpoint(
    cfg: MultiModelExperimentConfig,
    checkpoint: str | Path,
    *,
    checkpoint_cfg: MultiModelExperimentConfig,
    matrix_cache_dir: str | Path,
    output_json: str | Path,
    output_components: str | Path,
    expected_checkpoint_sha256: str,
    device_name: str = "cuda",
    batch_size_sequences: int = 2,
) -> dict[str, Any]:
    """Evaluate one ASPD checkpoint on V3 under a fixed cross-checkpoint batching policy."""

    if batch_size_sequences != 2:
        raise ValueError("registered V3 ASPD comparison uses batch_size_sequences=2")
    device = torch.device(device_name)
    cache_root = Path(matrix_cache_dir)
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    metadata = _load_jsonl(cache_root / manifest["metadata_file"])
    source_root = Path(manifest["paired_activation_cache"]["path"])
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    if manifest["cache_fingerprint"] != _sha256_json(
        {key: value for key, value in manifest.items() if key != "cache_fingerprint"}
    ):
        raise ValueError("V3 matrix cache fingerprint mismatch")
    if source_manifest["cache_fingerprint"] != manifest["paired_activation_cache"][
        "fingerprint"
    ]:
        raise ValueError("paired activation cache changed after X capture")
    eval_cache = PairedActivationCache(cfg)
    validate_checkpoint_evaluation_compatibility(cfg, checkpoint_cfg)
    training_cache = PairedActivationCache(checkpoint_cfg)
    matrix_names = validate_matrix_alignment(cfg, eval_cache)
    checkpoint_record = validate_checkpoint_provenance(
        checkpoint_cfg,
        training_cache,
        checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    model = build_model_from_cache(cfg, eval_cache)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    current = model.state_dict()
    for name, value in state["model"].items():
        if name.startswith("target_weights.") and not torch.equal(
            value.cpu(), current[name].cpu()
        ):
            raise ValueError(f"checkpoint weight {name} differs from V3 target snapshot")
    model.load_state_dict(state["model"], strict=True)
    checkpoint_step = int(state["step"])
    del state
    model = model.to(device).float().eval()
    norms = [float(row["r_rms_norm"]) for row in training_cache.model_manifests]
    matrix_dims = {
        name: int(eval_cache.model_manifests[0]["matrices"][name]["d_out"])
        for name in matrix_names
    }
    groups = build_group_definitions(metadata)
    accumulators = {
        group.name: GroupAccumulator(
            int(eval_cache.model_manifests[0]["activation_dim"]), matrix_dims
        )
        for group in groups
    }
    c = cfg.sparsity.n_features
    sequence_tensors = {
        "prompt_max": torch.zeros((len(metadata), c), dtype=torch.float32),
        "response_max": torch.zeros((len(metadata), c), dtype=torch.float32),
        "prompt_argmax": torch.full((len(metadata), c), -1, dtype=torch.int16),
        "response_argmax": torch.full((len(metadata), c), -1, dtype=torch.int16),
        "prompt_fire_count": torch.zeros((len(metadata), c), dtype=torch.int16),
        "response_fire_count": torch.zeros((len(metadata), c), dtype=torch.int16),
    }
    selected_pairs = 0
    for split in ("train", "validation"):
        for entry in source_manifest["splits"][split]["shards"]:
            source = torch.load(
                source_root / entry["path"], map_location="cpu", weights_only=True
            )
            x_shards = [
                torch.load(
                    cache_root / "models" / spec.name / entry["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                for spec in cfg.models
            ]
            if not all(
                torch.equal(shard["example_indices"], source["example_indices"])
                for shard in x_shards
            ):
                raise ValueError(f"X/example alignment failed for {entry['path']}")
            rows = int(source["input_ids"].shape[0])
            for start in range(0, rows, batch_size_sequences):
                stop = min(start + batch_size_sequences, rows)
                valid = source["valid_tokens"][start:stop].to(device)
                example_indices = source["example_indices"][start:stop]
                activations = [
                    source["activations"][start:stop, :, n, :].to(device).float()
                    / norms[n]
                    for n in range(2)
                ]
                inputs = [
                    {
                        name: x_shards[n]["X"][name][start:stop].to(device).float()
                        for name in matrix_names
                    }
                    for n in range(2)
                ]
                code = model.encode(activations, valid)
                selected_pairs += code.n_selected
                for row_index, example_index in enumerate(example_indices.tolist()):
                    prompt_length = int(metadata[example_index]["prompt_token_count"])
                    length = int(valid[row_index].sum())
                    for region, begin, end in (
                        ("prompt", 0, prompt_length),
                        ("response", prompt_length, length),
                    ):
                        values = code.values[row_index, begin:end]
                        maxima, argmax = values.max(dim=0)
                        sequence_tensors[f"{region}_max"][example_index] = maxima.cpu()
                        sequence_tensors[f"{region}_argmax"][example_index] = (
                            argmax + begin
                        ).to(dtype=torch.int16).cpu()
                        sequence_tensors[f"{region}_fire_count"][example_index] = (
                            (values > 0).sum(dim=0).to(dtype=torch.int16).cpu()
                        )
                group_masks = {
                    group.name: _mask_for_group(
                        group, example_indices, valid, metadata
                    )
                    for group in groups
                }
                for n, model_name in enumerate(("base", "target")):
                    reconstruction = model.activation_decoders.reconstruct(
                        code.values, n, valid
                    ).float()
                    for group in groups:
                        accumulators[group.name].activation[model_name].update(
                            activations[n], reconstruction, group_masks[group.name]
                        )
                for name in matrix_names:
                    x_base, x_target = inputs[0][name], inputs[1][name]
                    w_base, w_target = model._target(0, name), model._target(1, name)
                    y_base = w_base.apply(x_base).float()
                    y_target = w_target.apply(x_target).float()
                    component_base = model.components[0][name.replace(".", "__dot__")]
                    component_target = model.components[1][name.replace(".", "__dot__")]
                    yhat_base = component_base.reconstruct(x_base, code.gate, valid).float()
                    yhat_target = component_target.reconstruct(
                        x_target, code.gate, valid
                    ).float()
                    delta_total = y_target - y_base
                    delta_total_hat = yhat_target - yhat_base
                    delta_parameter = torch.nn.functional.linear(
                        x_base, w_target.weight.float() - w_base.weight.float()
                    )
                    target_on_base_hat = component_target.reconstruct(
                        x_base, code.gate, valid
                    ).float()
                    delta_parameter_hat = target_on_base_hat - yhat_base
                    delta_x = x_target - x_base
                    delta_input = w_target.apply(delta_x).float()
                    delta_input_hat = component_target.reconstruct(
                        delta_x, code.gate, valid
                    ).float()
                    for group in groups:
                        mask = group_masks[group.name]
                        accumulator = accumulators[group.name]
                        for metric, target, prediction in (
                            ("base", y_base, yhat_base),
                            ("target", y_target, yhat_target),
                            ("delta_total", delta_total, delta_total_hat),
                            (
                                "delta_parameter",
                                delta_parameter,
                                delta_parameter_hat,
                            ),
                            ("delta_input", delta_input, delta_input_hat),
                        ):
                            accumulator.matrices[name][metric].update(
                                target, prediction, mask
                            )
                        accumulator.errors[name].update(
                            y_base, yhat_base, y_target, yhat_target, mask
                        )
                    del (
                        y_base,
                        y_target,
                        yhat_base,
                        yhat_target,
                        delta_total,
                        delta_total_hat,
                        delta_parameter,
                        delta_parameter_hat,
                        delta_input,
                        delta_input_hat,
                        target_on_base_hat,
                    )
                del activations, inputs, code, group_masks
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del source, x_shards

    expected_pairs = sum(
        int(row["sequence_length"]) for row in metadata
    ) * cfg.sparsity.top_k
    if selected_pairs != expected_pairs:
        raise ValueError(
            f"BatchTopK selected {selected_pairs} pairs, expected {expected_pairs}"
        )
    Path(output_components).parent.mkdir(parents=True, exist_ok=True)
    save_file(sequence_tensors, str(output_components))
    result = {
        "schema_version": 1,
        "kind": "aspd_v3_grouped_computation_fidelity",
        "scientific_status": "EXPLORATORY_REUSE_OF_V3",
        "checkpoint": {**checkpoint_record, "step": checkpoint_step},
        "checkpoint_training_cache": {
            "path": checkpoint_cfg.cache.root,
            "fingerprint": training_cache.token_manifest["cache_fingerprint"],
            "grounding_r_rms_norm": norms,
        },
        "v3_matrix_cache": {
            "path": str(cache_root),
            "fingerprint": manifest["cache_fingerprint"],
        },
        "measurement": {
            "sequences": len(metadata),
            "tokens": sum(int(row["sequence_length"]) for row in metadata),
            "batch_size_sequences": batch_size_sequences,
            "batching_policy": (
                "source shard order, contiguous pairs of sequences; identical across checkpoints"
            ),
            "batch_topk_selected_pairs": selected_pairs,
            "neutral_behavioral_labels": "not available and not imputed",
        },
        "groups": {
            group.name: {
                **_group_metadata(group, metadata),
                **accumulators[group.name].result(),
            }
            for group in groups
        },
        "component_sequence_summary": {
            # Store a path relative to the JSON artifact.  Slurm writes both files
            # in a job-specific temporary directory and atomically renames that
            # directory after validation, so an absolute temporary path would be
            # stale in the completed artifact.
            "path": Path(output_components).name,
            "sha256": sha256_file(output_components),
            "tensors": {name: list(value.shape) for name, value in sequence_tensors.items()},
        },
    }
    write_json_immutable(output_json, result)
    return result
