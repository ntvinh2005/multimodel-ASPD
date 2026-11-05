"""Read-only statistics and provenance helpers for the ASPD v3 500M diagnostic.

This module deliberately contains no optimizer or training entry point.  The GPU
evaluator feeds it frozen tensors; CPU tests exercise the reductions independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DiagnosticContract:
    schema_version: int
    status: str
    source_run: Path
    source_config: Path
    checkpoint: Path
    output_root: Path
    source_git_sha: str
    source_config_sha256: str
    source_config_fingerprint: str
    dataset_manifest_sha256: str
    alignment_manifest_sha256: str
    checkpoint_manifest_sha256: str
    split: str
    dev_anchors: int
    main_anchors: int
    pool_anchors: int
    seed: int
    bootstrap_replicates: int
    jaccard_top_features: int
    jaccard_min_fires: int

    @classmethod
    def load(cls, path: str | Path) -> DiagnosticContract:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported ASPD v3 diagnostic contract schema")
        evaluation = payload["evaluation"]
        provenance = payload["provenance"]
        paths = payload["paths"]
        contract = cls(
            schema_version=1,
            status=str(payload["status"]),
            source_run=Path(paths["source_run"]),
            source_config=Path(paths["source_config"]),
            checkpoint=Path(paths["checkpoint"]),
            output_root=Path(paths["output_root"]),
            source_git_sha=str(provenance["source_git_sha"]),
            source_config_sha256=str(provenance["source_config_sha256"]),
            source_config_fingerprint=str(provenance["source_config_fingerprint"]),
            dataset_manifest_sha256=str(provenance["dataset_manifest_sha256"]),
            alignment_manifest_sha256=str(provenance["alignment_manifest_sha256"]),
            checkpoint_manifest_sha256=str(provenance["checkpoint_manifest_sha256"]),
            split=str(evaluation["split"]),
            dev_anchors=int(evaluation["dev_anchors"]),
            main_anchors=int(evaluation["main_anchors"]),
            pool_anchors=int(evaluation["pool_anchors"]),
            seed=int(evaluation["seed"]),
            bootstrap_replicates=int(evaluation["bootstrap_replicates"]),
            jaccard_top_features=int(evaluation["jaccard_top_features"]),
            jaccard_min_fires=int(evaluation["jaccard_min_fires"]),
        )
        contract.validate_static()
        return contract

    def validate_static(self) -> None:
        if self.status != "RESEARCHER_APPROVED":
            raise ValueError("diagnostic contract must be RESEARCHER_APPROVED")
        if self.split != "validation":
            raise ValueError("the checkpoint diagnostic must use the frozen validation split")
        if self.pool_anchors != 2048 or self.dev_anchors <= 0 or self.main_anchors <= 0:
            raise ValueError("invalid diagnostic anchor budget")
        if self.dev_anchors % self.pool_anchors or self.main_anchors % self.pool_anchors:
            raise ValueError("diagnostic anchor budgets must contain complete BatchTopK pools")
        if self.main_anchors < self.dev_anchors:
            raise ValueError("main diagnostic cannot be smaller than development diagnostic")
        if self.bootstrap_replicates <= 0:
            raise ValueError("cluster bootstrap replicate count must be positive")
        if self.jaccard_top_features <= 1 or self.jaccard_min_fires <= 0:
            raise ValueError("invalid activity-aware Jaccard policy")

    def fingerprint(self) -> str:
        return canonical_json_sha256(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in self.__dict__.items()
            }
        )


def validate_frozen_provenance(
    contract: DiagnosticContract, *, source_root: str | Path = "."
) -> dict[str, Any]:
    root = Path(source_root)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    source_run = resolved(contract.source_run)
    source_config = resolved(contract.source_config)
    checkpoint = resolved(contract.checkpoint)
    run_manifest_path = source_run / "run_manifest.json"
    alignment_path = root / Path(
        "out/multimodel/aspd_v3/main_2b_v1/data/pile_uncopyrighted/alignment_manifest.json"
    )
    dataset_path = root / Path(
        "out/multimodel/aspd_v3/main_2b_v1/data/pile_uncopyrighted/manifest.json"
    )
    required = (
        source_run,
        source_config,
        checkpoint,
        run_manifest_path,
        alignment_path,
        dataset_path,
        checkpoint / "manifest.json",
        checkpoint / "COMPLETE",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"frozen diagnostic input is missing: {missing}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads(
        (checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    marker = json.loads((checkpoint / "COMPLETE").read_text(encoding="utf-8"))
    observed = {
        "source_git_sha": run_manifest.get("git_sha"),
        "source_config_sha256": sha256_file(source_config),
        "source_config_fingerprint": run_manifest.get("config_fingerprint"),
        "dataset_manifest_sha256": run_manifest.get("dataset_manifest_sha256"),
        "alignment_manifest_sha256": run_manifest.get("alignment_manifest_sha256"),
        "checkpoint_manifest_sha256": sha256_file(checkpoint / "manifest.json"),
        "checkpoint_step": checkpoint_manifest.get("step"),
        "checkpoint_status": checkpoint_manifest.get("status"),
        "checkpoint_world_size": checkpoint_manifest.get("world_size"),
        "model_order": run_manifest.get("model_order"),
    }
    expected = {
        "source_git_sha": contract.source_git_sha,
        "source_config_sha256": contract.source_config_sha256,
        "source_config_fingerprint": contract.source_config_fingerprint,
        "dataset_manifest_sha256": contract.dataset_manifest_sha256,
        "alignment_manifest_sha256": contract.alignment_manifest_sha256,
        "checkpoint_manifest_sha256": contract.checkpoint_manifest_sha256,
        "checkpoint_step": 61036,
        "checkpoint_status": "COMPLETE",
        "checkpoint_world_size": 3,
        "model_order": ["qwen", "gemma", "smol"],
    }
    mismatch = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if observed[key] != value
    }
    if marker.get("manifest_sha256") != observed["checkpoint_manifest_sha256"]:
        mismatch["checkpoint_complete_marker"] = {
            "expected": observed["checkpoint_manifest_sha256"],
            "observed": marker.get("manifest_sha256"),
        }
    if mismatch:
        raise ValueError(f"frozen diagnostic provenance mismatch: {mismatch}")
    return {"status": "PASS", **observed, "diagnostic_contract_sha256": contract.fingerprint()}


def canonical_metric_rows(lines: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Retain the final physical row for each optimizer step after a recovery replay."""

    by_step: dict[int, dict[str, Any]] = {}
    physical_rows = 0
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["step"])
        if step <= 0:
            raise ValueError("optimizer steps must be positive")
        by_step[step] = row
        physical_rows += 1
    steps = sorted(by_step)
    if not steps or steps != list(range(1, steps[-1] + 1)):
        raise ValueError("canonical metric trajectory has missing optimizer steps")
    rows = [by_step[step] for step in steps]
    return rows, {
        "physical_rows": physical_rows,
        "canonical_rows": len(rows),
        "replayed_rows": physical_rows - len(rows),
        "final_step": steps[-1],
    }


def iter_canonical_metric_file(
    path: str | Path,
) -> tuple[Iterator[dict[str, Any]], dict[str, int]]:
    """Two-pass canonical reader with O(number-of-steps) indexing memory."""

    metrics_path = Path(path)
    last_line: dict[int, int] = {}
    physical_rows = 0
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            last_line[int(row["step"])] = line_number
            physical_rows += 1
    steps = sorted(last_line)
    if not steps or steps != list(range(1, steps[-1] + 1)):
        raise ValueError("canonical metric trajectory has missing optimizer steps")

    def rows() -> Iterator[dict[str, Any]]:
        with metrics_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                if last_line[int(row["step"])] == line_number:
                    yield row

    return rows(), {
        "physical_rows": physical_rows,
        "canonical_rows": len(steps),
        "replayed_rows": physical_rows - len(steps),
        "final_step": steps[-1],
    }


def compact_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Collapse hundreds of matrix/layer fields into audit-level signed trajectories."""

    loss = {key: float(value) for key, value in row["loss"].items()}
    compact_loss = {key: loss[key] for key in ("total", "activation", "auxk", "internal")}
    for objective in ("activation", "auxk"):
        for model in ("qwen", "gemma", "smol"):
            for site in ("pre", "mid"):
                prefix = f"loss/{objective}/{site}/{model}/"
                values = [value for key, value in loss.items() if key.startswith(prefix)]
                if len(values) != 8:
                    raise ValueError(f"missing {objective} layer metrics for {site}/{model}")
                compact_loss[f"{objective}/{model}/{site}"] = sum(values) / len(values)
    for model in ("qwen", "gemma", "smol"):
        for matrix in ("q", "k", "v", "o", "mlp_gate", "mlp_up", "mlp_down"):
            prefix = f"loss/internal/{model}."
            suffix = f".{matrix}"
            values = [
                value for key, value in loss.items() if key.startswith(prefix) and key.endswith(suffix)
            ]
            if len(values) != 8:
                raise ValueError(f"missing internal layer metrics for {model}/{matrix}")
            compact_loss[f"internal/{model}/{matrix}"] = sum(values) / len(values)
    return {
        "step": int(row["step"]),
        "accepted_exposures": int(row["accepted_exposures"]),
        "seconds": float(row["seconds"]),
        "anchors_per_second": float(row["anchors_per_second"]),
        "loss": compact_loss,
    }


def _window_mean(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in keys:
        values = [float(row["loss"][key]) for row in rows if key in row["loss"]]
        if values:
            result[key] = sum(values) / len(values)
    return result


def summarize_training_trajectory(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 1000:
        raise ValueError("training trajectory audit requires at least 1000 canonical steps")
    all_loss_keys = sorted({key for row in rows for key in row["loss"]})
    if any(not math.isfinite(float(row["loss"][key])) for row in rows for key in row["loss"]):
        raise FloatingPointError("training trajectory contains non-finite loss")
    windows = {
        "first_1000": rows[:1000],
        "middle_1000": rows[len(rows) // 2 - 500 : len(rows) // 2 + 500],
        "last_1000": rows[-1000:],
    }
    summaries = {
        name: _window_mean(values, all_loss_keys) for name, values in windows.items()
    }
    model_site: dict[str, dict[str, dict[str, float]]] = {}
    for name, summary in summaries.items():
        for model in ("qwen", "gemma", "smol"):
            for site in ("pre", "mid"):
                compact_key = f"activation/{model}/{site}"
                prefix = f"loss/activation/{site}/{model}/"
                values = (
                    [summary[compact_key]]
                    if compact_key in summary
                    else [value for key, value in summary.items() if key.startswith(prefix)]
                )
                model_site.setdefault(name, {}).setdefault(model, {})[site] = (
                    sum(values) / len(values) if values else float("nan")
                )
    ratios: dict[str, Any] = {}
    for name, values in model_site.items():
        ratios[name] = {
            site: {
                "gemma_over_qwen": values["gemma"][site] / values["qwen"][site],
                "gemma_over_smol": values["gemma"][site] / values["smol"][site],
            }
            for site in ("pre", "mid")
        }
    return {
        "status": "PASS",
        "steps": len(rows),
        "window_means": summaries,
        "activation_model_site": model_site,
        "activation_ratios": ratios,
    }


@dataclass
class TensorMetricAccumulator:
    """Aggregate exact global reconstruction statistics from FP32 tensor batches."""

    width: int
    rows: int = 0
    squared_error: float = 0.0
    target_square_sum: float = 0.0
    prediction_square_sum: float = 0.0
    target_sum: Tensor = field(init=False)
    prediction_sum: Tensor = field(init=False)
    target_scalar_sum: float = 0.0
    prediction_scalar_sum: float = 0.0
    target_prediction_sum: float = 0.0

    def __post_init__(self) -> None:
        self.target_sum = torch.zeros(self.width, dtype=torch.float64)
        self.prediction_sum = torch.zeros(self.width, dtype=torch.float64)

    def update(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError("metric batches must be matching [anchors,width] tensors")
        if prediction.shape[1] != self.width or prediction.shape[0] == 0:
            raise ValueError("metric batch has invalid shape")
        pred = prediction.detach().to(device="cpu", dtype=torch.float64)
        truth = target.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
            raise FloatingPointError("non-finite reconstruction metric input")
        self.rows += truth.shape[0]
        residual = pred - truth
        self.squared_error += float(residual.square().sum())
        self.target_square_sum += float(truth.square().sum())
        self.prediction_square_sum += float(pred.square().sum())
        self.target_sum += truth.sum(dim=0)
        self.prediction_sum += pred.sum(dim=0)
        self.target_scalar_sum += float(truth.sum())
        self.prediction_scalar_sum += float(pred.sum())
        self.target_prediction_sum += float((truth * pred).sum())

    def merge_(self, other: TensorMetricAccumulator) -> None:
        if self.width != other.width:
            raise ValueError("cannot merge metric accumulators with different widths")
        self.rows += other.rows
        self.squared_error += other.squared_error
        self.target_square_sum += other.target_square_sum
        self.prediction_square_sum += other.prediction_square_sum
        self.target_sum += other.target_sum
        self.prediction_sum += other.prediction_sum
        self.target_scalar_sum += other.target_scalar_sum
        self.prediction_scalar_sum += other.prediction_scalar_sum
        self.target_prediction_sum += other.target_prediction_sum

    def finalize(self) -> dict[str, float | int | bool | None]:
        if self.rows <= 0:
            raise ValueError("cannot finalize an empty metric accumulator")
        values = self.rows * self.width
        centered_ss = self.target_square_sum - float(self.target_sum.square().sum()) / self.rows
        target_flat_ss = self.target_square_sum - self.target_scalar_sum**2 / values
        pred_flat_ss = self.prediction_square_sum - self.prediction_scalar_sum**2 / values
        covariance = (
            self.target_prediction_sum
            - self.target_scalar_sum * self.prediction_scalar_sum / values
        )
        correlation_denominator = math.sqrt(max(0.0, target_flat_ss * pred_flat_ss))
        if centered_ss <= 0:
            raise FloatingPointError("target has zero or negative centered variance")
        correlation_defined = correlation_denominator > 0
        fvu = self.squared_error / centered_ss
        result = {
            "anchor_rows": self.rows,
            "scalar_values": values,
            "sse": self.squared_error,
            "centered_target_ss": centered_ss,
            "zero_baseline_sse": self.target_square_sum,
            "mse": self.squared_error / values,
            "fvu": fvu,
            "r2": 1.0 - fvu,
            "relative_l2_error": math.sqrt(self.squared_error / self.target_square_sum),
            "target_rms": math.sqrt(self.target_square_sum / values),
            "prediction_rms": math.sqrt(self.prediction_square_sum / values),
            "residual_rms": math.sqrt(self.squared_error / values),
            # A constant prediction (notably the Matryoshka bias-only prefix) has
            # no defined Pearson correlation.  This is not a failed reconstruction
            # metric: retain every other sufficient-statistic result and mark the
            # correlation explicitly unavailable.
            "target_prediction_correlation": (
                covariance / correlation_denominator if correlation_defined else None
            ),
            "target_prediction_correlation_defined": correlation_defined,
            "mean_baseline_fvu": 1.0,
            "zero_baseline_fvu": self.target_square_sum / centered_ss,
        }
        if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
            raise FloatingPointError("non-finite finalized reconstruction metric")
        return result


def cluster_bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, replicates: int, alpha: float = 0.05
) -> dict[str, float | int]:
    """Bootstrap documents/sequences as clusters; token positions are never resampled."""

    if not values or replicates <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid cluster-bootstrap input")
    if any(not math.isfinite(value) for value in values):
        raise FloatingPointError("cluster-bootstrap values must be finite")
    rng = random.Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    )
    lower_index = max(0, math.floor((alpha / 2) * replicates))
    upper_index = min(replicates - 1, math.ceil((1 - alpha / 2) * replicates) - 1)
    return {
        "clusters": count,
        "replicates": replicates,
        "mean": sum(values) / count,
        "lower": samples[lower_index],
        "upper": samples[upper_index],
    }


def feature_health_from_counts(counts: Tensor, accepted_anchors: int) -> dict[str, Any]:
    values = counts.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 1 or accepted_anchors <= 0 or torch.any(values < 0):
        raise ValueError("feature counts must be a non-negative vector")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("feature health requires at least one firing event")
    probabilities = values / total
    positive = probabilities > 0
    entropy = float(-(probabilities[positive] * probabilities[positive].log()).sum())
    ordered = values.sort(descending=True).values

    def share(fraction: float) -> float:
        count = max(1, math.ceil(values.numel() * fraction))
        return float(ordered[:count].sum()) / total

    return {
        "accepted_anchors": accepted_anchors,
        "feature_count": values.numel(),
        "used_features": int(positive.sum()),
        "never_fired_features": int((~positive).sum()),
        "total_firing_events": int(total),
        "mean_active_features_per_anchor": total / accepted_anchors,
        "effective_dictionary_size_entropy": math.exp(entropy),
        "top_1_percent_event_share": share(0.01),
        "top_5_percent_event_share": share(0.05),
        "top_10_percent_event_share": share(0.10),
    }


def pairwise_jaccard(
    firing: Tensor, counts: Tensor, *, top_features: int, min_fires: int
) -> dict[str, Any]:
    """Exact Jaccard among an activity-filtered feature subset.

    ``firing`` is a boolean ``[anchors,features]`` matrix intended for bounded held-out
    diagnostics.  Filtering avoids meaningless high similarity among nearly inactive rows.
    """

    if firing.ndim != 2 or firing.dtype != torch.bool or counts.shape != (firing.shape[1],):
        raise ValueError("Jaccard inputs have incompatible shapes")
    eligible = (counts >= min_fires).nonzero(as_tuple=True)[0]
    if eligible.numel() < 2:
        return {
            "eligible_features": int(eligible.numel()),
            "evaluated_features": int(eligible.numel()),
            "evaluated_pairs": 0,
            "coverage_of_all_pairs": 0.0,
            "pairs": [],
        }
    ranked = eligible[counts[eligible].argsort(descending=True)]
    selected = ranked[:top_features]
    subset = firing[:, selected].T.to(torch.int32)
    intersections = subset @ subset.T
    selected_counts = counts[selected].to(torch.int32)
    pairs = []
    for left in range(selected.numel()):
        for right in range(left + 1, selected.numel()):
            intersection = int(intersections[left, right])
            union = int(selected_counts[left] + selected_counts[right] - intersection)
            pairs.append(
                {
                    "left": int(selected[left]),
                    "right": int(selected[right]),
                    "jaccard": intersection / union if union else 0.0,
                    "intersection": intersection,
                    "union": union,
                }
            )
    total_pairs = firing.shape[1] * (firing.shape[1] - 1) // 2
    return {
        "eligible_features": int(eligible.numel()),
        "evaluated_features": int(selected.numel()),
        "evaluated_pairs": len(pairs),
        "coverage_of_all_pairs": len(pairs) / total_pairs,
        "pairs": pairs,
    }
