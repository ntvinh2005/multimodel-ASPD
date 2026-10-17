"""Method adapters isolate measured ranking and retained-context artifact formats."""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from aspd.multimodel.semantic_triage.schema import MeasuredCandidate


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


class CandidateAdapter(ABC):
    """Generic boundary for current Crosscoder and future Delta-Crosscoder candidates."""

    @property
    @abstractmethod
    def method_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def sequence_length(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def select_candidates(
        self,
        *,
        base_tail_count: int,
        finetuned_tail_count: int,
        min_fire_count: int,
        feature_ids: Sequence[int] | None = None,
    ) -> list[MeasuredCandidate]:
        """Return method-measured candidates without semantic reranking."""

    @abstractmethod
    def load_contexts(self, feature_id: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def provenance(self) -> dict[str, Any]:
        raise NotImplementedError


class CrosscoderCandidateAdapter(CandidateAdapter):
    """Read completed BatchTopK Crosscoder JSON artifacts without mutating them."""

    context_backfill_name = "content_semantic_scan_contexts.json"

    def __init__(self, analysis_dir: str | Path) -> None:
        self.analysis_dir = Path(analysis_dir).resolve()
        self.summary = _read_json(self.analysis_dir / "summary.json")
        metrics = _read_json(self.analysis_dir / "feature_metrics.json")
        rows = metrics.get("features")
        if not isinstance(rows, list):
            raise ValueError("feature_metrics.json must contain a features list")
        self._rows = {int(row["feature_id"]): row for row in rows if isinstance(row, dict)}
        if len(self._rows) != len(rows):
            raise ValueError("feature_metrics.json contains malformed or duplicate feature IDs")

        canonical = _read_json(self.analysis_dir / "top_activation_examples.json")
        canonical_features = canonical.get("features", canonical)
        if not isinstance(canonical_features, dict):
            raise ValueError("top_activation_examples.json must contain a features object")
        self._contexts = {
            str(feature_id): examples
            for feature_id, examples in canonical_features.items()
            if isinstance(examples, list)
        }
        backfill_path = self.analysis_dir / self.context_backfill_name
        if backfill_path.exists():
            backfill = _read_json(backfill_path)
            if backfill.get("metadata", {}).get("method") != self.method_name:
                raise ValueError(f"incompatible context backfill artifact: {backfill_path}")
            backfill_features = backfill.get("features")
            if not isinstance(backfill_features, dict):
                raise ValueError(f"backfill artifact has no features object: {backfill_path}")
            self._contexts.update(
                {
                    str(feature_id): examples
                    for feature_id, examples in backfill_features.items()
                    if isinstance(examples, list)
                }
            )
        raw_groups = self.summary.get("candidate_groups", {})
        self._groups = raw_groups if isinstance(raw_groups, dict) else {}
        self._sequence_length = self._load_sequence_length()

    @property
    def method_name(self) -> str:
        return "batchtopk_crosscoder"

    @property
    def sequence_length(self) -> int | None:
        return self._sequence_length

    def _load_sequence_length(self) -> int | None:
        config_path = self.analysis_dir.parent / "config_resolved.yaml"
        if not config_path.exists():
            return None
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        data = value.get("data")
        length = data.get("sequence_length") if isinstance(data, dict) else None
        return int(length) if isinstance(length, int) and length > 0 else None

    def _candidate(self, row: Mapping[str, Any], scan_pools: list[str]) -> MeasuredCandidate:
        feature_id = int(row["feature_id"])
        groups = [
            str(name)
            for name, ids in self._groups.items()
            if isinstance(ids, list) and feature_id in ids
        ]
        return MeasuredCandidate(
            feature_id=feature_id,
            strict_classification=str(row["classification"]),
            relative_decoder_norm=float(row["relative_decoder_norm"]),
            fire_count=int(row["fire_count"]),
            blind_rank_within_class=int(row["blind_rank_within_class"]),
            blind_rank_base_tail=int(row["blind_rank_base_tail"]),
            blind_rank_finetuned_tail=int(row["blind_rank_finetuned_tail"]),
            original_inspection_groups=groups,
            scan_pools=scan_pools,
        )

    def select_candidates(
        self,
        *,
        base_tail_count: int,
        finetuned_tail_count: int,
        min_fire_count: int,
        feature_ids: Sequence[int] | None = None,
    ) -> list[MeasuredCandidate]:
        if base_tail_count < 0 or finetuned_tail_count < 0 or min_fire_count < 0:
            raise ValueError("tail counts and min_fire_count must be nonnegative")
        eligible = [row for row in self._rows.values() if int(row["fire_count"]) >= min_fire_count]
        if feature_ids is not None:
            if len(feature_ids) != len(set(feature_ids)):
                raise ValueError("feature IDs must be unique")
            missing = [feature_id for feature_id in feature_ids if feature_id not in self._rows]
            if missing:
                raise ValueError(f"feature IDs out of range or absent: {missing}")
            below_support = [
                feature_id
                for feature_id in feature_ids
                if int(self._rows[feature_id]["fire_count"]) < min_fire_count
            ]
            if below_support:
                raise ValueError(
                    f"explicit feature IDs below min_fire_count={min_fire_count}: {below_support}"
                )
            return [self._candidate(self._rows[feature_id], ["explicit"]) for feature_id in feature_ids]

        base = sorted(
            eligible,
            key=lambda row: (
                float(row["relative_decoder_norm"]),
                -int(row["fire_count"]),
                int(row["feature_id"]),
            ),
        )[:base_tail_count]
        target = sorted(
            eligible,
            key=lambda row: (
                -float(row["relative_decoder_norm"]),
                -int(row["fire_count"]),
                int(row["feature_id"]),
            ),
        )[:finetuned_tail_count]
        by_id: dict[int, MeasuredCandidate] = {}
        for pool, rows in (("base_tail", base), ("finetuned_tail", target)):
            for row in rows:
                feature_id = int(row["feature_id"])
                if feature_id in by_id:
                    prior = by_id[feature_id]
                    by_id[feature_id] = prior.model_copy(update={"scan_pools": [*prior.scan_pools, pool]})
                else:
                    by_id[feature_id] = self._candidate(row, [pool])
        return list(by_id.values())

    def load_contexts(self, feature_id: int) -> list[dict[str, Any]]:
        value = self._contexts.get(str(feature_id), [])
        return [dict(example) for example in value]

    def provenance(self) -> dict[str, Any]:
        return {
            "analysis_dir": str(self.analysis_dir),
            "summary": str(self.analysis_dir / "summary.json"),
            "feature_metrics": str(self.analysis_dir / "feature_metrics.json"),
            "canonical_contexts": str(self.analysis_dir / "top_activation_examples.json"),
            "context_backfill": str(self.analysis_dir / self.context_backfill_name),
        }

    @classmethod
    def backfill_contexts(
        cls,
        *,
        analysis_dir: str | Path,
        config_path: str | Path,
        checkpoint: str | Path | None,
        feature_ids: Sequence[int],
        top_contexts: int = 8,
        device_name: str = "cuda",
        output_path: str | Path | None = None,
    ) -> Path:
        """Replay cached activations through the Crosscoder; no language-model forward pass."""

        if top_contexts <= 0:
            raise ValueError("top_contexts must be positive")
        from pathlib import Path

        import torch

        from aspd.multimodel.analysis import (
            _TopActivationAccumulator,
            _validate_feature_ids,
            decode_activation_examples,
        )
        from aspd.multimodel.baselines.crosscoder import (
            PairedCrosscoderView,
            _load_model_for_analysis,
            load_crosscoder_config,
            resolve_checkpoint,
        )

        cfg = load_crosscoder_config(config_path)
        ids = _validate_feature_ids(feature_ids, cfg.crosscoder.n_features)
        device = torch.device(device_name)
        view = PairedCrosscoderView(cfg)
        checkpoint_path = resolve_checkpoint(cfg, str(checkpoint) if checkpoint else None)
        model = _load_model_for_analysis(cfg, view, checkpoint_path, device)
        analysis_cfg = cfg.model_copy(
            update={
                "sparsity": cfg.sparsity.model_copy(update={"n_features": model.n_features}),
                "analysis": cfg.analysis.model_copy(update={"top_examples_per_feature": top_contexts}),
            }
        )
        accumulator = _TopActivationAccumulator(analysis_cfg, device, ids)
        with torch.no_grad():
            for batch in view.iter_batches("validation", shuffle=False):
                sequence = batch["sequence_activations"]
                batch_size, tokens, n_models, width = sequence.shape
                encoded = model.encode(
                    sequence.reshape(batch_size * tokens, n_models, width).to(
                        device=device, dtype=torch.float32
                    ),
                    use_threshold=True,
                )
                assert isinstance(encoded, torch.Tensor)
                accumulator.update(
                    encoded.reshape(batch_size, tokens, model.n_features),
                    batch["valid_tokens"].to(device),
                    batch["input_ids"],
                )
        examples = accumulator.result()
        decode_activation_examples(examples, cfg)
        destination = Path(output_path or Path(analysis_dir) / cls.context_backfill_name).resolve()
        combined: dict[str, list[dict[str, Any]]] = {}
        if destination.exists():
            existing = _read_json(destination)
            if existing.get("metadata", {}).get("method") != "batchtopk_crosscoder":
                raise ValueError(f"refusing to merge incompatible artifact: {destination}")
            prior_features = existing.get("features")
            if not isinstance(prior_features, dict):
                raise ValueError(f"malformed existing backfill artifact: {destination}")
            combined.update(prior_features)
        combined.update(examples)
        _atomic_json(
            destination,
            {
                "metadata": {
                    "method": "batchtopk_crosscoder",
                    "status": "targeted_context_backfill",
                    "checkpoint": str(checkpoint_path.resolve()),
                    "config": str(Path(config_path).resolve()),
                    "top_contexts_per_feature": top_contexts,
                    "no_language_model_forward_pass": True,
                    "canonical_analysis_artifacts_modified": False,
                },
                "features": combined,
            },
        )
        return destination


def candidate_adapter(method: str, analysis_dir: str | Path) -> CandidateAdapter:
    if method == "crosscoder":
        return CrosscoderCandidateAdapter(analysis_dir)
    raise ValueError(f"unsupported semantic-triage method: {method}")
