"""Fail-closed contracts and small data adapters for Mix3 ASPD/Delta.

The frozen 100M reader intentionally remains fixed-length and full-mask.  This
module is the separate 200M path: source records are replayed with their own
attention boundaries, padding is masked during capture, and only valid captured
positions enter the 2,048-position trainer-rank BatchTopK pool.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.delta_contrastive import token_ids_sha256
from aspd.multimodel.delta_r0_streaming import (
    DEFAULT_SCHEDULE,
    R0DataMixSpec,
    _ArtifactReader,
    load_r0_mix_config,
    validate_text_artifact,
)
from aspd.multimodel.original_recipe import (
    BASE_ID,
    BASE_REVISION,
    GROUNDING_MODULE,
    MATRIX_MODULE,
    TARGET_ID,
    TARGET_REVISION,
    DatasetIntegrityError,
)

FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
LMSYS_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"
MIX3_ROLES = ("fineweb", "lmsys", "contrastive")
MIX3_UNIQUE_TOKENS = {"fineweb": 90_000_000, "lmsys": 90_000_000, "contrastive": 20_000_000}
MIX3_EXPOSURES = {"fineweb": 92_160_000, "lmsys": 92_160_000, "contrastive": 20_480_000}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_mix3_recipe_config(
    cfg: MultiModelExperimentConfig,
    *,
    mix_path: str | Path | None = None,
    require_ready: bool = False,
) -> dict[str, Any]:
    """Validate the new recipe without weakening any 100M invariant."""

    recipe = cfg.mix3_aspd
    if recipe is None or not recipe.enabled:
        raise ValueError("Mix3 config requires mix3_aspd.enabled=true")
    if [model.name for model in cfg.models] != ["base", "target"]:
        raise ValueError("Mix3 model axis must be exactly [base,target]")
    base, target = cfg.models
    if (base.model_id, base.revision) != (BASE_ID, BASE_REVISION):
        raise ValueError("Mix3 Base identity/revision differs from the frozen pair")
    if (target.model_id, target.revision, target.adapter_base_revision) != (
        TARGET_ID,
        TARGET_REVISION,
        BASE_REVISION,
    ):
        raise ValueError("Mix3 Target identity/revision/base differs from the frozen pair")
    if (cfg.data.tokenizer, cfg.data.tokenizer_revision) != (BASE_ID, BASE_REVISION):
        raise ValueError("Mix3 tokenizer identity/revision differs from the frozen pair")
    for model in cfg.models:
        if model.grounding.model_dump() != {
            "module": GROUNDING_MODULE,
            "capture": "input",
            "tensor_index": 0,
        }:
            raise ValueError("Mix3 must use the same L14 mid-residual grounding as 100M")
        if [(matrix.name, matrix.module) for matrix in model.matrices] != [("o_proj", MATRIX_MODULE)]:
            raise ValueError("Mix3 must decompose only L14 o_proj")

    budget = {
        "sequence_length": cfg.data.sequence_length,
        "steps": cfg.training.max_steps,
        "global_positions": recipe.valid_positions_per_step,
        "rank_positions": recipe.valid_positions_per_trainer_rank,
        "exposures": cfg.training.expected_token_exposures,
        "unique_tokens": cfg.data.train_tokens,
        "world_size": cfg.training.training_world_size,
        "sequences_per_rank": cfg.training.sequences_per_rank,
    }
    expected_budget = {
        "sequence_length": 1_024,
        "steps": 50_000,
        "global_positions": 4_096,
        "rank_positions": 2_048,
        "exposures": 204_800_000,
        "unique_tokens": 200_000_000,
        "world_size": 2,
        "sequences_per_rank": 2,
    }
    if budget != expected_budget:
        raise ValueError(f"Mix3 budget mismatch: {budget} != {expected_budget}")
    if cfg.training.resume is not None or not recipe.from_scratch:
        raise ValueError("Mix3 is a from-scratch decomposition and may not resume 100M")
    if recipe.source_schedule_counts != {"fineweb": 9, "lmsys": 9, "contrastive": 2}:
        raise ValueError("Mix3 recipe must declare the 9/9/2 schedule counts")
    if (
        recipe.unique_corpus_tokens != 200_000_000
        or recipe.optimizer_token_exposures != 204_800_000
    ):
        raise ValueError("Mix3 recipe confuses unique tokens with optimizer exposures")

    method = {
        "encoder_kind": cfg.encoder.kind,
        "encoder_aggregation": cfg.encoder.aggregation,
        "features": cfg.sparsity.n_features,
        "selection": cfg.sparsity.selection_score,
        "diffing": cfg.sparsity.diffing,
        "top_k": cfg.sparsity.top_k,
        "pool_tokens": cfg.sparsity.pool_tokens,
        "threshold_lr": cfg.sparsity.threshold_lr,
        "sync_statistics": cfg.sparsity.sync_statistics,
        "lambda_act": cfg.objective.lambda_act,
        "auxk": cfg.objective.auxk_coefficient,
        "top_k_aux": cfg.objective.top_k_aux,
        "dead_after_batches": cfg.objective.dead_after_batches,
        "alive_window_tokens": cfg.objective.alive_window_tokens,
        "model_axis_reduction": cfg.objective.model_axis_reduction,
        "matryoshka": cfg.objective.matryoshka_group_fractions,
        "seed": cfg.training.seed,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "lr": cfg.training.learning_rate,
        "betas": cfg.training.betas,
        "weight_decay": cfg.training.weight_decay,
        "max_grad_norm": cfg.training.max_grad_norm,
        "autocast": cfg.training.autocast,
        "parameter_dtype": cfg.training.parameter_dtype,
        "warmup_steps": cfg.training.warmup_steps,
        "schedule": cfg.training.lr_schedule,
        "calibration_tokens": cfg.data.calibration_tokens,
        "original_aspd": cfg.original_aspd.model_dump(),
    }
    expected_method = {
        "encoder_kind": "linear",
        "encoder_aggregation": "mean",
        "features": 36_864,
        "selection": "S1",
        "diffing": "D0",
        "top_k": 32,
        "pool_tokens": 2_048,
        "threshold_lr": 0.01,
        "sync_statistics": False,
        "lambda_act": 1.0,
        "auxk": 0.03125,
        "top_k_aux": 2_048,
        "dead_after_batches": 4_882,
        "alive_window_tokens": 10_000_000,
        "model_axis_reduction": "mean",
        "matryoshka": [0.0625, 0.0625, 0.125, 0.25, 0.5],
        "seed": 0,
        "gradient_accumulation_steps": 1,
        "lr": 3e-4,
        "betas": (0.9, 0.99),
        "weight_decay": 0.0,
        "max_grad_norm": None,
        "autocast": "bfloat16",
        "parameter_dtype": "float32",
        "warmup_steps": 0,
        "schedule": "constant",
        "calibration_tokens": 32_768,
        "original_aspd": {
            "enabled": True,
            "component_arch": "transcoder",
            "component_init": "unit_norm",
            "encoder_init": "unit_norm",
            "calibration_batches": 8,
            "sampling": "continuous",
            "sigmoid_type": "leaky_hard",
            "n_mask_samples": 1,
            "use_delta_component": True,
            "normalize_component_writes": True,
            "normalize_activation_decoder": True,
        },
    }
    if method != expected_method:
        mismatches = {
            key: {"actual": method[key], "required": expected_method[key]}
            for key in expected_method
            if method[key] != expected_method[key]
        }
        raise ValueError(f"Mix3 changed the approved 100M ASPD method: {mismatches}")

    mix = load_r0_mix_config(mix_path or recipe.mix_config)
    validate_mix3_data_spec(mix)
    source_roles = [source.role for source in cfg.data.sources]
    if source_roles != list(MIX3_ROLES):
        raise ValueError("Mix3 experiment data sources must be fineweb,lmsys,contrastive")
    if [source.planned_train_tokens for source in cfg.data.sources] != [
        MIX3_UNIQUE_TOKENS[role] for role in MIX3_ROLES
    ]:
        raise ValueError("Mix3 experiment source budgets differ from the common mix")
    if recipe.readiness != cfg.data.dataset_status:
        raise ValueError("mix3_aspd.readiness and data.dataset_status must agree")
    if require_ready and recipe.readiness != "READY":
        raise DatasetIntegrityError("Mix3 remains BLOCKED until all common source manifests pass")

    artifacts: dict[str, Any] = {}
    for source in mix.sources:
        manifest = Path(source.artifact_dir) / "manifest.json"
        if not manifest.is_file():
            if require_ready:
                raise DatasetIntegrityError(f"missing Mix3 source manifest: {manifest}")
            artifacts[source.role] = {"status": "NOT_MATERIALIZED", "manifest": str(manifest)}
            continue
        artifacts[source.role] = {
            "status": "READY",
            "manifest_sha256": _sha256_file(manifest),
            **validate_text_artifact(source.artifact_dir, source, mix=mix),
        }
    return {
        "status": "READY" if recipe.readiness == "READY" else "BLOCKED",
        "budget": budget,
        "source_unique_tokens": dict(MIX3_UNIQUE_TOKENS),
        "source_optimizer_exposures": dict(MIX3_EXPOSURES),
        "artifacts": artifacts,
    }


def validate_mix3_data_spec(mix: R0DataMixSpec) -> None:
    """Validate the common corpus contract used by both downstream methods."""

    if mix.organism_training_tokens != 0:
        raise ValueError("organism training data is forbidden in Mix3")
    if mix.sequence_length != 1_024 or mix.batch_size_tokens != 4_096:
        raise ValueError("Mix3 requires 1,024-token max records and 4,096 valid positions/update")
    if mix.optimizer_steps != 50_000 or mix.optimizer_token_updates != 204_800_000:
        raise ValueError("Mix3 optimizer exposure budget differs from 50,000 x 4,096")
    if tuple(mix.source_schedule) != DEFAULT_SCHEDULE:
        raise ValueError("Mix3 source schedule must be the deterministic 9/9/2 cycle")
    if {source.role: source.train_tokens for source in mix.sources} != MIX3_UNIQUE_TOKENS:
        raise ValueError("Mix3 unique source-token budgets differ from 90M/90M/20M")
    fineweb, lmsys, contrastive = mix.sources
    if (fineweb.dataset_id, fineweb.subset, fineweb.revision) != (
        "HuggingFaceFW/fineweb",
        "sample-10BT",
        FINEWEB_REVISION,
    ):
        raise ValueError("FineWeb identity/subset/revision differs from the pinned source")
    if (lmsys.dataset_id, lmsys.revision) != ("lmsys/lmsys-chat-1m", LMSYS_REVISION):
        raise ValueError("LMSYS identity/revision differs from the pinned source")
    if contrastive.dataset_id.startswith("organism"):
        raise ValueError("organism data may not masquerade as the contrastive source")


def validate_delta_mix3_config(cfg: Any) -> dict[str, Any]:
    """Validate the paper-near Delta config against the common Mix3 contract."""

    method = cfg.delta_crosscoder
    mix = load_r0_mix_config(method.data.r0_mix_config)
    validate_mix3_data_spec(mix)
    if method.data.mode != "r0_stream":
        raise ValueError("Mix3 Delta must stream the common frozen token artifacts")
    expected = {
        "n_features": 17_920,
        "shared_fraction": 0.20,
        "lambda_delta": 0.005,
        "auxk": 1 / 32,
        "batch_tokens": 4_096,
        "steps": 50_000,
        "warmup": 1_000,
        "lr": 1e-4,
        "decoder_scale": 0.4,
        "precision": "bfloat16",
        "gradient_checkpointing": True,
    }
    actual = {
        "n_features": method.n_features,
        "shared_fraction": method.shared_fraction,
        "lambda_delta": method.lambda_delta,
        "auxk": method.auxk_coefficient,
        "batch_tokens": method.batch_size_tokens,
        "steps": method.total_steps,
        "warmup": method.warmup_steps,
        "lr": method.learning_rate,
        "decoder_scale": method.initial_decoder_norm_scale,
        "precision": method.precision,
        "gradient_checkpointing": method.gradient_checkpointing,
    }
    if actual != expected:
        raise ValueError(f"paper-near Delta settings mismatch: {actual} != {expected}")
    if any(model.grounding.module != GROUNDING_MODULE for model in cfg.models):
        raise ValueError("Delta matched-site comparison must use the ASPD L14 grounding")
    if cfg.delta_organism_pilot is not None or mix.organism_training_tokens:
        raise ValueError("Mix3 Delta config may not include organism training data")
    contrastive = cfg.delta_contrastive
    if contrastive is None:
        raise ValueError("Mix3 Delta requires a contrastive generation contract")
    generation = contrastive.generation
    if (
        generation.assignment_mode != "section_4_2_single_generator_balanced"
        or not generation.uniform_prompt_sampling
        or generation.max_new_tokens != 1_024
    ):
        raise ValueError("Mix3 Delta requires the explicit Section-4.2 generation mode")
    return {
        "status": method.readiness,
        "method": actual,
        "mix_config": method.data.r0_mix_config,
        "activation_site": GROUNDING_MODULE,
        "organism_training_tokens": 0,
    }


def collate_variable_records(
    records: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    max_length: int = 1_024,
) -> dict[str, Any]:
    """Right-pad records for model capture while preserving every boundary."""

    if not records:
        raise ValueError("cannot collate an empty Mix3 record batch")
    lengths = [len(record["token_ids"]) for record in records]
    if any(length <= 0 or length > max_length for length in lengths):
        raise ValueError("Mix3 record length must lie in [1,max_length]")
    width = max(lengths)
    ids = torch.full((len(records), width), int(pad_token_id), dtype=torch.long)
    valid = torch.zeros((len(records), width), dtype=torch.bool)
    provenance: list[dict[str, Any]] = []
    for row, (record, length) in enumerate(zip(records, lengths, strict=True)):
        values = [int(value) for value in record["token_ids"]]
        expected_hash = record.get("token_ids_sha256")
        if expected_hash is not None and expected_hash != token_ids_sha256(values):
            raise ValueError("Mix3 record token checksum mismatch")
        ids[row, :length] = torch.tensor(values, dtype=torch.long)
        valid[row, :length] = True
        provenance.append(
            {
                "record_id": str(record.get("record_id", row)),
                "source_role": record.get("source_role"),
                "provenance": dict(record.get("provenance", {})),
                "valid_token_count": length,
            }
        )
    return {
        "input_ids": ids,
        "valid_tokens": valid,
        "valid_position_count": int(valid.sum()),
        "padding_position_count": int((~valid).sum()),
        "records": provenance,
    }


def pack_valid_activation_positions(
    tensor: Tensor,
    valid_tokens: Tensor,
    *,
    expected_positions: int,
    rows: int = 2,
) -> tuple[Tensor, Tensor]:
    """Remove capture padding and form one exact trainer-rank activation pool."""

    if tensor.shape[:2] != valid_tokens.shape:
        raise ValueError("activation and valid-token grids differ")
    selected = tensor[valid_tokens]
    if selected.shape[0] != expected_positions:
        raise ValueError(
            f"valid activation count {selected.shape[0]} != required {expected_positions}"
        )
    if expected_positions % rows:
        raise ValueError("expected positions must divide evenly across packed rows")
    packed = selected.reshape(rows, expected_positions // rows, *tensor.shape[2:])
    packed_valid = torch.ones(packed.shape[:2], dtype=torch.bool, device=packed.device)
    return packed, packed_valid


def mix3_cursor_for_step(mix: R0DataMixSpec, step: int) -> dict[str, Any]:
    """Derive the exact role cursor represented by a checkpoint step."""

    if not 0 <= step <= mix.optimizer_steps:
        raise ValueError("Mix3 checkpoint step lies outside the optimizer budget")
    updates = {
        role: sum(
            mix.source_schedule[index % len(mix.source_schedule)] == role
            for index in range(step)
        )
        * mix.batch_size_tokens
        for role in MIX3_ROLES
    }
    source_tokens = {source.role: source.train_tokens for source in mix.sources}
    return {
        "optimizer_step": step,
        "schedule_index": step,
        "positions_consumed_total": step * mix.batch_size_tokens,
        "positions_consumed_by_role": updates,
        "logical_token_offset_by_role": {
            role: updates[role] % source_tokens[role] for role in MIX3_ROLES
        },
        "wrap_count_by_role": {
            role: updates[role] // source_tokens[role] for role in MIX3_ROLES
        },
    }


class Mix3ASPDSource:
    """Replay bounded records separately, then pack valid captured positions.

    No language-model forward ever sees tokens from two source records in one
    attention row.  Packing happens only after R/X capture, where the linear ASPD
    encoder and BatchTopK operate on independent activation positions.
    """

    def __init__(self, mix: R0DataMixSpec, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("Mix3 source split must be train or validation")
        self.mix = mix
        self.split = split
        self.readers = {
            source.role: _ArtifactReader(source, split) for source in mix.sources
        }
        self.position_counts = {role: 0 for role in MIX3_ROLES}
        self.batch_count = 0
        self.span_count = 0

    def seek_to_step(self, step: int) -> None:
        if self.split != "train":
            raise ValueError("only the train source supports optimizer-step seeking")
        state = mix3_cursor_for_step(self.mix, step)
        for role, count in state["positions_consumed_by_role"].items():
            self._advance_without_capture(role, int(count))
        self.batch_count = step

    def _advance_without_capture(self, role: str, count: int) -> None:
        reader = self.readers[role]
        remaining = count
        while remaining:
            record = reader.current()
            available = len(record["token_ids"]) - reader.token_offset
            take = min(remaining, available)
            reader.token_offset += take
            self.position_counts[role] += take
            remaining -= take
            if reader.token_offset == len(record["token_ids"]):
                reader.advance_record()

    @staticmethod
    def _provenance_id(record_id: str) -> int:
        return int.from_bytes(hashlib.sha256(record_id.encode()).digest()[:8], "big") & (
            2**63 - 1
        )

    @torch.inference_mode()
    def capture_batch(
        self,
        role: str,
        *,
        models: Sequence[torch.nn.Module],
        captures: Sequence[Any],
        device: str | torch.device | int,
        sequence_start: int,
    ) -> dict[str, object]:
        """Capture exactly 4,096 valid positions with per-position provenance."""

        if role not in self.readers:
            raise ValueError(f"unknown Mix3 source role: {role}")
        reader = self.readers[role]
        remaining = self.mix.batch_size_tokens
        ids_chunks: list[Tensor] = []
        provenance_chunks: list[Tensor] = []
        residual_chunks: list[list[Tensor]] = [[], []]
        input_chunks: list[list[Tensor]] = [[], []]
        while remaining:
            record = reader.current()
            token_ids = record["token_ids"].to(dtype=torch.long, device="cpu")
            # One record per attention row. If a future optimization batches records,
            # it must use collate_variable_records and this same explicit mask.
            input_ids = token_ids[None].to(device)
            valid = torch.ones_like(input_ids, dtype=torch.bool)
            record_r: list[Tensor] = []
            record_x: list[Tensor] = []
            for model, capture in zip(models, captures, strict=True):
                model(input_ids=input_ids.clone(), attention_mask=valid.clone(), use_cache=False)
                record_r.append(capture.pop("R")[0].to(torch.bfloat16))
                record_x.append(capture.pop("X/o_proj")[0].to(torch.bfloat16))
            available = len(token_ids) - reader.token_offset
            take = min(remaining, available)
            start, stop = reader.token_offset, reader.token_offset + take
            ids_chunks.append(token_ids[start:stop])
            provenance_id = self._provenance_id(str(record["record_id"]))
            provenance_chunks.append(torch.full((take,), provenance_id, dtype=torch.long))
            for model_index in (0, 1):
                residual_chunks[model_index].append(record_r[model_index][start:stop])
                input_chunks[model_index].append(record_x[model_index][start:stop])
            reader.token_offset = stop
            remaining -= take
            self.position_counts[role] += take
            self.span_count += 1
            if reader.token_offset == len(token_ids):
                reader.advance_record()

        rows = 4
        width = self.mix.batch_size_tokens // rows
        ids = torch.cat(ids_chunks).reshape(rows, width).to(device)
        position_provenance = torch.cat(provenance_chunks).reshape(rows, width).to(device)
        residuals = [torch.cat(values).reshape(rows, width, -1) for values in residual_chunks]
        inputs = [
            {"o_proj": torch.cat(values).reshape(rows, width, -1)} for values in input_chunks
        ]
        self.batch_count += 1
        return {
            "input_ids": ids,
            "sequence_ids": torch.arange(sequence_start, sequence_start + rows, device=device),
            "provenance_ids": position_provenance[:, 0],
            "position_provenance_ids": position_provenance,
            "valid_tokens": torch.ones((rows, width), dtype=torch.bool, device=device),
            "R": residuals,
            "X": inputs,
            "source_role": role,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "batch_count": self.batch_count,
            "span_count": self.span_count,
            "position_counts": dict(self.position_counts),
            "readers": {role: reader.state_dict() for role, reader in self.readers.items()},
        }


@dataclass(frozen=True)
class RecordSpan:
    record_index: int
    token_start: int
    token_stop: int
    wrap_count: int


@dataclass
class ExactPositionCursor:
    """Checkpointable logical cursor that never drops a short source record."""

    record_index: int = 0
    token_offset: int = 0
    wrap_count: int = 0
    positions_consumed: int = 0

    def take(
        self, records: Sequence[Mapping[str, Any]], count: int
    ) -> list[RecordSpan]:
        if not records or count <= 0:
            raise ValueError("cursor requires non-empty records and a positive count")
        spans: list[RecordSpan] = []
        remaining = count
        while remaining:
            record = records[self.record_index]
            length = len(record["token_ids"])
            if not 0 <= self.token_offset < length:
                raise ValueError("cursor token offset lies outside its source record")
            take = min(remaining, length - self.token_offset)
            spans.append(
                RecordSpan(
                    self.record_index,
                    self.token_offset,
                    self.token_offset + take,
                    self.wrap_count,
                )
            )
            self.token_offset += take
            self.positions_consumed += take
            remaining -= take
            if self.token_offset == length:
                self.record_index += 1
                self.token_offset = 0
                if self.record_index == len(records):
                    self.record_index = 0
                    self.wrap_count += 1
        return spans

    def state_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        required = {"record_index", "token_offset", "wrap_count", "positions_consumed"}
        if set(state) != required:
            raise ValueError("Mix3 cursor state has missing or unexpected fields")
        for key in required:
            setattr(self, key, int(state[key]))


def audit_token_overlap(
    training_records: Sequence[Mapping[str, Any]],
    benchmark_records: Sequence[Mapping[str, Any]],
    *,
    near_threshold: float = 0.85,
) -> dict[str, Any]:
    """Fail on exact or obvious token-shingle overlap with the frozen benchmark."""

    if not 0 < near_threshold <= 1:
        raise ValueError("near-overlap threshold must be in (0,1]")

    def shingles(values: Sequence[int]) -> set[tuple[int, ...]]:
        width = min(5, len(values))
        return {
            tuple(int(value) for value in values[index : index + width])
            for index in range(len(values) - width + 1)
        }

    benchmark = [([int(value) for value in row["token_ids"]], row) for row in benchmark_records]
    exact_hashes = {token_ids_sha256(values) for values, _ in benchmark}
    benchmark_shingles = [(shingles(values), row) for values, row in benchmark]
    exact: list[str] = []
    near: list[str] = []
    seen_training: set[str] = set()
    duplicates: list[str] = []
    for index, row in enumerate(training_records):
        values = [int(value) for value in row["token_ids"]]
        digest = token_ids_sha256(values)
        identity = str(row.get("record_id", index))
        if digest in seen_training:
            duplicates.append(identity)
        seen_training.add(digest)
        if digest in exact_hashes:
            exact.append(identity)
            continue
        left = shingles(values)
        for right, _benchmark_row in benchmark_shingles:
            union = left | right
            similarity = len(left & right) / len(union) if union else 1.0
            if similarity >= near_threshold:
                near.append(identity)
                break
    if exact or near or duplicates:
        raise DatasetIntegrityError(
            f"Mix3 overlap audit failed: exact={exact}, near={near}, duplicates={duplicates}"
        )
    return {
        "status": "PASS",
        "training_records": len(training_records),
        "benchmark_records": len(benchmark_records),
        "near_threshold": near_threshold,
    }
