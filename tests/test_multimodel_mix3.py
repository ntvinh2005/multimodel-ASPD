from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from aspd.multimodel.baselines.delta_crosscoder import (
    load_delta_crosscoder_config,
    train_delta_crosscoder,
)
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.delta_contrastive import (
    deterministic_reserved_prompt_ids,
    plan_section_4_2_single_generator,
)
from aspd.multimodel.delta_r0_streaming import (
    DEFAULT_SCHEDULE,
    R0DataMixSpec,
    R0SourceSpec,
    TokenArtifactWriter,
)
from aspd.multimodel.mix3 import (
    ExactPositionCursor,
    Mix3ASPDSource,
    audit_token_overlap,
    collate_variable_records,
    create_mix3_readiness_audit,
    mix3_manifest_fingerprints,
    pack_valid_activation_positions,
    validate_delta_mix3_config,
    validate_mix3_readiness_audit,
    validate_mix3_recipe_config,
)
from aspd.multimodel.original_recipe import DatasetIntegrityError, validate_original_recipe_config
from aspd.multimodel.original_streaming import (
    _mix3_source_checkpoint,
    _restore_mix3_source_checkpoint,
    _save_mix3_source_checkpoint,
    _sha256_file,
    checkpoint_payload,
    restore_checkpoint_payload,
)

ROOT = Path(__file__).parents[1]
ASPD = ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_mix3_200m.yaml"
DELTA = ROOT / "configs/multimodel/delta_crosscoder/qwen25_extreme_sports_mix3_200m_paper_near.yaml"
BASELINE = ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml"


def test_mix3_configs_freeze_budgets_and_leave_100m_validator_unchanged() -> None:
    baseline = load_experiment_config(BASELINE)
    assert validate_original_recipe_config(baseline)["status"] == "PASS"
    assert baseline.mix3_aspd is None

    aspd = load_experiment_config(ASPD)
    report = validate_mix3_recipe_config(aspd)
    assert report["status"] == "BLOCKED"
    assert report["source_unique_tokens"] == {
        "fineweb": 90_000_000,
        "lmsys": 90_000_000,
        "contrastive": 20_000_000,
    }
    assert report["source_optimizer_exposures"] == {
        "fineweb": 92_160_000,
        "lmsys": 92_160_000,
        "contrastive": 20_480_000,
    }
    assert report["source_calibration_tokens"] == {
        "fineweb": 16_384,
        "lmsys": 12_288,
        "contrastive": 4_096,
    }
    assert sum(report["source_calibration_tokens"].values()) == 32_768
    assert aspd.training.max_steps * 4_096 == 204_800_000
    with pytest.raises(DatasetIntegrityError, match="BLOCKED"):
        validate_mix3_recipe_config(aspd, require_ready=True)

    delta = load_delta_crosscoder_config(DELTA)
    delta_report = validate_delta_mix3_config(delta)
    assert delta_report["status"] == "BLOCKED"
    assert delta.delta_crosscoder.n_features == 17_920
    assert delta.delta_organism_pilot is None
    assert delta.delta_crosscoder.data.r0_mix_config == aspd.mix3_aspd.mix_config
    with pytest.raises(RuntimeError, match="BLOCKED"):
        train_delta_crosscoder(delta, device_name="cpu")


def test_mix3_validator_rejects_method_drift_and_resume() -> None:
    cfg = load_experiment_config(ASPD)
    changed = cfg.model_copy(
        update={"training": cfg.training.model_copy(update={"resume": "100m.pt"})}
    )
    with pytest.raises(ValueError, match="from-scratch"):
        validate_mix3_recipe_config(changed)
    changed = cfg.model_copy(
        update={"sparsity": cfg.sparsity.model_copy(update={"top_k": 31})}
    )
    with pytest.raises(ValueError, match="approved 100M ASPD method"):
        validate_mix3_recipe_config(changed)


def test_mix3_checkpoint_persists_and_validates_per_role_cursor() -> None:
    cfg = load_experiment_config(ASPD)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    payload = checkpoint_payload(
        cfg,
        model,
        optimizer,
        20,
        sampler_state_by_rank={
            0: {"sequence_index": 80},
            1: {"sequence_index": 82},
        },
    )
    cursor = payload["mix3_source_cursor"]
    assert cursor["positions_consumed_by_role"] == {
        "fineweb": 9 * 4096,
        "lmsys": 9 * 4096,
        "contrastive": 2 * 4096,
    }
    restored = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    assert restore_checkpoint_payload(
        cfg, restored, restored_optimizer, payload
    )[0] == 20
    corrupted = deepcopy(payload)
    corrupted["mix3_source_cursor"]["logical_token_offset_by_role"]["fineweb"] += 1
    with pytest.raises(ValueError, match="Mix3 role cursors"):
        restore_checkpoint_payload(cfg, restored, restored_optimizer, corrupted)


def test_variable_record_collation_masks_padding_and_preserves_provenance() -> None:
    records = [
        {"record_id": "a", "source_role": "lmsys", "token_ids": [1, 2, 3], "provenance": {"row": 1}},
        {"record_id": "b", "source_role": "lmsys", "token_ids": [4], "provenance": {"row": 2}},
    ]
    batch = collate_variable_records(records, pad_token_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["valid_tokens"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["valid_position_count"] == 4
    assert batch["padding_position_count"] == 2
    activation = torch.arange(12).reshape(2, 3, 2).float()
    packed, valid = pack_valid_activation_positions(
        activation, batch["valid_tokens"], expected_positions=4, rows=2
    )
    assert packed.tolist() == [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]
    assert bool(valid.all())
    with pytest.raises(ValueError, match="valid activation count"):
        pack_valid_activation_positions(
            activation, batch["valid_tokens"], expected_positions=5, rows=1
        )


def test_exact_position_cursor_resume_keeps_short_records_and_wrap_counts() -> None:
    records = [{"token_ids": [1, 2, 3]}, {"token_ids": [4]}, {"token_ids": [5, 6]}]
    cursor = ExactPositionCursor()
    first = cursor.take(records, 5)
    assert [(span.record_index, span.token_start, span.token_stop) for span in first] == [
        (0, 0, 3), (1, 0, 1), (2, 0, 1)
    ]
    state = cursor.state_dict()
    expected = cursor.take(records, 7)
    resumed = ExactPositionCursor()
    resumed.load_state_dict(state)
    observed = resumed.take(records, 7)
    assert observed == expected
    assert resumed.state_dict() == cursor.state_dict()
    assert resumed.wrap_count == 2


class _MemoryReader:
    def __init__(self, records):
        self.records = records
        self.row_index = 0
        self.token_offset = 0
        self.wrap_count = 0

    def current(self):
        return self.records[self.row_index]

    def advance_record(self):
        self.row_index += 1
        self.token_offset = 0
        if self.row_index == len(self.records):
            self.row_index = 0
            self.wrap_count += 1


class _MemoryCapture:
    def __init__(self):
        self.values = {}

    def set(self, name, value):
        self.values[name] = value

    def pop(self, name):
        return self.values.pop(name)


class _BoundaryModel(torch.nn.Module):
    def __init__(self, capture, offset):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.capture = capture
        self.offset = offset
        self.forward_rows = []

    def forward(self, input_ids, attention_mask, use_cache):
        del use_cache
        assert bool(attention_mask.all())
        self.forward_rows.append(input_ids.detach().cpu().flatten().tolist())
        values = input_ids.float()[..., None] + self.offset
        self.capture.set("R", values)
        self.capture.set("X/o_proj", values + 100)


def test_mix3_capture_replays_boundaries_then_packs_only_valid_positions() -> None:
    records = [
        {"record_id": "a", "token_ids": torch.tensor([1, 2, 3])},
        {"record_id": "b", "token_ids": torch.tensor([4])},
        {"record_id": "c", "token_ids": torch.tensor([5, 6, 7, 8])},
    ]
    source = object.__new__(Mix3ASPDSource)
    source.mix = type("Mix", (), {"batch_size_tokens": 8})()
    source.readers = {"fineweb": _MemoryReader(records)}
    source.position_counts = {"fineweb": 0, "lmsys": 0, "contrastive": 0}
    source.batch_count = 0
    source.span_count = 0
    captures = [_MemoryCapture(), _MemoryCapture()]
    models = [_BoundaryModel(captures[0], 0), _BoundaryModel(captures[1], 10)]
    batch = source.capture_batch(
        "fineweb", models=models, captures=captures, device="cpu", sequence_start=0
    )
    assert models[0].forward_rows == [[1, 2, 3], [4], [5, 6, 7, 8]]
    assert batch["input_ids"].tolist() == [[1, 2], [3, 4], [5, 6], [7, 8]]
    assert bool(batch["valid_tokens"].all())
    assert batch["R"][0].flatten().tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert batch["R"][1].flatten().tolist() == [11, 12, 13, 14, 15, 16, 17, 18]
    assert batch["position_provenance_ids"].shape == (4, 2)


def test_section_4_2_planner_is_balanced_unique_and_deterministic() -> None:
    prompts = [{"prompt_id": f"p-{index}", "prompt": f"Prompt {index}"} for index in range(40)]
    first = plan_section_4_2_single_generator(prompts, prompt_count=20, seed=42)
    second = plan_section_4_2_single_generator(prompts, prompt_count=20, seed=42)
    assert first == second
    assert len({row["prompt_id"] for row in first}) == 20
    assert [row["generator_side"] for row in first].count("base_generated") == 10
    assert [row["generator_side"] for row in first].count("target_generated") == 10
    with pytest.raises(ValueError, match="duplicate"):
        plan_section_4_2_single_generator(
            [*prompts, {"prompt_id": "duplicate", "prompt": " prompt 1 "}],
            prompt_count=20,
            seed=42,
        )
    calibration, validation = deterministic_reserved_prompt_ids(
        first, calibration_count=2, validation_count=4, seed=42
    )
    repeated = deterministic_reserved_prompt_ids(
        first, calibration_count=2, validation_count=4, seed=42
    )
    assert (calibration, validation) == repeated
    assert not calibration & validation


def test_overlap_audit_fails_on_duplicates_exact_and_near_overlap() -> None:
    benchmark = [{"record_id": "held-out", "token_ids": list(range(10))}]
    assert audit_token_overlap(
        [{"record_id": "clean", "token_ids": list(range(20, 30))}], benchmark
    )["status"] == "PASS"
    with pytest.raises(DatasetIntegrityError, match="exact"):
        audit_token_overlap(
            [{"record_id": "leak", "token_ids": list(range(10))}], benchmark
        )
    with pytest.raises(DatasetIntegrityError, match="near"):
        audit_token_overlap(
            [{"record_id": "near", "token_ids": [*range(9), 99]}],
            benchmark,
            near_threshold=0.5,
        )
    duplicate_rows = [
        {"record_id": "one", "token_ids": [20, 21, 22]},
        {"record_id": "two", "token_ids": [20, 21, 22]},
    ]
    with pytest.raises(DatasetIntegrityError, match="duplicates"):
        audit_token_overlap(duplicate_rows, benchmark)
def _small_three_way_mix(
    tmp_path: Path, *, split_leak: bool = False, prompt_leak: bool = False
) -> R0DataMixSpec:
    sources = []
    for role in ("fineweb", "lmsys", "contrastive"):
        sources.append(
            R0SourceSpec(
                role=role,
                dataset_id=f"fixture/{role}",
                revision="a" * 40,
                train_tokens=13,
                calibration_tokens=3,
                validation_tokens=4,
                artifact_dir=str(tmp_path / role),
                text_field="text" if role == "fineweb" else None,
                conversation_field="conversation" if role == "lmsys" else None,
                input_manifest="input.json" if role == "contrastive" else None,
            )
        )
    mix = R0DataMixSpec(
        scientific_label="synthetic three-way fixture",
        tokenizer_id="fixture/tokenizer",
        tokenizer_revision="b" * 40,
        batch_size_tokens=4096,
        optimizer_steps=20,
        source_schedule=DEFAULT_SCHEDULE,
        sources=tuple(sources),
    )
    for role_index, source in enumerate(mix.sources):
        writer = TokenArtifactWriter(source, mix, overwrite=False)
        for split, budget in (("train", 13), ("calibration", 3), ("validation", 4)):
            remaining = budget
            record = 0
            while remaining:
                take = min(3 + ((record + role_index) % 4), remaining)
                base = role_index * 10_000 + {"train": 0, "calibration": 100, "validation": 200}[split]
                writer.add(
                    list(range(base + budget - remaining, base + budget - remaining + take)),
                    split=split,
                    provenance={
                        "source_row_id": (
                            f"{role}-leaked-row"
                            if split_leak and source.role == "fineweb" and record == 0
                            else f"{source.role}-{split}-{record}"
                        ),
                        "prompt_id": (
                            "contrastive-leaked-prompt"
                            if prompt_leak and source.role == "contrastive" and record == 0
                            else f"{source.role}-{split}-{record}"
                        ),
                    },
                )
                remaining -= take
                record += 1
        writer.finish({"fixture": True})
    return mix


def test_mix3_exact_source_state_resume_uses_live_mid_record_cursor(tmp_path: Path) -> None:
    mix = _small_three_way_mix(tmp_path)
    uninterrupted = Mix3ASPDSource(mix, "train")
    uninterrupted._advance_without_capture("fineweb", 4096)
    uninterrupted.batch_count = 1
    state = uninterrupted.state_dict()
    assert state["readers"]["fineweb"]["token_offset"] != 0

    def consume(source, role, count):
        reader = source.readers[role]
        observed = []
        remaining = count
        while remaining:
            record = reader.current()
            start = reader.token_offset
            take = min(remaining, len(record["token_ids"]) - start)
            stop = start + take
            observed.append(
                (
                    record["record_id"],
                    start,
                    stop,
                    reader.wrap_count,
                    record["token_ids"][start:stop].tolist(),
                    record["provenance"],
                )
            )
            reader.token_offset = stop
            source.position_counts[role] += take
            remaining -= take
            if stop == len(record["token_ids"]):
                reader.advance_record()
        return observed

    expected_batch = consume(uninterrupted, "lmsys", 4096)
    uninterrupted.batch_count = 2
    expected = uninterrupted.state_dict()

    resumed = Mix3ASPDSource(mix, "train")
    resumed.load_state_dict(state)
    observed_batch = consume(resumed, "lmsys", 4096)
    resumed.batch_count = 2
    assert observed_batch == expected_batch
    assert resumed.state_dict() == expected

    corrupt = deepcopy(state)
    corrupt["source_manifest_sha256"]["fineweb"] = "0" * 64
    with pytest.raises(DatasetIntegrityError, match="different manifests"):
        Mix3ASPDSource(mix, "train").load_state_dict(corrupt)


def test_coordinated_producer_sidecar_checksum_fails_closed(tmp_path: Path) -> None:
    mix = _small_three_way_mix(tmp_path / "data")
    cfg = load_experiment_config(ASPD)
    source = Mix3ASPDSource(mix, "train")
    source._advance_without_capture("fineweb", 4096)
    source.batch_count = 1
    _save_mix3_source_checkpoint(tmp_path, cfg, 1, source)
    sidecar = _mix3_source_checkpoint(tmp_path, 1)
    torch.save(
        {"step": 1, "producer_sha256": _sha256_file(sidecar)},
        tmp_path / "latest_checkpoint.pt",
    )
    restored = Mix3ASPDSource(mix, "train")
    _restore_mix3_source_checkpoint(tmp_path, cfg, 1, restored)
    assert restored.state_dict() == source.state_dict()
    with sidecar.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        _restore_mix3_source_checkpoint(tmp_path, cfg, 1, Mix3ASPDSource(mix, "train"))


def _write_readiness_report(mix, benchmark: Path, path: Path, **updates):
    benchmark_payload = json.loads(benchmark.read_text(encoding="utf-8"))
    prompt_path = Path(benchmark_payload["prompt_file"])
    checks = {
        "exact_duplicate_check": "PASS",
        "near_duplicate_check": "PASS",
        "split_disjointness": "PASS",
        "contrastive_prompt_isolation": "PASS",
        "organism_exclusion": "PASS",
        "shared_aspd_delta_contract": "PASS",
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "source_manifest_sha256": mix3_manifest_fingerprints(mix),
        "benchmark_manifest_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "benchmark_identity": {
            "benchmark_version": benchmark_payload["benchmark_version"],
            "prompt_file": str(prompt_path.resolve()),
            "prompt_sha256": benchmark_payload["prompt_sha256"],
        },
        "mix_contract_sha256": hashlib.sha256(
            json.dumps(mix.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "organism_training_tokens": 0,
        "checks": checks,
        **updates,
    }
    report["audit_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")


def _write_benchmark_fixture(tmp_path: Path) -> Path:
    prompts = tmp_path / "benchmark_prompts.jsonl"
    prompts.write_text('{"prompt_id":"held-out","prompt":"frozen"}\n', encoding="utf-8")
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(
        json.dumps(
            {
                "benchmark_version": "synthetic-frozen-v1",
                "prompt_file": str(prompts),
                "prompt_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_readiness_audit_is_hash_bound_and_status_flag_is_insufficient(tmp_path: Path) -> None:
    mix = _small_three_way_mix(tmp_path / "mix")
    benchmark = _write_benchmark_fixture(tmp_path)
    report = tmp_path / "audit.json"
    with pytest.raises(DatasetIntegrityError, match="missing"):
        validate_mix3_readiness_audit(mix, report_path=report, benchmark_manifest=benchmark)
    _write_readiness_report(mix, benchmark, report)
    assert validate_mix3_readiness_audit(
        mix, report_path=report, benchmark_manifest=benchmark
    )["status"] == "PASS"
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetIntegrityError, match="checksum"):
        validate_mix3_readiness_audit(mix, report_path=report, benchmark_manifest=benchmark)
    _write_readiness_report(mix, benchmark, report, benchmark_manifest_sha256="0" * 64)
    with pytest.raises(DatasetIntegrityError, match="different benchmark"):
        validate_mix3_readiness_audit(mix, report_path=report, benchmark_manifest=benchmark)
    wrong_sources = mix3_manifest_fingerprints(mix)
    wrong_sources["fineweb"] = "0" * 64
    _write_readiness_report(mix, benchmark, report, source_manifest_sha256=wrong_sources)
    with pytest.raises(DatasetIntegrityError, match="different source manifests"):
        validate_mix3_readiness_audit(mix, report_path=report, benchmark_manifest=benchmark)

    cfg = load_experiment_config(ASPD)
    falsely_ready = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(update={"dataset_status": "READY"}),
            "mix3_aspd": cfg.mix3_aspd.model_copy(update={"readiness": "READY"}),
        }
    )
    with pytest.raises(DatasetIntegrityError):
        validate_mix3_recipe_config(falsely_ready, require_ready=True)
    delta = load_delta_crosscoder_config(DELTA)
    falsely_ready_delta = delta.model_copy(
        update={
            "data": delta.data.model_copy(update={"dataset_status": "READY"}),
            "mix3_aspd": delta.mix3_aspd.model_copy(update={"readiness": "READY"}),
            "delta_crosscoder": delta.delta_crosscoder.model_copy(
                update={"readiness": "READY"}
            ),
        }
    )
    with pytest.raises(DatasetIntegrityError):
        train_delta_crosscoder(falsely_ready_delta, device_name="cpu")


def test_streaming_audit_rejects_split_and_overlap_fixtures(tmp_path: Path) -> None:
    mix = _small_three_way_mix(tmp_path / "clean")
    benchmark = _write_benchmark_fixture(tmp_path)
    output = tmp_path / "synthetic-audit.json"
    synthetic = create_mix3_readiness_audit(
        mix,
        benchmark_records=[{"token_ids": [90, 91, 92, 93, 94]}],
        benchmark_manifest=benchmark,
        output_path=output,
        _synthetic_fixture=True,
    )
    assert synthetic["status"] == "SYNTHETIC_TEST_ONLY"
    with pytest.raises(DatasetIntegrityError, match="not a production PASS"):
        validate_mix3_readiness_audit(mix, report_path=output, benchmark_manifest=benchmark)

    with pytest.raises(DatasetIntegrityError, match="exact="):
        create_mix3_readiness_audit(
            mix,
            benchmark_records=[{"token_ids": [0, 1, 2]}],
            benchmark_manifest=benchmark,
            output_path=output,
            near_threshold=0.5,
            _synthetic_fixture=True,
        )
    leaked = _small_three_way_mix(tmp_path / "leaked", split_leak=True)
    with pytest.raises(DatasetIntegrityError, match="split_leaks="):
        create_mix3_readiness_audit(
            leaked,
            benchmark_records=[{"token_ids": [90, 91, 92]}],
            benchmark_manifest=benchmark,
            output_path=output,
            _synthetic_fixture=True,
        )
    prompt_leaked = _small_three_way_mix(tmp_path / "prompt-leaked", prompt_leak=True)
    with pytest.raises(DatasetIntegrityError, match="prompt_leaks="):
        create_mix3_readiness_audit(
            prompt_leaked,
            benchmark_records=[{"token_ids": [90, 91, 92]}],
            benchmark_manifest=benchmark,
            output_path=output,
            _synthetic_fixture=True,
        )
