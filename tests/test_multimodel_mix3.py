from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from aspd.multimodel.baselines.delta_crosscoder import (
    load_delta_crosscoder_config,
    train_delta_crosscoder,
)
from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.delta_contrastive import plan_section_4_2_single_generator
from aspd.multimodel.mix3 import (
    ExactPositionCursor,
    Mix3ASPDSource,
    audit_token_overlap,
    collate_variable_records,
    pack_valid_activation_positions,
    validate_delta_mix3_config,
    validate_mix3_recipe_config,
)
from aspd.multimodel.original_recipe import DatasetIntegrityError, validate_original_recipe_config
from aspd.multimodel.original_streaming import checkpoint_payload, restore_checkpoint_payload

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


def test_overlap_audit_fails_on_duplicates_exact_and_near_overlap() -> None:
    benchmark = [{"record_id": "held-out", "token_ids": list(range(10))}]
    assert audit_token_overlap(
        [{"record_id": "clean", "token_ids": list(range(20, 30))}], benchmark
    )["status"] == "PASS"
    with pytest.raises(DatasetIntegrityError, match="exact"):
        audit_token_overlap(
            [{"record_id": "leak", "token_ids": list(range(10))}], benchmark
        )
    duplicate_rows = [
        {"record_id": "one", "token_ids": [20, 21, 22]},
        {"record_id": "two", "token_ids": [20, 21, 22]},
    ]
    with pytest.raises(DatasetIntegrityError, match="duplicates"):
        audit_token_overlap(duplicate_rows, benchmark)
