from __future__ import annotations

import json
from types import MethodType

import torch

from aspd.multimodel.baselines.delta_crosscoder import load_delta_crosscoder_config
from aspd.multimodel.delta_r0_streaming import (
    DEFAULT_SCHEDULE,
    R0DataMixSpec,
    R0SourceSpec,
    R0StreamingDeltaSource,
    TokenArtifactWriter,
    _ArtifactReader,
    _budgeted_chunks,
    load_r0_mix_config,
    validate_text_artifact,
    write_mix_summary,
)


def _source(tmp_path, role: str, train: int = 8192, validation: int = 4096):
    kwargs = {
        "role": role,
        "dataset_id": f"test/{role}",
        "revision": "a" * 40,
        "train_tokens": train,
        "validation_tokens": validation,
        "artifact_dir": str(tmp_path / role),
        "text_field": "text" if role == "fineweb" else None,
        "conversation_field": "conversation" if role == "lmsys" else None,
        "input_manifest": "input.json" if role == "contrastive" else None,
    }
    return R0SourceSpec.model_validate(kwargs)


def _mix(tmp_path) -> R0DataMixSpec:
    return R0DataMixSpec(
        scientific_label="artifact-faithful local test no-organism-training-data variant",
        tokenizer_id="test/tokenizer",
        tokenizer_revision="b" * 40,
        batch_size_tokens=4096,
        optimizer_steps=20,
        source_schedule=DEFAULT_SCHEDULE,
        sources=tuple(_source(tmp_path, role) for role in ("fineweb", "lmsys", "contrastive")),
    )


def _write_artifacts(mix: R0DataMixSpec) -> None:
    for source_index, source in enumerate(mix.sources):
        writer = TokenArtifactWriter(source, mix, overwrite=False)
        for split, budget in (
            ("train", source.train_tokens),
            ("validation", source.validation_tokens),
        ):
            offset = 0
            while offset < budget:
                take = min(701 + source_index, budget - offset)
                writer.add(
                    list(range(source_index * 10000 + offset, source_index * 10000 + offset + take)),
                    split=split,
                    provenance={"offset": offset},
                )
                offset += take
        writer.finish({"test": True})


def _fake_stream(mix: R0DataMixSpec) -> R0StreamingDeltaSource:
    stream = object.__new__(R0StreamingDeltaSource)
    stream.mix = mix
    stream._activation_dim = 2
    stream.readers = {
        source.role: _ArtifactReader(source, "train") for source in mix.sources
    }
    stream.schedule_index = 0
    stream.tokens_consumed_total = 0
    stream.tokens_consumed_by_source = {role: 0 for role in ("fineweb", "lmsys", "contrastive")}
    stream.paired_forward_seconds = 0.0
    stream.paired_forward_tokens = 0
    stream.same_token_checks = 0

    def paired(self, token_ids):
        values = token_ids.float()
        return torch.stack(
            (torch.stack((values, values + 1), dim=-1),
             torch.stack((values + 2, values + 3), dim=-1)),
            dim=1,
        )

    stream._paired_forward = MethodType(paired, stream)
    return stream


def test_production_mix_and_local_config_freeze_scientific_contract(tmp_path) -> None:
    mix = load_r0_mix_config("configs/multimodel/delta_crosscoder/r0_data_mix.yaml")
    cfg = load_delta_crosscoder_config(
        "configs/multimodel/delta_crosscoder/r0_extreme_sports_local.yaml"
    )
    pilot = load_delta_crosscoder_config(
        "configs/multimodel/delta_crosscoder/r0_extreme_sports_streaming_pilot.yaml"
    )
    assert mix.organism_training_tokens == 0
    assert [source.train_tokens for source in mix.sources] == [90_000_000, 90_000_000, 20_000_000]
    assert mix.unique_tokens == 200_000_000
    assert mix.optimizer_token_updates == 204_800_000
    assert cfg.delta_crosscoder.data.mode == "r0_stream"
    assert cfg.delta_crosscoder.total_steps == 50_000
    assert cfg.delta_contrastive is not None
    assert cfg.delta_contrastive.generation.max_new_tokens == 1024
    assert cfg.delta_contrastive.generation.materialization_batch_size == 4
    assert pilot.delta_crosscoder.total_steps == 200
    assert pilot.delta_crosscoder.data.r0_mix_config.endswith("r0_data_mix_pilot.yaml")
    output = tmp_path / "summary.json"
    write_mix_summary(mix, output)
    summary = json.loads(output.read_text())
    assert summary["repeat_tokens"] == 4_800_000
    assert summary["source_repeat_tokens"] == {
        "fineweb": 2_160_000,
        "lmsys": 2_160_000,
        "contrastive": 480_000,
    }


def test_budgeted_chunks_stop_at_exact_split_budgets(tmp_path) -> None:
    source = _source(tmp_path, "fineweb", train=7, validation=5)
    rows = [
        ([1, 2, 3, 4, 5, 6], {"row": 0}),
        ([7, 8, 9, 10, 11, 12], {"row": 1}),
        ([13, 14, 15, 16, 17], {"row": 2}),
    ]
    chunks = list(_budgeted_chunks(rows, source, sequence_length=4))
    assert sum(len(ids) for ids, split, _ in chunks if split == "train") == 7
    assert sum(len(ids) for ids, split, _ in chunks if split == "validation") == 5
    # Row 1 fills the final train token with 7. Its suffix is discarded, so the
    # held-out split starts at row 2 and cannot leak the same document.
    assert [value for ids, split, _ in chunks for value in ids if split == "train"] == list(
        range(1, 8)
    )
    assert [value for ids, split, _ in chunks for value in ids if split == "validation"] == list(
        range(13, 18)
    )


def test_token_artifact_validation_and_resume_cursor_are_exact(tmp_path) -> None:
    mix = _mix(tmp_path)
    _write_artifacts(mix)
    for source in mix.sources:
        report = validate_text_artifact(source.artifact_dir, source)
        assert report["tokens"] == {"train": 8192, "validation": 4096}

    uninterrupted = _fake_stream(mix)
    for _ in range(7):
        role = mix.source_schedule[uninterrupted.schedule_index % len(DEFAULT_SCHEDULE)]
        uninterrupted.schedule_index += 1
        uninterrupted._next_batch(role, uninterrupted.readers, count=True)
    state = uninterrupted.state_dict()

    expected = []
    for _ in range(5):
        role = mix.source_schedule[uninterrupted.schedule_index % len(DEFAULT_SCHEDULE)]
        uninterrupted.schedule_index += 1
        batch = uninterrupted._next_batch(role, uninterrupted.readers, count=True)
        expected.append((role, batch["activations"].clone()))

    resumed = _fake_stream(mix)
    resumed.load_state_dict(state)
    observed = []
    for _ in range(5):
        role = mix.source_schedule[resumed.schedule_index % len(DEFAULT_SCHEDULE)]
        resumed.schedule_index += 1
        batch = resumed._next_batch(role, resumed.readers, count=True)
        observed.append((role, batch["activations"]))

    assert [role for role, _ in observed] == [role for role, _ in expected]
    for (_, actual), (_, reference) in zip(observed, expected, strict=True):
        torch.testing.assert_close(actual, reference)
    assert resumed.state_dict()["readers"] == uninterrupted.state_dict()["readers"]
    assert resumed.tokens_consumed_by_source == uninterrupted.tokens_consumed_by_source
