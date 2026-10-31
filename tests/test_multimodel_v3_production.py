from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from aspd.multimodel.v3.data import HashSplitPolicy, SourceDocument, SQLiteExposureLedger
from aspd.multimodel.v3.distributed_checkpoint import (
    MainV1DistributedCheckpoint,
    _sharded_state_dict_options,
)
from aspd.multimodel.v3.production_stream import PileAlignedAnchorStream


class _CharacterTokenizer:
    is_fast = True

    def __call__(self, text: str, **_kwargs):
        return {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(value) for value in token_ids)


class _StatefulRows:
    def __init__(self, rows):
        self.rows = list(rows)
        self.index = 0

    def __iter__(self):
        while self.index < len(self.rows):
            row = self.rows[self.index]
            self.index += 1
            yield row

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state):
        self.index = int(state["index"])


def _train_texts(policy: HashSplitPolicy, count: int) -> list[str]:
    values = []
    candidate = 0
    while len(values) < count:
        text = f"document-{candidate}-abcdefghijk"
        document = SourceDocument(
            hashlib.sha256(text.encode()).hexdigest(), "unit", text, str(candidate)
        )
        if policy.split(document) == "train":
            values.append(text)
        candidate += 1
    return values


def _stream(tmp_path: Path, rows: _StatefulRows, policy: HashSplitPolicy):
    tokenizers = {key: _CharacterTokenizer() for key in ("qwen", "gemma", "phi")}
    revisions = {key: f"{key}-revision" for key in tokenizers}
    return PileAlignedAnchorStream(
        dataset=rows,
        dataset_id="unit/pile",
        dataset_revision="frozen-revision",
        tokenizers=tokenizers,
        tokenizer_revisions=revisions,
        split_policy=policy,
        dedup_path=tmp_path / "dedup.sqlite",
        sequence_length=64,
    )


def test_dcp_keeps_sharded_state_on_device_for_bounded_sync_writes() -> None:
    options = _sharded_state_dict_options()
    assert options.full_state_dict is False
    assert options.cpu_offload is False


def test_production_sqlite_uses_parallel_filesystem_safe_journaling(
    tmp_path: Path,
) -> None:
    policy = HashSplitPolicy(
        modulus=5,
        validation_buckets=frozenset({0}),
        test_buckets=frozenset({1}),
        seed=0,
    )
    text = _train_texts(policy, 1)[0]
    stream = _stream(
        tmp_path,
        _StatefulRows([{"text": text, "meta": {"pile_set_name": "unit"}}]),
        policy,
    )
    ledger = SQLiteExposureLedger(
        tmp_path / "ledger.sqlite",
        dataset_manifest_sha256="e" * 64,
        world_size=3,
    )
    try:
        stream_mode = stream.dedup.connection.execute("PRAGMA journal_mode").fetchone()[0]
        ledger_mode = ledger.connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(stream_mode).lower() == "delete"
        assert str(ledger_mode).lower() == "delete"
    finally:
        stream.close()
        ledger.close()


def test_production_stream_exact_pool_and_resume(tmp_path: Path) -> None:
    policy = HashSplitPolicy(
        modulus=5,
        validation_buckets=frozenset({0}),
        test_buckets=frozenset({1}),
        seed=0,
    )
    texts = _train_texts(policy, 5)
    source = _StatefulRows(
        [
            {"text": texts[0], "meta": {"pile_set_name": "unit"}},
            {"text": texts[0], "meta": {"pile_set_name": "unit"}},  # exact duplicate
            *(
                {"text": text, "meta": {"pile_set_name": "unit"}}
                for text in texts[1:]
            ),
        ]
    )
    stream = _stream(tmp_path, source, policy)
    first = stream.next_pool(pool_anchors=17)
    first.validate(expected_anchors=17, sequence_length=64)
    checkpoint = stream.state_dict()
    second = stream.next_pool(pool_anchors=19)
    second_duplicates = stream.exact_duplicates_rejected
    stream.load_state_dict(checkpoint)
    replay = stream.next_pool(pool_anchors=19)
    assert replay.to_payload() == second.to_payload()
    assert stream.exact_duplicates_rejected == second_duplicates
    stream.close()


def test_production_stream_rejects_provenance_mismatch(tmp_path: Path) -> None:
    policy = HashSplitPolicy(
        modulus=5,
        validation_buckets=frozenset({0}),
        test_buckets=frozenset({1}),
        seed=0,
    )
    text = _train_texts(policy, 1)[0]
    stream = _stream(
        tmp_path,
        _StatefulRows([{"text": text, "meta": {"pile_set_name": "unit"}}]),
        policy,
    )
    state = stream.state_dict()
    state["provenance_sha256"] = "0" * 64
    try:
        stream.load_state_dict(state)
    except ValueError as error:
        assert "provenance mismatch" in str(error)
    else:
        raise AssertionError("mismatched stream provenance was accepted")
    stream.close()


def test_dcp_atomic_save_restore_optimizer_and_runtime_state(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.GELU(), torch.nn.Linear(5, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    manager = MainV1DistributedCheckpoint(tmp_path / "checkpoints", keep_complete=2)
    manifest = manager.save(
        step=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        runtime_state={"accepted_exposures": 8192, "stream": {"cursor": 7}},
        config_fingerprint="a" * 64,
        config_file_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
    )
    assert manifest["status"] == "COMPLETE"
    checkpoint = tmp_path / "checkpoints" / "step-000000001"
    assert (checkpoint / "COMPLETE").is_file()
    assert not (tmp_path / "checkpoints" / ".step-000000001.incomplete").exists()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    runtime, loaded_manifest = manager.load(
        path=checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config_fingerprint="a" * 64,
        config_file_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
    )
    assert runtime == {"accepted_exposures": 8192, "stream": {"cursor": 7}}
    assert loaded_manifest == manifest
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])

    incomplete = tmp_path / "checkpoints" / ".step-000000002.incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="not atomically complete"):
        manager.load(
            path=incomplete,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config_fingerprint="a" * 64,
            config_file_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
        )


def test_streaming_raw_token_ledger_rolls_back_with_checkpoint_receipt(
    tmp_path: Path,
) -> None:
    ledger = SQLiteExposureLedger(
        tmp_path / "ledger.sqlite", dataset_manifest_sha256="d" * 64, world_size=4
    )
    ledger.observe_raw_tokens("doc-0", {"qwen": 5, "gemma": 6, "phi": 7})
    ledger.observe_global_optimizer_step(
        0,
        {
            rank: [f"step0-r{rank}-a{index}" for index in range(2048)]
            for rank in range(4)
        },
    )
    receipt = ledger.checkpoint_receipt(expected_optimizer_steps=1)
    ledger.observe_raw_tokens("doc-1", {"qwen": 11, "gemma": 12, "phi": 13})
    ledger.observe_global_optimizer_step(
        1,
        {
            rank: [f"step1-r{rank}-a{index}" for index in range(2048)]
            for rank in range(4)
        },
    )
    recovered = ledger.recover_to_receipt(receipt)
    assert recovered.optimization_steps == 1
    assert recovered.accepted_aligned_anchor_exposures == 8192
    assert (recovered.raw_tokens_qwen, recovered.raw_tokens_gemma, recovered.raw_tokens_phi) == (
        5,
        6,
        7,
    )
    ledger.close()


def test_replicated_model_parallel_step_counts_global_anchors_once(tmp_path: Path) -> None:
    ledger = SQLiteExposureLedger(
        tmp_path / "replicated.sqlite",
        dataset_manifest_sha256="e" * 64,
        world_size=3,
    )
    anchors = [f"anchor-{index}" for index in range(8192)]
    counters = ledger.observe_replicated_optimizer_step(0, anchors)

    assert counters.accepted_aligned_anchor_exposures == 8192
    assert counters.unique_aligned_anchors == 8192
    assert ledger.rank_exposure_counts() == {0: 8192, 1: 0, 2: 0}
    assert ledger.audit_full_consistency()["status"] == "PASS"
    ledger.close()
