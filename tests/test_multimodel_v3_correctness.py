from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aspd.multimodel.v3.alignment import (
    align_canonical_text,
    exact_prefix_decode_work_upper_bound,
)
from aspd.multimodel.v3.config import load_v3_config_bundle
from aspd.multimodel.v3.data import (
    HashSplitPolicy,
    ManifestAccumulator,
    SourceDocument,
    SQLiteExposureLedger,
)
from aspd.multimodel.v3.encoder import ChannelSlicePacker, HierarchicalV3Encoder
from aspd.multimodel.v3.hooks import (
    LogicalMatrixCapture,
    ResidualWindowCapture,
    freeze_backbone,
    validate_observation,
)
from aspd.multimodel.v3.models import validate_loaded_model_modules
from aspd.multimodel.v3.registry import build_model_registry, validate_runtime_config
from aspd.multimodel.v3.streaming import AlignedAnchorStream

ROOT = Path(__file__).parents[1]
CONFIG_ROOT = ROOT / "configs/multimodel/aspd_v3"


def _bundle():
    return load_v3_config_bundle(
        CONFIG_ROOT / "approved.yaml",
        CONFIG_ROOT / "candidates.yaml",
        CONFIG_ROOT / "runtime_template.yaml",
    )


def test_gemma_4b_registry_matches_independent_reference_metadata() -> None:
    reference = json.loads(
        (CONFIG_ROOT / "reference_metadata/gemma3_4b_text_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert reference["checkpoint_config_access"].startswith("NOT_VERIFIED_GATED_401")
    assert all(source.startswith("https://") for source in reference["independent_primary_sources"])
    expected = reference["text_config"]
    gemma = build_model_registry(_bundle().approved)["gemma"]
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "q_out",
        "kv_out",
    ):
        assert getattr(gemma, name) == expected[name]
    assert gemma.layers == tuple(range(13, 21))
    assert len(gemma.targets()) == 56
    first = {target.logical_name: target for target in gemma.targets()[:7]}
    assert (first["q"].d_in, first["q"].d_out) == (2560, 2048)
    assert (first["k"].d_in, first["k"].d_out) == (2560, 1024)
    assert (first["v"].d_in, first["v"].d_out) == (2560, 1024)
    assert (first["o"].d_in, first["o"].d_out) == (2048, 2560)
    assert (first["mlp_gate"].d_in, first["mlp_gate"].d_out) == (2560, 10240)
    assert (first["mlp_up"].d_in, first["mlp_up"].d_out) == (2560, 10240)
    assert (first["mlp_down"].d_in, first["mlp_down"].d_out) == (10240, 2560)
    assert first["o"].grounding_module_candidates[0].endswith(
        "pre_feedforward_layernorm"
    )

    config = SimpleNamespace(**{key: expected[key] for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )})
    assert validate_runtime_config(gemma, SimpleNamespace(text_config=config))["status"] == "PASS"
    config.head_dim = 128
    with pytest.raises(ValueError, match="q_out"):
        validate_runtime_config(gemma, SimpleNamespace(text_config=config))


class _Scale(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.value


class _FamilyLayer(nn.Module):
    def __init__(self, family: str):
        super().__init__()
        self.family = family
        self.input_layernorm = _Scale(0.5)
        self.post_attention_layernorm = _Scale(2.0)
        if family == "gemma":
            self.pre_feedforward_layernorm = _Scale(0.25)
        self.attention = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Linear(4, 4, bias=False)
        nn.init.eye_(self.attention.weight)
        nn.init.eye_(self.mlp.weight)
        self.last_pre = torch.empty(0)
        self.last_mid = torch.empty(0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.last_pre = hidden.detach().clone()
        residual = hidden
        attention_output = self.attention(self.input_layernorm(hidden))
        if self.family == "gemma":
            attention_output = self.post_attention_layernorm(attention_output)
            hidden = residual + attention_output
            self.last_mid = hidden.detach().clone()
            mlp_input = self.pre_feedforward_layernorm(hidden)
        else:
            hidden = residual + attention_output
            self.last_mid = hidden.detach().clone()
            mlp_input = self.post_attention_layernorm(hidden)
        return hidden + self.mlp(mlp_input)


class _FamilyModel(nn.Module):
    def __init__(self, family: str):
        super().__init__()
        layers = nn.ModuleList([_FamilyLayer(family) for _ in range(8)])
        self.model = nn.Module()
        if family == "gemma":
            self.model.language_model = nn.Module()
            self.model.language_model.layers = layers
        else:
            self.model.layers = layers
        self.layers = layers

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


@pytest.mark.parametrize("key", ["qwen", "gemma", "phi"])
def test_family_specific_residual_hooks_capture_true_streams(key: str) -> None:
    registry = build_model_registry(_bundle().approved)
    base = registry[key]
    prefix = "model.language_model.layers" if key == "gemma" else "model.layers"
    layout = replace(base, layers=tuple(range(8)), layer_prefixes=(prefix,), hidden_size=4)
    model = _FamilyModel(key)
    freeze_backbone(model)
    x = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    clean = model(x)
    expected_pre = [layer.last_pre for layer in model.layers]
    expected_mid = [layer.last_mid for layer in model.layers]
    with ResidualWindowCapture(model, layout, "residual_pre") as pre_capture:
        with ResidualWindowCapture(model, layout, "residual_mid") as mid_capture:
            hooked = model(x)
            captured_pre = pre_capture.ordered()
            captured_mid = mid_capture.ordered()
    assert torch.equal(clean, hooked)
    assert all(
        torch.equal(value, expected)
        for value, expected in zip(captured_pre, expected_pre, strict=True)
    )
    assert all(
        torch.equal(value, expected)
        for value, expected in zip(captured_mid, expected_mid, strict=True)
    )
    if key == "gemma":
        assert all("pre_feedforward_layernorm" in path for path in mid_capture.paths.values())
        assert not torch.equal(captured_mid[0], model.layers[0].attention(model.layers[0].input_layernorm(x)))
    else:
        assert all("post_attention_layernorm" in path for path in mid_capture.paths.values())
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())


class _SeparateAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(value) + self.k_proj(value) + self.v_proj(value))


class _SeparateMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4)
        self.up_proj = nn.Linear(4, 4)
        self.down_proj = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_proj(value) * self.up_proj(value))


class _SeparateLayer(nn.Module):
    def __init__(self, family: str):
        super().__init__()
        self.family = family
        self.input_layernorm = _Scale(0.5)
        self.post_attention_layernorm = _Scale(2.0)
        if family == "gemma":
            self.pre_feedforward_layernorm = _Scale(0.25)
            self.post_feedforward_layernorm = nn.Identity()
        self.self_attn = _SeparateAttention()
        self.mlp = _SeparateMLP()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        attention_output = self.self_attn(self.input_layernorm(hidden))
        if self.family == "gemma":
            hidden = residual + self.post_attention_layernorm(attention_output)
            mlp_output = self.mlp(self.pre_feedforward_layernorm(hidden))
            return hidden + self.post_feedforward_layernorm(mlp_output)
        hidden = residual + attention_output
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _SeparateModel(nn.Module):
    def __init__(self, family: str):
        super().__init__()
        layer = _SeparateLayer(family)
        self.model = nn.Module()
        if family == "gemma":
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleList([layer])
        else:
            self.model.layers = nn.ModuleList([layer])
        self.layer = layer

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layer(hidden)


@pytest.mark.parametrize("key", ["qwen", "gemma"])
def test_separate_projection_family_hooks_capture_exact_native_inputs_and_outputs(key: str) -> None:
    base = build_model_registry(_bundle().approved)[key]
    prefix = "model.language_model.layers" if key == "gemma" else "model.layers"
    layout = replace(
        base,
        layers=(0,),
        layer_prefixes=(prefix,),
        hidden_size=4,
        intermediate_size=4,
        q_out=4,
        kv_out=4,
    )
    model = _SeparateModel(key)
    freeze_backbone(model)
    module_report = validate_loaded_model_modules(model, layout)
    assert module_report["status"] == "PASS" and len(module_report["targets"]) == 7
    x = torch.randn(2, 3, 4)
    clean = model(x)
    for target in layout.targets():
        with LogicalMatrixCapture(model, target) as capture:
            hooked = model(x)
            observation = capture.observation()
        assert torch.equal(clean, hooked)
        assert validate_observation(model, target, observation)["status"] == "PASS"
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    model.layer.self_attn.q_proj = nn.Linear(3, 4)
    freeze_backbone(model)
    with pytest.raises(ValueError, match="sliced weight shape"):
        validate_loaded_model_modules(model, layout)


def _write_stream_fixture(root: Path, counts: tuple[int, ...]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shards = []
    next_value = 0
    for shard_index, count in enumerate(counts):
        path = root / f"stream_{shard_index}.jsonl"
        rows = []
        for _ in range(count):
            rows.append(
                {
                    "anchor_id": f"anchor-{next_value}",
                    "model_order": ["qwen", "gemma", "phi"],
                    "value": "\N{LATIN SMALL LETTER E WITH ACUTE}" * (1 + next_value % 3),
                }
            )
            next_value += 1
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        shards.append(
            {
                "split": "train",
                "shard_index": shard_index,
                "path": path.name,
                "records": count,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = root / "stream_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 3, "shards": shards}), encoding="utf-8")
    return manifest


def test_stream_is_linear_and_exact_across_mid_shard_resume(tmp_path: Path) -> None:
    work = []
    for count in (32, 128, 512):
        root = tmp_path / str(count)
        root.mkdir()
        manifest = _write_stream_fixture(root, (count,))
        with AlignedAnchorStream(manifest, "train", seed=7, shuffle_buffer_size=5) as stream:
            observed = [row["anchor_id"] for row in stream]
            work.append(stream.raw_lines_read)
        assert len(observed) == count
    assert work == [32, 128, 512]

    manifest = _write_stream_fixture(tmp_path / "resume", (7, 9))
    uninterrupted_stream = AlignedAnchorStream(manifest, "train", seed=42, shuffle_buffer_size=4)
    uninterrupted = [row["anchor_id"] for row in uninterrupted_stream]
    uninterrupted_stream.close()
    first = AlignedAnchorStream(manifest, "train", seed=42, shuffle_buffer_size=4)
    iterator = iter(first)
    prefix = [next(iterator)["anchor_id"] for _ in range(6)]
    state = first.state_dict()
    assert state["buffer"] and state["byte_offset"] > 0
    first.close()
    resumed = AlignedAnchorStream(manifest, "train", seed=42, shuffle_buffer_size=4)
    resumed.load_state_dict(state)
    suffix = [row["anchor_id"] for row in resumed]
    resumed.close()
    assert prefix + suffix == uninterrupted

    completed = AlignedAnchorStream(manifest, "train", seed=42, shuffle_buffer_size=4)
    list(completed)
    final_state = completed.state_dict()
    completed.close()
    final_resume = AlignedAnchorStream(manifest, "train", seed=42, shuffle_buffer_size=4)
    final_resume.load_state_dict(final_state)
    assert list(final_resume) == []
    final_resume.close()


def test_stream_resume_rejects_corrupt_offset_manifest_and_shard(tmp_path: Path) -> None:
    manifest = _write_stream_fixture(tmp_path, (5, 3))
    stream = AlignedAnchorStream(manifest, "train", seed=1, shuffle_buffer_size=2)
    iterator = iter(stream)
    next(iterator)
    state = stream.state_dict()
    stream.close()

    corrupt = dict(state)
    corrupt["byte_offset"] = state["byte_offset"] - 1
    resumed = AlignedAnchorStream(manifest, "train", seed=1, shuffle_buffer_size=2)
    with pytest.raises(ValueError, match="JSONL boundary|cursor proof"):
        resumed.load_state_dict(corrupt)
    resumed.close()

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({**manifest_payload, "extra": True}), encoding="utf-8")
    changed_manifest = AlignedAnchorStream(manifest, "train", seed=1, shuffle_buffer_size=2)
    with pytest.raises(ValueError, match="provenance mismatch"):
        changed_manifest.load_state_dict(state)
    changed_manifest.close()

    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    shard = tmp_path / manifest_payload["shards"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="missing or corrupt"):
        AlignedAnchorStream(manifest, "train", seed=1, shuffle_buffer_size=2)


def test_exact_exposure_accounting_survives_restart_and_rank_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "exposure.sqlite"
    manifest_hash = "a" * 64
    with SQLiteExposureLedger(path, dataset_manifest_sha256=manifest_hash) as ledger:
        counters = ledger.observe_optimizer_step(0, ["a", "a", "b"])
        assert (counters.unique_aligned_anchors, counters.repeated_anchor_exposures) == (2, 1)
        assert counters.accepted_aligned_anchor_exposures == 3
    with SQLiteExposureLedger(path, dataset_manifest_sha256=manifest_hash) as ledger:
        counters = ledger.observe_optimizer_step(1, ["b", "c"])
        assert counters.accepted_aligned_anchor_exposures == 5
        assert counters.newly_observed_anchor_exposures == 3
        assert counters.repeated_anchor_exposures == 2
        before = ledger.snapshot()
        assert ledger.observe_optimizer_step(1, ["b", "c"]) == before
        with pytest.raises(ValueError, match="different batch"):
            ledger.observe_optimizer_step(1, ["other"])
    with pytest.raises(ValueError, match="provenance mismatch"):
        SQLiteExposureLedger(path, dataset_manifest_sha256="b" * 64)

    distributed_path = tmp_path / "distributed.sqlite"
    with SQLiteExposureLedger(
        distributed_path, dataset_manifest_sha256=manifest_hash, world_size=2
    ) as ledger:
        counters = ledger.observe_global_optimizer_step(0, {0: ["x", "y"], 1: ["y", "z"]})
        assert counters.accepted_aligned_anchor_exposures == 4
        assert counters.unique_aligned_anchors == 3
        assert counters.repeated_anchor_exposures == 1
        assert ledger.rank_exposure_counts() == {0: 2, 1: 2}
        with pytest.raises(ValueError, match="every rank"):
            ledger.observe_global_optimizer_step(1, {0: ["x"]})
    with pytest.raises(ValueError, match="only rank 0"):
        SQLiteExposureLedger(
            tmp_path / "rank1.sqlite",
            dataset_manifest_sha256=manifest_hash,
            writer_rank=1,
        )


def test_exposure_raw_tokens_are_idempotent_and_interrupted_transaction_rolls_back(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exposure-recovery.sqlite"
    manifest_hash = "f" * 64
    alignment = _alignment("raw-doc", "abc")
    ledger = SQLiteExposureLedger(path, dataset_manifest_sha256=manifest_hash)
    ledger.observe_alignment("raw-doc", alignment)
    first = ledger.snapshot()
    assert (first.raw_tokens_qwen, first.raw_tokens_gemma, first.raw_tokens_phi) == (5, 5, 5)
    ledger.observe_alignment("raw-doc", alignment)
    assert ledger.snapshot() == first
    ledger.connection.execute("BEGIN IMMEDIATE")
    ledger.connection.execute(
        "INSERT INTO seen_anchors(anchor_id,first_step) VALUES ('uncommitted',99)"
    )
    ledger.close()
    with SQLiteExposureLedger(path, dataset_manifest_sha256=manifest_hash) as recovered:
        assert recovered.snapshot() == first
        assert recovered.connection.execute(
            "SELECT COUNT(*) FROM seen_anchors WHERE anchor_id='uncommitted'"
        ).fetchone()[0] == 0


class _SimpleTokenizer:
    is_fast = True

    def __init__(self, spans: list[tuple[int, int]]):
        self.spans = spans
        self.pieces: dict[int, str] = {}

    def __call__(self, text: str, **kwargs):
        self.pieces = {
            index + 1: text[start:stop] for index, (start, stop) in enumerate(self.spans)
        }
        return {
            "input_ids": [99, *range(1, len(self.spans) + 1), 100],
            "attention_mask": [1] * (len(self.spans) + 2),
            "offset_mapping": [(0, 0), *self.spans, (0, 0)],
        }

    def decode(self, token_ids, **kwargs):
        return "".join(self.pieces.get(int(token_id), "") for token_id in token_ids)


def _alignment(document_id: str, text: str = "ab"):
    spans = [(index, index + 1) for index in range(len(text))]
    return align_canonical_text(
        text,
        document_id=document_id,
        source_group="pile_uncopyrighted",
        tokenizers={key: _SimpleTokenizer(spans) for key in ("qwen", "gemma", "phi")},
    )


def test_manifest_resume_reconstructs_durable_aggregates_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "manifest.sqlite"
    policy = HashSplitPolicy(10, frozenset({0}), frozenset({1}), seed=42)
    kwargs = {
        "dataset_id": "monology/pile-uncopyrighted",
        "dataset_revision": "d" * 40,
        "split_policy": policy,
        "tokenizer_revisions": {"qwen": "q", "gemma": "g", "phi": "p"},
        "ledger_path": path,
    }
    first_doc = SourceDocument("doc-1", "pile_uncopyrighted", "ab", "row-1")
    with ManifestAccumulator(**kwargs) as accumulator:
        accumulator.add(first_doc, "train", _alignment("doc-1"))
        partial = accumulator.manifest()
    with ManifestAccumulator(**kwargs) as resumed:
        assert resumed.manifest() == partial
        second_doc = SourceDocument("doc-2", "pile_uncopyrighted", "abc", "row-2")
        resumed.add(second_doc, "validation", _alignment("doc-2", "abc"))
        complete = resumed.manifest()
    assert complete["documents_by_split"] == {"train": 1, "validation": 1, "test": 0}
    assert complete["anchors_by_split"] == {"train": 2, "validation": 3, "test": 0}
    wrong = dict(kwargs)
    wrong["dataset_revision"] = "e" * 40
    with pytest.raises(ValueError, match="provenance"):
        ManifestAccumulator(**wrong)

    corrupt_path = tmp_path / "manifest-corrupt.sqlite"
    shutil.copyfile(path, corrupt_path)
    connection = sqlite3.connect(corrupt_path)
    with connection:
        connection.execute(
            "UPDATE manifest_metadata SET provenance_sha256=? WHERE singleton=1",
            ("0" * 64,),
        )
    connection.close()
    corrupt = dict(kwargs)
    corrupt["ledger_path"] = corrupt_path
    with pytest.raises(ValueError, match="provenance"):
        ManifestAccumulator(**corrupt)


def test_encoder_rejects_cross_width_before_module_allocation() -> None:
    packers = {
        key: ChannelSlicePacker(768, 1) for key in ("qwen", "gemma", "phi")
    }
    with pytest.raises(ValueError, match=r"cross_width=773.*heads=12"):
        HierarchicalV3Encoder(
            packers,
            intra_layers=1,
            intra_heads=12,
            cross_layers=1,
            cross_heads=12,
            n_features=8,
            readout="mean",
        )


def test_alignment_performance_audit_reports_quadratic_exact_prefix_work() -> None:
    work = []
    for length in (8, 16, 32):
        text = "x" * length
        result = _alignment(f"doc-{length}", text)
        qwen = result.tokenized["qwen"]
        assert qwen.prefix_decode_calls == length
        assert qwen.prefix_decoded_token_ids == length * (length + 3) // 2
        assert qwen.prefix_decoded_token_ids <= exact_prefix_decode_work_upper_bound(length + 2)
        work.append(qwen.prefix_decoded_token_ids)
        assert result.report()["performance_status"].startswith("EXACT_BUT_QUADRATIC")
    assert work == [44, 152, 560]
