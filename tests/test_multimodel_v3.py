from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.v3.alignment import AlignmentError, align_canonical_text
from aspd.multimodel.v3.config import (
    V3ApprovedConfig,
    V3CandidateConfig,
    V3RuntimeConfig,
    load_v3_config_bundle,
)
from aspd.multimodel.v3.data import (
    HashSplitPolicy,
    ManifestAccumulator,
    SourceDocument,
    SQLiteDuplicateIndex,
    SQLiteExposureLedger,
    write_json_immutable,
)
from aspd.multimodel.v3.encoder import (
    ChannelSlicePacker,
    HierarchicalV3Encoder,
    pack_cross_model_tokens,
)
from aspd.multimodel.v3.feasibility import estimate_v3_feasibility
from aspd.multimodel.v3.gating import V3BatchTopK
from aspd.multimodel.v3.hooks import (
    LogicalMatrixCapture,
    ResidualWindowCapture,
    freeze_backbone,
    validate_observation,
)
from aspd.multimodel.v3.reconstruction import (
    RankOneMatrixBank,
    require_resolved_objective,
)
from aspd.multimodel.v3.registry import (
    LogicalMatrixTarget,
    OutputSlice,
    build_model_registry,
    validate_runtime_config,
)
from aspd.multimodel.v3.streaming import AlignedAnchorStream
from aspd.multimodel.v3.training import (
    AlignedStreamCursor,
    DistributedPlan,
    V3TrainingState,
    commit_coordinated_checkpoint,
    deterministic_rank_shard,
    encode_rng_state,
    minimum_optimizer_steps,
    restore_coordinated_checkpoint,
    validate_finite_tensors,
)

ROOT = Path(__file__).parents[1]
APPROVED = ROOT / "configs/multimodel/aspd_v3/approved.yaml"
CANDIDATES = ROOT / "configs/multimodel/aspd_v3/candidates.yaml"
RUNTIME = ROOT / "configs/multimodel/aspd_v3/runtime_template.yaml"


class ToyTokenizer:
    is_fast = True

    def __init__(
        self,
        spans: list[tuple[int, int]],
        *,
        mismatch: bool = False,
        padding: int = 0,
    ):
        self.spans = spans
        self.mismatch = mismatch
        self.padding = padding
        self._pieces: dict[int, str] = {}

    def __call__(self, text: str, **kwargs):
        max_length = kwargs.get("max_length")
        spans = self.spans
        ids = [99] + list(range(1, len(spans) + 1)) + [100]
        offsets = [(0, 0)] + spans + [(0, 0)]
        if max_length is not None:
            ids = ids[:max_length]
            offsets = offsets[:max_length]
        mask = [1] * len(ids)
        ids.extend([0] * self.padding)
        offsets.extend([(0, 0)] * self.padding)
        mask.extend([0] * self.padding)
        self._pieces = {index + 1: text[start:stop] for index, (start, stop) in enumerate(spans)}
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "offset_mapping": offsets,
        }

    def decode(self, token_ids, **kwargs):
        result = "".join(self._pieces.get(int(token_id), "") for token_id in token_ids)
        return result.replace("é", "e") if self.mismatch else result


def _bundle():
    return load_v3_config_bundle(APPROVED, CANDIDATES, RUNTIME)


def test_v3_config_is_separate_blocked_and_does_not_inherit_mix3() -> None:
    bundle = _bundle()
    report = bundle.report()
    assert report["status"] == "BLOCKED"
    assert report["derived"]["planned_logical_matrix_targets"] == 168
    assert report["derived"]["minimum_optimizer_steps_at_candidate_batch"] == 250_000
    assert any("dictionary_size" in blocker for blocker in report["blockers"])
    assert any("main_sequence_length" in blocker for blocker in report["blockers"])
    assert any("split_modulus" in blocker for blocker in report["blockers"])
    assert any("gemma" in blocker for blocker in report["blockers"])
    payload = bundle.approved.model_dump(mode="json")
    payload["mix3_aspd"] = {"optimizer_token_exposures": 204_800_000}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        V3ApprovedConfig.model_validate(payload)
    runtime = bundle.runtime.model_dump(mode="json")
    runtime["output_dir"] = "out/multimodel/runs/qwen25_aspd_mix3_200m"
    with pytest.raises(ValidationError, match="legacy namespace"):
        V3RuntimeConfig.model_validate(runtime)
    with pytest.raises(RuntimeError, match="ASPD v3 main run is blocked"):
        bundle.require_ready()
    candidates = bundle.candidates.model_dump(mode="json")
    candidates["simultaneous_matrix_scope"] = ["o", "not_a_matrix"]
    with pytest.raises(ValidationError, match="invalid logical matrices"):
        V3CandidateConfig.model_validate(candidates)


def test_v3_registry_has_three_families_eight_layers_and_168_targets() -> None:
    registry = build_model_registry(_bundle().approved)
    assert tuple(registry) == ("qwen", "gemma", "phi")
    targets = [target for layout in registry.values() for target in layout.targets()]
    assert len(targets) == 168
    assert all(len(layout.layers) == 8 for layout in registry.values())
    assert registry["qwen"].layers == tuple(range(14, 22))
    assert registry["gemma"].layers == tuple(range(13, 21))
    assert registry["phi"].layers == tuple(range(12, 20))
    phi = {target.logical_name: target for target in registry["phi"].targets()[:7]}
    assert phi["q"].output_slice == OutputSlice(0, 3072)
    assert phi["k"].output_slice == OutputSlice(3072, 4096)
    assert phi["v"].output_slice == OutputSlice(4096, 5120)
    assert phi["mlp_gate"].output_slice == OutputSlice(0, 8192)
    assert phi["mlp_up"].output_slice == OutputSlice(8192, 16384)
    assert phi["o"].d_in == 3072 and phi["mlp_down"].d_in == 8192


def test_registry_runtime_metadata_validation_is_fail_closed() -> None:
    layout = build_model_registry(_bundle().approved)["qwen"]
    config = SimpleNamespace(
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )
    assert validate_runtime_config(layout, config)["status"] == "PASS"
    config.hidden_size = 1
    with pytest.raises(ValueError, match="architecture mismatch"):
        validate_runtime_config(layout, config)


def test_alignment_accepts_only_shared_exact_unicode_boundaries() -> None:
    text = "ab é"
    tokenizers = {
        "qwen": ToyTokenizer([(0, 1), (1, 2), (2, 3), (3, 4)]),
        "gemma": ToyTokenizer([(0, 2), (2, 3), (3, 4)]),
        "phi": ToyTokenizer([(0, 1), (1, 3), (3, 4)]),
    }
    result = align_canonical_text(
        text,
        document_id="doc-1",
        source_group="pile_uncopyrighted",
        tokenizers=tokenizers,
    )
    assert [anchor.character_boundary for anchor in result.anchors] == [3, 4]
    assert result.anchors[-1].byte_boundary == len(text.encode("utf-8"))
    assert tuple(result.anchors[-1].positions) == ("qwen", "gemma", "phi")
    assert result.report()["accepted_anchors"] == 2
    assert result.rejection_counts["not_shared_by_all_tokenizers"] > 0


def test_alignment_rejects_normalization_mismatch_empty_and_wrong_order() -> None:
    text = "é"
    tokenizers = {
        "qwen": ToyTokenizer([(0, 1)]),
        "gemma": ToyTokenizer([(0, 1)], mismatch=True),
        "phi": ToyTokenizer([(0, 1)]),
    }
    result = align_canonical_text(
        text,
        document_id="doc-2",
        source_group="pile_uncopyrighted",
        tokenizers=tokenizers,
    )
    assert not result.anchors
    assert result.rejection_counts["gemma:prefix_roundtrip_mismatch"] == 1
    with pytest.raises(AlignmentError, match="empty documents"):
        align_canonical_text(
            "", document_id="x", source_group="x", tokenizers=tokenizers
        )
    with pytest.raises(AlignmentError, match="deterministic configured model order"):
        align_canonical_text(
            text,
            document_id="x",
            source_group="x",
            tokenizers={"phi": tokenizers["phi"], "gemma": tokenizers["gemma"], "qwen": tokenizers["qwen"]},
        )


def test_alignment_records_truncation_and_ignores_padding_masks() -> None:
    text = "repeated repeated repeated"
    spans = [(0, 8), (8, 17), (17, len(text))]
    tokenizers = {
        "qwen": ToyTokenizer(spans, padding=2),
        "gemma": ToyTokenizer(spans, padding=2),
        "phi": ToyTokenizer(spans, padding=2),
    }
    result = align_canonical_text(
        text,
        document_id="doc-truncated",
        source_group="pile_uncopyrighted",
        tokenizers=tokenizers,
        max_length=3,
    )
    assert all(document.truncated for document in result.tokenized.values())
    assert all(
        len(document.input_ids) == len(document.attention_mask)
        for document in result.tokenized.values()
    )
    assert all(document.attention_mask[-2:] == (0, 0) for document in result.tokenized.values())
    assert result.report()["truncated_models"] == ["gemma", "phi", "qwen"]
    assert result.rejection_counts["qwen:truncated_suffix"] == 1


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(4, 9, bias=True)
        self.o_proj = nn.Linear(3, 4, bias=False)


class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(4, 10, bias=True)
        self.down_proj = nn.Linear(5, 4, bias=False)


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, x):
        residual_pre = self.input_layernorm(x)
        qkv = self.self_attn.qkv_proj(residual_pre)
        attention_output = self.self_attn.o_proj(qkv[..., :3])
        residual_mid = self.post_attention_layernorm(x + attention_output)
        gate_up = self.mlp.gate_up_proj(residual_mid)
        return residual_mid + self.mlp.down_proj(gate_up[..., :5])


class ToyHookModel(nn.Module):
    def __init__(self, n_layers: int = 1):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyLayer() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


@pytest.mark.parametrize(
    ("name", "module", "output_slice", "d_in", "d_out", "grounding"),
    [
        ("q", "self_attn.qkv_proj", OutputSlice(0, 3), 4, 3, "input_layernorm"),
        ("k", "self_attn.qkv_proj", OutputSlice(3, 6), 4, 3, "input_layernorm"),
        ("v", "self_attn.qkv_proj", OutputSlice(6, 9), 4, 3, "input_layernorm"),
        ("o", "self_attn.o_proj", None, 3, 4, "post_attention_layernorm"),
        ("mlp_gate", "mlp.gate_up_proj", OutputSlice(0, 5), 4, 5, "post_attention_layernorm"),
        ("mlp_up", "mlp.gate_up_proj", OutputSlice(5, 10), 4, 5, "post_attention_layernorm"),
        ("mlp_down", "mlp.down_proj", None, 5, 4, "post_attention_layernorm"),
    ],
)
def test_fused_hook_slices_reconstruct_native_output_and_preserve_model(
    name, module, output_slice, d_in, d_out, grounding
) -> None:
    torch.manual_seed(0)
    model = ToyHookModel()
    freeze_backbone(model)
    target = LogicalMatrixTarget(
        "phi",
        "phi3",
        0,
        name,
        "residual_pre" if grounding == "input_layernorm" else "residual_mid",
        (f"model.layers.0.{grounding}",),
        (f"model.layers.0.{module}",),
        output_slice,
        d_in,
        d_out,
        "slice_if_present",
    )
    x = torch.randn(2, 3, 4)
    clean = model(x)
    with LogicalMatrixCapture(model, target) as capture:
        hooked = model(x)
        observation = capture.observation()
    assert torch.equal(clean, hooked)
    assert validate_observation(model, target, observation)["status"] == "PASS"
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def test_eight_layer_residual_capture_preserves_configured_order() -> None:
    model = ToyHookModel(n_layers=8)
    freeze_backbone(model)
    base = build_model_registry(_bundle().approved)["phi"]
    layout = base.__class__(
        **{
            **base.__dict__,
            "layers": tuple(range(8)),
            "layer_prefixes": ("model.layers",),
            "hidden_size": 4,
        }
    )
    x = torch.randn(1, 2, 4)
    clean = model(x)
    with ResidualWindowCapture(model, layout, "residual_pre") as capture:
        hooked = model(x)
        values = capture.ordered()
    assert torch.equal(clean, hooked)
    assert len(values) == 8 and all(value.shape == (1, 2, 4) for value in values)
    assert not torch.equal(values[0], values[-1])
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def test_hierarchical_encoder_shapes_order_separators_and_identity() -> None:
    torch.manual_seed(0)
    packers = {
        "qwen": ChannelSlicePacker(16, 4),
        "gemma": ChannelSlicePacker(24, 4),
        "phi": ChannelSlicePacker(32, 4),
    }
    encoder = HierarchicalV3Encoder(
        packers,
        intra_layers=1,
        intra_heads=2,
        cross_layers=1,
        cross_heads=1,
        n_features=7,
        readout="mean",
    )
    activations = {
        "qwen": [torch.randn(2, 3, 2) for _ in range(8)],
        "gemma": [torch.randn(2, 3, 3) for _ in range(8)],
        "phi": [torch.randn(2, 3, 4) for _ in range(8)],
    }
    valid = torch.tensor([[True, True, False], [True, True, True]])
    scores = encoder(activations, valid)
    assert scores.shape == (2, 3, 7)
    assert (scores >= 0).all() and not scores[~valid].any()
    tokens = {key: torch.randn(1, 2, 4, width) for key, width in zip(packers, (4, 6, 8), strict=True)}
    packed = pack_cross_model_tokens(tokens)
    assert packed.values.shape == (1, 2, 14, 13)
    assert packed.separator_positions == (4, 9)
    assert packed.identity_indices == (0, 0, 0, 0, 3, 1, 1, 1, 1, 4, 2, 2, 2, 2)


def test_v3_batch_topk_exact_budget_gradients_and_zero_scores() -> None:
    scores = torch.arange(1, 49, dtype=torch.float32).reshape(2, 3, 8).requires_grad_()
    valid = torch.tensor([[True, True, False], [True, True, True]])
    gate = V3BatchTopK(8, 2)
    code = gate(scores, valid)
    assert code.n_selected == 2 * int(valid.sum())
    code.values.sum().backward()
    assert torch.equal(scores.grad != 0, code.gate)
    zeros = torch.zeros(1, 2, 8)
    zero_code = gate(zeros, torch.ones(1, 2, dtype=torch.bool), update_statistics=False)
    assert zero_code.n_selected == 0
    with pytest.raises(ValueError, match="non-negative"):
        gate(-torch.ones(1, 1, 8), torch.ones(1, 1, dtype=torch.bool))


def test_rank_one_bank_supports_both_explicit_unapproved_semantics() -> None:
    target = LogicalMatrixTarget(
        "qwen", "qwen3", 0, "o", "residual_mid", ("g",), ("m",), None, 2, 3,
        "physical_if_present",
    )
    bank = RankOneMatrixBank([target], 4, {target.key: torch.ones(3)})
    x = torch.randn(1, 2, 2)
    code = torch.tensor([[[0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    binary = bank.reconstruct(target.key, x, code, valid, code_semantics="binary_support")
    continuous = bank.reconstruct(target.key, x, code, valid, code_semantics="continuous")
    assert binary.shape == continuous.shape == (1, 2, 3)
    assert not torch.equal(binary, continuous)
    with pytest.raises(RuntimeError, match="researcher approval"):
        require_resolved_objective(
            gate_grouping=None,
            code_semantics=None,
            gradient_route=None,
            activation_loss_weight=None,
            internal_loss_weight=None,
        )


def test_document_split_dedup_and_near_duplicate_are_deterministic(tmp_path: Path) -> None:
    policy = HashSplitPolicy(10, frozenset({0}), frozenset({1}), seed=42)
    document = SourceDocument("doc-1", "pile_uncopyrighted", "one two three four five six", "1")
    assert policy.split(document) == policy.split(document)
    with SQLiteDuplicateIndex(
        tmp_path / "dedup.sqlite",
        shingle_width=3,
        band_size=1,
        near_duplicate_threshold=0.8,
    ) as index:
        split = policy.split(document)
        assert index.classify(document, split)["status"] == "UNIQUE"
        duplicate = SourceDocument("doc-2", "pile_uncopyrighted", document.text, "2")
        assert index.classify(duplicate, "test")["status"] == "EXACT_DUPLICATE"
        words = [f"word-{index}" for index in range(40)]
        original = SourceDocument("doc-long", "pile_uncopyrighted", " ".join(words), "long")
        assert index.classify(original, "train")["status"] == "UNIQUE"
        words[20] = "replacement"
        near = SourceDocument("doc-3", "pile_uncopyrighted", " ".join(words), "3")
        assert index.classify(near, "validation")["status"] == "NEAR_DUPLICATE"
        clean = SourceDocument("doc-4", "pile_uncopyrighted", "alpha beta gamma", "4")
        assert index.classify(clean, "train")["status"] == "UNIQUE"


def test_manifest_uses_disk_ledger_and_rejects_cross_split_document(tmp_path: Path) -> None:
    tokenizers = {
        "qwen": ToyTokenizer([(0, 1), (1, 2)]),
        "gemma": ToyTokenizer([(0, 1), (1, 2)]),
        "phi": ToyTokenizer([(0, 1), (1, 2)]),
    }
    alignment = align_canonical_text(
        "ab",
        document_id="doc-ledger",
        source_group="pile_uncopyrighted",
        tokenizers=tokenizers,
    )
    policy = HashSplitPolicy(10, frozenset({0}), frozenset({1}), seed=42)
    document = SourceDocument("doc-ledger", "pile_uncopyrighted", "ab", "row-1")
    with ManifestAccumulator(
        dataset_id="monology/pile-uncopyrighted",
        dataset_revision="a" * 40,
        split_policy=policy,
        tokenizer_revisions={"qwen": "q", "gemma": "g", "phi": "p"},
        ledger_path=tmp_path / "split-ledger.sqlite",
    ) as accumulator:
        accumulator.add(document, "train", alignment)
        with pytest.raises(ValueError, match="already assigned"):
            accumulator.add(document, "validation", alignment)
        manifest = accumulator.manifest()
    assert manifest["split_ledger_rows"] == 1
    assert manifest["documents_by_split"] == {"train": 1, "validation": 0, "test": 0}


def _write_stream_fixture(root: Path) -> Path:
    rows = [
        {"anchor_id": f"a{index}", "model_order": ["qwen", "gemma", "phi"], "value": index}
        for index in range(7)
    ]
    shards = []
    for shard_index, subset in enumerate((rows[:3], rows[3:])):
        path = root / f"shard_{shard_index}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in subset), encoding="utf-8")
        shards.append(
            {
                "split": "train",
                "shard_index": shard_index,
                "path": path.name,
                "records": len(subset),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 3, "shards": shards}), encoding="utf-8")
    return manifest


def test_stream_resume_matches_uninterrupted_exactly(tmp_path: Path) -> None:
    manifest = _write_stream_fixture(tmp_path)
    uninterrupted = [row["anchor_id"] for row in AlignedAnchorStream(
        manifest, "train", seed=42, shuffle_buffer_size=3
    )]
    first = AlignedAnchorStream(manifest, "train", seed=42, shuffle_buffer_size=3)
    iterator = iter(first)
    prefix = [next(iterator)["anchor_id"] for _ in range(3)]
    state = first.state_dict()
    resumed = AlignedAnchorStream(manifest, "train", seed=42, shuffle_buffer_size=3)
    resumed.load_state_dict(state)
    suffix = [row["anchor_id"] for row in resumed]
    assert prefix + suffix == uninterrupted
    state["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance mismatch"):
        resumed.load_state_dict(state)


def test_exposure_checkpoint_manifest_and_finite_contracts(tmp_path: Path) -> None:
    with SQLiteExposureLedger(
        tmp_path / "exposure.sqlite", dataset_manifest_sha256="e" * 64
    ) as ledger:
        ledger.observe_optimizer_step(0, ["a", "b"])
        counters = ledger.observe_optimizer_step(1, ["b", "c"])
        exposure_snapshot = ledger.backup_to(
            tmp_path / "exposure-step-2.sqlite", expected_optimizer_steps=2
        )
    exposure_sha256 = hashlib.sha256(exposure_snapshot.read_bytes()).hexdigest()
    assert counters.accepted_aligned_anchor_exposures == 4
    assert counters.newly_observed_anchor_exposures == 3
    assert counters.repeated_anchor_exposures == 1
    assert minimum_optimizer_steps(2_048_000_000, 8192) == 250_000
    assert DistributedPlan(8, 128, 8).effective_anchors_per_optimizer_step == 8192
    assert deterministic_rank_shard(["a", "b", "c", "d"], 1, 2) == ["b", "d"]
    with pytest.raises(FloatingPointError, match="loss"):
        validate_finite_tensors({"loss": torch.tensor(float("nan"))})

    generator = __import__("random").Random(42)
    state = V3TrainingState(
        schema_version=3,
        optimizer_step=2,
        stream_cursor=AlignedStreamCursor(1, 2, 3, 0, 3, 2),
        counters=counters,
        dataset_manifest_sha256="e" * 64,
        alignment_manifest_sha256="b" * 64,
        config_fingerprint="c" * 64,
        sampler_rng_state_b64=encode_rng_state(generator),
        exposure_ledger_optimizer_step=2,
        exposure_ledger_sha256=exposure_sha256,
        per_rank_microbatch_index={"0": 4},
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"synthetic checkpoint")
    marker = commit_coordinated_checkpoint(
        tmp_path / "checkpoints",
        state=state,
        model_checkpoint=weights,
        exposure_ledger_snapshot=exposure_snapshot,
    )
    restored, restored_weights = restore_coordinated_checkpoint(
        marker,
        expected_dataset_manifest_sha256="e" * 64,
        expected_alignment_manifest_sha256="b" * 64,
        expected_config_fingerprint="c" * 64,
    )
    assert restored.payload() == state.payload() and restored_weights == weights.resolve()
    with pytest.raises(ValueError, match="provenance mismatch"):
        restore_coordinated_checkpoint(
            marker,
            expected_dataset_manifest_sha256="d" * 64,
            expected_alignment_manifest_sha256="b" * 64,
            expected_config_fingerprint="c" * 64,
        )
    exposure_snapshot.write_bytes(exposure_snapshot.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="exposure-ledger snapshot checksum"):
        restore_coordinated_checkpoint(
            marker,
            expected_dataset_manifest_sha256="e" * 64,
            expected_alignment_manifest_sha256="b" * 64,
            expected_config_fingerprint="c" * 64,
        )


def test_feasibility_exposes_full_scope_instead_of_silently_shrinking() -> None:
    bundle = _bundle()
    report = estimate_v3_feasibility(bundle, build_model_registry(bundle.approved))
    assert report["logical_matrix_targets"] == 168
    assert report["dictionary_scenarios"]["8192"]["rank_one_parameters"] > 10_000_000_000
    assert report["encoder_parameters"] is None
    assert report["status"] == "BLOCKED_FOR_MAIN_RUN"


def test_historical_mix3_remains_loadable_and_immutable_writer_refuses_overwrite(
    tmp_path: Path,
) -> None:
    historical = load_experiment_config(
        ROOT / "configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_mix3_200m.yaml"
    )
    assert historical.mix3_aspd is not None
    assert historical.mix3_aspd.optimizer_token_exposures == 204_800_000
    target = tmp_path / "manifest.json"
    write_json_immutable(target, {"status": "first"})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_json_immutable(target, {"status": "second"})
