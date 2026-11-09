from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from aspd.multimodel.cli.evaluate_aspd_v3_granite_normalized_target import _load_contract
from aspd.multimodel.cli.train_aspd_v3_2b import _load_training_config
from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
from aspd.multimodel.v3.objective import (
    auxk_dead_feature_loss,
    detached_binary_support,
    full_activation_fvu,
    masked_fvu,
    matryoshka_activation_fvu,
)
from aspd.multimodel.v3.reconstruction import ActivationDecoder
from aspd.multimodel.v3.sparse_support import prepare_sparse_support
from aspd.multimodel.v3.two_model_matryoshka_config import (
    load_two_model_matryoshka_config,
)

ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "configs/multimodel/aspd_v3"
BASELINE = CONFIG_DIR / "two_model_matryoshka_v1.yaml"
TREATMENT = CONFIG_DIR / "two_model_full_only_v1.yaml"
EVALUATION = CONFIG_DIR / "two_model_matryoshka_evaluation_v1.yaml"


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ActivationDecoder]:
    torch.manual_seed(7)
    code = torch.randn(2, 3, 16).relu().requires_grad_()
    code.data[code.data < 0.45] = 0
    target = torch.randn(2, 3, 5)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    decoder = ActivationDecoder(16, 5, bias=True, dtype=torch.float32)
    return code, target, valid, decoder


def test_full_reconstruction_matches_matryoshka_full_prefix() -> None:
    code, target, valid, decoder = _fixture()
    support = prepare_sparse_support(code, valid, binary=False)
    _matryoshka, matryoshka_full = matryoshka_activation_fvu(
        code,
        decoder,
        target,
        valid,
        support=support,
        sparse_backend="dense",
    )
    _full_loss, full = full_activation_fvu(
        code,
        decoder,
        target,
        valid,
        support=support,
        sparse_backend="dense",
    )
    # Grouped prefix accumulation and one full GEMM differ only by FP32 summation order.
    torch.testing.assert_close(full, matryoshka_full, rtol=1e-6, atol=1e-7)


def test_full_only_is_exact_full_fvu_and_gradients_match_reference() -> None:
    code, target, valid, decoder = _fixture()
    reference_code = code.detach().clone().requires_grad_()
    reference_decoder = ActivationDecoder(16, 5, bias=True, dtype=torch.float32)
    reference_decoder.load_state_dict(decoder.state_dict())

    loss, prediction = full_activation_fvu(
        code, decoder, target, valid, sparse_backend="dense"
    )
    reference_prediction = reference_decoder(reference_code)
    assert isinstance(reference_prediction, torch.Tensor)
    reference_prediction = reference_prediction * valid[..., None]
    reference_loss = masked_fvu(reference_prediction, target, valid)
    torch.testing.assert_close(prediction, reference_prediction, rtol=0, atol=0)
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(code.grad, reference_code.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        decoder.weight.grad, reference_decoder.weight.grad, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        decoder.bias.grad, reference_decoder.bias.grad, rtol=1e-6, atol=1e-7
    )
    assert torch.isfinite(decoder.weight.grad).all()


def test_auxk_uses_identical_full_residual_in_both_arms() -> None:
    code, target, valid, decoder = _fixture()
    support = prepare_sparse_support(code, valid, binary=False)
    _, baseline_full = matryoshka_activation_fvu(
        code, decoder, target, valid, support=support, sparse_backend="dense"
    )
    _, treatment_full = full_activation_fvu(
        code, decoder, target, valid, support=support, sparse_backend="dense"
    )
    dead = torch.zeros(16, dtype=torch.bool)
    dead[8:] = True
    baseline_aux = auxk_dead_feature_loss(
        code,
        decoder,
        target,
        baseline_full,
        valid,
        dead,
        top_k_aux=4,
        sparse_backend="dense",
    )
    treatment_aux = auxk_dead_feature_loss(
        code,
        decoder,
        target,
        treatment_full,
        valid,
        dead,
        top_k_aux=4,
        sparse_backend="dense",
    )
    torch.testing.assert_close(baseline_aux, treatment_aux, rtol=1e-6, atol=1e-7)
    baseline_gradients = torch.autograd.grad(
        baseline_aux,
        (code, decoder.weight, decoder.bias),
        retain_graph=True,
        allow_unused=True,
    )
    treatment_gradients = torch.autograd.grad(
        treatment_aux, (code, decoder.weight, decoder.bias), allow_unused=True
    )
    for baseline_gradient, treatment_gradient in zip(
        baseline_gradients, treatment_gradients, strict=True
    ):
        if baseline_gradient is None or treatment_gradient is None:
            assert baseline_gradient is treatment_gradient is None
        else:
            torch.testing.assert_close(
                baseline_gradient, treatment_gradient, rtol=1e-5, atol=1e-6
            )


def test_internal_binary_support_remains_detached() -> None:
    code = torch.tensor([[[0.0, 0.4, 0.0, 0.8]]], requires_grad=True)
    support = detached_binary_support(code)
    assert not support.requires_grad
    assert support.tolist() == [[[0.0, 1.0, 0.0, 1.0]]]


@pytest.mark.parametrize("backend", ["indexed", "csr", "dense"])
def test_full_reconstruction_sparse_dense_forward_and_gradient_parity(backend: str) -> None:
    code, target, valid, decoder = _fixture()
    active = (code.detach() > 0).to(code.dtype)
    raw_code = code.detach().clone().requires_grad_()
    code = raw_code * active
    reference_raw_code = raw_code.detach().clone().requires_grad_()
    reference_code = reference_raw_code * active
    reference_decoder = ActivationDecoder(16, 5, bias=True, dtype=torch.float32)
    reference_decoder.load_state_dict(decoder.state_dict())
    support = prepare_sparse_support(code, valid, binary=False)
    reference_support = prepare_sparse_support(reference_code, valid, binary=False)
    loss, prediction = full_activation_fvu(
        code,
        decoder,
        target,
        valid,
        support=support,
        sparse_backend=backend,
    )
    reference_loss, reference_prediction = full_activation_fvu(
        reference_code,
        reference_decoder,
        target,
        valid,
        support=reference_support,
        sparse_backend="dense",
    )
    torch.testing.assert_close(prediction, reference_prediction, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(loss, reference_loss, rtol=1e-6, atol=1e-6)
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(
        raw_code.grad, reference_raw_code.grad, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        decoder.weight.grad, reference_decoder.weight.grad, rtol=1e-5, atol=1e-6
    )


def test_configs_are_matched_and_fail_closed_against_qwen(tmp_path: Path) -> None:
    baseline = load_two_model_matryoshka_config(BASELINE)
    treatment = load_two_model_matryoshka_config(TREATMENT)
    assert baseline.fingerprint() != treatment.fingerprint()
    assert baseline.objective.activation_target_normalization == "per_anchor_l2"
    assert treatment.objective.activation_decoder_normalization == "none"
    assert baseline.architecture.encoder_mixer == treatment.architecture.encoder_mixer
    assert tuple(model.key for model in baseline.models) == ("gemma", "smol")
    assert all("qwen" not in model.model_id.lower() for model in baseline.models)
    assert _load_training_config(BASELINE).fingerprint() == baseline.fingerprint()
    assert _load_training_config(TREATMENT).fingerprint() == treatment.fingerprint()
    assert _load_contract(EVALUATION)["model_order"] == ["gemma", "smol"]

    raw = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    raw["models"][1]["model_id"] = "Qwen/forbidden"
    forbidden = tmp_path / "forbidden.yaml"
    forbidden.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="source registry mismatch|Qwen"):
        load_two_model_matryoshka_config(forbidden)


def test_checkpoint_provenance_rejects_other_objective_fingerprint(tmp_path: Path) -> None:
    baseline = load_two_model_matryoshka_config(BASELINE)
    treatment = load_two_model_matryoshka_config(TREATMENT)
    checkpoint = tmp_path / "step-000001221"
    rank = checkpoint / "rank-00000"
    dcp = rank / "dcp"
    dcp.mkdir(parents=True)
    (dcp / ".metadata").write_bytes(b"metadata")
    (rank / "sidecar.json").write_text("{}\n", encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "status": "COMPLETE",
        "world_size": 2,
        "config_fingerprint": baseline.fingerprint(),
        "config_file_sha256": "baseline-config",
        "dataset_manifest_sha256": "dataset",
        "rank_sidecar_sha256": {"rank-00000": digest(rank / "sidecar.json")},
        "rank_dcp_metadata_sha256": {"rank-00000": digest(dcp / ".metadata")},
    }
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (checkpoint / "COMPLETE").write_text(
        json.dumps({"manifest_sha256": digest(manifest_path)}), encoding="utf-8"
    )
    manager = ModelOwnedCheckpoint(tmp_path, keep_complete=2)
    with pytest.raises(ValueError, match="config_fingerprint"):
        manager._validate(
            checkpoint,
            config_fingerprint=treatment.fingerprint(),
            config_file_sha256="baseline-config",
            dataset_manifest_sha256="dataset",
            expected_world_size=2,
        )


def test_scientific_configs_differ_only_in_objective_and_namespaces() -> None:
    baseline = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    treatment = yaml.safe_load(TREATMENT.read_text(encoding="utf-8"))
    for payload in (baseline, treatment):
        payload.pop("arm_id")
        payload["objective"].pop("activation_loss")
        payload["objective"].pop("activation_objective_mode")
        for key in ("output_root", "checkpoint_root", "preflight_evidence"):
            payload["runtime"].pop(key)
    assert baseline == treatment
