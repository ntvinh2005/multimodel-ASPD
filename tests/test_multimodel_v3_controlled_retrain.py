from pathlib import Path

import torch
from torch import nn

from aspd.multimodel.v3.controlled_retrain import (
    configure_controlled_trainables,
    load_controlled_retrain_contract,
    reduce_controlled_shared_gradients_,
)

ROOT = Path(__file__).resolve().parents[1]


class _DummyControlledModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix_bank = nn.Linear(3, 3)
        self.activation_banks = nn.ModuleDict({"pre": nn.Linear(3, 3)})
        self.local_encoders = nn.ModuleDict({"pre": nn.Linear(3, 3)})
        self.shared_encoders = nn.ModuleDict({"pre": nn.Linear(3, 3)})

    def shared_parameters(self):
        yield from self.shared_encoders.parameters()


def test_controlled_contracts_freeze_exact_accounting_and_arms() -> None:
    observed = {}
    for name in (
        "controlled_retrain_10m_shared_v1.yaml",
        "controlled_retrain_10m_gemma_v1.yaml",
    ):
        contract = load_controlled_retrain_contract(
            ROOT / "configs/multimodel/aspd_v3" / name
        )
        assert contract.additional_optimizer_steps == 1221
        assert contract.additional_accepted_exposures == 10_002_432
        assert contract.additional_optimizer_steps * 8192 == 10_002_432
        assert contract.source_checkpoint_step == 61_036
        assert contract.source_accepted_exposures == 500_006_912
        observed[contract.arm] = contract
    assert set(observed) == {"shared_control", "gemma_specific"}
    shared = observed["shared_control"].model_dump()
    gemma = observed["gemma_specific"].model_dump()
    differing = {key for key in shared if shared[key] != gemma[key]}
    assert differing == {"arm", "output_root"}


def test_controlled_optimizer_excludes_rank_one_matrix_factors() -> None:
    model = _DummyControlledModel()
    groups = configure_controlled_trainables(model)  # type: ignore[arg-type]
    assert [group["scope"] for group in groups] == [
        "activation_decoders",
        "local_encoders",
        "shared_encoders",
    ]
    matrix_ids = {id(parameter) for parameter in model.matrix_bank.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in groups
        for parameter in group["params"]  # type: ignore[union-attr]
    }
    assert matrix_ids.isdisjoint(optimizer_ids)
    assert all(not parameter.requires_grad for parameter in model.matrix_bank.parameters())
    assert optimizer_ids == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_source_configuration_checksum_is_frozen() -> None:
    import hashlib

    source = ROOT / "configs/multimodel/aspd_v3/main_2b_v1.yaml"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    contract = load_controlled_retrain_contract(
        ROOT / "configs/multimodel/aspd_v3/controlled_retrain_10m_shared_v1.yaml"
    )
    assert digest == contract.source_config_sha256


def test_gradient_topology_separates_gemma_but_keeps_qwen_smol_paired(monkeypatch) -> None:
    model = _DummyControlledModel()
    for parameter in model.shared_parameters():
        parameter.grad = torch.ones_like(parameter)
    calls: list[object | None] = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 3)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda _tensor, op=None, group=None: calls.append(group),
    )
    subgroup = object()
    for rank, expected_calls in ((0, 2), (1, 0), (2, 2)):
        calls.clear()
        monkeypatch.setattr(torch.distributed, "get_rank", lambda rank=rank: rank)
        reduce_controlled_shared_gradients_(
            model,  # type: ignore[arg-type]
            arm="gemma_specific",
            qwen_smol_group=subgroup,  # type: ignore[arg-type]
        )
        assert len(calls) == expected_calls
        assert all(group is subgroup for group in calls)

    calls.clear()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    reduce_controlled_shared_gradients_(
        model,  # type: ignore[arg-type]
        arm="shared_control",
        qwen_smol_group=None,
    )
    assert calls == [None, None]
