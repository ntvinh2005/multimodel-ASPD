import torch

from aspd.multimodel.batch_topk import batch_topk
from aspd.multimodel.config import SparsitySpec


def test_d0_batch_topk_keeps_k_per_valid_token_on_average() -> None:
    pre = torch.arange(1, 1 + 2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    cfg = SparsitySpec(n_features=8, diffing="D0", top_k=2)
    code = batch_topk(pre, valid, cfg)
    assert code.gate.sum().item() == 2 * valid.sum().item()
    assert not code.gate[~valid].any()


def test_dual_k_applies_independent_shared_and_exclusive_budgets() -> None:
    pre = torch.rand(2, 2, 10) + 0.1
    valid = torch.ones(2, 2, dtype=torch.bool)
    cfg = SparsitySpec(
        n_features=10,
        diffing="D1",
        shared_fraction=0.6,
        top_k_shared=2,
        top_k_exclusive=1,
    )
    code = batch_topk(pre, valid, cfg)
    assert code.gate[..., :6].sum().item() == 2 * valid.sum().item()
    assert code.gate[..., 6:].sum().item() == valid.sum().item()


def test_s2_weight_changes_support_but_not_kept_activation_value() -> None:
    pre = torch.tensor([[[5.0, 4.0]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    cfg = SparsitySpec(n_features=2, diffing="D0", top_k=1, selection_score="S2")
    code = batch_topk(pre, valid, cfg, ranking_weights=torch.tensor([0.1, 10.0]))
    assert code.gate.tolist() == [[[False, True]]]
    assert code.values[0, 0, 1].item() == 4.0
