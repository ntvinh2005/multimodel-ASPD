import torch

from aspd.multimodel.components import RankOneComponents


def test_sparse_rank_one_reconstruction_matches_dense_equation() -> None:
    torch.manual_seed(0)
    component = RankOneComponents(n_features=5, d_in=3, d_out=4)
    x = torch.randn(2, 3, 3)
    gate = torch.rand(2, 3, 5) > 0.6
    valid = torch.tensor([[True, True, False], [True, True, True]])
    sparse = component.reconstruct(x, gate, valid)
    reads = torch.einsum("btd,cd->btc", x, component.V)
    dense = torch.einsum("btc,co->bto", reads * gate, component.U)
    dense[~valid] = 0
    torch.testing.assert_close(sparse, dense)
