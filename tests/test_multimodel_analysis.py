import torch

from aspd.multimodel.analysis import _factor_metrics
from aspd.multimodel.losses import residual_fvu


def test_p5_component_metrics_are_invariant_to_rank_one_gauge() -> None:
    torch.manual_seed(0)
    u = torch.randn(4, 5)
    v = torch.randn(4, 3)
    scale = torch.tensor([2.0, -3.0, 0.5, -0.25])
    metrics = _factor_metrics(u, v, u * scale[:, None], v / scale[:, None])
    torch.testing.assert_close(metrics["relative_component_change"], torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(metrics["component_cosine"], torch.ones(4), atol=1e-6, rtol=0)


def test_auxk_uses_original_target_variance() -> None:
    target = torch.tensor([[[0.0], [2.0]]])
    residual = torch.tensor([[[0.0], [1.0]]])
    residual_reconstruction = torch.zeros_like(residual)
    valid = torch.ones(1, 2, dtype=torch.bool)
    # Numerator is 1 and the original target's centered sum of squares is 2.
    torch.testing.assert_close(
        residual_fvu(target, residual, residual_reconstruction, valid),
        torch.tensor(0.5),
    )
