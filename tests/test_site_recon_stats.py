"""`site_recon_stats`: the FVU reported for L_internal."""

import torch

from aspd.loss_utils import site_recon_stats


def test_site_recon_stats_is_the_reported_fvu():
    """`sq_err / sq_tot` with the denominator centred PER BATCH -- `aspd`'s ReconAccumulator."""
    g = torch.Generator().manual_seed(11)
    target = torch.randn(9, 4, generator=g, dtype=torch.float64)
    pred = target + 0.1 * torch.randn(9, 4, generator=g, dtype=torch.float64)

    expected = (
        (pred - target).pow(2).sum() / (target - target.mean(dim=0, keepdim=True)).pow(2).sum()
    )
    torch.testing.assert_close(site_recon_stats(pred, target)["fvu"], expected)
    torch.testing.assert_close(
        site_recon_stats(target, target)["fvu"], torch.zeros((), dtype=torch.float64)
    )
