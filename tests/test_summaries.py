"""Per-component summaries: the alive restriction and key stability."""

import torch

from aspd.summaries import alive_summary, component_summary


def test_quantiles_and_mean_are_the_plain_ones():
    v = torch.arange(101).float()
    out = component_summary("pr", v)
    assert torch.isclose(out["pr_median"], torch.tensor(50.0))
    assert torch.isclose(out["pr_p10"], torch.tensor(10.0))
    assert torch.isclose(out["pr_p90"], torch.tensor(90.0))
    assert torch.isclose(out["pr_mean"], torch.tensor(50.0))


def test_alive_restriction_actually_restricts():
    """The reason this exists: a dead component's `nmse` is divided by an eps-clamped variance and
    explodes, so an unrestricted median over C=8192 can report the dead population rather than the
    decomposition.
    """
    v = torch.cat([torch.full((90,), 1e6), torch.arange(10).float()])  # 90 "dead", 10 live
    alive = torch.cat([torch.zeros(90, dtype=torch.bool), torch.ones(10, dtype=torch.bool)])
    out = component_summary("nmse", v, alive=alive)
    assert out["nmse_median"] == 1e6
    assert out["nmse_alive_median"] < 10


def test_keys_are_identical_whether_or_not_anything_is_alive():
    v = torch.randn(32)
    alive = torch.ones(32, dtype=torch.bool)
    assert set(component_summary("pr", v, alive=alive)) == set(
        component_summary("pr", v, alive=~alive)
    )


def test_legacy_median_alive_alias_is_opt_in():
    """`pr_median_alive` is `OutputSparsityLoss`'s pre-existing spelling and must keep resolving to
    the same number as the uniform `pr_alive_median` -- but ONLY where it has logged history.
    Emitting it everywhere put the same number under two names on five keys nothing refers to.
    """
    v, alive = torch.randn(64), torch.rand(64) > 0.5
    assert "pr_median_alive" not in component_summary("pr", v, alive=alive)
    out = component_summary("pr", v, alive=alive, legacy_median_alive_alias=True)
    assert out["pr_median_alive"] == out["pr_alive_median"]


def test_alive_summary_threshold():
    gbar = torch.tensor([0.0, 0.005, 0.02, 1.0])
    alive, stats = alive_summary(gbar, 0.01)
    assert alive.tolist() == [False, False, True, True]
    assert stats["n_alive"] == 2.0
    assert torch.isclose(stats["alive_frac"], torch.tensor(0.5))


