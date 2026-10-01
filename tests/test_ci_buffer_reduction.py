"""BatchTopK thresholds and dead counters are reduced over every data-parallel rank."""

import torch

from aspd.ci.pd_transcoder import PDTranscoderCiFn


class _Buffers:
    """Just the two buffers and the one config field the update methods read."""

    def __init__(self, n_features: int = 6, threshold_lr: float = 0.5):
        self.threshold = torch.zeros(())
        self.n_batches_not_active = torch.zeros(n_features)

        class _Cfg:
            pass

        self.cfg = _Cfg()
        self.cfg.threshold_lr = threshold_lr

    update_threshold = PDTranscoderCiFn._update_threshold
    update_inactive = PDTranscoderCiFn._update_inactive


class _FakeDist:

    def __init__(self, monkeypatch, other: torch.Tensor):
        self.other = other
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "all_reduce", self._all_reduce)

    def _all_reduce(self, t, op):
        other = self.other.to(t.dtype)
        if op is torch.distributed.ReduceOp.MIN:
            t.copy_(torch.minimum(t, other))
        elif op is torch.distributed.ReduceOp.MAX:
            t.copy_(torch.maximum(t, other))
        else:
            raise AssertionError(f"unexpected reduce op {op}")


def test_single_process_updates_are_unchanged():
    b = _Buffers()
    acts = torch.tensor([[0.0, 2.0, 0.0, 0.0, 5.0, 0.0]])
    b.update_threshold(acts)
    assert torch.isclose(b.threshold, torch.tensor(1.0)), b.threshold  # 0.5 * min(2, 5)
    b.update_inactive(acts)
    assert b.n_batches_not_active.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_the_threshold_takes_the_minimum_over_ALL_ranks(monkeypatch):
    _FakeDist(monkeypatch, torch.tensor(0.5))  # the other rank's smallest positive act
    b = _Buffers()
    b.update_threshold(torch.tensor([[0.0, 2.0, 0.0, 0.0, 5.0, 0.0]]))
    assert torch.isclose(b.threshold, torch.tensor(0.25)), b.threshold  # 0.5 * min(2, 0.5)


def test_a_rank_with_no_positive_activation_still_reaches_the_collective(monkeypatch):
    """`+inf` stands in for an empty shard: it loses the MIN and contributes nothing, and the rank
    still participates. An early return on `positive.any()` would deadlock every other rank.
    """
    _FakeDist(monkeypatch, torch.tensor(3.0))
    b = _Buffers()
    b.update_threshold(torch.zeros(1, 6))  # nothing positive on THIS rank
    assert torch.isclose(b.threshold, torch.tensor(1.5)), b.threshold  # 0.5 * 3.0, the other rank's


def test_the_threshold_is_untouched_when_NO_rank_fires(monkeypatch):
    _FakeDist(monkeypatch, torch.tensor(float("inf")))
    b = _Buffers()
    b.threshold = torch.tensor(0.75)
    b.update_threshold(torch.zeros(1, 6))
    assert torch.isclose(b.threshold, torch.tensor(0.75))


def test_a_latent_firing_on_ANOTHER_rank_is_not_counted_dead(monkeypatch):
    _FakeDist(monkeypatch, torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    b = _Buffers()
    b.update_inactive(torch.tensor([[0.0, 2.0, 0.0, 0.0, 5.0, 0.0]]))
    #                                 dead  live  dead  LIVE-ELSEWHERE  live  dead
    assert b.n_batches_not_active.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def test_the_dead_clock_still_advances_for_a_latent_no_rank_fires(monkeypatch):
    _FakeDist(monkeypatch, torch.zeros(6))
    b = _Buffers()
    b.n_batches_not_active = torch.full((6,), 7.0)
    b.update_inactive(torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]]))
    assert b.n_batches_not_active.tolist() == [8.0, 0.0, 8.0, 8.0, 8.0, 8.0]


def test_two_ranks_agree_with_one_rank_holding_the_same_tokens(monkeypatch):
    shard_a = torch.tensor([[0.0, 2.0, 0.0, 0.0, 5.0, 0.0]])
    shard_b = torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.0, 0.0]])

    single = _Buffers()
    single.update_threshold(torch.cat([shard_a, shard_b]))
    single.update_inactive(torch.cat([shard_a, shard_b]))

    _FakeDist(monkeypatch, torch.tensor(0.5))
    rank0 = _Buffers()
    rank0.update_threshold(shard_a)
    monkeypatch.setattr(torch.distributed, "all_reduce",
                        _FakeDist(monkeypatch, (shard_b.sum(0) > 0).float())._all_reduce)
    rank0.update_inactive(shard_a)

    assert torch.isclose(rank0.threshold, single.threshold), (rank0.threshold, single.threshold)
    assert rank0.n_batches_not_active.tolist() == single.n_batches_not_active.tolist()
