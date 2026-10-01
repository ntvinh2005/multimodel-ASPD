"""The alive set: components that fired within the last `window_tokens` tokens."""

import pytest
import torch

from aspd.alive import AliveTracker, alive_from_training_state

C, B, S = 6, 2, 8
TOKENS_PER_BATCH = B * S


def _ci(fired: list[int]) -> torch.Tensor:
    """Causal importances where exactly `fired` components are nonzero somewhere in the batch."""
    ci = torch.zeros(B, S, C)
    for c in fired:
        ci[0, 0, c] = 0.7
    return ci


def test_everything_starts_alive():
    """A fresh counter is 0 tokens since active, which is inside any window. Matching an SAE's
    dead-latent counter: nothing is dead until it has had a chance to fire and did not.
    """
    assert AliveTracker(C).alive.all()


def test_a_component_dies_only_after_the_full_window():
    tracker = AliveTracker(C, window_tokens=10 * TOKENS_PER_BATCH)
    for _ in range(9):
        tracker.observe(_ci([0]))
    assert tracker.alive.all(), "died before the window elapsed"
    tracker.observe(_ci([0]))
    alive = tracker.alive
    assert alive[0] and not alive[1:].any()


def test_one_firing_resurrects_for_a_full_window():
    """A component that fires on ONE token in the window is alive. This is the property a
    single-batch alive test would get wrong -- rare-but-real components are not dead.
    """
    tracker = AliveTracker(C, window_tokens=3 * TOKENS_PER_BATCH)
    for _ in range(5):
        tracker.observe(_ci([]))
    assert not tracker.alive.any()
    tracker.observe(_ci([2]))
    assert tracker.alive[2] and tracker.alive.sum() == 1


def test_firing_is_strictly_nonzero_not_a_magnitude_threshold():
    """`LowerLeakyHardSigmoid` clamps to EXACTLY zero, so `g > 0` is a real on/off event and needs
    no cutoff. A tiny-but-nonzero importance counts as firing; that is deliberate.
    """
    tracker = AliveTracker(C, window_tokens=TOKENS_PER_BATCH)
    ci = torch.zeros(B, S, C)
    ci[0, 0, 3] = 1e-9
    tracker.observe(ci)
    assert tracker.alive[3]


def test_state_round_trips_through_a_checkpoint():
    """The counter is the run's HISTORY -- a resumed run that reset it would report every
    long-dead component as alive again for a full window.
    """
    tracker = AliveTracker(C, window_tokens=2 * TOKENS_PER_BATCH)
    tracker.observe(_ci([1]))
    for _ in range(3):
        tracker.observe(_ci([]))

    restored = AliveTracker(C)
    restored.load_state_dict(tracker.state_dict())
    assert torch.equal(restored.alive, tracker.alive)


def test_the_window_travels_with_the_state():
    """The window IS the definition of alive, so offline eval must use the run's, not the
    current default.
    """
    tracker = AliveTracker(C, window_tokens=5)
    alive = alive_from_training_state(
        {"loss_metrics": {"ComponentAliveTracker": tracker.state_dict()}}, C
    )
    restored = AliveTracker(C)
    restored.load_state_dict(tracker.state_dict())
    assert restored.window.item() == 5
    assert torch.equal(alive, tracker.alive)


def test_a_checkpoint_without_the_tracker_fails_loudly():
    """Falling back to 'everything is alive' would make metric 1 read as though no component had
    ever died -- exactly the confound the alive restriction exists to remove.
    """
    with pytest.raises(AssertionError, match="alive set is the run's firing history"):
        alive_from_training_state({"loss_metrics": {"FaithfulnessLoss": {}}}, C)


def test_eval_batches_do_not_advance_the_counter():
    """`optimize.py` re-runs every metric on the EVAL split. Counting those tokens would stretch
    the window by an amount that depends on eval cadence, and would resurrect a component on data
    the run never trained on.
    """
    from aspd.losses import ComponentAliveTracker, ComponentAliveTrackerConfig

    cfg = ComponentAliveTrackerConfig(coeff=0.0, module="m", window_tokens=100)
    metric = ComponentAliveTracker.__new__(ComponentAliveTracker)
    metric.cfg, metric.device = cfg, "cpu"
    metric.tracker = AliveTracker(C, window_tokens=100)
    metric.reset()

    class _Ctx:
        def __init__(self, is_eval):
            self.is_eval = is_eval
            self.ci = type("CI", (), {"lower_leaky": {"m": _ci([])}})()

    metric.update(_Ctx(is_eval=True))
    assert metric.tracker.tokens_since_active.sum().item() == 0.0
    metric.update(_Ctx(is_eval=False))
    assert metric.tracker.tokens_since_active.sum().item() == C * TOKENS_PER_BATCH
