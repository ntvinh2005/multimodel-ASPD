"""Which components are alive: fired at least once within the last `window_tokens` tokens."""

import torch
import torch.distributed as dist
from jaxtyping import Bool, Float
from torch import Tensor, nn


class AliveTracker(nn.Module):
    """Per-component "tokens since it last fired", and the alive mask that falls out of it."""

    def __init__(self, n_components: int, window_tokens: int = 10_000_000):
        super().__init__()
        self.register_buffer("tokens_since_active", torch.zeros(n_components))
        self.register_buffer("window", torch.tensor(float(window_tokens)))

    @torch.no_grad()
    def observe(self, ci: Float[Tensor, "b s c"]) -> None:
        """Fold one batch of causal importances into the counter."""
        assert ci.ndim == 3, f"expected [batch, seq, components], got {tuple(ci.shape)}"
        fired = (ci > 0).any(dim=0).any(dim=0).float()
        n_tokens = ci.shape[0] * ci.shape[1]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(fired, op=dist.ReduceOp.MAX)
            n_tokens *= dist.get_world_size()
        self.tokens_since_active += n_tokens
        self.tokens_since_active[fired > 0] = 0.0

    @property
    def alive(self) -> Bool[Tensor, " c"]:
        return self.tokens_since_active < self.window

    @property
    def alive_frac(self) -> Float[Tensor, ""]:
        return self.alive.float().mean()


def alive_from_training_state(
    state, n_components: int, *, metric_key: str = "ComponentAliveTracker"
) -> Bool[Tensor, " c"]:
    """Recover the alive mask from a `training_<step>.pth` blob."""
    metrics = state["loss_metrics"] if isinstance(state, dict) else state.loss_metrics
    assert metric_key in metrics, (
        f"no {metric_key!r} state in this checkpoint (has: {sorted(metrics)}). Metric 1's alive "
        "set is the run's firing history over the last window of tokens and cannot be "
        "reconstructed from weights -- the tracker must be in the config from step 0."
    )
    tracker = AliveTracker(n_components)
    tracker.load_state_dict(metrics[metric_key])
    return tracker.alive
