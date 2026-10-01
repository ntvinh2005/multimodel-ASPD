"""Ensures ASPD's shared encoder reads the residual stream of the frozen target model.

The encoder's input is captured by a forward hook; this guard disables the hook during the
component model's own (masked) forwards so only the target's clean residual stream is recorded.
"""

from collections.abc import Generator
from contextlib import contextmanager

_ARMED = True
"""Whether a routed CI fn's gate-site hook should record what it sees.

Process-global rather than per-CI-fn: what it tracks is a property of the forward in flight, and a
whole-model run has 24 encoders all firing inside that one forward. Not thread-local -- training,
eval and every offline tool here run one forward at a time in one thread, and a per-thread flag
would silently arm a background thread's capture rather than failing.
"""


def capture_armed() -> bool:
    """Whether the forward in flight is the frozen target's. Read by every gate-site hook."""
    return _ARMED


@contextmanager
def disarmed() -> Generator[None]:
    """Suppress gate-site capture for the duration. Nests; restores rather than re-arming."""
    global _ARMED
    previous = _ARMED
    _ARMED = False
    try:
        yield
    finally:
        _ARMED = previous


def install_clean_capture_guard() -> None:
    """Disarm capture inside masked `ComponentModel.forward`s. Idempotent; call unconditionally."""
    from param_decomp.component_model import ComponentModel

    if getattr(ComponentModel.forward, "_fs_capture_guarded", False):
        return
    stock = ComponentModel.forward

    def _forward(self, batch, mask_infos=None, cache_type="none"):
        if mask_infos is None:
            return stock(self, batch, mask_infos=mask_infos, cache_type=cache_type)
        with disarmed():
            return stock(self, batch, mask_infos=mask_infos, cache_type=cache_type)

    _forward._fs_capture_guarded = True  # pyright: ignore[reportFunctionMemberAccess]
    ComponentModel.forward = _forward  # pyright: ignore[reportAttributeAccessIssue]
