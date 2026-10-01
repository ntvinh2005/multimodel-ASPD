"""The weight edit W' = W - sum_c u_c v_c^T, and the random-component control."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from param_decomp.components import LinearComponents
from torch import Tensor, nn
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from aspd.eval.editing.attribution import is_rank_one


def component_delta_weight(
    components: LinearComponents, selection: Int[Tensor, " s"]
) -> Float[Tensor, "d_out d_in"]:
    """`dW = sum_{c in S} U_c (x) V_c`, in PD's `[d_out, d_in]` convention."""
    assert is_rank_one(components), (
        f"the rank-1 edit is defined for components whose weight is `U_c (x) V_c`; got "
        f"{type(components).__name__}, which is not one. Attribution still runs on such an arm; "
        "the edit does not"
    )
    dtype = torch.promote_types(components.U.dtype, torch.float32)
    u = components.U.detach()[selection].to(dtype)  # [s, d_out]
    v = components.V.detach()[:, selection].to(dtype)  # [d_in, s]
    return u.t() @ v.t()


def random_selection(n_components: int, k: int, generator: torch.Generator) -> Tensor:
    """`k` distinct component indices, uniform without replacement, on the CPU generator."""
    assert 0 < k <= n_components, (k, n_components)
    return torch.randperm(n_components, generator=generator)[:k].sort().values


@contextmanager
def patched_target_weight(
    model: nn.Module, module_path: str, delta: Float[Tensor, "d_out d_in"]
):
    """Subtract `delta` from the real module's weight for the duration of the block."""
    module = model.get_submodule(module_path)
    match module:
        case RadfordConv1D():
            update = delta.t()
        case nn.Linear():
            update = delta
        case _:
            raise AssertionError(
                f"{module_path} is a {type(module).__name__}; the edit only knows how to write "
                "into nn.Linear and Radford Conv1D weights"
            )
    weight = module.weight
    assert update.shape == weight.shape, (
        f"edit is {tuple(update.shape)} but {module_path}.weight is {tuple(weight.shape)}"
    )
    original = weight.detach().clone()
    try:
        with torch.no_grad():
            weight.sub_(update.to(weight.dtype))
        yield
    finally:
        with torch.no_grad():
            weight.copy_(original)


@dataclass(frozen=True)
class EditSpec:

    kind: str
    """`ranked`, `random`, or `norm_matched`."""
    k: int
    feature_id: int | None
    """The feature whose ranking produced it. `None` for the shared controls."""
    rep: int = 0
    """Which draw, for the controls. Always 0 for `ranked`."""

    @property
    def name(self) -> str:
        target = "shared" if self.feature_id is None else f"f{self.feature_id}"
        rep = "" if self.kind == "ranked" else f"_r{self.rep}"
        return f"{self.kind}_k{self.k}_{target}{rep}"


def norm_matched_delta(
    delta: Float[Tensor, "d_out d_in"], target_norm: float
) -> Float[Tensor, "d_out d_in"]:
    """`delta` rescaled to `||delta||_F == target_norm`."""
    norm = float(delta.norm())
    assert norm > 0, "random selection produced a zero edit; every component is dead"
    return delta * (target_norm / norm)
