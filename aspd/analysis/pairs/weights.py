"""Memory-mapped access to a run's component directions u_c and v_c."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from aspd.analysis.pairs.spaces import ModuleSpaces, Side, head_dim_for_model, module_spaces

_COMPONENT_PREFIX = "_components."
_CI_PREFIX = "ci_fn._ci_fns."
_CI_ENCODER_PREFIX = "ci_fn._encoders."


@dataclass(frozen=True)
class Directions:
    """One endpoint's directions plus their norms (cached: the norms are needed on every query)."""

    mat: Tensor  # [C, d], row c is component c's direction
    norms: Tensor  # [C]


def dashed(module: str) -> str:
    """`transformer.h.6.mlp.c_fc` -> `transformer-h-6-mlp-c_fc`, the checkpoint's own key form."""
    return module.replace(".", "-")


def undashed(key: str) -> str:
    return key.replace("-", ".")


class RunWeights:
    """A run's decomposed modules and their component directions."""

    def __init__(self, ckpt_path: Path, model_name: str, *, cache_bytes: int = 6 << 30):
        self.ckpt_path = Path(ckpt_path)
        self.model_name = model_name
        self._sd: dict[str, Tensor] = torch.load(
            self.ckpt_path, map_location="cpu", mmap=True, weights_only=True
        )
        self._cache: OrderedDict[tuple[str, Side], Directions] = OrderedDict()
        self._cache_bytes = cache_bytes
        self._used = 0

        head_dim = head_dim_for_model(model_name)
        self.head_dim = head_dim
        self.spaces: dict[str, ModuleSpaces] = {}
        for key in self._sd:
            if not (key.startswith(_COMPONENT_PREFIX) and key.endswith(".V")):
                continue
            module = undashed(key[len(_COMPONENT_PREFIX) : -len(".V")])
            d_in, c_v = self._sd[key].shape
            c_u, d_out = self._sd[f"{_COMPONENT_PREFIX}{dashed(module)}.U"].shape
            assert c_v == c_u, f"{module}: V has C={c_v} but U has C={c_u}"
            self.spaces[module] = module_spaces(module, int(d_in), int(d_out), head_dim)
        assert self.spaces, f"no `{_COMPONENT_PREFIX}*` entries in {self.ckpt_path}"

    @property
    def modules(self) -> list[str]:
        return sorted(self.spaces, key=lambda m: (self.spaces[m].layer, self.spaces[m].role))

    def n_components(self, module: str) -> int:
        return int(self._sd[f"{_COMPONENT_PREFIX}{dashed(module)}.V"].shape[1])

    def dead_counter(self, module: str) -> Tensor | None:
        """`n_batches_not_active` from the CI fn, `[C]` -- the AuxK dead-latent clock, if present."""
        from aspd.sites import resid_site_for_module

        key = f"{_CI_PREFIX}{dashed(module)}.n_batches_not_active"
        if key in self._sd:
            return self._sd[key].clone()
        for candidate in (dashed(module), dashed(resid_site_for_module(module))):
            encoder_key = f"{_CI_ENCODER_PREFIX}{candidate}.n_batches_not_active"
            if encoder_key in self._sd:
                return self._sd[encoder_key].clone()
        return None

    def directions(self, module: str, side: Side) -> Directions:
        """`[C, d]` directions for one endpoint, materialised from the mmap on first use."""
        hit = self._cache.get((module, side))
        if hit is not None:
            self._cache.move_to_end((module, side))
            return hit
        base = f"{_COMPONENT_PREFIX}{dashed(module)}"
        # V is [d_in, C] and U is [C, d_out]; both become [C, d].
        raw = self._sd[f"{base}.V"].t() if side == "read" else self._sd[f"{base}.U"]
        mat = raw.to(torch.float32).contiguous()
        out = Directions(mat=mat, norms=mat.norm(dim=1))
        self._cache[(module, side)] = out
        self._used += mat.element_size() * mat.nelement()
        while self._used > self._cache_bytes and len(self._cache) > 2:
            _, evicted = self._cache.popitem(last=False)
            self._used -= evicted.mat.element_size() * evicted.mat.nelement()
        return out
