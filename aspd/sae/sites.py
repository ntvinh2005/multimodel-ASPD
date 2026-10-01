"""Hook sites for SAE training: the activations entering and leaving the decomposed matrix."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HookSite:
    """ONE extraction point: which module to hook, and whether to take its input or its output."""

    key: str
    module: str
    take: Literal["input", "output"] = "output"

    def resolve(self, model: nn.Module) -> nn.Module:
        """Fail fast at setup rather than mid-run if the path does not exist on this model."""
        return model.get_submodule(self.module)


class SiteCapture:
    """`HookSite` capture -- the input-or-output generalization of `OutputCapture`, keyed by `key`."""

    def __init__(self, model: nn.Module, sites: list[HookSite], *, detach: bool = False):
        keys = [s.key for s in sites]
        assert len(set(keys)) == len(keys), f"duplicate site key in {keys}"
        self.model = model
        self.sites = sites
        self.detach = detach
        self.acts: dict[str, Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _store(self, key: str, tensor: object) -> None:
        assert isinstance(tensor, Tensor), (
            f"site {key!r} captured {type(tensor).__name__}, expected Tensor -- adjacency "
            "requires hooking the projection itself, not a block that returns a tuple"
        )
        self.acts[key] = tensor.detach() if self.detach else tensor

    def _output_hook(self, key: str):
        def hook(_module: nn.Module, _args: tuple, output: object) -> None:
            self._store(key, output)

        return hook

    def _input_hook(self, key: str):
        def hook(_module: nn.Module, args: tuple) -> None:
            assert args, f"site {key!r}: module called with no positional args, nothing to capture"
            self._store(key, args[0])

        return hook

    def __enter__(self) -> "SiteCapture":
        for site in self.sites:
            module = site.resolve(self.model)
            handle = (
                module.register_forward_pre_hook(self._input_hook(site.key))
                if site.take == "input"
                else module.register_forward_hook(self._output_hook(site.key))
            )
            self._handles.append(handle)
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __getitem__(self, key: str) -> Tensor:
        assert key in self.acts, f"no activation captured for site {key!r}; did the forward run?"
        return self.acts[key]

    def require_all(self) -> dict[str, Tensor]:
        missing = [s.key for s in self.sites if s.key not in self.acts]
        assert not missing, (
            f"forward completed without firing {missing} -- the module is dead code on this model, "
            "or sits behind a branch this input did not take. The captured set is incomplete."
        )
        return dict(self.acts)


@dataclass(frozen=True)
class SitePair:
    """The two activations bracketing one decomposed matrix, each named by module path."""

    module: str  # the decomposed matrix, e.g. "transformer.h.0.mlp.c_fc"
    input_site: str
    output_site: str  # == module; named separately so the asymmetry stays visible
    input_take: Literal["input", "output"] = "output"
    input_module: str | None = None  # module to hook for `x`; None => `input_site` is the path

    def __post_init__(self) -> None:
        assert self.output_site == self.module, (
            f"output site must be the decomposed module itself (adjacency): "
            f"{self.output_site!r} != {self.module!r}"
        )
        assert self.input_site != self.output_site, (
            f"input and output site are both {self.input_site!r}; this class keys its captures, "
            "its saved files and its report by module path, so the two would overwrite each "
            "other. Give the input site a distinct KEY and name the module in `input_module`."
        )
        assert self.input_module is None or self.input_take == "input", (
            f"input_module={self.input_module!r} with input_take='output': the only reason to "
            "name the hooked module separately is that the key is not a module path, and that "
            "only arises when `x` is captured on the way INTO a module. An OUTPUT take is "
            "addressable by its own path, so use it as the key."
        )

    @property
    def takes(self) -> dict[str, Literal["input", "output"]]:
        """Per-key capture mode, in the form `OutputCapture` wants."""
        return {self.input_site: self.input_take, self.output_site: "output"}

    @property
    def capture_modules(self) -> dict[str, str]:
        """Per-key module to hook, in the form `OutputCapture` wants. Identity but for `down_proj`."""
        return {
            self.input_site: self.input_module or self.input_site,
            self.output_site: self.output_site,
        }


def gpt2_mlp_c_fc(layer: int) -> SitePair:
    """GPT2 `c_fc`: reads post-`ln_2` resid, writes the pre-GELU MLP hidden."""
    return SitePair(
        module=f"transformer.h.{layer}.mlp.c_fc",
        input_site=f"transformer.h.{layer}.ln_2",
        output_site=f"transformer.h.{layer}.mlp.c_fc",
    )


def gpt2_mlp_c_fc_resid(layer: int) -> SitePair:
    """GPT2 `c_fc` with the input extractor on the RESIDUAL STREAM -- `ln_2`'s input, not output."""
    return SitePair(
        module=f"transformer.h.{layer}.mlp.c_fc",
        input_site=f"transformer.h.{layer}.ln_2",
        output_site=f"transformer.h.{layer}.mlp.c_fc",
        input_take="input",
    )


def llama_simple_mlp_c_fc(layer: int) -> SitePair:
    return SitePair(
        module=f"h.{layer}.mlp.c_fc",
        input_site=f"h.{layer}.rms_2",
        output_site=f"h.{layer}.mlp.c_fc",
    )


def gemma2_mlp_down_proj(layer: int) -> SitePair:
    """Gemma-2 `down_proj`: reads the gated MLP hidden `act(gate(h))*up(h)`, writes the MLP output."""
    module = f"model.layers.{layer}.mlp.down_proj"
    return SitePair(
        module=module,
        input_site=f"{module}.in",
        output_site=module,
        input_take="input",
        input_module=module,
    )


def attn_o_proj(layer: int) -> SitePair:
    """Attention OUTPUT matrix on a `model.layers.N` block: reads `z`, writes the residual delta."""
    module = f"model.layers.{layer}.self_attn.o_proj"
    return SitePair(
        module=module,
        input_site=f"{module}.in",
        output_site=module,
        input_take="input",
        input_module=module,
    )


def resolve_sites(model: nn.Module, sites: SitePair) -> None:
    """Fail fast at setup rather than mid-run if a path does not exist on this model."""
    for path in (sites.module, *sites.capture_modules.values()):
        model.get_submodule(path)


class _CaptureComplete(Exception):
    """Internal sentinel: every requested site has fired, so the rest of the forward is dead work."""


class OutputCapture:
    """Forward hooks capturing named modules' outputs. Used by SAE TRAINING and width probing."""

    def __init__(
        self,
        model: nn.Module,
        paths: list[str],
        *,
        detach: bool = False,
        stop_when_complete: bool = False,
        takes: dict[str, Literal["input", "output"]] | None = None,
        modules: dict[str, str] | None = None,
    ):
        self.model = model
        self.paths = paths
        self.detach = detach
        self.stop_when_complete = stop_when_complete
        self.takes: dict[str, Literal["input", "output"]] = takes or {}
        self.modules: dict[str, str] = modules or {}
        unknown = set(self.takes) - set(paths)
        assert not unknown, f"takes names paths that are not captured: {sorted(unknown)}"
        unknown_modules = set(self.modules) - set(paths)
        assert not unknown_modules, (
            f"modules names keys that are not captured: {sorted(unknown_modules)}"
        )
        self.acts: dict[str, Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, path: str):
        take = self.takes.get(path, "output")

        def hook(_module: nn.Module, args: tuple, output: object) -> None:
            if take == "input":
                assert args, (
                    f"hook site {path!r} was called with no positional args, so there is no input "
                    "tensor to capture"
                )
                captured = args[0]
            else:
                captured = output
            assert isinstance(captured, Tensor), (
                f"hook site {path!r} ({take}) is {type(captured).__name__}, expected Tensor -- "
                "adjacency requires hooking the projection/norm itself, not a block"
            )
            self.acts[path] = captured.detach() if self.detach else captured
            if self.stop_when_complete and len(self.acts) == len(self.paths):
                raise _CaptureComplete

        return hook

    def run(self, *args: object, **kwargs: object) -> None:
        """Run the model only as far as the last requested site, then unwind."""
        self.acts.clear()
        try:
            self.model(*args, **kwargs)
        except _CaptureComplete:
            pass
        missing = [p for p in self.paths if p not in self.acts]
        assert not missing, (
            f"forward completed without firing {missing}; the model ran to the end and these "
            "sites were never reached. Check the SitePair against this model's module tree."
        )

    def __enter__(self) -> "OutputCapture":
        for path in self.paths:
            module = self.model.get_submodule(self.modules.get(path, path))
            self._handles.append(module.register_forward_hook(self._make_hook(path)))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __getitem__(self, path: str) -> Tensor:
        assert path in self.acts, f"no activation captured for {path!r}; did the forward run?"
        return self.acts[path]


def site_widths(model: nn.Module, sites: SitePair, probe: Tensor) -> dict[str, int]:
    """`d_in`/`d_out` read off a real forward -- never inferred from config field names."""
    paths = [sites.input_site, sites.output_site]
    with torch.no_grad(), OutputCapture(
        model,
        paths,
        detach=True,
        stop_when_complete=True,
        takes=sites.takes,
        modules=sites.capture_modules,
    ) as cap:
        cap.run(probe)
        return {
            sites.input_site: cap[sites.input_site].shape[-1],
            sites.output_site: cap[sites.output_site].shape[-1],
        }


def iter_token_batches(loader: Iterator[object]) -> Iterator[Tensor]:
    """Normalize a param_decomp LM loader's yield into bare token-id tensors."""
    for batch in loader:
        if isinstance(batch, Tensor):
            yield batch
        elif isinstance(batch, dict):
            yield batch["input_ids"]
        else:
            yield batch[0]
