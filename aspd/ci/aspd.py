"""ASPD's causal-importance function: the shared sparse encoder g^s.

g^s reads the residual stream r_t (resid-pre for Q/K/V, resid-mid for O and the MLP matrices)
through a forward hook on the target model, computes relu((r_t - b_dec) W_enc) and keeps the batch
top-k (k per token on average); component c is active iff feature c is selected,
g_{t,c} = 1[g^s_{t,c}(R) > 0]. Its decoder W_dec reconstructs r_t (L_act). Matrices reading the
same site share one encoder when `share_encoders` is set, so component c is the same feature across
them.
"""

from typing import Literal

import torch
from jaxtyping import Float
from param_decomp.components import get_module_input_dim
from pydantic import PositiveInt
from torch import Tensor, nn
from torch.autograd import forward_ad
from torch.utils.hooks import RemovableHandle

from aspd.capture_guard import capture_armed
from aspd.ci.pd_transcoder import PDTranscoderCiConfig, PDTranscoderCiFn


class ASPDCiConfig(PDTranscoderCiConfig):
    """PD Transcoder's BatchTopK gate, computed by a separate encoder on the residual stream."""

    mode: Literal["aspd"] = "aspd"  # pyright: ignore[reportIncompatibleVariableOverride]
    encoder: Literal["untied"] = "untied"  # pyright: ignore[reportIncompatibleVariableOverride]

    resid_site: str = ""
    """Module whose input is the residual stream the encoder reads (single decomposed matrix)."""
    resid_sites: dict[str, str] | None = None
    """`{decomposed module: residual site}` when several matrices are decomposed."""
    d_act: PositiveInt | None = None
    """Width of the residual stream (d_model). Required."""
    share_encoders: bool = True
    """Matrices reading the same site share one encoder (same feature index across them)."""
    encoder_init: Literal["reference", "unit_norm"] = "reference"
    """How W_enc is drawn at construction (seeding redraws it unit-norm before training)."""


class SharedEncoder(PDTranscoderCiFn):
    """g^s on one residual stream: W_enc [d_act, C], b_dec [d_act], W_dec [C, d_act]."""

    allowed_encoders = ("untied",)
    # g_c = 1 constrains the encoder's feature, not the component activation v_c^T x_t, whose sign
    # is free.
    gate_implies_positive_preact: bool = False

    def __init__(self, site: str, n_components: int, d_act: int, cfg: ASPDCiConfig):
        super().__init__(module=site, n_components=n_components, d_in=d_act, cfg=cfg)
        assert site, "the shared encoder needs the module whose input is its residual stream"
        self.cfg: ASPDCiConfig = cfg
        self.site = site
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_act, n_components)))
        self.b_dec = nn.Parameter(torch.zeros(d_act))
        if cfg.encoder_init == "unit_norm":
            with torch.no_grad():
                self.W_enc.div_(self.W_enc.norm(dim=0, keepdim=True).clamp_min(1e-8))
        self.W_dec = nn.Parameter(torch.empty(n_components, d_act))
        self.tie_decoder()
        self._resid: Tensor | None = None
        self._resid_handle: RemovableHandle | None = None

    @torch.no_grad()
    def tie_decoder(self) -> None:
        """W_dec <- W_enc^T with unit-norm rows."""
        self.W_dec.copy_(self.W_enc.t())
        self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    def attach_resid_site(self, target_model: nn.Module) -> "SharedEncoder":
        """Register the forward hook that captures the input of `site` on every target forward."""
        assert self._resid_handle is None, f"residual site {self.site!r} is already attached"
        module = target_model.get_submodule(self.site)

        def hook(_module: nn.Module, args: tuple, _output: object) -> None:
            if not capture_armed():
                return
            assert args and isinstance(args[0], Tensor), f"no tensor input at {self.site!r}"
            r = args[0]
            assert r.shape[-1] == self.d_in, (
                f"residual site {self.site!r} has width {r.shape[-1]}, config says d_act={self.d_in}"
            )
            self._resid = r.detach()

        self._resid_handle = module.register_forward_hook(hook)
        return self

    def detach_resid_site(self) -> "SharedEncoder":
        """Remove the hook. Idempotent."""
        if self._resid_handle is not None:
            self._resid_handle.remove()
        self._resid_handle = None
        self._resid = None
        return self

    def _captured_resid(self, reference: Float[Tensor, "... d_x"]) -> Float[Tensor, "... d_act"]:
        """The captured r_t, checked to be on the same token grid as `reference`."""
        assert forward_ad.unpack_dual(reference).tangent is None, (
            f"the gate does not depend on the activation at {self.module!r} (it reads {self.site!r}), "
            "so derivatives with respect to it are not available"
        )
        assert self._resid is not None, (
            f"nothing captured at {self.site!r}: a target forward must run before the CI function"
        )
        assert self._resid.shape[:-1] == reference.shape[:-1], (
            f"captured {tuple(self._resid.shape)} at {self.site!r} but was handed "
            f"{tuple(reference.shape)} for {self.module!r}"
        )
        return self._resid

    def resid_site_acts(self, reference: Float[Tensor, "... d_x"]) -> Float[Tensor, "... d_act"]:
        """r_t, the target of L_act."""
        return self._captured_resid(reference)

    def preacts(
        self, x: Float[Tensor, "... d_x"], z: Float[Tensor, "... f"] | None = None
    ) -> Float[Tensor, "... f"]:
        """relu((r_t - b_dec) W_enc); `x` fixes the token grid only."""
        del z
        r = self._captured_resid(x)
        return torch.relu((r - self.b_dec) @ self.W_enc)


class ASPDCiFn(SharedEncoder):
    """The shared encoder for a single decomposed matrix."""

    def __init__(self, module: str, n_components: int, d_x: int, d_act: int, cfg: ASPDCiConfig):
        site = cfg.resid_site or (cfg.resid_sites or {}).get(module, "")
        assert site, f"no residual site for {module!r}: set `resid_site` or `resid_sites`"
        super().__init__(site=site, n_components=n_components, d_act=d_act, cfg=cfg)
        self.module = module
        self.d_x = d_x

    def preacts(
        self, x: Float[Tensor, "... d_x"], z: Float[Tensor, "... f"] | None = None
    ) -> Float[Tensor, "... f"]:
        assert x.shape[-1] == self.d_x, (
            f"{self.module!r} was handed width {x.shape[-1]}, its components read {self.d_x}"
        )
        return super().preacts(x, z)


def encoder_key_map(sites: dict[str, str], share_encoders: bool) -> dict[str, str]:
    """`{decomposed module: key of its encoder}`: the site when shared, the module otherwise."""
    return {m: (site if share_encoders else m) for m, site in sites.items()}


class ASPDCiFnSet(nn.Module):
    """Shared encoders for several decomposed matrices (the model-wide decomposition)."""

    def __init__(
        self,
        encoders: dict[str, SharedEncoder],
        module_encoders: dict[str, str],
        d_x: dict[str, int],
    ):
        super().__init__()
        assert encoders and module_encoders, "empty encoder set"
        assert set(module_encoders.values()) == set(encoders), "module->encoder map and encoders disagree"
        assert set(module_encoders) == set(d_x), "module->encoder map and module->d_x disagree"
        self.module_names = sorted(module_encoders)
        self.encoder_names = sorted(encoders)
        self._encoders = nn.ModuleDict({s.replace(".", "-"): encoders[s] for s in self.encoder_names})
        self._module_encoders = dict(module_encoders)
        self._d_x = dict(d_x)
        self.cfg = encoders[self.encoder_names[0]].cfg
        self._tied: list[dict[str, nn.Module]] = []

    def encoders(self) -> dict[str, SharedEncoder]:
        return {s: self._encoders[s.replace(".", "-")] for s in self.encoder_names}

    def encoder_for(self, module: str) -> SharedEncoder:
        assert module in self._module_encoders, f"no encoder for {module!r}"
        return self._encoders[self._module_encoders[module].replace(".", "-")]

    def fns(self) -> dict[str, SharedEncoder]:
        """`{module: its encoder}`; several modules map to one object when shared."""
        return {m: self.encoder_for(m) for m in self.module_names}

    @property
    def shared(self) -> bool:
        return len(self.encoder_names) < len(self.module_names)

    def sites(self) -> dict[str, str]:
        """`{encoder key: residual site it reads}`."""
        return {key: encoder.site for key, encoder in self.encoders().items()}

    def modules_by_encoder(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {s: [] for s in self.encoder_names}
        for module in self.module_names:
            out[self._module_encoders[module]].append(module)
        return out

    def attach_components(self, components: dict[str, nn.Module]) -> "ASPDCiFnSet":
        """Check every matrix's V against its encoder's feature axis (component c = feature c)."""
        assert set(components) == set(self.module_names), "CI function and components disagree"
        for module in self.module_names:
            comp = components[module]
            assert hasattr(comp, "b_dec"), f"{module!r} needs `component_arch: transcoder`"
            expected = (self._d_x[module], self.encoder_for(module).n_features)
            assert tuple(comp.V.shape) == expected, (
                f"{module!r}: V is {tuple(comp.V.shape)}, expected {expected}"
            )
        self._tied = [dict(components)]
        return self

    @property
    def components(self) -> dict[str, nn.Module]:
        assert self._tied, "call `aspd.ci.aspd_setup.tie_aspd_ci_fn` after building the Trainer"
        return self._tied[0]

    def detach_resid_site(self) -> "ASPDCiFnSet":
        for encoder in self.encoders().values():
            encoder.detach_resid_site()
        return self

    def forward(
        self, input_acts: dict[str, Float[Tensor, "... d_in"]]
    ) -> dict[str, Float[Tensor, "... c"]]:
        """Gates for every module in `input_acts`, computing each encoder's gate once."""
        unknown = set(input_acts) - set(self.module_names)
        assert not unknown, f"no encoder for {sorted(unknown)}"
        wanted: dict[str, list[str]] = {}
        for module in sorted(input_acts):
            wanted.setdefault(self._module_encoders[module], []).append(module)
        out: dict[str, Float[Tensor, "... c"]] = {}
        for key, modules in wanted.items():
            encoder = self._encoders[key.replace(".", "-")]
            head = modules[0]
            assert input_acts[head].shape[-1] == self._d_x[head], (
                f"{head!r} was handed width {input_acts[head].shape[-1]}, expected {self._d_x[head]}"
            )
            gate = encoder.gate(input_acts[head])
            for module in modules:
                out[module] = gate
        return out


def make_aspd_ci_fn(
    *, target_model: nn.Module, module_to_c: dict[str, int], ci_config: ASPDCiConfig
) -> ASPDCiFn | ASPDCiFnSet:
    """Build ASPD's CI function (one encoder, or a set of them) and register the residual hooks."""
    assert module_to_c, "no decomposition targets"
    assert ci_config.d_act is not None, "ASPDCiConfig.d_act (the residual stream width) is required"
    d_act = ci_config.d_act

    if len(module_to_c) == 1:
        module, n_components = next(iter(module_to_c.items()))
        return ASPDCiFn(
            module=module,
            n_components=n_components,
            d_x=get_module_input_dim(target_model.get_submodule(module)),
            d_act=d_act,
            cfg=ci_config,
        ).attach_resid_site(target_model)

    sites = ci_config.resid_sites
    assert sites is not None, "several decomposition targets need `resid_sites: {module: site}`"
    assert set(sites) == set(module_to_c), "`resid_sites` must cover exactly the decomposition targets"

    module_encoders = encoder_key_map({m: sites[m] for m in sorted(module_to_c)}, ci_config.share_encoders)
    by_key: dict[str, list[str]] = {}
    for module in sorted(module_to_c):
        by_key.setdefault(module_encoders[module], []).append(module)

    encoders: dict[str, SharedEncoder] = {}
    for key, modules in by_key.items():
        widths = {module_to_c[m] for m in modules}
        assert len(widths) == 1, f"matrices sharing encoder {key!r} declare different C: {sorted(widths)}"
        encoders[key] = SharedEncoder(
            site=sites[modules[0]], n_components=widths.pop(), d_act=d_act, cfg=ci_config
        ).attach_resid_site(target_model)

    return ASPDCiFnSet(
        encoders=encoders,
        module_encoders=module_encoders,
        d_x={m: get_module_input_dim(target_model.get_submodule(m)) for m in module_to_c},
    )
