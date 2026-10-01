"""Pair endpoints: decomposed matrices and pretrained SAEs, and how two endpoints link."""

from dataclasses import dataclass
from typing import Literal

from aspd.analysis.pairs import features as ft
from aspd.analysis.pairs.sae_config import SITES
from aspd.analysis.pairs.spaces import Side, Space

Kind = Literal["module", "sae"]

STREAM_ORDER = {"resid_pre": 0, "attn_out": 1, "resid_mid": 2, "mlp_out": 3, "resid_post": 4}

ROLE_POSITION: dict[tuple[str, Side], str] = {
    ("attn.q", "read"): "resid_pre",
    ("attn.k", "read"): "resid_pre",
    ("attn.v", "read"): "resid_pre",
    ("attn.v", "write"): "attn_z",
    ("attn.o", "read"): "attn_z",
    ("attn.o", "write"): "attn_out",
    ("mlp.in", "read"): "resid_mid",
    ("mlp.gate", "read"): "resid_mid",
    ("mlp.up", "read"): "resid_mid",
    ("mlp.out", "write"): "mlp_out",
}


@dataclass(frozen=True)
class Endpoint:
    """A `[C, d]` block of directions plus everything needed to say what it means."""

    kind: Kind
    key: str
    side: Side
    role: str  # module role, or `sae.<site>`
    layer: int
    space: Space
    n_components: int
    label: str
    centred: bool
    position: str | None  # the point in the block it occupies, if it is on the stream

    @property
    def is_sae(self) -> bool:
        return self.kind == "sae"


def module_endpoint(module: str, side: Side, spec, n_components: int) -> Endpoint:
    return Endpoint(
        kind="module",
        key=module,
        side=side,
        role=spec.role,
        layer=spec.layer,
        space=spec.space(side),
        n_components=n_components,
        label=f"{module} · {side}",
        centred=False,
        position=ROLE_POSITION.get((spec.role, side)),
    )


def sae_endpoint(ep: ft.SaeEndpoint, side: Side) -> Endpoint:
    return Endpoint(
        kind="sae",
        key=ep.key,
        side=side,
        role=f"sae.{ep.site}",
        layer=ep.layer,
        space=ep.space,
        n_components=ep.d_sae,
        label=f"{ep.label} · {'encoder' if side == 'read' else 'decoder'}",
        centred=ep.centred,
        position=ep.site if ep.site in STREAM_ORDER else None,
    )


def reads_through_layernorm(e: Endpoint, model_name: str) -> ft.NormSpec | None:
    """The LayerNorm between this endpoint's direction and the residual stream, if any."""
    if e.is_sae or e.side != "read":
        return None
    return ft.norms_for_model(model_name).get(e.role)


def sae_link(a: Endpoint, b: Endpoint, model_name: str) -> str | None:
    """What separates two endpoints when at least one of them is a dictionary."""
    if a.space.key != b.space.key and not (
        a.position in STREAM_ORDER and b.position in STREAM_ORDER
    ):
        return None
    ln = reads_through_layernorm(a, model_name) or reads_through_layernorm(b, model_name)
    if a.position == "attn_z" or b.position == "attn_z":
        # The z space is head-structured and layer-local; equal keys already mean the same point.
        return "same_point" if a.space.key == b.space.key else None
    if a.position not in STREAM_ORDER or b.position not in STREAM_ORDER:
        return None
    if ln is not None:
        return "layernorm"
    if (a.layer, STREAM_ORDER[a.position]) == (b.layer, STREAM_ORDER[b.position]):
        return "same_point"
    return "residual"


@dataclass(frozen=True)
class SaeTemplate:
    """A suggested pairing where at least one endpoint is a dictionary."""

    key: str
    label: str
    a: str
    a_side: Side
    b: str
    b_side: Side
    link: str
    layer_mode: str  # "same" | "cross"


WEIGHT_FEATURE_TEMPLATES: list[SaeTemplate] = [
    SaeTemplate("wf_mlp_out", "MLP out components -> MLP-out features",
                "mlp.out", "write", "sae:mlp_out", "read", "same_point", "same"),
    SaeTemplate("wf_attn_out", "Attention out components -> attn-out features",
                "attn.o", "write", "sae:attn_out", "read", "same_point", "same"),
    SaeTemplate("wf_attn_v", "Value components -> attention-z features",
                "attn.v", "write", "sae:attn_z", "read", "same_point", "same"),
    SaeTemplate("wf_attn_o_read", "Output-projection reads -> attention-z features",
                "attn.o", "read", "sae:attn_z", "read", "same_point", "same"),
    SaeTemplate("wf_mlp_in_read", "MLP in reads -> resid-mid features",
                "mlp.in", "read", "sae:resid_mid", "read", "layernorm", "same"),
    SaeTemplate("wf_mlp_gate_read", "MLP gate reads -> resid-mid features",
                "mlp.gate", "read", "sae:resid_mid", "read", "layernorm", "same"),
    SaeTemplate("wf_attn_q_read", "Query reads -> resid-pre features",
                "attn.q", "read", "sae:resid_pre", "read", "layernorm", "same"),
]

FEATURE_FEATURE_TEMPLATES: list[SaeTemplate] = [
    SaeTemplate("ff_mlp_to_resid", "MLP-out features -> resid-post features",
                "sae:mlp_out", "write", "sae:resid_post", "read", "residual", "same"),
    SaeTemplate("ff_attn_to_resid", "Attn-out features -> resid-post features",
                "sae:attn_out", "write", "sae:resid_post", "read", "residual", "same"),
    SaeTemplate("ff_resid_to_resid", "Resid features -> later resid features",
                "sae:resid_pre", "write", "sae:resid_pre", "read", "residual", "cross"),
]


def _site_of(name: str) -> str | None:
    return name[4:] if name.startswith("sae:") else None


def available_sae_templates(
    module_spaces: dict, model_name: str, sae_catalogue: list[dict]
) -> list[dict[str, object]]:
    """Templates this run can instantiate, each with the layers each endpoint exists on."""
    from aspd.analysis.pairs.suggest import LINKS

    role_layers: dict[str, list[int]] = {}
    for s in module_spaces.values():
        role_layers.setdefault(s.role, []).append(s.layer)
    site_layers: dict[str, list[int]] = {}
    for entry in sae_catalogue:
        site_layers.setdefault(str(entry["site"]), []).extend(entry["layers"])  # pyright: ignore[reportArgumentType]

    def layers_of(name: str) -> list[int]:
        site = _site_of(name)
        return sorted(set(site_layers.get(site, []) if site else role_layers.get(name, [])))

    out: list[dict[str, object]] = []
    for t in WEIGHT_FEATURE_TEMPLATES + FEATURE_FEATURE_TEMPLATES:
        la, lb = layers_of(t.a), layers_of(t.b)
        if not la or not lb:
            continue
        if t.layer_mode == "same":
            shared = sorted(set(la) & set(lb))
            if not shared:
                continue
            layers: dict[str, list[int]] = {"same": shared}
        else:
            layers = {"a": la, "b": lb}
        out.append(
            {
                "key": t.key,
                "label": t.label,
                "a": t.a, "a_side": t.a_side,
                "b": t.b, "b_side": t.b_side,
                "a_site": _site_of(t.a), "b_site": _site_of(t.b),
                "link": t.link,
                "link_label": LINKS[t.link]["label"],
                "caveat": LINKS[t.link]["caveat"],
                "layer_mode": t.layer_mode,
                "layers": layers,
                "mode": "feature" if _site_of(t.a) and _site_of(t.b) else "weight_feature",
            }
        )
    return out


def site_label(site: str) -> str:
    return SITES[site].label


@dataclass(frozen=True)
class Reconciled:
    """Two endpoints' directions brought into one basis, and the record of what that took."""

    a: ft.Directions
    b: ft.Directions
    d_eff_drop: int  # 1 once centred: the all-ones direction is gone from the space
    applied: list[str]


def reconcile(
    a: Endpoint,
    b: Endpoint,
    da: ft.Directions,
    db: ft.Directions,
    *,
    model_name: str,
    norms: ft.TargetNorms,
) -> Reconciled:
    """Fold the LayerNorm and reconcile the residual basis, or do nothing at all."""
    if not (a.is_sae or b.is_sae):
        return Reconciled(da, db, 0, [])

    applied: list[str] = []
    mats = {}
    for name, e, d in (("a", a, da), ("b", b, db)):
        spec = reads_through_layernorm(e, model_name)
        if spec is None:
            mats[name] = d.mat
            continue
        gain = norms.gain(spec, e.layer)
        mats[name] = ft.fold_layernorm(d.mat, gain, centres=spec.centres)
        applied.append(
            f"{name}: folded {spec.key.format(layer=e.layer)}"
            + (" and centred (LayerNorm)" if spec.centres else " (RMSNorm, no centring)")
        )

    drop = 0
    if a.centred or b.centred:
        mats = {k: ft.centre(v) for k, v in mats.items()}
        drop = 1
        who = a.key if a.centred else b.key
        applied.append(f"centred both sides: {who} was fit on mean-centred (TransformerLens) activations")

    out = {k: ft.Directions(mat=v, norms=v.norm(dim=1)) for k, v in mats.items()}
    return Reconciled(out["a"], out["b"], drop, applied)
