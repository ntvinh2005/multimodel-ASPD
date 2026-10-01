"""The module pairings the viewer offers (OV, QK, MLP, cross-layer residual)."""

from dataclasses import dataclass

from aspd.analysis.pairs.spaces import ModuleSpaces, Side

LINKS: dict[str, dict[str, str]] = {
    "identity": {
        "label": "coordinate-wise identity",
        "caveat": "Attention mixes positions, not coordinates, so the inner product is the exact "
        "composition strength up to the (positive) attention weights.",
    },
    "pointwise": {
        "label": "through a pointwise nonlinearity",
        "caveat": "A writes pre-activation and B reads post-activation. The inner product is the "
        "composition only in the linear regime of the activation; the Jacobian-weighted metric is "
        "the first-order correction.",
    },
    "bilinear": {
        "label": "through a gated (bilinear) MLP",
        "caveat": "B reads phi(gate(x)) * up(x). One branch alone does not determine what B sees, "
        "so this score is a marginal, not the composition.",
    },
    "bilinear_form": {
        "label": "the QK bilinear form",
        "caveat": "The attention logit is a PER-HEAD dot product. A flat score over all coordinates "
        "sums the heads' logits, which is not a quantity the model forms -- read the per-head score.",
    },
    "residual": {
        "label": "across the residual stream",
        "caveat": "A LayerNorm and every intervening block sit between the two endpoints. The score "
        "ignores the LayerNorm's centering and per-token scaling, and ignores whether the two "
        "components are ever active at compatible positions.",
    },
    "same_point": {
        "label": "the same point in the stream",
        "caveat": "Both endpoints are directions at the SAME activation site, so the inner product "
        "is exact. It is still geometry: it says nothing about whether the component and the "
        "feature are ever active on the same token.",
    },
    "layernorm": {
        "label": "through a LayerNorm",
        "caveat": "The component reads after the block's normalisation. Its direction is folded "
        "through the norm's gain (and centred, on LayerNorm) so the two live in one space; the "
        "per-token 1/sigma(x) and the norm's bias term are not directions and are ignored.",
    },
    "shared_input": {
        "label": "shared input direction",
        "caveat": "Both endpoints READ this space; the score asks whether they respond to the same "
        "input, not whether one feeds the other.",
    },
}


@dataclass(frozen=True)
class Template:
    key: str
    label: str
    a_role: str
    a_side: Side
    b_role: str
    b_side: Side
    link: str
    layer_mode: str  # "same" | "cross"


TEMPLATES: list[Template] = [
    Template("mlp_in_out", "MLP in -> out (through the activation)",
             "mlp.in", "write", "mlp.out", "read", "pointwise", "same"),
    Template("mlp_gate_down", "MLP gate -> down",
             "mlp.gate", "write", "mlp.out", "read", "bilinear", "same"),
    Template("mlp_up_down", "MLP up -> down",
             "mlp.up", "write", "mlp.out", "read", "bilinear", "same"),
    Template("attn_ov", "Attention OV (value -> output projection)",
             "attn.v", "write", "attn.o", "read", "identity", "same"),
    Template("attn_qk", "Attention QK (query . key)",
             "attn.q", "write", "attn.k", "write", "bilinear_form", "same"),
    Template("attn_qk_inputs", "Query vs key input directions",
             "attn.q", "read", "attn.k", "read", "shared_input", "same"),
    Template("resid_mlp_to_mlp", "MLP out -> later MLP in (residual)",
             "mlp.out", "write", "mlp.in", "read", "residual", "cross"),
    Template("resid_mlp_to_gate", "MLP out -> later MLP gate (residual)",
             "mlp.out", "write", "mlp.gate", "read", "residual", "cross"),
    Template("resid_mlp_to_up", "MLP out -> later MLP up (residual)",
             "mlp.out", "write", "mlp.up", "read", "residual", "cross"),
    Template("resid_attn_to_mlp", "Attention out -> later MLP in (residual)",
             "attn.o", "write", "mlp.in", "read", "residual", "cross"),
    Template("resid_mlp_to_v", "MLP out -> later value (residual)",
             "mlp.out", "write", "attn.v", "read", "residual", "cross"),
    Template("resid_mlp_to_q", "MLP out -> later query (residual)",
             "mlp.out", "write", "attn.q", "read", "residual", "cross"),
    Template("resid_attn_to_attn", "Attention out -> later value (residual)",
             "attn.o", "write", "attn.v", "read", "residual", "cross"),
]


def by_role(spaces: dict[str, ModuleSpaces]) -> dict[tuple[str, int], str]:
    """`(role, layer) -> module path` for everything the run decomposed."""
    return {(s.role, s.layer): m for m, s in spaces.items()}


def available_templates(spaces: dict[str, ModuleSpaces]) -> list[dict[str, object]]:
    """Templates a run can actually instantiate, each with the layers it is defined on."""
    roles = by_role(spaces)

    def layers_of(role: str) -> list[int]:
        return sorted({layer for (r, layer) in roles if r == role})

    out: list[dict[str, object]] = []
    for t in TEMPLATES:
        la, lb = layers_of(t.a_role), layers_of(t.b_role)
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
                "a_role": t.a_role, "a_side": t.a_side,
                "b_role": t.b_role, "b_side": t.b_side,
                "link": t.link,
                "link_label": LINKS[t.link]["label"],
                "caveat": LINKS[t.link]["caveat"],
                "layer_mode": t.layer_mode,
                "layers": layers,
            }
        )
    return out


def link_for(a: ModuleSpaces, a_side: Side, b: ModuleSpaces, b_side: Side) -> str | None:
    """The link kind of an arbitrary (manually selected) pairing, or None if it matches no template."""

    def matched(x: ModuleSpaces, xs: Side, y: ModuleSpaces, ys: Side) -> str | None:
        for t in TEMPLATES:
            if (t.a_role, t.a_side, t.b_role, t.b_side) != (x.role, xs, y.role, ys):
                continue
            if t.layer_mode == "same" and x.layer != y.layer:
                continue
            return t.link
        return None

    link = matched(a, a_side, b, b_side) or matched(b, b_side, a, a_side)
    if link is not None:
        return link
    if a.space(a_side).key == b.space(b_side).key:
        return "identity"
    return None
