"""Causal-importance functions added to `param_decomp`.

- `pd_transcoder`: PD Transcoder. g_{t,c} = 1[c in BatchTopK(relu(v_c^T (x_t - b_dec)))]; the
  gate is read off the components' own encoder V.
- `aspd`: ASPD. g_{t,c} = phi(g^s_{t,c}(R)), where g^s is a shared BatchTopK encoder on the
  residual stream r_t and phi is the indicator.
"""

from aspd.ci.aspd import (
    ASPDCiConfig,
    ASPDCiFn,
    ASPDCiFnSet,
    SharedEncoder,
    encoder_key_map,
    make_aspd_ci_fn,
)
from aspd.ci.pd_transcoder import (
    PDTranscoderCiConfig,
    PDTranscoderCiFn,
    PDTranscoderCiFnSet,
    gate_for,
    make_transcoder_ci_fn,
)

__all__ = [
    "SharedEncoder",
    "ASPDCiConfig",
    "ASPDCiFn",
    "ASPDCiFnSet",
    "PDTranscoderCiConfig",
    "PDTranscoderCiFn",
    "PDTranscoderCiFnSet",
    "gate_for",
    "make_aspd_ci_fn",
    "make_transcoder_ci_fn",
    "encoder_key_map",
]
