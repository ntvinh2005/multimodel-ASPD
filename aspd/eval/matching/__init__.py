"""Meaning localization (matching): does a component write the meaning it fires on?

Procedure: pair each component c with the output-SAE feature it affects most,
pi(c) = argmax_j E_{t in A_j}[|zeta_c(t)|] M_{j,c} with M_{j,c} = <W_enc[:, j], u_c>; show an LLM
judge activating examples of both and score SIMILAR / MAYBE / DIFFERENT as 3 / 2 / 1; report the
mean score minus that of random component-feature pairs.
"""

import re

MODES = ("c2o", "i2o")
DEFAULT_MODE = "c2o"
"""What `run_all_evals.sh` submits, and what every reader prefers when a directory holds both."""

REPORT_RE = re.compile(r"^matching_(?:(?P<mode>c2o|i2o)_)?step(?P<step>\d+)\.json$")
"""`matching_c2o_step250000.json`, `matching_i2o_step250000.json`, or the LEGACY
`matching_step<N>.json` written before the modes were split, whose mode is only knowable from
`meta.scheme`."""


def report_name(mode: str, step: int) -> str:
    assert mode in MODES, f"unknown matching mode {mode!r}; have {MODES}"
    return f"matching_{mode}_step{step}.json"


def mode_of(meta: dict) -> str:
    scheme = (meta or {}).get("scheme")
    if scheme is None or scheme == "c2o":
        return "c2o"
    if scheme in ("i2o", "in_to_out"):
        return "i2o"
    raise AssertionError(f"unknown matching scheme {scheme!r}; have {MODES}")
