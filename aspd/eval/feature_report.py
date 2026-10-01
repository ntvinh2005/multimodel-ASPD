"""Static HTML report of SAE latents and their activation examples from a `harvest.db`."""

import html
import json
import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path

from param_decomp_lab.harvest.db import HarvestDB
from param_decomp_lab.harvest.schemas import ActivationExample, ComponentData

ACTIVATION_KEY = "activation"


def _token_span(text: str, act: float, peak: float) -> str:
    """One token rendered with background opacity proportional to its activation."""
    alpha = 0.0 if peak <= 0 else max(0.0, min(1.0, act / peak))
    safe = html.escape(text).replace(" ", "&nbsp;")
    return f'<span class="tok" style="background:rgba(34,139,230,{alpha:.3f})" title="{act:.3g}">{safe}</span>'


def _example_html(ex: ActivationExample, decode: Callable[[list[int]], list[str]], window: int) -> str:
    acts = ex.activations.get(ACTIVATION_KEY, [0.0] * len(ex.token_ids))
    toks = decode(ex.token_ids)
    peak = max(acts) if acts else 0.0
    center = acts.index(peak) if acts else 0
    lo, hi = max(0, center - window), min(len(toks), center + window + 1)

    def render(a: int, b: int) -> str:
        return "".join(_token_span(toks[i], acts[i], peak) for i in range(a, b))

    narrow = render(lo, hi)
    full = render(0, len(toks))
    # <details> gives click-to-expand with zero JS; the narrow window is the summary line.
    return (
        f'<details class="ex"><summary><span class="peak">{peak:.3g}</span> {narrow}</summary>'
        f'<div class="full">{full}</div></details>'
    )


def _pmi_html(label: str, pmi, decode: Callable[[list[int]], list[str]]) -> str:
    if pmi is None or not getattr(pmi, "top", None):
        return ""
    chips = "".join(
        f'<span class="chip">{html.escape(decode([tid])[0])} <em>{score:.2f}</em></span>'
        for tid, score in pmi.top[:10]
    )
    return f'<div class="pmi"><span class="pmi-label">{label}</span>{chips}</div>'


def _card_html(
    comp: ComponentData,
    decode: Callable[[list[int]], list[str]],
    logit_lens: dict[str, list[tuple[str, float]]] | None,
    labels: dict[str, tuple[str, str]] | None,
    *,
    examples_per_latent: int,
    window: int,
) -> str:
    mean_act = comp.mean_activations.get(ACTIVATION_KEY, 0.0)
    label_html = ""
    if labels and comp.component_key in labels:
        label, reasoning = labels[comp.component_key]
        label_html = (
            f'<div class="label" title="{html.escape(reasoning)}">'
            f'{html.escape(label)}</div>'
        )
    head = (
        f'<div class="head"><b>{html.escape(comp.component_key)}</b>'
        f'<span class="stat">density {comp.firing_density:.2e}</span>'
        f'<span class="stat">mean {mean_act:.3g}</span></div>'
        f"{label_html}"
    )
    ll = ""
    if logit_lens and comp.component_key in logit_lens:
        chips = "".join(
            f'<span class="chip">{html.escape(t)} <em>{s:.2f}</em></span>'
            for t, s in logit_lens[comp.component_key]
        )
        ll = f'<div class="pmi"><span class="pmi-label">logit lens</span>{chips}</div>'
    pmi_in = _pmi_html("input PMI", comp.input_token_pmi, decode)
    pmi_out = _pmi_html("output PMI", comp.output_token_pmi, decode)
    examples = "".join(
        _example_html(ex, decode, window) for ex in comp.activation_examples[:examples_per_latent]
    )
    return f'<div class="card">{head}{ll}{pmi_in}{pmi_out}<div class="examples">{examples}</div></div>'


_STYLE = """
.fs-root{font-family:system-ui,sans-serif;background:#0f1115;color:#e6e6e6;min-height:100vh}
.fs-root *{box-sizing:border-box}
.fs-root header{position:sticky;top:0;background:#171a21;padding:10px 16px;border-bottom:1px solid #2a2f3a;z-index:2}
header input{width:260px;padding:6px 10px;background:#0f1115;border:1px solid #2a2f3a;color:#e6e6e6;border-radius:6px}
.wrap{padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:12px}
.card{background:#171a21;border:1px solid #2a2f3a;border-radius:10px;padding:12px;overflow:hidden}
.head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}
.head b{font-size:14px}
.stat{font-size:11px;color:#9aa4b2;background:#0f1115;padding:2px 6px;border-radius:5px}
.label{font-size:13px;color:#c0caf5;background:#1e2330;border-left:3px solid #7aa2f7;padding:4px 8px;border-radius:4px;margin:2px 0 6px}
.pmi{margin:4px 0;display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.pmi-label{font-size:11px;color:#9aa4b2;margin-right:4px}
.chip{font-size:12px;background:#0f1115;border:1px solid #2a2f3a;padding:1px 6px;border-radius:5px}
.chip em{color:#7aa2f7;font-style:normal}
.examples{margin-top:8px}
.ex{margin:3px 0;border-top:1px solid #22262f;padding-top:4px}
.ex summary{cursor:pointer;line-height:1.9;white-space:nowrap;overflow-x:auto}
.ex summary::-webkit-scrollbar{height:5px}.ex summary::marker{color:#5a6472}
.peak{display:inline-block;min-width:44px;color:#7aa2f7;font-size:11px}
.full{line-height:2;margin-top:6px;padding:6px;background:#0f1115;border-radius:6px;white-space:pre-wrap;overflow-wrap:anywhere}
.tok{border-radius:3px;padding:0 1px}
"""

_SCRIPT = """
const q=document.getElementById('q');
q.addEventListener('input',()=>{const v=q.value.toLowerCase();
document.querySelectorAll('.card').forEach(c=>{c.style.display=c.querySelector('.head b').textContent.toLowerCase().includes(v)?'':'none';});});
"""


def load_labels(interp_db: Path) -> dict[str, tuple[str, str]]:
    """`{component_key -> (label, reasoning)}` from an autointerp `interp.db`."""
    con = sqlite3.connect(f"file:{interp_db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT component_key, label, reasoning FROM interpretations").fetchall()
    finally:
        con.close()
    return {k: (label, reasoning) for k, label, reasoning in rows}


def build_html(
    components: Iterable[ComponentData],
    decode: Callable[[list[int]], list[str]],
    *,
    logit_lens: dict[str, list[tuple[str, float]]] | None = None,
    labels: dict[str, tuple[str, str]] | None = None,
    examples_per_latent: int = 12,
    window: int = 12,
    title: str = "SAE features",
    fragment: bool = False,
) -> str:
    """Full standalone HTML document, or (when `fragment`) just the `<style>` + content, for
    embedding in a host page that supplies its own `<head>`/`<body>`.
    """
    cards = [
        _card_html(
            c, decode, logit_lens, labels, examples_per_latent=examples_per_latent, window=window
        )
        for c in components
    ]
    body = "".join(cards)
    inner = (
        f"<style>{_STYLE}</style><div class='fs-root'>"
        f"<header><b>{html.escape(title)}</b> &nbsp; {len(cards)} latents &nbsp; "
        f"<input id='q' placeholder='filter by site:idx'></header>"
        f"<div class='wrap'>{body}</div></div><script>{_SCRIPT}</script>"
    )
    if fragment:
        return inner
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"</head><body>{inner}</body></html>"
    )


def write_feature_report(
    db_path: Path,
    decode: Callable[[list[int]], list[str]],
    out_html: Path,
    *,
    logit_lens_json: Path | None = None,
    interp_db: Path | None = None,
    max_latents: int | None = None,
    examples_per_latent: int = 12,
    window: int = 12,
    title: str = "SAE features",
) -> Path:
    """Render every fired latent in `db_path` to a single static HTML file."""
    db = HarvestDB(db_path, readonly=True)
    if max_latents is not None:
        # Show the most active latents across all sites, not the first N of the first site.
        densities = sorted(db.get_component_densities(min_examples=1), key=lambda kv: -kv[1])
        keys = [k for k, _ in densities[:max_latents]]
    else:
        keys = db.get_component_keys()
    components = [c for c in (db.get_component(k) for k in keys) if c is not None]

    logit_lens: dict[str, list[tuple[str, float]]] | None = None
    if logit_lens_json is not None:
        raw = json.loads(Path(logit_lens_json).read_text())
        # {key: {"top": [...], "bottom": [...]}}; the static card shows the top row.
        logit_lens = {k: [(t, float(s)) for t, s in v["top"]] for k, v in raw.items()}

    labels = load_labels(interp_db) if interp_db is not None else None

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        build_html(
            components,
            decode,
            logit_lens=logit_lens,
            labels=labels,
            examples_per_latent=examples_per_latent,
            window=window,
            title=title,
        )
    )
    print(f"[feature-report] {len(components)} latents -> {out_html}", flush=True)
    return out_html
