"""Small web viewer over a `harvest.db`: one latent's activation examples at a time."""

import json
from collections.abc import Callable
from pathlib import Path

from param_decomp_lab.harvest.db import HarvestDB
from param_decomp_lab.harvest.schemas import ComponentData

ACTIVATION_KEY = "activation"


def list_sites(db: HarvestDB) -> list[dict[str, object]]:
    """`[{site, n_features}]` derived from the harvested `component_key`s (`<site>:<idx>`)."""
    counts: dict[str, int] = {}
    for key in db.get_component_keys():
        site = key.rsplit(":", 1)[0]
        counts[site] = counts.get(site, 0) + 1
    return [{"site": s, "n_features": n} for s, n in sorted(counts.items())]


def _example_view(
    token_ids: list[int], acts: list[float], decode: Callable[[list[int]], list[str]], window: int
) -> dict[str, object]:
    toks = decode(token_ids)
    peak = max(acts) if acts else 0.0
    center = acts.index(peak) if acts else 0
    lo, hi = max(0, center - window), min(len(toks), center + window + 1)
    return {
        "peak": peak,
        "short": {"tokens": toks[lo:hi], "acts": acts[lo:hi], "target_idx": center - lo},
        "full": {"tokens": toks, "acts": acts, "target_idx": center},
    }


def feature_detail(
    comp: ComponentData,
    decode: Callable[[list[int]], list[str]],
    *,
    n_intervals: int,
    examples_per_interval: int,
    window: int,
    activation_key: str = ACTIVATION_KEY,
) -> dict[str, object]:
    """Assemble one feature's payload: examples binned by peak activation into `n_intervals`,
    at most `examples_per_interval` per bin (highest-activation first).
    """
    keys = sorted({k for ex in comp.activation_examples for k in ex.activations})
    assert activation_key in keys or not keys, (
        f"{comp.component_key} has no {activation_key!r} series; harvested keys are {keys}"
    )
    views = [
        _example_view(ex.token_ids, ex.activations.get(activation_key, []), decode, window)
        for ex in comp.activation_examples
    ]
    views = [v for v in views if v["peak"] > 0]
    max_act = max((v["peak"] for v in views), default=0.0)

    intervals: list[dict[str, object]] = []
    if max_act > 0 and views:
        step = max_act / n_intervals
        for i in range(n_intervals, 0, -1):
            lo, hi = step * (i - 1), step * i
            in_bin = [
                v for v in views if (v["peak"] >= lo and (v["peak"] < hi or i == n_intervals))
            ]
            if not in_bin:
                continue
            in_bin.sort(key=lambda v: -v["peak"])
            intervals.append({"min": lo, "max": hi, "examples": in_bin[:examples_per_interval]})

    return {
        "feature": comp.component_key,
        "site": comp.layer,
        "idx": comp.component_idx,
        "density": comp.firing_density,
        "max_act": max_act,
        "mean_act": comp.mean_activations.get(activation_key, 0.0),
        "activation_key": activation_key,
        "activation_keys": keys,
        "input_pmi": _pmi_list(comp.input_token_pmi, decode),
        "output_pmi": _pmi_list(comp.output_token_pmi, decode),
        "intervals": intervals,
    }


def _pmi_list(pmi: object, decode: Callable[[list[int]], list[str]]) -> list[tuple[str, float]]:
    top = getattr(pmi, "top", None)
    if not top:
        return []
    return [(decode([tid])[0], float(score)) for tid, score in top[:10]]


def build_app(
    harvest_db: Path,
    decode: Callable[[list[int]], list[str]],
    *,
    interp_db: Path | None = None,
    logit_lens_json: Path | None = None,
    n_intervals: int = 5,
    examples_per_interval: int = 5,
    window: int = 10,
):
    """A FastAPI app serving the feature viewer over one `harvest.db` (opened read-only per request)."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    labels = _load_labels(interp_db)
    logit_lens = (
        json.loads(Path(logit_lens_json).read_text()) if logit_lens_json is not None else {}
    )
    startup_db = HarvestDB(harvest_db, readonly=True)
    sites = list_sites(startup_db)
    browse = _browse_lists(startup_db, labels)

    app = FastAPI(title="SAE feature viewer")

    @app.get("/api/sites")
    def _sites() -> JSONResponse:
        return JSONResponse(sites)

    @app.get("/api/browse/{site}")
    def _browse(site: str) -> JSONResponse:
        return JSONResponse(browse.get(site, []))

    @app.get("/api/feature/{site}/{idx}")
    def _feature(site: str, idx: int) -> JSONResponse:
        key = f"{site}:{idx}"
        comp = HarvestDB(harvest_db, readonly=True).get_component(key)
        if comp is None:
            raise HTTPException(404, f"{key} not found (dead latent or out of range)")
        detail = feature_detail(
            comp, decode, n_intervals=n_intervals,
            examples_per_interval=examples_per_interval, window=window,
        )
        detail["label"] = labels.get(key)
        detail["logit_lens"] = logit_lens.get(key, {})
        return JSONResponse(detail)

    @app.get("/", response_class=HTMLResponse)
    def _index() -> str:
        return _INDEX_HTML

    return app


_BROWSE_CAP = 300
_UNINTERP_MARKERS = ("unclear", "uninterpretable", "unknown", "no clear", "ambiguous", "none")


def _browse_rank(label: str | None) -> int:
    """0 = interpretable label, 1 = 'unclear'-style label, 2 = unlabeled."""
    if label is None:
        return 2
    return 1 if any(m in label.lower() for m in _UNINTERP_MARKERS) else 0


def _browse_lists(
    db: HarvestDB, labels: dict[str, dict[str, str]]
) -> dict[str, list[dict[str, object]]]:
    """Per-site browse dropdown: interpretable labels first, then unclear, then unlabeled by density."""
    per_site: dict[str, list[tuple[int, float, int, str | None]]] = {}
    for key, density in db.get_component_densities(min_examples=1):
        site, idx = key.rsplit(":", 1)
        label = labels.get(key, {}).get("label")
        per_site.setdefault(site, []).append((_browse_rank(label), density, int(idx), label))
    out: dict[str, list[dict[str, object]]] = {}
    for site, rows in per_site.items():
        rows.sort(key=lambda r: (r[0], -r[1]))
        out[site] = [
            {"idx": idx, "label": label, "density": density}
            for _rank, density, idx, label in rows[:_BROWSE_CAP]
        ]
    return out


def _load_labels(interp_db: Path | None) -> dict[str, dict[str, str]]:
    if interp_db is None:
        return {}
    import sqlite3

    con = sqlite3.connect(f"file:{interp_db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT component_key, label, reasoning FROM interpretations").fetchall()
    finally:
        con.close()
    return {k: {"label": lab, "reasoning": rea} for k, lab, rea in rows}


# Escape the reference example's static assets into one page. Vanilla JS: no build step, CSP-free.
_INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>SAE feature viewer</title><style>
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f8f9fa;color:#333;margin:0}
.bar{position:sticky;top:0;background:#fff;border-bottom:1px solid #e5e7eb;padding:12px 20px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;z-index:5}
.bar b{font-size:15px;margin-right:6px}
select,input,button{padding:6px 10px;border:1px solid #ced4da;border-radius:6px;font-size:14px}
button{background:#0d6efd;color:#fff;border-color:#0d6efd;cursor:pointer}
button:hover{background:#0b5ed7}
#meta{padding:10px 20px;color:#555;font-size:13px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.label{background:#eef2ff;border-left:3px solid #6366f1;padding:6px 10px;border-radius:4px;color:#3730a3;font-size:14px;margin:8px 20px}
.chips{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:2px 20px}
.chips .lab{font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.4px;margin-right:4px}
.chip{font-size:12px;background:#f1f3f5;border:1px solid #e5e7eb;border-radius:5px;padding:1px 6px}
.chip em{color:#0d6efd;font-style:normal}
.container{max-width:1100px;margin:14px auto;background:#fff;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.05);overflow:visible}
.interval-header{background:#f1f3f5;padding:5px 20px;font-size:.72rem;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:.5px;display:flex;justify-content:space-between}
.example-row{display:flex;align-items:flex-start;padding:6px 15px;border-bottom:1px solid #f4f4f4}
.meta-col{width:56px;flex-shrink:0;display:flex;flex-direction:column;align-items:center;margin-right:14px;padding-top:2px}
.act-badge{background:#e9ecef;color:#495057;padding:1px 5px;border-radius:3px;font-size:.62rem;font-weight:700;text-transform:uppercase}
.act-val{color:#28a745;font-weight:700;font-size:.85rem}
.seq-col{flex-grow:1;min-width:0;font-family:'Consolas',Monaco,monospace;font-size:13px;line-height:1.9;cursor:pointer}
/* wrap (not scroll) so the hover tooltip is never clipped by an overflow container */
.short-view{white-space:pre-wrap;overflow-wrap:anywhere}
.full-view{white-space:pre-wrap;overflow-wrap:anywhere;margin-top:6px;padding:6px;background:#fafafa;border-radius:6px}
.token{border-radius:2px;position:relative;cursor:help}
.token:hover{outline:1px solid #444;z-index:10}
.target-token{border-bottom:2px solid #333;font-weight:700}
.token .tt{visibility:hidden;opacity:0;position:absolute;bottom:130%;left:50%;transform:translateX(-50%);background:#333;color:#fff;padding:3px 7px;border-radius:4px;font:11px sans-serif;white-space:nowrap;pointer-events:none;transition:opacity .15s;z-index:100}
.token:hover .tt{visibility:visible;opacity:1}
#empty{padding:40px;text-align:center;color:#999}
</style></head><body>
<div class="bar"><b>SAE feature viewer</b>
  <label>SAE <select id="site"></select></label>
  <label>browse <select id="feat" style="max-width:340px"><option value="">— pick a feature —</option></select></label>
  <label>or id <input id="idx" type="number" min="0" value="0" style="width:100px" placeholder="id"></label>
  <button onclick="load()">Search</button>
  <span id="hint" style="color:#888;font-size:12px"></span>
</div>
<div id="out"><div id="empty">Pick an SAE and a feature index, then Load.</div></div>
<script>
let SITES=[];
async function init(){
  SITES=await (await fetch('/api/sites')).json();
  const sel=document.getElementById('site');
  SITES.forEach(s=>{const o=document.createElement('option');o.value=s.site;o.textContent=`${s.site} (${s.n_features})`;sel.appendChild(o);});
  sel.onchange=()=>{updateHint();fillBrowse();}; updateHint(); fillBrowse();
  document.getElementById('idx').addEventListener('keydown',e=>{if(e.key==='Enter')load();});
  document.getElementById('feat').addEventListener('change',e=>{
    if(e.target.value==='')return;
    document.getElementById('idx').value=e.target.value; load();
  });
}
function updateHint(){
  const s=SITES.find(x=>x.site===document.getElementById('site').value);
  document.getElementById('hint').textContent=s?`0 … ${s.n_features-1}`:'';
  document.getElementById('idx').max=s?s.n_features-1:0;
}
async function fillBrowse(){
  const site=document.getElementById('site').value;
  const rows=await (await fetch(`/api/browse/${encodeURIComponent(site)}`)).json();
  const sel=document.getElementById('feat');
  sel.innerHTML='<option value="">— pick a feature —</option>';
  rows.forEach(r=>{const o=document.createElement('option');o.value=r.idx;
    o.textContent=r.label?`${r.idx} — ${r.label}`:`${r.idx} (density ${r.density.toExponential(1)})`;
    sel.appendChild(o);});
}
function esc(t){return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'\\u21b5');}
function renderTokens(o,maxAct){
  let h='';
  for(let i=0;i<o.tokens.length;i++){
    const a=o.acts[i]||0, alpha=maxAct>0?Math.max(0,Math.min(1,a/maxAct)):0;
    const bg=a>0?`rgba(0,200,83,${alpha})`:'transparent';
    const cls='token'+(i===o.target_idx?' target-token':'');
    h+=`<span class="${cls}" style="background:${bg}">${esc(o.tokens[i])}<span class="tt">${a.toFixed(4)}</span></span>`;
  }
  return h;
}
function toggle(el){const s=el.querySelector('.short-view'),f=el.querySelector('.full-view');
  const showFull=s.style.display!=='none'; s.style.display=showFull?'none':'block'; f.style.display=showFull?'block':'none';}
async function load(){
  const site=document.getElementById('site').value, idx=document.getElementById('idx').value;
  const out=document.getElementById('out'); out.innerHTML='<div id="empty">Loading…</div>';
  const r=await fetch(`/api/feature/${encodeURIComponent(site)}/${idx}`);
  if(!r.ok){out.innerHTML=`<div id="empty">${(await r.json()).detail||'not found'}</div>`;return;}
  const d=await r.json();
  let h=`<div id="meta"><b>${d.feature}</b>`+
    `<span>density ${d.density.toExponential(2)}</span><span>max act ${d.max_act.toFixed(3)}</span><span>mean ${d.mean_act.toFixed(3)}</span></div>`;
  if(d.label)h+=`<div class="label" title="${esc(d.label.reasoning||'')}">${esc(d.label.label)}</div>`;
  const chips=(lab,arr)=>arr&&arr.length?`<div class="chips"><span class="lab">${lab}</span>`+arr.map(([t,s])=>`<span class="chip">${esc(t)} <em>${s.toFixed(2)}</em></span>`).join('')+'</div>':'';
  const ll=d.logit_lens||{};
  h+=chips('logit lens ↑ top',ll.top)+chips('logit lens ↓ bottom',ll.bottom);
  h+=chips('input pmi',d.input_pmi)+chips('output pmi',d.output_pmi);
  h+='<div class="container">';
  if(!d.intervals.length)h+='<div id="empty">No activating examples stored for this feature.</div>';
  for(const iv of d.intervals){
    h+=`<div class="interval-header"><span>Interval ${iv.min.toFixed(3)} – ${iv.max.toFixed(3)}</span><span>${iv.examples.length} example(s)</span></div>`;
    for(const ex of iv.examples){
      h+=`<div class="example-row"><div class="meta-col"><span class="act-badge">max</span><span class="act-val">${ex.peak.toFixed(2)}</span></div>`+
         `<div class="seq-col" onclick="toggle(this)"><div class="short-view">${renderTokens(ex.short,d.max_act)}</div>`+
         `<div class="full-view" style="display:none">${renderTokens(ex.full,d.max_act)}</div></div></div>`;
    }
  }
  h+='</div>'; out.innerHTML=h;
}
init();
</script></body></html>"""
