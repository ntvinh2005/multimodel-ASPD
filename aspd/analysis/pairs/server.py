"""FastAPI app for the pair viewer (inline page, no build step, CPU only)."""

import json
import threading

from aspd.analysis.pairs import endpoints as ep
from aspd.analysis.pairs import features as ft
from aspd.analysis.pairs import scores as sc
from aspd.analysis.pairs.clean import (byte_decoder, decode_token, decode_tokens,
                                   dedupe_activations, tidy_explanations)
from aspd.analysis.pairs.lens import LogitLens
from aspd.analysis.pairs.spaces import Side, compatibility
from aspd.analysis.pairs.store import HarvestStore, InterpStore, KappaStore, PairScoreStore
from aspd.analysis.pairs.suggest import LINKS, available_templates, link_for
from aspd.analysis.pairs.weights import RunWeights


def _space_json(s) -> dict[str, object]:
    return {
        "key": s.key,
        "label": s.label,
        "dim": s.dim,
        "n_heads": s.heads.n_heads if s.heads else None,
        "head_dim": s.heads.head_dim if s.heads else None,
    }


def build_app(
    *,
    run_name: str,
    weights: RunWeights,
    harvest: HarvestStore,
    interp: InterpStore,
    pair_scores: PairScoreStore,
    kappa: KappaStore,
    saes: ft.SaeStore,
    norms: ft.TargetNorms,
    neuronpedia: bool = True,
    neuronpedia_timeout: float = 6.0,
):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title=f"component pairs — {run_name}")
    overview_cache: dict[tuple, dict] = {}
    lens = LogitLens(weights.model_name)
    threading.Thread(target=lambda: lens.available, daemon=True).start()
    np_cache: dict[tuple[str, int], tuple[dict | None, str | None]] = {}
    sae_catalogue = ft.catalogue(weights.model_name) if saes.available else []
    sae_by_key = {str(c["key"]): c for c in sae_catalogue}

    def resolve(key: str, side: Side) -> ep.Endpoint:
        """One endpoint from a module path OR an SAE `<release>:L<layer>` key."""
        if key in weights.spaces:
            return ep.module_endpoint(key, side, weights.spaces[key], weights.n_components(key))
        if not ft.is_sae_key(key):
            raise HTTPException(404, f"{key} is not a decomposed module of this run")
        if not saes.available:
            raise HTTPException(404, f"no SAE releases are configured for {weights.model_name}")
        release_key, layer = ft.parse_endpoint_key(key)
        entry = sae_by_key.get(release_key)
        if entry is None:
            raise HTTPException(404, f"no SAE release {release_key!r} for {weights.model_name}")
        if layer not in entry["layers"]:  # pyright: ignore[reportOperatorIssue]
            raise HTTPException(404, f"release {release_key!r} has no layer {layer}")
        return ep.sae_endpoint(saes.endpoint(key), side)

    def block(e: ep.Endpoint):
        return saes.directions(e.key, e.side) if e.is_sae else weights.directions(e.key, e.side)

    def link_of(a: ep.Endpoint, b: ep.Endpoint) -> str | None:
        if a.is_sae or b.is_sae:
            return ep.sae_link(a, b, weights.model_name)
        return link_for(weights.spaces[a.key], a.side, weights.spaces[b.key], b.side)

    def endpoints(a_module: str, a_side: Side, b_module: str, b_side: Side):
        a, b = resolve(a_module, a_side), resolve(b_module, b_side)
        return a, b, a.space, b.space

    def _basis_json(a: ep.Endpoint, b: ep.Endpoint) -> dict[str, object]:
        """What `reconcile` will do to this pair, declared BEFORE any number is produced."""
        if not (a.is_sae or b.is_sae):
            return {"corrections": [], "centred": False, "d_eff_drop": 0}
        folds = []
        for name, e in (("a", a), ("b", b)):
            spec = ep.reads_through_layernorm(e, weights.model_name)
            if spec is not None:
                folds.append({"endpoint": name, "norm": spec.key.format(layer=e.layer),
                              "centres": spec.centres})
        centred = a.centred or b.centred
        corrections = [f"fold {f['norm']}" for f in folds]
        if centred:
            corrections.append("centre both sides (TransformerLens basis)")
        return {"corrections": corrections, "layernorm_folds": folds,
                "centred": centred, "d_eff_drop": 1 if centred else 0}

    def ep_json(e: ep.Endpoint) -> dict[str, object]:
        return {"module": e.key, "key": e.key, "side": e.side, "role": e.role, "layer": e.layer,
                "kind": e.kind, "label": e.label, "space": _space_json(e.space),
                "n_components": e.n_components, "centred": e.centred, "position": e.position}

    def density_mask(b: ep.Endpoint, max_density: float):
        """Bool mask over B keeping components that fire, but not too often. `(mask, note)`."""
        if max_density <= 0:
            return None, None
        if b.is_sae:
            return None, "no density filter on a dictionary: features have no harvested density"
        d = harvest.densities(b.key, b.n_components)
        if d is None:
            return None, "no density filter: this run has no harvest for that module"
        mask = (d > 0) & (d <= max_density)
        return mask, None

    def kappa_ok(a: ep.Endpoint, b: ep.Endpoint) -> bool:
        """Whether `dot_coact` can be served for this pairing."""
        return (kappa.available and not a.is_sae and not b.is_sae
                and kappa.covers(a.key, b.key))

    def kappa_reason(a: ep.Endpoint, b: ep.Endpoint) -> str:
        if a.is_sae or b.is_sae:
            return ("undefined against a dictionary: a feature has no gate and no component "
                    "activation, so there is no co-activation coefficient to measure")
        if not kappa.available:
            return "no pair_coactivation.pt — run slurm/app_coactivation.sbatch"
        return ("not harvested for this module pair; only the same-layer templates (OV, QK, "
                "MLP in→out) are covered")

    def resolve_mode(mode: str, compat: dict, link: str | None) -> bool:
        """`auto` picks the mode that answers the question the link poses."""
        if mode == "flat":
            return False
        if mode == "head":
            return True
        if link == "bilinear_form" and compat["per_head"]:
            return True
        return not compat["flat"] and bool(compat["per_head"])

    @app.get("/api/meta")
    def meta() -> JSONResponse:
        mods = [
            {
                "module": m,
                "role": s.role,
                "layer": s.layer,
                "n_components": weights.n_components(m),
                "read": _space_json(s.read),
                "write": _space_json(s.write),
            }
            for m, s in ((m, weights.spaces[m]) for m in weights.modules)
        ]
        return JSONResponse(
            {
                "run": run_name,
                "model": weights.model_name,
                "checkpoint": str(weights.ckpt_path),
                "head_dim": weights.head_dim,
                "modules": mods,
                "templates": available_templates(weights.spaces),
                "saes": sae_catalogue,
                "sae_templates": ep.available_sae_templates(
                    weights.spaces, weights.model_name, sae_catalogue
                ),
                "sae_sites": {k: v.label for k, v in ft.SITES.items()},
                "metrics": [m.__dict__ for m in sc.METRIC_SPECS],
                "links": LINKS,
                "harvest": harvest.available,
                "harvest_path": str(harvest.path) if harvest.path else None,
                "interp": interp.available,
                "interp_path": str(interp.path) if interp.path else None,
                "pair_scores": pair_scores.available,
                "kappa": kappa.available,
                "kappa_path": str(kappa.path) if kappa.path else None,
                "kappa_meta": kappa.meta,
            }
        )

    def _metric_json(m, a: ep.Endpoint, b: ep.Endpoint, computable: bool,
                     covered: set[str], compat: dict) -> dict[str, object]:
        if m.source == "kappa":
            ok = computable and kappa_ok(a, b)
            reason = None if ok else (str(compat["reason"]) if not computable
                                      else kappa_reason(a, b))
        elif m.source == "weights":
            ok = computable
            reason = None if ok else str(compat["reason"])
        else:
            ok = m.key in covered
            reason = None if ok else "no sidecar DB entry"
        return {"key": m.key, "label": m.label, "formula": m.formula, "note": m.note,
                "available": ok, "reason": reason}

    @app.get("/api/pairspace")
    def pairspace(
        a_module: str, b_module: str, a_side: Side = "write", b_side: Side = "read"
    ) -> JSONResponse:
        a, b, sa, sb = endpoints(a_module, a_side, b_module, b_side)
        compat = compatibility(sa, sb)
        covered = (
            pair_scores.covered_metrics(a_module, a_side, b_module, b_side)
            if pair_scores.available
            else set()
        )
        computable = bool(compat["flat"] or compat["per_head"])
        link = link_of(a, b)
        return JSONResponse(
            {
                "a": ep_json(a),
                "b": ep_json(b),
                "compat": compat,
                "basis": _basis_json(a, b),
                "link": link,
                "link_label": LINKS[link]["label"] if link else None,
                "caveat": LINKS[link]["caveat"] if link else
                          "No template describes this pairing; the score is a raw inner product "
                          "between two spaces whose relationship is unstated.",
                "metrics": [_metric_json(m, a, b, computable, covered, compat)
                            for m in sc.METRIC_SPECS],
                "shared_gate": bool(kappa.available and kappa.shared_gate(a.key, b.key)),
                "kappa_pool": {"a": kappa.pool_size(a.key), "b": kappa.pool_size(b.key),
                               "full": kappa.full_pool} if kappa_ok(a, b) else None,
            }
        )

    @app.get("/api/rank")
    def rank(
        a_module: str,
        b_module: str,
        idx: int,
        a_side: Side = "write",
        b_side: Side = "read",
        metric: str = "cosine",
        mode: str = "auto",
        head: int | None = None,
        k: int = Query(25, ge=1, le=200),
        max_density: float = Query(0.0, ge=0.0, le=1.0),
    ) -> JSONResponse:
        a, b, sa, sb = endpoints(a_module, a_side, b_module, b_side)
        compat = compatibility(sa, sb)
        if metric not in sc.DIRECTION_METRICS:
            rows = pair_scores.row(metric, a_module, a_side, idx, b_module, b_side)
            if not rows:
                raise HTTPException(
                    409,
                    f"metric '{metric}' has no stored data for this pair space — it needs a data "
                    "pass written into the pair-score DB",
                )
            rows.sort(key=lambda r: -r["score"])
            return JSONResponse(
                {"top": rows[:k], "bottom": rows[::-1][:k], "source": "db", "per_head": False}
            )
        if not (compat["flat"] or compat["per_head"]):
            raise HTTPException(409, str(compat["reason"]))
        per_head = resolve_mode(mode, compat, link_of(a, b))
        if per_head and not compat["per_head"]:
            raise HTTPException(409, "this pair space has no head structure")
        if not per_head and not compat["flat"]:
            raise HTTPException(409, str(compat["reason"]))
        if not 0 <= idx < a.n_components:
            raise HTTPException(404, f"component {idx} out of range (0..{a.n_components - 1})")
        kap, covered = None, None
        if metric in sc.KAPPA_METRICS:
            if not kappa_ok(a, b):
                raise HTTPException(409, f"'{metric}' is unavailable here: {kappa_reason(a, b)}")
            got = kappa.row(a.key, idx, b.key, b.n_components)
            if got is None:
                raise HTTPException(
                    409,
                    f"component {idx} of {a.key} is outside the kappa harvest's pool "
                    f"({kappa.pool_size(a.key)} of {a.n_components} components were measured)",
                )
            kap, covered = got
        dens, dens_note = density_mask(b, max_density)
        if dens is not None:
            covered = dens if covered is None else (covered & dens)
        rec = ep.reconcile(a, b, block(a), block(b), model_name=weights.model_name, norms=norms)
        xa, yb = rec.a, rec.b
        row = sc.score_row(
            xa, yb, idx, metric=metric, space_a=sa, space_b=sb,  # pyright: ignore[reportArgumentType]
            per_head=bool(per_head), head=head, kappa=kap, keep=covered,
        )
        same = a.key == b.key and a.side == b.side
        out = sc.top_bottom(row, k=k, exclude=idx if same else None, covered=covered)
        d_eff = (sa.heads.head_dim if (per_head and sa.heads) else sa.dim) - rec.d_eff_drop
        for row in out["top"] + out["bottom"]:  # pyright: ignore[reportOperatorIssue]
            row["z_theory"] = row["score"] * (d_eff**0.5) if metric == "cosine" else None
        out |= {"source": "weights", "per_head": bool(per_head), "d_eff": d_eff,
                "self_norm": float(xa.norms[idx]), "applied": rec.applied,
                "shared_gate": bool(metric in sc.KAPPA_METRICS
                                    and kappa.shared_gate(a.key, b.key)),
                "max_density": max_density,
                "n_dense_excluded": int((~dens).sum()) if dens is not None else 0,
                "density_note": dens_note}
        return JSONResponse(out)

    def _lens_json(u, space) -> dict[str, object]:
        """`u_c -> ln_f -> unembed` for one write direction, or why there is none."""
        out: dict[str, object] = {
            "available": False,
            "in_residual_stream": space.get("key") == "resid",
            "reason": None,
        }
        if not lens.available:
            out["reason"] = lens.reason
            return out
        if u.numel() != lens.d_model:
            out["reason"] = (
                f"{space.get('label')} is {u.numel()}-dimensional and the final norm is "
                f"{lens.d_model}-dimensional"
            )
            return out
        return out | {"available": True, **lens.top(u, k=10)}

    def _neuronpedia(np_id: str, idx: int) -> tuple[dict | None, str | None]:
        """One feature's dashboard from Neuronpedia, or `(None, reason)`."""
        import json as _json
        import urllib.error
        import urllib.request

        ck = (np_id, idx)
        if ck in np_cache:
            return np_cache[ck]
        url = f"https://www.neuronpedia.org/api/feature/{np_id}/{idx}"
        try:
            with urllib.request.urlopen(url, timeout=neuronpedia_timeout) as r:
                out = (_json.loads(r.read()), None)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            out = (None, f"Neuronpedia unreachable ({type(exc).__name__}): {url}")
        np_cache[ck] = out
        return out

    def _feature_body(key: str, idx: int) -> dict[str, object]:
        """The card for one SAE feature: its two directions, and its public dashboard."""
        e = saes.endpoint(key)
        if not 0 <= idx < e.d_sae:
            raise HTTPException(404, f"feature {idx} out of range (0..{e.d_sae - 1})")
        enc = saes.directions(key, "read").mat[idx]
        dec = saes.directions(key, "write").mat[idx]
        body: dict[str, object] = {
            "module": key, "idx": idx, "kind": "sae",
            "role": f"sae.{e.site}", "layer": e.layer, "hook": e.hook,
            "release": e.release.release, "release_label": e.release.label,
            "d_sae": e.d_sae, "centred": e.centred,
            "read_space": _space_json(e.space), "write_space": _space_json(e.space),
            "read_norm": float(enc.norm()), "write_norm": float(dec.norm()),
            "enc_dec_cosine": float(
                (enc @ dec) / (enc.norm().clamp_min(1e-12) * dec.norm().clamp_min(1e-12))
            ),
            "neuronpedia_id": e.neuronpedia,
            "logit_lens": _lens_json(dec, _space_json(e.space)),
            "harvest": False,
        }
        if e.neuronpedia is None:
            body["reason"] = f"release {e.release.release} has no Neuronpedia dashboard"
            return body
        body["neuronpedia_url"] = f"https://www.neuronpedia.org/{e.neuronpedia}/{idx}"
        if not neuronpedia:
            body["reason"] = "Neuronpedia lookups are disabled (--no-neuronpedia)"
            return body
        data, reason = _neuronpedia(e.neuronpedia, idx)
        if data is None:
            body["reason"] = reason
            return body
        dec_bytes = byte_decoder(weights.model_name)
        # `pos_str`/`neg_str` are unrelated tokens; an activating window is one byte stream.
        detok = lambda ts: [decode_token(t, dec_bytes) for t in ts]
        explanations, dropped = tidy_explanations(
            [x["description"] for x in (data.get("explanations") or []) if x.get("description")]
        )
        acts, acts_dropped = dedupe_activations(data.get("activations") or [])
        body |= {
            "label": {"label": explanations[0]} if explanations else None,
            "explanations": explanations,
            "explanations_dropped": dropped,
            "examples_dropped": acts_dropped,
            "firing_density": data.get("frac_nonzero"),
            "max_activation": data.get("maxActApprox"),
            "output_pmi": list(zip(detok(data.get("pos_str") or []), data.get("pos_values") or [], strict=False)),
            "input_pmi": list(zip(detok(data.get("neg_str") or []), data.get("neg_values") or [], strict=False)),
            "pmi_available": bool(data.get("pos_str")),
            "examples": [
                {
                    "tokens": decode_tokens(a["tokens"], dec_bytes),
                    "window": [0, len(a["tokens"])],
                    "firings": [v > 0 for v in a["values"]],
                    "peak": a.get("maxValue") or 0.0,
                    "series": {k: a["values"] for k in ("effective", "activation", "ci")},
                }
                for a in acts[:24]
            ],
        }
        return body

    @app.get("/api/component")
    def component(
        module: str, idx: int, sort: str = "effective", window: int = Query(10, ge=1, le=60)
    ) -> JSONResponse:
        if module not in weights.spaces and ft.is_sae_key(module):
            resolve(module, "read")  # validates the release and layer before any download
            return JSONResponse(_feature_body(module, idx))
        if module not in weights.spaces:
            raise HTTPException(404, f"{module} is not a decomposed module of this run")
        spec = weights.spaces[module]
        dead = weights.dead_counter(module)
        body: dict[str, object] = {
            "module": module,
            "idx": idx,
            "role": spec.role,
            "layer": spec.layer,
            "read_space": _space_json(spec.read),
            "write_space": _space_json(spec.write),
            "read_norm": float(weights.directions(module, "read").norms[idx]),
            "write_norm": float(weights.directions(module, "write").norms[idx]),
            "n_batches_not_active": int(dead[idx]) if dead is not None else None,
            "label": interp.label(module, idx),
            "logit_lens": _lens_json(weights.directions(module, "write").mat[idx],
                                     _space_json(spec.write)),
            "harvest": harvest.available,
        }
        if not harvest.available:
            body["reason"] = "this run has no harvest, so there are no activation examples"
            return JSONResponse(body)
        rec = harvest.component(module, idx, sort=sort, window=window)  # pyright: ignore[reportArgumentType]
        if rec is None:
            body["reason"] = f"no harvested row for {module}:{idx} (a dead component is not stored)"
            return JSONResponse(body)
        body |= {
            "firing_density": rec.firing_density,
            "mean_activations": rec.mean_activations,
            "examples": rec.examples,
            "input_pmi": rec.input_pmi,
            "output_pmi": rec.output_pmi,
            "pmi_available": rec.pmi_available,
        }
        return JSONResponse(body)

    @app.post("/api/label")
    def label(module: str, idx: int) -> JSONResponse:
        """Autointerp on demand. Refuses with the missing requirement rather than guessing."""
        missing = []
        if not harvest.available:
            missing.append("a harvest with activation examples")
        else:
            rec = harvest.component(module, idx, sort="effective", window=10)
            if rec is None:
                missing.append(f"a harvested row for {module}:{idx}")
            elif not rec.pmi_available:
                missing.append("token stats / PMI in the harvest (dropped by default since 24cb6ac)")
        if interp.path is None:
            missing.append("an --interp-db path to write the label to")
        raise HTTPException(
            501,
            "cannot generate a label: missing " + "; ".join(missing) if missing
            else "label generation is not wired up in this build",
        )

    @app.get("/api/overview")
    def overview(
        a_module: str, b_module: str, a_side: Side = "write", b_side: Side = "read",
        metric: str = "cosine", mode: str = "auto",
    ) -> JSONResponse:
        a, b, sa, sb = endpoints(a_module, a_side, b_module, b_side)
        compat = compatibility(sa, sb)
        if metric not in sc.WEIGHT_METRICS:
            raise HTTPException(
                409,
                "the overview is only defined for the weight-computed metrics: it is a per-source "
                "best/worst over EVERY partner, and under dot_coact that is a kappa-weighted "
                "product the harvest covers only for the same-layer templates",
            )
        if not (compat["flat"] or compat["per_head"]):
            raise HTTPException(409, str(compat["reason"]))
        per_head = resolve_mode(mode, compat, link_of(a, b))
        ck = (a_module, a_side, b_module, b_side, metric, per_head)
        if ck not in overview_cache:
            rec = ep.reconcile(a, b, block(a), block(b), model_name=weights.model_name, norms=norms)
            res = sc.best_match_summary(
                rec.a, rec.b,
                metric=metric, space_a=sa, space_b=sb, per_head=bool(per_head),  # pyright: ignore[reportArgumentType]
            )
            d_eff = (sa.heads.head_dim if (per_head and sa.heads) else sa.dim) - rec.d_eff_drop
            overview_cache[ck] = {**res, "d_eff": d_eff, "per_head": bool(per_head),
                                  "applied": rec.applied}
        return JSONResponse(overview_cache[ck])

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML.replace("__RUN__", json.dumps(run_name)))

    return app


_INDEX_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>component pairs</title>
<style>
:root{--bg:#ffffff;--fg:#1b1e23;--dim:#6b7280;--line:#e3e6ea;--card:#fbfcfd;--hi:#1f5fc4;
      --pos:#12703a;--neg:#c0271f;--warn:#8a5a00;--bar:#f3f5f8;--hover:#eef3fb;
      --field:#ffffff;--mask:#f2f3f5;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:10px 14px;border-bottom:1px solid var(--line);background:var(--bar);position:sticky;top:0;z-index:5}
h1{font-size:13px;margin:0 0 8px;font-weight:600}
h1 span{color:var(--dim);font-weight:400}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
label{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
select,input,button{background:var(--field);color:var(--fg);border:1px solid var(--line);border-radius:4px;
  padding:4px 6px;font:12px ui-monospace,Menlo,monospace}
button{cursor:pointer}button:hover{border-color:var(--hi)}
button:disabled{opacity:.4;cursor:not-allowed}
option:disabled{color:var(--dim)}
.note{padding:6px 14px;color:var(--dim);border-bottom:1px solid var(--line);background:var(--bar)}
.warn{color:var(--warn)}
.err{color:var(--neg)}
main{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px;min-width:0}
.card h2{font-size:12px;margin:0 0 6px;color:var(--hi);font-weight:600}
.meta{color:var(--dim);font-size:11px;margin-bottom:6px}
.badge{display:inline-block;background:var(--bar);border:1px solid var(--line);border-radius:3px;
  padding:1px 5px;margin-right:4px;font-size:11px}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:2px 5px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:400;font-size:11px}
tbody tr{cursor:pointer}tbody tr:hover{background:var(--hover)}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.ex{margin:5px 0;padding:4px;border:1px solid var(--line);border-radius:4px;overflow-x:auto;white-space:pre}
.ex .pk{color:var(--dim);font-size:11px}
.tok{padding:1px 0;border-radius:2px}
.fire{border-bottom:2px solid var(--hi)}
.pmi{display:flex;gap:14px;flex-wrap:wrap}
.pmi div{min-width:140px}
.bar{height:9px;background:var(--mask);position:relative}
.bar i{position:absolute;top:0;bottom:0;background:var(--hi)}
.hist{display:flex;align-items:flex-end;gap:1px;height:70px;margin-top:6px}
.hist i{background:var(--hi);flex:1;min-width:1px}
.rowsplit{display:grid;grid-template-columns:1fr 1fr;gap:10px}
</style></head><body>
<header>
  <h1>component pairs <span id="runname"></span>
      <a href="/prompt" style="color:var(--hi);font-weight:400;margin-left:10px">prompt trace &rarr;</a></h1>
  <div class="controls">
    <label>mode</label><select id="pairmode">
      <option value="weight">weight × weight</option>
      <option value="feature">weight × features</option>
    </select>
    <label>suggested</label><select id="tpl"></select>
  </div>
  <div class="controls" style="margin-top:6px">
    <label>A</label><select id="kindA"><option value="module">module</option><option value="sae">SAE</option></select>
    <select id="modA"></select><select id="siteA"></select><select id="relA"></select>
    <select id="layA"></select><select id="sideA"><option>write</option><option>read</option></select>
    <span style="color:var(--dim)">→</span>
    <label>B</label><select id="kindB"><option value="module">module</option><option value="sae">SAE</option></select>
    <select id="modB"></select><select id="siteB"></select><select id="relB"></select>
    <select id="layB"></select><select id="sideB"><option>read</option><option>write</option></select>
  </div>
  <div class="controls" style="margin-top:6px">
    <label>metric</label><select id="metric"></select>
    <label>mode</label><select id="mode"><option value="auto">auto</option><option value="flat">flat</option><option value="head">per-head</option></select>
    <label>head</label><select id="head"><option value="">best</option></select>
    <label>top k</label><input id="k" type="number" value="25" min="1" max="200" style="width:60px">
    <label title="Exclude partners that fire on more than this fraction of tokens, and dead ones. kappa is an unconditional mean, so it rewards density on its own.">density ≤</label>
    <select id="dens">
      <option value="5e-3">5e-3 (pipeline band)</option>
      <option value="1e-3">1e-3 (strict)</option>
      <option value="1e-2">1e-2 (loose)</option>
      <option value="0">off — rank everything</option>
    </select>
    <label>sort/heat</label><select id="sort"><option value="effective">effective (g·a)</option><option value="activation">activation (a)</option><option value="ci">gate (g)</option></select>
    <button id="ovbtn">overview</button>
  </div>
</header>
<div class="note" id="note"></div>
<main>
  <section class="card" id="cardA"></section>
  <section class="card" id="cardB"></section>
</main>
<div style="padding:0 12px 20px"><section class="card" id="ov" style="display:none"></section></div>
<script>
const $=id=>document.getElementById(id);
const RUN=__RUN__;
let META=null, PS=null, SEL={A:0,B:0};
const esc=t=>String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;');
const q=o=>Object.entries(o).filter(([,v])=>v!==null&&v!==undefined&&v!=='').map(([k,v])=>k+'='+encodeURIComponent(v)).join('&');
async function api(p,o){const r=await fetch(p+'?'+q(o));const j=await r.json().catch(()=>({detail:'bad json'}));if(!r.ok)throw new Error(j.detail||r.status);return j;}

function fill(sel,items,val){sel.innerHTML='';for(const it of items){const o=document.createElement('option');
  o.value=it.value;o.textContent=it.text;if(it.disabled)o.disabled=true;
  if(it.title)o.title=it.title;sel.appendChild(o);}
  if(val!==undefined&&items.some(i=>String(i.value)===String(val)))sel.value=val;}

function modsByRole(role){return META.modules.filter(m=>m.role===role);}
function saeList(){return META.saes||[];}
function sites(){return [...new Set(saeList().map(s=>s.site))];}
function relsFor(site){return saeList().filter(s=>s.site===site);}
function tplList(){return $('pairmode').value==='weight'?META.templates:(META.sae_templates||[]);}
function fillTpl(){fill($('tpl'),[{value:'',text:'(manual)'}].concat(
  tplList().map(t=>({value:t.key,text:t.label}))));}
function fillRel(w){fill($('rel'+w),relsFor($('site'+w).value).map(r=>({value:r.key,text:r.label})),$('rel'+w).value);}
function modLayers(){return [...new Set(META.modules.map(m=>m.layer))].sort((a,b)=>a-b);}
function fillLay(w){
  let ls;
  if($('kind'+w).value==='sae'){const r=saeList().find(x=>x.key===$('rel'+w).value);ls=r?r.layers:[];}
  else ls=modLayers();
  fill($('lay'+w),ls.map(l=>({value:l,text:'L'+l})),$('lay'+w).value);
}
function syncKind(w){
  const sae=$('kind'+w).value==='sae';
  $('mod'+w).style.display=sae?'none':'';
  $('site'+w).style.display=sae?'':'none';
  $('rel'+w).style.display=sae?'':'none';
  if(sae){if(!$('site'+w).value)$('site'+w).value=sites()[0];fillRel(w);}
  fillLay(w);
}
function onLayer(w){
  if($('kind'+w).value==='sae')return;
  const cur=META.modules.find(m=>m.module===$('mod'+w).value); if(!cur)return;
  const m=META.modules.find(m=>m.role===cur.role&&String(m.layer)===String($('lay'+w).value));
  if(m)$('mod'+w).value=m.module;
}
function epKey(w){
  return $('kind'+w).value==='sae'?$('rel'+w).value+':L'+$('lay'+w).value:$('mod'+w).value;
}

async function boot(){
  META=await api('/api/meta',{});
  $('runname').textContent='— '+META.run+' · '+META.model+(META.harvest?'':' · NO HARVEST')
    +' · '+(saeList().length?saeList().length+' SAE releases':'no SAEs configured for this model');
  for(const w of ['A','B']){
    fill($('mod'+w),META.modules.map(m=>({value:m.module,text:m.module})));
    fill($('site'+w),sites().map(x=>({value:x,text:(META.sae_sites||{})[x]||x})));
  }
  fill($('metric'),META.metrics.map(m=>({value:m.key,text:m.label})));
  if(!saeList().length){$('pairmode').disabled=true;$('kindA').disabled=true;$('kindB').disabled=true;}
  $('pairmode').onchange=()=>{fillTpl();$('tpl').value=(tplList()[0]||{}).key||'';applyTemplate();};
  $('tpl').onchange=applyTemplate;
  for(const w of ['A','B']){
    $('kind'+w).onchange=()=>{syncKind(w);refresh(true);};
    $('site'+w).onchange=()=>{fillRel(w);fillLay(w);refresh(true);};
    $('rel'+w).onchange=()=>{fillLay(w);refresh(true);};
    $('lay'+w).onchange=()=>{onLayer(w);refresh(true);};
    $('mod'+w).onchange=()=>refresh(true);
    $('side'+w).onchange=()=>refresh(true);
  }
  for(const id of ['metric','mode','head'])$(id).onchange=()=>refresh(true);
  $('k').onchange=()=>refresh(false); $('dens').onchange=()=>{drawRank('A');drawRank('B');}; $('sort').onchange=()=>{drawComp('A');drawComp('B');};
  $('ovbtn').onclick=overview;
  fillTpl();
  const t=tplList().find(t=>t.key==='mlp_in_out')||tplList()[0];
  if(t)$('tpl').value=t.key;
  for(const w of ['A','B'])syncKind(w);
  applyTemplate();
}

function setEndpoint(w,name,layer){
  const site=name.startsWith('sae:')?name.slice(4):null;
  $('kind'+w).value=site?'sae':'module';
  syncKind(w);
  if(site){$('site'+w).value=site;fillRel(w);fillLay(w);}
  if(layer!=null&&[...$('lay'+w).options].some(o=>String(o.value)===String(layer)))$('lay'+w).value=layer;
  if(!site){
    const m=META.modules.find(m=>m.role===name&&String(m.layer)===String(layer));
    if(m)$('mod'+w).value=m.module;
  }
}

function applyTemplate(){
  const t=tplList().find(x=>x.key===$('tpl').value);
  if(!t)return refresh(true);
  const isW=$('pairmode').value==='weight';
  const an=isW?t.a_role:t.a, bn=isW?t.b_role:t.b;
  const la=t.layer_mode==='same'?t.layers.same:t.layers.a;
  const lb=t.layer_mode==='same'?t.layers.same:t.layers.b;
  const pa=la.includes(+$('layA').value)?+$('layA').value:la[0];
  const pb=t.layer_mode==='same'?pa:(lb.includes(+$('layB').value)?+$('layB').value:lb[0]);
  setEndpoint('A',an,pa); setEndpoint('B',bn,pb);
  $('sideA').value=t.a_side; $('sideB').value=t.b_side;
  refresh(true);
}

function ep(){return {a_module:epKey('A'),a_side:$('sideA').value,
                      b_module:epKey('B'),b_side:$('sideB').value};}

async function refresh(reset){
  let note='';
  // A dictionary is fetched on first selection and that takes seconds; say so rather than
  // leaving the previous pair space's note and caveat on screen as though they still applied.
  $('note').innerHTML='<span class="meta">resolving pair space… '
    +'(an SAE is downloaded the first time it is selected)</span>';
  try{ PS=await api('/api/pairspace',ep()); }
  catch(e){ $('note').innerHTML='<span class="err">'+e.message+'</span>'; return; }
  const c=PS.compat;
  // METRIC_SPECS is ordered so the first available entry is the one to default to: dot_coact where
  // kappa was harvested, cosine on weight x feature where kappa cannot exist. The tooltip carries
  // the reason a disabled entry is disabled, which differs between "not harvested" and "undefined".
  fill($('metric'),PS.metrics.map(m=>({value:m.key,text:m.label+(m.available?'':' — no data'),
        disabled:!m.available,title:m.reason||m.note})),$('metric').value);
  if(!$('metric').value||($('metric').selectedOptions[0]&&$('metric').selectedOptions[0].disabled)){
    const first=PS.metrics.find(m=>m.available); if(first)$('metric').value=first.key;}
  const perHead=($('mode').value==='head')||($('mode').value==='auto'&&
      ((PS.link==='bilinear_form'&&c.per_head)||(!c.flat&&c.per_head)));
  $('head').disabled=!perHead;
  fill($('head'),[{value:'',text:perHead?('best of '+c.n_head_pairs):'no head'}].concat(
      Array.from({length:perHead?c.n_head_pairs:0},(_,i)=>({value:i,text:'head '+i}))),$('head').value);
  note='<b>'+PS.a.space.label+'</b> ('+PS.a.space.dim+'d'+(PS.a.space.n_heads?', '+PS.a.space.n_heads+'×'+PS.a.space.head_dim:'')+')'
     +' → <b>'+PS.b.space.label+'</b> ('+PS.b.space.dim+'d'+(PS.b.space.n_heads?', '+PS.b.space.n_heads+'×'+PS.b.space.head_dim:'')+')'
     +' · link: '+(PS.link_label||'<span class="warn">unclassified</span>')
     +' · flat '+(c.flat?'yes':'no')+' · per-head '+(c.per_head?c.n_head_pairs+' pairs':'no')
     +'<br><span class="warn">'+PS.caveat+'</span>';
  if(PS.basis&&PS.basis.corrections&&PS.basis.corrections.length)
    note+='<br><span class="badge" style="border-color:var(--hi)">basis</span> '
        +PS.basis.corrections.join(' · ')
        +(PS.basis.centred?' <span class="meta">(d_eff drops by 1: the all-ones direction is gone)</span>':'');
  if(!c.flat&&!c.per_head)note+='<br><span class="err">'+c.reason+'</span>';
  if(PS.link==='bilinear_form'&&!perHead&&c.flat)
    note+='<br><span class="err">flat mode on a QK pair sums the per-head logits — switch mode to per-head.</span>';
  $('note').innerHTML=note;
  if(reset){SEL.A=Math.min(SEL.A,PS.a.n_components-1);SEL.B=Math.min(SEL.B,PS.b.n_components-1);}
  drawComp('A');drawComp('B');
}

function heat(v,max,neg){const t=Math.min(1,Math.abs(v)/(max||1));
  return v>=0?`rgba(31,95,196,${t*0.55})`:`rgba(192,39,31,${t*0.55})`;}

function exHtml(ex,key){
  const s=ex.series[key], [lo,hi]=ex.window;
  const max=Math.max(...s.map(Math.abs),1e-9);
  let h='<div class="ex"><span class="pk">peak '+ex.peak.toFixed(3)+'</span>  ';
  for(let i=lo;i<hi;i++){
    const t=ex.tokens[i].replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/\n/g,'⏎');
    h+='<span class="tok'+(ex.firings[i]?' fire':'')+'" title="'+s[i].toFixed(4)+'" style="background:'+heat(s[i],max)+'">'+t+'</span>';
  }
  return h+'</div>';
}

function pmiHtml(rows,title){
  if(!rows||!rows.length)return '';
  return '<div><b>'+title+'</b><br>'+rows.slice(0,12).map(([t,v])=>
    '<span class="badge">'+String(t).replace(/</g,'&lt;')+' '+v.toFixed(2)+'</span>').join('')+'</div>';
}

// u_c through the final norm and the unembedding. Shape is the only gate: a write side that is not
// d_model wide says so in `reason`, everything else has one.
function lensHtml(l){
  if(!l)return '';
  if(!l.available)return '<div class="warn">no logit lens — '+String(l.reason||'unavailable').replace(/</g,'&lt;')+'</div>';
  const row=(rows,title)=>'<div><b>'+title+'</b><br>'+rows.map(([t,v])=>
    '<span class="badge">'+JSON.stringify(t).slice(1,-1).replace(/</g,'&lt;')+' '+v.toFixed(2)+'</span>').join('')+'</div>';
  return '<div class="pmi">'+row(l.promoted,'logit lens ↑')+row(l.suppressed,'logit lens ↓')+'</div>'
   + (l.in_residual_stream?'':'<div class="warn">this write side is not the residual stream — the '
     + 'arithmetic is defined but the model never carries this vector to <code>ln_f</code> without '
     + 'putting it through the attention machinery first</div>');
}

async function drawComp(which){
  const side=which==='A'?PS.a:PS.b, el=$('card'+which);
  const n=side.n_components;
  el.innerHTML='<h2>'+which+' · '+side.module+' · '+side.side+'</h2><div class="meta">loading…</div>';
  let d,h;
  try{ d=await api('/api/component',{module:side.module,idx:SEL[which],sort:$('sort').value,window:12}); }
  catch(e){ el.innerHTML='<h2>'+which+'</h2><div class="err">'+e.message+'</div>'; return; }
  const isSae=d.kind==='sae';
  h='<h2>'+which+' · '+(side.label||side.module)+' · <span style="color:var(--fg)">side '+side.side+'</span></h2>';
  h+='<div class="controls" style="margin-bottom:6px">'
   + '<button data-nav="'+which+'" data-d="-1">◀</button>'
   + '<input id="idx'+which+'" type="number" value="'+SEL[which]+'" min="0" max="'+(n-1)+'" style="width:80px">'
   + '<button data-nav="'+which+'" data-d="1">▶</button>'
   + '<span class="meta">of '+n+(isSae?' features':' components')+'</span>'
   + (isSae?'':'<button data-label="'+which+'">autointerp</button>')
   + (isSae&&d.neuronpedia_url?'<a href="'+d.neuronpedia_url+'" target="_blank" style="color:var(--hi)">neuronpedia ↗</a>':'')
   + '</div>';
  h+='<div class="meta">';
  if(d.label)h+='<span class="badge" style="border-color:var(--hi)">'+esc(d.label.label)+'</span>';
  if(d.firing_density!=null)h+='density '+(d.firing_density*100).toFixed(3)+'% · ';
  h+=(isSae?'|enc| ':'|read| ')+d.read_norm.toFixed(3)+' · '+(isSae?'|dec| ':'|write| ')+d.write_norm.toFixed(3);
  if(d.enc_dec_cosine!=null)h+=' · enc·dec '+d.enc_dec_cosine.toFixed(3);
  if(d.max_activation!=null)h+=' · max act '+d.max_activation.toFixed(2);
  if(d.centred)h+=' · <span class="warn">centred basis</span>';
  if(d.n_batches_not_active!=null)h+=' · dead-clock '+d.n_batches_not_active;
  h+='</div>';
  if(d.explanations&&d.explanations.length>1)
    h+='<div class="meta">'+d.explanations.slice(1,3).map(x=>'· '+esc(x)).join('<br>')+'</div>';
  if(d.reason)h+='<div class="warn">'+d.reason+'</div>';
  if(d.pmi_available===false)h+='<div class="warn">no token PMI in this harvest</div>';
  h+=lensHtml(d.logit_lens);
  if(d.input_pmi||d.output_pmi)h+='<div class="pmi">'+pmiHtml(d.input_pmi,'input PMI')+pmiHtml(d.output_pmi,'output PMI')+'</div>';
  if(d.explanations_dropped&&d.explanations_dropped.length)
    h+='<div class="meta">'+d.explanations_dropped.length+' duplicate autointerp description'
      +(d.explanations_dropped.length>1?'s':'')+' hidden</div>';
  h+='<div id="rank'+which+'"><div class="meta">ranking…</div></div>';
  if(d.examples&&d.examples.length){
    h+='<div class="meta" style="margin-top:8px">'+d.examples.length+' examples, sorted by '+$('sort').value+'</div>';
    h+=d.examples.slice(0,12).map(e=>exHtml(e,$('sort').value)).join('');
  }
  el.innerHTML=h;
  el.querySelectorAll('[data-nav]').forEach(b=>b.onclick=()=>{
    SEL[which]=Math.max(0,Math.min(n-1,SEL[which]+ +b.dataset.d));drawComp(which);});
  $('idx'+which).onchange=e=>{SEL[which]=Math.max(0,Math.min(n-1,+e.target.value));drawComp(which);};
  const lb=el.querySelector('[data-label]');
  if(lb)lb.onclick=async()=>{
    const r=await fetch('/api/label?'+q({module:side.module,idx:SEL[which]}),{method:'POST'});
    const j=await r.json().catch(()=>({detail:'bad json'}));
    alert(r.ok?JSON.stringify(j):('cannot generate: '+j.detail));};
  drawRank(which);
}

async function drawRank(which){
  const from=which, to=which==='A'?'B':'A';
  const e=ep(), args=(from==='A')?e:{a_module:e.b_module,a_side:e.b_side,b_module:e.a_module,b_side:e.a_side};
  const el=$('rank'+which); if(!el)return;
  let r;
  try{ r=await api('/api/rank',{...args,idx:SEL[from],metric:$('metric').value,
        mode:$('mode').value,head:$('head').value,k:$('k').value,
        max_density:$('dens').value}); }
  catch(err){ el.innerHTML='<div class="err">'+err.message+'</div>'; return; }
  // dot_coact is a product with kappa ~1e-5, so a fixed 4-decimal format prints every score as
  // 0.0000. Pick the format from the magnitude actually present rather than per metric.
  const big=[...r.top,...r.bottom].some(x=>Math.abs(x.score)>=1e-3);
  const fmt=v=>big?v.toFixed(4):v.toExponential(2);
  const rows=l=>l.map(x=>'<tr data-to="'+to+'" data-i="'+x.idx+'"><td>'+x.idx+'</td>'
    +'<td class="'+(x.score>=0?'pos':'neg')+'">'+fmt(x.score)+'</td>'
    +'<td>'+(x.z_theory!=null?x.z_theory.toFixed(1)+'σ':'—')+'</td>'
    +'<td>'+(x.z_empirical!=null?x.z_empirical.toFixed(1):'—')+'</td>'
    +'<td>'+(x.head!==undefined?'h'+x.head:'')+'</td></tr>').join('');
  const hd='<tr><th>'+to+' idx</th><th>score</th><th>z(null)</th><th>z(row)</th><th>head</th></tr>';
  let foot='<div class="meta" style="margin-top:8px">partners in '+to+' · '+r.source
    +(r.per_head?' · per-head (d='+r.d_eff+')':' · flat (d='+r.d_eff+')')
    +' · row mean '+ (r.row_mean!=null?fmt(r.row_mean):'—')
    +' sd '+(r.row_std!=null?fmt(r.row_std):'—');
  // Coverage is stated only when something was actually left out; on a full-pool kappa harvest
  // there is nothing to caveat and a permanent "6144 of 6144" line is noise.
  if(r.n_covered!=null&&r.n_components!=null&&r.n_covered<r.n_components){
    const why=r.n_dense_excluded?(r.n_dense_excluded+' too dense or dead'
        +(r.n_covered+r.n_dense_excluded<r.n_components?', the rest unmeasured':'')):'no kappa';
    foot+=' · <span class="warn">ranked over '+r.n_covered+' of '+r.n_components
        +' ('+why+' — excluded, not scored 0)</span>';}
  if(r.density_note)foot+=' · <span class="warn">'+r.density_note+'</span>';
  foot+='</div>';
  if(r.shared_gate)foot+='<div class="meta"><span class="badge" style="border-color:var(--warn)">'
    +'shared encoder</span> both modules read one gate on this arm, so kappa here measures gate '
    +'sharing as much as composition.</div>';
  el.innerHTML=foot
    +'<div class="rowsplit"><div><table>'+hd+rows(r.top)+'</table></div>'
    +'<div><table>'+hd+rows(r.bottom)+'</table></div></div>';
  el.querySelectorAll('tr[data-i]').forEach(tr=>tr.onclick=()=>{
    SEL[tr.dataset.to]=+tr.dataset.i; drawComp(tr.dataset.to);});
}

async function overview(){
  const el=$('ov'); el.style.display='block'; el.innerHTML='<h2>overview</h2><div class="meta">computing…</div>';
  let d;
  try{ d=await api('/api/overview',{...ep(),metric:$('metric').value,mode:$('mode').value}); }
  catch(e){ el.innerHTML='<h2>overview</h2><div class="err">'+e.message+'</div>'; return; }
  const bins=60, lo=Math.min(...d.worst), hi=Math.max(...d.best);
  const mk=(arr,title,col)=>{const h=new Array(bins).fill(0);
    for(const v of arr)h[Math.max(0,Math.min(bins-1,Math.floor((v-lo)/(hi-lo)*bins)))]++;
    const mx=Math.max(...h);
    return '<div><b>'+title+'</b><div class="hist">'+h.map(c=>'<i style="height:'+(100*c/mx)+'%;background:'+col+'"></i>').join('')+'</div>'
      +'<div class="meta">'+lo.toFixed(3)+' … '+hi.toFixed(3)+'</div></div>';};
  const mean=a=>a.reduce((x,y)=>x+y,0)/a.length, srt=[...d.best].sort((a,b)=>a-b);
  el.innerHTML='<h2>best / worst match per component of A (d_eff '+d.d_eff+', null sd '+(1/Math.sqrt(d.d_eff)).toFixed(4)+')</h2>'
   +'<div class="rowsplit">'+mk(d.best,'best match','var(--pos)')+mk(d.worst,'worst match','var(--neg)')+'</div>'
   +'<div class="meta">best: mean '+mean(d.best).toFixed(4)+' · median '+srt[srt.length>>1].toFixed(4)
   +' · p90 '+srt[Math.floor(srt.length*0.9)].toFixed(4)+' · max '+srt[srt.length-1].toFixed(4)
   +' · max in null σ: '+(srt[srt.length-1]*Math.sqrt(d.d_eff)).toFixed(1)+'</div>';
}
boot();
</script></body></html>"""
