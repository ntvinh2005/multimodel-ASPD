"""Harvest the co-activation term of the component interaction score.

For every module pair the pair viewer ranks, accumulate kappa(c1, c2) = E_{X,t}[g_{t,c2} e_{t,c1}]
over a corpus, so Interact(c1, c2) = kappa(c1, c2) <u_c1, v_c2> can be read at query time.
"""

import argparse
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor

from aspd.analysis.pairs.spaces import parse_role
from aspd.analysis.pairs.suggest import TEMPLATES


def eligible_indices(harvest_db: Path, module: str, band: tuple[float, float], per_module: int) -> Tensor:
    """Up to `per_module` component indices of `module` that fire inside the density `band`."""
    con = sqlite3.connect(f"file:{harvest_db}?mode=ro", uri=True, timeout=5.0)
    with con:
        rows = con.execute(
            "SELECT component_key, firing_density FROM components WHERE component_key LIKE ?",
            (f"{module}:%",),
        ).fetchall()
    assert rows, f"harvest has no component keyed `{module}:<idx>`"
    keep = [int(k.rsplit(":", 1)[1]) for k, d in rows if band[0] <= d <= band[1]]
    assert keep, f"{module}: no component inside density band {band}"
    return torch.tensor(sorted(keep)[:per_module], dtype=torch.long)


def pair_key(src: str, dst: str) -> str:
    return f"{src}|{dst}"


def resid_sites(run_dir: Path) -> dict[str, str]:
    """`module -> gate site`, empty on a per-module-gate arm where every module has its own."""
    import yaml

    cfg = yaml.safe_load((run_dir / "experiment_config.yaml").read_text())
    return dict(cfg["pd"]["ci_config"].get("resid_sites") or {})


def template_pairs(modules: list[str]) -> dict[str, dict[str, object]]:
    """`{"<src>|<dst>": {"src_module", "dst_module", "templates"}}` -- the SAME-LAYER templates only."""
    by_role: dict[tuple[str, int], str] = {}
    for m in modules:
        role, layer = parse_role(m)
        by_role[(role, layer)] = m
    layers = sorted({layer for _, layer in by_role})

    out: dict[str, dict[str, object]] = {}
    for t in TEMPLATES:
        if t.layer_mode != "same":
            continue
        for L in layers:
            src, dst = by_role.get((t.a_role, L)), by_role.get((t.b_role, L))
            if src is None or dst is None:
                continue
            entry = out.setdefault(
                pair_key(src, dst),
                {"src_module": src, "dst_module": dst, "templates": []},
            )
            entry["templates"].append(t.key)  # pyright: ignore[reportAttributeAccessIssue]
    return out


def accumulate(harvest_fn, dataloader, n_batches: int, pairs, index, device):
    """`(K, n_co, n_tok, seen)` with `K[key] = sum_x (g_src a_src) (x) g_dst`."""
    K = {k: torch.zeros(len(index[str(p["src_module"])]), len(index[str(p["dst_module"])]),
                        dtype=torch.float64, device=device) for k, p in pairs.items()}
    n_co = {k: torch.zeros_like(v) for k, v in K.items()}
    by_layer: dict[int, list[str]] = {}
    for k, p in pairs.items():
        by_layer.setdefault(parse_role(str(p["src_module"]))[1], []).append(k)
    mods = sorted({str(p[s]) for p in pairs.values() for s in ("src_module", "dst_module")})
    idx = {m: index[m].to(device) for m in mods}
    full = {m: int(index[m].numel()) == int(index[m][-1]) + 1 for m in mods}

    def gather(out, m: str):
        """`(g*a, g, firings)` as fp64 `[T, C]`. Slicing is skipped when the pool is everything."""
        si = idx[m]
        acts = out.activations[m]
        g = acts["causal_importance"]
        a = acts["component_activation"]
        f = out.firings[m]
        if not full[m]:
            g, a, f = g[..., si], a[..., si], f[..., si]
        n = g.shape[-1]
        g = g.reshape(-1, n).double()
        a = a.reshape(-1, n).double()
        return g * a, g, f.reshape(-1, n).double()

    n_tok, seen = 0, 0
    t0 = time.time()
    it = iter(dataloader)
    for i in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            print(f"[appcoact] dataset exhausted after {i} batches", flush=True)
            break
        out = harvest_fn(batch)
        counted = False
        for L, keys in by_layer.items():
            layer_mods = sorted({str(pairs[k][s]) for k in keys
                                 for s in ("src_module", "dst_module")})
            blocks = {m: gather(out, m) for m in layer_mods}
            for k in keys:
                sm, dm = str(pairs[k]["src_module"]), str(pairs[k]["dst_module"])
                K[k] += blocks[sm][0].t() @ blocks[dm][1]
                n_co[k] += blocks[sm][2].t() @ blocks[dm][2]
            if not counted:
                n_tok += next(iter(blocks.values()))[0].shape[0]
                counted = True
            del blocks
        seen += 1
        if (i + 1) % 50 == 0 or i + 1 == n_batches:
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            print(f"[appcoact] {i + 1}/{n_batches} batches  {dt / (i + 1):.2f} s/batch  "
                  f"eta {(n_batches - i - 1) * dt / (i + 1) / 60:.0f} min  peak {peak:.1f} GiB",
                  flush=True)
    assert seen > 0, "no batches processed"
    return K, n_co, n_tok, seen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--n-batches", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--per-module", type=int, default=0,
                    help="0 (default) takes EVERY component; a positive value keeps that many "
                         "per module from inside --density-band, for a cheap smoke run")
    ap.add_argument("--density-band", type=float, nargs=2, default=(0.0, 5e-3))
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to <run-dir>/harvest/pair_coactivation.pt")
    args = ap.parse_args()

    from aspd.lab_compat import widen_lab_config_parsing
    from aspd.cli.harvest import _local_adapter, downstream_id_for
    from param_decomp_lab.harvest.config import ParamDecompHarvestConfig
    from param_decomp_lab.harvest.harvest_fn import make_harvest_fn

    widen_lab_config_parsing()
    run_dir = args.run_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = _local_adapter(run_dir, run_dir.name, args.step)
    modules = sorted(dict(adapter.layer_activation_sizes))

    harvests = sorted((run_dir / "harvest").glob("h-*/harvest.db"), reverse=True)
    assert harvests, f"no harvest under {run_dir}/harvest"
    harvest_db = harvests[0]

    band = (args.density_band[0], args.density_band[1])
    pairs = template_pairs(modules)
    assert pairs, f"no template expands against the {len(modules)} modules this run decomposed"
    sites = resid_sites(run_dir)
    for p in pairs.values():
        s_site, d_site = sites.get(str(p["src_module"])), sites.get(str(p["dst_module"]))
        p["shared_gate"] = bool(s_site) and s_site == d_site
    n_shared = sum(bool(p["shared_gate"]) for p in pairs.values())
    used = sorted({str(p[s]) for p in pairs.values() for s in ("src_module", "dst_module")})
    sizes = dict(adapter.layer_activation_sizes)
    index = {
        m: (torch.arange(sizes[m], dtype=torch.long) if args.per_module <= 0
            else eligible_indices(harvest_db, m, band, args.per_module))
        for m in used
    }
    cells = sum(len(index[str(p["src_module"])]) * len(index[str(p["dst_module"])])
                for p in pairs.values())
    print(f"[appcoact] run={run_dir.name} ckpt={adapter.pd_run.checkpoint_path.name} "
          f"harvest={harvest_db.parent.name} dev={device}",
          flush=True)
    pool = "every component" if args.per_module <= 0 else f"top {args.per_module}/module in {band}"
    print(f"[appcoact] {len(pairs)} module pairs over {len(used)} modules, {cells:,} pair cells "
          f"({pool}); accumulators {cells * 16 / 2**30:.1f} GiB", flush=True)
    print(f"[appcoact] gate sites: {len(set(sites.values())) or 'per module'}; "
          f"{n_shared} pairs share an encoder (kappa there measures gate sharing, not composition)",
          flush=True)

    method = ParamDecompHarvestConfig(wandb_path=downstream_id_for(run_dir.name),
                                      activation_threshold=0.0)
    harvest_fn = make_harvest_fn(device, method, adapter)
    with torch.no_grad():
        K, n_co, n_tok, seen = accumulate(
            harvest_fn, adapter.dataloader(args.batch_size), args.n_batches, pairs, index, device)

    out = args.out or (run_dir / "harvest" / "pair_coactivation.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kappa": {k: (v / n_tok).cpu() for k, v in K.items()},
        "n_co": {k: v.cpu() for k, v in n_co.items()},
        "index": {m: index[m] for m in used},
        "pairs": {k: {"src_module": p["src_module"], "dst_module": p["dst_module"],
                      "templates": sorted(set(p["templates"])),  # pyright: ignore[reportArgumentType]
                      "shared_gate": bool(p["shared_gate"])}
                  for k, p in pairs.items()},
        "meta": {"run_dir": str(run_dir), "checkpoint": adapter.pd_run.checkpoint_path.name,
                 "harvest": harvest_db.parent.name,
                 "n_batches": seen, "batch_size": args.batch_size, "n_tokens": n_tok,
                 "density_band": list(band), "per_module": args.per_module,
                 "full_pool": args.per_module <= 0,
                 "n_pairs": len(pairs), "n_modules": len(used),
                 "resid_sites": len(set(sites.values())), "n_shared_gate_pairs": n_shared,
                 "kappa": "E_x[g_c(x) g_c'(x) a_c(x)], unconditional over all tokens, src -> dst",
                 "n_co": "tokens where both gates are open",
                 "created": datetime.now().isoformat()},
    }, out)

    g = torch.Generator(device="cpu").manual_seed(0)
    sample = torch.cat([
        (K[key] / n_tok).abs().flatten().cpu()[
            torch.randint(K[key].numel(), (2 ** 20 // len(K),), generator=g)]
        for key in K
    ])
    q = torch.quantile(sample.float(), torch.tensor([0.5, 0.99]))
    never = sum(float((n_co[key] == 0).sum()) for key in n_co) / sum(n_co[key].numel() for key in n_co)
    print(f"[appcoact] |kappa| p50 {q[0]:.3e} p99 {q[1]:.3e} (over {sample.numel():,} sampled cells)"
          f"  pairs that never co-fire: {never:.1%}", flush=True)
    print(f"[appcoact] {n_tok:,} tokens; wrote {out} "
          f"({out.stat().st_size / 2**30:.1f} GiB)", flush=True)


if __name__ == "__main__":
    main()
