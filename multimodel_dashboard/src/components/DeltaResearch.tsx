import { useEffect, useState } from "react";
import { api } from "../api";
import type { DeltaCandidatesPayload, DeltaOverviewPayload } from "../types";

export default function DeltaResearch({ mode }: { mode: "overview" | "features" | "comparison" }) {
  const [overview, setOverview] = useState<DeltaOverviewPayload | null>(null);
  const [rows, setRows] = useState<DeltaCandidatesPayload | null>(null);
  const [partition, setPartition] = useState("");
  useEffect(() => { api.deltaOverview().then(setOverview).catch((e) => setOverview({ available: false, diagnostics: [String(e)] })); }, []);
  useEffect(() => { if (mode === "features") api.deltaCandidates(partition || undefined).then(setRows).catch((e) => setRows({ available: false, diagnostics: [String(e)], items: [] })); }, [mode, partition]);
  if (!overview) return <div className="loading">Loading Delta-Crosscoder artifacts…</div>;
  if (!overview.available) return <section className="empty-state"><h2>Delta positive control unavailable</h2><p>Real Delta artifacts are required; no demo values are shown.</p>{overview.diagnostics.map((d) => <p key={d}>{d}</p>)}</section>;
  if (mode === "comparison") return <section className="panel"><h2>Delta ↔ ASPD comparison</h2><p>Correspondence is measured only after the exact-token-grid artifact is present.</p></section>;
  const metrics = overview.metrics ?? {};
  const exact = (metrics.validation_modes as Record<string, Record<string, unknown>> | undefined)?.exact_batchtopk ?? {};
  return <section className="panel"><h2>{mode === "overview" ? "Delta-Crosscoder Overview" : "Delta Feature Explorer"}</h2>
    <p>Reduced-scale positive control · {overview.n_features ?? "?"} components · partitions: {JSON.stringify(overview.partition_counts ?? {})}</p>
    {mode === "overview" && <div className="metric-grid"><div><span>FVE total</span><strong>{typeof exact.fve_total === "number" ? exact.fve_total.toFixed(5) : "—"}</strong></div><div><span>Delta FVE</span><strong>{typeof exact.delta_fve === "number" ? exact.delta_fve.toFixed(5) : "—"}</strong></div><div><span>Exact L0</span><strong>{typeof exact.l0_total === "number" ? exact.l0_total.toFixed(0) : "—"}</strong></div><div><span>Dead fraction</span><strong>{typeof exact.dead_fraction_total === "number" ? `${(exact.dead_fraction_total * 100).toFixed(2)}%` : "—"}</strong></div></div>}
    {mode === "features" && <><p className="muted">Orientation: R_base near 0 = target-heavy; R_base near 1 = base-heavy.</p><label>Partition <select value={partition} onChange={(e) => setPartition(e.target.value)}><option value="">all</option><option value="shared">shared</option><option value="delta">delta</option></select></label>{rows?.items.map((row) => <div className="table-row" key={row.feature_id}><strong>#{row.feature_id}</strong> {row.partition} · fire {row.validation_activation_count ?? row.fire_count ?? 0} · R_base {row.relative_decoder_norm_base?.toFixed(3) ?? "—"} · {String(row.heldout_generalization_status ?? "measured stats pending")}</div>)}</>}
  </section>;
}
