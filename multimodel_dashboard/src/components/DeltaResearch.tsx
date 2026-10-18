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
  return <section className="panel"><h2>{mode === "overview" ? "Delta-Crosscoder Overview" : "Delta Feature Explorer"}</h2><p>{overview.n_features ?? "?"} components · partitions: {JSON.stringify(overview.partition_counts ?? {})}</p>{mode === "features" && <><label>Partition <select value={partition} onChange={(e) => setPartition(e.target.value)}><option value="">all</option><option value="shared">shared</option><option value="delta">delta</option></select></label>{rows?.items.map((row) => <div className="table-row" key={row.feature_id}><strong>#{row.feature_id}</strong> {row.partition} · fire {row.validation_activation_count ?? row.fire_count ?? 0} · base norm {row.relative_decoder_norm_base?.toFixed(3) ?? "—"}</div>)}</>}</section>;
}
