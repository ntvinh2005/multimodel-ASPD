import { useMemo, useState } from "react";
import type { FeatureSummary, Metadata, TaxonomyItem } from "../types";
import { Badge, Metric, Section } from "./Common";

interface Filters {
  category: string;
  activationMin: string; activationMax: string; mechanismMin: string; mechanismMax: string;
  betaMin: string; fireMin: string; componentMin: string; componentMax: string;
  readMin: string; readMax: string; writeMin: string; writeMax: string;
  relativeMin: string; relativeMax: string; locus: string;
}

const initial: Filters = {
  category: "shared_activation_concentrated_mechanism", activationMin: "", activationMax: "",
  mechanismMin: "", mechanismMax: "", betaMin: "", fireMin: "", componentMin: "",
  componentMax: "", readMin: "", readMax: "", writeMin: "", writeMax: "",
  relativeMin: "", relativeMax: "", locus: "",
};

export default function CandidateFinder({
  points, metadata, taxonomy, presetCategory, selected, onFeature, onToggleSelected,
}: {
  points: FeatureSummary[]; metadata: Metadata; taxonomy: TaxonomyItem[]; presetCategory: string;
  selected: number[]; onFeature: (id: number) => void; onToggleSelected: (id: number) => void;
}) {
  const [filters, setFilters] = useState<Filters>({ ...initial, category: presetCategory || initial.category });
  const [sort, setSort] = useState("beta_total");
  const [descending, setDescending] = useState(true);
  const number = (raw: string) => raw === "" ? null : Number(raw);
  const between = (value: number | null, lo: string, hi: string) => (number(lo) === null || (value !== null && value >= number(lo)!)) && (number(hi) === null || (value !== null && value <= number(hi)!));
  const filtered = useMemo(() => points.filter((row) => {
    const a = row.activation_rho?.[0] ?? null, m = row.mechanism_rho?.[0] ?? null;
    return (!filters.category || row.taxonomy_categories.includes(filters.category)) && between(a, filters.activationMin, filters.activationMax)
      && between(m, filters.mechanismMin, filters.mechanismMax) && (number(filters.betaMin) === null || (row.beta_total !== null && row.beta_total >= number(filters.betaMin)!))
      && (number(filters.fireMin) === null || (row.fire_count !== null && row.fire_count >= number(filters.fireMin)!))
      && between(row.p5_min_component_cosine, filters.componentMin, filters.componentMax)
      && between(row.p5_min_read_cosine, filters.readMin, filters.readMax)
      && between(row.p5_min_write_cosine, filters.writeMin, filters.writeMax)
      && between(row.p5_max_relative_change, filters.relativeMin, filters.relativeMax)
      && (!filters.locus || Object.values(row.dominant_locus).includes(filters.locus));
  }).sort((a, b) => {
    const av = a[sort as keyof FeatureSummary], bv = b[sort as keyof FeatureSummary];
    const numeric = typeof av === "number" && typeof bv === "number" ? av - bv : String(av ?? "").localeCompare(String(bv ?? ""));
    return descending ? -numeric : numeric;
  }), [points, filters, sort, descending]);

  function update(key: keyof Filters, value: string) { setFilters({ ...filters, [key]: value }); }
  function exportCsv() {
    const headers = ["feature_id", "taxonomy", "activation_rho_base", "mechanism_rho_base", "beta_total", "fire_count", "rho_gap", "locus_shift", "p5_min_component_cosine", "p5_min_read_cosine", "p5_min_write_cosine", "p5_max_relative_change"];
    const lines = [headers.join(","), ...filtered.map((row) => [row.feature_id, row.taxonomy, row.activation_rho?.[0], row.mechanism_rho?.[0], row.beta_total, row.fire_count, row.rho_gap, row.locus_shift, row.p5_min_component_cosine, row.p5_min_read_cosine, row.p5_min_write_cosine, row.p5_max_relative_change].map((v) => JSON.stringify(v ?? "")).join(","))];
    const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" })); link.download = "filtered_aspd_candidates.csv"; link.click(); URL.revokeObjectURL(link.href);
  }

  return <div className="page-stack">
    <Section title="Candidate Finder" kicker="Transparent filtering — no hidden interestingness score">
      <div className="filter-grid">
        <label>Taxonomy<select value={filters.category} onChange={(e) => update("category", e.target.value)}><option value="">All categories</option>{taxonomy.map((x) => <option key={x.category}>{x.category}</option>)}</select></label>
        <Range label="Activation rho — base" lo={filters.activationMin} hi={filters.activationMax} setLo={(v) => update("activationMin", v)} setHi={(v) => update("activationMax", v)} />
        <Range label="Mechanism rho — base" lo={filters.mechanismMin} hi={filters.mechanismMax} setLo={(v) => update("mechanismMin", v)} setHi={(v) => update("mechanismMax", v)} />
        <label>beta_total minimum<input type="number" step="any" value={filters.betaMin} onChange={(e) => update("betaMin", e.target.value)} /></label>
        <label>fire_count minimum<input type="number" value={filters.fireMin} onChange={(e) => update("fireMin", e.target.value)} /></label>
        <label>Locus matrix<select value={filters.locus} onChange={(e) => update("locus", e.target.value)}><option value="">Any dominant locus</option>{metadata.matrix_names.map((x) => <option key={x}>{x}</option>)}</select></label>
      </div>
      <details className="advanced"><summary>Advanced P5 filters</summary><div className="filter-grid">
        <Range label="Minimum component cosine across matrices" lo={filters.componentMin} hi={filters.componentMax} setLo={(v) => update("componentMin", v)} setHi={(v) => update("componentMax", v)} />
        <Range label="Minimum read cosine across matrices" lo={filters.readMin} hi={filters.readMax} setLo={(v) => update("readMin", v)} setHi={(v) => update("readMax", v)} />
        <Range label="Minimum write cosine across matrices" lo={filters.writeMin} hi={filters.writeMax} setLo={(v) => update("writeMin", v)} setHi={(v) => update("writeMax", v)} />
        <Range label="Maximum relative change across matrices" lo={filters.relativeMin} hi={filters.relativeMax} setLo={(v) => update("relativeMin", v)} setHi={(v) => update("relativeMax", v)} />
      </div></details>
      <div className="toolbar"><strong>{filtered.length.toLocaleString()} components</strong><label>Sort<select value={sort} onChange={(e) => setSort(e.target.value)}><option value="beta_total">beta total</option><option value="fire_count">fire count</option><option value="rho_gap">rho gap</option><option value="locus_shift">locus shift</option><option value="p5_min_component_cosine">min component cosine</option><option value="p5_max_relative_change">max relative change</option><option value="feature_id">feature ID</option></select></label><button onClick={() => setDescending(!descending)}>{descending ? "Descending" : "Ascending"}</button><button onClick={exportCsv}>Export filtered CSV</button>{selected.length > 0 && <a className="button-link" href={`/api/export/selected.json?${selected.map((id) => `ids=${id}`).join("&")}`}>Export selected JSON</a>}</div>
    </Section>
    <div className="table-scroll candidate-table"><table><thead><tr><th>Compare</th><th>Feature</th><th>Taxonomy</th><th>activation ρ base / FT</th><th>β base / FT / total</th><th>mechanism ρ base / FT</th><th>Fires</th><th>Dominant loci</th><th>ρ gap</th><th>Locus L1 shift</th><th>Min comp cos</th><th>Min read / write</th><th>Max rel change</th></tr></thead>
      <tbody>{filtered.slice(0, 1000).map((row) => <tr key={row.feature_id}><td><input type="checkbox" checked={selected.includes(row.feature_id)} onChange={() => onToggleSelected(row.feature_id)} /></td><td><button className="id-button" onClick={() => onFeature(row.feature_id)}>{row.feature_id}</button>{row.low_support && <Badge tone="warn">low support</Badge>}</td><td>{row.taxonomy?.replaceAll("_", " ") || "N/A"}</td><td>{row.activation_rho?.map((v) => v.toFixed(4)).join(" / ") || "N/A"}</td><td>{row.beta?.map((v) => v.toExponential(3)).join(" / ") || "N/A"} / <Metric value={row.beta_total} kind="beta" /></td><td>{row.mechanism_rho?.map((v) => v.toFixed(4)).join(" / ") || "N/A"}</td><td><Metric value={row.fire_count} kind="integer" /></td><td>{metadata.model_names.map((m) => row.dominant_locus[m] || "N/A").join(" → ")}</td><td><Metric value={row.rho_gap} /></td><td><Metric value={row.locus_shift} /></td><td><Metric value={row.p5_min_component_cosine} /></td><td><Metric value={row.p5_min_read_cosine} /> / <Metric value={row.p5_min_write_cosine} /></td><td><Metric value={row.p5_max_relative_change} /></td></tr>)}</tbody></table>
      {filtered.length > 1000 && <div className="table-note">Showing first 1,000 filtered rows. Export includes all {filtered.length.toLocaleString()} rows.</div>}</div>
    <div className="formula-panel"><strong>Derived columns</strong><code>rho_gap = |activation_rho_base − mechanism_rho_base|</code><code>beta_total = Σₙ betaₙ</code><code>locus_shift = Σⱼ |π_base(j) − π_finetuned(j)|</code><p>P5 summaries are extrema over real per-matrix values, not a weighted “interestingness” score.</p><small>Sources per row: <code>activation_rho_NC[:,feature_id]</code>, <code>mechanism_beta_NC[:,feature_id]</code>, <code>mechanism_rho_NC[:,feature_id]</code>, <code>fire_count_C[feature_id]</code>, and discovered <code>locus/…</code>/<code>pair/…</code> tensors.</small></div>
  </div>;
}

function Range({ label, lo, hi, setLo, setHi }: { label: string; lo: string; hi: string; setLo: (v: string) => void; setHi: (v: string) => void }) {
  return <label>{label}<span className="range-inputs"><input type="number" step="any" placeholder="min" value={lo} onChange={(e) => setLo(e.target.value)} /><input type="number" step="any" placeholder="max" value={hi} onChange={(e) => setHi(e.target.value)} /></span></label>;
}
