import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { FeatureDetail, FeatureSummary, Metadata, P5Row, ResearcherNote, TaxonomyItem } from "../types";
import { Badge, Metric, ModelBars, Section, Source } from "./Common";

const emptyNote: ResearcherNote = {
  tentative_label: "", notes: "", semantic_evidence: "", alternative_interpretation: "", confidence: "low",
  why_interesting: "", mentor_notes: "", candidate_status: "unreviewed",
};

export default function FeatureExplorer({
  featureId,
  metadata,
  points,
  taxonomy,
  bookmarked,
  onNavigate,
  onBookmark,
  onCompare,
}: {
  featureId: number;
  metadata: Metadata;
  points: FeatureSummary[];
  taxonomy: TaxonomyItem[];
  bookmarked: boolean;
  onNavigate: (id: number) => void;
  onBookmark: (id: number) => void;
  onCompare: (id: number) => void;
}) {
  const [feature, setFeature] = useState<FeatureDetail | null>(null);
  const [error, setError] = useState("");
  const [note, setNote] = useState<ResearcherNote>(emptyNote);
  const [saved, setSaved] = useState("");
  const [p5Sort, setP5Sort] = useState<keyof P5Row>("matrix");
  const [control, setControl] = useState<Record<string, unknown> | null>(null);
  const [matchAttributes, setMatchAttributes] = useState(["fire_count", "beta_total", "activation_rho"]);

  useEffect(() => {
    setFeature(null); setError(""); setControl(null);
    api.feature(featureId).then((value) => { setFeature(value); setNote(value.note || emptyNote); }).catch((e: Error) => setError(e.message));
  }, [featureId]);

  const p5 = useMemo(() => [...(feature?.p5 || [])].sort((a, b) => {
    const av = a[p5Sort], bv = b[p5Sort];
    return typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv));
  }), [feature, p5Sort]);
  if (error) return <div className="error-box">{error}</div>;
  if (!feature) return <div className="loading">Loading measured component data…</div>;
  const modelSources = (key: string) => metadata.model_names.map((_, i) => `posthoc.safetensors → ${key}[${i},${featureId}]`);

  async function saveNote() {
    await api.saveNote(featureId, note); setSaved("Saved separately to researcher_notes.json"); setTimeout(() => setSaved(""), 2500);
  }
  async function findControl() {
    setControl(await api.matchedControl(featureId, matchAttributes));
  }

  return <div className="page-stack">
    <div className="feature-header">
      <div><span className="eyebrow">Feature Explorer</span><h1>Component {featureId}</h1>
        {feature.taxonomy ? <Badge tone="target">{feature.taxonomy.replaceAll("_", " ")}</Badge> : <Badge>N/A — taxonomy unavailable</Badge>}
        {feature.low_support && <Badge tone="warn">Low support heuristic: &lt; {metadata.low_support_threshold} fires</Badge>}
      </div>
      <div className="feature-actions">
        <button onClick={() => onNavigate(Math.max(0, featureId - 1))}>← Previous</button>
        <input type="number" min="0" max={(metadata.n_features || 1) - 1} value={featureId} onChange={(e) => onNavigate(Number(e.target.value))} />
        <button onClick={() => onNavigate(Math.min((metadata.n_features || 1) - 1, featureId + 1))}>Next →</button>
        <select aria-label="Jump to taxonomy category" value={feature.taxonomy || ""} onChange={(e) => { const found = points.find((point) => point.taxonomy_categories.includes(e.target.value)); if (found) onNavigate(found.feature_id); }}><option value="">Jump to category</option>{taxonomy.map((item) => <option key={item.category}>{item.category}</option>)}</select>
        <button className={bookmarked ? "active" : ""} onClick={() => onBookmark(featureId)}>★ Bookmark</button>
        <button onClick={() => onCompare(featureId)}>Add to compare</button>
        <a className="button-link" href={`/api/export/feature/${featureId}.md`}>Export Markdown</a>
      </div>
    </div>
    {feature.missing.length > 0 && <div className="warn-box">{feature.missing.map((x) => <div key={x}>{x}</div>)}</div>}

    <Section title="My semantic interpretation" kicker="Researcher interpretation — never measured data">
      <div className="notes-grid">
        <label>Tentative label<input value={note.tentative_label} onChange={(e) => setNote({ ...note, tentative_label: e.target.value })} /></label>
        <label>Confidence<select value={note.confidence} onChange={(e) => setNote({ ...note, confidence: e.target.value as ResearcherNote["confidence"] })}><option>low</option><option>medium</option><option>high</option></select></label>
        <label>Candidate status<select value={note.candidate_status} onChange={(e) => setNote({ ...note, candidate_status: e.target.value as ResearcherNote["candidate_status"] })}><option>unreviewed</option><option>promising</option><option>control</option><option>reject</option><option>P6 candidate</option></select></label>
        <label className="wide">Semantic evidence<textarea value={note.semantic_evidence} onChange={(e) => setNote({ ...note, semantic_evidence: e.target.value })} /></label>
        <label className="wide">Notes<textarea value={note.notes} onChange={(e) => setNote({ ...note, notes: e.target.value })} /></label>
        <label className="wide">Alternative interpretation<textarea value={note.alternative_interpretation} onChange={(e) => setNote({ ...note, alternative_interpretation: e.target.value })} /></label>
        <label className="wide">Why interesting<textarea value={note.why_interesting} onChange={(e) => setNote({ ...note, why_interesting: e.target.value })} /></label>
        <label className="wide">Mentor notes<textarea value={note.mentor_notes} onChange={(e) => setNote({ ...note, mentor_notes: e.target.value })} /></label>
      </div><button className="primary" onClick={saveNote}>Save researcher notes</button> <span className="save-status">{saved}</span>
    </Section>

    <Section title="Activation examples" kicker="Measured shared latent activations">
      {feature.examples === null ? <p className="na">N/A — this component was not included in the analyzer’s capped context sample.</p> : feature.examples.length === 0 ?
        <p className="na">No retained activation examples for this component in the analyzed context set.</p> :
        <div className="examples-list">{[...feature.examples].sort((a, b) => (b.g_s || 0) - (a.g_s || 0)).map((example, index) =>
          <article className="example-card" key={index} title={example.source}><div className="example-meta"><Badge tone="target">{example.center_token || "N/A — center token unavailable"}</Badge><span>g_s <Metric value={example.g_s} kind="gs" source={`${example.source}.g_s`} /></span></div>
            <p>{example.text || "N/A — decoded text unavailable"}</p><small>Shared latent activation strength at this token</small>
            <details><summary>Token IDs and exact center index</summary><code>{JSON.stringify(example.token_ids ?? "N/A")}</code><div>center_in_window: {example.center_in_window ?? "N/A"}</div></details>
          </article>)}</div>}
    </Section>

    <div className="two-column">
      <Section title="P1 — Activation Representation"><h3>Absolute decoder norms</h3><ModelBars labels={metadata.model_names} values={feature.decoder_norm} sources={modelSources("decoder_norm_NC")} metricKind="beta" />
        <h3>Normalized activation rho</h3><ModelBars labels={metadata.model_names} values={feature.activation_rho} sources={modelSources("activation_rho_NC")} />
        <p className="explain">{feature.taxonomy?.startsWith("shared_activation") ? "Classified as shared activation by the current taxonomy threshold." : feature.taxonomy?.startsWith("concentrated_activation") ? "Classified as concentrated activation by the current taxonomy threshold." : "No shared/concentrated P1 claim is available from the current taxonomy."}</p>
        <Source>{String(feature.sources.activation_rho)}</Source></Section>
      <Section title="P2 — Parameter Mechanism Strength"><h3>Absolute beta</h3><ModelBars labels={metadata.model_names} values={feature.beta} sources={modelSources("mechanism_beta_NC")} metricKind="beta" />
        <h3>Normalized mechanism rho</h3><ModelBars labels={metadata.model_names} values={feature.mechanism_rho} sources={modelSources("mechanism_rho_NC")} />
        <div className="stat-row"><div><span>beta total</span><Metric value={feature.beta_total} kind="beta" /></div><div><span>fire count</span><Metric value={feature.fire_count} kind="integer" /></div><div><span>fire density</span><Metric value={feature.fire_density} /></div></div>
        <p className="explain">Normalized ownership and absolute mechanism mass are shown together. A concentrated rho alone does not establish importance.</p></Section>
    </div>

    <Section title="P3 — Where the Mechanism Lives">
      <p className="explain">{feature.locus_statement || "N/A — dominant locus comparison unavailable"}. This is a mechanical locus comparison, not a causal relocation claim.</p>
      <div className="locus-grid">{metadata.model_names.map((model, modelIndex) => <div key={model}><h3>{model}</h3>{metadata.matrix_names.map((matrix) => { const value = feature.locus[model]?.[matrix]; return <div className="bar-row" key={matrix} title={`posthoc.safetensors → locus/${model}/${matrix}[${featureId}]`}><span>{matrix}</span><div className="bar-track"><div className={`bar-fill model-${modelIndex}`} style={{ width: `${(value || 0) * 100}%` }} /></div><Metric value={value} /></div>; })}<strong>Dominant: {feature.dominant_locus[model] || "N/A"}</strong></div>)}</div>
      <div className="formula">Derived locus shift = Σⱼ |π_base(j) − π_finetuned(j)| = <Metric value={feature.locus_shift} /></div>
    </Section>

    <Section title="P5 — How the Parameter Mechanism Changed">
      <p className="explain">All values come directly from same-architecture pair tensors. Cosines describe alignment; relative change describes magnitude of Frobenius change. No value is labeled “bad”.</p>
      <div className="table-scroll"><table><thead><tr><th onClick={() => setP5Sort("matrix")}>Matrix</th><th>Base locus</th><th>FT locus</th>{(["component_cosine", "relative_component_change", "read_cosine", "write_cosine"] as const).map((key) => <th key={key} onClick={() => setP5Sort(key)}>{key.replaceAll("_", " ")}</th>)}</tr></thead>
        <tbody>{p5.map((row) => <tr key={row.matrix}><td>{row.matrix}</td><td><Metric value={feature.locus[metadata.model_names[0]]?.[row.matrix]} /></td><td><Metric value={feature.locus[metadata.model_names[1]]?.[row.matrix]} /></td><Heat value={row.component_cosine} source={row.sources.component_cosine} cosine /><Heat value={row.relative_component_change} source={row.sources.relative_component_change} /><Heat value={row.read_cosine} source={row.sources.read_cosine} cosine /><Heat value={row.write_cosine} source={row.sources.write_cosine} cosine /></tr>)}</tbody></table></div>
    </Section>

    <Section title="Matched control" kicker="Transparent numeric nearest neighbor">
      <p>Pool: <code>shared_activation_shared_mechanism</code>. Default variables: fire_count, beta_total, activation rho. Distance is range-normalized Euclidean distance.</p>
      <div className="match-attributes">{["fire_count", "beta_total", "activation_rho"].map((attribute) => <label key={attribute}><input type="checkbox" checked={matchAttributes.includes(attribute)} onChange={() => setMatchAttributes((old) => old.includes(attribute) ? old.filter((x) => x !== attribute) : [...old, attribute])} /> {attribute}</label>)}</div>
      <button onClick={findControl} disabled={matchAttributes.length === 0}>Find matched control</button>
      {control && <pre className="json-preview">{JSON.stringify(control, null, 2)}</pre>}
    </Section>
  </div>;
}

function Heat({ value, source, cosine = false }: { value: number | null; source: string; cosine?: boolean }) {
  const background = value === null ? "transparent" : cosine ? `rgba(${value < 0 ? "255,90,110" : "68,181,255"},${Math.min(.7, .12 + Math.abs(value) * .55)})` : `rgba(255,180,84,${Math.min(.72, .1 + Math.log1p(Math.max(0, value)) * .25)})`;
  return <td style={{ background }} title={source}><Metric value={value} source={source} /></td>;
}
