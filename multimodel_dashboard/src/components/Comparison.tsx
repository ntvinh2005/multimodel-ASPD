import { useEffect, useState } from "react";
import { api } from "../api";
import type { FeatureDetail, Metadata } from "../types";
import { Badge, Metric, ModelBars, Section } from "./Common";

export default function Comparison({ ids, metadata, onRemove, onFeature }: { ids: number[]; metadata: Metadata; onRemove: (id: number) => void; onFeature: (id: number) => void }) {
  const [features, setFeatures] = useState<FeatureDetail[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    if (ids.length < 2) { setFeatures([]); return; }
    api.compare(ids).then((x) => setFeatures(x.items)).catch((e: Error) => setError(e.message));
  }, [ids]);
  if (ids.length < 2) return <div className="empty-state">Select 2–5 components from Candidate Finder or Feature Explorer to compare measured data side-by-side.</div>;
  if (error) return <div className="error-box">{error}</div>;
  return <div className="page-stack"><div className="feature-header"><div><span className="eyebrow">Mentor presentation mode</span><h1>Component comparison</h1></div><a className="button-link" href={`/api/export/selected.json?${ids.map((id) => `ids=${id}`).join("&")}`}>Export selected JSON</a></div>
    <div className="compare-grid">{features.map((feature) => <article className="compare-card" key={feature.feature_id}><header><button className="id-button" onClick={() => onFeature(feature.feature_id)}>Component {feature.feature_id}</button><button onClick={() => onRemove(feature.feature_id)}>×</button></header><Badge>{feature.taxonomy?.replaceAll("_", " ") || "N/A"}</Badge>
      <h3>Activation examples</h3>{feature.examples?.slice(0, 2).map((ex, i) => <p className="compare-context" key={i}><strong>{ex.center_token || "N/A"}</strong> · g_s <Metric value={ex.g_s} kind="gs" /><br />{ex.text || "N/A — text unavailable"}</p>) || <p className="na">N/A — no retained examples</p>}
      <h3>P1 activation rho</h3><ModelBars labels={metadata.model_names} values={feature.activation_rho} />
      <h3>P2 beta</h3><ModelBars labels={metadata.model_names} values={feature.beta} metricKind="beta" /><div>beta total <Metric value={feature.beta_total} kind="beta" /> · fires <Metric value={feature.fire_count} kind="integer" /></div>
      <h3>P3 dominant loci</h3>{metadata.model_names.map((model) => <div key={model}>{model}: <strong>{feature.dominant_locus[model] || "N/A"}</strong></div>)}<div>L1 shift <Metric value={feature.locus_shift} /></div>
      <h3>P5 extrema</h3><div>min read cosine <Metric value={feature.p5_min_read_cosine} /></div><div>min write cosine <Metric value={feature.p5_min_write_cosine} /></div><div>max relative change <Metric value={feature.p5_max_relative_change} /></div>
      <h3>Researcher note</h3><p>{feature.note?.tentative_label || "N/A — no tentative label"} · {feature.note?.candidate_status || "unreviewed"}</p>
    </article>)}</div>
    <Section title="Comparison caveat"><p>Side-by-side differences are descriptive. They do not establish semantic identity or causal behavioral control.</p></Section>
  </div>;
}
