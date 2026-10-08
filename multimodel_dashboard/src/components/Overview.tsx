import type { OverviewPayload } from "../types";
import { Badge, Metric, Section, Source } from "./Common";
import { Histogram, ScatterPlot } from "./Charts";

export default function Overview({
  data,
  onFeature,
  onCategory,
}: {
  data: OverviewPayload;
  onFeature: (id: number) => void;
  onCategory: (category: string) => void;
}) {
  const { metadata, points } = data;
  return <div className="page-stack">
    <Section title="Experiment metadata" kicker="Real run provenance">
      <div className="metadata-grid">
        <Meta label="Run" value={metadata.run_name} />
        <Meta label="Models" value={metadata.model_names.join(" ↔ ")} />
        <Meta label="Checkpoint" value={metadata.checkpoint} />
        <Meta label="Components C" value={metadata.n_features?.toLocaleString()} />
        <Meta label="BatchTopK K" value={metadata.top_k?.toString()} />
        <Meta label="Analyzed tokens" value={metadata.validation_tokens?.toLocaleString()} />
        <Meta label="Layer / site" value={metadata.selected_layers?.join(" · ")} />
        <Meta label="Matrices" value={metadata.matrix_names.join(", ")} />
        <Meta label="Taxonomy thresholds" value={metadata.shared_epsilon !== null && metadata.concentrated_threshold !== null ? `ε=${metadata.shared_epsilon}, τ=${metadata.concentrated_threshold}` : null} />
        <Meta label="Git commit" value={metadata.git_commit} mono />
      </div>
      <Source>{metadata.sources.experiment || "N/A — experiment config unavailable"}</Source>
    </Section>

    <Section title="P4 — Feature Taxonomy" kicker="Exact category membership">
      <p className="explain">The target category means activation representation is classified as shared while the parameter mechanism is concentrated in one model. It is not a causal finding.</p>
      <div className="taxonomy-grid">
        {data.taxonomy.map((item) => <button className="taxonomy-card" key={item.category} onClick={() => onCategory(item.category)} title={item.source}>
          <span>{item.category.replaceAll("_", " ")}</span>
          <strong>{item.count?.toLocaleString() ?? "N/A"}</strong>
          <Metric value={item.percentage} suffix="%" />
          <div className="taxonomy-bar"><i style={{ width: `${item.percentage || 0}%` }} /></div>
        </button>)}
      </div>
    </Section>

    <Section title="P1 vs P2 ownership" kicker="Central research plot">
      <p className="explain">Each point is one component. Position is measured rho; size reflects <code>log(1 + beta_total)</code>. Shaded bands use the configured taxonomy thresholds.</p>
      {data.scatter_available ? <ScatterPlot points={points} metadata={metadata} onSelect={onFeature} /> :
        <div className="empty-state">This visualization is disabled because the loaded analysis does not contain exactly two models.</div>}
      <div className="legend-row">
        {data.taxonomy.map((item) => <Badge key={item.category}>{item.category.replaceAll("_", " ")}</Badge>)}
      </div>
    </Section>

    <Section title="Measured distributions" kicker="No interpolated bins">
      <div className="hist-grid">
        <Histogram title={`Activation rho — ${metadata.model_names[0] || "model 0"}`} values={points.map((p) => p.activation_rho?.[0] ?? null)} />
        <Histogram title={`Mechanism rho — ${metadata.model_names[0] || "model 0"}`} values={points.map((p) => p.mechanism_rho?.[0] ?? null)} />
        <Histogram title="Beta total" values={points.map((p) => p.beta_total)} />
        <Histogram title="Fire count" values={points.map((p) => p.fire_count)} />
      </div>
    </Section>
  </div>;
}

function Meta({ label, value, mono = false }: { label: string; value: string | null | undefined; mono?: boolean }) {
  return <div className="meta-item"><span>{label}</span><strong className={mono ? "mono" : ""}>{value || "N/A — data unavailable"}</strong></div>;
}
