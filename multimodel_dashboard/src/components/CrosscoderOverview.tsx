import type { CrosscoderOverviewPayload } from "../types";
import { RelativeNormHistogram } from "./Charts";
import { Badge, Metric, Section, Source } from "./Common";

export default function CrosscoderOverview({
  data,
  onFeature,
}: {
  data: CrosscoderOverviewPayload;
  onFeature: (id: number) => void;
}) {
  if (!data.available || !data.metadata || !data.metrics) {
    return <Unavailable diagnostics={data.diagnostics} />;
  }
  const { metadata, metrics } = data;
  const classOrder = ["base-specific", "shared", "intermediate", "finetuned-specific"];
  return <div className="page-stack">
    <div className="feature-header">
      <div><span className="eyebrow">Representation-space model diffing</span><h1>BatchTopK Crosscoder</h1></div>
      <Badge tone="pass">Measured artifact</Badge>
    </div>

    <Section title="Experiment metadata" kicker="Measured provenance">
      <div className="metadata-grid">
        <Meta label="Run" value={metadata.run_name} />
        <Meta label="Models" value={metadata.model_ids.filter(Boolean).join(" ↔ ")} />
        <Meta label="Layer / site" value={metadata.sites.filter(Boolean).join(" · ")} />
        <Meta label="Dictionary" value={metadata.n_features?.toLocaleString()} />
        <Meta label="Completed steps" value={metadata.training_steps?.toLocaleString()} />
        <Meta label="Train BatchTopK k" value={metadata.train_batch_topk?.toLocaleString()} />
        <Meta label="Validation tokens" value={metadata.validation_tokens?.toLocaleString()} />
        <Meta label="Checkpoint" value={metadata.checkpoint} mono />
        <Meta label="Reference method" value={metadata.reference.method} />
        <Meta label="Reference commit" value={metadata.reference.library_commit} mono />
      </div>
      <Source>{data.sources?.config ?? metadata.config_path}</Source>
    </Section>

    <Section title="Training and evaluation diagnostics" kicker="Measured values">
      <div className="metric-card-grid">
        <MetricCard label="Total FVE" value={metrics.fve_total} source={data.sources?.summary} />
        <MetricCard label="Base FVE" value={metrics.fve_base} source={data.sources?.summary} />
        <MetricCard label="Fine-tuned FVE" value={metrics.fve_finetuned} source={data.sources?.summary} />
        <MetricCard label="Validation L0" value={metrics.validation_l0} source={data.sources?.summary} />
        <MetricCard label="Dead features" value={metrics.dead_count} integer source={data.sources?.summary} />
        <MetricCard label="Dead fraction" value={metrics.dead_fraction} percent source={data.sources?.summary} />
      </div>
      <p className="explain">Validation L0 is measured after threshold fitting. It is not assumed to equal the training BatchTopK value.</p>
    </Section>

    <Section title="Relative decoder norm distribution" kicker="Central measured result">
      <p className="explain">{data.relative_decoder_norm_convention}. Shaded regions are the stored published thresholds; unshaded values are intermediate.</p>
      {data.strict_exclusive_absent && <div className="result-callout"><strong>{data.strict_result}</strong><span>This is a scientific result, not an error state.</span></div>}
      <div className="taxonomy-grid crosscoder-class-grid">
        {classOrder.map((classification) => <div className="taxonomy-card" key={classification}>
          <span>{classification}</span>
          <strong>{data.classification_counts?.[classification]?.toLocaleString() ?? "N/A"}</strong>
          <Metric value={data.classification_fractions?.[classification]} kind="percent" />
        </div>)}
      </div>
      <RelativeNormHistogram
        bins={data.relative_norm_histogram ?? []}
        convention={data.relative_decoder_norm_convention}
        thresholds={data.thresholds ?? {}}
      />
      <div className="stat-row">
        <div><span>Minimum measured score</span><Metric value={metrics.relative_norm_min} /></div>
        <div><span>Maximum measured score</span><Metric value={metrics.relative_norm_max} /></div>
        <div><span>Histogram source</span><strong>feature_metrics.json</strong></div>
      </div>
    </Section>

    <Section title="Blind inspection groups" kicker="Ranking groups — not strict classes">
      <p className="explain">{data.candidate_selection}. A tail name describes inspection rank only and never changes strict classification.</p>
      <div className="inspection-groups">
        {Object.entries(data.candidate_groups ?? {}).map(([group, ids]) => <div key={group} className="inspection-group">
          <h3>{group}</h3><span>{ids.length} frozen candidates</span>
          <div>{ids.map((id) => <button className="id-button" key={id} onClick={() => onFeature(id)}>{id}</button>)}</div>
        </div>)}
      </div>
    </Section>
    {data.diagnostics.length > 0 && <div className="warn-box">{data.diagnostics.map((item) => <div key={item}>{item}</div>)}</div>}
  </div>;
}

function MetricCard({ label, value, source, integer = false, percent = false }: { label: string; value: number | null; source?: string; integer?: boolean; percent?: boolean }) {
  return <div className="metric-card"><span>{label}</span><strong><Metric value={value} kind={integer ? "integer" : percent ? "percent" : undefined} source={source} /></strong><small>MEASURED</small></div>;
}

function Meta({ label, value, mono = false }: { label: string; value: string | null | undefined; mono?: boolean }) {
  return <div className="meta-item"><span>{label}</span><strong className={mono ? "mono" : ""}>{value || "N/A — data unavailable"}</strong></div>;
}

function Unavailable({ diagnostics }: { diagnostics: string[] }) {
  return <div className="empty-state unavailable-panel"><strong>Crosscoder artifacts unavailable</strong><p>No scientific value is substituted.</p>{diagnostics.map((item) => <div key={item}>{item}</div>)}</div>;
}
