import type { MethodComparisonPayload, SemanticAnnotation } from "../types";
import { Badge, crosscoderArtifactTitle, crosscoderDisplayLabel, Metric, Section, Source } from "./Common";

export default function MethodComparison({
  data,
  onCrosscoderFeature,
  onAspdFeature,
}: {
  data: MethodComparisonPayload;
  onCrosscoderFeature: (id: number) => void;
  onAspdFeature: (id: number) => void;
}) {
  if (!data.available) {
    return <div className="empty-state unavailable-panel"><strong>Structured comparison unavailable</strong><p>No feature relationship is inferred in the browser.</p>{data.diagnostics.map((item) => <div key={item}>{item}</div>)}</div>;
  }
  const measured = data.global_measured;
  return <div className="page-stack">
    <div className="feature-header">
      <div><span className="eyebrow">Phenomenon-level evidence</span><h1>Cross-method Comparison</h1></div>
      <div><Badge tone="warn">post-ranking synthesis</Badge><Badge>not feature identity</Badge><Badge>not causal</Badge></div>
    </div>

    <div className="method-summary-grid">
      <article className="method-summary-card">
        <span className="scientific-kind">METHOD</span><h2>Crosscoder</h2><strong>Representation-space model diffing</strong>
        <div className="mini-metrics"><span>FVE <Metric value={measured?.crosscoder?.metrics.fve_total} /></span><span title="Artifact role: finetuned">Strict base / target/post-trained exclusive <strong>{measured?.crosscoder?.classification_counts["base-specific"] ?? "N/A"} / {measured?.crosscoder?.classification_counts["finetuned-specific"] ?? "N/A"}</strong></span></div>
        <div className="global-counts">{Object.entries(measured?.crosscoder?.classification_counts ?? {}).map(([label, count]) => <span key={label} title={crosscoderArtifactTitle(label)}>{crosscoderDisplayLabel(label).replaceAll("-", " ")} <strong>{count.toLocaleString()}</strong></span>)}</div>
        <p>{measured?.crosscoder?.strict_result ?? "Measured Crosscoder summary unavailable."}</p>
        <ul>{data.global_summary?.crosscoder?.map((item) => <li key={item}>{item}</li>)}</ul>
      </article>
      <article className="method-summary-card">
        <span className="scientific-kind">METHOD</span><h2>Multimodel ASPD</h2><strong>Activation + parameter-mechanism decomposition</strong>
        <div className="mini-metrics"><span>Components <Metric value={measured?.aspd.n_features} kind="integer" /></span><span>Validation tokens <Metric value={measured?.aspd.validation_tokens} kind="integer" /></span></div>
        <div className="global-counts">{Object.entries(measured?.aspd.taxonomy_counts ?? {}).map(([label, count]) => <span key={label}>{label.replaceAll("_", " ")} <strong>{count?.toLocaleString() ?? "N/A"}</strong></span>)}</div>
        <ul>{data.global_summary?.aspd?.map((item) => <li key={item}>{item}</li>)}</ul>
      </article>
    </div>

    {data.global_summary?.current_finding && <div className="result-callout synthesis-callout"><span className="scientific-kind">POST-RANKING RESEARCH SYNTHESIS</span><strong>{data.global_summary.current_finding}</strong></div>}

    <Section title="Current phenomenon-level evidence" kicker="No automatic semantic matching">
      <div className="phenomena-list">{data.phenomena.map((phenomenon) => <article className={`phenomenon-card status-${phenomenon.status}`} key={phenomenon.id}>
        <header><div><span className="scientific-kind">POST-RANKING COMPARISON</span><h2>{phenomenon.label}</h2></div><Badge tone={phenomenon.status === "low_support_hypothesis" ? "warn" : "neutral"}><span title={phenomenon.status_definition}>{phenomenon.status.replaceAll("_", " ")}</span></Badge></header>
        <p className="status-definition"><strong>Status meaning:</strong> {phenomenon.status_definition}</p>
        <div className="evidence-columns">
          <div><h3>Crosscoder evidence</h3>{phenomenon.crosscoder_evidence.length === 0 ? <p className="na">No strict-exclusive corresponding Crosscoder feature is referenced.</p> : phenomenon.crosscoder_evidence.map((feature) => <div className="evidence-card" key={feature.feature_id}>
            <button className="id-button" onClick={() => onCrosscoderFeature(feature.feature_id)}>Feature {feature.feature_id}</button>
            <div><span>Strict class</span><strong title={crosscoderArtifactTitle(feature.classification)}>{crosscoderDisplayLabel(feature.classification)}</strong></div>
            <div><span>Inspection group</span><strong title={feature.inspection_groups.map(crosscoderArtifactTitle).filter(Boolean).join(" ") || undefined}>{feature.inspection_groups.join(", ") || "none"}</strong></div>
            <div><span>Relative norm</span><Metric value={feature.relative_decoder_norm} /></div>
            <div><span>Fires</span><Metric value={feature.fire_count} kind="integer" /></div>
            <AnnotationSummary annotation={feature.semantic_annotation} />
            <ContextSummary examples={feature.examples} />
          </div>)}</div>
          <div><h3>ASPD evidence</h3>{phenomenon.aspd_evidence.map((component) => <div className="evidence-card" key={component.component_id}>
            <button className="id-button" onClick={() => onAspdFeature(component.component_id)}>Component {component.component_id}</button>
            {component.low_support && <Badge tone="warn">LOW SUPPORT</Badge>}
            <div><span>Taxonomy</span><strong>{component.taxonomy?.replaceAll("_", " ") ?? "N/A"}</strong></div>
            <div><span>Activation ρ base / FT</span><strong>{component.activation_rho?.map((value) => value.toFixed(4)).join(" / ") ?? "N/A"}</strong></div>
            <div><span>Mechanism ρ base / FT</span><strong>{component.mechanism_rho?.map((value) => value.toFixed(4)).join(" / ") ?? "N/A"}</strong></div>
            <div><span>Fires</span><Metric value={component.fire_count} kind="integer" /></div>
            <AnnotationSummary annotation={component.semantic_annotation} />
            <ContextSummary examples={component.examples} />
          </div>)}</div>
        </div>
        <div className="interpretation-panel compact"><span className="scientific-kind">INTERPRETATION</span><p>{phenomenon.interpretation}</p><strong>Limitations</strong><ul>{phenomenon.limitations.map((item) => <li key={item}>{item}</li>)}</ul></div>
      </article>)}</div>
      <Source>{data.source ?? "N/A — comparison artifact source unavailable"}</Source>
    </Section>
  </div>;
}

function AnnotationSummary({ annotation }: { annotation: SemanticAnnotation | null }) {
  if (!annotation?.tentative_label) return <p className="na">No post-ranking semantic label.</p>;
  return <div className="annotation-summary"><span>Post-ranking interpretation</span><strong>{annotation.tentative_label}</strong><small>Interpretation confidence: <Metric value={annotation.interpretation_confidence} /> · not measured ground truth</small>{annotation.warning && <Badge tone="warn">{annotation.warning}</Badge>}</div>;
}

function ContextSummary({ examples }: { examples: import("../types").ActivationExample[] | null }) {
  if (examples === null) return <p className="na">Context extraction unavailable.</p>;
  if (!examples.length) return <p className="na">No retained positive context.</p>;
  const example = examples[0];
  return <details className="decoded-context"><summary>Representative retained context</summary><p><strong>{example.center_token ?? "center unavailable"}</strong> · g<sup>s</sup> <Metric value={example.g_s} kind="gs" /><br />{example.text ?? "Decoded text unavailable"}</p></details>;
}
