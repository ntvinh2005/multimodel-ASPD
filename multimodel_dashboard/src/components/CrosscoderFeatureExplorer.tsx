import { useEffect, useState } from "react";
import { api } from "../api";
import type { CrosscoderFeatureDetail, CrosscoderMetadata } from "../types";
import { ActivationExamples } from "./ActivationExamples";
import { Badge, crosscoderArtifactTitle, crosscoderDisplayConvention, crosscoderDisplayLabel, Metric, Section, Source } from "./Common";

export default function CrosscoderFeatureExplorer({
  featureId,
  metadata,
  onNavigate,
  onAspdFeature,
}: {
  featureId: number;
  metadata: CrosscoderMetadata;
  onNavigate: (id: number) => void;
  onAspdFeature: (id: number) => void;
}) {
  const [feature, setFeature] = useState<CrosscoderFeatureDetail | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setFeature(null); setError("");
    api.crosscoderFeature(featureId).then(setFeature).catch((reason: Error) => setError(reason.message));
  }, [featureId]);

  if (error) return <div className="error-box">{error}</div>;
  if (!feature) return <div className="loading">Loading measured Crosscoder feature data…</div>;
  const annotation = feature.semantic_annotation;
  return <div className="page-stack">
    <div className="feature-header">
      <div><span className="eyebrow">Crosscoder Feature Explorer</span><h1>Feature {featureId}</h1></div>
      <div className="feature-actions">
        <button onClick={() => onNavigate(Math.max(0, featureId - 1))}>← Previous</button>
        <input type="number" min="0" max={(metadata.n_features ?? 1) - 1} value={featureId} onChange={(event) => onNavigate(Number(event.target.value))} />
        <button onClick={() => onNavigate(Math.min((metadata.n_features ?? 1) - 1, featureId + 1))}>Next →</button>
      </div>
    </div>

    <div className="classification-split">
      <div className="measured-panel">
        <span className="scientific-kind">MEASURED · STRICT CLASSIFICATION</span>
        <strong title={crosscoderArtifactTitle(feature.classification)}>{crosscoderDisplayLabel(feature.classification)}</strong>
        <small title={feature.relative_decoder_norm_convention}>{crosscoderDisplayConvention(feature.relative_decoder_norm_convention)}</small>
      </div>
      <div className="derived-panel">
        <span className="scientific-kind">DERIVED RANK · INSPECTION GROUP</span>
        <strong title={feature.inspection_groups.map(crosscoderArtifactTitle).filter(Boolean).join(" ") || undefined}>{feature.inspection_groups.length ? feature.inspection_groups.join(", ") : "not a frozen inspection candidate"}</strong>
        <small>Tail membership does not imply strict model specificity.</small>
      </div>
    </div>

    <Section title="Measured feature metrics" kicker="Canonical feature_metrics.json row">
      <div className="metric-card-grid">
        <MetricCard label="Relative decoder norm" value={feature.relative_decoder_norm} />
        <MetricCard label="Base decoder norm" value={feature.decoder_norm_base} />
        <MetricCard label="Target/post-trained decoder norm" value={feature.decoder_norm_finetuned} title="Artifact role: finetuned" />
        <MetricCard label="Decoder cosine" value={feature.decoder_cosine} />
        <MetricCard label="Fire count" value={feature.fire_count} integer />
        <MetricCard label="Activation density" value={feature.activation_density} percent />
      </div>
      <div className="rank-grid">
        <div><span>Strict-class rank</span><Metric value={feature.blind_rank_within_class} kind="integer" /></div>
        <div><span>Base-tail rank</span><Metric value={feature.blind_rank_base_tail} kind="integer" /></div>
        <div title={crosscoderArtifactTitle("finetuned-tail")}><span>Target/post-trained-tail rank</span><Metric value={feature.blind_rank_finetuned_tail} kind="integer" /></div>
      </div>
      <Source>{feature.sources.measured}</Source>
    </Section>

    <Section title="Activation examples" kicker="Exact tokenizer-aligned measured contexts">
      <div className="stat-row activation-stats">
        <div><span>Context status</span><strong>{feature.context.status}</strong></div>
        <div><span>Saved contexts</span><Metric value={feature.context.saved_contexts} kind="integer" /></div>
        <div><span>Artifact</span><strong>{feature.context.source}</strong></div>
      </div>
      <ActivationExamples examples={feature.examples} missingMessage={feature.context.message} />
    </Section>

    <Section title="Post-ranking semantic interpretation" kicker="NOT MEASURED GROUND TRUTH">
      {!annotation || !annotation.tentative_label ? <div className="empty-state">No structured semantic interpretation is available for this feature.</div> : <div className="interpretation-panel">
        <div className="interpretation-header"><strong>{annotation.tentative_label}</strong><Badge tone="warn">post-ranking interpretation</Badge></div>
        <dl>
          <dt>Interpretation confidence</dt><dd><Metric value={annotation.interpretation_confidence} /> <small>research-review confidence, not a model probability</small></dd>
          <dt>Annotation source</dt><dd>{annotation.annotation_source ?? "N/A — source unavailable"}</dd>
          <dt>Model/version</dt><dd>{[annotation.model, annotation.model_version].filter(Boolean).join(" · ") || "N/A — unavailable"}</dd>
          <dt>Evidence context IDs</dt><dd>{annotation.evidence_validation_sequence_indices?.join(", ") || "N/A — unavailable"}</dd>
        </dl>
        {annotation.warning && <div className="warn-box">{annotation.warning}</div>}
        {annotation.limitations?.length ? <ul>{annotation.limitations.map((item) => <li key={item}>{item}</li>)}</ul> : null}
        <Source>{annotation.source ?? "N/A — annotation source unavailable"}</Source>
      </div>}
    </Section>

    {feature.related_aspd?.length ? <Section title="Related ASPD evidence" kicker="Structured phenomenon references only">
      {feature.related_aspd.map((relationship) => <div className="relationship-row" key={relationship.phenomenon_id}>
        <div><strong>{relationship.phenomenon_label}</strong><Badge tone={relationship.status === "low_support_hypothesis" ? "warn" : "neutral"}>{relationship.status.replaceAll("_", " ")}</Badge><p>{relationship.status_definition}</p></div>
        <div>{relationship.aspd_component_ids.map((id) => <button className="id-button" key={id} onClick={() => onAspdFeature(id)}>ASPD {id}</button>)}</div>
      </div>)}
      <p className="explain">These links do not establish latent identity or causal equivalence.</p>
    </Section> : null}
  </div>;
}

function MetricCard({ label, value, integer = false, percent = false, title }: { label: string; value: number | null; integer?: boolean; percent?: boolean; title?: string }) {
  return <div className="metric-card" title={title}><span>{label}</span><strong><Metric value={value} kind={integer ? "integer" : percent ? "percent" : undefined} /></strong><small>MEASURED</small></div>;
}
