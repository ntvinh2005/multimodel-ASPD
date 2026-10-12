import type { HealthPayload } from "../types";
import { Badge, Section } from "./Common";

export default function DataHealth({ data }: { data: HealthPayload }) {
  const tone = (status: string) => status === "PASS" ? "pass" : status === "WARN" ? "warn" : "fail";
  return <div className="page-stack">
    <div className="feature-header"><div><span className="eyebrow">Scientific correctness gate</span><h1>Data Health</h1></div><Badge tone={tone(data.overall_status)}>{data.overall_status}</Badge></div>
    {data.startup_diagnostics.length > 0 && <div className="warn-box">{data.startup_diagnostics.map((x) => <div key={x}>{x}</div>)}</div>}
    <Section title="Activation-context coverage" kicker="Key presence is distinct from firing">
      <div className="stat-row">
        <div><span>Components</span><strong>{data.context_coverage.components?.toLocaleString() ?? "N/A"}</strong></div>
        <div><span>Context extracted</span><strong>{data.context_coverage.components_with_context_extraction.toLocaleString()}</strong></div>
        <div><span>Saved context ≥ 1</span><strong>{data.context_coverage.components_with_at_least_one_saved_context.toLocaleString()}</strong></div>
        <div><span>Not extracted</span><strong>{data.context_coverage.components_not_extracted.toLocaleString()}</strong></div>
        <div><span>Fired but not extracted</span><strong>{data.context_coverage.components_with_fire_count_gt_zero_but_no_extracted_context.toLocaleString()}</strong></div>
      </div>
      <p className="explain">Configured mode: <code>{data.context_coverage.configured_mode ?? "unknown"}</code>. Absence from <code>{data.context_coverage.source}</code> does not mean that a component was inactive.</p>
    </Section>
    <div className="health-grid">{data.checks.map((check) => <article className="health-card" key={check.name}><header><h2>{check.name}</h2><Badge tone={tone(check.status)}>{check.status}</Badge></header><pre>{JSON.stringify(check, null, 2)}</pre></article>)}</div>
    <Section title="Tensor inventory" kicker="Enumerated at startup">
      {data.missing_expected_keys.length > 0 && <div className="error-box"><strong>Missing expected keys</strong>{data.missing_expected_keys.map((key) => <div key={key}>N/A — data unavailable: {key}</div>)}</div>}
      <div className="key-groups">{Object.entries(data.categorized_keys).map(([group, keys]) => <details key={group}><summary>{group} ({keys.length})</summary>{keys.map((key) => <div className="tensor-key" key={key}>{key} <span>{JSON.stringify(data.tensor_shapes[key])}</span></div>)}</details>)}</div>
    </Section>
  </div>;
}
