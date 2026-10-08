import type { HealthPayload } from "../types";
import { Badge, Section } from "./Common";

export default function DataHealth({ data }: { data: HealthPayload }) {
  const tone = (status: string) => status === "PASS" ? "pass" : status === "WARN" ? "warn" : "fail";
  return <div className="page-stack">
    <div className="feature-header"><div><span className="eyebrow">Scientific correctness gate</span><h1>Data Health</h1></div><Badge tone={tone(data.overall_status)}>{data.overall_status}</Badge></div>
    {data.startup_diagnostics.length > 0 && <div className="warn-box">{data.startup_diagnostics.map((x) => <div key={x}>{x}</div>)}</div>}
    <div className="health-grid">{data.checks.map((check) => <article className="health-card" key={check.name}><header><h2>{check.name}</h2><Badge tone={tone(check.status)}>{check.status}</Badge></header><pre>{JSON.stringify(check, null, 2)}</pre></article>)}</div>
    <Section title="Tensor inventory" kicker="Enumerated at startup">
      {data.missing_expected_keys.length > 0 && <div className="error-box"><strong>Missing expected keys</strong>{data.missing_expected_keys.map((key) => <div key={key}>N/A — data unavailable: {key}</div>)}</div>}
      <div className="key-groups">{Object.entries(data.categorized_keys).map(([group, keys]) => <details key={group}><summary>{group} ({keys.length})</summary>{keys.map((key) => <div className="tensor-key" key={key}>{key} <span>{JSON.stringify(data.tensor_shapes[key])}</span></div>)}</details>)}</div>
    </Section>
  </div>;
}
