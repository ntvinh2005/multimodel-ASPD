import type { ReactNode } from "react";

export function formatMetric(value: number | null | undefined, kind = "default"): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "N/A — data unavailable";
  if (kind === "integer") return Math.round(value).toLocaleString();
  if (kind === "beta" || (Math.abs(value) > 0 && Math.abs(value) < 1e-4)) return value.toExponential(5);
  if (kind === "gs") return value.toPrecision(6);
  return value.toFixed(4);
}

export function Metric({
  value,
  source,
  kind,
  suffix = "",
}: {
  value: number | null | undefined;
  source?: string | null;
  kind?: string;
  suffix?: string;
}) {
  return (
    <span className={value === null || value === undefined ? "na" : "metric"} title={source || undefined}>
      {formatMetric(value, kind)}{value !== null && value !== undefined ? suffix : ""}
    </span>
  );
}

export function Section({ title, kicker, children }: { title: string; kicker?: string; children: ReactNode }) {
  return (
    <section className="section-card">
      <header className="section-heading">
        {kicker && <span className="kicker">{kicker}</span>}
        <h2>{title}</h2>
      </header>
      {children}
    </section>
  );
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: string }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function ModelBars({
  labels,
  values,
  sources,
  metricKind,
}: {
  labels: string[];
  values: number[] | null;
  sources?: string[];
  metricKind?: string;
}) {
  if (!values) return <p className="na">N/A — expected tensor is missing or has incompatible shape.</p>;
  const scale = Math.max(...values.map((x) => Math.abs(x)), 1e-12);
  return (
    <div className="model-bars">
      {labels.map((label, index) => (
        <div className="bar-row" key={label} title={sources?.[index]}>
          <span>{label}</span>
          <div className="bar-track"><div className={`bar-fill model-${index}`} style={{ width: `${(Math.abs(values[index]) / scale) * 100}%` }} /></div>
          <Metric value={values[index]} kind={metricKind} source={sources?.[index]} />
        </div>
      ))}
    </div>
  );
}

export function Source({ children }: { children: ReactNode }) {
  return <div className="source-line">Source: {children}</div>;
}
