import { useEffect, useRef, useState, type MouseEvent } from "react";
import type { FeatureSummary, Metadata } from "../types";
import { formatMetric } from "./Common";

const palette: Record<string, string> = {
  shared_activation_shared_mechanism: "#5ea6ff",
  shared_activation_concentrated_mechanism: "#ffb454",
  concentrated_activation_shared_mechanism: "#b67cff",
  concentrated_activation_concentrated_mechanism: "#ff6b7a",
  mixed_or_subset_mass: "#8792a7",
};

export function ScatterPlot({
  points,
  metadata,
  onSelect,
}: {
  points: FeatureSummary[];
  metadata: Metadata;
  onSelect: (id: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<FeatureSummary | null>(null);
  const width = 760, height = 500, left = 58, top = 24, right = 22, bottom = 52;
  const innerW = width - left - right, innerH = height - top - bottom;

  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--panel") || "#111827";
    context.fillRect(0, 0, width, height);
    const eps = metadata.shared_epsilon;
    const threshold = metadata.concentrated_threshold;
    if (eps !== null) {
      context.fillStyle = "rgba(94,166,255,.10)";
      context.fillRect(left + (0.5 - eps) * innerW, top, 2 * eps * innerW, innerH);
    }
    if (threshold !== null) {
      context.fillStyle = "rgba(255,107,122,.09)";
      context.fillRect(left, top, innerW, (1 - threshold) * innerH);
      context.fillRect(left, top + threshold * innerH, innerW, (1 - threshold) * innerH);
    }
    context.strokeStyle = "#44506a";
    context.lineWidth = 1;
    context.strokeRect(left, top, innerW, innerH);
    context.setLineDash([4, 5]);
    context.strokeStyle = "#73809b";
    context.beginPath(); context.moveTo(left + innerW / 2, top); context.lineTo(left + innerW / 2, top + innerH); context.stroke();
    context.beginPath(); context.moveTo(left, top + innerH / 2); context.lineTo(left + innerW, top + innerH / 2); context.stroke();
    context.setLineDash([]);
    context.font = "12px ui-monospace, monospace";
    context.fillStyle = "#aeb8cb";
    for (let tick = 0; tick <= 10; tick += 2) {
      const value = tick / 10;
      context.fillText(value.toFixed(1), left + value * innerW - 8, top + innerH + 22);
      context.fillText((1 - value).toFixed(1), 20, top + value * innerH + 4);
    }
    context.font = "13px system-ui";
    context.fillText(`Activation rho — ${metadata.model_names[0] || "model 0"}`, left + innerW / 2 - 90, height - 8);
    context.save(); context.translate(14, top + innerH / 2 + 75); context.rotate(-Math.PI / 2);
    context.fillText(`Mechanism rho — ${metadata.model_names[0] || "model 0"}`, 0, 0); context.restore();
    for (const point of points) {
      const x = point.activation_rho?.[0], y = point.mechanism_rho?.[0];
      if (x === undefined || y === undefined) continue;
      const radius = point.beta_total === null ? 1.6 : Math.min(4.5, 1.5 + Math.log1p(Math.max(point.beta_total, 0)) * 2);
      context.beginPath(); context.arc(left + x * innerW, top + (1 - y) * innerH, radius, 0, Math.PI * 2);
      context.fillStyle = palette[point.taxonomy || ""] || "#8792a7"; context.globalAlpha = .58; context.fill();
    }
    context.globalAlpha = 1;
  }, [points, metadata]);

  function nearest(event: MouseEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const mx = (event.clientX - rect.left) * width / rect.width;
    const my = (event.clientY - rect.top) * height / rect.height;
    let best: FeatureSummary | null = null, distance = 12;
    for (const point of points) {
      const x = point.activation_rho?.[0], y = point.mechanism_rho?.[0];
      if (x === undefined || y === undefined) continue;
      const d = Math.hypot(left + x * innerW - mx, top + (1 - y) * innerH - my);
      if (d < distance) { distance = d; best = point; }
    }
    return best;
  }

  return (
    <div className="scatter-wrap">
      <canvas ref={canvas} width={width} height={height} onMouseMove={(e) => setHover(nearest(e))}
        onMouseLeave={() => setHover(null)} onClick={(e) => { const p = nearest(e); if (p) onSelect(p.feature_id); }} />
      {hover && <div className="chart-tooltip">
        <strong>Component {hover.feature_id}</strong><br />
        {hover.taxonomy || "N/A — taxonomy unavailable"}<br />
        activation rho: {hover.activation_rho?.map((v) => formatMetric(v)).join(" / ")}<br />
        mechanism rho: {hover.mechanism_rho?.map((v) => formatMetric(v)).join(" / ")}<br />
        beta: {hover.beta?.map((v) => formatMetric(v, "beta")).join(" / ")}<br />
        beta total: {formatMetric(hover.beta_total, "beta")} · fires: {formatMetric(hover.fire_count, "integer")}<br />
        <span className="tooltip-source">activation: posthoc.safetensors → activation_rho_NC[:,{hover.feature_id}]<br />mechanism: posthoc.safetensors → mechanism_rho_NC[:,{hover.feature_id}]<br />beta: posthoc.safetensors → mechanism_beta_NC[:,{hover.feature_id}]</span>
      </div>}
    </div>
  );
}

export function Histogram({ values, title }: { values: Array<number | null>; title: string }) {
  const usable = values.filter((value): value is number => value !== null && Number.isFinite(value));
  if (!usable.length) return <div className="mini-chart"><h3>{title}</h3><p className="na">N/A — data unavailable</p></div>;
  const min = Math.min(...usable), max = Math.max(...usable), bins = 24;
  const counts = Array.from({ length: bins }, () => 0);
  for (const value of usable) counts[Math.min(bins - 1, Math.floor(((value - min) / (max - min || 1)) * bins))]++;
  const high = Math.max(...counts, 1);
  return <div className="mini-chart"><h3>{title}</h3><svg viewBox="0 0 320 150" role="img" aria-label={title}>
    <line x1="30" y1="125" x2="310" y2="125" className="axis" />
    {counts.map((count, index) => <rect key={index} x={30 + index * 280 / bins} y={125 - count / high * 105}
      width={Math.max(2, 280 / bins - 1)} height={count / high * 105} className="hist-bar" />)}
    <text x="28" y="145">{formatMetric(min, Math.abs(min) < 1e-4 ? "beta" : "default")}</text>
    <text x="270" y="145">{formatMetric(max, Math.abs(max) < 1e-4 ? "beta" : "default")}</text>
  </svg></div>;
}
