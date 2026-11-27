import type { ActivationExample, DependencyEdge } from "./types";

export interface GraphViewBox { x: number; y: number; w: number; h: number; }
export type ActivationContextView = "stacked" | "snippet" | "full";

export const DEFAULT_GRAPH_VIEWBOX: GraphViewBox = { x: 0, y: 0, w: 760, h: 510 };
const MIN_GRAPH_WIDTH = 380;
const MAX_GRAPH_WIDTH = 3040;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function zoomGraphViewBox(
  current: GraphViewBox,
  factor: number,
  anchorX = .5,
  anchorY = .5,
): GraphViewBox {
  const width = clamp(current.w * factor, MIN_GRAPH_WIDTH, MAX_GRAPH_WIDTH);
  const appliedFactor = width / current.w;
  const height = current.h * appliedFactor;
  const x = current.x + (current.w - width) * clamp(anchorX, 0, 1);
  const y = current.y + (current.h - height) * clamp(anchorY, 0, 1);
  return { x, y, w: width, h: height };
}

export function panGraphViewBox(
  origin: GraphViewBox,
  deltaClientX: number,
  deltaClientY: number,
  viewportWidth: number,
  viewportHeight: number,
): GraphViewBox {
  if (viewportWidth <= 0 || viewportHeight <= 0) return origin;
  return {
    ...origin,
    x: origin.x - deltaClientX * origin.w / viewportWidth,
    y: origin.y - deltaClientY * origin.h / viewportHeight,
  };
}

function quantile(sorted: number[], fraction: number) {
  if (sorted.length === 0) return 0;
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

export function edgeThresholdScale(edges: DependencyEdge[]) {
  const values = edges
    .map((edge) => edge.normalizedMagnitude)
    .filter((value): value is number => value !== null && Number.isFinite(value) && value >= 0)
    .sort((left, right) => left - right);
  const maximum = values.at(-1) ?? 0;
  return {
    count: values.length,
    maximum,
    median: quantile(values, .5),
    upperQuartile: quantile(values, .75),
    step: maximum > 0 ? Math.max(maximum / 240, 1e-7) : 1e-7,
  };
}

export function formatEdgeMagnitude(value: number) {
  if (value === 0) return "0";
  if (Math.abs(value) < 1e-4) return value.toExponential(2);
  return value.toFixed(5);
}

export function activationWindow(example: ActivationExample, view: ActivationContextView) {
  if (view === "full") return { start: 0, end: example.activations.length };
  const anchors = view === "stacked"
    ? [example.centerPosition]
    : example.firings.flatMap((firing, position) => firing ? [position] : []);
  const positions = anchors.length > 0 ? anchors : [example.centerPosition];
  const radius = view === "stacked" ? 6 : 8;
  return {
    start: Math.max(0, Math.min(...positions) - radius),
    end: Math.min(example.activations.length, Math.max(...positions) + radius + 1),
  };
}

export function activationOpacity(activation: number, featureMaximum: number) {
  if (activation <= 0 || featureMaximum <= 0) return 0;
  return .06 + Math.min(1, activation / featureMaximum) * .88;
}
