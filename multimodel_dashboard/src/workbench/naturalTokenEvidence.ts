import { componentIdentity } from "./circuitComposition";
import type { ExpandedSnapshot } from "./expandedNaturalValidation";
import type { EvalModel } from "./negationWorkbenchData";

export const naturalTokenMetrics = {
  gate: "Shared gate activation gˢ",
  reading: "Ungated reading vᵀx",
  coefficient: "Signed coefficient e",
  magnitude: "Contribution magnitude ‖eu‖₂",
};
export type NaturalTokenMetric = keyof typeof naturalTokenMetrics;
export type NaturalScalar = { reading: number; coefficient: number; magnitude: number; normalized: number };
export type NaturalTokenPoint = {
  anchor: string;
  pool: number;
  characterBoundary: number | null;
  positions: Partial<Record<EvalModel, number>>;
  gates: Record<string, number>;
  graphCoverage: "unavailable";
};
export type NaturalTokenWindow = {
  documentId: string;
  source: string;
  center: number;
  tokens: string[];
  tokenIds: number[];
  points: Record<number, NaturalTokenPoint>;
};
export type NaturalTokenAsset = {
  schemaVersion: 1;
  checkpoint: string;
  candidate: string;
  dataset: string;
  split: "test";
  provenance: { archiveManifestSha256: string; candidateEvidenceSha256: string; poolSize: number };
  examples: Record<string, NaturalTokenWindow>;
};
export type NaturalLocal = { source: string; target: string; before: number; after: number; delta: number };
export type NaturalScalarAsset = {
  schemaVersion: 1;
  checkpoint: string;
  dataset: string;
  split: "test";
  archiveManifestSha256: string;
  rows: Record<string, { pool: number; position: number; components: Record<string, NaturalScalar>; local?: NaturalLocal[] }>;
};

function replayAt(asset: NaturalTokenAsset, replay: NaturalScalarAsset | undefined,
  point: NaturalTokenPoint, component: string) {
  if (!replay || replay.schemaVersion !== 1 || replay.checkpoint !== asset.checkpoint
    || replay.dataset !== asset.dataset || replay.split !== asset.split
    || replay.archiveManifestSha256 !== asset.provenance.archiveManifestSha256) return undefined;
  const row = replay.rows[point.anchor];
  const { modelId } = componentIdentity(component);
  const position = point.positions[modelId as EvalModel];
  if (!row || row.pool !== point.pool || position === undefined || row.position !== position) return undefined;
  return row;
}

export function naturalScalar(asset: NaturalTokenAsset, replay: NaturalScalarAsset | undefined,
  point: NaturalTokenPoint, component: string): NaturalScalar | undefined {
  const value = replayAt(asset, replay, point, component)?.components[component];
  const { gateId } = componentIdentity(component);
  if (!value || !Object.values(value).every(Number.isFinite)) return undefined;
  return value.coefficient === (point.gates[gateId] > 0 ? value.reading : 0)
    && value.magnitude >= 0 && value.normalized >= 0
    && (value.coefficient !== 0 || value.magnitude === 0 && value.normalized === 0) ? value : undefined;
}

export function naturalLocal(asset: NaturalTokenAsset, replay: NaturalScalarAsset | undefined,
  point: NaturalTokenPoint, component: string) {
  return (replayAt(asset, replay, point, component)?.local ?? []).filter(({ source, target, before, after, delta }) =>
    (source === component || target === component) && [before, after, delta].every(Number.isFinite));
}

export function naturalMetric(metric: NaturalTokenMetric, point: NaturalTokenPoint | undefined, gate: string,
  scalar?: NaturalScalar) {
  if (!point) return undefined;
  if (metric === "gate") return point.gates[gate] ?? 0;
  if (metric === "reading") return scalar?.reading;
  return scalar?.[metric] ?? (point.gates[gate] === undefined ? 0 : undefined);
}

export function naturalScale(values: (number | undefined)[]) {
  return Math.max(0, ...values.map(value => Math.abs(value ?? 0)));
}

export type TestTokenScope = { anchor: string; pool: number; dataset: string; archiveManifestSha256: string };
export type NaturalTokenGraphAsset = TestTokenScope & {
  schemaVersion: 1;
  checkpoint: string;
  split: "test";
  coverage: "full" | "partial";
  snapshot: ExpandedSnapshot;
};

export function matchingNaturalGraph(checkpoint: string, scope: TestTokenScope | undefined,
  graph: NaturalTokenGraphAsset | undefined) {
  return scope && graph?.schemaVersion === 1 && graph.checkpoint === checkpoint && graph.split === "test"
    && graph.anchor === scope.anchor && graph.snapshot?.anchor === scope.anchor && graph.pool === scope.pool
    && graph.dataset === scope.dataset && graph.archiveManifestSha256 === scope.archiveManifestSha256
    && (graph.coverage === "full" || graph.coverage === "partial") ? graph : undefined;
}
