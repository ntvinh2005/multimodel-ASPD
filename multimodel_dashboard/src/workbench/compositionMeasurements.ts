import { componentIdentity, type CompositionEdge, type Measurement, type Scope } from "./circuitComposition";
import type { QueryTarget } from "./InductionQueryGraph";
import type { TokenExample, TokenGraphAsset } from "./tokenDiscovery";
import type { DependencyEdge } from "./types";

export function measurementFromGraph(scope: Scope, edges: readonly DependencyEdge[], targets?: Iterable<string>): Measurement {
  const measured = new Set(targets);
  const records = edges.flatMap(edge => {
    if (edge.anchorId !== scope.anchor || edge.signedEffect === null || edge.normalizedMagnitude === null
      || edge.evidence === "taylor-only") return [];
    measured.add(edge.target);
    return [{ source: edge.source, target: edge.target, signed: edge.signedEffect, weight: edge.normalizedMagnitude }];
  });
  return { scope, edges: records, targets: [...measured], complete: false };
}

export function measurementFromToken(asset: TokenGraphAsset, scope: Scope,
  example: TokenExample, anchor: TokenExample["positions"][number]): Measurement {
  if (asset.checkpointSha256 !== scope.checkpoint || asset.anchorId !== scope.anchor || asset.anchorId !== anchor.anchor_id
    || asset.exampleId !== scope.example || asset.exampleId !== example.example_id
    || asset.poolId !== scope.pool || asset.poolId !== anchor.pool_id) {
    throw new Error("The measured asset belongs to a different checkpoint, example, pool, or token.");
  }
  const rows = new Map<string, { coefficient: number; edges: CompositionEdge[] }>();
  function add(target: QueryTarget) {
    rows.set(target.component.id, {
      coefficient: target.coefficient,
      edges: target.parents.map(parent => ({
        source: parent.component.id, target: target.component.id, signed: parent.E, weight: parent.w,
      })),
    });
  }
  for (const graph of Object.values(asset.official)) {
    if (graph.anchorId !== scope.anchor) throw new Error("Parent records contain a different token.");
    for (const model of Object.values(graph.models)) model.targets.forEach(add);
  }
  Object.values(asset.supplementalTargets ?? {}).forEach(add);
  for (const [target, [coefficient, norm, parents]] of Object.entries(asset.componentTargets)) {
    if (rows.has(target)) continue;
    const scale = asset.outputScale[target.slice(0, target.lastIndexOf("/"))];
    if (!(scale > 0)) throw new Error("Missing matrix-output scale.");
    const model = componentIdentity(target).modelId;
    const edges = parents.map(([source, signed]) => {
      if (componentIdentity(source).modelId !== model || (asset.completeComponents && !asset.componentTargets[source])) {
        throw new Error("Parent component is outside the measured model or scope.");
      }
      return { source, target, signed, weight: Math.abs(signed) * norm / scale };
    });
    rows.set(target, { coefficient, edges });
  }
  return { scope, complete: asset.completeComponents, targets: [...rows.keys()],
    edges: [...rows.values()].flatMap(row => row.edges),
    coefficients: Object.fromEntries([...rows].map(([id, row]) => [id, row.coefficient])) };
}
