import type { QueryGraph, QueryTarget } from "./InductionQueryGraph";
import { activationAt, type MatrixLocator, type NegationPosition } from "./negationWorkbenchData";
import type { MatrixType } from "./types";

export type AssetRef = { path: string; sha256: string };
export type TokenIndex = {
  checkpointSha256: string;
  datasets: Record<string, Record<string, AssetRef>>;
  graphs: Record<string, AssetRef>;
};
export type TokenExample = {
  example_id: string;
  group_id: string;
  condition: string;
  prompt: string;
  positions: (NegationPosition & { pool_id: string })[];
};
export type TokenGroup = { checkpointSha256: string; dataset: string; groupId: string; examples: TokenExample[] };
export type TokenGraphAsset = {
  checkpointSha256: string;
  anchorId: string;
  exampleId: string;
  poolId: string;
  official: Record<string, QueryGraph>;
  componentTargets: Record<string, [number, number, [string, number][]]>;
  outputScale: Record<string, number>;
  completeComponents: boolean;
  supplementalTargets?: Record<string, QueryTarget>;
};

export function selectToken(example: TokenExample | undefined, anchorId: string) {
  return example?.positions.find((row) => row.anchor_id === anchorId)
    ?? example?.positions.find((row) => row.final_query);
}

export function tokenGates(position?: NegationPosition) {
  return position ? (["pre", "mid"] as const).flatMap((site) => position[site].active_gate_ids.map((id, index) => ({
    id: `${site.toUpperCase()}:${id}`, value: position[site].values[index],
  }))).sort((a, b) => b.value - a.value) : [];
}

export function tokenGraph(
  asset: TokenGraphAsset, example: TokenExample, anchor: TokenExample["positions"][number],
  checkpoint: string, gate: string, model: string, locator: MatrixLocator[] = [],
) {
  if (asset.anchorId !== anchor.anchor_id || asset.exampleId !== example.example_id
      || asset.poolId !== anchor.pool_id || asset.checkpointSha256 !== checkpoint) {
    throw new Error("Graph provenance differs from the selected token.");
  }
  const graph = asset.official[gate];
  if (!graph && !asset.completeComponents) return {};
  if (!graph || graph.anchorId !== anchor.anchor_id || graph.activation !== activationAt(anchor, gate)) {
    throw new Error("Measured graph and natural firing disagree for the selected token.");
  }
  const matrices = new Map(locator.map((row) => [`${row.model}/${row.layer}/${row.matrix}`, row]));
  const identity = (id: string): QueryTarget["component"] => {
    const parsed = /^([^.]+)\.L(\d+)\.(q|k|v|o|mlp_gate|mlp_up|mlp_down)\/(\d+)$/.exec(id);
    if (!parsed) throw new Error("Invalid component identity.");
    const [, modelId, layerRaw, matrixRaw, featureRaw] = parsed;
    const layer = Number(layerRaw);
    const matrix = (matrixRaw.startsWith("mlp_") ? matrixRaw.slice(4) : matrixRaw.toUpperCase()) as MatrixType;
    const pre = ["Q", "K", "V"].includes(matrix);
    const feature = Number(featureRaw);
    return { id, modelId, layer, matrix, gateId: `${pre ? "PRE" : "MID"}:${feature}`,
      beta: matrices.get(`${modelId}/${layer}/${matrix}`)?.beta[feature] ?? null, rho: null };
  };
  const targets = new Map<string, { target: QueryTarget; representative: boolean }>();
  for (const other of Object.values(asset.official)) {
    if (other.anchorId !== anchor.anchor_id) throw new Error("Mixed graph anchors.");
    for (const target of other.models[model]?.targets ?? []) {
      targets.set(target.component.id, { target, representative: true });
    }
  }
  for (const [id, target] of Object.entries(asset.supplementalTargets ?? {})) {
    if (target.component.modelId === model) targets.set(id, { target, representative: false });
  }
  for (const [id, [coefficient, norm, parents]] of Object.entries(asset.componentTargets)) {
    if (!id.startsWith(model + ".") || targets.has(id)) continue;
    const component = identity(id);
    const scale = asset.outputScale[id.slice(0, id.lastIndexOf("/"))];
    if (!(scale > 0)) throw new Error("Missing matrix-output scale.");
    targets.set(id, { representative: false, target: {
      component, coefficient, contributionNorm: Math.abs(coefficient) * norm, outputScale: scale,
      parents: parents.map(([parent, effect]) => {
        const upstream = identity(parent);
        if (upstream.modelId !== model || !asset.componentTargets[parent]) throw new Error("Missing same-model active upstream component.");
        return { component: upstream, E: effect, w: Math.abs(effect) * norm / scale };
      }),
    } });
  }
  return { graph, targets };
}
