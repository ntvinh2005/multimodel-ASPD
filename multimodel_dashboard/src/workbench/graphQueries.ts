import type { Component, DependencyEdge, TraceDirection, TraceResult } from "./types";

export function searchSharedGates<T extends { id: string }>(gates: T[], query: string): T[] {
  const term = query.trim().toUpperCase().replace(/^(PRE|MID)[\s:/_-]*/, "$1:");
  if (!term) return gates;
  const numeric = /^\d+$/.test(term);
  // Stable sort preserves the view's existing order among equally good matches.
  return gates.flatMap((gate) => {
    const key = numeric ? gate.id.split(":")[1] : gate.id;
    if (!key?.includes(term)) return [];
    const rank = key === term ? 0 : key.startsWith(term) ? 1 : 2;
    return [{ gate, rank }];
  }).sort((left, right) => left.rank - right.rank).map(({ gate }) => gate);
}

export function componentsForSelection(components: Component[], gateIds: Set<string>, modelIds: Set<string>, representativeOnly: boolean) {
  return components.filter((component) => gateIds.has(component.gateId) && modelIds.has(component.modelId) && (!representativeOnly || component.representative));
}

export function edgesForAnchor(edges: DependencyEdge[], anchorId: string, visibleComponentIds: Set<string>, threshold: number) {
  return edges.filter((edge) => edge.anchorId === anchorId && visibleComponentIds.has(edge.source) && visibleComponentIds.has(edge.target) && edge.normalizedMagnitude !== null && edge.normalizedMagnitude >= threshold);
}

export function traceGraph(edges: DependencyEdge[], startId: string, direction: TraceDirection, maxDepth: number): TraceResult {
  const nodes = new Set([startId]);
  const selectedEdges = new Set<string>();
  const distance = new Map([[startId, 0]]);
  let frontier = [startId];
  for (let depth = 1; depth <= maxDepth && frontier.length; depth += 1) {
    const next: string[] = [];
    for (const nodeId of frontier) {
      for (const edge of edges) {
        const incoming = (direction === "parents" || direction === "both") && edge.target === nodeId;
        const outgoing = (direction === "children" || direction === "both") && edge.source === nodeId;
        if (!incoming && !outgoing) continue;
        const neighbor = incoming ? edge.source : edge.target;
        selectedEdges.add(edge.id);
        if (!nodes.has(neighbor)) {
          nodes.add(neighbor);
          distance.set(neighbor, depth);
          next.push(neighbor);
        }
      }
    }
    frontier = next;
  }
  return { nodes, edges: selectedEdges, distance };
}

export function normalizedDistribution(raw: Record<string, number>): { status: "measured" | "undefined"; values: Record<string, number> | null; total: number } {
  const total = Object.values(raw).reduce((sum, value) => sum + value, 0);
  if (total <= 0) return { status: "undefined", values: null, total };
  return { status: "measured", values: Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, value / total])), total };
}

export function jaccard(left: Record<string, number> | null, right: Record<string, number> | null): number | null {
  if (!left || !right) return null;
  const a = new Set(Object.keys(left)); const b = new Set(Object.keys(right));
  const union = new Set([...a, ...b]);
  if (!union.size) return null;
  return [...a].filter((key) => b.has(key)).length / union.size;
}

export function tvDistance(left: Record<string, number> | null, right: Record<string, number> | null): number | null {
  if (!left || !right) return null;
  const keys = new Set([...Object.keys(left), ...Object.keys(right)]);
  return [...keys].reduce((sum, key) => sum + Math.abs((left[key] ?? 0) - (right[key] ?? 0)), 0) / 2;
}
