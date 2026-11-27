import { componentIdentity } from "./circuitComposition";
import catalog from "./data/expandedNaturalValidationCatalog.json";
import { searchSharedGates } from "./graphQueries";
import type { Component, DependencyEdge } from "./types";

export type ExpandedCandidate = typeof catalog.candidates[number];
type MeasuredEdge = { source: string; target: string; E: number; w: number; rank0: number }
  & ({ shard: string; record: number; flat: number } | { profileRow: number });
export type ExpandedSnapshot = { anchor: string; kind: string; edges: MeasuredEdge[]; representatives: string[] };
export type FiringExample = { anchor: string; source: string; activation: number; before: string; token: string; after: string };
export type ExpandedEvidence = {
  snapshots: ExpandedSnapshot[];
  hypothesis: string;
  alternative: string;
  test: string;
  documents: number;
  examples: Record<string, FiringExample[]>;
};

export const expandedCatalog = catalog;
const byId = new Map(catalog.candidates.map((candidate) => [candidate.id, candidate]));
const gates = [...new Set(catalog.candidates.flatMap((candidate) => candidate.gates))].map((id) => ({ id }));

export function getExpandedCandidate(id: string) {
  return byId.get(id);
}

export function expandedCandidateInNote(observation: string) {
  const id = /^\s*(NVX-\d{3})\b/.exec(observation)?.[1];
  return id ? byId.get(id) : undefined;
}

export function searchExpandedCandidates(query: string): ExpandedCandidate[] {
  const term = query.trim();
  if (!term) return [];
  const code = /^NVX[ _-]*(\d{0,3})$/i.exec(term);
  if (code) {
    const key = `NVX-${code[1]}`;
    const exact = byId.get(key);
    return exact ? [exact] : catalog.candidates.filter(({ id }) => id.startsWith(key));
  }
  if (/^(?:PRE|MID)[\s:/_-]*\d*$/i.test(term) || /^\d+$/.test(term)) {
    const matches = searchSharedGates(gates, term);
    const canonical = term.toUpperCase().replace(/^(PRE|MID)[\s:/_-]*/, "$1:");
    const exact = matches.filter(({ id }) => id === canonical || id.split(":")[1] === canonical);
    const eligible = new Set((exact.length ? exact : matches).map(({ id }) => id));
    return catalog.candidates.filter((candidate) => candidate.gates.some(id => eligible.has(id)));
  }
  const tokenize = (text: string) => text.toLowerCase().replaceAll("-", " ").trim().split(/\s+/);
  const words = tokenize(term);
  return catalog.candidates.filter(({ title }) => words.every(word => tokenize(title).some(part => part.includes(word))));
}

export async function loadExpandedEvidence(id: string): Promise<ExpandedEvidence> {
  const graphModule = await import("./data/expandedNaturalValidationGraphs.json");
  const graphs: Record<string, ExpandedEvidence> = graphModule.default;
  const graph = graphs[id];
  if (!graph) throw new Error(`No saved graph exists for ${id}.`);
  return graph;
}

export function renderableExpandedGraph(snapshot: ExpandedSnapshot) {
  const representatives = new Set(snapshot.representatives);
  const components: Component[] = [...new Set(snapshot.edges.flatMap(({ source, target }) => [source, target]))]
    .map((id) => ({ ...componentIdentity(id), beta: null, rho: null, representative: representatives.has(id) }));
  const index = new Map(components.map(component => [component.id, component]));
  const edges: DependencyEdge[] = snapshot.edges.map((edge) => {
    const modelId = index.get(edge.source)!.modelId;
    if (modelId !== index.get(edge.target)!.modelId) throw new Error("An expanded graph edge crosses models.");
    return {
      id: `${snapshot.anchor}:${edge.source}:${edge.target}`,
      anchorId: snapshot.anchor, modelId, source: edge.source, target: edge.target,
      signedEffect: edge.E, normalizedMagnitude: edge.w, upstreamCoefficient: null, gradient: null,
      normalization: "Stored normalized parent weight", evidence: "measured-taylor",
    };
  });
  return { components, edges };
}
