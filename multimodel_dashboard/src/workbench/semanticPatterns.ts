import { componentIdentity } from "./circuitComposition";
import type { Witness } from "./SemanticDiscovery";

export type PatternEdge = { target: string; source: string };
export type WitnessAt = (component: string, anchor: string) => Witness | undefined;

function ranked(counts: Map<string, number>) {
  return [...counts].map(([id, count]) => ({ id, count }))
    .sort((a, b) => b.count - a.count || a.id.localeCompare(b.id));
}

function measureParents(witness: Witness) {
  return new Set(witness.edges.filter(edge => edge.target === witness.component).map(edge => edge.source));
}

export function recurringParents(ids: readonly string[], target: string, witnessAt: WitnessAt) {
  let measured = 0;
  const counts = new Map<string, number>();
  for (const anchor of ids) {
    const witness = witnessAt(target, anchor);
    if (!witness) continue;
    measured++;
    for (const source of measureParents(witness)) counts.set(source, (counts.get(source) ?? 0) + 1);
  }
  return { measured, rows: ranked(counts) };
}

export function sharedParentGates(ids: readonly string[], targets: readonly [string, string], witnessAt: WitnessAt) {
  let measured = 0;
  const counts = new Map<string, number>();
  for (const anchor of ids) {
    const left = witnessAt(targets[0], anchor), right = witnessAt(targets[1], anchor);
    if (!left || !right) continue;
    measured++;
    const gates = (witness: Witness) => new Set([...measureParents(witness)].map(source => componentIdentity(source).gateId));
    const rightIds = gates(right);
    for (const gate of gates(left)) {
      if (rightIds.has(gate)) counts.set(gate, (counts.get(gate) ?? 0) + 1);
    }
  }
  return { measured, rows: ranked(counts) };
}

export function anchorsForPattern(ids: readonly string[], edges: readonly PatternEdge[], witnessAt: WitnessAt) {
  const targets = [...new Set(edges.map(edge => edge.target))];
  const present: string[] = [], contrast: string[] = [];
  if (!targets.length) return { present, contrast, unavailable: 0 };

  let unavailable = 0;
  for (const anchor of ids) {
    const found = new Map(targets.map(target => [target, witnessAt(target, anchor)]));
    if ([...found.values()].some(witness => !witness)) {
      unavailable++;
      continue;
    }
    const patternPresent = edges.every(({ target, source }) => found.get(target)?.edges.some(edge =>
      edge.source === source && edge.target === target,
    ));
    (patternPresent ? present : contrast).push(anchor);
  }
  return { present, contrast, unavailable };
}
