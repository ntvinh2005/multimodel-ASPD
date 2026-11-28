import type { EvalModel } from "./negationWorkbenchData";

export type Cluster = {
  id: string;
  members: string[];
  medoid: string;
  outlier: string;
  withinDistance: number | null;
};

export type Component = { id: string; model: EvalModel; gate: string; beta: number; uNorm: number; sigma: number };
export type Partition = {
  componentOrder: string[];
  clusters: Cluster[];
  tree: [number, number, number, number][];
  distances: number[][];
  heldoutDistances: number[][];
  evaluation: {
    heldoutCoassignmentJaccard: number | null;
    heldout: { mean: number | null; pairs: number };
    meanEmbedding: { mean: number | null };
    sharedGate: { mean: number | null; pairs: number };
    random: { mean: number; quantiles: [number, number] } | null;
  };
};
export type Scope = {
  anchor: string;
  document: string;
  source: string;
  pool: number;
  boundary: number;
  prefix: string;
  native?: Record<EvalModel, [number, number, number, number]>;
  positions?: Record<EvalModel, number>;
  tokenIds?: Record<EvalModel, number>;
  split?: string;
};
export type Example = {
  anchor: string;
  document: string;
  coefficient: number;
  strength: number;
  heatmap?: [string, number, number, number, number, number, number][];
};
export type Evidence = {
  runId: string;
  examples: Record<string, Example[]>;
  witnesses: Record<string, Scope>;
};
export type Taylor = {
  from: string;
  to: string;
  model: EvalModel;
  anchor: string;
  scope: Scope;
  status: "complete" | "truncated";
  score: number | null;
  observedTargetMean: number | null;
  eligiblePairs: number;
  observedPairs: number;
  targets: { target: string; signed: number; weight: number; effects: Record<string, number> }[];
};
export type Match = {
  left: string;
  right: string;
  discovery: number;
  heldout: number | null;
  accepted: boolean;
  heldoutSameDocumentMass?: number;
  heldoutExactAnchorMass?: number;
  heldoutSameTerminalMass?: number;
  heldoutNoSameDocument?: number | null;
  transported?: { left: string; right: string; mass: number }[];
};
export type Ref = { file: string; sha256: string };
export type SupernodeRelease = {
  schemaVersion: 1;
  runId: string;
  checkpoint: string;
  catalog: Ref;
};
export type SupernodeCatalog = {
  schemaVersion: 1;
  runId: string;
  checkpoint: string;
  dataset: string;
  models: Record<EvalModel, Partition>;
  components: Component[];
  reference: { id: string; dimensions: number; scope: string; groundCost: string };
  settings: { support: number; epsilon: number | null };
  counts: Record<string, Record<string, { distinctNonzeroDocuments: number; exportedNonzero: number }>>;
  taylor: { status: string; witnesses: Taylor[]; matchedTargetMeasurements?: number };
  matches: Match[];
  exactAblation: { status: string; result?: { measurements: {
    model: EvalModel; anchor: string; pool: number; document: string; from: string; to: string;
    condition: string; taylor: { score: number | null };
    exact: { supernode: number[] | null; js?: number[]; functional?: Record<string, number[]> };
  }[] } };
  examples: Record<string, Ref>;
  transport: Ref;
};
export const SUPERNODE_PATH = "semantic-supernodes";
const digest = /^[a-f0-9]{64}$/;

export function releasePath(run: string, ref: Ref | undefined) {
  return ref && digest.test(run) && digest.test(ref.sha256)
    && /^[a-z0-9-]+\.json(?:\.gz)?$/.test(ref.file)
    ? `${SUPERNODE_PATH}/${run}/${ref.file}` : undefined;
}

export function partitionAt(partition: Partition, { count, threshold }: { count?: number; threshold?: number }) {
  const n = partition.componentOrder.length;
  const active = new Map<number, number[]>(partition.componentOrder.map((_, i) => [i, [i]]));
  for (const [i, [left, right, distance]] of partition.tree.entries()) {
    if (count !== undefined && active.size <= count) break;
    if (threshold !== undefined && distance > threshold) break;
    const a = active.get(left);
    const b = active.get(right);
    if (!a || !b) throw new Error("Invalid clustering merge");
    active.delete(left);
    active.delete(right);
    active.set(n + i, [...a, ...b]);
  }
  return [...active.values()].sort((a, b) => b.length - a.length || a[0] - b[0]);
}

export function measuredTaylorAt(rows: Taylor[], model: EvalModel, anchor: string, threshold: number) {
  return rows.filter(row => row.model === model && row.anchor === anchor
    && (row.score ?? row.observedTargetMean ?? -1) >= threshold);
}
