/// <reference types="vite/client" />

import type { QueryGraph, QueryTarget } from "./InductionQueryGraph";
import type { MatrixType } from "./types";

export const NEGATION_LABELS = {
  affirmative: "Affirmative",
  negative: "Negative",
  counterfactual_affirmative: "Counterfactual affirmative",
  counterfactual_negative: "Counterfactual negative",
  scope_control: "Scope control",
  counterfactual_scope_control: "Counterfactual scope control",
};
export const FAMILY_LABELS = {
  binary: "Binary exclusion", attribute: "Attribute negation",
  multiple: "Multiple alternatives", scope: "Scope control",
};
export const CONDITION_OUTCOMES = {
  successful_negation: "Successful negation",
  negation_specific_failure: "Repeated the excluded answer",
  affirmative_task_failure: "Affirmative prerequisite failed",
  unrelated_prediction_error: "Unrelated prediction",
  invalid_evaluation: "Invalid measurement",
};
export const MODEL_NAMES = { gemma: "Gemma 2", smol: "SmolLM2", granite: "Granite 3.3" };
export const MODEL_IDS = ["gemma", "smol", "granite"] as const;
export type EvalModel = typeof MODEL_IDS[number];
export type NegationCondition = keyof typeof NEGATION_LABELS;
export type Outcome = keyof typeof CONDITION_OUTCOMES;
export type Family = keyof typeof FAMILY_LABELS;

export type NegationGate = {
  id: string;
  contrast: number;
  referenceBeta: Record<EvalModel, number | null>;
  conditions: Partial<Record<NegationCondition, {
    sequenceFires: number;
    queryFires: number;
    queryMean: number;
    mean: number;
    meanWhenFiring: number | null;
    sd: number;
    roles: Record<string, number>;
  }>>;
};
export type GroupInfo = {
  path: string;
  family: Family;
  conditions: NegationCondition[];
  outcomes: Partial<Record<NegationCondition, Record<EvalModel, Outcome>>>;
};
export type NegationSplit = {
  groups: Record<string, GroupInfo>;
  gateGroups: Record<string, string[]>;
  selection: Record<string, { kind: string; groups: string[] }>;
  graphProvenance: { discoveryManifestSha256: string; runtime: { slurm_job_id: string } } | null;
  denominators: Record<string, { examples: number; queries: number; anchors: number }>;
  gates: NegationGate[];
  poolCount: number;
  manifestSha256: string;
};
export type NegationIndex = {
  schemaVersion: number;
  mechanism: "negation";
  checkpointStep: number;
  checkpointSha256: string;
  splits: Record<"smoke" | "discovery", NegationSplit>;
  assetHashes: Record<string, string>;
};
export type MatrixLocator = {
  model: EvalModel;
  layer: number;
  site: "PRE" | "MID";
  matrix: MatrixType;
  beta: Array<number | null>;
  representatives: number[];
};
type Support = { active_gate_ids: number[]; values: number[] };
export type NegationPosition = {
  anchor_id: string;
  character_boundary: number;
  roles_ending: string[];
  final_query: boolean;
  model_positions: Record<EvalModel, number>;
  pre: Support;
  mid: Support;
};
export type NativePrediction = {
  predicted_token: string;
  fixed_margin: number;
  correctness_margin: number;
  exact_correct: boolean;
  candidate_correct: boolean;
  valid_mass: number;
  excluded_mass: number;
  unrelated_mass: number;
  negation_outcome?: Outcome;
  candidates: Record<string, { logit: number; probability: number }>;
};
export type NegationExample = {
  example_id: string;
  group_id: string;
  pool_id: string;
  prompt: string;
  condition: NegationCondition;
  candidate_orientation: [string, string];
  valid_alternatives: string[];
  excluded_candidates: string[];
  positions: NegationPosition[];
  predictions: Record<EvalModel, NativePrediction>;
  graphs?: Record<string, QueryGraph>;
};
export type NegationGroup = { mechanism: "negation"; groupId: string; examples: NegationExample[] };
export type ExpandedNegationIndex = {
  schemaVersion: 1;
  mechanism: "negation";
  split: "discovery";
  checkpointSha256: string;
  discoveryManifestSha256: string;
  groups: Record<string, { path: string; sha256: string; gates: string[]; jobId: string }>;
};
type ExpandedQuery = {
  exampleId: string;
  groupId: string;
  condition: NegationCondition;
  anchorId: string;
  poolId: string;
  modelPositions: Record<EvalModel, number>;
  graphs: Record<string, QueryGraph>;
  traceTargets: Record<string, QueryTarget>;
};
export type ExpandedNegationGroup = Pick<ExpandedNegationIndex,
  "schemaVersion" | "mechanism" | "split" | "checkpointSha256" | "discoveryManifestSha256"
> & { groupId: string; examples: ExpandedQuery[] };

export function readNegationAsset<T>(path: string, signal: AbortSignal, hash?: string): Promise<T> {
  return readWorkbenchAsset<T>(`negation/${path}`, signal, hash);
}

export async function readWorkbenchAsset<T>(path: string, signal: AbortSignal, hash?: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}${path}`, { signal });
  const compressed = path.endsWith(".json.gz");
  const contentType = response.headers.get("content-type") ?? "";
  if (!response.ok || !(compressed ? /gzip|octet-stream/.test(contentType) : contentType.includes("json"))) {
    throw new Error("This build has no measured artifact at the requested path.");
  }
  const bytes = await response.arrayBuffer();
  if (hash && globalThis.crypto?.subtle) {
    const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
      .map((value) => value.toString(16).padStart(2, "0")).join("");
    if (digest !== hash) throw new Error("Measured artifact hash mismatch.");
  }
  if (compressed) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text()) as T;
  }
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

export function validateNegationIndex(index: NegationIndex) {
  const discovery = index.splits?.discovery;
  if (index.mechanism !== "negation" || index.schemaVersion !== 1 || index.checkpointStep !== 61036
      || !index.splits?.smoke.gates.length || !discovery?.gates.length
      || discovery.graphProvenance?.discoveryManifestSha256 !== discovery.manifestSha256) {
    throw new Error("Negation index and measured discovery graphs disagree.");
  }
  return index;
}

export function graphAtFinalQuery(example: Pick<NegationExample, "positions" | "graphs">, gate: string) {
  const graph = example.graphs?.[gate];
  if (!graph) return undefined;
  const final = example.positions.find((item) => item.final_query);
  if (graph.anchorId !== final?.anchor_id || graph.activation !== activationAt(final, gate)) {
    throw new Error("Graph and full-sequence activation record disagree at the final query.");
  }
  return graph;
}

export function matchesExpandedDiscovery(
  expansion: Pick<ExpandedNegationIndex, "schemaVersion" | "mechanism" | "split" | "checkpointSha256" | "discoveryManifestSha256">,
  index: NegationIndex,
) {
  return expansion.schemaVersion === 1 && expansion.mechanism === "negation" && expansion.split === "discovery"
    && expansion.checkpointSha256 === index.checkpointSha256
    && expansion.discoveryManifestSha256 === index.splits.discovery.manifestSha256;
}

export function expandedAtFinalQuery(
  group: ExpandedNegationGroup,
  example: NegationExample,
  gate: string,
  model: EvalModel,
) {
  const row = group.examples.find((item) => item.exampleId === example.example_id);
  const final = example.positions.find((item) => item.final_query);
  if (!row || !final || group.groupId !== example.group_id || row.groupId !== example.group_id
      || row.condition !== example.condition || row.poolId !== example.pool_id
      || row.anchorId !== final.anchor_id
      || MODEL_IDS.some((key) => row.modelPositions[key] !== final.model_positions[key])) {
    throw new Error("Expanded dependency and activation records disagree on the example or anchor.");
  }
  const graph = graphAtFinalQuery({ positions: example.positions, graphs: row.graphs }, gate);
  if (!graph && activationAt(final, gate) > 0) {
    throw new Error("An active gate is missing from the comprehensive artifact.");
  }
  const targets = new Map<string, { target: QueryTarget; representative: boolean }>();
  for (const candidate of Object.values(row.graphs)) {
    if (candidate.anchorId !== final.anchor_id) throw new Error("Mixed dependency anchors.");
    for (const target of candidate.models[model]?.targets ?? []) {
      if (target.component.modelId !== model || targets.has(target.component.id)) {
        throw new Error("Invalid representative component identity in expanded graph.");
      }
      targets.set(target.component.id, { target, representative: true });
    }
  }
  for (const [id, target] of Object.entries(row.traceTargets)) {
    if (target.component.modelId !== model) continue;
    if (id !== target.component.id || targets.has(id)) throw new Error("Invalid supplemental component identity.");
    targets.set(id, { target, representative: false });
  }
  return { graph, targets };
}

export function activationAt(position: NegationPosition, gate: string): number {
  const site = gate.startsWith("PRE:") ? position.pre : position.mid;
  const index = site.active_gate_ids.indexOf(Number(gate.split(":")[1]));
  return index === -1 ? 0 : site.values[index];
}
