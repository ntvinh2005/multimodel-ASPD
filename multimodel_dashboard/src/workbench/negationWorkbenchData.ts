/// <reference types="vite/client" />

import type { QueryGraph } from "./InductionQueryGraph";
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

export async function readNegationAsset<T>(path: string, signal: AbortSignal, hash?: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}negation/${path}`, { signal });
  if (!response.ok || !response.headers.get("content-type")?.includes("json")) {
    throw new Error("This build has no measured Negation artifact at the requested path.");
  }
  const bytes = await response.arrayBuffer();
  if (hash && globalThis.crypto?.subtle) {
    const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
      .map((value) => value.toString(16).padStart(2, "0")).join("");
    if (digest !== hash) throw new Error("Negation artifact hash mismatch.");
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

export function activationAt(position: NegationPosition, gate: string): number {
  const site = gate.startsWith("PRE:") ? position.pre : position.mid;
  const index = site.active_gate_ids.indexOf(Number(gate.split(":")[1]));
  return index === -1 ? 0 : site.values[index];
}
