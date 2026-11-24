import rawDataset from "./data/inductionGateDataset.json";

export type InductionCondition = "induction" | "counterfactual" | "no_match" | "repeated_no_association";

export interface InductionGate {
  id: string;
  site: "PRE" | "MID";
  feature: number;
  contrast: number;
  meanActivation: number;
  meanAnnotatedActivation: number;
  sequenceFires: Record<InductionCondition, number>;
  finalQueryFires: Record<InductionCondition, number>;
  finalQueryActivationSum: Record<InductionCondition, number>;
  betaByModel: Record<string, { max: number; mean: number }>;
}

export interface InductionPosition {
  boundary: number;
  activation: number;
  finalQuery: boolean;
  roles: string[];
}

export interface InductionContext {
  exampleId: string;
  groupId: string;
  family: string;
  condition: InductionCondition;
  prompt: string;
  expectedContinuation: string;
  nativePredictions: Record<string, { predicted_token: string; logit_margin: number }>;
  spans: Array<{ role: string; start: number; stop: number }>;
  positions: InductionPosition[];
}

export interface InductionComponent {
  id: string;
  model: string;
  layer: number;
  matrix: "Q" | "K" | "V" | "O" | "gate" | "up" | "down";
  beta: number | null;
  representative: boolean;
}

export interface MeasuredInductionSplit {
  status: "measured";
  split: "smoke" | "discovery";
  sourceRevision: string;
  evaluationScope: string;
  artifactRoot: string;
  manifestSha256: string;
  groups: number;
  examples: number;
  pools: number;
  gates: InductionGate[];
  contextGateIds: string[];
  contexts: Record<string, InductionContext[]>;
  components: Record<string, InductionComponent[]>;
}

export interface PendingInductionSplit { status: "pending" }
export interface InductionGateDataset {
  schemaVersion: 1;
  label: string;
  checkpointStep: number;
  checkpointPath: string;
  scientificBoundary: string;
  referencePosthocRoot: string;
  referenceImportanceCompleteSha256: string;
  splits: { smoke: MeasuredInductionSplit; discovery: MeasuredInductionSplit | PendingInductionSplit };
}

export function validateInductionGateDataset(value: unknown): InductionGateDataset {
  const dataset = value as Partial<InductionGateDataset>;
  if (dataset.schemaVersion !== 1 || dataset.checkpointStep !== 61036) throw new Error("unsupported induction gate bundle");
  if (dataset.splits?.smoke?.status !== "measured") throw new Error("induction smoke split is not measured");
  const smoke = dataset.splits.smoke;
  if (smoke.groups !== 32 || smoke.examples !== 128 || smoke.pools !== 2) throw new Error("induction smoke denominators changed");
  if (!smoke.gates.length || smoke.gates.some((gate) => !/^(PRE|MID):\d+$/.test(gate.id))) throw new Error("induction gate ids are malformed");
  if (smoke.contextGateIds.some((gateId) => !smoke.contexts[gateId] || !smoke.components[gateId])) throw new Error("bounded induction contexts/components are incomplete");
  for (const gateId of smoke.contextGateIds) {
    for (const context of smoke.contexts[gateId]) {
      if (!context.positions.length || context.positions.at(-1)?.boundary !== context.prompt.length) throw new Error("induction context lost its final canonical boundary");
    }
  }
  return value as InductionGateDataset;
}

export const inductionGateDataset = validateInductionGateDataset(rawDataset);
