export type GateStage = "PRE" | "MID";
export type MatrixType = "Q" | "K" | "V" | "O" | "gate" | "up" | "down";
export type Availability = "measured" | "missing" | "undefined" | "filtered";

export interface DatasetManifest {
  schemaVersion: "1.0";
  datasetId: string;
  label: string;
  synthetic: boolean;
  sourceRevision: string;
  trainingBranch: string;
  trainingConfig: string;
  specification: { status: "provided-in-request" | "local-pdf"; path: string | null; sha256?: string };
  dataset: { id: string; revision: string; splitPolicy: string };
  screening: { metric: "beta"; kRep: number; tauRep: number; method: string };
  edgeScoring: { method: string; normalization: string };
  display?: {
    defaultGateIds: string[];
    defaultSpotlight: string;
    defaultAnchorId: string;
    defaultEdgeThreshold: number;
  };
  export?: {
    artifactRoot: string;
    checkpointStep: number;
    selectionRule: string;
    graphBound: string;
    selectedScenarios: Array<{
      gateId: string;
      rawGateId: string;
      anchorId: string;
      displayParentGateIds: string[];
      edgeShard: string;
    }>;
    sourceHashes: Record<string, unknown>;
    parentBackend?: {
      stage: "profiles";
      status: string;
      sourceRevision: string;
      slurmJobId: string;
      parentRows: number;
      distributionSource: string;
      parentSetSource: string;
      rule: { k_par: number; tau: number };
    };
    activationExamples?: {
      artifactRoot: string;
      checkpointStep: number;
      checkpointManifestSha256: string;
      evaluationGitSha: string;
      experimentFamily: string;
      extractionManifestSha256: string;
      featureContextSha256: Record<string, string>;
      naturalGateOnly: boolean;
      selectedSplit: string;
      splitUnit: string;
      windowTokens: number;
      relationToGraph: string;
    };
    verification: string;
    limitations: string[];
  };
}

export interface ModelMetadata {
  id: string;
  label: string;
  checkpoint: string;
  tokenizer: string;
  selectedLayers: number[];
}

export interface SharedGate { id: string; stage: GateStage; featureIndex: number; label: string; }

export interface AnchorRecord {
  id: string;
  label: string;
  text: string;
  start: number;
  end: number;
  gateActivations: Record<string, number | null>;
  firingGateIds?: string[];
  modelTokens: Record<string, Array<{ text: string; position: number; isAnchor: boolean }>>;
}

export interface Component {
  id: string;
  modelId: string;
  gateId: string;
  layer: number;
  matrix: MatrixType;
  beta: number | null;
  rho: number | null;
  representative: boolean;
}

export interface DependencyEdge {
  id: string;
  modelId: string;
  anchorId: string;
  source: string;
  target: string;
  signedEffect: number | null;
  normalizedMagnitude: number | null;
  upstreamCoefficient: number | null;
  gradient: number | null;
  normalization: string;
  evidence: "taylor-only" | "measured-taylor" | "intervention-available";
}

export interface ParentDistribution {
  gateId: string;
  modelId: string;
  anchorId: string;
  status: Availability;
  totalStrength: number | null;
  values: Record<string, number> | null;
}

export interface ActivationExample {
  activation: number;
  activations: number[];
  anchorId: string;
  anchorIndex: number;
  centerPosition: number;
  context: string;
  documentId: string;
  firings: boolean[];
  sourceGroup: string;
}

export interface GateActivationExamples {
  gateId: string;
  site: GateStage;
  featureIndex: number;
  examples: ActivationExample[];
}

export interface WorkbenchDataset {
  manifest: DatasetManifest;
  models: ModelMetadata[];
  gates: SharedGate[];
  anchors: AnchorRecord[];
  components: Component[];
  edges: DependencyEdge[];
  parentDistributions: ParentDistribution[];
  activationExamples?: GateActivationExamples[];
}

export type TraceDirection = "parents" | "children" | "both";
export interface TraceResult { nodes: Set<string>; edges: Set<string>; distance: Map<string, number>; }

export interface ResearchHypothesis {
  id: string;
  title: string;
  notes: string;
  status: "Unreviewed" | "Interesting candidate" | "Needs validation" | "Inconclusive" | "Supported by intervention" | "Rejected";
  gateIds: string[];
  componentId: string | null;
  edgeId: string | null;
  anchorId: string;
  modelIds: string[];
  sourceRevision: string;
  createdAt: string;
}
