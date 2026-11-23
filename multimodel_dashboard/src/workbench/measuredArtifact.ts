export interface MeasuredPairSummary {
  modelPair: "gemma_granite" | "gemma_smol" | "smol_granite";
  jaccard: number;
  totalVariation: number;
  validTvRows: number;
}

export interface MeasuredValidationArtifact {
  schemaVersion: 1;
  artifactType: "aspd_v3_1_posthoc_parent_set_validation_record";
  scientificStatus: "VALIDATION_RUN_PRE_MERGE";
  experimentId: string;
  sourceCommit: string;
  repositoryRevision: string;
  rawArtifactBytes: number;
  rawArtifactsLocal: false;
  checkpoint: {
    step: number;
    manifestSha256: string;
    trainingConfigSha256: string;
  };
  evaluation: {
    split: "validation";
    sourceDtype: "bfloat16";
    importanceAnchors: number;
    screenedGates: number;
    comparedGates: number;
    edgeRecords: number;
    pairComparisons: number;
  };
  pairs: MeasuredPairSummary[];
  excludedZeroEdgeFiringGates: string[];
  summaryStatus: "DESCRIPTIVE_ANCHOR_MEANS_NOT_A_VALIDATED_STATISTIC";
}

// Typed transcription of the hash-verified record committed on origin/main.
// It contains aggregate measurements only; component and edge tensors remain off-repository.
export const measuredValidationArtifact: MeasuredValidationArtifact = {
  schemaVersion: 1,
  artifactType: "aspd_v3_1_posthoc_parent_set_validation_record",
  scientificStatus: "VALIDATION_RUN_PRE_MERGE",
  experimentId: "posthoc_v31_500m_validation_v1",
  sourceCommit: "9c9f492c02dafc3a79ab51fd310637b74a1caa05",
  repositoryRevision: "aa1eb4c1b1bbddfb725229c64ef4442e75ebdc77",
  rawArtifactBytes: 29394517308,
  rawArtifactsLocal: false,
  checkpoint: {
    step: 61036,
    manifestSha256: "735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969",
    trainingConfigSha256: "f7a6d87436304956ca0baca955ec2e0e929a7207142f57fc35362369822f63be",
  },
  evaluation: {
    split: "validation",
    sourceDtype: "bfloat16",
    importanceAnchors: 102400,
    screenedGates: 15675,
    comparedGates: 15673,
    edgeRecords: 2756154,
    pairComparisons: 936837,
  },
  pairs: [
    { modelPair: "gemma_granite", jaccard: 0.35601906763455843, totalVariation: 0.48011483936815286, validTvRows: 299769 },
    { modelPair: "gemma_smol", jaccard: 0.341954632813974, totalVariation: 0.493454463120278, validTvRows: 294369 },
    { modelPair: "smol_granite", jaccard: 0.3255328320394026, totalVariation: 0.5135182676074135, validTvRows: 293559 },
  ],
  excludedZeroEdgeFiringGates: ["MID:7969", "PRE:2364"],
  summaryStatus: "DESCRIPTIVE_ANCHOR_MEANS_NOT_A_VALIDATED_STATISTIC",
};

export function formatBytes(bytes: number): string {
  return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
}

export function modelPairLabel(pair: MeasuredPairSummary["modelPair"]): string {
  return ({ gemma_granite: "Gemma ↔ Granite", gemma_smol: "Gemma ↔ SmolLM2", smol_granite: "SmolLM2 ↔ Granite" })[pair];
}
