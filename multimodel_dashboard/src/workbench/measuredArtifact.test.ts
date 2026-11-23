import { describe, expect, it } from "vitest";
import { formatBytes, measuredValidationArtifact, modelPairLabel } from "./measuredArtifact";

describe("measured validation artifact adapter", () => {
  it("preserves the immutable provenance and aggregate counts", () => {
    expect(measuredValidationArtifact.sourceCommit).toHaveLength(40);
    expect(measuredValidationArtifact.checkpoint.step).toBe(61036);
    expect(measuredValidationArtifact.evaluation.edgeRecords).toBe(2756154);
    expect(measuredValidationArtifact.rawArtifactsLocal).toBe(false);
  });

  it("keeps all reported pair summaries bounded and explicitly descriptive", () => {
    expect(measuredValidationArtifact.pairs).toHaveLength(3);
    for (const pair of measuredValidationArtifact.pairs) {
      expect(pair.jaccard).toBeGreaterThanOrEqual(0);
      expect(pair.jaccard).toBeLessThanOrEqual(1);
      expect(pair.totalVariation).toBeGreaterThanOrEqual(0);
      expect(pair.totalVariation).toBeLessThanOrEqual(1);
      expect(pair.validTvRows).toBeGreaterThan(0);
    }
    expect(measuredValidationArtifact.summaryStatus).toBe("DESCRIPTIVE_ANCHOR_MEANS_NOT_A_VALIDATED_STATISTIC");
  });

  it("formats artifact metadata for the presentation layer", () => {
    expect(formatBytes(29394517308)).toBe("29.39 GB");
    expect(modelPairLabel("gemma_smol")).toBe("Gemma ↔ SmolLM2");
  });
});
