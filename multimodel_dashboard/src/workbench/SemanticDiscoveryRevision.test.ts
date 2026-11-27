import { describe, expect, it } from "vitest";
import { hasSemanticCoverage, type RevisedCatalog } from "./SemanticDiscoveryRevision";

function catalog(a: number, b: number): RevisedCatalog {
  return {
    schemaVersion: 2, runId: "a".repeat(64), dataset: "natural-validation", checkpoint: "test",
    graphCoverage: "unavailable",
    pairs: [{
      left: "a", right: "b", results: { discovery: {}, heldout: {} },
      fullSupportJaccard: { discovery: 0, heldout: 0 },
      heldoutAdvantageInterval: [0.1, 0.2],
      controls: { left: { discovery: [], heldout: [] }, right: { discovery: [], heldout: [] } },
    }],
    groups: [{ components: ["a", "b"] }],
    components: {
      a: { model: "gemma", gate: "pre/7", coverage: { discovery: 20, heldout: a }, beta: 0.1 },
      b: { model: "smol", gate: "pre/8", coverage: { discovery: 20, heldout: b }, beta: 0.1 },
    },
    examples: {
      a: { file: "component-0.json.gz", sha256: "a".repeat(64) },
      b: { file: "component-1.json.gz", sha256: "b".repeat(64) },
    },
  };
}

describe("revised Semantic Discovery release", () => {
  it("requires both endpoints and a mean of twenty documents", () => {
    expect(hasSemanticCoverage(catalog(10, 30))).toBe(true);
    expect(hasSemanticCoverage(catalog(9, 31))).toBe(false);
    expect(hasSemanticCoverage(catalog(10, 29))).toBe(false);
  });
  it("rejects a reference outside its immutable component asset namespace", () => {
    const result = catalog(20, 20);
    result.examples.b.file = "../semantic-discovery/catalog.json";
    expect(hasSemanticCoverage(result)).toBe(false);
  });
});
