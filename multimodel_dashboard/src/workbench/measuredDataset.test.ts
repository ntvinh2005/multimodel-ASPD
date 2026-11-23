import { describe, expect, it } from "vitest";
import rawDataset from "./data/measuredWorkbenchDataset.json";
import { measuredWorkbenchDataset, validateMeasuredWorkbenchDataset } from "./measuredDataset";

describe("measured Workbench bundle", () => {
  it("loads a bounded non-synthetic graph with immutable provenance", () => {
    expect(measuredWorkbenchDataset.manifest.synthetic).toBe(false);
    expect(measuredWorkbenchDataset.manifest.export?.checkpointStep).toBe(61036);
    expect(measuredWorkbenchDataset.models).toHaveLength(3);
    expect(measuredWorkbenchDataset.anchors).toHaveLength(4);
    expect(measuredWorkbenchDataset.components.length).toBeGreaterThan(0);
    expect(measuredWorkbenchDataset.edges.length).toBeGreaterThan(0);
    expect(measuredWorkbenchDataset.edges.every((edge) => edge.evidence === "measured-taylor")).toBe(true);
  });

  it("contains no cross-model dependency edges", () => {
    const components = new Map(measuredWorkbenchDataset.components.map((component) => [component.id, component]));
    for (const edge of measuredWorkbenchDataset.edges) {
      expect(components.get(edge.source)?.modelId).toBe(edge.modelId);
      expect(components.get(edge.target)?.modelId).toBe(edge.modelId);
    }
  });

  it("rejects a synthetic flag or cross-model edge", () => {
    const synthetic = structuredClone(rawDataset) as unknown as Record<string, any>;
    synthetic.manifest.synthetic = true;
    expect(() => validateMeasuredWorkbenchDataset(synthetic)).toThrow(/synthetic=false/);

    const crossed = structuredClone(rawDataset) as unknown as Record<string, any>;
    crossed.edges[0].modelId = crossed.edges[0].modelId === "gemma" ? "smol" : "gemma";
    expect(() => validateMeasuredWorkbenchDataset(crossed)).toThrow(/crosses models/);
  });
});
