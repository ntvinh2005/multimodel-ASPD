import { describe, expect, it } from "vitest";
import { inductionGateDataset } from "./inductionGateDataset";

describe("measured induction shared-gate bundle", () => {
  it("preserves smoke and discovery denominators", () => {
    expect(inductionGateDataset.checkpointStep).toBe(61036);
    expect(inductionGateDataset.splits.smoke).toMatchObject({ status: "measured", groups: 32, examples: 128, pools: 2 });
    expect(inductionGateDataset.splits.discovery).toMatchObject({ status: "measured", groups: 256, examples: 1024, pools: 12, evaluationScope: "gates" });
  });

  it("keeps full paired conditions and exact final boundaries in bounded contexts", () => {
    for (const split of [inductionGateDataset.splits.smoke, inductionGateDataset.splits.discovery]) {
      if (split.status !== "measured") throw new Error("expected measured split");
      expect(split.contextGateIds).toHaveLength(64);
      for (const gateId of split.contextGateIds) {
        const contexts = split.contexts[gateId];
        expect(new Set(contexts.map((context) => context.condition))).toEqual(new Set(["induction", "counterfactual", "no_match", "repeated_no_association"]));
        for (const context of contexts) expect(context.positions.at(-1)?.boundary).toBe(context.prompt.length);
      }
    }
  });

  it("does not contain dependency edges or a universal-mechanism claim", () => {
    expect(inductionGateDataset.scientificBoundary).toMatch(/no induction circuit or universal mechanism/i);
    expect(JSON.stringify(inductionGateDataset)).not.toContain('"edges"');
  });
});
