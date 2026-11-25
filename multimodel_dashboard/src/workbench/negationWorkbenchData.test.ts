import { describe, expect, it } from "vitest";
import { activationAt, graphAtFinalQuery, validateNegationIndex, type NegationIndex, type NegationPosition } from "./negationWorkbenchData";

const position: NegationPosition = {
  anchor_id: "exact-anchor", character_boundary: 25, roles_ending: ["query_prefix"], final_query: true,
  model_positions: { gemma: 6, smol: 5, granite: 4 },
  pre: { active_gate_ids: [4312], values: [0.2] },
  mid: { active_gate_ids: [4312], values: [0.6] },
};

describe("Negation workbench provenance and graph coverage", () => {
  it("distinguishes PRE, MID, zero support and a missing graph", () => {
    expect(activationAt(position, "PRE:4312")).toBe(.2);
    expect(activationAt(position, "MID:4312")).toBe(.6);
    expect(activationAt(position, "PRE:5")).toBe(0);
    const example = {
      positions: [position],
      graphs: { "PRE:5": { anchorId: position.anchor_id, activation: 0, models: {}, comparisons: [] } },
    };
    expect(graphAtFinalQuery(example, "PRE:5")?.activation).toBe(0);
    expect(graphAtFinalQuery(example, "PRE:4312")).toBeUndefined();
  });

  it("rejects a graph from a different query or dataset", () => {
    const example = {
      positions: [position],
      graphs: { "PRE:4312": { anchorId: "other-example", activation: .2, models: {}, comparisons: [] } },
    };
    expect(() => graphAtFinalQuery(example, "PRE:4312")).toThrow("final query");
    const index = {
      schemaVersion: 1, mechanism: "negation", checkpointStep: 61036,
      splits: {
        smoke: { gates: [{}] },
        discovery: { gates: [{}], manifestSha256: "negation", graphProvenance: { discoveryManifestSha256: "induction" } },
      },
    } as unknown as NegationIndex;
    expect(() => validateNegationIndex(index)).toThrow("disagree");
  });
});
