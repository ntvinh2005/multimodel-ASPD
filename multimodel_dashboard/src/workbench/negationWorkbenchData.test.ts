import { afterEach, describe, expect, it, vi } from "vitest";
import {
  activationAt, expandedAtFinalQuery, graphAtFinalQuery, readNegationAsset, validateNegationIndex,
  type ExpandedNegationGroup, type NegationExample, type NegationIndex, type NegationPosition,
} from "./negationWorkbenchData";

const position: NegationPosition = {
  anchor_id: "exact-anchor", character_boundary: 25, roles_ending: ["query_prefix"], final_query: true,
  model_positions: { gemma: 6, smol: 5, granite: 4 },
  pre: { active_gate_ids: [4312], values: [0.2] },
  mid: { active_gate_ids: [4312], values: [0.6] },
};

describe("Negation workbench provenance and graph coverage", () => {
  afterEach(() => vi.unstubAllGlobals());
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

  it("fetches split-scoped assets and rejects tampered bytes", async () => {
    const text = '{"mechanism":"negation","groupId":"example"}';
    const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)))]
      .map((byte) => byte.toString(16).padStart(2, "0")).join("");
    const fetchMock = vi.fn(() => Promise.resolve(new Response(text, {
      headers: { "content-type": "application/json" },
    })));
    vi.stubGlobal("fetch", fetchMock);
    const { signal } = new AbortController();
    await expect(readNegationAsset("discovery/group-001.json", signal, hash))
      .resolves.toEqual({ mechanism: "negation", groupId: "example" });
    expect(fetchMock).toHaveBeenCalledWith("/negation/discovery/group-001.json", { signal });
    await expect(readNegationAsset("discovery/group-001.json", signal, "wrong"))
      .rejects.toThrow("hash mismatch");
  });

  it("resolves exact-component trace targets only at the matching model, anchor and condition", () => {
    const example = {
      example_id: "negative-1", group_id: "matched", condition: "negative",
      pool_id: "original-2048-pool", positions: [position],
    } as NegationExample;
    const representative = {
      component: { id: "gemma.L11.q/4312", modelId: "gemma", gateId: "PRE:4312", layer: 11, matrix: "Q" as const, beta: .2, rho: .3 },
      coefficient: .5, parents: [],
    };
    const trace = {
      ...representative,
      component: { ...representative.component, id: "gemma.L09.q/4312", layer: 9 },
    };
    const graph = {
      anchorId: position.anchor_id, activation: .2, comparisons: [],
      models: { gemma: { targets: [representative], strength: 0, distribution: null } },
    };
    const row = {
      exampleId: example.example_id, groupId: example.group_id, poolId: example.pool_id, condition: example.condition,
      anchorId: position.anchor_id, modelPositions: position.model_positions,
      graphs: { "PRE:4312": graph }, traceTargets: { [trace.component.id]: trace },
    };
    const group: ExpandedNegationGroup = {
      schemaVersion: 1, mechanism: "negation", split: "discovery", groupId: "matched",
      checkpointSha256: "c", discoveryManifestSha256: "d", examples: [row],
    };
    const measure = expandedAtFinalQuery(group, example, "PRE:4312", "gemma");
    expect(measure.targets.get(trace.component.id)).toEqual({ target: trace, representative: false });
    expect(measure.targets.get(representative.component.id)?.representative).toBe(true);
    expect(measure.graph?.models.gemma.targets).toEqual([representative]);
    expect(expandedAtFinalQuery(group, example, "PRE:4312", "granite").targets.size).toBe(0);
    expect(expandedAtFinalQuery(group, example, "PRE:2", "gemma").graph).toBeUndefined();
    expect(() => expandedAtFinalQuery(group, example, "MID:4312", "gemma")).toThrow("active gate");
    const wrongAnchor = { ...group, examples: [{ ...row, anchorId: "another" }] };
    expect(() => expandedAtFinalQuery(wrongAnchor, example, "PRE:4312", "gemma")).toThrow("anchor");
    const wrongPosition = { ...group, examples: [{ ...row, modelPositions: { ...position.model_positions, smol: 10 } }] };
    expect(() => expandedAtFinalQuery(wrongPosition, example, "PRE:4312", "gemma")).toThrow("anchor");
    const wrongPool = { ...group, examples: [{ ...row, poolId: "another" }] };
    expect(() => expandedAtFinalQuery(wrongPool, example, "PRE:4312", "gemma")).toThrow("anchor");
  });
});
