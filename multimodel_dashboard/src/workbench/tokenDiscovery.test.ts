import { afterEach, describe, expect, it, vi } from "vitest";
import { readWorkbenchAsset } from "./negationWorkbenchData";
import { filterTokenGates, graphWitnessForGate, selectExample, selectGroup, selectToken, tokenGates, tokenGraph, type GraphWitnessDataset, type TokenExample, type TokenGraphAsset } from "./tokenDiscovery";

const anchor: TokenExample["positions"][number] = {
  anchor_id: "exact", character_boundary: 3, final_query: false, roles_ending: [],
  model_positions: { gemma: 2, smol: 3, granite: 1 }, pool_id: "frozen-pool",
  pre: { active_gate_ids: [42], values: [.3] }, mid: { active_gate_ids: [42], values: [.4] },
};
const final = { ...anchor, character_boundary: 9, anchor_id: "final", final_query: true };
const example: TokenExample = {
  example_id: "example", group_id: "group", condition: "affirmative", prompt: "The story",
  positions: [anchor, final],
};
const asset: TokenGraphAsset = {
  anchorId: "exact", exampleId: "example", poolId: "frozen-pool", checkpointSha256: "sha",
  official: {
    "MID:42": {
      anchorId: "exact", activation: .4, comparisons: [],
      models: { smol: { targets: [], strength: null, distribution: null, status: "no_reference_representatives" } },
    },
  },
  componentTargets: {
    "smol.L08.q/42": [1, 2, []],
    "smol.L09.mlp_up/42": [1.3, 3, [["smol.L08.q/42", -.2]]],
  },
  outputScale: { "smol.L08.q": 2, "smol.L09.mlp_up": 4 },
  completeComponents: true,
};

describe("token discovery", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("loads and decompresses a hash-verified graph asset", async () => {
    const text = JSON.stringify(asset);
    const gzip = await new Response(new Blob([text]).stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer();
    const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", gzip))]
      .map((byte) => byte.toString(16).padStart(2, "0")).join("");
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(gzip, { headers: { "content-type": "application/gzip" } }))));
    const signal = new AbortController().signal;
    await expect(readWorkbenchAsset("tokens/graph.json.gz", signal, digest)).resolves.toEqual(asset);
    await expect(readWorkbenchAsset("tokens/graph.json.gz", signal, "invalid")).rejects.toThrow("hash mismatch");
  });
  it("defaults to the final anchor and exposes both naturally active sites", () => {
    expect(selectToken(example, "")).toBe(final);
    expect(selectToken(example, "exact")).toBe(anchor);
    expect(tokenGates(anchor)).toEqual([{ id: "MID:42", value: .4 }, { id: "PRE:42", value: .3 }]);
  });

  it("keeps an explicitly selected group and returns to the final anchor of another condition", () => {
    expect(selectGroup(["preferred", "selected"], "selected", ["preferred"])).toBe("selected");
    expect(selectGroup(["preferred", "selected"], "previous-split", ["missing", "preferred"])).toBe("preferred");
    const neg = { ...example, condition: "negative", positions: [{ ...final, anchor_id: "negative-final" }] };
    expect(selectExample([example, neg], "negative", "affirmative")).toBe(neg);
    expect(selectToken(neg, anchor.anchor_id)?.anchor_id).toBe("negative-final");
  });

  it("preserves a gate that is active at the selected token and updates the shared sidebar filters", () => {
    const active = tokenGates(anchor);
    expect(filterTokenGates(active, "ALL", "", "PRE:42").selected).toBe("PRE:42");
    expect(filterTokenGates(active, "ALL", "", "PRE:91").selected).toBe("MID:42");
    expect(filterTokenGates(active, "PRE", "42", "MID:42").rows).toEqual([{ id: "PRE:42", value: .3 }]);
    expect(filterTokenGates([], "ALL", "", "MID:42")).toEqual({ rows: [], selected: undefined });
  });

  it("navigates to a stored witness for this gate and model", () => {
    const inventory: GraphWitnessDataset = { anchors: [
      { group: "one", condition: "negative", anchor: "earlier-token", final: false },
      { group: "one", condition: "negative", anchor: "last-token", final: true },
      { group: "two", condition: "affirmative", anchor: "other-group", final: true },
    ], gates: { "PRE:42": [[0, 2], [1, 3], [2, 4]] } };
    expect(graphWitnessForGate(inventory, "PRE:42", "smol", "one", "negative", "earlier-token")?.anchor).toBe("earlier-token");
    expect(graphWitnessForGate(inventory, "PRE:42", "gemma", "one", "negative", "earlier-token")?.anchor).toBe("last-token");
    expect(graphWitnessForGate(inventory, "PRE:42", "granite", "one", "negative")?.anchor).toBe("other-group");
    expect(graphWitnessForGate(inventory, "MID:42", "smol")).toBeUndefined();
  });

  it("requires exact anchor, example and pool identities", () => {
    expect(() => tokenGraph(asset, example, final, "sha", "MID:42", "smol")).toThrow("selected token");
    expect(() => tokenGraph({ ...asset, poolId: "other" }, example, anchor, "sha", "MID:42", "smol")).toThrow("selected token");
    expect(tokenGraph({ ...asset, completeComponents: false }, example, anchor, "sha", "PRE:42", "smol")).toEqual({});
  });

  it("exposes model-specific paths and empty parent sets without inventing representatives", () => {
    const result = tokenGraph(asset, example, anchor, "sha", "MID:42", "smol");
    expect(result.graph?.models.smol.distribution).toBeNull();
    expect(result.targets?.get("smol.L08.q/42")?.target.parents).toEqual([]);
    const target = result.targets?.get("smol.L09.mlp_up/42");
    expect(target?.representative).toBe(false);
    expect(target?.target.parents[0].E).toBe(-.2);
    expect(target?.target.parents[0].w).toBeCloseTo(.15);
    expect(target?.target.parents[0].component.gateId).toBe("PRE:42");
  });
});
