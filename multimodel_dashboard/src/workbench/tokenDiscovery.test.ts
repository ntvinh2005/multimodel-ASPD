import { afterEach, describe, expect, it, vi } from "vitest";
import { readWorkbenchAsset } from "./negationWorkbenchData";
import { selectToken, tokenGates, tokenGraph, type TokenExample, type TokenGraphAsset } from "./tokenDiscovery";

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
