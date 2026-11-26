import { describe, expect, it } from "vitest";
import { matchingNaturalGraph, naturalMetric, naturalScale, naturalScalar,
  type NaturalTokenAsset, type NaturalTokenGraphAsset, type NaturalScalarAsset, type NaturalTokenPoint } from "./naturalTokenEvidence";

const active: NaturalTokenPoint = { anchor: "a", pool: 4, characterBoundary: 15, positions: { smol: 7 },
  gates: { "MID:1117": 1.75 }, graphCoverage: "unavailable" };
const inactive: NaturalTokenPoint = { ...active, anchor: "inactive", gates: {} };
const asset: NaturalTokenAsset = { schemaVersion: 1, checkpoint: "ck", candidate: "NVX-025",
  dataset: "archive", split: "test",
  provenance: { archiveManifestSha256: "mf", candidateEvidenceSha256: "e", poolSize: 2048 }, examples: {} };
const component = "smol.L14.mlp_up/1117";
const measured = { reading: -2, coefficient: 0, magnitude: 0, normalized: 0 };
const replay: NaturalScalarAsset = { schemaVersion: 1, checkpoint: "ck", dataset: "archive", split: "test",
  archiveManifestSha256: "mf", rows: { inactive: { pool: 4, position: 7, components: { [component]: measured } } } };

describe("Natural Validation exact token evidence", () => {
  it("distinguishes nonalignment, unmeasured active components and known inactive zeros", () => {
    expect(naturalMetric("gate", undefined, "MID:1117")).toBeUndefined();
    expect(naturalMetric("gate", active, "MID:1117")).toBe(1.75);
    expect(naturalMetric("gate", inactive, "MID:1117")).toBe(0);
    expect(naturalMetric("coefficient", active, "MID:1117")).toBeUndefined();
    expect(naturalMetric("reading", inactive, "MID:1117")).toBeUndefined();
    expect(naturalMetric("magnitude", inactive, "MID:1117")).toBe(0);
  });

  it("accepts an ungated negative reading on exact inactive support only", () => {
    expect(naturalMetric("reading", inactive, "MID:1117", naturalScalar(asset, replay, inactive, component))).toBe(-2);
    expect(naturalScalar(asset, replay, { ...inactive, pool: 5 }, component)).toBeUndefined();
    expect(naturalScalar(asset, replay, { ...inactive, positions: { smol: 6 } }, component)).toBeUndefined();
    expect(naturalScalar(asset, replay, { ...active, anchor: inactive.anchor }, component)).toBeUndefined();
    expect(naturalScalar(asset, { ...replay, archiveManifestSha256: "foreign" }, inactive, component)).toBeUndefined();
    expect(naturalScale([undefined, -2, 1.75, 0])).toBe(2);
  });

  it("rejects nearby graphs and retains measured empty graphs", () => {
    const scope = { anchor: "a", pool: 4, dataset: "archive", archiveManifestSha256: "mf" };
    const graph: NaturalTokenGraphAsset = { ...scope, schemaVersion: 1, checkpoint: "ck", split: "test",
      coverage: "full", snapshot: { anchor: "a", kind: "measured", representatives: [], edges: [] } };
    expect(matchingNaturalGraph("ck", scope, graph)).toBe(graph);
    expect(matchingNaturalGraph("ck", scope, { ...graph, snapshot: { ...graph.snapshot, anchor: "validation" } })).toBeUndefined();
    expect(matchingNaturalGraph("ck", scope, { ...graph, pool: 5 })).toBeUndefined();
    expect(matchingNaturalGraph("ck", scope, undefined)).toBeUndefined();
  });
});
