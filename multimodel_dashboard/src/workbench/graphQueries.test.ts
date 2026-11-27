import { describe, expect, it } from "vitest";
import { edgesForAnchor, jaccard, normalizedDistribution, searchSharedGates, traceGraph, tvDistance } from "./graphQueries";
import type { DependencyEdge } from "./types";

const edge = (id: string, source: string, target: string, anchorId = "a", magnitude = .5): DependencyEdge => ({
  id, source, target, anchorId, modelId: "model", signedEffect: magnitude,
  normalizedMagnitude: magnitude, upstreamCoefficient: .2, gradient: .3,
  normalization: "test", evidence: "taylor-only",
});

describe("shared gate search", () => {
  const gates = ["MID:1660", "PRE:6608", "MID:6608", "MID:660", "PRE:66"].map((id) => ({ id }));
  const ids = (query: string) => searchSharedGates(gates, query).map(({ id }) => id);

  it("accepts full gate IDs regardless of case and separator", () => {
    expect(ids(" mid: 6608 ")).toEqual(["MID:6608"]);
    expect(ids("MID 660")).toEqual(["MID:660", "MID:6608"]);
    expect(ids("pre/66")).toEqual(["PRE:66", "PRE:6608"]);
  });

  it("prioritizes exact feature numbers across sites and preserves unfiltered order", () => {
    expect(ids("6608")).toEqual(["PRE:6608", "MID:6608"]);
    expect(ids("660")).toEqual(["MID:660", "PRE:6608", "MID:6608", "MID:1660"]);
    expect(ids("999")).toEqual([]);
    expect(searchSharedGates(gates, " ")).toBe(gates);
  });
});

describe("graph selection and traversal", () => {
  const edges = [edge("ab", "a", "b"), edge("bc", "b", "c"), edge("db", "d", "b"), edge("other", "x", "y", "other")];

  it("keeps anchor and threshold semantics explicit", () => {
    expect(edgesForAnchor(edges, "a", new Set(["a", "b", "c", "d"]), .4).map((item) => item.id)).toEqual(["ab", "bc", "db"]);
    expect(edgesForAnchor(edges, "a", new Set(["a", "b"]), .4).map((item) => item.id)).toEqual(["ab"]);
  });

  it("distinguishes direct parents from two-hop ancestors", () => {
    const oneHop = traceGraph(edges, "c", "parents", 1);
    expect([...oneHop.nodes]).toEqual(["c", "b"]);
    const twoHop = traceGraph(edges, "c", "parents", 2);
    expect(twoHop.nodes).toEqual(new Set(["c", "b", "a", "d"]));
    expect(twoHop.distance.get("a")).toBe(2);
  });

  it("traces children without reversing edge direction", () => {
    expect(traceGraph(edges, "a", "children", 2).nodes).toEqual(new Set(["a", "b", "c"]));
  });
});

describe("scientific missing-value handling", () => {
  it("does not turn zero total strength into a zero distribution", () => {
    expect(normalizedDistribution({ parent: 0 })).toEqual({ status: "undefined", values: null, total: 0 });
  });

  it("keeps undefined comparisons distinct from valid zero", () => {
    expect(jaccard(null, { a: 1 })).toBeNull();
    expect(tvDistance(null, { a: 1 })).toBeNull();
    expect(tvDistance({ a: 1 }, { a: 1 })).toBe(0);
    expect(jaccard({ a: 1 }, { a: 1 })).toBe(1);
  });
});
