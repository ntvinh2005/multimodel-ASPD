import { describe, expect, it } from "vitest";
import { measuredTaylorAt, partitionAt, releasePath, type Partition, type Taylor } from "./supernodeDiscovery";

describe("Semantic Supernode Discovery", () => {
  it("cuts the original tree by count and distance without recomputing causal evidence", () => {
    const partition = { componentOrder: ["a", "b", "c"], tree: [[0, 1, .25, 2], [2, 3, .75, 3]] } as Partition;
    expect(partitionAt(partition, { count: 2 })).toEqual([[0, 1], [2]]);
    expect(partitionAt(partition, { threshold: .2 })).toEqual([[0], [1], [2]]);
    expect(partitionAt(partition, { threshold: .9 })).toEqual([[2, 0, 1]]);
  });

  it("filters by exact extraction scope and uses partial Taylor magnitude for display only", () => {
    const a = { model: "gemma", anchor: "a", score: null, observedTargetMean: .1, status: "truncated" } as Taylor;
    const b = { ...a, anchor: "b" };
    const c = { ...a, model: "smol" } as Taylor;
    expect(measuredTaylorAt([a, b, c], "gemma", "a", .05)).toEqual([a]);
    expect(measuredTaylorAt([a, b, c], "gemma", "a", .2)).toEqual([]);
    expect(a.score).toBeNull();
  });

  it("allows only versioned and hashed asset paths", () => {
    const run = "f".repeat(64);
    expect(releasePath(run, { file: "component-2.json.gz", sha256: run })).toContain(run);
    expect(releasePath(run, { file: "../catalog.json", sha256: run })).toBeUndefined();
  });
});
