import { describe, expect, it } from "vitest";
import type { Witness } from "./SemanticDiscovery";
import { anchorsForPattern, recurringParents, sharedParentGates } from "./semanticPatterns";

const granite = "granite.L23.mlp_down/515";
const smol = "smol.L09.mlp_up/515";
const graniteParent = "granite.L23.mlp_gate/515";
const graniteSecond = "granite.L22.mlp_down/515";
const smolParent = "smol.L08.mlp_up/515";
const witness = (anchor: string, component: string, sources: string[]): Witness => ({
  anchor, component, model: component === granite ? "granite" : "smol", artifact: "test",
  edges: sources.map(source => ({ source, target: component, E: -0.1, w: 0.02 })),
});
const table: Record<string, Witness> = {
  [granite + "|a"]: witness("a", granite, [graniteParent, graniteSecond, graniteParent]),
  [smol + "|a"]: witness("a", smol, [smolParent]),
  [granite + "|b"]: witness("b", granite, [graniteParent]),
  [smol + "|b"]: witness("b", smol, []),
  [granite + "|c"]: witness("c", granite, [graniteParent]),
};
const at = (component: string, anchor: string) => table[component + "|" + anchor];
const ids = ["a", "b", "c"];

describe("recurrence in measured Semantic Discovery graphs", () => {
  it("uses each endpoint's measured denominator and counts exact parents once per anchor", () => {
    expect(recurringParents(ids, granite, at)).toEqual({
      measured: 3, rows: [{ id: graniteParent, count: 3 }, { id: graniteSecond, count: 1 }],
    });
    expect(recurringParents(ids, smol, at)).toEqual({ measured: 2, rows: [{ id: smolParent, count: 1 }] });
  });

  it("counts a shared parent gate once and uses only paired exact anchors", () => {
    expect(sharedParentGates(ids, [granite, smol], at)).toEqual({
      measured: 2, rows: [{ id: "MID:515", count: 1 }],
    });
  });

  it("requires all selected edges together and excludes missing target measurements from contrasts", () => {
    expect(anchorsForPattern(ids, [
      { target: granite, source: graniteParent },
      { target: smol, source: smolParent },
    ], at)).toEqual({ present: ["a"], contrast: ["b"], unavailable: 1 });
    expect(anchorsForPattern(ids, [
      { target: granite, source: graniteParent },
      { target: granite, source: graniteSecond },
    ], at)).toEqual({ present: ["a"], contrast: ["b", "c"], unavailable: 0 });
  });
});
