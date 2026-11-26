import { expect, it } from "vitest";
import { matchesDiscovery, type DiscoveryCandidate } from "./DiscoveredCircuits";

it("filters by hypotheses, exact gates, direction, recurrence and cross-model evidence", () => {
  const candidate = {
    id: "CXD-check", dataset: "test", directions: ["bottom-up", "composition"], anchorCount: 8,
    structure: { nodes: 6 }, gateIds: ["MID:7417"], componentIds: ["gemma.L15.mlp_up/7417"],
    crossModel: { models: ["gemma", "granite"] }, interpretation: { hypothesis: "Local relay" },
  } as DiscoveryCandidate;
  const filters = { query: "relay", direction: "composition", nodes: 6, recurrence: 8, models: 2 };
  expect(matchesDiscovery(candidate, filters)).toBe(true);
  expect(matchesDiscovery(candidate, {...filters, query: "mid:7417"})).toBe(true);
  expect(matchesDiscovery(candidate, {...filters, nodes: 7})).toBe(false);
  expect(matchesDiscovery(candidate, {...filters, direction: "top-down"})).toBe(false);
  expect(matchesDiscovery(candidate, {...filters, recurrence: 9})).toBe(false);
  expect(matchesDiscovery(candidate, {...filters, models: 3})).toBe(false);
});
