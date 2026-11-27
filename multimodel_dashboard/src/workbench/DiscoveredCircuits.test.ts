import { expect, it } from "vitest";
import { conditionLabel, discoveryWording, selectionFrameFamily } from "./discoveryWording";
import { matchesDiscovery, type DiscoveryCandidate } from "./DiscoveredCircuits";

const candidate: DiscoveryCandidate = {
  id: "CXD-check", dataset: "selection-frame", directions: ["bottom-up", "composition"],
  anchorCount: 8, status: "Unvalidated hypothesis", model: "gemma", paretoFront: 1,
  structure: { nodes: 6, edges: 8, depth: 3, convergence: 2, branching: 1, diamonds: 0 },
  gateIds: ["MID:7417"], componentIds: ["gemma.L15.mlp_up/7417"],
  crossModel: { models: ["gemma", "granite"], sharedExactAnchors: 1, peers: [] },
  matchedNull: { trials: 0, hits: null },
  interpretation: {
    reviewTitle: "A reviewed graph", hypothesis: "Local relay",
    alternative: "", counterevidence: "", proposedTest: "", independentActivationExamples: [],
  },
  witnesses: [{
    anchor: "position", model: "gemma", dataset: "selection-frame", split: "heldout",
    family: "nonselection_voice_by_polarity", text: "The book was carried.", boundary: 21,
    example: null, pool: null, artifact: "test", edges: [], components: {}, activations: {},
  }],
  recurrence: [{ dataset: "selection-frame", split: "heldout", anchors: 10, groups: 2 }],
  witnessSignStability: 0.75,
};

it("filters by hypotheses, displayed contexts, exact gates, method, recurrence and model evidence", () => {
  const filters = { query: "relay", direction: "composition", nodes: 6, recurrence: 8, models: 2 };
  expect(matchesDiscovery(candidate, filters)).toBe(true);
  expect(matchesDiscovery(candidate, { ...filters, query: "carrying a book" })).toBe(true);
  expect(matchesDiscovery(candidate, { ...filters, query: "mid:7417" })).toBe(true);
  expect(matchesDiscovery(candidate, { ...filters, nodes: 7 })).toBe(false);
  expect(matchesDiscovery(candidate, { ...filters, direction: "top-down" })).toBe(false);
  expect(matchesDiscovery(candidate, { ...filters, recurrence: 9 })).toBe(false);
  expect(matchesDiscovery(candidate, { ...filters, models: 3 })).toBe(false);
});

it("explains the sentence controls while suppressing generic semantic claims", () => {
  const copy = discoveryWording({
    ...candidate,
    interpretation: {
      ...candidate.interpretation, reviewTitle: undefined,
      hypothesis: "The convergent or serial dependencies may coordinate processing.",
    },
  });
  expect(copy.hypothesis).toBeUndefined();
  expect(copy.sources).toEqual(["Carrying a book in active and passive voice"]);
  expect(copy.heldout).toContain("10 held-out token positions");
  expect(copy.signs).toContain("75% of edges");
  expect(conditionLabel("selection-frame", "counterfactual_negative",
    selectionFrameFamily("selection-frame-discovery-nonselection_voice_by_polarity"))).toBe("Passive, negative");
  expect(conditionLabel("negation-discovery", "counterfactual_negative", "binary")).toBe("Counterfactual negative");
});
