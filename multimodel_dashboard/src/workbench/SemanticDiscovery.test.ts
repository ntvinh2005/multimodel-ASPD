import { expect, it } from "vitest";
import { matchesSemanticPair, sharedGraphAnchors } from "./SemanticDiscovery";

it("filters the actual component pair regardless of endpoint order", () => {
  expect(matchesSemanticPair("smol", "gemma", "gemma/smol")).toBe(true);
  expect(matchesSemanticPair("gemma", "granite", "gemma/smol")).toBe(false);
  expect(matchesSemanticPair("gemma", "gemma", "gemma/smol")).toBe(false);
  expect(matchesSemanticPair("smol", "granite", "cross-model")).toBe(true);
  expect(matchesSemanticPair("granite", "granite", "cross-model")).toBe(false);
  expect(matchesSemanticPair("granite", "granite", "granite/granite")).toBe(true);
  expect(matchesSemanticPair("granite", "granite", "all")).toBe(true);
});

it("offers a shared token only if every displayed model has a graph there", () => {
  const positions = (ids: string[]) => ids.map(anchor => ({ anchor }));
  expect(sharedGraphAnchors([
    positions(["different", "shared", "pair-only"]),
    positions(["shared", "pair-only"]),
    positions(["shared", "different"]),
  ])).toEqual(["shared"]);
  expect(sharedGraphAnchors([positions(["a"]), positions(["a"]), []])).toEqual([]);
});
