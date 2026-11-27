import { expect, it } from "vitest";
import { matchesSemanticPair } from "./SemanticDiscovery";

it("filters the actual component pair regardless of endpoint order", () => {
  expect(matchesSemanticPair("smol", "gemma", "gemma/smol")).toBe(true);
  expect(matchesSemanticPair("gemma", "granite", "gemma/smol")).toBe(false);
  expect(matchesSemanticPair("gemma", "gemma", "gemma/smol")).toBe(false);
  expect(matchesSemanticPair("smol", "granite", "cross-model")).toBe(true);
  expect(matchesSemanticPair("granite", "granite", "cross-model")).toBe(false);
  expect(matchesSemanticPair("granite", "granite", "granite/granite")).toBe(true);
  expect(matchesSemanticPair("granite", "granite", "all")).toBe(true);
});
