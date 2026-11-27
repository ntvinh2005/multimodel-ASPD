import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { expandedCatalog, renderableExpandedGraph } from "./expandedNaturalValidation";
import { measuredWorkbenchDataset } from "./measuredDataset";
import { decodeDictionaryGate, dictionaryGates, dictionaryLocation, searchDictionaryGates, validateDictionary, type DictionaryBlock, type DictionaryIndex } from "./naturalDictionary";
import NaturalValidationScenarios from "./NaturalValidationScenarios";

const testIndex: DictionaryIndex = {
  schemaVersion: 1,
  checkpoint: expandedCatalog.checkpoint,
  checkpointStep: 61036,
  featureCount: 8192,
  blockSize: 8192,
  flags: "i".repeat(1855) + "n" + "i".repeat(6336) + "o" + "n" + "i".repeat(8190),
  blocks: [{ path: "pre.json.gz", sha256: "pre" }, { path: "mid.json.gz", sha256: "mid" }],
  coverage: { incoming: 16381, outgoingOnly: 1, withoutEdges: 2, withExamples: 4 },
  outgoingLimitPerModel: 8,
  provenance: { source: "example", profileHashes: {} },
};

const testBlock: DictionaryBlock = {
  checkpoint: testIndex.checkpoint, start: 8192,
  components: [["gemma.L09.o/0", 1], ["gemma.L09.mlp_down/2", 0]],
  records: [["validation-anchor", 123, 1, [[0, 1, -0.5, 0.2, 2, 543]], []], [null, 0, 0, [], []]],
};

describe("Natural Validation gate dictionary", () => {
  it("searches the entire dictionary by site-qualified or bare gate ID", () => {
    const gates = dictionaryGates(validateDictionary(testIndex));
    expect(gates).toHaveLength(16384);
    const ids = (query: string) => searchDictionaryGates(gates, query).matches.map(({ id }) => id);
    expect(ids("mid/0")).toEqual(["MID:0"]);
    expect(ids("1")).toEqual(["PRE:1", "MID:1"]);
    expect(ids("MID 8192")).toEqual([]);
    expect(ids("PRE")).toHaveLength(8192);
  });

  it("preserves a stored anchor, model, measurements, provenance and missing-edge status", () => {
    expect(dictionaryLocation(testIndex, "MID:0")).toMatchObject({ path: "mid.json.gz", offset: 8192, row: 0, coverage: "o" });
    const gate = decodeDictionaryGate(testIndex, "MID:0", testBlock);
    expect(gate).toMatchObject({ fireCount: 123, coverage: "o", snapshot: { edges: [{ profileRow: 543 }] } });
    const { components, edges } = renderableExpandedGraph(gate.snapshot!);
    expect(components[0]).toMatchObject({ id: "gemma.L09.o/0", gateId: "MID:0", representative: true });
    expect(edges).toMatchObject([{ modelId: "gemma", anchorId: "validation-anchor", signedEffect: -0.5, normalizedMagnitude: 0.2 }]);
    expect(decodeDictionaryGate(testIndex, "MID:1", testBlock)).toMatchObject({ snapshot: null, coverage: "n" });
    expect(() => decodeDictionaryGate(testIndex, "PRE:0", testBlock)).toThrow(/does not match/);
  });

  it("renders gate hits outside the starter bundle alongside the candidate selector", () => {
    const id = "PRE:1855";
    expect(measuredWorkbenchDataset.gates.some(gate => gate.id === id)).toBe(false);
    const result = renderToStaticMarkup(<NaturalValidationScenarios manifest={measuredWorkbenchDataset.manifest}
      activeAnchorId={null} query={id} onSelect={() => undefined} onQueryChange={() => undefined}
      onOpenCandidate={() => undefined} dictionary={testIndex} onOpenDictionaryGate={() => undefined} />);
    expect(result).toContain("Shared gate dictionary results");
    expect(result).toContain("No retained dependency edges");
    expect(result).toContain("Random measured gate");
    expect(result).toContain("Candidate graphs (123)");
  });
});
