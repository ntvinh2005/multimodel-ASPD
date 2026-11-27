import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import rawDataset from "./data/measuredWorkbenchDataset.json";
import { measuredWorkbenchDataset, validateMeasuredWorkbenchDataset } from "./measuredDataset";
import { componentsForSelection, discoveryScenarioForGate, edgesForAnchor, measuredViewForGate, searchSharedGates } from "./graphQueries";
import NaturalValidationScenarios from "./NaturalValidationScenarios";

describe("measured Workbench bundle", () => {
  it("loads a bounded non-synthetic graph with immutable provenance", () => {
    expect(measuredWorkbenchDataset.manifest.synthetic).toBe(false);
    expect(measuredWorkbenchDataset.manifest.export?.checkpointStep).toBe(61036);
    expect(measuredWorkbenchDataset.manifest.sourceRevision).toBe("1348bf9587122c396e2b741cc8ba7e7394a13253");
    expect(measuredWorkbenchDataset.manifest.export?.parentBackend?.slurmJobId).toBe("52480152");
    expect(measuredWorkbenchDataset.manifest.export?.parentBackend?.parentRows).toBe(24819978);
    expect(measuredWorkbenchDataset.models).toHaveLength(3);
    expect(measuredWorkbenchDataset.anchors).toHaveLength(14);
    expect(measuredWorkbenchDataset.components.length).toBeGreaterThan(0);
    expect(measuredWorkbenchDataset.edges.length).toBeGreaterThan(0);
    expect(measuredWorkbenchDataset.edges.every((edge) => edge.evidence === "measured-taylor")).toBe(true);
  });

  it("opens a measured discovery graph when searching for MID:7027", () => {
    const data = measuredWorkbenchDataset;
    const [result] = searchSharedGates(data.gates, "MID:7027");
    expect(result?.id).toBe("MID:7027");
    expect(searchSharedGates(data.gates, "7027")[0]?.id).toBe(result.id);
    const scenario = discoveryScenarioForGate(data.manifest, result.id);
    expect(scenario?.discoveryGateIds).toEqual(["MID:7027", "PRE:483", "PRE:3182"]);
    const selected = componentsForSelection(data.components, new Set(scenario!.discoveryGateIds), new Set(data.models.map(({ id }) => id)), false);
    const gatesByComponent = new Map(selected.map(({ id, gateId }) => [id, gateId]));
    const edges = edgesForAnchor(data.edges, scenario!.anchorId, new Set(gatesByComponent.keys()), 0);
    for (const { id } of data.models) {
      const branches = new Map<string, Set<string>>();
      for (const edge of edges) {
        if (edge.modelId !== id || gatesByComponent.get(edge.source) !== result.id) continue;
        const targets = branches.get(edge.source) ?? new Set<string>();
        targets.add(gatesByComponent.get(edge.target)!);
        branches.set(edge.source, targets);
      }
      expect([...branches.values()].some((targets) => targets.has("PRE:483") && targets.has("PRE:3182"))).toBe(true);
    }
    expect(data.activationExamples?.find(({ gateId }) => gateId === result.id)?.examples).toHaveLength(4);
  });

  it("includes a searchable measured scenario for the other discovery hypotheses", () => {
    const data = measuredWorkbenchDataset;
    for (const id of ["MID:6439", "PRE:4587", "MID:3437", "PRE:5171",
      "MID:2548", "MID:4214", "PRE:6041", "PRE:7189", "MID:1629"]) {
      expect(searchSharedGates(data.gates, id)[0]?.id).toBe(id);
      const scenario = discoveryScenarioForGate(data.manifest, id);
      expect(scenario?.gateId).toBe(id);
      expect(data.edges.filter((edge) => edge.anchorId === scenario?.anchorId).length).toBeGreaterThan(0);
    }
  });

  it("opens an exact interactive witness for every gate in the measured subset", () => {
    const data = measuredWorkbenchDataset;
    const models = new Set(data.models.map(model => model.id));
    const initial = data.manifest.display!.defaultAnchorId!;
    for (const { id } of data.gates) {
      const view = measuredViewForGate(id, initial, models, data.components, data.edges, data.manifest);
      const selected = componentsForSelection(data.components, view.gates, models, false);
      const byId = new Map(selected.map(component => [component.id, component.gateId]));
      expect(view.measuredAnchors, id).toBeGreaterThan(0);
      expect(edgesForAnchor(data.edges, view.anchorId, new Set(byId.keys()), 0)
        .some(edge => byId.get(edge.source) === id || byId.get(edge.target) === id), id).toBe(true);
    }
    const empty = measuredViewForGate("PRE:99999", initial, models, data.components, data.edges, data.manifest);
    expect(empty).toEqual({ anchorId: initial, gates: new Set(["PRE:99999"]), measuredAnchors: 0 });
  });

  it("lists every measured anchor without a search and identifies the active discovery graph", () => {
    const { manifest, anchors } = measuredWorkbenchDataset;
    const activeAnchorId = discoveryScenarioForGate(manifest, "MID:7027")!.anchorId;
    const html = renderToStaticMarkup(createElement(NaturalValidationScenarios, {
      manifest, activeAnchorId, onSelect: () => undefined,
    }));
    expect(html.match(/<button /g)).toHaveLength(anchors.length);
    expect(html).toContain("10 discovery hypotheses · unvalidated");
    expect(html).toContain('aria-pressed="true"><strong>Opening-parenthesis fan-out</strong>');
    expect(html).toContain("MID:7027 · PRE:483 · PRE:3182");
    expect(html).toContain("Other measured anchors");
  });

  it("contains no cross-model dependency edges", () => {
    const components = new Map(measuredWorkbenchDataset.components.map((component) => [component.id, component]));
    for (const edge of measuredWorkbenchDataset.edges) {
      expect(components.get(edge.source)?.modelId).toBe(edge.modelId);
      expect(components.get(edge.target)?.modelId).toBe(edge.modelId);
    }
  });

  it("covers every displayed gate with hash-provenanced natural firing contexts", () => {
    expect(measuredWorkbenchDataset.manifest.export?.activationExamples?.naturalGateOnly).toBe(true);
    expect(measuredWorkbenchDataset.manifest.export?.activationExamples?.selectedSplit).toBe("test");
    expect(measuredWorkbenchDataset.manifest.export?.activationExamples?.displayTokenizer.modelKey).toBe("gemma");
    expect(measuredWorkbenchDataset.manifest.export?.activationExamples?.displayTokenizer.revision).toBe("299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8");
    expect(measuredWorkbenchDataset.activationExamples).toHaveLength(measuredWorkbenchDataset.gates.length);
    for (const record of measuredWorkbenchDataset.activationExamples ?? []) {
      expect(record.examples.length).toBeGreaterThan(0);
      expect(record.examples.length).toBeLessThanOrEqual(4);
      for (const example of record.examples) {
        expect(example.firings[example.centerPosition]).toBe(true);
        expect(example.activations[example.centerPosition]).toBe(example.activation);
        expect(example.tokenIds).toHaveLength(example.activations.length);
        expect(example.tokenPieces).toHaveLength(example.activations.length);
        expect(example.tokenTexts).toHaveLength(example.activations.length);
      }
    }
  });

  it("rejects a synthetic flag or cross-model edge", () => {
    const synthetic = structuredClone(rawDataset) as unknown as Record<string, any>;
    synthetic.manifest.synthetic = true;
    expect(() => validateMeasuredWorkbenchDataset(synthetic)).toThrow(/synthetic=false/);

    const crossed = structuredClone(rawDataset) as unknown as Record<string, any>;
    crossed.edges[0].modelId = crossed.edges[0].modelId === "gemma" ? "smol" : "gemma";
    expect(() => validateMeasuredWorkbenchDataset(crossed)).toThrow(/crosses models/);

    const brokenContext = structuredClone(rawDataset) as unknown as Record<string, any>;
    brokenContext.activationExamples[0].examples[0].firings[brokenContext.activationExamples[0].examples[0].centerPosition] = false;
    expect(() => validateMeasuredWorkbenchDataset(brokenContext)).toThrow(/natural center firing/);

    const wrongParentRule = structuredClone(rawDataset) as unknown as Record<string, any>;
    wrongParentRule.manifest.export.parentBackend.rule.k_par = 9;
    expect(() => validateMeasuredWorkbenchDataset(wrongParentRule)).toThrow(/parent backend rule/);

    const misalignedTokens = structuredClone(rawDataset) as unknown as Record<string, any>;
    misalignedTokens.activationExamples[0].examples[0].tokenPieces.pop();
    expect(() => validateMeasuredWorkbenchDataset(misalignedTokens)).toThrow(/one-to-one token/);
  });
});
