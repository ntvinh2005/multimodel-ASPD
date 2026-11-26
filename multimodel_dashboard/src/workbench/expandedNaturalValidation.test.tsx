import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { expandedCandidateInNote, expandedCatalog, loadExpandedEvidence, renderableExpandedGraph, searchExpandedCandidates } from "./expandedNaturalValidation";
import { measuredWorkbenchDataset } from "./measuredDataset";
import NaturalValidationScenarios from "./NaturalValidationScenarios";
import { NaturalWitness } from "./ExpandedNaturalValidationGraph";
import { ResearchNotesProvider } from "./ResearchNotes";

describe("expanded Natural Validation graph search", () => {
  it("finds a sheet row by its candidate ID, member gate, or title", () => {
    for (const query of ["NVX-025", "nvx025", "MID:7417", "mid/7417", "7417", "day first month"]) {
      expect(searchExpandedCandidates(query).map(row => row.id)).toContain("NVX-025");
    }
    expect(searchExpandedCandidates("PRE:7417").some(row => row.id === "NVX-025")).toBe(false);
    expect(expandedCandidateInNote("NVX-025 · MID:1117 → MID:7417 → PRE:1529")?.id).toBe("NVX-025");
  });

  it("renders matches as selectable graph results and retains the legacy measured browser", () => {
    const render = (query: string) => renderToStaticMarkup(<NaturalValidationScenarios
      query={query} manifest={measuredWorkbenchDataset.manifest} activeAnchorId={null} activeCandidateId={null}
      matchingGates={[]} onSelect={() => undefined} onQueryChange={() => undefined}
      onOpenGate={() => undefined} onOpenCandidate={() => undefined} />);
    const result = render("MID:7417");
    expect(result).toContain("Measured graph search results");
    expect(result).toContain("NVX-025 · Day-first month position");
    expect(result).toContain("MID:7417");
    expect(render("")).toContain("Other measured anchors");
    expect(render("")).toContain("Candidate graphs (123)");
  });

  it("loads a same-anchor graph for each sheet candidate and separates held-out examples", async () => {
    expect(expandedCatalog.candidates).toHaveLength(123);
    const selectedLayers = new Map(measuredWorkbenchDataset.models.map(model => [model.id, model.selectedLayers]));
    let extensions = 0;
    for (const candidate of expandedCatalog.candidates) {
      const evidence = await loadExpandedEvidence(candidate.id);
      expect(evidence.snapshots[0].edges).toHaveLength(6);
      extensions += evidence.snapshots.length - 1;
      for (const snapshot of evidence.snapshots) {
        const { components, edges } = renderableExpandedGraph(snapshot);
        expect(new Set(edges.map(item => item.anchorId))).toEqual(new Set([snapshot.anchor]));
        expect(edges).toHaveLength(3 * (snapshot.kind === "Core motif" ? 2 : 3));
        expect(new Set(edges.map(item => item.modelId))).toEqual(new Set(["gemma", "smol", "granite"]));
        expect(components.every(component => selectedLayers.get(component.modelId)?.includes(component.layer))).toBe(true);
        expect(Object.values(evidence.examples).flat().every(example => example.anchor !== snapshot.anchor)).toBe(true);
      }
    }
    expect(extensions).toBe(8);
    const witness = await loadExpandedEvidence("NVX-025");
    expect(witness.snapshots[1].edges).toContainEqual(expect.objectContaining({
      source: "gemma.L15.mlp_up/7417", target: "gemma.L15.mlp_down/7417",
    }));
  });

  it("renders the same data-scaled threshold control for candidate graphs and dictionary neighborhoods", async () => {
    const candidate = expandedCatalog.candidates.find(({ id }) => id === "NVX-025")!;
    const evidence = await loadExpandedEvidence(candidate.id);
    const { edges } = evidence.snapshots[0];
    const maximum = Math.max(...edges.map(({ w }) => w));
    for (const gates of [candidate.gates, [candidate.gates[0]]]) {
      const graph = renderToStaticMarkup(<ResearchNotesProvider>
        <NaturalWitness gates={gates} reference={candidate.id} artifact="test"
          evidence={evidence} Graph={() => null} onExploreGate={gates.length === 1 ? () => undefined : undefined} />
      </ResearchNotesProvider>);
      expect(graph).toContain('type="range"');
      expect(graph).toContain(`max="${maximum}"`);
      expect(graph).toContain('aria-label="Minimum normalized edge weight"');
      expect(graph).toContain(`${edges.length}/${edges.length} measured edges shown`);
      expect(graph).toContain('aria-label="Displayed component edges"');
    }
  });
});
