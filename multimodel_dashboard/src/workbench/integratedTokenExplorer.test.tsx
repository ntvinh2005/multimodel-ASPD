import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { graphCoverageLabel, SelectedNoteEditor, TokenAnchorBar, TokenSequence, useSelectedNote } from "./TokenCircuitExplorer";
import { QueryGraphPanel, type QueryGraph } from "./InductionQueryGraph";
import { ResearchNotesProvider } from "./ResearchNotes";
import type { TokenExample, TokenPosition } from "./tokenDiscovery";

const first: TokenPosition = {
  anchor_id: "first", character_boundary: 1, final_query: false, roles_ending: [],
  model_positions: { gemma: 4, smol: 8, granite: 6 }, pool_id: "test",
  pre: { active_gate_ids: [7], values: [.5] }, mid: { active_gate_ids: [], values: [] },
};
const final = { ...first, anchor_id: "final", character_boundary: 3, final_query: true };
const example: TokenExample = {
  example_id: "row", group_id: "group", condition: "negative", prompt: "😀 x", positions: [first, final],
};

it("renders exact clickable prompt spans inside the existing activation heatmap", () => {
  const html = renderToStaticMarkup(<TokenSequence example={example} gate="PRE:7" scale={1} model="smol" anchor="first" onSelect={() => {}} />);
  expect(html).toContain('class="ig-sequence td-sequence"');
  expect(html).toMatch(/aria-pressed="true"[^>]*title="SmolLM2 token 8 · gˢ 0.5 · g 1">😀<\/button>/);
  expect(html).toMatch(/class="final-query" aria-pressed="false"[^>]*> x<\/button>/);
});

it("offers the final-token reset only when reading an earlier anchor", () => {
  const actions = { model: "smol" as const, activeCount: 1, anchorOnly: true, onReset: () => {} };
  expect(renderToStaticMarkup(<TokenAnchorBar {...actions} anchor={first} />)).toContain("Return to final token");
  expect(renderToStaticMarkup(<TokenAnchorBar {...actions} anchor={final} />)).not.toContain("Return to final token");
});

it("distinguishes verified empty graph data from missing and failed reads", () => {
  expect(graphCoverageLabel(false, false, true, false)).toBe("Full component graphs measured at this token");
  expect(graphCoverageLabel(false, false, false, true)).toBe("Some component graphs measured at this token");
  expect(graphCoverageLabel(false, false, false, false)).toBe("Dependency graphs unmeasured at this token");
  expect(graphCoverageLabel(false, true, false, false)).toBe("Graph data could not be loaded or verified");
});

it("uses one note editor for graph annotations even when a measured target has zero retained parents", () => {
  const checkpoint = "a".repeat(64);
  const gate = "PRE:7";
  const graph: QueryGraph = { anchorId: first.anchor_id, activation: .5, comparisons: [], models: {
    smol: { strength: 0, distribution: null, targets: [{
      component: { id: "smol.L08.q/7", gateId: gate, layer: 8, matrix: "Q", modelId: "smol", beta: .1, rho: null },
      coefficient: 1, parents: [],
    }] },
  } };
  function View() {
    const note = useSelectedNote(gate);
    return <ResearchNotesProvider>
      <QueryGraphPanel graph={graph} gateId={gate} model="smol" annotationCheckpoint={checkpoint}
        onAnnotate={note.annotate} Graph={() => <svg />} />
      <SelectedNoteEditor gate={gate} checkpoint={checkpoint} note={note} />
    </ResearchNotesProvider>;
  }
  const html = renderToStaticMarkup(<View />);
  expect(html.match(/class="rn-editor"/g)).toHaveLength(1);
  expect(html).toContain("Annotate this model’s gate");
  expect(html).toContain("No parent passed the strict");
  expect(html).not.toContain("No downstream component was measured");
});
