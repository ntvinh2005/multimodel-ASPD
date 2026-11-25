import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ResearchGateLabel, ResearchNotesEditor, ResearchNotesLibrary, ResearchNotesProvider } from "./ResearchNotes";
import {
  EMPTY_NOTE, EMPTY_NOTE_FILTERS, NOTE_STORAGE_PREFIX, exportNotesCsv, exportNotesJson, filterNotes,
  importNotesJson, loadNotes, noteKey, notesForGate, notesForSelection, selectionNotes, storeNote,
  type NoteTarget, type ResearchNote,
} from "./researchNotes";

const checkpoint = "a".repeat(64);
const gateId = "MID:2844";
const [shared, model, component] = notesForSelection(checkpoint, gateId, "gemma", { layer: 10, matrix: "up" });

function makeNote(target: NoteTarget, fields: Partial<ResearchNote["fields"]>): ResearchNote {
  return { target, fields: { ...EMPTY_NOTE, ...fields }, updatedAt: "2026-11-25T18:00:00.000Z" };
}

function makeStorage(): Storage {
  const data = new Map<string, string>();
  return {
    get length() { return data.size; },
    clear: () => data.clear(),
    key: (index) => [...data.keys()][index] ?? null,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => { data.set(key, value); },
    removeItem: (key) => { data.delete(key); },
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("checkpoint-scoped research notes", () => {
  it("restores distinct checkpoint, site, model, layer and matrix identities after reload", () => {
    const locations = [
      shared, model, component,
      { ...component, layer: 11 } as NoteTarget,
      { ...component, matrix: "down" } as NoteTarget,
      { ...component, modelId: "smol" } as NoteTarget,
      { ...component, checkpoint: "b".repeat(64) },
      { ...shared, gateId: "PRE:2844" },
    ];
    const storage = makeStorage();
    locations.forEach((target, i) => storeNote(storage, makeNote(target, { label: `label ${i}` })));
    const { records } = loadNotes(storage);
    expect(Object.keys(records)).toHaveLength(locations.length);
    for (const [i, target] of locations.entries()) expect(records[noteKey(target)].fields.label).toBe(`label ${i}`);
    expect(selectionNotes(records, [shared, model, component]).map(({ fields }) => fields.label)).toEqual(["label 2", "label 1", "label 0"]);
    expect(selectionNotes(records, notesForSelection(checkpoint, gateId, "granite")).map(({ fields }) => fields.label)).toEqual(["label 0"]);
  });

  it("keeps evidence separate, skips corrupt entries and clears an empty draft", () => {
    const storage = makeStorage();
    const note = makeNote(component, { observation: "Positive on controls", evidence: "anchor abc", hypothesis: "Selection effect" });
    storeNote(storage, note);
    storage.setItem(NOTE_STORAGE_PREFIX + noteKey(model), JSON.stringify(note));
    expect(loadNotes(storage)).toEqual({ records: { [noteKey(component)]: note }, invalid: 1 });
    storeNote(storage, makeNote(component, {}));
    expect(loadNotes(storage).records).toEqual({});
  });

  it("filters by hypothesis text, model, status, tag and ordinary search", () => {
    const a = makeNote(model, { label: "Selection", hypothesis: "Semantic exclusion", tags: "Negation, controls", status: "Hypothesis proposed" });
    const b = makeNote(shared, { label: "Selection", hypothesis: "General constraint", tags: "Negation" });
    const c = makeNote(component, { label: "Bound", hypothesis: "Semantic exclusion", tags: "controls", status: "Rejected" });
    const records = [a, b, c];
    expect(filterNotes(records, { ...EMPTY_NOTE_FILTERS, model: "shared" })).toEqual([b]);
    expect(filterNotes(records, { ...EMPTY_NOTE_FILTERS, search: "selection", hypothesis: "EXCLUSION", model: "gemma", tag: "controls", status: "Hypothesis proposed" })).toEqual([a]);
    expect(filterNotes(records, { ...EMPTY_NOTE_FILTERS, tag: "controls", status: "Rejected" })).toEqual([c]);
  });

  it("round-trips JSON, checks imported identities and escapes spreadsheet formulas", () => {
    const records = [makeNote(component, { label: '=HYPERLINK("x")', observation: 'A,"B"\nC' })];
    const serialized = exportNotesJson(records);
    expect(importNotesJson(serialized)).toEqual(records);
    expect(() => importNotesJson(serialized.replace(checkpoint, "61036"))).toThrow("valid");
    const csv = exportNotesCsv(records);
    expect(csv).toContain(`"${checkpoint}","component","MID:2844","gemma","10","up"`);
    expect(csv).toContain('"\'=HYPERLINK(""x"")"');
    expect(csv).toContain('"A,""B""\nC"');
  });

  it("renders the scoped editors and saved labels from browser storage", () => {
    const storage = makeStorage();
    vi.stubGlobal("window", { localStorage: storage });
    storeNote(storage, makeNote(shared, { label: "Selection", hypothesis: "Shared hypothesis" }));
    storeNote(storage, makeNote(component, { observation: "Local observation", status: "Under investigation" }));
    const html = renderToStaticMarkup(<ResearchNotesProvider>
      <ResearchGateLabel checkpoint={checkpoint} gateId={gateId} />
      <ResearchNotesEditor targets={notesForGate(checkpoint, gateId)} />
      <ResearchNotesEditor targets={[shared, model, component]} openByDefault />
      <ResearchNotesLibrary />
    </ResearchNotesProvider>);
    expect(html).toContain("Shared hypothesis");
    expect(html).toContain("Local observation");
    expect(html).toContain("Model-specific gate");
    expect(html).toContain("Under investigation");
    expect(html).toContain("Export CSV");
  });
});
