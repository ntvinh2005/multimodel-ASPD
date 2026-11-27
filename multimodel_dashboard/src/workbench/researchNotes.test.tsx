import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ResearchGateLabel, ResearchNotesEditor, ResearchNotesLibrary, ResearchNotesProvider } from "./ResearchNotes";
import { preliminaryNotes, seedBottomUpBatch2Notes, seedBottomUpNotes, seedExpandedNaturalValidationNotes, seedPreliminaryNotes, seedRelatedCircuitNotes } from "./preliminaryResearchNotes";
import CircuitNoteLinks from "./CircuitNoteLinks";
import bottomUpBackup from "./bottomUpResearchNotes.json";
import bottomUpBatch2Backup from "./bottomUpResearchNotesBatch2.json";
import circuitBackup from "./relatedCircuitResearchNotes.json";
import expandedBackup from "./expandedNaturalValidationResearchNotes.json";
import circuitLinks from "./relatedCircuitLinks.json";
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
  it("imports nine unvalidated SmolLM2 drafts once and preserves existing notes", () => {
    const storage = makeStorage();
    const existing = makeNote(preliminaryNotes[0].target, { label: "My own note", status: "Rejected" });
    storeNote(storage, existing);
    seedPreliminaryNotes(storage);
    const { records } = loadNotes(storage);
    expect(Object.keys(records)).toHaveLength(9);
    expect(records[noteKey(existing.target)]).toEqual(existing);
    const drafts = Object.values(records).filter((note) => note.fields.label !== "My own note");
    expect(drafts.every(({ target, fields }) => target.scope === "model" && target.modelId === "smol"
      && fields.status === "Hypothesis proposed")).toBe(true);
    expect(drafts.some(({ target }) => target.gateId === "PRE:953")).toBe(true);
    storage.removeItem(NOTE_STORAGE_PREFIX + noteKey(preliminaryNotes[1].target));
    seedPreliminaryNotes(storage);
    expect(Object.keys(loadNotes(storage).records)).toHaveLength(8);
  });
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

  it("imports five natural-validation hypotheses once and preserves researcher annotations at occupied targets", () => {
    const storage = makeStorage();
    const drafts = importNotesJson(JSON.stringify(bottomUpBackup));
    const existing = makeNote(drafts[0].target, { label: "Researcher annotation", status: "Rejected" });
    const rawKey = NOTE_STORAGE_PREFIX + noteKey(existing.target);
    storeNote(storage, existing);
    const before = storage.getItem(rawKey);
    seedPreliminaryNotes(storage);
    seedBottomUpNotes(storage);
    const { records } = loadNotes(storage);
    expect(drafts).toHaveLength(5);
    expect(storage.getItem(rawKey)).toBe(before);
    expect(Object.keys(records)).toHaveLength(14);
    expect(drafts.slice(1).map((draft) => records[noteKey(draft.target)])).toEqual(drafts.slice(1));
    expect(drafts.every(({ target, fields }) => target.scope === "shared"
      && fields.status === "Hypothesis proposed")).toBe(true);
    storage.removeItem(NOTE_STORAGE_PREFIX + noteKey(drafts[1].target));
    seedBottomUpNotes(storage);
    expect(Object.keys(loadNotes(storage).records)).toHaveLength(13);
  });

  it("imports batch two into browsers that have already imported the first batch", () => {
    const storage = makeStorage();
    seedBottomUpNotes(storage);
    const firstBatch = loadNotes(storage).records;
    const drafts = importNotesJson(JSON.stringify(bottomUpBatch2Backup));
    const prior = makeNote(drafts[0].target, { label: "Existing assessment", status: "Under investigation" });
    storeNote(storage, prior);
    seedBottomUpBatch2Notes(storage);
    const { records } = loadNotes(storage);
    expect(Object.keys(records)).toHaveLength(10);
    expect(records).toMatchObject(firstBatch);
    expect(records[noteKey(prior.target)]).toEqual(prior);
    expect(drafts.slice(1).map(({ target }) => records[noteKey(target)].fields.status))
      .toEqual(Array(4).fill("Hypothesis proposed"));
  });

  it("adds circuit-relative notes once, with links rendered in both directions", () => {
    const storage = makeStorage();
    seedBottomUpNotes(storage);
    seedBottomUpBatch2Notes(storage);
    const founders = loadNotes(storage).records;
    const drafts = importNotesJson(JSON.stringify(circuitBackup));
    const existing = makeNote(drafts[0].target, { label: "My own circuit note" });
    storeNote(storage, existing);
    seedRelatedCircuitNotes(storage);
    seedRelatedCircuitNotes(storage);
    const { records } = loadNotes(storage);
    expect(Object.keys(records)).toHaveLength(22);
    expect(records).toMatchObject(founders);
    expect(records[noteKey(existing.target)]).toEqual(existing);
    expect(drafts).toHaveLength(circuitLinks.relations.length);
    const target = { checkpoint: circuitLinks.checkpoint, gateId: "MID:7027", scope: "shared" as const };
    const founder = renderToStaticMarkup(<CircuitNoteLinks target={target} onSelect={() => undefined} />);
    const child = renderToStaticMarkup(<CircuitNoteLinks target={{ ...target, gateId: "PRE:1884" }} onSelect={() => undefined} />);
    expect(founder).toContain("Open related note · PRE:1884");
    expect(child).toContain("Open original hypothesis · MID:7027");
    expect(child).toContain("gemma.L09.mlp_gate/7027 → gemma.L15.v/1884");
    expect(renderToStaticMarkup(<CircuitNoteLinks target={{ ...target, checkpoint }} />)).toBe("");
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

  it("seeds 123 expanded hypotheses once and preserves researcher annotations", () => {
    const storage = makeStorage();
    seedPreliminaryNotes(storage);
    seedBottomUpNotes(storage);
    seedBottomUpBatch2Notes(storage);
    seedRelatedCircuitNotes(storage);
    const prior = loadNotes(storage).records;
    const drafts = importNotesJson(JSON.stringify(expandedBackup));
    const researcher = makeNote(drafts[0].target, { label: "Existing researcher assessment", status: "Rejected" });
    storeNote(storage, researcher);
    seedExpandedNaturalValidationNotes(storage);
    seedExpandedNaturalValidationNotes(storage);
    const snapshot = loadNotes(storage);
    expect(snapshot.invalid).toBe(0);
    expect(drafts).toHaveLength(123);
    expect(snapshot.records).toMatchObject(prior);
    expect(snapshot.records[noteKey(researcher.target)]).toEqual(researcher);
    expect(Object.keys(snapshot.records)).toHaveLength(Object.keys(prior).length + drafts.length);
    expect(drafts.slice(1).map((note) => snapshot.records[noteKey(note.target)])).toEqual(drafts.slice(1));
    expect(drafts.every((note) => note.fields.status === "Hypothesis proposed")).toBe(true);
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
