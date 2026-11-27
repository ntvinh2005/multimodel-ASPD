import { EMPTY_NOTE, NOTE_STORAGE_PREFIX, importNotesJson, noteKey, storeNote, type ResearchNote } from "./researchNotes";
import bottomUpBackup from "./bottomUpResearchNotes.json";
import bottomUpBatch2Backup from "./bottomUpResearchNotesBatch2.json";
import relatedCircuitBackup from "./relatedCircuitResearchNotes.json";
import expandedNaturalValidationBackup from "./expandedNaturalValidationResearchNotes.json";

const checkpoint = "735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969";
const importMarker = "aspd.research-note-import.smol-selection-frame.v1";
export const selectionFrameTitle = "Selection Frame Mechanism";
export const selectionFrameHypothesis =
  "These interconnected SmolLM2 gates may participate in a Selection Frame Mechanism representing the selection action, candidate set, and object reference.";
const observations = [
  ["PRE:3207", "the", "Definite article sensitivity or object reference"],
  ["MID:6020", "the", "Definite object reference"],
  ["MID:2323", "choose, selected by, from", "Selection action representation"],
  ["PRE:6968", "Strongest at choose", "Selection action representation"],
  ["PRE:8169", "Strongest at from", "Candidate set or source representation"],
  ["MID:2844", "choose one code from, choose a code", "Selection structure representation"],
  ["MID:4084", "selected by, will choose, will not choose", "Context-dependent selection structure"],
  ["MID:2122", "selected, from, is, choose the, punctuation", "Syntactic structure processing"],
  ["PRE:953", "will be selected by, from, the, punctuation", "Syntactic structure processing"],
] as const;
export const selectionFrameGateIds = observations.map(([gateId]) => gateId);

export const preliminaryNotes: ResearchNote[] = observations.map(([gateId, observation, hypothesis]) => ({
  target: { checkpoint, gateId, scope: "model", modelId: "smol" },
  fields: {
    ...EMPTY_NOTE,
    label: hypothesis,
    observation,
    hypothesis: gateId === "MID:2844" ? `${hypothesis}\n\n${selectionFrameHypothesis}` : hypothesis,
    evidence: [
      "Preliminary observation supplied by the researcher. Intervention evidence is pending.",
      gateId === "PRE:953" && "Site resolved against the four measured Negation groups: PRE:953 activated at 171 aligned positions, including the reported words. MID:953 had zero activations on those sequences.",
    ].filter(Boolean).join("\n"),
    tags: "selection frame, preliminary",
    status: "Hypothesis proposed",
  },
  updatedAt: "2026-11-25T21:12:30.000Z",
}));

function seedMissingNotes(storage: Storage, marker: string, drafts: ResearchNote[]) {
  if (storage.getItem(marker)) return;
  for (const note of drafts) {
    if (storage.getItem(NOTE_STORAGE_PREFIX + noteKey(note.target)) === null) storeNote(storage, note);
  }
  storage.setItem(marker, "complete");
}

export function seedPreliminaryNotes(storage: Storage) {
  seedMissingNotes(storage, importMarker, preliminaryNotes);
}

export function seedBottomUpNotes(storage: Storage) {
  seedMissingNotes(storage, "aspd.research-note-import.natural-validation-61036.v1",
    importNotesJson(JSON.stringify(bottomUpBackup)));
}

export function seedBottomUpBatch2Notes(storage: Storage) {
  seedMissingNotes(storage, "aspd.research-note-import.natural-validation-61036-batch2.v1",
    importNotesJson(JSON.stringify(bottomUpBatch2Backup)));
}

export function seedRelatedCircuitNotes(storage: Storage) {
  seedMissingNotes(storage, "aspd.research-note-import.natural-validation-61036-circuit-relatives.v1",
    importNotesJson(JSON.stringify(relatedCircuitBackup)));
}

export function seedExpandedNaturalValidationNotes(storage: Storage) {
  seedMissingNotes(storage, "aspd.research-note-import.natural-validation-61036-expanded.v1",
    importNotesJson(JSON.stringify(expandedNaturalValidationBackup)));
}
