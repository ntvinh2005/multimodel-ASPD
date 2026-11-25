import type { Component, MatrixType } from "./types";

export const NOTE_STATUSES = [
  "Unreviewed", "Under investigation", "Hypothesis proposed", "Causally validated", "Rejected",
] as const;
export type NoteStatus = typeof NOTE_STATUSES[number];

type GateIdentity = { checkpoint: string; gateId: string };
export type NoteTarget = GateIdentity & (
  { scope: "shared" }
  | { scope: "model"; modelId: string }
  | { scope: "component"; modelId: string; layer: number; matrix: MatrixType }
);
export type NoteFields = {
  label: string;
  observation: string;
  hypothesis: string;
  evidence: string;
  status: NoteStatus;
  tags: string;
};
export type ResearchNote = { target: NoteTarget; fields: NoteFields; updatedAt: string };
export type NoteRecords = Record<string, ResearchNote>;

export const NOTE_STORAGE_PREFIX = "aspd.research-note.v1/";
export const EMPTY_NOTE: NoteFields = {
  label: "", observation: "", hypothesis: "", evidence: "", status: "Unreviewed", tags: "",
};
export const SCOPE_LABEL = { shared: "Shared gate", model: "Model-specific gate", component: "Component" };
export const NOTE_MODELS: Record<string, string> = { gemma: "Gemma 2", smol: "SmolLM2", granite: "Granite 3.3" };
const matrices: readonly MatrixType[] = ["Q", "K", "V", "O", "gate", "up", "down"];

export function noteKey(target: NoteTarget): string {
  const identity = [target.checkpoint, target.scope, target.gateId];
  if (target.scope === "shared") return JSON.stringify(identity);
  if (target.scope === "model") return JSON.stringify([...identity, target.modelId]);
  return JSON.stringify([...identity, target.modelId, target.layer, target.matrix]);
}

export function notesForSelection(
  checkpoint: string,
  gateId: string,
  modelId?: string,
  component?: Pick<Component, "layer" | "matrix">,
): NoteTarget[] {
  const targets: NoteTarget[] = [{ checkpoint, gateId, scope: "shared" }];
  if (modelId) {
    targets.push({ checkpoint, gateId, scope: "model", modelId });
    if (component) targets.push({ checkpoint, gateId, scope: "component", modelId, layer: component.layer, matrix: component.matrix });
  }
  return targets;
}

export function notesForGate(checkpoint: string, gateId: string, modelIds = Object.keys(NOTE_MODELS)): NoteTarget[] {
  return [
    { checkpoint, gateId, scope: "shared" },
    ...modelIds.map((modelId) => ({ checkpoint, gateId, scope: "model" as const, modelId })),
  ];
}

export function targetLabel(target: NoteTarget) {
  if (target.scope === "shared") return target.gateId;
  const model = NOTE_MODELS[target.modelId] ?? target.modelId;
  return `${target.gateId} · ${model}${target.scope === "component" ? ` · L${target.layer}/${target.matrix}` : ""}`;
}

export function statusClass(status: NoteStatus) {
  return status.toLowerCase().replaceAll(" ", "-");
}

export function selectionNotes(records: NoteRecords, targets: NoteTarget[]) {
  return targets.slice().reverse().flatMap((target) => {
    const note = records[noteKey(target)];
    return note ? [note] : [];
  });
}

export function noteTags(value: string) {
  return [...new Set(value.split(",").map((tag) => tag.trim()).filter(Boolean))];
}

export function hasNoteContent(fields: NoteFields) {
  return fields.status !== "Unreviewed"
    || [fields.label, fields.observation, fields.hypothesis, fields.evidence, fields.tags].some((value) => value.trim());
}

function validNote(value: unknown): value is ResearchNote {
  if (!value || typeof value !== "object") return false;
  const note = value as ResearchNote;
  const t = note.target;
  const f = note.fields;
  return Boolean(t && f
    && typeof t.checkpoint === "string" && /^[a-f0-9]{64}$/.test(t.checkpoint)
    && typeof t.gateId === "string" && /^(PRE|MID):\d+$/.test(t.gateId)
    && (t.scope === "shared"
      || ((t.scope === "model" || t.scope === "component") && typeof t.modelId === "string" && t.modelId.length > 0
        && (t.scope === "model" || Number.isInteger(t.layer) && t.layer >= 0 && matrices.includes(t.matrix))))
    && [f.label, f.observation, f.hypothesis, f.evidence, f.tags].every((field) => typeof field === "string")
    && NOTE_STATUSES.includes(f.status)
    && typeof note.updatedAt === "string" && Number.isFinite(Date.parse(note.updatedAt)));
}

export function readNote(raw: string): ResearchNote {
  const note: unknown = JSON.parse(raw);
  if (!validNote(note)) throw new Error("Invalid research note or checkpoint identity.");
  return note;
}

export function loadNotes(storage: Storage): { records: NoteRecords; invalid: number } {
  const records: NoteRecords = {};
  let invalid = 0;
  for (let index = 0; index < storage.length; index++) {
    const key = storage.key(index);
    if (!key?.startsWith(NOTE_STORAGE_PREFIX)) continue;
    try {
      const note = readNote(storage.getItem(key) ?? "");
      if (key !== NOTE_STORAGE_PREFIX + noteKey(note.target)) throw new Error("Note key mismatch.");
      records[noteKey(note.target)] = note;
    } catch {
      invalid++;
    }
  }
  return { records, invalid };
}

export function storeNote(storage: Storage, note: ResearchNote) {
  const key = NOTE_STORAGE_PREFIX + noteKey(note.target);
  if (hasNoteContent(note.fields)) storage.setItem(key, JSON.stringify(note));
  else storage.removeItem(key);
}

export type NoteFilters = { search: string; hypothesis: string; model: string; status: string; tag: string };
export const EMPTY_NOTE_FILTERS: NoteFilters = { search: "", hypothesis: "", model: "", status: "", tag: "" };

export function filterNotes(records: ResearchNote[], filters: NoteFilters) {
  const match = (content: string, query: string) => content.toLowerCase().includes(query.trim().toLowerCase());
  return records.filter(({ target, fields }) =>
    match([targetLabel(target), target.checkpoint, fields.label, fields.observation, fields.hypothesis, fields.evidence, fields.tags].join(" "), filters.search)
    && match(fields.hypothesis, filters.hypothesis)
    && (!filters.model || filters.model === (target.scope === "shared" ? "shared" : target.modelId))
    && (!filters.status || filters.status === fields.status)
    && (!filters.tag || noteTags(fields.tags).includes(filters.tag)),
  ).sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
}

export function exportNotesJson(records: ResearchNote[]) {
  return JSON.stringify({ schemaVersion: 1, notes: records }, null, 2);
}

export function importNotesJson(raw: string): ResearchNote[] {
  const backup = JSON.parse(raw);
  if (backup?.schemaVersion !== 1 || !Array.isArray(backup.notes) || !backup.notes.every(validNote)) {
    throw new Error("Choose a valid Research Notes JSON export.");
  }
  return backup.notes;
}

export function exportNotesCsv(records: ResearchNote[]) {
  const rows: (string | number)[][] = [[
    "checkpoint_manifest_sha256", "scope", "gate_id", "model_id", "layer", "matrix",
    "short_label", "observation", "hypothesis", "evidence", "status", "tags", "updated_at",
  ]];
  for (const { target, fields, updatedAt } of records) {
    rows.push([
      target.checkpoint, target.scope, target.gateId, target.scope === "shared" ? "" : target.modelId,
      target.scope === "component" ? target.layer : "", target.scope === "component" ? target.matrix : "",
      fields.label, fields.observation, fields.hypothesis, fields.evidence, fields.status, fields.tags, updatedAt,
    ]);
  }
  return rows.map((row) => row.map((cell) => {
    const text = String(cell);
    const safe = /^\s*[=+@-]|^[\t\r\n]/.test(text) ? "'" + text : text;
    return '"' + safe.replaceAll('"', '""') + '"';
  }).join(",")).join("\r\n");
}
