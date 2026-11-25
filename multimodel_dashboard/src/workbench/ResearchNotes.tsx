import { createContext, useContext, useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import {
  EMPTY_NOTE, EMPTY_NOTE_FILTERS, NOTE_MODELS, NOTE_STATUSES, NOTE_STORAGE_PREFIX, SCOPE_LABEL,
  exportNotesCsv, exportNotesJson, filterNotes, hasNoteContent, importNotesJson, loadNotes,
  noteKey, noteTags, statusClass, storeNote, targetLabel,
  type NoteFields, type NoteRecords, type NoteTarget, type ResearchNote,
} from "./researchNotes";
import "./researchNotes.css";

type NotesState = { records: NoteRecords; warning: string };
type NotesContext = NotesState & {
  edit: (target: NoteTarget, fields: Partial<NoteFields>) => void;
  restore: (notes: ResearchNote[]) => number;
};
const Notes = createContext<NotesContext | null>(null);

function readSaved(): NotesState {
  try {
    const { records, invalid } = loadNotes(window.localStorage);
    return { records, warning: invalid ? `${invalid} unreadable browser note records were skipped.` : "" };
  } catch {
    return { records: {}, warning: "Browser storage is unavailable. Export a backup before closing this page." };
  }
}

export function ResearchNotesProvider({ children }: { children: ReactNode }) {
  const [saved, setSaved] = useState(readSaved);
  const current = useRef(saved);
  function publish(next: NotesState) {
    current.current = next;
    setSaved(next);
  }
  useEffect(() => {
    function sync(event: StorageEvent) {
      if (event.key === null || event.key.startsWith(NOTE_STORAGE_PREFIX)) publish(readSaved());
    }
    window.addEventListener("storage", sync);
    return () => window.removeEventListener("storage", sync);
  }, []);

  function save(note: ResearchNote, records: NoteRecords) {
    const key = noteKey(note.target);
    if (hasNoteContent(note.fields)) records[key] = note;
    else delete records[key];
    try {
      storeNote(window.localStorage, note);
      return true;
    } catch {
      return false;
    }
  }
  function edit(target: NoteTarget, fields: Partial<NoteFields>) {
    const records = { ...current.current.records };
    const previous = records[noteKey(target)]?.fields ?? EMPTY_NOTE;
    const note = { target, fields: { ...previous, ...fields }, updatedAt: new Date().toISOString() };
    const persisted = save(note, records);
    publish({
      records, warning: persisted ? current.current.warning
        : "Autosave failed. Your edit is available on this page. Export a backup before closing it.",
    });
  }
  function restore(notes: ResearchNote[]) {
    const records = { ...current.current.records };
    let restored = 0;
    let failed = false;
    for (const note of notes) {
      const existing = records[noteKey(note.target)];
      if (existing && Date.parse(existing.updatedAt) >= Date.parse(note.updatedAt)) continue;
      if (!hasNoteContent(note.fields)) continue;
      failed = !save(note, records) || failed;
      restored++;
    }
    publish({
      records, warning: failed ? "Browser storage could not save every imported note. Export a backup."
        : current.current.warning,
    });
    return restored;
  }
  return <Notes.Provider value={{ ...saved, edit, restore }}>{children}</Notes.Provider>;
}

export function useResearchNotes() {
  const context = useContext(Notes);
  if (!context) throw new Error("Missing ResearchNotesProvider");
  return context;
}

export function ResearchGateLabel({ checkpoint, gateId }: { checkpoint: string; gateId: string }) {
  const { records } = useResearchNotes();
  const fields = records[noteKey({ checkpoint, gateId, scope: "shared" })]?.fields;
  if (!fields) return gateId;
  return <span className="rn-gate-label" title={`Shared gate · ${fields.status}${fields.hypothesis ? `\nHypothesis: ${fields.hypothesis}` : ""}`}>
    {gateId}{fields.label && <small>{fields.label}</small>}
    <i aria-label={fields.status} className={`rn-dot ${statusClass(fields.status)}`} />
  </span>;
}

export function NotesLocatorCell({ target, title, beta, representative, onSelect, children }: {
  target?: NoteTarget; title: string; beta: number | null; representative: boolean;
  onSelect: (target: NoteTarget) => void; children: ReactNode;
}) {
  const { records } = useResearchNotes();
  const fields = target && records[noteKey(target)]?.fields;
  const hover = fields ? `${title}\nComponent · ${fields.status}${fields.label ? ` · ${fields.label}` : ""}${fields.hypothesis ? `\nHypothesis: ${fields.hypothesis}` : ""}` : title;
  return <i className={[representative ? "representative" : "", target ? "rn-locator-cell" : ""].join(" ")}
    style={{ "--beta": Math.min(1, beta ?? 0) } as CSSProperties}
    title={hover} role={target ? "button" : undefined} tabIndex={target ? 0 : undefined}
    aria-label={target ? `Research notes for ${targetLabel(target)} · ${hover}` : undefined}
    onClick={target ? () => onSelect(target) : undefined}
    onKeyDown={target ? (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onSelect(target);
      }
    } : undefined}>
    {children}
    {fields && <em className={`rn-dot ${statusClass(fields.status)}`} />}
  </i>;
}

export function ResearchNotesEditor({ targets, evidenceReference, openByDefault = false }: {
  targets: NoteTarget[];
  evidenceReference?: string;
  openByDefault?: boolean;
}) {
  const defaultTarget = targets.find((target) => target.scope === "component") ?? targets[0];
  const [chosen, setChosen] = useState(() => noteKey(defaultTarget));
  const { records } = useResearchNotes();
  const target = targets.find((item) => noteKey(item) === chosen) ?? defaultTarget;
  const note = records[noteKey(target)];
  return <details className="rn-editor" open={openByDefault || undefined}>
    <summary>Research notes {note && <span className={`rn-status ${statusClass(note.fields.status)}`}>{note.fields.status}</span>}</summary>
    <div className="rn-editor-body">
      {targets.length > 1 && <label className="rn-field">Annotation scope
        <select value={noteKey(target)} onChange={(event) => setChosen(event.target.value)}>
          {targets.map((item) => <option key={noteKey(item)} value={noteKey(item)}>{SCOPE_LABEL[item.scope]} · {targetLabel(item)}</option>)}
        </select>
      </label>}
      <NoteFieldsEditor target={target} evidenceReference={evidenceReference} />
    </div>
  </details>;
}

function NoteFieldsEditor({ target, evidenceReference }: { target: NoteTarget; evidenceReference?: string }) {
  const { records, edit, warning } = useResearchNotes();
  const fields = records[noteKey(target)]?.fields ?? EMPTY_NOTE;
  return <div className="rn-fields">
    <p className="rn-identity">{SCOPE_LABEL[target.scope]} · {targetLabel(target)} · <code title={target.checkpoint}>{target.checkpoint.slice(0, 12)}</code></p>
    <label className="rn-field">Short label
      <input type="text" value={fields.label} maxLength={80} onChange={(event) => edit(target, { label: event.target.value })} />
    </label>
    <label className="rn-field">Observation
      <textarea rows={2} value={fields.observation} onChange={(event) => edit(target, { observation: event.target.value })} />
    </label>
    <label className="rn-field">Hypothesis
      <textarea rows={2} value={fields.hypothesis} onChange={(event) => edit(target, { hypothesis: event.target.value })} />
    </label>
    <label className="rn-field">Evidence <small>Supporting or contradicting examples, dependencies, or interventions</small>
      <textarea rows={3} value={fields.evidence} onChange={(event) => edit(target, { evidence: event.target.value })} />
    </label>
    {evidenceReference && <button type="button" disabled={fields.evidence.includes(evidenceReference)}
      onClick={() => edit(target, { evidence: [fields.evidence, evidenceReference].filter(Boolean).join("\n") })}>
      Add current context reference
    </button>}
    <label className="rn-field">Researcher status
      <select value={fields.status} onChange={(event) => {
        const status = NOTE_STATUSES.find((item) => item === event.target.value);
        if (status) edit(target, { status });
      }}>
        {NOTE_STATUSES.map((status) => <option key={status}>{status}</option>)}
      </select>
    </label>
    <label className="rn-field">Tags <small>Comma separated</small>
      <input type="text" value={fields.tags} onChange={(event) => edit(target, { tags: event.target.value })} />
    </label>
    <p className="rn-storage" role={warning ? "alert" : undefined}>{warning || "Autosaves in this browser. Export JSON to transfer notes to another browser or tunnel port."}</p>
  </div>;
}

function download(content: string, type: string, suffix: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `aspd-research-notes.${suffix}`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function ResearchNotesLibrary() {
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const [filters, setFilters] = useState(EMPTY_NOTE_FILTERS);
  const [selected, setSelected] = useState<NoteTarget | null>(null);
  const [message, setMessage] = useState("");
  const { records, restore, warning } = useResearchNotes();
  const notes = Object.values(records);
  const tags = [...new Set(notes.flatMap((note) => noteTags(note.fields.tags)))].sort();
  const models = [...new Set(notes.flatMap(({ target }) => target.scope === "shared" ? [] : [target.modelId]))];
  const results = filterNotes(notes, filters);
  return <>
    <button type="button" className="rn-library-trigger" onClick={() => dialog.current?.showModal()}>Research notes <small>{notes.length}</small></button>
    <dialog ref={dialog} className="rn-library" aria-label="Research notes and hypotheses">
      <header><h2>Research notes</h2><button type="button" onClick={() => dialog.current?.close()}>Close</button></header>
      <div className="rn-library-actions">
        <button type="button" disabled={!notes.length} onClick={() => download(exportNotesJson(notes), "application/json", "json")}>Export JSON</button>
        <button type="button" disabled={!notes.length} onClick={() => download(exportNotesCsv(notes), "text/csv;charset=utf-8", "csv")}>Export CSV</button>
        <button type="button" onClick={() => input.current?.click()}>Import JSON</button>
        <input hidden ref={input} type="file" accept=".json,application/json" onChange={async (event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (!file) return;
          try {
            const count = restore(importNotesJson(await file.text()));
            setMessage(count ? `Imported ${count} new or newer notes.` : "Your browser already has notes at least as recent as this backup.");
          } catch {
            setMessage("Could not import this file. Choose a Research Notes JSON export.");
          }
        }} />
      </div>
      {(message || warning) && <p className="rn-storage" role="status">{[message, warning].filter(Boolean).join(" ")}</p>}
      <div className="rn-library-filters">
        <label className="rn-field">Search
          <input type="search" value={filters.search} onChange={(event) => setFilters({ ...filters, search: event.target.value })} placeholder="Gate ID, label, evidence…" />
        </label>
        <label className="rn-field">Hypothesis
          <input type="search" value={filters.hypothesis} onChange={(event) => setFilters({ ...filters, hypothesis: event.target.value })} />
        </label>
        <label className="rn-field">Model
          <select value={filters.model} onChange={(event) => setFilters({ ...filters, model: event.target.value })}>
            <option value="">All scopes</option><option value="shared">Shared gates</option>
            {models.map((model) => <option key={model} value={model}>{NOTE_MODELS[model] ?? model}</option>)}
          </select>
        </label>
        <label className="rn-field">Status
          <select value={filters.status} onChange={(event) => setFilters({ ...filters, status: event.target.value })}>
            <option value="">All statuses</option>{NOTE_STATUSES.map((status) => <option key={status}>{status}</option>)}
          </select>
        </label>
        <label className="rn-field">Tag
          <select value={filters.tag} onChange={(event) => setFilters({ ...filters, tag: event.target.value })}>
            <option value="">All tags</option>{tags.map((tag) => <option key={tag}>{tag}</option>)}
          </select>
        </label>
      </div>
      <div className="rn-table">
        <table>
          <caption>{results.length} of {notes.length} saved notes</caption>
          <thead><tr><th scope="col">Target</th><th scope="col">Label</th><th scope="col">Hypothesis</th><th scope="col">Status</th><th scope="col">Tags</th></tr></thead>
          <tbody>{results.map(({ target, fields }) => <tr key={noteKey(target)}>
            <th scope="row"><button type="button" onClick={() => setSelected(target)}>{targetLabel(target)}</button>
              <small title={target.checkpoint}>{SCOPE_LABEL[target.scope]} · {target.checkpoint.slice(0, 12)}</small>
            </th>
            <td>{fields.label}</td><td>{fields.hypothesis}</td>
            <td><span className={`rn-status ${statusClass(fields.status)}`}>{fields.status}</span></td>
            <td>{noteTags(fields.tags).join(", ")}</td>
          </tr>)}</tbody>
        </table>
      </div>
      {!results.length && <p>{notes.length ? "No notes match these filters." : "Select a gate or component to start recording notes."}</p>}
      {selected && <section className="rn-library-note" aria-label="Edit selected note">
        <header><strong>Edit selected note</strong><button type="button" onClick={() => setSelected(null)}>Done</button></header>
        <NoteFieldsEditor target={selected} />
      </section>}
    </dialog>
  </>;
}
