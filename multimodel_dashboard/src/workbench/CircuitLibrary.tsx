import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import seed from "./data/bottomUpCircuits.json";
import { NOTE_MODELS } from "./researchNotes";
import {
  CIRCUIT_STATUSES, COMPOSITION_STORAGE, componentIdentity, EMPTY_LIBRARY, loadCircuitLibrary, mergeCircuits, mergeMeasurements, resolveComponents, saveCircuitLibrary,
  restrictMeasurement, validateLibrary, type Circuit, type CircuitClaim, type CircuitLibrary, type Measurement,
} from "./circuitComposition";
import "./circuitComposition.css";

type Draft = { components: string[]; measurements: Measurement[] };
const EMPTY_DRAFT: Draft = { components: [], measurements: [] };
type LibraryContext = {
  library: CircuitLibrary;
  drafts: Record<string, Draft>;
  warning: string;
  toggle: (checkpoint: string, component: string, measurement?: Measurement) => void;
  clear: (checkpoint: string) => void;
  persist: (library: CircuitLibrary) => void;
};
const Library = createContext<LibraryContext | null>(null);

function readLibrary() {
  return loadCircuitLibrary(window.localStorage, seed);
}

export function CircuitLibraryProvider({ children }: { children: ReactNode }) {
  const [initial] = useState(() => {
    try { return { library: readLibrary(), warning: "" }; }
    catch { return { library: EMPTY_LIBRARY, warning: "Circuit storage is unavailable or invalid. Existing data was preserved." }; }
  });
  const [library, setLibrary] = useState(initial.library);
  const [warning, setWarning] = useState(initial.warning);
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  useEffect(() => {
    function onStorage(event: StorageEvent) {
      if (event.key !== COMPOSITION_STORAGE && event.key !== null) return;
      try { setLibrary(readLibrary()); setWarning(""); }
      catch { setWarning("A circuit storage update could not be read."); }
    }
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);
  function persist(next: CircuitLibrary) {
    saveCircuitLibrary(window.localStorage, next);
    setLibrary(next);
    setWarning("");
  }
  function clear(checkpoint: string) { setDrafts(old => ({ ...old, [checkpoint]: EMPTY_DRAFT })); }
  function toggle(checkpoint: string, id: string, measurement?: Measurement) {
    componentIdentity(id);
    setDrafts(old => {
      const draft = old[checkpoint] ?? EMPTY_DRAFT;
      const components = draft.components.includes(id) ? draft.components.filter(value => value !== id)
        : [...draft.components, id];
      const members = new Set(components);
      const measurements = (measurement?.scope.checkpoint === checkpoint
        ? [...draft.measurements, measurement] : draft.measurements)
        .map(record => restrictMeasurement(record, members)).filter(record => record.targets.length || record.edges.length);
      return { ...old, [checkpoint]: { components, measurements } };
    });
  }
  return <Library.Provider value={{ library, drafts, warning, toggle, clear, persist }}>{children}</Library.Provider>;
}

export function useCircuitLibrary() {
  return useContext(Library);
}

export function CircuitCaptureButton({ checkpoint, id, measurement }: { checkpoint: string; id: string; measurement?: Measurement }) {
  const state = useCircuitLibrary();
  if (!state) return null;
  const draft = state.drafts[checkpoint] ?? EMPTY_DRAFT;
  const selected = draft.components.includes(id);
  return <div className="cc-capture">
    <button type="button" aria-pressed={selected} onClick={() => state.toggle(checkpoint, id, measurement)}>
      {selected ? "Remove from circuit draft" : "Add to circuit draft"}
    </button>
    <small>{draft.components.length} in draft</small>
  </div>;
}

function blank(checkpoint: string): Circuit {
  return { id: "", checkpoint, name: "", description: "", hypothesis: "", evidence: "", status: "Unreviewed",
    components: [], children: [], snapshots: [], claims: [] };
}

export function CircuitLibraryButton({ checkpoint }: { checkpoint?: string }) {
  const state = useCircuitLibrary();
  const dialog = useRef<HTMLDialogElement>(null);
  const [opened, setOpened] = useState(false);
  if (!checkpoint) return null;
  const draftCount = state?.drafts[checkpoint]?.components.length ?? 0;
  return <>
    <button type="button" title="Shift-click graph components to draft a circuit" onClick={() => { dialog.current?.showModal(); setOpened(true); }}>
      Circuits{draftCount > 0 ? ` (${draftCount} drafted)` : ""}
    </button>
    <dialog className="rn-library cc-dialog" ref={dialog} onClose={() => setOpened(false)} aria-label="Circuit library">
      <header><h2>Circuit library</h2><button type="button" onClick={() => dialog.current?.close()}>Close</button></header>
      {opened && <CircuitManager checkpoint={checkpoint} />}
    </dialog>
  </>;
}

function CircuitManager({ checkpoint }: { checkpoint: string }) {
  const state = useCircuitLibrary()!;
  const draft = state.drafts[checkpoint] ?? EMPTY_DRAFT;
  const circuits = state.library.circuits.filter(circuit => circuit.checkpoint === checkpoint);
  const [editor, setEditor] = useState<Circuit | null>(null);
  const [componentText, setComponentText] = useState("");
  const [message, setMessage] = useState("");
  const byId = new Map(circuits.map(circuit => [circuit.id, circuit]));
  function edit(circuit: Circuit) {
    setMessage("");
    setEditor(circuit);
    setComponentText(circuit.components.join("\n"));
  }
  function save() {
    if (!editor) return;
    try {
      const components = [...new Set(componentText.split(/\s+/).filter(Boolean))];
      const id = editor.id || globalThis.crypto?.randomUUID?.()
        || `circuit-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const next = { ...editor, id, name: editor.name.trim(), components };
      const pending = new Map(byId).set(id, next);
      const members = new Set(resolveComponents(id, pending));
      const captured = draft.measurements.map(measurement => restrictMeasurement(measurement, members))
        .filter(measurement => measurement.targets.length);
      next.snapshots = mergeMeasurements([...editor.snapshots, ...captured]);
      state.persist(mergeCircuits(state.library, { version: 1, circuits: [next] }, true));
      setEditor(next);
      setMessage("Circuit saved.");
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); }
  }
  function exportLibrary() {
    const blob = new Blob([JSON.stringify(state.library, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const file = document.createElement("a");
    file.href = url;
    file.download = "aspd-circuits.json";
    file.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  function remove(id: string) {
    try {
      state.persist(validateLibrary({ version: 1, circuits: state.library.circuits.filter(circuit => circuit.id !== id) }));
      if (editor?.id === id) setEditor(null);
      setMessage("Circuit removed.");
    } catch { setMessage("Remove references to this circuit from parent circuits and relationship notes first."); }
  }
  return <>
    <p>Definitions are saved in this browser. Checkpoint <code title={checkpoint}>{checkpoint.slice(0, 12)}</code></p>
    <div className="cc-actions">
      <button type="button" onClick={() => edit({ ...blank(checkpoint), components: draft.components })}>New circuit</button>
      <button type="button" onClick={() => state.clear(checkpoint)} disabled={!draft.components.length}>Clear draft ({draft.components.length})</button>
      <button type="button" onClick={exportLibrary}>Export circuits</button>
      <label className="cc-import">Import circuit JSON <input type="file" accept=".json,application/json" onChange={async event => {
        const file = event.currentTarget.files?.[0];
        event.currentTarget.value = "";
        if (!file) return;
        try {
          state.persist(mergeCircuits(state.library, validateLibrary(JSON.parse(await file.text()))));
          setMessage("Imported circuit definitions. Existing IDs were kept.");
        } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); }
      }} /></label>
    </div>
    {(message || state.warning) && <p role="status">{message || state.warning}</p>}
    <ul className="cc-library-list">{circuits.map(circuit => <li key={circuit.id}>
      <button type="button" onClick={() => edit(circuit)}>{circuit.name}</button>
      <span>{resolveComponents(circuit.id, byId).length} components · {circuit.children.length} nested · {circuit.status}</span>
    </li>)}</ul>
    {editor && <section className="cc-editor">
      <div className="cc-form">
        <label>Name <input value={editor.name} onChange={event => setEditor({ ...editor, name: event.target.value })} /></label>
        <label>Validation status <select value={editor.status} onChange={event => setEditor({ ...editor, status: event.target.value as Circuit["status"] })}>
          {CIRCUIT_STATUSES.map(status => <option key={status}>{status}</option>)}
        </select></label>
        {(["description", "hypothesis", "evidence"] as const).map(field => <label key={field}>
          {field[0].toUpperCase() + field.slice(1)}
          <textarea value={editor[field]} rows={2} onChange={event => setEditor({ ...editor, [field]: event.target.value })} />
        </label>)}
        <label>Direct component IDs <textarea value={componentText} rows={4}
          placeholder="smol.L08.mlp_down/6020" onChange={event => setComponentText(event.target.value)} /></label>
      </div>
      <button type="button" disabled={!draft.components.length}
        onClick={() => setComponentText([...new Set([...componentText.split(/\s+/).filter(Boolean), ...draft.components])].join("\n"))}>
        Add draft components
      </button>
      <fieldset><legend>Nested circuits</legend>
        {circuits.filter(circuit => circuit.id !== editor.id).map(circuit => <label key={circuit.id}>
          <input type="checkbox" checked={editor.children.includes(circuit.id)} onChange={event => setEditor({
            ...editor, children: event.target.checked ? [...editor.children, circuit.id] : editor.children.filter(id => id !== circuit.id),
          })} /> {circuit.name}
        </label>)}
      </fieldset>
      <fieldset><legend>Researcher relationship notes</legend>
        {editor.claims.map((claim, index) => <ClaimEditor key={index} claim={claim} circuits={circuits.filter(circuit => circuit.id !== editor.id)}
          update={value => setEditor({ ...editor, claims: editor.claims.map((item, i) => i === index ? value : item) })}
          remove={() => setEditor({ ...editor, claims: editor.claims.filter((_, i) => i !== index) })} />)}
        <button type="button" disabled={!circuits.some(circuit => circuit.id !== editor.id)} onClick={() => {
          const target = circuits.find(circuit => circuit.id !== editor.id);
          if (target) setEditor({ ...editor, claims: [...editor.claims, { target: target.id, statement: "", evidence: "", status: "Hypothesis proposed" }] });
        }}>Add relationship note</button>
      </fieldset>
      <div className="cc-actions"><button type="button" onClick={save}>Save circuit</button>
        {editor.id && <button type="button" onClick={() => { if (window.confirm(`Remove “${editor.name}”?`)) remove(editor.id); }}>Delete circuit</button>}
        <span>{editor.snapshots.length} saved measurement anchors</span>
      </div>
    </section>}
  </>;
}

function ClaimEditor({ claim, circuits, update, remove }: {
  claim: CircuitClaim; circuits: Circuit[]; update: (claim: CircuitClaim) => void; remove: () => void;
}) {
  return <div className="cc-form">
    <label>Target circuit <select value={claim.target} onChange={event => update({ ...claim, target: event.target.value })}>
      {circuits.map(circuit => <option key={circuit.id} value={circuit.id}>{circuit.name}</option>)}
    </select></label>
    <label>Status <select value={claim.status} onChange={event => update({ ...claim, status: event.target.value as CircuitClaim["status"] })}>
      {CIRCUIT_STATUSES.map(status => <option key={status}>{status}</option>)}
    </select></label>
    <label>Relationship <input value={claim.statement} onChange={event => update({ ...claim, statement: event.target.value })} /></label>
    <label>Supporting evidence <textarea value={claim.evidence} onChange={event => update({ ...claim, evidence: event.target.value })} /></label>
    <button type="button" onClick={remove}>Remove relationship note</button>
  </div>;
}

export const compositionModelLabel = (id: string) => NOTE_MODELS[id] ?? id;
