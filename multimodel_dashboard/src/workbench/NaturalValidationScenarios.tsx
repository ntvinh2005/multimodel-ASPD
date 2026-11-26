import { useMemo, useState } from "react";
import bottomUpNotes from "./bottomUpResearchNotes.json";
import bottomUpNotesBatch2 from "./bottomUpResearchNotesBatch2.json";
import { expandedCatalog, searchExpandedCandidates } from "./expandedNaturalValidation";
import SharedGateSearch from "./SharedGateSearch";
import { coverageLabel, dictionaryGates, searchDictionaryGates, type DictionaryIndex } from "./naturalDictionary";
import type { DatasetManifest } from "./types";

const pageSize = 24;
const discoveryTitles = new Map(
  [...bottomUpNotes.notes, ...bottomUpNotesBatch2.notes].map(({ target, fields }) => [target.gateId, fields.label]),
);

export default function NaturalValidationScenarios({ manifest, activeAnchorId, onSelect, query = "", onQueryChange,
  matchingGates = [], onOpenGate, activeCandidateId = null, onOpenCandidate, dictionary, dictionaryError,
  activeDictionaryGate = null, onOpenDictionaryGate }: {
  manifest: DatasetManifest;
  activeAnchorId: string | null;
  onSelect: (anchorId: string) => void;
  query?: string;
  onQueryChange?: (query: string) => void;
  matchingGates?: string[];
  onOpenGate?: (gateId: string) => void;
  activeCandidateId?: string | null;
  onOpenCandidate?: (id: string) => void;
  dictionary?: DictionaryIndex;
  dictionaryError?: string;
  activeDictionaryGate?: string | null;
  onOpenDictionaryGate?: (id: string) => void;
}) {
  const [browsing, setBrowsing] = useState(false);
  const [currentPage, setCurrentPage] = useState({ query: "", number: 0 });
  const matches = useMemo(() => searchExpandedCandidates(query), [query]);
  const gates = useMemo(() => dictionary ? dictionaryGates(dictionary) : [], [dictionary]);
  const dictionarySearch = useMemo(() => searchDictionaryGates(gates, query), [gates, query]);
  const measuredGates = useMemo(() => gates.filter(gate => gate.coverage !== "n"), [gates]);
  const scenarios = manifest.export?.selectedScenarios ?? [];
  const discovery = scenarios.filter(({ discoveryGateIds }) => discoveryGateIds);
  const baseline = scenarios.filter(({ discoveryGateIds }) => !discoveryGateIds);
  const searching = Boolean(query.trim());
  const canonical = query.trim().toUpperCase().replace(/^(MID|PRE)[\s:/_-]*/, "$1:");
  const exactGate = matchingGates.find(id => id === canonical || id.split(":")[1] === canonical);
  const showDictionary = dictionary && (browsing || searching);
  const dictionaryMatches = showDictionary ? dictionarySearch.matches : [];
  const pageCount = Math.ceil(dictionaryMatches.length / pageSize);
  const page = currentPage.query === query ? Math.min(currentPage.number, Math.max(0, pageCount - 1)) : 0;
  const visibleGates = dictionaryMatches.slice(page * pageSize, (page + 1) * pageSize);
  const starterMatches = dictionary ? [] : matchingGates;

  return <section className="uw-scenario-browser" aria-labelledby="uw-scenario-title">
    <header>
      <h2 id="uw-scenario-title">Browse measured graphs</h2>
      <span>{discovery.length + (onOpenCandidate ? expandedCatalog.candidates.length : 0)} discovery hypotheses · unvalidated</span>
    </header>
    {onQueryChange && onOpenCandidate && <div className="uw-expanded-search">
      <SharedGateSearch value={query} label="Search shared gates and candidates" placeholder="MID:7027, PRE:1341, NVX-025, or candidate title"
        onChange={value => { onQueryChange(value); setCurrentPage({ query: value, number: 0 }); }} onSelectFirst={() => {
          if (dictionarySearch.exact[0]) onOpenDictionaryGate?.(dictionarySearch.exact[0].id);
          else if (exactGate && !dictionary) onOpenGate?.(exactGate);
          else if (matches[0]) onOpenCandidate(matches[0].id);
          else if (visibleGates[0]) onOpenDictionaryGate?.(visibleGates[0].id);
          else if (matchingGates[0]) onOpenGate?.(matchingGates[0]);
        }} />
      <label>Choose a graph
        <select value={activeDictionaryGate ? `gate:${activeDictionaryGate}` : activeCandidateId ?? (activeAnchorId ? `anchor:${activeAnchorId}` : "")} onChange={event => {
          const id = event.target.value;
          if (id.startsWith("anchor:")) onSelect(id.slice("anchor:".length));
          else if (id.startsWith("gate:")) onOpenDictionaryGate?.(id.slice("gate:".length));
          else onOpenCandidate(id);
        }}>
          <option value="" disabled>Select a measured graph</option>
          {activeDictionaryGate && <option value={`gate:${activeDictionaryGate}`}>{activeDictionaryGate} · Shared gate</option>}
          <optgroup label="Measured anchors">
            {scenarios.map(({ anchorId, gateId }) => <option key={anchorId} value={`anchor:${anchorId}`}>
              {discoveryTitles.get(gateId) ?? gateId}
            </option>)}
          </optgroup>
          <optgroup label={`Candidate graphs (${expandedCatalog.candidates.length})`}>
            {expandedCatalog.candidates.map(({ id, title }) => <option key={id} value={id}>{id} · {title}</option>)}
          </optgroup>
        </select>
      </label>
    </div>}
    {onOpenDictionaryGate && <div className="uw-scenario-baseline uw-dictionary-controls">
      {dictionary ? <>
        <span>{gates.length.toLocaleString()} dictionary gates · {measuredGates.length.toLocaleString()} with measured edges</span>
        <button type="button" aria-pressed={browsing} onClick={() => {
          setBrowsing(value => !value || searching);
          onQueryChange?.("");
          setCurrentPage({ query: "", number: 0 });
        }}>{browsing && !searching ? "Hide gate list" : "Browse all gates"}</button>
        <button type="button" disabled={!measuredGates.length} onClick={() => {
          const gate = measuredGates[Math.floor(Math.random() * measuredGates.length)];
          if (gate) onOpenDictionaryGate(gate.id);
        }}>Random measured gate</button>
      </> : <span role="status">{dictionaryError
        ? `Full gate dictionary unavailable. Starter gates and candidate graphs are available. ${dictionaryError}`
        : "Loading the gate dictionary…"}</span>}
    </div>}
    {visibleGates.length > 0 && <>
      <div className="uw-scenario-list uw-expanded-results" aria-label="Shared gate dictionary results">
        {visibleGates.map(({ id, coverage }) => <button key={id} type="button" aria-pressed={activeDictionaryGate === id}
          onClick={() => onOpenDictionaryGate?.(id)}><strong>{id}</strong><span>{coverageLabel[coverage]}</span></button>)}
      </div>
      {pageCount > 1 && <div className="uw-scenario-baseline uw-dictionary-controls" aria-label="Gate result pages">
        <span aria-live="polite">{(page * pageSize + 1).toLocaleString()}–{(page * pageSize + visibleGates.length).toLocaleString()} of {dictionaryMatches.length.toLocaleString()} shared gates</span>
        <button type="button" disabled={page === 0} onClick={() => setCurrentPage({ query, number: page - 1 })}>Previous</button>
        <button type="button" disabled={page + 1 >= pageCount} onClick={() => setCurrentPage({ query, number: page + 1 })}>Next</button>
      </div>}
    </>}
    {searching ? <>
      {(matches.length > 0 || starterMatches.length > 0) && <div className="uw-scenario-list uw-expanded-results" aria-label="Measured graph search results">
        {matches.map(({ id, title, gates }) => <button key={id} type="button" aria-pressed={activeCandidateId === id}
          onClick={() => onOpenCandidate?.(id)}><strong>{id} · {title}</strong><span>{gates.join(" · ")}</span></button>)}
        {starterMatches.map(id => <button type="button" key={id} onClick={() => onOpenGate?.(id)}>
          <strong>{discoveryTitles.get(id) ?? id}</strong><span>{id} · measured subset</span>
        </button>)}
      </div>}
      {!matches.length && !starterMatches.length && !dictionaryMatches.length && <p role="status">No gate or candidate matches this search.</p>}
    </> : browsing && dictionary ? null : <><div className="uw-scenario-list">
      {discovery.map(({ anchorId, gateId, discoveryGateIds }) => (
        <button key={anchorId} type="button" aria-pressed={anchorId === activeAnchorId} onClick={() => onSelect(anchorId)}>
          <strong>{discoveryTitles.get(gateId) ?? gateId}</strong>
          <span>{discoveryGateIds?.join(" · ")}</span>
        </button>
      ))}
    </div>
    {baseline.length > 0 && <div className="uw-scenario-baseline" aria-label="Other measured anchors">
      <span>Other measured anchors</span>
      {baseline.map(({ anchorId, gateId }) => (
        <button key={anchorId} type="button" aria-pressed={anchorId === activeAnchorId} onClick={() => onSelect(anchorId)}>{gateId}</button>
      ))}
    </div>}</>}
  </section>;
}
