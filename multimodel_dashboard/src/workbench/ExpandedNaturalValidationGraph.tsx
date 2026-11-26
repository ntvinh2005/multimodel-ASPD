import { useEffect, useMemo, useRef, useState } from "react";
import type { ComponentType } from "react";
import { measurementFromGraph } from "./compositionMeasurements";
import ComponentComputationInspector from "./ComponentComputationInspector";
import { edgeThresholdScale, formatEdgeMagnitude } from "./displayUtils";
import { expandedCatalog, loadExpandedEvidence, renderableExpandedGraph, type ExpandedCandidate, type ExpandedEvidence, type FiringExample } from "./expandedNaturalValidation";
import { edgesForAnchor, traceGraph } from "./graphQueries";
import { ResearchNotesEditor } from "./ResearchNotes";
import { notesForGate, notesForSelection } from "./researchNotes";
import type { ModelGraph } from "./UniversalWorkbench";
import { coverageLabel, decodeDictionaryGate, dictionaryLocation, dictionaryPath, type DictionaryBlock, type DictionaryIndex } from "./naturalDictionary";
import { useWorkbenchAsset } from "./TokenCircuitExplorer";

type GraphPanel = ComponentType<Parameters<typeof ModelGraph>[0]>;
const models = ["gemma", "smol", "granite"];
const colors = ["#51d6c3", "#f0b45b", "#8ca8ff"];

export default function ExpandedNaturalValidationGraph({ candidate, Graph }: { candidate: ExpandedCandidate; Graph: GraphPanel }) {
  const heading = useRef<HTMLHeadingElement>(null);
  const [evidence, setEvidence] = useState<ExpandedEvidence>();
  const [error, setError] = useState("");
  useEffect(() => {
    let current = true;
    const frame = requestAnimationFrame(() => heading.current?.scrollIntoView({ block: "nearest" }));
    loadExpandedEvidence(candidate.id).then(
      value => { if (current) setEvidence(value); },
      reason => { if (current) setError(String(reason)); },
    );
    return () => {
      current = false;
      cancelAnimationFrame(frame);
    };
  }, [candidate.id]);
  return <section className="uw-expanded-graph" aria-labelledby="uw-expanded-graph-title">
    <header className="uw-context uw-candidate-context">
      <h2 id="uw-expanded-graph-title" ref={heading}>{candidate.id} · {candidate.title}</h2>
      <span>{candidate.shared}/{candidate.opportunities} shared validation anchors</span>
      <small className="rn-status hypothesis-proposed">Hypothesis proposed</small>
    </header>
    {error ? <p role="alert">The measured graph could not load. {error}</p>
      : evidence ? <NaturalWitness gates={candidate.gates} reference={candidate.id}
        artifact={`${expandedCatalog.artifact} · ${expandedCatalog.evidenceSha256}`} evidence={evidence} Graph={Graph} />
      : <p role="status">Loading measured component graphs…</p>}
  </section>;
}

export function DictionaryNaturalValidationGraph({ gate, index, Graph, onExploreGate }: {
  gate: string; index: DictionaryIndex; Graph: GraphPanel; onExploreGate: (id: string) => void;
}) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    const frame = requestAnimationFrame(() => heading.current?.scrollIntoView({ block: "nearest" }));
    return () => cancelAnimationFrame(frame);
  }, []);
  const { path, sha256, coverage } = dictionaryLocation(index, gate);
  const asset = useWorkbenchAsset<DictionaryBlock>(`${dictionaryPath}/${path}`, sha256);
  const result = useMemo(() => {
    if (!asset?.data) return {};
    try {
      return { data: decodeDictionaryGate(index, gate, asset.data) };
    } catch (error) {
      return { error: String(error) };
    }
  }, [asset?.data, index, gate]);
  const data = result.data;
  const error = asset?.error ?? result.error;
  const artifact = `${index.provenance.source}/profiles/parents_per_anchor.safetensors · SHA-256 ${index.provenance.profileHashes["parents_per_anchor.safetensors"]}`;

  return <section className="uw-expanded-graph" aria-labelledby="uw-expanded-graph-title">
    <header className="uw-context uw-candidate-context">
      <h2 id="uw-expanded-graph-title" ref={heading}>{gate} · Shared gate</h2>
      <span>{coverageLabel[coverage]}</span>
      {data && <small>{data.fireCount.toLocaleString()} natural validation firings</small>}
    </header>
    {error ? <p role="alert">The shared gate could not load. {error}</p>
      : !data ? <p role="status">Loading the saved gate neighborhood…</p>
      : data.snapshot ? <NaturalWitness gates={[gate]} reference={gate} artifact={artifact} Graph={Graph}
        evidence={{ snapshots: [data.snapshot], examples: { [gate]: data.examples } }}
        neighborhoodLimit={index.outgoingLimitPerModel} onExploreGate={onExploreGate} />
      : <div className="uw-main">
        <section className="uw-graph-area"><p className="uw-edge-state">The Natural Validation artifact retained no dependency edge incident to {gate}. Held-out examples and gate notes are available in the inspector.</p></section>
        <aside className="uw-inspector">
          <Examples gates={[gate]} examples={{ [gate]: data.examples }} />
          <ResearchNotesEditor targets={notesForGate(index.checkpoint, gate, models)} evidenceReference={`${gate} · Natural Validation`} />
        </aside>
      </div>}
  </section>;
}

type WitnessEvidence = Pick<ExpandedEvidence, "snapshots" | "examples"> &
  Partial<Pick<ExpandedEvidence, "hypothesis" | "alternative" | "test" | "documents">>;

export function NaturalWitness({ gates, reference, artifact, evidence, Graph, neighborhoodLimit, onExploreGate }: {
  gates: string[]; reference: string; artifact: string; evidence: WitnessEvidence; Graph: GraphPanel;
  neighborhoodLimit?: number; onExploreGate?: (id: string) => void;
}) {
  const [snapshotIndex, setSnapshotIndex] = useState(0);
  const snapshot = evidence.snapshots[snapshotIndex];
  const [selection, select] = useState<{ anchor: number; id: string } | null>(null);
  const [tab, setTab] = useState<"details" | "examples">("details");
  const [exampleGate, setExampleGate] = useState(gates[0]);
  const [threshold, setThreshold] = useState(0);
  const { components, edges } = useMemo(() => renderableExpandedGraph(snapshot), [snapshot]);
  const thresholdScale = useMemo(() => edgeThresholdScale(edges), [edges]);
  const visibleEdges = useMemo(() => edgesForAnchor(edges, snapshot.anchor, new Set(components.map(({ id }) => id)), threshold),
    [components, edges, snapshot.anchor, threshold]);
  const displayed = new Set(visibleEdges);
  const checkpoint = expandedCatalog.checkpoint;
  const measurement = useMemo(() => measurementFromGraph({
    checkpoint, dataset: "natural-validation", anchor: snapshot.anchor, example: null, pool: null,
    artifact,
  }, edges), [checkpoint, snapshot.anchor, edges, artifact]);
  const selected = selection?.anchor === snapshotIndex ? selection.id : null;
  const node = components.find(({ id }) => id === selected);
  const edge = visibleEdges.find(({ id }) => id === selected);
  const raw = edge && snapshot.edges[edges.indexOf(edge)];
  const trace = useMemo(() => node ? traceGraph(visibleEdges, node.id, "both", 3) : null, [visibleEdges, node]);
  const gateColors = Object.fromEntries(gates.map((id, index) => [id, colors[index]]));
  const focus = (id: string) => {
    select({ anchor: snapshotIndex, id });
    setTab("details");
  };

  return <div className="uw-main">
    <section className="uw-graph-area">
      <div className="uw-graph-toolbar">
        {evidence.snapshots.length > 1 && <label>Witness <select value={snapshotIndex} onChange={event => {
          setSnapshotIndex(Number(event.target.value));
          select(null);
          setThreshold(0);
        }}>
          {evidence.snapshots.map((item, index) => <option key={item.kind} value={index}>{item.kind}</option>)}
        </select></label>}
        <button type="button" onClick={() => { select(null); setThreshold(0); }}>Restore graph</button>
        <span>{components.length} components · {visibleEdges.length}/{edges.length} measured edges shown</span>
      </div>
      <div className="uw-edge-filter">
        <label className="uw-threshold-label">
          <span>Minimum normalized edge weight <i>w</i></span>
          <output>{formatEdgeMagnitude(threshold)}</output>
          <input type="range" min="0" max={thresholdScale.maximum || thresholdScale.step}
            step={Math.min(thresholdScale.maximum || thresholdScale.step, thresholdScale.step)}
            aria-label="Minimum normalized edge weight" aria-valuetext={formatEdgeMagnitude(threshold)}
            value={threshold} disabled={thresholdScale.maximum === 0} onChange={event => {
              const value = Number(event.target.value);
              if (edge?.normalizedMagnitude != null && edge.normalizedMagnitude < value) select(null);
              setThreshold(value);
            }} />
        </label>
        <small>Range 0–{formatEdgeMagnitude(thresholdScale.maximum)} at this anchor.</small>
      </div>
      <div className="uw-graph-key"><span><i className="gate-fill" />{onExploreGate ? `Teal = ${gates[0]}` : "Fill color = shared gate"}</span><span><i className="selection-ring" />Ring = selected component</span></div>
      {neighborhoodLimit !== undefined && <div className="uw-graph-key">One saved anchor · retained parent edges · up to {neighborhoodLimit} outgoing edges per model</div>}
      <div className="uw-panels">
        {models.map(modelId => <Graph key={modelId} modelId={modelId}
          components={components.filter(component => component.modelId === modelId)}
          edges={visibleEdges.filter(item => item.modelId === modelId)}
          spotlight={onExploreGate ? gates[0] : null} selectedComponent={node?.id ?? null} selectedEdge={edge?.id ?? null} trace={trace}
          direction="both" widthScale="sqrt" colors={gateColors} annotationCheckpoint={checkpoint}
          compositionMeasurement={measurement} onComponent={focus} onEdge={focus} />)}
      </div>
    </section>
    <aside className="uw-inspector">
      <div className="uw-tabs">
        <button type="button" className={tab === "details" ? "active" : ""} onClick={() => setTab("details")}>Details</button>
        <button type="button" className={tab === "examples" ? "active" : ""} onClick={() => setTab("examples")}>Examples</button>
      </div>
      {tab === "details" ? <div className="uw-detail uw-candidate-detail">
        <h2>{edge ? "Dependency edge" : node ? "Rank-one component" : onExploreGate ? "Gate neighborhood" : "Candidate evidence"}</h2>
        <dl>
          <dt>Exact validation anchor</dt><dd>{snapshot.anchor}</dd>
          {node && <><dt>Component</dt><dd>{node.id}</dd></>}
          {raw && <><dt>Source</dt><dd>{raw.source}</dd><dt>Target</dt><dd>{raw.target}</dd>
            <dt>Signed E</dt><dd>{raw.E.toPrecision(5)}</dd><dt>Normalized w</dt><dd>{raw.w.toPrecision(5)}</dd>
            <dt>Rank</dt><dd>{raw.rank0 + 1}</dd><dt>Artifact location</dt>
            <dd>{"profileRow" in raw ? `Parent profile · row ${raw.profileRow}` : `${raw.shard} · record ${raw.record} · offset ${raw.flat}`}</dd></>}
        </dl>
        {node && onExploreGate && node.gateId !== gates[0] && <button type="button" className="uw-text-button"
          onClick={() => onExploreGate(node.gateId)}>Explore {node.gateId} at its saved anchor</button>}
        {node && <ComponentComputationInspector id={node.id} checkpoint={checkpoint} anchor={snapshot.anchor}
          measurement={measurement} candidateId={reference} />}
        {node && <ResearchNotesEditor key={node.id} targets={notesForSelection(checkpoint, node.gateId, node.modelId, node)}
          evidenceReference={`${reference} · Natural Validation · anchor ${snapshot.anchor}`} />}
        {!node && !edge && onExploreGate && <ResearchNotesEditor targets={notesForGate(checkpoint, gates[0], models)}
          evidenceReference={`${reference} · Natural Validation · anchor ${snapshot.anchor}`} />}
        <section className="uw-connections" aria-label="Displayed component edges">
          <header><strong>Displayed component edges</strong><span>{visibleEdges.length}/{edges.length}</span></header>
          {snapshot.edges.map((row, index) => displayed.has(edges[index]) && <button key={edges[index].id} type="button" aria-pressed={edges[index].id === selected}
            onClick={() => focus(edges[index].id)}>
            <b>{index + 1}</b><span>{row.source} → {row.target}</span>
            <small className={row.E < 0 ? "neg" : "pos"}>E {row.E.toPrecision(3)}</small><strong>w {row.w.toPrecision(3)}</strong>
          </button>)}
        </section>
        {evidence.hypothesis && <p><b>Hypothesis</b> {evidence.hypothesis}</p>}
        {evidence.alternative && <p><b>Alternative</b> {evidence.alternative}</p>}
        {evidence.test && <p><b>Proposed test</b> {evidence.test}</p>}
        {onExploreGate && <details><summary>Artifact provenance</summary><p>{artifact}</p></details>}
      </div> : <Examples gates={gates} gate={exampleGate} onGateChange={setExampleGate} examples={evidence.examples} documents={evidence.documents} />}
    </aside>
  </div>;
}

function Examples({ gates, gate = gates[0], onGateChange, examples, documents }: {
  gates: string[]; gate?: string; onGateChange?: (id: string) => void;
  examples: WitnessEvidence["examples"]; documents?: number;
}) {
  const rows = examples[gate] ?? [];
  return <div className="uw-examples uw-candidate-examples">
    <h2>Held-out firing examples</h2>
    <p>{documents !== undefined && `${documents} co-firing documents. `}Test examples use separate anchors from the graph.</p>
    {gates.length > 1 && <label>Gate <select value={gate} onChange={event => onGateChange?.(event.target.value)}>
      {gates.map(id => <option key={id}>{id}</option>)}
    </select></label>}
    {!rows.length && <p>No saved held-out example for {gate}.</p>}
    {rows.map((example) => <article className="uw-example-card" key={example.anchor}>
      <blockquote>{example.before}<mark>{example.token.replaceAll("▁", " ").replaceAll("\n", "↵")}</mark>{example.after}</blockquote>
      <dl><dt>Source</dt><dd>{example.source}</dd><dt>Activation</dt><dd>{example.activation.toPrecision(5)}</dd>
        <dt>Test anchor</dt><dd>{example.anchor}</dd></dl>
    </article>)}
  </div>;
}
