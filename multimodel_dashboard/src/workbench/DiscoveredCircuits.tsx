import { useMemo, useState } from "react";
import { CircuitCaptureButton } from "./CircuitLibrary";
import CircuitCompositionExplorer from "./CircuitCompositionExplorer";
import ComponentComputationInspector from "./ComponentComputationInspector";
import type { GraphRenderer } from "./InductionQueryGraph";
import { ResearchNotesEditor } from "./ResearchNotes";
import { useWorkbenchAsset } from "./TokenCircuitExplorer";
import { componentIdentity, validateMeasurement, type Measurement } from "./circuitComposition";
import { conditionLabel, datasetLabel, discoveryDirections, discoveryWording, familyLabel, splitLabel } from "./discoveryWording";
import { edgeThresholdScale, formatEdgeMagnitude } from "./displayUtils";
import { traceGraph } from "./graphQueries";
import { MODEL_NAMES, type EvalModel } from "./negationWorkbenchData";
import { notesForSelection } from "./researchNotes";
import type { DependencyEdge } from "./types";
import "./discoveredCircuits.css";

type Witness = {
  model: EvalModel; anchor: string; dataset: string; split: string; example: string | null; pool: string | null;
  artifact: string; text: string | null; boundary: number | null; condition?: string | null; group?: string; family?: string | null;
  edges: Array<{ source: string; target: string; E: number; w: number }>;
  components: Record<string, { coefficient: number | null; normalized: number | null }>;
  activations: Record<string, number>;
};

export type DiscoveryCandidate = {
  id: string; directions: string[]; model: EvalModel; dataset: string; anchorCount: number; paretoFront: number;
  witnessSignStability?: number | null;
  status: string; componentIds: string[]; gateIds: string[];
  structure: { nodes: number; edges: number; depth: number; branching: number; convergence: number; diamonds: number };
  crossModel: { models: EvalModel[]; topologyOnlyModels?: EvalModel[]; sharedExactAnchors: number; peers: Array<{id: string; model: EvalModel; sharedExactAnchors: number}> };
  recurrence: Array<{dataset: string; split: string; anchors: number; groups: number}>;
  interpretation: { reviewTitle?: string; hypothesis: string; alternative: string; counterevidence: string; proposedTest: string;
    componentRoles?: Array<{componentId: string; structuralRole: string; coefficient?: number | null}>;
    counterexampleContexts?: Array<{anchor: string; text: string; condition: string; split: string}>;
    independentActivationExamples: Array<{anchor: string; gate: string; before: string; token: string; after: string}> };
  witnesses: Witness[];
  matchedNull: {trials: number; hits: number | null};
  conditionalGraphNull?: {trials: number; hits: number | null; acceptedSwaps: number};
  measuredContextSelectivity?: {
    denominators: Array<{dataset: string; family: string; split: string; condition: string;
      roleAnchors: number; allGatesActive: number; requiredTargetsMeasured: number; exactCircuitPresent: number}>;
    counterexamples: Array<{anchor: string; text: string; condition: string; split: string}>;
  };
  prefixAudit?: {distinctHeldoutPrefixes: number; unseenHeldoutPrefixes: number; heldoutPrefixesSeenInDiscovery: number};
};

type Catalog = { schemaVersion: 1; provenance: { checkpoint: string }; candidates: DiscoveryCandidate[] };

export function matchesDiscovery(candidate: DiscoveryCandidate, filters: {
  query: string; direction: string; nodes: number; recurrence: number; models: number;
}) {
  const term = filters.query.trim().toLowerCase();
  if ((filters.direction && !candidate.directions.includes(filters.direction))
    || candidate.structure.nodes < filters.nodes || candidate.anchorCount < filters.recurrence
    || candidate.crossModel.models.length < filters.models) return false;
  if (!term) return true;
  const copy = discoveryWording(candidate);
  return [candidate.id, datasetLabel(candidate.dataset), copy.title, copy.hypothesis,
    ...copy.sources, ...candidate.gateIds, ...candidate.componentIds].some(value => value?.toLowerCase().includes(term));
}

function WitnessGraph({ witness, candidate, checkpoint, Graph }: {
  witness: Witness; candidate: DiscoveryCandidate; checkpoint: string; Graph: GraphRenderer;
}) {
  const [selected, setSelected] = useState<string>();
  const [selectedEdge, selectEdge] = useState<string>();
  const [threshold, setThreshold] = useState(0);
  const graph = useMemo(() => {
    const components = candidate.componentIds.map(id => ({ ...componentIdentity(id), beta: null, rho: null, representative: false }));
    const exact = new Set(components.map(item => item.id));
    const measurement: Measurement = {
      scope: { checkpoint, dataset: witness.dataset, anchor: witness.anchor, pool: witness.pool,
        example: witness.example, artifact: witness.artifact },
      complete: false, targets: [...exact],
      coefficients: Object.fromEntries(Object.entries(witness.components)
        .filter(([, value]) => value.coefficient !== null).map(([id, value]) => [id, value.coefficient! ])),
      edges: witness.edges.map(edge => ({ source: edge.source, target: edge.target, signed: edge.E, weight: edge.w })),
    };
    validateMeasurement(measurement, checkpoint);
    if (components.some(component => component.modelId !== witness.model) || witness.edges.some(({ source, target }) => !exact.has(source) || !exact.has(target))) {
      throw new Error("Candidate components and witness scope differ");
    }
    const edges: DependencyEdge[] = witness.edges.map(({ source, target, E, w }) => ({
      id: `${witness.anchor}:${source}:${target}`, anchorId: witness.anchor, modelId: witness.model, source, target,
      signedEffect: E, normalizedMagnitude: w, upstreamCoefficient: witness.components[source]?.coefficient ?? null,
      gradient: null, normalization: "Stored parent weight", evidence: "measured-taylor",
    }));
    return { components, edges, measurement };
  }, [witness, candidate, checkpoint]);
  const range = edgeThresholdScale(graph.edges);
  const visible = graph.edges.filter(edge => (edge.normalizedMagnitude ?? 0) >= threshold);
  const trace = selected ? traceGraph(visible, selected, "both", 3) : null;
  const node = graph.components.find(component => component.id === selected);
  const edge = visible.find(item => item.id === selectedEdge);
  const positions = witness.text === null ? [] : Array.from(witness.text);
  const boundary = witness.boundary ?? 0;
  const colors = Object.fromEntries(candidate.gateIds.map((gate, i) => [gate, `hsl(${(i * 137.5) % 360} 60% 67%)`]));
  return <div className="uw-main discovered-workspace">
    <section className="uw-graph-area">
      <p className="uw-graph-key">{[MODEL_NAMES[witness.model], datasetLabel(witness.dataset), splitLabel(witness.split), familyLabel(witness.family),
        conditionLabel(witness.dataset, witness.condition, witness.family)].filter(Boolean).join(" · ")}</p>
      {positions.length > 0 && <div className="uw-context discovered-context">
        <p className="ig-sequence td-sequence">{positions.slice(0, Math.max(0, boundary - 1)).join("")}
          <mark title="The aligned token ends after this character">{positions[boundary - 1]}</mark>{positions.slice(boundary).join("")}</p>
      </div>}
      <div className="uw-edge-filter">
        <label className="uw-threshold-label">
          <span>Minimum edge weight w</span>
          <output>{formatEdgeMagnitude(threshold)}</output>
          <input type="range" value={threshold} min={0} max={range.maximum || range.step}
            step={range.step} aria-valuetext={formatEdgeMagnitude(threshold)}
            onChange={event => setThreshold(Number(event.target.value))} />
        </label>
      </div>
      <Graph modelId={witness.model} components={graph.components} edges={visible} spotlight={null}
        selectedComponent={selected ?? null} selectedEdge={edge?.id ?? null} trace={trace} direction="both"
        widthScale="sqrt" colors={colors} annotationCheckpoint={checkpoint} compositionMeasurement={graph.measurement}
        onComponent={id => { setSelected(id); selectEdge(undefined); }}
        onEdge={id => { selectEdge(id); setSelected(undefined); }} />
      <details className="uw-provenance-drawer"><summary>Measurement source</summary><code>{witness.anchor}</code>
        <p>Pool {witness.pool ?? "Unavailable"} · archive {witness.artifact}</p></details>
      <CircuitCompositionExplorer checkpoint={checkpoint} measurement={graph.measurement}
        anchor={witness.anchor} model={witness.model} />
    </section>
    <aside className="uw-inspector uw-detail">
      {edge && <p><code>{edge.source} → {edge.target}</code><br />
        Signed Taylor effect E {edge.signedEffect?.toPrecision(5)} · normalized weight w {edge.normalizedMagnitude?.toPrecision(5)}</p>}
      {node && <><CircuitCaptureButton id={node.id} checkpoint={checkpoint} measurement={graph.measurement} />
        <ComponentComputationInspector id={node.id} checkpoint={checkpoint} anchor={witness.anchor}
          measurement={graph.measurement} token={{ dataset: witness.dataset, pool: witness.pool ?? undefined,
            activation: witness.activations[node.gateId], support: witness.activations[node.gateId] === undefined ? undefined : true,
            normalized: witness.components[node.id]?.normalized ?? undefined }} />
        <ResearchNotesEditor key={node.id} targets={notesForSelection(checkpoint, node.gateId, node.modelId, node)}
          evidenceReference={`${candidate.id} · ${witness.anchor}`} /></>}
      {!node && !edge && <p>Select a component or edge to inspect it.</p>}
    </aside>
  </div>;
}

export default function DiscoveredCircuits({ checkpoint, Graph }: {checkpoint: string; Graph: GraphRenderer}) {
  const [query, setQuery] = useState("");
  const [direction, setDirection] = useState("");
  const [nodes, setNodes] = useState(5);
  const [recurrence, setRecurrence] = useState(1);
  const [models, setModels] = useState(1);
  const [id, select] = useState("");
  const [witness, selectWitness] = useState(0);
  const asset = useWorkbenchAsset<Catalog>("circuit-discovery/catalog.json");
  const catalog = asset?.data?.schemaVersion === 1 && asset.data.provenance.checkpoint === checkpoint ? asset.data : undefined;
  const matching = catalog?.candidates.filter(row => matchesDiscovery(row, {query, direction, nodes, recurrence, models})) ?? [];
  const selected = matching.find(row => row.id === id) ?? matching[0];
  const copy = selected && discoveryWording(selected);
  const anchor = selected?.witnesses[witness] ?? selected?.witnesses[0];
  return <main className="td-section discovered-circuits" aria-label="Circuit discovery candidates">
    <div className="discovered-content">
      {asset?.error && <p role="status">Circuit Discovery data could not be loaded.</p>}
      {!asset && <p role="status">Loading measured candidates…</p>}
      {asset?.data && !catalog && <p role="status">This catalog is incompatible with the Workbench or its checkpoint.</p>}
      {catalog && <>
        <div className="uw-expanded-search discovered-filters">
          <label>Search <input className="uw-form-control" type="search" value={query} placeholder="Context, gate, hypothesis or circuit ID"
            onChange={event => { setQuery(event.target.value); selectWitness(0); }} /></label>
          <label>Discovery method <select className="uw-form-control" value={direction} onChange={event => { setDirection(event.target.value); selectWitness(0); }}>
            <option value="">All methods</option>{Object.entries(discoveryDirections).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <label>Min. components <input className="uw-form-control" type="number" min={5} value={nodes} onChange={event => setNodes(Number(event.target.value))} /></label>
          <label>Min. token positions <input className="uw-form-control" type="number" min={1} value={recurrence} onChange={event => setRecurrence(Number(event.target.value))} /></label>
          <label>Min. models <input className="uw-form-control" type="number" min={1} max={3} value={models} onChange={event => setModels(Number(event.target.value))} /></label>
        </div>
        <label className="discovered-picker">Candidate ({matching.length}/{catalog.candidates.length}) <select className="uw-form-control" value={selected?.id ?? ""} onChange={event => {select(event.target.value); selectWitness(0);}}>
          {!matching.length && <option value="">No matching candidates</option>}
          {matching.map(row => <option key={row.id} value={row.id}>{discoveryWording(row).title} · {row.structure.nodes} components · {MODEL_NAMES[row.model]} · {datasetLabel(row.dataset)}</option>)}
        </select></label>
        {selected && copy && <div className="discovered-selection">
          <header className="discovered-summary">
            <h2>{copy.title}</h2>
            <p><strong>Observed</strong> · {selected.structure.nodes} components and {selected.structure.edges} edges
              {" · "}{selected.anchorCount} measured positions · {MODEL_NAMES[selected.model]}</p>
            {copy.hypothesis ? <p><strong>Hypothesis</strong> · {copy.hypothesis}</p>
              : <p>Computational role unknown.</p>}
            <small>{copy.hypothesis ? "Unvalidated hypothesis · " : ""}{copy.sources.length ? copy.sources.join(" · ") : datasetLabel(selected.dataset)}</small>
          </header>
          <label className="discovered-picker">Measured example <select className="uw-form-control" value={anchor ? selected.witnesses.indexOf(anchor) : ""} onChange={event => selectWitness(Number(event.target.value))}>
            {selected.witnesses.map((row, i) => <option key={row.anchor + "/" + row.model} value={i}>{
              [splitLabel(row.split), conditionLabel(row.dataset, row.condition, row.family), row.anchor.slice(0, 12)].filter(Boolean).join(" · ")
            }</option>)}
          </select></label>
          {anchor && <WitnessGraph key={selected.id + "/" + anchor.anchor + "/" + anchor.model}
            witness={anchor} candidate={selected} checkpoint={checkpoint} Graph={Graph} />}
          <details className="uw-provenance-drawer"><summary>Graph shape and recurrence</summary>
            <p>Found by {selected.directions.map(direction => discoveryDirections[direction as keyof typeof discoveryDirections] ?? direction).join(" / ")} · ranking tier {selected.paretoFront}</p>
            <p>Longest path {selected.structure.depth} edges · {selected.structure.convergence} points where paths meet
              {" · "}{selected.structure.diamonds} paths that split and rejoin</p>
            <p>{copy.heldout}</p>
            {selected.prefixAudit && <p>{selected.prefixAudit.unseenHeldoutPrefixes} new held-out text prefixes
              {" · "}{selected.prefixAudit.heldoutPrefixesSeenInDiscovery} held-out prefixes also seen in discovery</p>}
            {selected.recurrence.map(row => <p key={row.dataset + "/" + row.split}>
              {datasetLabel(row.dataset)} · {splitLabel(row.split)} · {row.anchors} token positions · {row.groups} example groups</p>)}
          </details>
          {selected.interpretation.componentRoles?.length ? <details><summary>Where components sit in the graph</summary>
            <ul>{selected.interpretation.componentRoles.map(role => <li key={role.componentId}>
              <code>{role.componentId}</code> · {role.structuralRole === "relay" ? "Has incoming and outgoing edges"
                : role.structuralRole === "input arm" ? "Has outgoing edges only" : "Has incoming edges only"}
            </li>)}</ul>
          </details> : null}
          <details><summary>Other explanations and checks</summary>
            {copy.alternative && <p><strong>Another explanation</strong> · {copy.alternative}</p>}
            {copy.signs && <p>{copy.signs}</p>}
            {copy.proposedTest && <p><strong>Next test</strong> · {copy.proposedTest}</p>}
            {(selected.measuredContextSelectivity?.counterexamples ?? selected.interpretation.counterexampleContexts)?.map(row => <blockquote key={row.anchor}>
              {row.text} <small>{splitLabel(row.split)} · {conditionLabel(selected.dataset, row.condition)} · {row.anchor.slice(0, 12)}</small>
            </blockquote>)}
            {!!selected.interpretation.independentActivationExamples.length && <details><summary>Other examples where individual gates fire</summary>
              <p>Each example shows one gate. Open a measured example above to see edges.</p>
              {selected.interpretation.independentActivationExamples.map(row => <blockquote key={row.gate + row.anchor}>
                {row.before}<mark>{row.token.replaceAll("▁", " ")}</mark>{row.after} <small>{row.gate}</small>
              </blockquote>)}
            </details>}
            {!!selected.conditionalGraphNull?.trials && <p>{selected.conditionalGraphNull.hits}/{selected.conditionalGraphNull.trials}
              {" "}shuffled graphs with the same node degrees reproduced this pattern at the tested position.</p>}
            <p>{selected.matchedNull.trials ? `${selected.matchedNull.hits}/${selected.matchedNull.trials} trials with similar replacement components reproduced this pattern`
              : "No comparison with similar replacement components is available."}</p>
          </details>
          {!!selected.measuredContextSelectivity?.denominators.length && <details><summary>Where the circuit appears</summary>
            <div className="cc-scroll"><table className="cc-table"><thead><tr><th>Source / condition</th>
              <th>Comparable positions</th><th>All gates active</th><th>All targets measured</th><th>Same circuit found</th>
            </tr></thead><tbody>{selected.measuredContextSelectivity.denominators.map(row =>
              <tr key={[row.dataset, row.family, row.split, row.condition].join("/")}>
                <th>{[datasetLabel(row.dataset), familyLabel(row.family), splitLabel(row.split),
                  conditionLabel(row.dataset, row.condition, row.family)].filter(Boolean).join(" · ")}</th>
                <td>{row.roleAnchors}</td><td>{row.allGatesActive}</td>
                <td>{row.requiredTargetsMeasured}</td><td>{row.exactCircuitPresent}</td>
              </tr>)}</tbody></table></div>
          </details>}
          {!!selected.crossModel.topologyOnlyModels?.length && <p>Same graph shape, with gate IDs ignored, in {
            selected.crossModel.topologyOnlyModels.map(model => MODEL_NAMES[model]).join(", ")
          }.</p>}
          {selected.crossModel.peers.length > 0 && <details><summary>Similar circuits in other models</summary>
            {selected.crossModel.peers.map(peer => <button type="button" key={peer.id} onClick={() => {
              select(peer.id); setDirection(""); setNodes(5); setRecurrence(1); setModels(1); setQuery(""); selectWitness(0);
            }}>{MODEL_NAMES[peer.model]} · {peer.id} · {peer.sharedExactAnchors} shared positions</button>)}</details>}

        </div>}
      </>}
    </div>
  </main>;
}
