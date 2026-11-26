import { useId, useMemo, useState, type CSSProperties } from "react";
import { compositionModelLabel, useCircuitLibrary } from "./CircuitLibrary";
import {
  activationTimeline, analyzeRelationships, compareStructure, componentIdentity, mergeMeasurements, projectCompositionGraph, resolveCircuit, scopeKey,
  type CircuitMembership, type CompositionEdge, type Measurement,
} from "./circuitComposition";
import { MODEL_IDS } from "./negationWorkbenchData";
import { tokenGates, type TokenExample } from "./tokenDiscovery";

type Analysis = ReturnType<typeof analyzeRelationships>;
const format = (value: number) => value.toPrecision(4);

export default function CircuitCompositionExplorer({ checkpoint, measurement, example, anchor, onSelectToken, unavailable, model }: {
  checkpoint: string;
  measurement?: Measurement;
  example?: TokenExample;
  anchor?: string;
  onSelectToken?: (anchor: string) => void;
  unavailable?: string;
  model?: string;
}) {
  const [open, setOpen] = useState(false);
  return <details className="td-section cc-explorer" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>Circuit composition</summary>
    {open && <CompositionPanel checkpoint={checkpoint} measurement={measurement} example={example}
      anchor={anchor} onSelectToken={onSelectToken} unavailable={unavailable} activeModel={model} />}
  </details>;
}

function CompositionPanel({ checkpoint, measurement, example, anchor, onSelectToken, unavailable, activeModel }: {
  checkpoint: string; measurement?: Measurement; example?: TokenExample; anchor?: string;
  onSelectToken?: (anchor: string) => void; unavailable?: string; activeModel?: string;
}) {
  const state = useCircuitLibrary()!;
  const [selection, select] = useState<string[] | null>(null);
  const [graphModel, setModel] = useState<string>("smol");
  const model = activeModel ?? graphModel;
  const [threshold, setThreshold] = useState("0");
  const [savedContext, setSavedContext] = useState({ key: "", anchor });
  const snapshot = savedContext.anchor === anchor ? savedContext.key : "";
  const setSnapshot = (key: string) => setSavedContext({ key, anchor });
  const [expanded, setExpanded] = useState<string[]>([]);
  const available = state.library.circuits.filter(circuit => circuit.checkpoint === checkpoint);
  const selected = available.filter(circuit => (selection ?? available.slice(0, 2).map(entry => entry.id)).includes(circuit.id));
  const selectedKey = JSON.stringify(selected.map(circuit => circuit.id));
  const byId = new Map(available.map(circuit => [circuit.id, circuit]));
  const circuits = selected.map(circuit => ({ id: circuit.id, name: circuit.name, ...resolveCircuit(circuit.id, byId) }));
  const parentId = new Map(available.map(circuit => [circuit.id, circuit.name]));
  const graphCorpus = useMemo(() => {
    try {
      const captured = new Map(state.library.circuits.map(circuit => [circuit.id, circuit]));
      const scoped = new Set<string>(JSON.parse(selectedKey) as string[]);
      for (const id of scoped) resolveCircuit(id, captured).children.forEach(child => scoped.add(child));
      const archived = [...scoped].flatMap(id => captured.get(id)?.snapshots ?? []);
      const measured = mergeMeasurements(measurement ? [measurement, ...archived] : archived);
      return { measured, error: "" };
    } catch (error) {
      return { measured: [], error: error instanceof Error ? error.message : String(error) };
    }
  }, [state.library, selectedKey, measurement]);
  const selectedMeasurement = graphCorpus.measured.find(series => scopeKey(series.scope) === (snapshot || (measurement ? scopeKey(measurement.scope) : "")));
  const minimum = Math.max(0, Number(threshold) || 0);
  const results = MODEL_IDS.map(id => ({ id, analysis: analyzeRelationships(circuits, selectedMeasurement, id, minimum) }));
  const analysis = results.find(result => result.id === model)!.analysis;
  const nested = circuits.flatMap((left, i) => circuits.slice(i + 1).map(right => ({
    left: left.id, right: right.id,
    shared: left.children.filter(child => right.children.includes(child)),
    nested: left.children.includes(right.id) || right.children.includes(left.id),
  })).filter(pair => pair.shared.length || pair.nested));
  const claims = selected.flatMap(circuit => circuit.claims.filter(claim => selected.some(other => other.id === claim.target))
    .map(claim => ({ ...claim, source: circuit.id })));
  const liveSelected = !!measurement && !!selectedMeasurement && scopeKey(measurement.scope) === scopeKey(selectedMeasurement.scope);

  return <div className="cc-panel">
    <fieldset className="cc-selection"><legend>Circuits to compare</legend>
      {available.map(circuit => <label key={circuit.id} title={circuit.hypothesis}>
        <input type="checkbox" checked={selected.some(value => value.id === circuit.id)} onChange={event => {
          const ids = selected.map(value => value.id);
          select(event.target.checked ? [...ids, circuit.id] : ids.filter(id => id !== circuit.id));
        }} /> {circuit.name}
      </label>)}
      {!available.length && <p>Save a circuit from the graph or import circuit definitions through the Circuits menu.</p>}
    </fieldset>
    <div className="ig-toolbar">
      <label>Composition context <select value={snapshot} onChange={event => setSnapshot(event.target.value)}>
        <option value="">Current context and token</option>
        {graphCorpus.measured.filter(item => !measurement || scopeKey(item.scope) !== scopeKey(measurement.scope)).map(item =>
          <option key={scopeKey(item.scope)} value={scopeKey(item.scope)}>
            {item.scope.dataset} · {item.scope.example ?? "validation anchor"} · {item.scope.anchor.slice(0, 12)}
          </option>)}
      </select></label>
      {!activeModel && <label>Graph model <select value={model} onChange={event => setModel(event.target.value)}>
        {MODEL_IDS.map(id => <option key={id} value={id}>{compositionModelLabel(id)}</option>)}
      </select></label>}
      <label>Minimum normalized weight <input type="number" min={0} step="any" value={threshold}
        onChange={event => setThreshold(event.target.value)} /></label>
    </div>
    {graphCorpus.error && <p role="alert">{graphCorpus.error}</p>}
    {selectedMeasurement ? <details className="cc-provenance">
      <summary>{selectedMeasurement.scope.dataset} · {selectedMeasurement.scope.anchor.slice(0, 12)} · {selectedMeasurement.complete ? "complete retained graph" : "partial evidence"}</summary>
      <dl>
        <dt>Checkpoint</dt><dd>{selectedMeasurement.scope.checkpoint}</dd>
        <dt>Exact anchor</dt><dd>{selectedMeasurement.scope.anchor}</dd>
        <dt>Example</dt><dd>{selectedMeasurement.scope.example ?? "Unavailable"}</dd>
        <dt>BatchTopK pool</dt><dd>{selectedMeasurement.scope.pool ?? "Unavailable"}</dd>
        <dt>Measurement artifact</dt><dd>{selectedMeasurement.scope.artifact}</dd>
      </dl>
    </details> : <p role="status">{snapshot ? "That saved measurement is unavailable for these selected circuits."
      : unavailable ?? "No dependency measurement is available for the current exact context."}</p>}
    {selected.length > 0 && <>
      {selected.length < 2 && <p>Select two or more circuits to inspect composition.</p>}
      <div className="cc-actions">
        {circuits.map(circuit => <button key={circuit.id} type="button" aria-expanded={expanded.includes(circuit.id)}
          onClick={() => setExpanded(old => old.includes(circuit.id) ? old.filter(id => id !== circuit.id) : [...old, circuit.id])}>
          {expanded.includes(circuit.id) ? "Collapse" : "Expand"} {circuit.name}
        </button>)}
      </div>
      <CompositionGraph analysis={analysis} expanded={expanded} measurement={selectedMeasurement} />
      <RelationDetails analysis={analysis} names={parentId} measured={!!selectedMeasurement} />
      {nested.length > 0 && <div className="cc-section"><h3>Shared subcircuits</h3><ul>{nested.map(pair =>
        <li key={JSON.stringify([pair.left, pair.right])}>{parentId.get(pair.left)} / {parentId.get(pair.right)}
          {pair.nested && " · one contains the other"}{pair.shared.length > 0 && ` · ${pair.shared.map(id => parentId.get(id)).join(", ")}`}
        </li>)}</ul></div>}
      {claims.length > 0 && <div className="cc-section"><h3>Researcher relationship notes</h3><ul>
        {claims.map((claim, i) => <li key={i}><b>{parentId.get(claim.source)} → {parentId.get(claim.target)}</b>
          {" · "}{claim.status}<p>{claim.statement}</p>{claim.evidence && <p>{claim.evidence}</p>}</li>)}
      </ul></div>}
      {example && onSelectToken && <Timeline circuits={circuits} example={example} anchor={anchor}
        coefficients={liveSelected ? selectedMeasurement?.coefficients : measurement?.coefficients}
        onSelect={id => { setSnapshot(""); onSelectToken(id); }} model={model} />}
      <CrossModel results={results} measured={!!selectedMeasurement} />
    </>}
  </div>;
}

function RelationDetails({ analysis, measured, names }: { analysis: Analysis; measured: boolean; names: ReadonlyMap<string, string> }) {
  const { direct, bridges, convergence, paths, shared, coverage } = analysis;
  return <div className="cc-section cc-findings">
    <h3>Relationships at this anchor</h3>
    <p>{coverage.map(item => `${names.get(item.id)} ${item.measured}/${item.members} component rows${item.empty === null ? "" : ` (${item.empty} empty)`}`).join(" · ")}</p>
    {measured && coverage.some(item => item.measured > 0) && !paths.length && !convergence.length
      && <p>No inter-circuit edge or path meets the weight filter in the available records for this model.</p>}
    {measured && !coverage.some(item => item.measured > 0) && !convergence.length
      && <p>No downstream component rows are available for the selected circuits in this model.</p>}
    <details><summary>Directed connections ({direct.length})</summary><ul>{direct.map(({ from, to, edge }) =>
      <li key={JSON.stringify([from, to, edge.source, edge.target])}>{names.get(from)} → {names.get(to)} · <EdgeLine edge={edge} /></li>)}</ul></details>
    <details><summary>Two-hop bridges ({bridges.length})</summary><ul>{bridges.map(({ from, to, via, incoming, outgoing }) =>
      <li key={JSON.stringify([from, to, incoming.source, via, outgoing.target])}>
        {names.get(from)} → {names.get(to)} · <code>{incoming.source} → {via} → {outgoing.target}</code>
        <div><EdgeLine edge={incoming} /> · <EdgeLine edge={outgoing} /></div>
      </li>)}</ul></details>
    <details><summary>Convergent targets ({convergence.length})</summary><ul>{convergence.map(({ target, arms }) =>
      <li key={target}><code>{target}</code><ul>{arms.map(({ circuit, edge }) =>
        <li key={JSON.stringify([circuit, edge.source])}>{names.get(circuit)} · <EdgeLine edge={edge} /></li>)}</ul></li>)}</ul></details>
    <details><summary>Shortest measured paths ({paths.length})</summary>
      <p>One shortest path per ordered circuit pair at this anchor.</p>
      <ul>{paths.map(({ from, to, edges }) => <li key={JSON.stringify([from, to])}>{names.get(from)} → {names.get(to)}
        <ol>{edges.map(edge => <li key={edge.target}><EdgeLine edge={edge} /></li>)}</ol>
      </li>)}</ul>
    </details>
    <details><summary>Shared component identities ({shared.length})</summary><ul>{shared.map(item =>
      <li key={item.id}><code>{item.id}</code> · {item.circuits.map(id => names.get(id)).join(", ")}</li>)}</ul></details>
  </div>;
}

function EdgeLine({ edge }: { edge: CompositionEdge }) {
  return <><code>{edge.source} → {edge.target}</code> · E {format(edge.signed)} · w {format(edge.weight)}
    {edge.row && <small> · {edge.row.shard}/{edge.row.record} · flat index {edge.row.flat} · rank {edge.row.rank} (zero-based)</small>}</>;
}

function CompositionGraph({ analysis, expanded, measurement }: { analysis: Analysis; expanded: string[]; measurement?: Measurement }) {
  const arrow = useId();
  const [chosen, choose] = useState("");
  const { nodes, projected, width, height } = projectCompositionGraph(analysis, expanded);
  const selected = projected.get(chosen);
  const selectedNode = nodes.get(chosen);
  const weighted = [...projected].map(([key, item]) => ({
    key, ...item, strength: item.records.reduce((maximum, edge) => Math.max(maximum, edge.weight), 0),
  }));
  const maximum = weighted.reduce((max, item) => Math.max(max, item.strength), 0);
  return <div className="cc-section">
    <h3>Composition graph</h3>
    <p>Collapsed arrows group several component records. Expand the circuits or inspect a shortest measured path to follow exact component identities.</p>
    <div className="cc-graph-scroll"><svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-label="Directed circuit composition graph">
      <defs>{(["positive", "negative", "zero"] as const).map(sign => <marker key={sign} id={`${arrow}-${sign}`}
        markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">
        <path d="M 0 0 L 8 4 L 0 8 Z" className={`cc-arrow ${sign}`} /></marker>)}</defs>
      {weighted.map(({ key, from, to, records, strength }) => {
        const a = nodes.get(from)!;
        const b = nodes.get(to)!;
        const sign = records[0].signed > 0 ? "positive" : records[0].signed < 0 ? "negative" : "zero";
        const dir = a.x < b.x ? 1 : -1;
        const offset = sign === "positive" ? -4 : sign === "negative" ? 4 : 0;
        const d = from === to
          ? `M ${a.x - 25 + offset},${a.y - 9} C ${a.x - 52},${a.y - 35 + offset * 2} ${a.x + 52},${a.y - 35 + offset * 2} ${a.x + 25 + offset},${a.y - 9}`
          : a.x === b.x
            ? `M ${a.x + 80},${a.y + offset} C ${a.x + 120 + offset},${a.y + offset} ${b.x + 120 + offset},${b.y + offset} ${b.x + 80},${b.y + offset}`
            : `M ${a.x + dir * 80},${a.y + offset} C ${(a.x + b.x) / 2},${a.y + offset} ${(a.x + b.x) / 2},${b.y + offset} ${b.x - dir * 80},${b.y + offset}`;
        return <path key={key} d={d} fill="none" role="button" aria-label={`${a.text} to ${b.text}, ${sign}, maximum weight ${format(strength)}`}
          tabIndex={0} className={`cc-edge ${sign} ${chosen === key ? "selected" : ""}`} markerEnd={`url(#${arrow}-${sign})`}
          style={{ "--cc-edge-width": 1.5 + (maximum ? 2.5 * Math.sqrt(strength / maximum) : 0) } as CSSProperties}
          onClick={() => choose(key)} onKeyDown={event => { if (event.key === "Enter") choose(key); }}>
          <title>{records.length} measured {sign} edge(s) · max w {format(strength)}</title>
        </path>;
      })}
      {[...nodes].map(([id, node]) => <g key={id} transform={`translate(${node.x} ${node.y})`}
        role="button" tabIndex={0} className={`cc-node ${node.component ? "component" : ""}`}
        aria-label={node.text} onClick={() => choose(id)} onKeyDown={event => { if (event.key === "Enter") choose(id); }}>
        <rect x={-80} y={-10} width={160} height={20} rx={5} />
        <text textAnchor="middle" y={3}>{node.text.length > 26 ? node.text.slice(0, 25) + "…" : node.text}</text><title>{node.text}</title>
      </g>)}
    </svg></div>
    <p className="cc-legend">Blue = positive signed E · rust = negative · grey = zero · width follows maximum normalized w</p>
    {selected && <div className="cc-selected">
      <p>{nodes.get(selected.from)?.text} → {nodes.get(selected.to)?.text} · anchor <code>{measurement?.scope.anchor}</code></p>
      <ul>{selected.records.map(edge => <li key={JSON.stringify([edge.source, edge.target])}><EdgeLine edge={edge} /></li>)}</ul>
    </div>}
    {selectedNode && <div className="cc-selected"><p>{selectedNode.text}
      {selectedNode.component && ` · ${componentIdentity(selectedNode.component).gateId}`}</p></div>}
  </div>;
}

function Timeline({ circuits, example, anchor, onSelect, coefficients, model }: {
  circuits: CircuitMembership[]; example: TokenExample; anchor?: string; onSelect: (id: string) => void;
  coefficients?: Record<string, number>; model: string;
}) {
  const [details, setDetails] = useState(false);
  const chars = Array.from(example.prompt);
  const tokens = example.positions.map((pos, i) => ({ id: pos.anchor_id, pos,
    text: chars.slice(example.positions[i - 1]?.character_boundary ?? 0, pos.character_boundary).join(""),
    values: new Map(tokenGates(pos).map(({ id, value }) => [id, value])),
  }));
  const rows = circuits.flatMap(circuit => MODEL_IDS.map(id => ({
    circuit, model: id, columns: activationTimeline(circuit, id, tokens),
  })).filter(row => row.columns.some(column => column.total)));
  const overlaps = rows.flatMap((left, i) => rows.slice(i + 1).filter(right => left.model === right.model)
    .map(right => ({ left, right, shared: left.columns.filter((cell, j) => cell.active && right.columns[j].active).length })));
  return <div className="cc-section">
    <h3>Circuit activation timeline</h3>
    <p>Fraction of exact component members whose shared gate fired. Select a token to inspect its components and measured dependencies.</p>
    <label><input type="checkbox" checked={details} onChange={event => setDetails(event.target.checked)} /> Show component activation for {compositionModelLabel(model)}</label>
    <div className="cc-scroll">
      <table className="cc-table"><thead><tr><th>Circuit / model</th>{tokens.map(({ id, text, pos }) =>
        <th key={id}><button type="button" onClick={() => onSelect(id)} aria-pressed={id === anchor}
          title={`${id} · ${MODEL_IDS.map(key => `${compositionModelLabel(key)} ${pos.model_positions[key]}`).join(", ")}`}>{text}</button></th>)}</tr></thead>
        <tbody>{rows.map(row => <RowWithComponents key={JSON.stringify([row.circuit.id, row.model])}
          row={row} anchor={anchor} coefficients={coefficients} details={details && row.model === model} />)}</tbody>
      </table>
    </div>
    <details><summary>Co-activation within this sentence ({tokens.length} aligned tokens)</summary><ul>
      {overlaps.map(({ left, right, shared }) => <li key={JSON.stringify([left.circuit.id, right.circuit.id, left.model])}>
        {compositionModelLabel(left.model)} · {left.circuit.name} / {right.circuit.name} · {shared}/{tokens.length} tokens with active members in both
      </li>)}
    </ul></details>
  </div>;
}

function RowWithComponents({ row, anchor, coefficients, details }: {
  row: { circuit: CircuitMembership; model: string; columns: ReturnType<typeof activationTimeline> };
  anchor?: string; coefficients?: Record<string, number>; details: boolean;
}) {
  const ids = row.columns[0]?.rows.map(component => component.id) ?? [];
  return <>
    <tr><th>{row.circuit.name} · {compositionModelLabel(row.model)}</th>{row.columns.map(cell =>
      <td key={cell.id} className={cell.id === anchor ? "cc-current-token" : ""} title={`${cell.active}/${cell.total} active`}
        style={{ backgroundColor: `rgb(69 125 122 / ${(cell.fraction ?? 0) * 0.24})` }}>{cell.active}/{cell.total}</td>)}</tr>
    {details && ids.map((id, i) => <tr key={id}><th><code>{id}</code> <small>gˢ</small></th>{row.columns.map(cell =>
      <td key={cell.id} className={cell.id === anchor ? "cc-current-token" : ""}>
        {format(cell.rows[i].activation)}
        {cell.id === anchor && coefficients?.[id] !== undefined && <small> · e {format(coefficients[id])}</small>}
      </td>)}</tr>)}
  </>;
}

function CrossModel({ results, measured }: { results: { id: string; analysis: Analysis }[]; measured: boolean }) {
  function covered(analysis: Analysis) {
    return analysis.coverage.some(circuit => circuit.measured > 0) || analysis.convergence.length > 0;
  }
  return <div className="cc-section">
    <h3>Cross-model structure</h3>
    <p>Counts refer to the selected context and weight filter. Overlap compares directed circuit and shared-gate motifs.</p>
    <div className="cc-scroll"><table className="cc-table"><thead><tr><th>Model</th><th>Measured members</th><th>Direct</th><th>Bridges</th><th>Convergence</th><th>Exact bridge components</th></tr></thead>
      <tbody>{results.map(({ id, analysis }) => <tr key={id}><th>{compositionModelLabel(id)}</th>
        <td>{measured ? `${analysis.coverage.reduce((sum, c) => sum + c.measured, 0)}/${analysis.coverage.reduce((sum, c) => sum + c.members, 0)}` : "Unavailable"}</td>
        <td>{covered(analysis) ? analysis.direct.length : "—"}</td><td>{covered(analysis) ? analysis.bridges.length : "—"}</td>
        <td>{covered(analysis) ? analysis.convergence.length : "—"}</td>
        <td>{[...new Set(analysis.bridges.map(bridge => bridge.via))].map(key => <code key={key} className="cc-code-line">{key}</code>)}</td>
      </tr>)}</tbody></table></div>
    {measured && <ul>{results.flatMap((left, i) => results.slice(i + 1).map(right => {
      const comparison = compareStructure(left.analysis, right.analysis);
      const pair = `${compositionModelLabel(left.id)} / ${compositionModelLabel(right.id)}`;
      if (!covered(left.analysis) || !covered(right.analysis)) return <li key={left.id + right.id}>{pair} · comparison unavailable</li>;
      return <li key={left.id + right.id}>{pair}
        {" · motif overlap "}{comparison.jaccard === null ? "undefined (no retained motifs)" : format(comparison.jaccard)}
        {" · shared bridge gates "}{comparison.sharedBridges.join(", ") || "none observed"}
        {comparison.leftBridges.length > 0 && ` · observed only in ${compositionModelLabel(left.id)} ${comparison.leftBridges.join(", ")}`}
        {comparison.rightBridges.length > 0 && ` · observed only in ${compositionModelLabel(right.id)} ${comparison.rightBridges.join(", ")}`}
      </li>;
    }))}</ul>}
  </div>;
}
