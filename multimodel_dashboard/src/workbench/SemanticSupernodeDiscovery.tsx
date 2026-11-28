import { Fragment, useId, useMemo, useState } from "react";
import { MODEL_IDS, MODEL_NAMES, type EvalModel } from "./negationWorkbenchData";
import { useWorkbenchAsset } from "./TokenCircuitExplorer";
import type { GraphRenderer } from "./InductionQueryGraph";
import type { Component, DependencyEdge, MatrixType } from "./types";
import {
  SUPERNODE_PATH, measuredTaylorAt, partitionAt, releasePath,
  type Cluster, type Evidence, type Example, type Partition, type Scope,
  type SupernodeCatalog, type SupernodeRelease, type Taylor, type Ref,
} from "./supernodeDiscovery";
import "./semanticSupernodes.css";

const fmt = (value: number | null | undefined) => value == null ? "Unavailable" : value.toPrecision(3);

export default function SemanticSupernodeDiscovery({ checkpoint, onBaseline, Graph }: {
  checkpoint: string; onBaseline: () => void; Graph: GraphRenderer;
}) {
  const [view, setView] = useState<"shared" | "baseline">("shared");
  const shared = useWorkbenchAsset<SupernodeRelease>(`${SUPERNODE_PATH}/shared-reference.json`, undefined, "no-store");
  const baseline = useWorkbenchAsset<SupernodeRelease>(`${SUPERNODE_PATH}/index.json`, undefined, "no-store");
  const release = view === "shared" ? shared : baseline;
  const pointer = release?.data;
  const path = pointer?.schemaVersion === 1 && pointer.checkpoint === checkpoint
    ? releasePath(pointer.runId, pointer.catalog) : undefined;
  const asset = useWorkbenchAsset<SupernodeCatalog>(path, pointer?.catalog?.sha256);
  const catalog = asset?.data;
  const valid = catalog && catalog.runId === pointer?.runId && catalog.checkpoint === checkpoint
    && catalog.dataset === "natural-validation" && catalog.schemaVersion === 1
    && (view === "shared" ? catalog.mode === "shared-reference-joint" && !!catalog.jointPartition : !catalog.jointPartition);
  return <div className="sn-view">
    <div className="sn-toolbar">
      <label>Supernode method <select value={view} onChange={event => setView(event.target.value as typeof view)}>
        <option value="shared">ASPD shared reference · complete linkage</option>
        <option value="baseline">Archived MiniLM · average linkage</option>
      </select></label>
    </div>
    {valid ? <SupernodeWorkspace key={catalog.runId} catalog={catalog} onBaseline={onBaseline} Graph={Graph} />
      : <section className="uw-empty" role="status">
        {release?.error || asset?.error || (pointer && !path) || catalog
          ? view === "shared"
            ? "Shared reference results are pending export of the pooled PRE/MID anchor vectors."
            : "No verified archived supernode pilot is available for this checkpoint."
          : "Loading the Semantic Supernode pilot…"}
        <button type="button" onClick={onBaseline}>Open component-level Semantic Discovery</button>
      </section>}
  </div>;
}

function usePilotAsset<T>(run: string, ref?: Ref) {
  return useWorkbenchAsset<T>(releasePath(run, ref), ref?.sha256);
}

function SupernodeWorkspace({ catalog, onBaseline, Graph }: {
  catalog: SupernodeCatalog; onBaseline: () => void; Graph: GraphRenderer;
}) {
  const [model, setModel] = useState<EvalModel>("gemma");
  const [selection, setSelection] = useState("");
  const [search, setSearch] = useState("");
  const [labelsUpdated, setLabelsUpdated] = useState(0);
  const [withGraphs, setWithGraphs] = useState(false);
  const [mode, setMode] = useState<"supernodes" | "components" | "comparison">("supernodes");
  const [component, setComponent] = useState("");
  const joint = catalog.jointPartition;
  const partition = joint ?? catalog.models[model];
  const clusters = useMemo(() => [...partition.clusters].sort((a, b) =>
    b.members.length - a.members.length || (a.withinDistance ?? Infinity) - (b.withinDistance ?? Infinity)),
  [partition]);
  const graphed = useMemo(() => new Set(catalog.taylor.witnesses.flatMap(row => [row.from, row.to])), [catalog]);
  const candidates = withGraphs ? clusters.filter(cluster => graphed.has(cluster.id)) : clusters;
  const chosen = candidates.find(cluster => cluster.id === selection) ?? candidates[0] ?? clusters[0];
  const activeComponent = chosen.members.includes(component) ? component : chosen.medoid;
  const byComponent = useMemo(() => new Map(catalog.components.map(c => [c.id, c])), [catalog]);
  const recurrence = partition.evaluation;
  const visible = candidates.filter(cluster => `${cluster.id} ${cluster.members.join(" ")} ${labelFor(catalog, cluster)}`
    .toLowerCase().includes(search.trim().toLowerCase()));
  return <div className="sn-workbench">
    <div className="sn-heading">
      <div><h2>Semantic Supernode Discovery</h2>
        <p>Natural Validation · {catalog.components.length} representative components · {joint ? "joint clustering" : "per-model clustering"} · up to {catalog.settings.support} {joint ? "anchors" : "documents"} per distribution</p></div>
      <button type="button" onClick={onBaseline}>Component-level baseline</button>
    </div>
    <details className="sn-section">
      <summary>How to use this workspace</summary>
      <ol>
        <li>{joint ? "Select a joint cluster, then compare its members’ top contribution anchors across models." : "Choose a model and cluster. Compare its central and farthest members on held-out documents."}</li>
        <li>{joint ? "The distance inspector compares any two members. A cutoff bounds every pair in a joint cluster." : "Open the cross-model view to read the context pairs that carry the most transport mass."}</li>
        <li>Inspect archived Taylor witnesses at their exact anchors. Use component mode to see which terms were measured.</li>
        <li>Save a provisional label. Independent intervention evidence is listed under Causal Validation.</li>
      </ol>
    </details>
    <div className="sn-toolbar">
      <label>{joint ? "Graph model" : "Model"} <select value={model} onChange={event => { setModel(event.target.value as EvalModel); setSearch(""); }}>
        {MODEL_IDS.map(m => <option key={m} value={m}>{MODEL_NAMES[m]}</option>)}</select></label>
      <label>Find a supernode or component <input value={search} onChange={event => setSearch(event.target.value)} placeholder="Gate, matrix or label" /></label>
      <label><input type="checkbox" checked={withGraphs} onChange={event => setWithGraphs(event.target.checked)} /> Taylor witnesses only</label>
      <label>Graph view <select value={mode} onChange={event => setMode(event.target.value as typeof mode)}>
        <option value="supernodes">Semantic supernodes</option><option value="components">Component-level graph</option>
        {!joint && <option value="comparison">Cross-model comparison</option>}
      </select></label>
    </div>
    <div className="sn-grid">
      <aside className="sn-section sn-clusters" aria-label="Semantic Supernode Explorer">
        <h3>Supernodes <small>{visible.length} of {clusters.length}</small></h3>
        {visible.map(cluster => <button key={cluster.id} type="button" aria-pressed={chosen.id === cluster.id}
          onClick={() => setSelection(cluster.id)}>
          <strong>{labelFor(catalog, cluster) || cluster.medoid}</strong>
          <small>{cluster.members.length} {cluster.members.length === 1 ? "member" : "members"} · {joint ? `diameter ${fmt(cluster.diameter)}` : `within OT ${cluster.withinDistance == null ? "—" : fmt(cluster.withinDistance)}`}{graphed.has(cluster.id) && " · Taylor witnesses"}</small>
        </button>)}
        {!visible.length && <p>No matching supernode.</p>}
      </aside>
      <main>
        <section className="sn-section">
          <div className="sn-heading"><h3>Supernode members</h3><small>Study ID <code>{chosen.id}</code></small></div>
          <LabelEditor key={chosen.id} catalog={catalog} cluster={chosen} onSave={() => setLabelsUpdated(labelsUpdated + 1)} />
          {joint && <p>{MODEL_IDS.map(id => `${MODEL_NAMES[id]} ${chosen.members.filter(key => byComponent.get(key)?.model === id).length}`).join(" · ")}. Maximum discovery pairwise OT {fmt(chosen.diameter)}.</p>}
          <p>Central member <code>{chosen.medoid}</code> · Farthest member <code>{chosen.outlier}</code></p>
          <div className="sn-members">{chosen.members.map(key => <button type="button" key={key}
            aria-pressed={activeComponent === key} onClick={() => setComponent(key)}>
            {key} · β {fmt(byComponent.get(key)?.beta)}
          </button>)}</div>
        </section>
        <EvidencePanel key={activeComponent} catalog={catalog} component={activeComponent} />
        {mode === "comparison" ? <CrossModel catalog={catalog} selected={chosen} />
          : <CircuitPanel key={`${chosen.id}-${model}-${mode}`} catalog={catalog} cluster={chosen} model={model} componentMode={mode === "components"} Graph={Graph} />}
      </main>
    </div>
    <details className="sn-section">
      <summary>Clustering &amp; distance inspector</summary>
      <p>{catalog.reference.groundCost} ground cost · {catalog.reference.dimensions} dimensions · {catalog.reference.scope}{joint ? " before gate projection" : " causal text"}.</p>
      <p>Held-out mean within OT {fmt(recurrence.heldout.mean)} ({recurrence.heldout.pairs} measured pairs).
        {joint ? ` Discovery cut τ = ${joint.distanceCut}.`
          : ` Random-size-matched mean ${fmt(recurrence.random?.mean)}. Mean embedding ${fmt(recurrence.meanEmbedding?.mean)}. Shared-gate grouping ${fmt(recurrence.sharedGate?.mean)}. Pair co-assignment Jaccard ${fmt(recurrence.heldoutCoassignmentJaccard)}.`}</p>
      <Clustering key={joint ? "joint" : model} partition={partition} component={activeComponent} method={joint ? "Complete" : "Average"} />
    </details>
    <CausalValidation catalog={catalog} cluster={chosen} />
  </div>;
}

function CausalValidation({ catalog, cluster }: { catalog: SupernodeCatalog; cluster: Cluster }) {
  const rows = catalog.exactAblation.result?.measurements.filter(row => row.from === cluster.id || row.to === cluster.id) ?? [];
  return <details className="sn-section">
    <summary>Causal Validation</summary>
    <p>Simultaneous frozen-gate ablation at the same anchor tests whether the Taylor estimate predicts a downstream change.</p>
    {rows.length ? <div className="sn-table-scroll"><table>
      <thead><tr><th>Model and anchor</th><th>Condition</th><th>Taylor</th><th>Exact normalized effect</th><th>Output JS, nats</th></tr></thead>
      <tbody>{rows.map(row => <tr key={`${row.from}/${row.to}/${row.anchor}/${row.condition}`}>
        <td title={row.anchor}>{MODEL_NAMES[row.model]} · pool {row.pool} · {row.anchor.slice(0, 10)}</td>
        <td>{row.condition}</td><td>{fmt(row.taylor.score)}</td><td>{fmt(row.exact.supernode?.[0])}</td><td>{fmt(row.exact.js?.[0])}</td>
      </tr>)}</tbody>
    </table></div> : <p>No exact ablation measurement for this supernode.</p>}
  </details>;
}

function labelFor(catalog: SupernodeCatalog, cluster: Cluster) {
  try { return localStorage.getItem(`aspd-supernode-label/${catalog.checkpoint}/${catalog.runId}/${cluster.id}`) ?? ""; }
  catch { return ""; }
}

function LabelEditor({ catalog, cluster, onSave }: { catalog: SupernodeCatalog; cluster: Cluster; onSave: () => void }) {
  const [label, setLabel] = useState(() => labelFor(catalog, cluster));
  const [saved, setSaved] = useState(false);
  return <form className="sn-toolbar" onSubmit={event => {
    event.preventDefault();
    try {
      localStorage.setItem(`aspd-supernode-label/${catalog.checkpoint}/${catalog.runId}/${cluster.id}`, label.trim());
      setSaved(true);
      onSave();
    } catch { setSaved(false); }
  }}>
    <label>Provisional researcher label <input value={label} onChange={event => { setLabel(event.target.value); setSaved(false); }} maxLength={100} /></label>
    <button type="submit">Save label</button>{saved && <small role="status">Saved on this browser</small>}
  </form>;
}

function EvidencePanel({ catalog, component }: { catalog: SupernodeCatalog; component: string }) {
  const [split, setSplit] = useState("heldout");
  const [index, setIndex] = useState(0);
  const asset = usePilotAsset<Evidence>(catalog.runId, catalog.examples[component]);
  const data = asset?.data?.runId === catalog.runId ? asset?.data : undefined;
  const examples = data?.examples[split] ?? [];
  const item = examples[index];
  const scope = item && data?.witnesses[item.anchor];
  const counts = catalog.counts[component];
  const joint = !!catalog.jointPartition;
  const max = Math.max(1e-12, ...Object.values(data?.examples ?? {}).flat().flatMap(row =>
    [row.strength, ...(row.heatmap ?? []).map(span => span[6])]));
  return <section className="sn-section" aria-label="Semantic Evidence Inspector">
    <h3>Contribution examples</h3>
    <p><code>{component}</code> · {counts.discovery.exportedNonzero} discovery and {counts.heldout.exportedNonzero} held-out {joint ? "anchors" : "documents"} shown.
      {joint ? ` These cover ${counts.discovery.distinctNonzeroDocuments} and ${counts.heldout.distinctNonzeroDocuments} distinct documents respectively.` : ""}
      Color shows {joint ? "raw contribution κ" : "normalized contribution"} on one component-wide scale. The anchor has an outline.</p>
    <div className="sn-toolbar">
      <label>Examples <select value={split} onChange={event => { setSplit(event.target.value); setIndex(0); }}>
        <option value="heldout">Held out · strongest</option><option value="discovery">Discovery · strongest</option>
        {!joint && <><option value="heldoutContrast">Held out · measured zero</option><option value="discoveryContrast">Discovery · measured zero</option></>}
      </select></label>
      <button type="button" disabled={index === 0} onClick={() => setIndex(index - 1)}>{joint ? "Previous anchor" : "Previous document"}</button>
      <button type="button" disabled={index + 1 >= examples.length} onClick={() => setIndex(index + 1)}>{joint ? "Next anchor" : "Next document"}</button>
      {examples.length > 0 && <small>{index + 1} / {examples.length}</small>}
    </div>
    {scope && item ? <ExampleText scope={scope} item={item} model={catalog.components.find(c => c.id === component)!.model} max={max} strengthLabel={joint ? "κ" : "r"} />
      : <p role="status">{asset?.error ? "Evidence file unavailable." : data ? "No examples for this selection." : "Loading measured contexts…"}</p>}
  </section>;
}

function ExampleText({ scope, item, model, max, strengthLabel = "r" }: { scope: Scope; item?: Example; model: EvalModel; max?: number; strengthLabel?: string }) {
  const span = scope.native?.[model];
  const position = span?.[0] ?? scope.positions?.[model];
  const token = span?.[1] ?? scope.tokenIds?.[model];
  const characters = Array.from(scope.prefix);
  const textAt = (start: number, end?: number) => characters.slice(start, end).join("");
  let offset = 0;
  return <div className="sn-example">
    <small>{scope.source} · pool {scope.pool} · native token {position} (ID {token}){scope.split && ` · ${scope.split}`}{item ? ` · ${strengthLabel} ${fmt(item.strength)} · signed e ${fmt(item.coefficient)}` : ""}</small>
    <div className="sn-text">
      {item?.heatmap ? <>{item.heatmap.map(([anchor, start, end, pos, tokenId, e, value]) => {
        const gap = textAt(offset, start);
        const text = textAt(Math.max(offset, start), end);
        offset = Math.max(offset, end);
        return <Fragment key={anchor}>{gap}<span className={anchor === scope.anchor ? "sn-anchor" : undefined}
          style={{ backgroundColor: `rgba(44,148,117,${max ? .75 * value / max : 0})` }}
          title={`Anchor ${anchor} · token ${pos} (ID ${tokenId}) · e ${e} · ${strengthLabel} ${value}`}>{text}</span></Fragment>;
      })}{textAt(offset)}</>
        : span ? <>{textAt(Math.max(0, span[2] - 400), span[2])}<mark className="sn-anchor">{textAt(span[2], span[3])}</mark>{textAt(span[3])}</> : textAt(Math.max(0, scope.boundary - 400))}
    </div>
    <small title={scope.anchor}>{scope.anchor.slice(0, 12)}{item?.heatmap ? " · aligned positions measured" : span ? " · anchor outline only" : " · archive prefix, exact token ID above"}</small>
  </div>;
}

function CircuitPanel({ catalog, cluster, model, componentMode, Graph }: {
  catalog: SupernodeCatalog; cluster: Cluster; model: EvalModel; componentMode: boolean; Graph?: GraphRenderer;
}) {
  const rows = catalog.taylor.witnesses.filter(row => row.model === model && (row.from === cluster.id || row.to === cluster.id));
  const [anchor, setAnchor] = useState(rows[0]?.anchor ?? "");
  const [threshold, setThreshold] = useState(0);
  const [edge, setEdge] = useState("");
  const samples = [...new Map(rows.map(row => [row.anchor, row.scope])).values()];
  const max = Math.max(1e-8, ...rows.map(row => row.score ?? row.observedTargetMean ?? 0));
  const edges = measuredTaylorAt(rows, model, anchor, threshold);
  const focused = edges.find(row => `${row.from}/${row.to}` === edge) ?? edges[0];
  return <section className="sn-section">
    <h3>{componentMode ? "Component-level graph" : "Supernode circuit graph"}</h3>
    <p>Each graph uses one anchor in one model. Dashed edges summarize saved top-K terms. Unmeasured terms can increase or cancel those effects.</p>
    {rows.length ? <>
      <div className="sn-toolbar"><label>Measured anchor <select value={anchor} onChange={e => setAnchor(e.target.value)}>
        {samples.map(scope => <option key={scope.anchor} value={scope.anchor}>{scope.source} · {scope.anchor.slice(0, 10)}</option>)}
      </select></label>
      <label>Minimum displayed weight {fmt(threshold)} <input type="range" min={0} max={max} step={max / 100}
        value={threshold} onChange={e => setThreshold(Number(e.target.value))} /></label></div>
      {focused && <ExampleText scope={focused.scope} model={model} />}
      {componentMode && Graph && <ComponentGraph catalog={catalog} model={model} anchor={anchor} rows={edges} Graph={Graph} />}
      {!componentMode && <DirectedGraph catalog={catalog} cluster={cluster} model={model} edges={edges} focused={focused} onEdge={setEdge} />}
      <TaylorEdges edges={edges} focused={focused} setEdge={setEdge} componentMode={componentMode} />
      {!edges.length && <p>No cached edges pass the display threshold.</p>}
    </> : <p className="uw-empty">No cached Taylor edge connects this supernode's members to another supernode in this pilot.</p>}
  </section>;
}

const matrixType: Record<string, MatrixType> = {
  q: "Q", k: "K", v: "V", o: "O", mlp_gate: "gate", mlp_up: "up", mlp_down: "down",
};

function ComponentGraph({ catalog, model, anchor, rows, Graph }: {
  catalog: SupernodeCatalog; model: EvalModel; anchor: string; rows: Taylor[]; Graph: GraphRenderer;
}) {
  const [component, setComponent] = useState<string | null>(null);
  const [edge, setEdge] = useState<string | null>(null);
  const metadata = new Map(catalog.components.map(item => [item.id, item]));
  const edges: DependencyEdge[] = rows.flatMap(row => row.targets.flatMap(target =>
    Object.entries(target.effects).map(([source, signedEffect]) => {
      const factor = metadata.get(target.target)!;
      return {
        id: `${source}->${target.target}`, modelId: model, anchorId: anchor,
        source, target: target.target, signedEffect,
        normalizedMagnitude: Math.abs(signedEffect) * factor.uNorm / factor.sigma,
        normalization: "native matrix-output scale", evidence: "measured-taylor",
        upstreamCoefficient: null, gradient: null,
      };
    })));
  if (!edges.length) return null;
  const components: Component[] = [...new Set(edges.flatMap(item => [item.source, item.target]))].map(id => {
    const item = metadata.get(id)!;
    const [, layer, matrix] = id.split("/")[0].split(".");
    return {
      id, modelId: model, gateId: item.gate.replace("/", ":").toUpperCase(),
      layer: Number(layer.slice(1)), matrix: matrixType[matrix],
      beta: item.beta, rho: null, representative: true,
    };
  });
  return <Graph modelId={model} components={components} edges={edges} selectedComponent={component}
    selectedEdge={edge} spotlight={null} trace={null} direction="parents" widthScale="sqrt"
    onComponent={setComponent} onEdge={setEdge} annotationCheckpoint={catalog.checkpoint} />;
}

function DirectedGraph({ catalog, cluster, model, edges, focused, onEdge }: {
  catalog: SupernodeCatalog; cluster: Cluster; model: EvalModel; edges: Taylor[];
  focused?: Taylor; onEdge: (edge: string) => void;
}) {
  const marker = useId();
  const other = [...new Set(edges.map(row => row.from === cluster.id ? row.to : row.from))];
  if (!other.length) return null;
  const title = (id: string) => {
    const node = (catalog.jointPartition ?? catalog.models[model]).clusters.find(item => item.id === id);
    return node && (labelFor(catalog, node) || node.medoid) || id;
  };
  const height = Math.max(125, other.length * 49 + 24);
  const y = (id: string) => id === cluster.id ? height / 2 : other.indexOf(id) * 49 + 37;
  const x = (id: string) => id === cluster.id ? 86 : 425;
  return <svg className="sn-circuit" viewBox={`0 0 512 ${height}`} role="group" aria-label="Directed measured supernode graph">
    <defs><marker id={marker} markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <path d="M 0 0 L 8 3.5 L 0 7 Z" fill="#52738c" />
    </marker></defs>
    {edges.map(row => {
      const dx = row.from === cluster.id ? 1 : -1;
      const start = x(row.from) + 77 * dx;
      const end = x(row.to) - 77 * dx;
      const offset = dx * 7;
      const path = `M ${start} ${y(row.from) + offset} C 235 ${y(row.from) + offset}, 275 ${y(row.to) + offset}, ${end} ${y(row.to) + offset}`;
      const key = `${row.from}/${row.to}`;
      return <g key={key} role="button" tabIndex={0}
        aria-label={`${title(row.from)} to ${title(row.to)}, weight ${fmt(row.score ?? row.observedTargetMean)}`}
        onClick={() => onEdge(key)} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onEdge(key); } }}>
        <path d={path} fill="none" stroke="transparent" strokeWidth="16" />
        <path d={path} fill="none" stroke={focused === row ? "#247693" : "#7995aa"} strokeWidth={focused === row ? 2.6 : 1.5}
          strokeDasharray={row.status === "truncated" ? "4 2" : undefined} markerEnd={`url(#${marker})`} />
      </g>;
    })}
    {[cluster.id, ...other].map(id => <g key={id}>
      <rect x={x(id) - 77} y={y(id) - 16} width="154" height="32" rx="6"
        fill={id === cluster.id ? "#e0f0f5" : "#fff"} stroke="#7495a7" />
      <text x={x(id)} y={y(id)} dominantBaseline="middle" textAnchor="middle" fontSize="10" fill="currentColor">
        {title(id).length > 25 ? `${title(id).slice(0, 24)}…` : title(id)}
      </text>
      <title>{title(id)}</title>
    </g>)}
  </svg>;
}

function TaylorEdges({ edges, focused, setEdge, componentMode }: {
  edges: Taylor[]; focused: Taylor | undefined; setEdge: (v: string) => void; componentMode: boolean;
}) {
  return <div className="sn-edge-list">
    {!componentMode && edges.map(row => <button type="button" key={`${row.from}/${row.to}`}
      aria-pressed={focused === row} onClick={() => setEdge(`${row.from}/${row.to}`)}>
      {row.from} → {row.to} · {fmt(row.score ?? row.observedTargetMean)} · {row.status}
    </button>)}
    {focused && <details open={componentMode}>
      <summary>{focused.observedPairs} / {focused.eligiblePairs} structurally eligible terms measured</summary>
      {(componentMode ? edges : [focused]).flatMap(row => row.targets.flatMap(target => Object.entries(target.effects)
        .map(([source, effect]) => <div className="sn-edge" key={`${source}/${target.target}`}>
          <code>{source}</code><span aria-label={effect < 0 ? "Negative Taylor effect" : "Positive Taylor effect"}>→</span>
          <code>{target.target}</code><small>E {fmt(effect)}</small>
        </div>)))}
    </details>}
  </div>;
}

function CrossModel({ catalog, selected }: { catalog: SupernodeCatalog; selected: Cluster }) {
  const [index, setIndex] = useState(0);
  const [pair, setPair] = useState(0);
  const lookup = new Map(MODEL_IDS.flatMap(model => catalog.models[model].clusters.map(cluster => [cluster.id, { model, cluster }] as const)));
  const links = catalog.matches.filter(match => [match.left, match.right].includes(selected.id));
  const match = links[index] ?? links[0];
  const transported = match?.transported?.[pair];
  const scopeAsset = usePilotAsset<{ runId: string; witnesses: Record<string, Scope> }>(catalog.runId, catalog.transport);
  const scopes = scopeAsset?.data?.runId === catalog.runId ? scopeAsset.data.witnesses : undefined;
  return <section className="sn-section">
    <h3>Cross-model circuit comparison</h3>
    <p>Mixture OT compares distributions across documents. Side-by-side circuit witnesses may be from different anchors.</p>
    {links.length ? <>
      <div className="sn-toolbar"><label>Counterpart <select value={match ? links.indexOf(match) : 0}
        onChange={e => { setIndex(Number(e.target.value)); setPair(0); }}>
        {links.map((other, i) => <option key={`${other.left}/${other.right}`} value={i}>
          {other.left === selected.id ? other.right : other.left} · OT {fmt(other.discovery)}
        </option>)}</select></label>
        {match?.transported?.length ? <label>Transported held-out context pair <select value={pair} onChange={e => setPair(Number(e.target.value))}>
          {match.transported.map((item, i) => <option key={`${item.left}/${item.right}`} value={i}>{i + 1} · mass {fmt(item.mass)}</option>)}
        </select></label> : null}
      </div>
      <p>Discovery OT {fmt(match?.discovery)} · held-out OT {fmt(match?.heldout)} · {match?.accepted ? "Passes exploratory distance screen" : "Above exploratory distance screen"}</p>
      {match.heldoutSameDocumentMass !== undefined && <p>Same-document mass {(100 * match.heldoutSameDocumentMass).toFixed(0)}%
        · exact-anchor mass {(100 * (match.heldoutExactAnchorMass ?? 0)).toFixed(0)}%
        · held-out OT with same-document pairings blocked {fmt(match.heldoutNoSameDocument)}.</p>}
      <div className="sn-comparison">
        {MODEL_IDS.map(model => {
          const primary = [match.left, match.right].find(key => lookup.get(key)?.model === model);
          const alternate = links.find(link => [link.left, link.right].some(key => lookup.get(key)?.model === model));
          const id = primary ?? [alternate?.left, alternate?.right].find(key => key && lookup.get(key)?.model === model);
          const entry = id && lookup.get(id);
          const comparison = primary ? match : alternate;
          const context = primary ? transported : alternate?.transported?.[0];
          const aid = id === comparison?.left ? context?.left : context?.right;
          return <section key={model}><h4>{MODEL_NAMES[model]}</h4>{entry ? <>
            <strong>{labelFor(catalog, entry.cluster) || entry.cluster.medoid}</strong>
            {!primary && <p>Separate comparison with the selected supernode · held-out OT {fmt(comparison?.heldout)}.</p>}
            {aid && scopes?.[aid] && <ExampleText scope={scopes[aid]} model={model} />}
            <CircuitPanel key={id} catalog={catalog} cluster={entry.cluster} model={model} componentMode={false} />
          </> : <p>No counterpart in this selected pair.</p>}</section>;
        })}
      </div>
    </> : <p>No transport comparison was computed for this supernode.</p>}
  </section>;
}

function Clustering({ partition, component, method }: { partition: Partition; component: string; method: "Average" | "Complete" }) {
  const [cut, setCut] = useState<"count" | "distance">("count");
  const [count, setCount] = useState(partition.clusters.length);
  const [threshold, setThreshold] = useState(partition.tree.at(-1)?.[2] ?? 0);
  const [target, setTarget] = useState(0);
  const preview = partitionAt(partition, cut === "count" ? { count } : { threshold });
  const max = partition.distances.reduce((largest, row) => Math.max(largest, ...row), .01);
  const index = partition.componentOrder.indexOf(component);
  const displayed = [index, ...preview.find(group => group.includes(index))!.filter(i => i !== index),
    ...partition.componentOrder.map((_, i) => i)].filter((i, pos, all) => all.indexOf(i) === pos).slice(0, 30);
  return <>
    <div className="sn-toolbar">
      <label>Cut by <select value={cut} onChange={e => setCut(e.target.value as typeof cut)}>
        <option value="count">Cluster count</option><option value="distance">Merge distance</option>
      </select></label>
      {cut === "count" ? <label>Clusters {count} <input type="range" min={1} max={partition.componentOrder.length}
        value={count} onChange={e => setCount(Number(e.target.value))} /></label>
        : <label>Cut {fmt(threshold)} <input type="range" min={0} max={max} step={max / 200}
          value={threshold} onChange={e => setThreshold(Number(e.target.value))} /></label>}
      <small>{preview.length} preview clusters. Saved measurements use the published partition.</small>
    </div>
    <div className="sn-diagnostics">
      <div>
        <h4>OT heatmap</h4><small>Up to 30 components around the current selection. Darker means farther apart.</small>
        <svg className="sn-heatmap" viewBox={`0 0 ${displayed.length * 9} ${displayed.length * 9}`} role="img" aria-label="Discovery OT distance heatmap">
          {displayed.flatMap((i, row) => displayed.map((j, col) => <rect key={`${i}/${j}`}
            x={col * 9} y={row * 9} width={9} height={9} fill={`rgba(36,100,129,${partition.distances[i][j] / max})`}>
            <title>{partition.componentOrder[i]} ↔ {partition.componentOrder[j]} · OT {fmt(partition.distances[i][j])}</title>
          </rect>))}
        </svg>
      </div>
      <div><h4>{method}-linkage dendrogram</h4><Dendrogram partition={partition} threshold={cut === "distance" ? threshold : undefined} /></div>
    </div>
    <div className="sn-toolbar">
      <label>Compare component with <select value={target} onChange={e => setTarget(Number(e.target.value))}>
        {partition.componentOrder.map((id, j) => <option key={id} value={j}>{id}</option>)}
      </select></label>
      <small>Discovery OT {fmt(partition.distances[index][target])} · held-out OT {fmt(partition.heldoutDistances[index][target])}</small>
    </div>
  </>;
}

function Dendrogram({ partition, threshold }: { partition: Partition; threshold?: number }) {
  const n = partition.componentOrder.length;
  const root = n * 2 - 2;
  const points = new Map<number, [number, number]>();
  let leaf = 0;
  const xmax = partition.tree.at(-1)?.[2] || 1;
  const paths: string[] = [];
  function visit(i: number): [number, number] {
    const prior = points.get(i);
    if (prior) return prior;
    if (i < n) return [0, leaf++ * 180 / n];
    const [a, b, distance] = partition.tree[i - n];
    const [ax, ay] = visit(a);
    const [bx, by] = visit(b);
    const x = distance * 275 / xmax;
    const point: [number, number] = [x, (ay + by) / 2];
    paths.push(`M ${ax} ${ay} H ${x} V ${by} H ${bx}`);
    points.set(i, point);
    return point;
  }
  visit(root);
  return <svg viewBox="-5 -5 290 190" role="img" aria-label="Hierarchical clustering dendrogram">
    {paths.map((d, i) => <path key={i} d={d} fill="none" stroke="#42738b" strokeWidth=".65" />)}
    {threshold != null && <line x1={threshold * 275 / xmax} y1={0} x2={threshold * 275 / xmax} y2={180} stroke="#977137" strokeDasharray="4 2" />}
  </svg>;
}
