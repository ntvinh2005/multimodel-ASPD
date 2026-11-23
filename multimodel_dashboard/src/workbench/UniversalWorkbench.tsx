import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { componentsForSelection, edgesForAnchor, jaccard, traceGraph, tvDistance } from "./graphQueries";
import { formatBytes, measuredValidationArtifact, modelPairLabel } from "./measuredArtifact";
import { measuredWorkbenchDataset as data } from "./measuredDataset";
import type { Component, DependencyEdge, ResearchHypothesis, TraceDirection } from "./types";
import "./workbench.css";
import "./researcherFixes.css";
import "./lightTheme.css";
import "./activationExamples.css";

const palette = ["#51d6c3", "#f0b45b", "#8ca8ff", "#de7cdf", "#73c982", "#ed7d77", "#55a6d9", "#c49a6c"];
const gateColors: Record<string, string> = Object.fromEntries(data.gates.map((gate, index) => [gate.id, palette[index % palette.length]]));
const defaultGates = new Set(data.manifest.display?.defaultGateIds ?? data.gates.slice(0, 3).map((gate) => gate.id));

export default function UniversalWorkbench() {
  const [gateIds, setGateIds] = useState(defaultGates);
  const [modelIds, setModelIds] = useState(new Set(data.models.map((model) => model.id)));
  const [spotlight, setSpotlight] = useState<string | null>(data.manifest.display?.defaultSpotlight ?? data.gates[0]?.id ?? null);
  const [anchorId, setAnchorId] = useState(data.manifest.display?.defaultAnchorId ?? data.anchors[0].id);
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [direction, setDirection] = useState<TraceDirection>("parents");
  const [depth, setDepth] = useState(1);
  const [threshold, setThreshold] = useState(data.manifest.display?.defaultEdgeThreshold ?? 0);
  const [representativeOnly, setRepresentativeOnly] = useState(false);
  const [widthScale, setWidthScale] = useState<"linear" | "sqrt" | "log">("sqrt");
  const [tab, setTab] = useState<"details" | "examples" | "locator" | "parents" | "pinned" | "notes">("details");
  const [hypotheses, setHypotheses] = useState<ResearchHypothesis[]>(() => JSON.parse(localStorage.getItem("aspd-universal-hypotheses") || "[]") as ResearchHypothesis[]);
  const [pinnedComponentIds, setPinnedComponentIds] = useState<string[]>(() => JSON.parse(localStorage.getItem("aspd-pinned-components") || "[]") as string[]);
  const [pinnedEdgeIds, setPinnedEdgeIds] = useState<string[]>(() => JSON.parse(localStorage.getItem("aspd-pinned-edges") || "[]") as string[]);
  const [note, setNote] = useState("");
  const [showMeasuredSummary, setShowMeasuredSummary] = useState(true);

  useEffect(() => {
    const root = document.documentElement;
    const body = document.body;
    const previous = { rootScheme: root.style.colorScheme, rootBackground: root.style.background, bodyBackground: body.style.background };
    root.style.colorScheme = "light";
    root.style.background = "#f6f8fb";
    body.style.background = "#f6f8fb";
    return () => {
      root.style.colorScheme = previous.rootScheme;
      root.style.background = previous.rootBackground;
      body.style.background = previous.bodyBackground;
    };
  }, []);

  useEffect(() => localStorage.setItem("aspd-universal-hypotheses", JSON.stringify(hypotheses)), [hypotheses]);
  useEffect(() => localStorage.setItem("aspd-pinned-components", JSON.stringify(pinnedComponentIds)), [pinnedComponentIds]);
  useEffect(() => localStorage.setItem("aspd-pinned-edges", JSON.stringify(pinnedEdgeIds)), [pinnedEdgeIds]);
  const visibleComponents = useMemo(() => componentsForSelection(data.components, gateIds, modelIds, representativeOnly), [gateIds, modelIds, representativeOnly]);
  const visibleIds = useMemo(() => new Set(visibleComponents.map((component) => component.id)), [visibleComponents]);
  const visibleEdges = useMemo(() => edgesForAnchor(data.edges, anchorId, visibleIds, threshold), [anchorId, visibleIds, threshold]);
  const anchorEdgeRecords = useMemo(() => data.edges.filter((item) => item.anchorId === anchorId && visibleIds.has(item.source) && visibleIds.has(item.target)), [anchorId, visibleIds]);
  const trace = useMemo(() => selectedComponent ? traceGraph(visibleEdges, selectedComponent, direction, depth) : null, [visibleEdges, selectedComponent, direction, depth]);
  const edge = data.edges.find((item) => item.id === selectedEdge) ?? null;
  const component = data.components.find((item) => item.id === selectedComponent) ?? null;
  const inspectedGateId = component?.gateId ?? spotlight;
  const anchor = data.anchors.find((item) => item.id === anchorId)!;

  useEffect(() => {
    if (selectedComponent && !visibleIds.has(selectedComponent)) setSelectedComponent(null);
    if (selectedEdge) {
      const current = data.edges.find((item) => item.id === selectedEdge);
      if (!current || !visibleIds.has(current.source) || !visibleIds.has(current.target)) setSelectedEdge(null);
    }
  }, [selectedComponent, selectedEdge, visibleIds]);

  function toggleGate(id: string) { setGateIds((old) => { const next = new Set(old); next.has(id) ? next.delete(id) : next.add(id); return next; }); }
  function toggleModel(id: string) { setModelIds((old) => { const next = new Set(old); next.has(id) ? next.delete(id) : next.add(id); return next; }); }
  function selectAnchor(id: string) {
    setAnchorId(id);
    const scenario = data.manifest.export?.selectedScenarios.find((item) => item.anchorId === id);
    if (scenario) {
      setGateIds(new Set([scenario.gateId, ...scenario.displayParentGateIds]));
      setSpotlight(scenario.gateId);
      setSelectedComponent(null);
      setSelectedEdge(null);
    }
  }
  function togglePinnedComponent(id: string) { setPinnedComponentIds((old) => old.includes(id) ? old.filter((item) => item !== id) : [...old, id]); }
  function togglePinnedEdge(id: string) { setPinnedEdgeIds((old) => old.includes(id) ? old.filter((item) => item !== id) : [...old, id]); }
  function saveHypothesis() {
    if (!note.trim()) return;
    setHypotheses((old) => [...old, { id: crypto.randomUUID(), title: note.trim().slice(0, 72), notes: note.trim(), status: "Unreviewed", gateIds: [...gateIds], componentId: selectedComponent, edgeId: selectedEdge, anchorId, modelIds: [...modelIds], sourceRevision: data.manifest.sourceRevision, createdAt: new Date().toISOString() }]);
    setNote("");
  }

  return <div className="uw-shell">
    <header className="uw-header">
      <div><span className="uw-kicker">MULTI-MODEL ASPD · UNIVERSAL MECHANISM DISCOVERY</span><h1>Circuit Workbench</h1><p>Select a gate, reveal its components, trace measured dependencies, compare model-specific realizations.</p></div>
      <div className="uw-provenance"><span>POST-HOC SOURCE</span><code>{data.manifest.sourceRevision.slice(0, 12)}</code><small>Gemma · SmolLM2 · Granite</small></div>
    </header>
    <nav className="uw-data-switch" aria-label="Data view"><button className={showMeasuredSummary ? "active" : ""} onClick={() => setShowMeasuredSummary(true)}>Measured validation summary</button><button className={!showMeasuredSummary ? "active" : ""} onClick={() => setShowMeasuredSummary(false)}>Measured graph subset</button></nav>
    {showMeasuredSummary ? <MeasuredSummary onExplore={() => setShowMeasuredSummary(false)} /> : <>
    <div className="uw-warning"><strong>MEASURED · PROFILE-BACKED BOUNDED VISUALIZATION</strong> Components, stored parent sets, per-anchor S/π, and signed edges come from the completed hash-verified post-hoc artifact. The deterministic browser subset is not a dictionary-wide estimate, and first-order Taylor edges are not general causal validation.</div>
    <section className="uw-controls">
      <div className="uw-control-group uw-gates"><label>SHARED GATES <b>{gateIds.size} selected</b>{spotlight && <em>Spotlight: {spotlight}</em>}</label><div>{data.gates.map((gate) => { const selected = gateIds.has(gate.id); const activate = () => { if (!selected) toggleGate(gate.id); setSpotlight(gate.id); }; return <button key={gate.id} aria-pressed={selected} aria-label={`${selected ? "Selected" : "Available"} gate ${gate.id}${spotlight === gate.id ? ", spotlighted" : ""}`} className={`${selected ? "selected" : "available"} ${spotlight === gate.id ? "spotlit" : ""}`} style={{ "--gate": gateColors[gate.id] } as React.CSSProperties} onClick={activate} onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); activate(); } }} onDoubleClick={() => { setGateIds(new Set([gate.id])); setSpotlight(gate.id); setSelectedComponent(null); setSelectedEdge(null); }}>{gate.label}<span title={selected ? "Remove from selection" : "Add to selection"} onClick={(event) => { event.stopPropagation(); toggleGate(gate.id); if (selected && spotlight === gate.id) setSpotlight(null); }}>{selected ? "×" : "+"}</span></button>; })}</div><button className="uw-text-button" onClick={() => { setGateIds(new Set()); setSpotlight(null); setSelectedComponent(null); setSelectedEdge(null); }}>Clear selection</button></div>
      <div className="uw-control-group"><label>MODELS</label>{data.models.map((model) => <label className="uw-check" key={model.id}><input type="checkbox" checked={modelIds.has(model.id)} onChange={() => toggleModel(model.id)} />{model.label.split(" ").slice(0, 2).join(" ")}</label>)}</div>
      <div className="uw-control-group"><label htmlFor="uw-anchor">ANCHOR</label><select id="uw-anchor" aria-label="Aligned anchor" value={anchorId} onChange={(event) => selectAnchor(event.target.value)}>{data.anchors.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select><small>Measured canonical anchor identity; original text was not retained in the edge artifact.</small></div>
      <div className="uw-control-group"><label>DISPLAY</label><label className="uw-check"><input type="checkbox" checked={representativeOnly} onChange={(event) => setRepresentativeOnly(event.target.checked)} />Representative only</label><label>Edge threshold {threshold.toFixed(2)}<input aria-label="Minimum normalized edge magnitude" type="range" min="0" max="0.9" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label></div>
    </section>
    <div className="uw-context"><button aria-label="Previous aligned anchor" title="Previous aligned anchor" onClick={() => selectAnchor(data.anchors[Math.max(0, data.anchors.findIndex((a) => a.id === anchorId) - 1)].id)}>‹</button><p>{anchor.text}</p><span>{spotlight ? anchor.firingGateIds?.includes(spotlight) ? "Natural binary firing: yes · continuous g^s unavailable" : "No firing label stored for this gate/anchor" : "Select a gate"}</span><button aria-label="Next aligned anchor" title="Next aligned anchor" onClick={() => selectAnchor(data.anchors[Math.min(data.anchors.length - 1, data.anchors.findIndex((a) => a.id === anchorId) + 1)].id)}>›</button></div>
    <main className="uw-main">
      <section className="uw-graph-area">
        <div className="uw-graph-toolbar">
          <div className="uw-segment">{(["parents", "children", "both"] as TraceDirection[]).map((value) => <button className={direction === value ? "active" : ""} onClick={() => setDirection(value)} key={value}>{value.toUpperCase()}</button>)}</div>
          <label>Depth <select aria-label="Traversal depth" value={depth} onChange={(event) => setDepth(Number(event.target.value))}><option>1</option><option>2</option><option>3</option></select></label>
          <label>Width <select aria-label="Edge width scale" value={widthScale} onChange={(event) => setWidthScale(event.target.value as typeof widthScale)}><option value="linear">Linear</option><option value="sqrt">Square root</option><option value="log">Logarithmic</option></select></label>
          <button onClick={() => { setSelectedComponent(null); setSelectedEdge(null); }}>Restore graph</button>
          <span>Full subset: {visibleComponents.length} components · {visibleEdges.length} displayed edges{trace && <> | Current trace: {trace.nodes.size} components · {trace.edges.size} edges</>}</span>
        </div>
        {visibleEdges.length === 0 && <div className="uw-edge-state">{anchorEdgeRecords.length === 0 ? "Missing: no edge records exist for this selection and anchor." : anchorEdgeRecords.every((item) => item.normalizedMagnitude === 0) ? "Valid zero: edge records were measured, but all normalized magnitudes are exactly zero." : `Filtered: ${anchorEdgeRecords.length} measured edge records are below the current threshold ${threshold.toFixed(2)}.`}</div>}
        <div className="uw-panels">{data.models.filter((model) => modelIds.has(model.id)).map((model) => <ModelGraph key={model.id} modelId={model.id} components={visibleComponents.filter((item) => item.modelId === model.id)} edges={visibleEdges.filter((item) => item.modelId === model.id)} spotlight={spotlight} selectedComponent={selectedComponent} selectedEdge={selectedEdge} trace={trace} widthScale={widthScale} onComponent={(id) => { setSelectedComponent(id); setSelectedEdge(null); setTab("details"); }} onEdge={(id) => { setSelectedEdge(id); setTab("details"); }} />)}</div>
        {visibleComponents.length === 0 && <div className="uw-empty">Select at least one gate and model to materialize a bounded graph subset.</div>}
      </section>
      <aside className="uw-inspector">
        <div className="uw-tabs"><button className={tab === "details" ? "active" : ""} onClick={() => setTab("details")}>Details</button><button className={tab === "examples" ? "active" : ""} onClick={() => setTab("examples")}>Examples</button><button className={tab === "locator" ? "active" : ""} onClick={() => setTab("locator")}>Locator</button><button className={tab === "parents" ? "active" : ""} onClick={() => setTab("parents")}>Parents</button><button className={tab === "pinned" ? "active" : ""} onClick={() => setTab("pinned")}>Pinned {pinnedComponentIds.length + pinnedEdgeIds.length}</button><button className={tab === "notes" ? "active" : ""} onClick={() => setTab("notes")}>Notes {hypotheses.length}</button></div>
        {tab === "details" && <Details edge={edge} component={component} anchorId={anchorId} trace={trace} direction={direction} depth={depth} edges={visibleEdges} componentPinned={component ? pinnedComponentIds.includes(component.id) : false} edgePinned={edge ? pinnedEdgeIds.includes(edge.id) : false} onPinComponent={togglePinnedComponent} onPinEdge={togglePinnedEdge} onSelectEdge={(id) => setSelectedEdge(id)} />}
        {tab === "examples" && <ActivationExamples gateId={inspectedGateId} />}
        {tab === "locator" && <Locator gateId={spotlight} modelIds={modelIds} selected={selectedComponent} onSelect={(id) => { setSelectedComponent(id); setTab("details"); }} />}
        {tab === "parents" && <ParentComparison gateId={spotlight} anchorId={anchorId} modelIds={modelIds} />}
        {tab === "pinned" && <PinnedSubgraph componentIds={pinnedComponentIds} edgeIds={pinnedEdgeIds} onSelectComponent={(id) => { setSelectedComponent(id); setSelectedEdge(null); setTab("details"); }} onSelectEdge={(id) => { setSelectedEdge(id); setTab("details"); }} onRemoveComponent={togglePinnedComponent} onRemoveEdge={togglePinnedEdge} />}
        {tab === "notes" && <div className="uw-notes"><h2>Research hypotheses</h2><textarea value={note} onChange={(event) => setNote(event.target.value)} placeholder="Record a testable hypothesis or finding…" /><button onClick={saveHypothesis}>Save current view</button>{hypotheses.map((item) => <article key={item.id}><span>{item.status}</span><strong>{item.title}</strong><small>{item.gateIds.join(", ")} · {item.anchorId}</small></article>)}</div>}
      </aside>
    </main>
    <details className="uw-provenance-drawer"><summary>Experimental provenance</summary><dl><dt>Dataset</dt><dd>{data.manifest.dataset.id}@{data.manifest.dataset.revision}</dd><dt>Graph split</dt><dd>{data.manifest.dataset.splitPolicy}</dd><dt>Parent backend</dt><dd>job {data.manifest.export?.parentBackend?.slurmJobId} · {data.manifest.export?.parentBackend?.parentRows.toLocaleString()} stored parent rows · K={data.manifest.export?.parentBackend?.rule.k_par}, τ={data.manifest.export?.parentBackend?.rule.tau}</dd><dt>Activation examples</dt><dd>{data.manifest.export?.activationExamples?.experimentFamily} · split {data.manifest.export?.activationExamples?.selectedSplit} · independent graph anchors</dd><dt>Source</dt><dd>{data.manifest.sourceRevision}</dd><dt>Config</dt><dd>{data.manifest.trainingConfig}</dd><dt>Specification</dt><dd>{data.manifest.specification.path} · SHA-256 {data.manifest.specification.sha256}</dd><dt>Screening</dt><dd>β · K_rep={data.manifest.screening.kRep} · τ_rep={data.manifest.screening.tauRep}</dd><dt>Bundle selection</dt><dd>{data.manifest.export?.selectionRule}</dd><dt>Graph bound</dt><dd>{data.manifest.export?.graphBound}</dd><dt>Verification</dt><dd>{data.manifest.export?.verification}</dd>{data.models.map((model) => <Fragment key={model.id}><dt>{model.id} tokenizer</dt><dd>{model.tokenizer}</dd></Fragment>)}</dl></details>
    <footer className="uw-footer"><span>Display is bounded to selected gates/models/anchor; no cross-model dependency edges are rendered.</span><span>Config: {data.manifest.trainingConfig}</span></footer></>}
  </div>;
}

function MeasuredSummary({ onExplore }: { onExplore: () => void }) {
  const artifact = measuredValidationArtifact;
  const e = artifact.evaluation;
  return <main className="uw-measured">
    <section className="uw-measured-hero"><div><span className="uw-eyebrow">MEASURED POST-HOC VALIDATION ARTIFACT</span><h2>{artifact.experimentId}</h2><p>Aggregate parent-set results from the ASPD v3.1 500M checkpoint. The run is technically complete and hash-verified; its summary statistics are descriptive, not a validated claim of universality.</p></div><div className="uw-run-state"><strong>Validation result</strong><span>Checkpoint step {artifact.checkpoint.step.toLocaleString()}</span><code>{artifact.sourceCommit.slice(0, 12)}</code></div></section>
    <section className="uw-stat-grid"><Metric label="Aligned importance anchors" value={e.importanceAnchors.toLocaleString()} /><Metric label="Screened gates" value={e.screenedGates.toLocaleString()} /><Metric label="Compared gates" value={e.comparedGates.toLocaleString()} /><Metric label="Dependency edge records" value={e.edgeRecords.toLocaleString()} /><Metric label="Pair comparisons" value={e.pairComparisons.toLocaleString()} /><Metric label="Raw tensors on HiPerGator" value={formatBytes(artifact.rawArtifactBytes)} /></section>
    <section className="uw-measured-grid"><article><header><div><span className="uw-eyebrow">CROSS-MODEL SUMMARY</span><h3>Parent identity and distribution</h3></div><small>Anchor-weighted descriptive means</small></header><div className="uw-results-table"><div className="head"><span>Model pair</span><span>Jaccard</span><span>TV</span><span>Valid TV rows</span></div>{artifact.pairs.map((pair) => <div key={pair.modelPair}><strong>{modelPairLabel(pair.modelPair)}</strong><span>{pair.jaccard.toFixed(3)}</span><span>{pair.totalVariation.toFixed(3)}</span><span>{pair.validTvRows.toLocaleString()}</span></div>)}</div><p className="uw-caution">Jaccard measures parent-gate identity overlap; TV measures differences between normalized parent distributions. Neither quantity alone validates a universal mechanism.</p></article>
      <aside><span className="uw-eyebrow">MEASURED GRAPH AVAILABLE</span><h3>Browser-safe export from the raw tensors</h3><p>The interactive graph contains {data.components.length.toLocaleString()} measured components and {data.edges.length.toLocaleString()} stored parent edges selected reproducibly from the full parent-profile backend. Per-anchor S/π is read from the completed profile tensors rather than recomputed in the browser adapter. Natural gate contexts come from a separate hash-verified test-split extraction.</p><dl><dt>Split</dt><dd>{e.split}</dd><dt>Parent backend</dt><dd>Slurm {data.manifest.export?.parentBackend?.slurmJobId}</dd><dt>Stored parent rows</dt><dd>{data.manifest.export?.parentBackend?.parentRows.toLocaleString()}</dd><dt>Bundle</dt><dd>{data.manifest.datasetId}</dd><dt>Excluded gates</dt><dd>{artifact.excludedZeroEdgeFiringGates.join(", ")} — no natural firing in the disjoint edge split</dd><dt>Profile revision</dt><dd><code>{data.manifest.export?.parentBackend?.sourceRevision.slice(0, 12)}</code></dd><dt>Training config hash</dt><dd><code>{artifact.checkpoint.trainingConfigSha256}</code></dd></dl><button onClick={onExplore}>Open measured graph subset →</button></aside>
    </section>
    <section className="uw-measured-boundary"><strong>Scientific status</strong><span>{artifact.summaryStatus.replaceAll("_", " ")}</span><p>Exact-ablation diagnostics reported sign agreement for 56/56 Gemma checks, but they do not validate Taylor magnitudes across all models or nonlinear multi-hop paths.</p></section>
  </main>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <article><span>{label}</span><strong>{value}</strong></article>;
}

function ModelGraph({ modelId, components, edges, spotlight, selectedComponent, selectedEdge, trace, widthScale, onComponent, onEdge }: { modelId: string; components: Component[]; edges: DependencyEdge[]; spotlight: string | null; selectedComponent: string | null; selectedEdge: string | null; trace: ReturnType<typeof traceGraph> | null; widthScale: "linear" | "sqrt" | "log"; onComponent: (id: string) => void; onEdge: (id: string) => void }) {
  const svgRef = useRef<SVGSVGElement>(null); const [viewBox, setViewBox] = useState({ x: 0, y: 0, w: 760, h: 510 }); const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);
  const model = data.models.find((item) => item.id === modelId)!;
  const positions = useMemo(() => new Map(components.map((component) => { const layerIndex = model.selectedLayers.indexOf(component.layer); const sameCell = components.filter((other) => other.layer === component.layer && other.matrix === component.matrix); const offset = sameCell.findIndex((other) => other.id === component.id); return [component.id, { x: 72 + layerIndex * 86 + offset * 13, y: 74 + ["Q", "K", "V", "O", "gate", "up", "down"].indexOf(component.matrix) * 58 + offset * 9 }]; })), [components, model]);
  function edgeWidth(value: number | null) { if (value === null) return 1; const scaled = widthScale === "linear" ? value : widthScale === "sqrt" ? Math.sqrt(value) : Math.log1p(value * 9) / Math.log(10); return 1 + scaled * 6; }
  return <article className="uw-model-panel"><header><div><span>{model.label}</span><small>{model.checkpoint}</small></div><button onClick={() => setViewBox({ x: 0, y: 0, w: 760, h: 510 })}>Fit all</button></header><svg ref={svgRef} viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`} onWheel={(event) => { event.preventDefault(); const factor = event.deltaY > 0 ? 1.12 : .89; setViewBox((old) => ({ ...old, w: old.w * factor, h: old.h * factor })); }} onPointerDown={(event) => { drag.current = { x: event.clientX, y: event.clientY, vx: viewBox.x, vy: viewBox.y }; event.currentTarget.setPointerCapture(event.pointerId); }} onPointerMove={(event) => { if (!drag.current) return; setViewBox((old) => ({ ...old, x: drag.current!.vx - (event.clientX - drag.current!.x) * old.w / 760, y: drag.current!.vy - (event.clientY - drag.current!.y) * old.h / 510 })); }} onPointerUp={() => { drag.current = null; }}>
    <defs><marker id={`arrow-${modelId}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" /></marker></defs>
    {model.selectedLayers.map((layer, index) => <g key={layer}><line x1={72 + index * 86} y1="50" x2={72 + index * 86} y2="472" className="uw-layer-line" /><text x={72 + index * 86} y="28" className="uw-layer-label">L{layer}</text></g>)}
    {["Q", "K", "V", "O", "gate", "up", "down"].map((matrix, index) => <text key={matrix} x="15" y={79 + index * 58} className="uw-matrix-label">{matrix}</text>)}
    {edges.map((edge) => { const a = positions.get(edge.source); const b = positions.get(edge.target); if (!a || !b) return null; const relevant = !selectedComponent || trace?.edges.has(edge.id); const measure = (value: number | null) => value === null ? "missing" : value.toFixed(4); const label = `${edge.source} to ${edge.target}; signed E ${measure(edge.signedEffect)}; normalized w ${measure(edge.normalizedMagnitude)}`; return <line key={edge.id} role="button" tabIndex={0} aria-label={label} x1={a.x} y1={a.y} x2={b.x} y2={b.y} markerEnd={`url(#arrow-${modelId})`} className={`uw-edge ${edge.signedEffect === null ? "unknown" : edge.signedEffect < 0 ? "negative" : "positive"} ${selectedEdge === edge.id ? "selected" : ""} ${relevant ? "" : "dim"}`} strokeWidth={edgeWidth(edge.normalizedMagnitude)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onEdge(edge.id); }} onClick={(event) => { event.stopPropagation(); onEdge(edge.id); }} />; })}
    {components.map((component) => { const point = positions.get(component.id)!; const relevant = selectedComponent ? Boolean(trace?.nodes.has(component.id)) : (!spotlight || component.gateId === spotlight); const distance = trace?.distance.get(component.id); const measure = (value: number | null) => value === null ? "missing" : value.toFixed(3); const label = `${component.gateId}, ${model.label}, layer ${component.layer}, matrix ${component.matrix}, beta ${measure(component.beta)}, rho ${measure(component.rho)}`; return <g key={component.id} role="button" tabIndex={0} aria-label={label} transform={`translate(${point.x} ${point.y})`} className={`uw-node ${relevant ? "" : "dim"} ${selectedComponent === component.id ? "selected" : ""}`} onPointerDown={(event) => event.stopPropagation()} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onComponent(component.id); }} onClick={() => onComponent(component.id)}><circle r={component.representative ? 12 : 9} fill={gateColors[component.gateId] ?? "#8fa1b8"} /><circle r={component.representative ? 16 : 13} className="uw-node-ring" /><text y="4">{component.gateId.split(":")[1]}</text>{distance !== undefined && distance > 0 && <text className="uw-hop" x="13" y="-12">{distance}</text>}<title>{label}</title></g>; })}
  </svg><div className="uw-legend"><span><i className="positive" />positive E</span><span><i className="negative" />negative E</span><span><b>◎</b> representative</span><span><b className="uw-color-dot" />node fill = gate identity, not parent status</span><span>Wheel zoom · drag pan</span></div></article>;
}

function ActivationExamples({ gateId }: { gateId: string | null }) {
  if (!gateId) return <div className="uw-empty">Spotlight a gate or select a component to inspect natural firing contexts.</div>;
  const record = data.activationExamples?.find((item) => item.gateId === gateId);
  const provenance = data.manifest.export?.activationExamples;
  if (!record || record.examples.length === 0) return <div className="uw-empty">No saved natural firing contexts are available for {gateId}.</div>;
  return <div className="uw-examples">
    <span className="uw-eyebrow">NATURAL SHARED-GATE FIRINGS · TEST SPLIT</span>
    <h2>{gateId} activation examples</h2>
    <p>These are observed contexts where the trained shared gate fired naturally. Compare repeated lexical, syntactic, formatting, or structural patterns to form a hypothesis; four examples do not validate a semantic label.</p>
    <div className="uw-example-boundary"><strong>Separate extraction</strong><span>These contexts are not the contribution-graph anchor currently selected.</span></div>
    {record.examples.map((example, index) => {
      const max = Math.max(...example.activations, 1e-9);
      return <article key={example.anchorId} className="uw-example-card">
        <header><strong>Example {index + 1}</strong><span>g<sup>s</sup> = {example.activation.toFixed(4)}</span></header>
        <div className="uw-firing-strip" aria-label={`65-position activation window; saved firing at center position ${example.centerPosition}`}>
          {example.activations.map((activation, position) => <i key={position} className={`${example.firings[position] ? "firing" : ""} ${position === example.centerPosition ? "center" : ""}`} style={{ height: `${Math.max(2, activation / max * 100)}%` }} title={`position ${position}: activation ${activation.toFixed(4)}${example.firings[position] ? ", selected firing" : ""}`} />)}
        </div>
        <small>{example.activations.length}-position saved window (maximum 65) · amber bar = this recorded firing · selected firing positions are dark</small>
        <blockquote>{example.context}</blockquote>
        <dl><dt>Source</dt><dd>{example.sourceGroup}</dd><dt>Document</dt><dd>{example.documentId.slice(0, 16)}…</dd><dt>Anchor</dt><dd>{example.anchorId.slice(0, 16)}…</dd></dl>
      </article>;
    })}
    <p className="uw-caution">Natural binary support and continuous g<sup>s</sup> are distinct. The displayed scalar is the saved continuous activation at the center firing; the graph uses a separate natural-firing edge extraction. Source: job family {provenance?.experimentFamily}, checkpoint step {provenance?.checkpointStep}, split {provenance?.selectedSplit}.</p>
  </div>;
}

function Details({ edge, component, anchorId, trace, direction, depth, edges, componentPinned, edgePinned, onPinComponent, onPinEdge, onSelectEdge }: { edge: DependencyEdge | null; component: Component | null; anchorId: string; trace: ReturnType<typeof traceGraph> | null; direction: TraceDirection; depth: number; edges: DependencyEdge[]; componentPinned: boolean; edgePinned: boolean; onPinComponent: (id: string) => void; onPinEdge: (id: string) => void; onSelectEdge: (id: string) => void }) {
  if (edge) { const source = data.components.find((item) => item.id === edge.source); const target = data.components.find((item) => item.id === edge.target); const name = (item: Component | undefined) => item ? `${item.gateId} · ${item.modelId} · L${item.layer}/${item.matrix} · ${item.id.split("/").at(-1)}` : "Unknown component"; return <div className="uw-detail"><span className="uw-eyebrow">DEPENDENCY EDGE · {edge.evidence}</span><h2>{name(source)} → {name(target)}</h2><button className="uw-pin-action" aria-pressed={edgePinned} onClick={() => onPinEdge(edge.id)}>{edgePinned ? "Remove pinned edge" : "Pin edge"}</button><dl><dt>Anchor</dt><dd>{anchorId}</dd><dt>Signed E</dt><dd className={edge.signedEffect !== null && edge.signedEffect < 0 ? "neg" : "pos"}>{edge.signedEffect?.toFixed(4) ?? "Missing"}</dd><dt>Normalized w</dt><dd>{edge.normalizedMagnitude?.toFixed(4) ?? "Missing"}</dd><dt>Upstream coefficient</dt><dd>{edge.upstreamCoefficient?.toFixed(4) ?? "Unavailable — not stored separately"}</dd><dt>Gradient</dt><dd>{edge.gradient?.toFixed(4) ?? "Unavailable — not stored separately"}</dd><dt>Normalization</dt><dd>{edge.normalization}</dd><dt>Exact intervention</dt><dd>{edge.evidence === "intervention-available" ? "A matched exact-ablation record is available" : "Not included for this exported edge"}</dd></dl><p className="uw-caution">Signed E is a measured first-order Taylor attribution on this anchor; it is not general causal validation.</p></div>; }
  if (component) { const incoming = edges.filter((item) => item.target === component.id).sort((a, b) => (b.normalizedMagnitude ?? -1) - (a.normalizedMagnitude ?? -1)); const outgoing = edges.filter((item) => item.source === component.id).sort((a, b) => (b.normalizedMagnitude ?? -1) - (a.normalizedMagnitude ?? -1)); return <div className="uw-detail"><span className="uw-eyebrow">RANK-ONE COMPONENT</span><h2>{component.gateId} · {component.modelId}</h2><button className="uw-pin-action" aria-pressed={componentPinned} onClick={() => onPinComponent(component.id)}>{componentPinned ? "Remove pinned component" : "Pin component"}</button>{trace && <div className="uw-trace-summary"><strong>{direction.toUpperCase()} · depth {depth}</strong><span>{trace.nodes.size - 1} related components · {trace.edges.size} directed edges</span></div>}<dl><dt>Component ID</dt><dd>{component.id}</dd><dt>Location</dt><dd>Layer {component.layer} · {component.matrix}</dd><dt>β</dt><dd>{component.beta?.toFixed(3) ?? "Missing"}</dd><dt>ρ</dt><dd>{component.rho?.toFixed(3) ?? "Missing"}</dd><dt>Representative</dt><dd>{component.representative ? "Yes" : "No — retained in all-components view"}</dd></dl><ConnectionList title="Direct incoming" edges={incoming} otherEnd="source" onSelect={onSelectEdge} /><ConnectionList title="Direct outgoing" edges={outgoing} otherEnd="target" onSelect={onSelectEdge} /></div>; }
  return <div className="uw-detail"><span className="uw-eyebrow">INSPECTION</span><h2>Select a component or edge</h2><p>Component selection reveals its measured neighborhood. Edge selection preserves signed E and normalized w as distinct values.</p><div className="uw-method"><strong>Scientific boundary</strong><p>Shared gate identity establishes correspondence, not a universal mechanism and not one-to-one component matching.</p></div></div>;
}

function ConnectionList({ title, edges, otherEnd, onSelect }: { title: string; edges: DependencyEdge[]; otherEnd: "source" | "target"; onSelect: (id: string) => void }) {
  return <section className="uw-connections"><header><strong>{title}</strong><span>{edges.length}</span></header>{edges.length === 0 ? <p>No displayed direct edges at this anchor and threshold.</p> : edges.map((edge, index) => { const other = data.components.find((item) => item.id === edge[otherEnd]); return <button key={edge.id} onClick={() => onSelect(edge.id)}><b>{index + 1}</b><span>{other ? `${other.gateId} · L${other.layer}/${other.matrix}` : edge[otherEnd]}</span><small className={edge.signedEffect !== null && edge.signedEffect < 0 ? "neg" : "pos"}>E {edge.signedEffect?.toFixed(3) ?? "Missing"}</small><strong>w {edge.normalizedMagnitude?.toFixed(3) ?? "Missing"}</strong></button>; })}</section>;
}

function PinnedSubgraph({ componentIds, edgeIds, onSelectComponent, onSelectEdge, onRemoveComponent, onRemoveEdge }: { componentIds: string[]; edgeIds: string[]; onSelectComponent: (id: string) => void; onSelectEdge: (id: string) => void; onRemoveComponent: (id: string) => void; onRemoveEdge: (id: string) => void }) {
  const components = componentIds.map((id) => data.components.find((item) => item.id === id)).filter((item): item is Component => Boolean(item));
  const edges = edgeIds.map((id) => data.edges.find((item) => item.id === id)).filter((item): item is DependencyEdge => Boolean(item));
  return <div className="uw-pinned"><span className="uw-eyebrow">RESEARCHER EVIDENCE SUBSET</span><h2>Pinned subgraph</h2><p>This is a researcher-curated view. Pinning does not create a dependency, aggregate components, or validate causality.</p>{components.length === 0 && edges.length === 0 ? <div className="uw-empty">Select a component or edge in Details, then pin it here.</div> : <><section><header><strong>Components</strong><span>{components.length}</span></header>{components.map((component) => <article key={component.id}><button onClick={() => onSelectComponent(component.id)}><strong>{component.gateId} · {component.modelId}</strong><small>L{component.layer}/{component.matrix} · β {component.beta?.toFixed(3) ?? "Missing"}</small></button><button aria-label={`Remove pinned component ${component.id}`} onClick={() => onRemoveComponent(component.id)}>×</button></article>)}</section><section><header><strong>Directed edges</strong><span>{edges.length}</span></header>{edges.map((edge) => <article key={edge.id}><button onClick={() => onSelectEdge(edge.id)}><strong>{data.components.find((item) => item.id === edge.source)?.gateId ?? edge.source} → {data.components.find((item) => item.id === edge.target)?.gateId ?? edge.target}</strong><small>E {edge.signedEffect?.toFixed(3) ?? "Missing"} · w {edge.normalizedMagnitude?.toFixed(3) ?? "Missing"} · {edge.anchorId}</small></button><button aria-label={`Remove pinned edge ${edge.id}`} onClick={() => onRemoveEdge(edge.id)}>×</button></article>)}</section></>}</div>;
}

function Locator({ gateId, modelIds, selected, onSelect }: { gateId: string | null; modelIds: Set<string>; selected: string | null; onSelect: (id: string) => void }) {
  if (!gateId) return <div className="uw-empty">Spotlight a gate to inspect component localization.</div>;
  const selectedModels = data.models.filter((model) => modelIds.has(model.id));
  return <div className="uw-locator"><h2>{gateId} component locator</h2><p className="uw-model-coverage">Models included ({selectedModels.length}/{data.models.length}): {selectedModels.map((model) => model.label).join(" · ") || "None"}</p><p className="uw-locator-legend">Cell metric: β (normalized component contribution). White inset = representative under imported β screening, K_rep={data.manifest.screening.kRep}, τ_rep={data.manifest.screening.tauRep}.</p>{selectedModels.map((model) => <section key={model.id}><strong>{model.label}</strong><div className="uw-heatmap"><span />{model.selectedLayers.map((layer) => <small key={layer}>L{layer}</small>)}{["Q", "K", "V", "O", "gate", "up", "down"].flatMap((matrix) => [<small key={`${matrix}-label`}>{matrix}</small>, ...model.selectedLayers.map((layer) => { const component = data.components.find((item) => item.modelId === model.id && item.gateId === gateId && item.layer === layer && item.matrix === matrix); return <button key={`${matrix}-${layer}`} aria-label={component ? `${model.label}, layer ${layer}, matrix ${matrix}, beta ${component.beta?.toFixed(3) ?? "missing"}, ${component.representative ? "representative" : "not representative"}` : `${model.label}, layer ${layer}, matrix ${matrix}, no component`} disabled={!component} className={component?.id === selected ? "selected" : component?.representative ? "representative" : ""} style={{ opacity: component ? .25 + (component.beta ?? 0) * .8 : .08 }} onClick={() => component && onSelect(component.id)} title={component ? `β ${component.beta}` : "No component"}>{component ? component.beta?.toFixed(2) : ""}</button>; })])}</div></section>)}</div>;
}

function ParentComparison({ gateId, anchorId, modelIds }: { gateId: string | null; anchorId: string; modelIds: Set<string> }) {
  if (!gateId) return <div className="uw-empty">Spotlight a downstream gate to compare parent-gate distributions.</div>;
  const selectedModels = data.models.filter((model) => modelIds.has(model.id));
  const rows = data.parentDistributions.filter((item) => item.gateId === gateId && item.anchorId === anchorId && modelIds.has(item.modelId));
  const pairs = rows.flatMap((left, index) => rows.slice(index + 1).map((right) => ({ left, right, jac: jaccard(left.values, right.values), tv: tvDistance(left.values, right.values) })));
  return <div className="uw-parent"><h2>{gateId} parent distributions</h2><p className="uw-model-coverage">Models included ({selectedModels.length}/{data.models.length}): {selectedModels.map((model) => model.label).join(" · ") || "None"}</p>{pairs.length ? <div className="uw-pairwise">{pairs.map((pair) => <div key={`${pair.left.modelId}-${pair.right.modelId}`}><strong>{pair.left.modelId} ↔ {pair.right.modelId}</strong><span>Jaccard {pair.jac === null ? "Undefined" : pair.jac.toFixed(3)}</span><span>TV {pair.tv === null ? "Undefined" : pair.tv.toFixed(3)}</span></div>)}</div> : <p className="uw-undefined">Select at least two models for pairwise comparison.</p>}{rows.map((row) => <section key={row.modelId}><header><strong>{row.modelId}</strong><span>M = {row.totalStrength?.toFixed(3) ?? "Missing"}</span></header>{row.status === "undefined" ? <p className="uw-undefined">Undefined — total parent strength is zero.</p> : Object.entries(row.values ?? {}).sort((a, b) => b[1] - a[1]).map(([gate, value]) => <button key={gate} style={{ "--value": value, "--gate": gateColors[gate] ?? "#789" } as React.CSSProperties}><span>{gate}</span><i /><b>{value.toFixed(3)}</b></button>)}</section>)}<p className="uw-caution">Jaccard compares parent identity; TV compares normalized distributions. Neither confirms universality.</p></div>;
}
