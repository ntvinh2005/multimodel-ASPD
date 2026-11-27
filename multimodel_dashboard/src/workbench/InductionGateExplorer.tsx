import { useMemo, useState } from "react";
import { CONDITIONS, CONDITION_LABEL, inductionCheckpointManifest, inductionGateDataset, type InductionComponent, type InductionContext, type InductionCondition, type InductionGate, type MeasuredInductionSplit } from "./inductionGateDataset";
import InductionQueryGraph, { useInductionQueryGraphs, type GraphRenderer } from "./InductionQueryGraph";
import { searchSharedGates } from "./graphQueries";
import SharedGateSearch from "./SharedGateSearch";
import { NotesLocatorCell, ResearchGateLabel } from "./ResearchNotes";
import type { NoteTarget } from "./researchNotes";
import { ModelSelect, SelectedNoteEditor, TokenAnchorBar, TokenDependencyInspector, TokenSequence, useGraphWitnesses, useSelectedNote, useTokenGroup, useTokenIndex, useWorkbenchAsset } from "./TokenCircuitExplorer";
import { filterTokenGates, graphWitnessForGate, selectExample, selectGroup, selectToken, tokenGates, type GraphWitness, type TokenExample } from "./tokenDiscovery";
import { activationAt, type EvalModel, type MatrixLocator } from "./negationWorkbenchData";
import ComponentBetaLocator from "./ComponentBetaLocator";
import WorkbenchHeader, { type CircuitView } from "./WorkbenchHeader";

const MODEL_LABEL: Record<string, string> = { gemma: "Gemma 2", smol: "SmolLM2", granite: "Granite 3.3" };

export default function InductionGateExplorer({ onCircuitView, onOpenCandidate, Graph }: { onCircuitView: (value: CircuitView) => void; onOpenCandidate: (id: string) => void; Graph: GraphRenderer }) {
  const discovery = inductionGateDataset.splits.discovery;
  const [splitName, setSplitName] = useState<"smoke" | "discovery">(discovery.status === "measured" ? "discovery" : "smoke");
  const split = (splitName === "discovery" && discovery.status === "measured" ? discovery : inductionGateDataset.splits.smoke) as MeasuredInductionSplit;
  const { bundle, loadError } = useInductionQueryGraphs(splitName === "discovery" ? split.manifestSha256 : null, inductionCheckpointManifest);
  const graphSelection = splitName === "discovery" ? bundle?.selection : undefined;
  const [onlyMeasured, setOnlyMeasured] = useState(false);
  const [site, setSite] = useState<"ALL" | "PRE" | "MID">("ALL");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<"contrast" | "fires" | "beta">("contrast");
  const [selectedId, setSelectedId] = useState("");
  const [groupId, setGroupId] = useState("");
  const [condition, setCondition] = useState("induction");
  const [anchorId, setAnchorId] = useState("");
  const [anchorOnly, setAnchorOnly] = useState(false);
  const [model, setModel] = useState<EvalModel>("gemma");
  const { index: tokenIndex, error: tokenIndexError } = useTokenIndex(inductionCheckpointManifest);
  const locator = useWorkbenchAsset<{ matrices: MatrixLocator[] }>("negation/locator.json");
  const dataset = `induction-${splitName}`;
  const witnesses = useGraphWitnesses(tokenIndex, dataset);
  const sortedGates = useMemo(() => {
    const rows = split.gates.filter((gate) =>
      (site === "ALL" || gate.site === site)
      && (!onlyMeasured || Boolean((witnesses?.gates ?? graphSelection)?.[gate.id])),
    );
    return rows.sort((left, right) => {
      if (sort === "fires") return positiveFires(right) - positiveFires(left);
      if (sort === "beta") return maxBeta(right) - maxBeta(left);
      return Math.abs(right.contrast) - Math.abs(left.contrast);
    });
  }, [graphSelection, witnesses, onlyMeasured, site, sort, split]);
  const gates = useMemo(() => searchSharedGates(sortedGates, query), [sortedGates, query]);
  const preferred = gates.find((gate) => gate.id === selectedId) ?? gates[0];
  const groups = Object.keys(tokenIndex?.datasets[dataset] ?? {});
  const defaultWitness = !groupId && preferred ? graphWitnessForGate(witnesses, preferred.id, model, undefined, condition) : undefined;
  const preferredGroups = preferred ? [...(defaultWitness ? [defaultWitness.group] : []), ...(graphSelection?.[preferred.id]?.groups ?? []),
    ...(split.contexts[preferred.id] ?? []).map(context => context.groupId)] : [];
  const selectedGroup = selectGroup(groups, groupId, preferredGroups);
  const aligned = useTokenGroup(tokenIndex, dataset, selectedGroup);
  const examples = aligned.group?.examples;
  const initialWitness = selectedGroup && defaultWitness?.group === selectedGroup ? defaultWitness : undefined;
  const example = selectExample(examples, initialWitness?.condition ?? condition, "induction");
  const anchor = selectToken(example, anchorId || initialWitness?.anchor || "");
  const active = tokenGates(anchor);
  const tokenSelection = filterTokenGates(active, site, query, selectedId);
  const gateKey = anchorOnly ? tokenSelection.selected : preferred?.id;
  const selected = gateKey ? split.gates.find(gate => gate.id === gateKey) : undefined;
  const shown = anchorOnly ? tokenSelection.rows.map(row => ({ id: row.id, value: row.value, aggregate: undefined }))
    : gates.map(row => ({ id: row.id, value: undefined, aggregate: row }));
  const contexts = selected ? split.contexts[selected.id] ?? [] : [];
  const components = selected ? split.components[selected.id] ?? [] : [];
  const contextByExample = useMemo(() => {
    const map = new Map<string, InductionContext>();
    for (const rows of Object.values(split.contexts)) for (const row of rows) map.set(row.exampleId, row);
    return map;
  }, [split]);
  const summary = example && bundle?.examples.find(row => row.exampleId === example.example_id && row.prompt === example.prompt);
  const legacyGraph = anchor?.final_query && gateKey && summary ? summary.graphs[gateKey] : undefined;
  const evidenceReference = example && anchor ? `${dataset} · ${example.example_id} · anchor ${anchor.anchor_id}` : undefined;
  const note = useSelectedNote(`${evidenceReference}/${gateKey}/${model}`);
  const availableWitness = gateKey ? graphWitnessForGate(witnesses, gateKey, model, selectedGroup, example?.condition, anchor?.anchor_id) : undefined;

  function inspectToken(row: TokenExample, id: string) {
    setGroupId(row.group_id);
    setCondition(row.condition);
    setAnchorId(id);
    setAnchorOnly(true);
    setQuery("");
    setSite("ALL");
  }

  function selectGate(id: string) {
    setSelectedId(id);
    if (anchorOnly) return;
    const found = graphWitnessForGate(witnesses, id, model, selectedGroup, example?.condition, anchor?.anchor_id);
    if (found) openWitness(found);
    else { setGroupId(""); setAnchorId(""); }
  }

  function openWitness(found: GraphWitness) {
    setGroupId(found.group);
    setCondition(found.condition);
    setAnchorId(found.anchor);
  }

  function changeSplit(value: "smoke" | "discovery") {
    if (value === "discovery" && discovery.status !== "measured") return;
    setSplitName(value);
    setOnlyMeasured(false);
    setSelectedId("");
    setGroupId("");
    setCondition("induction");
    setAnchorId("");
    setAnchorOnly(false);
    setQuery("");
  }

  return <div className="uw-shell induction-shell">
    <WorkbenchHeader title="Induction Gate Explorer" view="induction" onChangeView={onCircuitView}
      checkpoint={inductionCheckpointManifest} onOpenCandidate={onOpenCandidate} />
    <div className="uw-warning"><strong>MEASURED · NO CIRCUIT CLAIM</strong>{inductionGateDataset.scientificBoundary}</div>
    <section className="ig-toolbar ig-dataset-toolbar">
      <label>Split<select value={splitName} onChange={(event) => changeSplit(event.target.value as "smoke" | "discovery")}><option value="smoke">Smoke · 32 paired groups</option><option value="discovery" disabled={discovery.status !== "measured"}>Discovery {discovery.status === "measured" ? `· ${discovery.groups} paired groups` : "· running / not bundled"}</option></select></label>
      <div className="ig-run-meta"><strong>{split.gates.length.toLocaleString()}</strong><span>active gates</span><strong>{split.pools}</strong><span>BatchTopK pools</span></div>
    </section>
    <main className="ig-layout">
      <aside className="ig-gate-list" aria-label="Shared gates">
        <header>
          <div><strong>{anchorOnly ? "Active gates at token" : "Shared gates"}</strong><span>{Math.min(250, shown.length)} / {shown.length}</span></div>
          <SharedGateSearch value={query} onChange={setQuery} onSelectFirst={() => {
            if (shown[0]) selectGate(shown[0].id);
          }} />
          <div className="ig-gate-filters">
            <label>Site<select value={site} onChange={(event) => setSite(event.target.value as typeof site)}><option>ALL</option><option>PRE</option><option>MID</option></select></label>
            {!anchorOnly && <label>Sort<select value={sort} onChange={(event) => setSort(event.target.value as typeof sort)}><option value="contrast">|Induction − control|</option><option value="fires">Positive final-query fires</option><option value="beta">Maximum representative β</option></select></label>}
          </div>
          {anchorOnly && <button type="button" onClick={() => { setAnchorOnly(false); setOnlyMeasured(false); setQuery(""); }}>Browse all gates</button>}
          {!anchorOnly && (witnesses || graphSelection) && <label className="ig-graph-filter">
            <input type="checkbox" checked={onlyMeasured} onChange={(event) => setOnlyMeasured(event.target.checked)} /> With measured graphs ({Object.keys(witnesses?.gates ?? graphSelection ?? {}).length})
          </label>}
        </header>
        {shown.slice(0, 250).map(({ id, value, aggregate }) => <button type="button" key={id} onClick={() => selectGate(id)}
          className={id === gateKey ? "selected" : ""} aria-pressed={id === gateKey}>
          <strong>{inductionCheckpointManifest ? <ResearchGateLabel checkpoint={inductionCheckpointManifest} gateId={id} /> : id}</strong>
          <span>{value !== undefined ? `gˢ ${value.toPrecision(4)}` : aggregate && `Δ ${signed(aggregate.contrast)}`}</span>
          {aggregate && <small>{positiveFires(aggregate)} positive query fires · β {maxBeta(aggregate).toFixed(3)}</small>}
          {aggregate && (witnesses?.gates ?? graphSelection)?.[id] && <small className="ig-graph-indicator">Measured graph available</small>}
        </button>)}
        {!shown.length && <p className="uw-empty">{anchorOnly && anchor && !active.length ? "No gates fired at this token." : "No gates match these filters."}</p>}
      </aside>
      <section className="ig-detail">
        {gateKey && <header><div><span className="uw-eyebrow">SHARED GATE · {gateKey.split(":")[0]}</span>
          <h2>{inductionCheckpointManifest ? <ResearchGateLabel checkpoint={inductionCheckpointManifest} gateId={gateKey} /> : gateKey}</h2></div>
          {selected && <div className="ig-badges"><span>Final-query contrast {signed(selected.contrast)}</span></div>}
        </header>}
        {tokenIndex && groups.length && inductionCheckpointManifest ? <section className="ig-examples">
          <header><h3>Matched examples</h3><p>Click a token to inspect its active gates and dependencies.</p></header>
          <div className="ig-toolbar negation-example-toolbar">
            <label>Example group<select value={selectedGroup ?? ""} onChange={event => { setGroupId(event.target.value); setAnchorId(""); }}>
              {[...new Set([...preferredGroups.filter(id => id in tokenIndex.datasets[dataset]), ...groups])]
                .map(id => <option key={id} value={id}>{id}</option>)}
            </select></label>
            <label>Condition<select value={example?.condition ?? condition} onChange={event => {
              setGroupId(selectedGroup ?? ""); setCondition(event.target.value); setAnchorId("");
            }}>
              {(examples ?? []).map(row => <option key={row.example_id} value={row.condition}>
                {CONDITION_LABEL[row.condition as InductionCondition] ?? row.condition}
              </option>)}
            </select></label>
            <ModelSelect model={model} onChange={setModel} />
          </div>
          {examples && example && anchor ? <>
            <div className="negation-context-grid">{examples.map(row => {
              const context = contextByExample.get(row.example_id);
              const matchingContext = context?.prompt === row.prompt ? context : undefined;
              const legacy = bundle?.examples.find(item => item.exampleId === row.example_id && item.prompt === row.prompt);
              const scale = gateKey ? Math.max(0, ...row.positions.map(position => activationAt(position, gateKey))) : 0;
              const current = row.example_id === example.example_id;
              const expected = matchingContext?.expectedContinuation ?? legacy?.expected;
              const predictions = matchingContext?.nativePredictions ?? legacy?.prediction;
              return <article className={`ig-context ${current ? "negation-context-selected" : ""}`} key={row.example_id}>
                <header><div><strong>{CONDITION_LABEL[row.condition as InductionCondition] ?? row.condition}</strong><span>{row.example_id}</span></div>
                  {expected !== undefined && <code>Target {JSON.stringify(expected)}</code>}</header>
                <TokenSequence example={row} gate={gateKey} scale={scale} model={model} anchor={current ? anchor.anchor_id : undefined}
                  onSelect={id => inspectToken(row, id)} />
                {predictions && <div className="ig-predictions">{Object.entries(predictions).map(([id, prediction]) =>
                  <span key={id}><b>{MODEL_LABEL[id] ?? id}</b> {JSON.stringify(prediction.predicted_token)} · margin {prediction.logit_margin.toFixed(2)}</span>)}</div>}
              </article>;
            })}</div>
            <TokenAnchorBar anchor={anchor} model={model} activeCount={active.length} anchorOnly={anchorOnly}
              onToggle={!anchorOnly ? () => inspectToken(example, anchor.anchor_id) : undefined}
              onReset={() => setAnchorId(example.positions.find(position => position.final_query)?.anchor_id ?? "")} />
            <TokenDependencyInspector checkpoint={inductionCheckpointManifest} index={tokenIndex} dataset={dataset}
              example={example} anchor={anchor} gate={gateKey} model={model} matrices={locator?.data?.matrices} Graph={Graph} onAnnotate={note.annotate}
              examples={examples} onSelectExampleToken={inspectToken}
              onOpenMeasuredExample={availableWitness && availableWitness.anchor !== anchor.anchor_id
                ? () => { setSelectedId(gateKey ?? ""); openWitness(availableWitness); } : undefined}
              onSelectToken={id => inspectToken(example, id)} onExploreGate={id => {
                if (!active.some(item => item.id === id)) return;
                inspectToken(example, anchor.anchor_id);
                setSelectedId(id);
              }}
              legacy={{ graph: legacyGraph, scope: legacyGraph && bundle && summary ? { checkpoint: inductionCheckpointManifest,
                dataset, example: example.example_id, anchor: legacyGraph.anchorId, pool: summary.poolId,
                artifact: `inductionQueryGraphs.json · ${bundle.provenance.runtime.git_sha} · job ${bundle.provenance.runtime.slurm_job_id}` } : undefined }} />
          </> : <p className="ig-graph-status" role="status">{aligned.error ?? "Loading matched examples…"}</p>}
        </section> : tokenIndexError && selected ? <>
          {bundle && graphSelection?.[selected.id] ? <InductionQueryGraph key={selected.id} gate={selected.id} bundle={bundle} Graph={Graph} onAnnotate={note.annotate} />
            : <p className="ig-graph-status" role="status">{loadError ?? "Token measurements are unavailable in this build."}</p>}
          {contexts.length > 0 && <section className="ig-examples"><header><h3>Activation examples</h3></header>
            {contexts.map(context => <ContextCard key={context.exampleId} context={context} />)}</section>}
        </> : <p className="ig-graph-status" role="status">{tokenIndex ? "No aligned examples for this split." : "Loading aligned examples…"}</p>}
        {inductionCheckpointManifest && gateKey && <>
          <SelectedNoteEditor checkpoint={inductionCheckpointManifest} gate={gateKey} evidenceReference={evidenceReference} note={note} />
          {locator?.data || !selected ? <ComponentBetaLocator key={gateKey} gate={gateKey} checkpoint={inductionCheckpointManifest}
            matrices={locator?.data?.matrices} onAnnotate={note.annotate} error={locator?.error} />
            : <BetaLocator key={gateKey} gate={selected} components={components} onAnnotate={note.annotate} />}
        </>}
        {selected && <details className="ig-examples"><summary>Split-wide gate statistics</summary>
          <div className="ig-metrics"><Metric label="Final activation contrast" value={signed(selected.contrast)} /><Metric label="Mean selected gˢ" value={selected.meanActivation.toFixed(4)} /><Metric label="Positive query fires" value={positiveFires(selected).toString()} /><Metric label="Control query fires" value={controlFires(selected).toString()} /></div>
          <section className="ig-condition-grid">{CONDITIONS.map(key => <div key={key}><strong>{CONDITION_LABEL[key]}</strong><span>{selected.finalQueryFires[key]} final-query fires</span><small>{selected.sequenceFires[key]} sequences fire anywhere</small></div>)}</section>
        </details>}
      </section>
    </main>
    <footer className="uw-footer"><span>Dataset: {split.split} · {split.groups} groups · {split.examples} conditions · exact 2,048-anchor BatchTopK pools.</span><span>Source {split.sourceRevision.slice(0, 12)}</span></footer>
  </div>;
}

function positiveFires(gate: InductionGate) { return gate.finalQueryFires.induction + gate.finalQueryFires.counterfactual; }
function controlFires(gate: InductionGate) { return gate.finalQueryFires.no_match + gate.finalQueryFires.repeated_no_association; }
function maxBeta(gate: InductionGate) { return Math.max(...Object.values(gate.betaByModel).map((value) => value.max)); }
function signed(value: number) { return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`; }
function Metric({ label, value }: { label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong></div>; }

function BetaLocator({ gate, components, onAnnotate }: {
  gate: InductionGate; components: InductionComponent[]; onAnnotate: (target: NoteTarget) => void;
}) {
  if (!components.length) return <section className="ig-beta"><header><h3>β locator</h3><p>Detailed component cells are outside this bounded gate export.</p></header></section>;
  return <section className="ig-beta"><header><div><span className="uw-eyebrow">IMPORTED NATURAL-VALIDATION β</span><h3>Model-specific component locator</h3></div><p>β was not recomputed on induction prompts. White inset marks the frozen K_rep=3, τ_rep=0.07 representatives.</p></header>
    <div className="ig-beta-models">{Object.entries(gate.betaByModel).map(([model, summary]) => <div key={model}>
      <strong>{MODEL_LABEL[model] ?? model}</strong><span>max β {summary.max.toFixed(3)}</span>
      <div className="ig-beta-grid">{components.filter((item) => item.model === model).map((item) => <NotesLocatorCell
        key={item.id} beta={item.beta} representative={item.representative}
        title={`L${item.layer}/${item.matrix} · β ${item.beta?.toFixed(4) ?? "undefined"}`}
        target={inductionCheckpointManifest ? { checkpoint: inductionCheckpointManifest, gateId: gate.id, scope: "component", modelId: model, layer: item.layer, matrix: item.matrix } : undefined}
        onSelect={onAnnotate}><small>L{item.layer}</small><b>{item.matrix}</b></NotesLocatorCell>)}</div>
    </div>)}</div>
  </section>;
}

function ContextCard({ context }: { context: InductionContext }) {
  const maximum = Math.max(...context.positions.map((position) => position.activation), 1e-8);
  let start = 0;
  return <article className={`ig-context condition-${context.condition}`}><header><div><strong>{CONDITION_LABEL[context.condition]}</strong><span>{context.family} · {context.groupId}</span></div><code>expected{JSON.stringify(context.expectedContinuation)}</code></header><p className="ig-sequence">{context.positions.map((position) => { const text = context.prompt.slice(start, position.boundary); start = position.boundary; const opacity = position.activation > 0 ? .14 + .76 * position.activation / maximum : 0; return <span key={position.boundary} className={position.finalQuery ? "final-query" : ""} style={{ "--activation": opacity } as React.CSSProperties} title={`gˢ ${position.activation.toFixed(5)}${position.roles.length ? ` · ${position.roles.join(", ")}` : ""}`}>{text}</span>; })}</p><div className="ig-predictions">{Object.entries(context.nativePredictions).map(([model, prediction]) => <span key={model}><b>{MODEL_LABEL[model] ?? model}</b> {JSON.stringify(prediction.predicted_token)} · margin {prediction.logit_margin.toFixed(2)}</span>)}</div></article>;
}
