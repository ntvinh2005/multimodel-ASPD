import { useEffect, useMemo, useState } from "react";
import ComponentBetaLocator from "./ComponentBetaLocator";
import { QueryGraphPanel, type GraphRenderer } from "./InductionQueryGraph";
import { activationAt, MODEL_IDS, MODEL_NAMES, readWorkbenchAsset, type EvalModel, type MatrixLocator } from "./negationWorkbenchData";
import { ResearchGateLabel, ResearchNotesEditor } from "./ResearchNotes";
import { notesForGate } from "./researchNotes";
import { selectToken, tokenGates, tokenGraph, type TokenGraphAsset, type TokenGroup, type TokenIndex } from "./tokenDiscovery";
import "./tokenDiscovery.css";

function useAsset<T>(path?: string, hash?: string) {
  const [result, setResult] = useState<{ path: string; data?: T; error?: string }>();
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    void readWorkbenchAsset<T>(path, controller.signal, hash).then(
      (data) => { if (!controller.signal.aborted) setResult({ path, data }); },
      (error: Error) => { if (!controller.signal.aborted) setResult({ path, error: error.message }); },
    );
    return () => controller.abort();
  }, [path, hash]);
  return result?.path === path ? result : undefined;
}

const datasets = {
  "negation-smoke": "Negation Smoke",
  "negation-discovery": "Negation Discovery",
  "induction-smoke": "Induction Smoke",
  "induction-discovery": "Induction Discovery",
  "selection-frame": "Selection Frame controls",
};

export default function TokenCircuitExplorer({ checkpoint, Graph, initial }: {
  checkpoint?: string; Graph: GraphRenderer; initial: keyof typeof datasets;
}) {
  const [open, setOpen] = useState(false);
  if (!checkpoint) return null;
  return <details className="td-section" onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>Explore by token</summary>
    {open && <TokenPanel checkpoint={checkpoint} Graph={Graph} initial={initial} />}
  </details>;
}

function TokenPanel({ checkpoint, Graph, initial }: {
  checkpoint: string; Graph: GraphRenderer; initial: keyof typeof datasets;
}) {
  const indexAsset = useAsset<TokenIndex>("tokens/index.json");
  const index = indexAsset?.data?.checkpointSha256 === checkpoint ? indexAsset.data : undefined;
  const [dataset, setDataset] = useState<string>(initial);
  const [groupId, setGroup] = useState("");
  const [picked, setPicked] = useState({ example: "", anchor: "" });
  const [wantedGate, setGate] = useState("");
  const [search, setSearch] = useState("");
  const [model, setModel] = useState<EvalModel>("smol");
  const groups = index?.datasets[dataset] ?? {};
  const groupKey = groupId in groups ? groupId : Object.keys(groups)[0];
  const groupRef = groups[groupKey];
  const groupAsset = useAsset<TokenGroup>(groupRef && `tokens/${groupRef.path}`, groupRef?.sha256);
  const group = groupAsset?.data?.checkpointSha256 === checkpoint && groupAsset.data.dataset === dataset
    && groupAsset.data.groupId === groupKey ? groupAsset.data : undefined;
  const example = group?.examples.find((row) => row.example_id === picked.example)
    ?? group?.examples.find((row) => row.condition === "negative") ?? group?.examples[0];
  const anchor = selectToken(example, picked.anchor);
  const active = tokenGates(anchor);
  const matching = active.filter(({ id }) => id.toLowerCase().includes(search.trim().toLowerCase()));
  const gate = matching.find((item) => item.id === wantedGate) ?? matching[0];
  const graphRef = anchor && index?.graphs[anchor.anchor_id];
  const graphAsset = useAsset<TokenGraphAsset>(graphRef && `tokens/${graphRef.path}`, graphRef?.sha256);
  const locator = useAsset<{ matrices: MatrixLocator[] }>("negation/locator.json");
  const matrices = locator?.data?.matrices;
  const dependency = useMemo<Partial<ReturnType<typeof tokenGraph>> & { error?: string }>(() => {
    if (!graphAsset?.data || !example || !anchor || !gate) return {};
    try {
      return tokenGraph(graphAsset.data, example, anchor, checkpoint, gate.id, model, matrices);
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error) };
    }
  }, [graphAsset, example, anchor, checkpoint, gate?.id, model, matrices]);
  if (!index) return <p className="ig-graph-status" role="status">
    {indexAsset?.error ?? (indexAsset?.data ? "Token inventory belongs to a different checkpoint." : "Loading token inventory…")}
  </p>;

  function select(exampleId: string, anchorId = "") {
    setPicked({ example: exampleId, anchor: anchorId });
    setSearch("");
  }

  const evidenceReference = example && anchor ? `${dataset} · ${example.example_id} · anchor ${anchor.anchor_id}` : undefined;
  return <div className="td-pane">
    <section className="ig-toolbar">
      <label>Dataset <select value={dataset} onChange={(event) => { setDataset(event.target.value); select(""); setGroup(""); }}>
        {Object.keys(index.datasets).map((key) => <option key={key} value={key}>{datasets[key as keyof typeof datasets] ?? key}</option>)}
      </select></label>
      <label>Example group <select value={groupKey ?? ""} onChange={(event) => { setGroup(event.target.value); select(""); }}>
        {Object.keys(groups).map((key) => <option key={key}>{key}</option>)}
      </select></label>
      <label>Model <select value={model} onChange={(event) => setModel(event.target.value as EvalModel)}>
        {MODEL_IDS.map((id) => <option key={id} value={id}>{MODEL_NAMES[id]}</option>)}
      </select></label>
    </section>
    <p className="ig-graph-status">Select an aligned token in any sentence. Each selectable span ends at an exact shared boundary. Graphs show same-anchor dependencies.</p>
    <div className="td-contexts">
      {group?.examples.map((row) => {
        const chars = Array.from(row.prompt);
        const max = gate ? Math.max(0, ...row.positions.map((position) => activationAt(position, gate.id))) : 0;
        return <article className="ig-context" key={row.example_id}>
          <header><strong>{row.condition.replaceAll("_", " ")}</strong></header>
          <p className="ig-sequence">{row.positions.map((position, i) => {
            const previous = row.positions[i - 1]?.character_boundary ?? 0;
            const token = chars.slice(previous, position.character_boundary).join("");
            const value = gate ? activationAt(position, gate.id) : undefined;
            return <button type="button" key={position.anchor_id}
              className={`${position.final_query ? "td-final" : ""} ${position.anchor_id === anchor?.anchor_id ? "td-selected" : ""}`}
              aria-pressed={position.anchor_id === anchor?.anchor_id}
              style={{ "--td-activation": max > 0 && value ? .1 + .6 * value / max : 0 } as React.CSSProperties}
              title={`${MODEL_NAMES[model]} token ${position.model_positions[model]}${value === undefined ? "" : ` · gˢ ${value.toPrecision(4)}`}`}
              onClick={() => select(row.example_id, position.anchor_id)}>{token}</button>;
          })}</p>
        </article>;
      })}
    </div>
    {!group && <p className="ig-graph-status" role="status">{groupAsset?.error ?? "Loading aligned examples…"}</p>}
    {example && anchor && <>
      <p className="ig-graph-status"><strong>{anchor.final_query ? "Final query" : "Selected anchor"}</strong> · {example.condition.replaceAll("_", " ")}
        {" · "}{MODEL_NAMES[model]} token {anchor.model_positions[model]} · <code>{anchor.anchor_id.slice(0, 12)}</code>
        {!anchor.final_query && <button type="button" onClick={() => select(example.example_id)}>Return to final token</button>}
      </p>
      <section className="td-gates">
        <label className="uw-gate-search">Shared gates · {active.length} active
          <input type="search" placeholder="e.g. MID:2844" value={search} onChange={(event) => setSearch(event.target.value)} />
        </label>
        <div>{matching.map(({ id, value }) => <button type="button" key={id} aria-pressed={id === gate?.id} onClick={() => setGate(id)}>
          <ResearchGateLabel checkpoint={checkpoint} gateId={id} /> <small>gˢ {value.toPrecision(4)}</small>
        </button>)}</div>
        {!matching.length && <p>{active.length ? "No active gates match the search." : "No shared gates fired at this anchor."}</p>}
      </section>
      {gate && <>
        <ResearchNotesEditor key={gate.id} targets={notesForGate(checkpoint, gate.id)} evidenceReference={evidenceReference} />
        {dependency.graph && dependency.targets ? <QueryGraphPanel
          key={`${anchor.anchor_id}/${gate.id}/${model}`} graph={dependency.graph} model={model} Graph={Graph} gateId={gate.id}
          annotationCheckpoint={checkpoint} evidenceReference={evidenceReference} traceTargets={dependency.targets}
          completeComponents={graphAsset?.data?.completeComponents} anchorLabel={anchor.final_query ? "FINAL QUERY" : "SELECTED TOKEN"}
          onExploreGate={(id) => { setSearch(""); setGate(id); }} />
          : <p className="ig-graph-status" role="status">{dependency.error ?? graphAsset?.error ?? (graphRef && !graphAsset?.data
            ? "Loading this token’s dependencies…" : "Dependencies are unmeasured at this exact anchor in this build.")}</p>}
        <ComponentBetaLocator key={gate.id} gate={gate.id} checkpoint={checkpoint} matrices={matrices} evidenceReference={evidenceReference} />
      </>}
    </>}
  </div>;
}
