import { useMemo, useState } from "react";
import { inductionGateDataset, type InductionCondition, type InductionContext, type InductionGate, type MeasuredInductionSplit } from "./inductionGateDataset";

const CONDITIONS: InductionCondition[] = ["induction", "counterfactual", "no_match", "repeated_no_association"];
const CONDITION_LABEL: Record<InductionCondition, string> = {
  induction: "Induction",
  counterfactual: "Counterfactual (+)",
  no_match: "No prior match",
  repeated_no_association: "Repeated / wrong association",
};
const MODEL_LABEL: Record<string, string> = { gemma: "Gemma 2", smol: "SmolLM2", granite: "Granite 3.3" };

export default function InductionGateExplorer({ onCircuitView }: { onCircuitView: (value: "natural" | "induction") => void }) {
  const discovery = inductionGateDataset.splits.discovery;
  const [splitName, setSplitName] = useState<"smoke" | "discovery">("smoke");
  const split = (splitName === "discovery" && discovery.status === "measured" ? discovery : inductionGateDataset.splits.smoke) as MeasuredInductionSplit;
  const [site, setSite] = useState<"ALL" | "PRE" | "MID">("ALL");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<"contrast" | "fires" | "beta">("contrast");
  const [selectedId, setSelectedId] = useState(split.gates[0]?.id ?? "");
  const gates = useMemo(() => {
    const normalized = query.trim().toUpperCase();
    const rows = split.gates.filter((gate) => (site === "ALL" || gate.site === site) && (!normalized || gate.id.includes(normalized)));
    return [...rows].sort((left, right) => {
      if (sort === "fires") return positiveFires(right) - positiveFires(left);
      if (sort === "beta") return maxBeta(right) - maxBeta(left);
      return Math.abs(right.contrast) - Math.abs(left.contrast);
    });
  }, [query, site, sort, split]);
  const selected = split.gates.find((gate) => gate.id === selectedId) ?? gates[0] ?? split.gates[0];
  const contexts = selected ? split.contexts[selected.id] ?? [] : [];
  const components = selected ? split.components[selected.id] ?? [] : [];

  function changeSplit(value: "smoke" | "discovery") {
    if (value === "discovery" && discovery.status !== "measured") return;
    const next = inductionGateDataset.splits[value] as MeasuredInductionSplit;
    setSplitName(value);
    setSelectedId(next.gates[0]?.id ?? "");
  }

  return <div className="uw-shell induction-shell">
    <header className="uw-header">
      <div><span className="uw-kicker">MULTI-MODEL ASPD · SHARED-GATE SCREENING</span><h1>Induction Gate Explorer</h1><p>Inspect natural shared-gate firing and imported β before selecting a targeted circuit analysis.</p></div>
      <CircuitSelect value="induction" onChange={onCircuitView} />
    </header>
    <div className="uw-warning"><strong>MEASURED · NO CIRCUIT CLAIM</strong>{inductionGateDataset.scientificBoundary}</div>
    <section className="ig-toolbar">
      <label>Split<select value={splitName} onChange={(event) => changeSplit(event.target.value as "smoke" | "discovery")}><option value="smoke">Smoke · 32 paired groups</option><option value="discovery" disabled={discovery.status !== "measured"}>Discovery {discovery.status === "measured" ? `· ${discovery.groups} paired groups` : "· running / not bundled"}</option></select></label>
      <label>Site<select value={site} onChange={(event) => setSite(event.target.value as typeof site)}><option>ALL</option><option>PRE</option><option>MID</option></select></label>
      <label>Sort<select value={sort} onChange={(event) => setSort(event.target.value as typeof sort)}><option value="contrast">|Induction − control|</option><option value="fires">Positive final-query fires</option><option value="beta">Maximum representative β</option></select></label>
      <label>Gate search<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="PRE:1386" /></label>
      <div className="ig-run-meta"><strong>{split.gates.length.toLocaleString()}</strong><span>active gates</span><strong>{split.pools}</strong><span>BatchTopK pools</span></div>
    </section>
    <main className="ig-layout">
      <aside className="ig-gate-list" aria-label="Shared gates"><header><strong>Shared gates</strong><span>showing {Math.min(250, gates.length)} / {gates.length}</span></header>{gates.slice(0, 250).map((gate) => <button className={gate.id === selected?.id ? "selected" : ""} key={gate.id} onClick={() => setSelectedId(gate.id)}><strong>{gate.id}</strong><span>Δ {signed(gate.contrast)}</span><small>{positiveFires(gate)} positive query fires · β {maxBeta(gate).toFixed(3)}</small></button>)}</aside>
      {selected && <section className="ig-detail">
        <header><div><span className="uw-eyebrow">SHARED GATE · {selected.site}</span><h2>{selected.id}</h2></div><div className="ig-badges"><span>Feature {selected.feature}</span><span>{contexts.length ? "Contexts exported" : "Context outside bounded top-gate export"}</span></div></header>
        <div className="ig-metrics"><Metric label="Final activation contrast" value={signed(selected.contrast)} /><Metric label="Mean selected gˢ" value={selected.meanActivation.toFixed(4)} /><Metric label="Positive query fires" value={positiveFires(selected).toString()} /><Metric label="Control query fires" value={controlFires(selected).toString()} /></div>
        <section className="ig-condition-grid">{CONDITIONS.map((condition) => <div key={condition}><strong>{CONDITION_LABEL[condition]}</strong><span>{selected.finalQueryFires[condition]} final-query fires</span><small>{selected.sequenceFires[condition]} sequences fire anywhere</small></div>)}</section>
        <BetaLocator gate={selected} components={components} />
        <section className="ig-examples"><header><div><span className="uw-eyebrow">PAIRED FULL-SEQUENCE CONTEXTS</span><h3>Where this shared gate fires</h3></div><p>Color intensity is measured continuous gˢ at exact shared canonical anchors. An outline marks the final query position.</p></header>{contexts.length ? contexts.map((context) => <ContextCard key={context.exampleId} context={context} />) : <div className="uw-empty">This gate is measurable in the sortable table, but full contexts are bounded to the top {split.contextGateIds.length} gates. Search a top-ranked gate to inspect sequences.</div>}</section>
      </section>}
    </main>
    <footer className="uw-footer"><span>Dataset: {split.split} · {split.groups} groups · {split.examples} conditions · exact 2,048-anchor BatchTopK pools.</span><span>Source {split.sourceRevision.slice(0, 12)}</span></footer>
  </div>;
}

export function CircuitSelect({ value, onChange }: { value: "natural" | "induction"; onChange: (value: "natural" | "induction") => void }) {
  return <label className="uw-circuit-select"><span>CIRCUIT / EVALUATION VIEW</span><select aria-label="Circuit or evaluation view" value={value} onChange={(event) => onChange(event.target.value as "natural" | "induction")}><option value="natural">Natural validation · measured dependencies</option><option value="induction">Induction · shared gates and β</option></select><small>Changing view does not reuse edges across datasets.</small></label>;
}

function positiveFires(gate: InductionGate) { return gate.finalQueryFires.induction + gate.finalQueryFires.counterfactual; }
function controlFires(gate: InductionGate) { return gate.finalQueryFires.no_match + gate.finalQueryFires.repeated_no_association; }
function maxBeta(gate: InductionGate) { return Math.max(...Object.values(gate.betaByModel).map((value) => value.max)); }
function signed(value: number) { return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`; }
function Metric({ label, value }: { label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong></div>; }

function BetaLocator({ gate, components }: { gate: InductionGate; components: Array<{ id: string; model: string; layer: number; matrix: string; beta: number | null; representative: boolean }> }) {
  if (!components.length) return <section className="ig-beta"><header><h3>β locator</h3><p>Detailed component cells are outside this bounded gate export.</p></header></section>;
  return <section className="ig-beta"><header><div><span className="uw-eyebrow">IMPORTED NATURAL-VALIDATION β</span><h3>Model-specific component locator</h3></div><p>β was not recomputed on induction prompts. White inset marks the frozen K_rep=3, τ_rep=0.07 representatives.</p></header><div className="ig-beta-models">{Object.entries(gate.betaByModel).map(([model, summary]) => <div key={model}><strong>{MODEL_LABEL[model] ?? model}</strong><span>max β {summary.max.toFixed(3)}</span><div className="ig-beta-grid">{components.filter((item) => item.model === model).map((item) => <i key={item.id} className={item.representative ? "representative" : ""} style={{ "--beta": Math.min(1, item.beta ?? 0) } as React.CSSProperties} title={`L${item.layer}/${item.matrix} · β ${item.beta?.toFixed(4) ?? "undefined"}`}><small>L{item.layer}</small><b>{item.matrix}</b></i>)}</div></div>)}</div></section>;
}

function ContextCard({ context }: { context: InductionContext }) {
  const maximum = Math.max(...context.positions.map((position) => position.activation), 1e-8);
  let start = 0;
  return <article className={`ig-context condition-${context.condition}`}><header><div><strong>{CONDITION_LABEL[context.condition]}</strong><span>{context.family} · {context.groupId}</span></div><code>expected{JSON.stringify(context.expectedContinuation)}</code></header><p className="ig-sequence">{context.positions.map((position) => { const text = context.prompt.slice(start, position.boundary); start = position.boundary; const opacity = position.activation > 0 ? .14 + .76 * position.activation / maximum : 0; return <span key={position.boundary} className={position.finalQuery ? "final-query" : ""} style={{ "--activation": opacity } as React.CSSProperties} title={`gˢ ${position.activation.toFixed(5)}${position.roles.length ? ` · ${position.roles.join(", ")}` : ""}`}>{text}</span>; })}</p><div className="ig-predictions">{Object.entries(context.nativePredictions).map(([model, prediction]) => <span key={model}><b>{MODEL_LABEL[model] ?? model}</b> {JSON.stringify(prediction.predicted_token)} · margin {prediction.logit_margin.toFixed(2)}</span>)}</div></article>;
}
