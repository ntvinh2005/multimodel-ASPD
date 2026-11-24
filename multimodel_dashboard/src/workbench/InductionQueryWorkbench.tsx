/// <reference types="vite/client" />

import { useEffect, useState, type ComponentProps, type ComponentType } from "react";
import { traceGraph } from "./graphQueries";
import { CircuitSelect, CONDITIONS, CONDITION_LABEL, type CircuitView } from "./InductionGateExplorer";
import type { InductionCondition } from "./inductionGateDataset";
import type { Component, DependencyEdge } from "./types";
import type { ModelGraph } from "./UniversalWorkbench";

type QueryComponent = Omit<Component, "representative">;
type ModelParents = {
  targets: Array<{
    component: QueryComponent;
    coefficient: number;
    parents: Array<{ component: QueryComponent; E: number; w: number }>;
  }>;
  strength: number;
  distribution: Record<string, number> | null;
};
type QueryGraph = {
  anchorId: string;
  activation: number;
  models: Record<string, ModelParents>;
  comparisons: Array<{ models: [string, string]; jaccard: number | null; total_variation: number | null }>;
};
type Example = {
  exampleId: string;
  groupId: string;
  family: string;
  condition: InductionCondition;
  prompt: string;
  queryStart: number;
  expected: string;
  poolId: string;
  prediction: Record<string, { predicted_token: string; logit_margin: number }>;
  modelPositions: Record<string, number>;
  graphs: Record<string, QueryGraph>;
};
type Bundle = {
  schemaVersion: 1;
  split: "discovery";
  provenance: { scope: string; kPar: number; tau: number; runtime: { slurm_job_id: string; git_sha: string } };
  selection: Record<string, { kind: "induction_contrast" | "fires_across_conditions"; groups: string[] }>;
  examples: Example[];
};
type GraphRenderer = ComponentType<ComponentProps<typeof ModelGraph>>;
const MODEL_LABELS = { gemma: "Gemma 2", smol: "SmolLM2", granite: "Granite 3.3" };
const MODELS = ["gemma", "smol", "granite"] as const;
const number = (value: number | null) => value === null ? "—" : value.toPrecision(4);

export default function InductionQueryWorkbench({ onCircuitView, Graph }: {
  onCircuitView: (view: CircuitView) => void;
  Graph: GraphRenderer;
}) {
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [loadError, setLoadError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch(`${import.meta.env.BASE_URL}inductionQueryGraphs.json`, {
          signal: controller.signal,
        });
        if (!response.ok || !response.headers.get("content-type")?.includes("json")) {
          throw new Error("This frontend build does not include the discovery graph artifact.");
        }
        const value: Bundle = await response.json();
        if (value.schemaVersion !== 1 || value.split !== "discovery"
            || value.provenance?.scope !== "final_query_same_anchor_frozen_discovery_support"
            || !value.examples?.length || !Object.keys(value.selection).length) {
          throw new Error("The discovery graph artifact has an unsupported format.");
        }
        setBundle(value);
      } catch (error) {
        if (!controller.signal.aborted) setLoadError(error instanceof Error ? error.message : String(error));
      }
    }
    void load();
    return () => controller.abort();
  }, []);
  return <div className="uw-shell">
    <header className="uw-header">
      <div>
        <span className="uw-kicker">MULTI-MODEL ASPD · DISCOVERY</span>
        <h1>Induction at the final query</h1>
        <p>Measured same-anchor dependencies into representative components at the prediction position.</p>
      </div>
      <CircuitSelect value="induction-query" onChange={onCircuitView} />
    </header>
    {bundle
      ? <QueryPanel bundle={bundle} Graph={Graph} />
      : <p className="uw-empty" role="status">{loadError || "Loading measured discovery graphs…"}</p>}
  </div>;
}

function QueryPanel({ bundle, Graph }: { bundle: Bundle; Graph: GraphRenderer }) {
  const gates = Object.keys(bundle.selection).sort();
  const [gate, setGate] = useState(
    gates.find((id) => bundle.selection[id].kind === "induction_contrast") ?? gates[0],
  );
  const [group, setGroup] = useState("");
  const [condition, setCondition] = useState<InductionCondition>("induction");
  const [model, setModel] = useState<(typeof MODELS)[number]>("gemma");
  const gateExamples = bundle.examples.filter((example) => gate in example.graphs);
  const groups = bundle.selection[gate].groups;
  const groupId = groups.includes(group) ? group : groups[0];
  const example = gateExamples.find((item) => item.groupId === groupId && item.condition === condition);
  const graph = example?.graphs[gate];
  const prompt = Array.from(example?.prompt ?? "");
  return <>
    <div className="uw-warning">
      <strong>MEASURED · DISCOVERY SHORTLIST</strong>
      Parent candidates include all naturally active upstream components at this anchor. Each representative shows up to {bundle.provenance.kPar} parents.
    </div>
    <section className="ig-toolbar iq-toolbar">
      <label>Gate<select value={gate} onChange={(event) => { setGate(event.target.value); setGroup(""); }}>
        {gates.map((id) => <option key={id} value={id}>{id} · {bundle.selection[id].kind === "induction_contrast" ? "induction contrast" : "fires across conditions"}</option>)}
      </select></label>
      <label>Example group<select value={groupId} onChange={(event) => setGroup(event.target.value)}>
        {groups.map((id) => <option key={id} value={id}>{id} · {gateExamples.find((item) => item.groupId === id)?.family.replaceAll("_", " ")}</option>)}
      </select></label>
      <label>Condition<select value={condition} onChange={(event) => setCondition(event.target.value as InductionCondition)}>
        {CONDITIONS.map((id) => <option key={id} value={id}>{CONDITION_LABEL[id]}</option>)}
      </select></label>
      <label>Model<select value={model} onChange={(event) => setModel(event.target.value as typeof model)}>
        {MODELS.map((id) => <option key={id} value={id}>{MODEL_LABELS[id]}</option>)}
      </select></label>
    </section>
    {example && graph ? <>
      <article className="ig-context iq-context">
        <header><strong>{CONDITION_LABEL[condition]}</strong><span>{example.family.replaceAll("_", " ")} · {example.poolId}</span></header>
        <p className="ig-sequence">{prompt.slice(0, example.queryStart).join("")}<span className="final-query">{prompt.slice(example.queryStart).join("")}</span></p>
        <div className="ig-predictions">
          <span><b>{condition === "induction" || condition === "counterfactual" ? "Expected" : "Contrast target"}</b> {JSON.stringify(example.expected)}</span>
          <span><b>Shared gˢ</b> {number(graph.activation)}</span>
          {MODELS.map((id) => <span key={id}><b>{MODEL_LABELS[id]}</b> {JSON.stringify(example.prediction[id]?.predicted_token)} · margin {number(example.prediction[id]?.logit_margin ?? null)}</span>)}
        </div>
      </article>
      {graph.activation === 0 ? <p className="uw-empty" role="status">{gate} did not fire at this query in the original BatchTopK pool.</p> :
        <QueryGraphPanel key={`${gate}/${example.exampleId}/${model}`} graph={graph} model={model} Graph={Graph} />}
      <footer className="uw-footer"><span>Anchor {graph.anchorId} · {MODEL_LABELS[model]} token index {example.modelPositions[model]}</span><span>Job {bundle.provenance.runtime.slurm_job_id} · {bundle.provenance.runtime.git_sha.slice(0, 12)}</span></footer>
    </> : <p className="uw-empty">No stored graph record for this selection.</p>}
  </>;
}

function QueryGraphPanel({ graph, model, Graph }: {
  graph: QueryGraph;
  model: (typeof MODELS)[number];
  Graph: GraphRenderer;
}) {
  const [selectedComponent, setComponent] = useState<string | null>(null);
  const [selectedEdge, setEdge] = useState<string | null>(null);
  const parents = graph.models[model];
  if (!parents) return <p className="uw-empty">Missing graph artifact for {MODEL_LABELS[model]}.</p>;
  const components = new Map<string, Component>();
  const coefficients = new Map<string, number>();
  const edges: DependencyEdge[] = [];
  for (const target of parents.targets) {
    const id = target.component.id;
    components.set(id, { ...target.component, representative: true });
    coefficients.set(id, target.coefficient);
    for (const parent of target.parents) {
      const source = parent.component.id;
      if (!components.has(source)) components.set(source, { ...parent.component, representative: false });
      edges.push({
        id: `${source}→${id}`, modelId: model, anchorId: graph.anchorId, source, target: id,
        signedEffect: parent.E, normalizedMagnitude: parent.w,
        upstreamCoefficient: null, gradient: null, normalization: "abs(E) × norm(u) / σ",
        evidence: "measured-taylor",
      });
    }
  }
  const colors = Object.fromEntries([...new Set([...components.values()].map((c) => c.gateId))].map((gate) => {
    const [site, feature] = gate.split(":");
    const hue = (Number(feature) * 137.508 + (site === "MID" ? 67 : 0)) % 360;
    return [gate, `hsl(${hue} 60% 69%)`];
  }));
  const trace = selectedComponent ? traceGraph(edges, selectedComponent, "both", 1) : null;
  const node = selectedComponent ? components.get(selectedComponent) : null;
  const edge = edges.find((item) => item.id === selectedEdge);
  return <main className="uw-main iq-main">
    <section className="uw-graph-area">
      <div className="uw-graph-toolbar">
        <button onClick={() => { setComponent(null); setEdge(null); }}>Restore graph</button>
        <span>{components.size} components · {edges.length} stored parents</span>
      </div>
      <Graph modelId={model} components={[...components.values()]} edges={edges}
        spotlight={null} selectedComponent={selectedComponent} selectedEdge={selectedEdge}
        trace={trace} colors={colors} direction="both" widthScale="sqrt"
        onComponent={(id) => { setComponent(id); setEdge(null); }}
        onEdge={(id) => { setEdge(id); setComponent(null); }} />
      {!edges.length && <p className="uw-empty">No parent passed the strict |E| threshold at this anchor.</p>}
    </section>
    <aside className="uw-inspector uw-detail">
      <span className="uw-eyebrow">FINAL QUERY · SAME-ANCHOR PARENTS</span>
      <h2>{edge ? "Selected dependency" : node?.gateId ?? "Parent summary"}</h2>
      {edge && <dl><dt>From</dt><dd>{edge.source}</dd><dt>To</dt><dd>{edge.target}</dd><dt>Signed E</dt><dd>{number(edge.signedEffect)}</dd><dt>Normalized w</dt><dd>{number(edge.normalizedMagnitude)}</dd></dl>}
      {node && <dl><dt>Component</dt><dd>{node.id}</dd><dt>Reference β</dt><dd>{number(node.beta)}</dd><dt>Coefficient e</dt><dd>{number(coefficients.get(node.id) ?? null)}</dd></dl>}
      <p>Parent strength M = <b>{number(parents.strength)}</b></p>
      {parents.distribution && <dl>{Object.entries(parents.distribution).sort((left, right) => right[1] - left[1]).map(([id, value]) =>
        <DistributionRow key={id} gate={id} value={value} />
      )}</dl>}
      <p>Cross-model comparisons at this anchor</p>
      <dl>{graph.comparisons.map((pair) => <ComparisonRow key={pair.models.join("/")} pair={pair} />)}</dl>
    </aside>
  </main>;
}

function DistributionRow({ gate, value }: { gate: string; value: number }) {
  return <><dt>{gate}</dt><dd>π {number(value)}</dd></>;
}

function ComparisonRow({ pair }: { pair: QueryGraph["comparisons"][number] }) {
  return <><dt>{pair.models.join(" / ")}</dt><dd>J {number(pair.jaccard)} · TV {number(pair.total_variation)}</dd></>;
}
