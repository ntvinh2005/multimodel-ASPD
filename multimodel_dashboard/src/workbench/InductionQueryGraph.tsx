/// <reference types="vite/client" />

import { useEffect, useState, type ComponentProps, type ComponentType } from "react";
import { edgeThresholdScale, formatEdgeMagnitude } from "./displayUtils";
import { edgesForAnchor, traceGraph } from "./graphQueries";
import { CONDITIONS, CONDITION_LABEL, inductionCheckpointManifest, type InductionCondition } from "./inductionGateDataset";
import { ResearchNotesEditor } from "./ResearchNotes";
import { notesForSelection } from "./researchNotes";
import type { Component, DependencyEdge } from "./types";
import type { ModelGraph } from "./UniversalWorkbench";

type QueryComponent = Omit<Component, "representative">;
export type QueryTarget = {
  component: QueryComponent;
  coefficient: number;
  contributionNorm?: number;
  outputScale?: number;
  parents: Array<{ component: QueryComponent; E: number; w: number }>;
};
type ModelParents = {
  targets: QueryTarget[];
  status?: "measured" | "no_reference_representatives";
  strength: number | null;
  distribution: Record<string, number> | null;
  scores?: Record<string, number> | null;
};
export type QueryGraph = {
  anchorId: string;
  activation: number;
  models: Record<string, ModelParents>;
  comparisons: Array<{ models: [string, string]; jaccard: number | null; total_variation: number | null; available?: boolean }>;
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
  provenance: {
    scope: string;
    checkpointManifestSha256: string;
    discoveryManifestSha256: string;
    kPar: number;
    runtime: { slurm_job_id: string; git_sha: string };
  };
  selection: Record<string, { kind: "induction_contrast" | "fires_across_conditions"; groups: string[] }>;
  examples: Example[];
};
export type GraphRenderer = ComponentType<ComponentProps<typeof ModelGraph>>;
const MODEL_LABELS = { gemma: "Gemma 2", smol: "SmolLM2", granite: "Granite 3.3" };
const MODELS = ["gemma", "smol", "granite"] as const;
const number = (value: number | null) => value === null ? "—" : value.toPrecision(4);

export function useInductionQueryGraphs(discoveryManifest: string | null) {
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [loadError, setLoadError] = useState("");
  useEffect(() => {
    if (!discoveryManifest || bundle || loadError) return;
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
            || value.provenance.discoveryManifestSha256 !== discoveryManifest
            || (inductionCheckpointManifest && value.provenance.checkpointManifestSha256 !== inductionCheckpointManifest)
            || !value.examples?.length || !Object.keys(value.selection).length) {
          throw new Error("The graph artifact does not match this discovery dataset.");
        }
        setBundle(value);
      } catch (error) {
        if (!controller.signal.aborted) setLoadError(error instanceof Error ? error.message : String(error));
      }
    }
    void load();
    return () => controller.abort();
  }, [discoveryManifest, bundle, loadError]);
  return { bundle, loadError };
}

export default function InductionQueryGraph({ gate, bundle, Graph }: {
  gate: string;
  bundle: Bundle;
  Graph: GraphRenderer;
}) {
  const [group, setGroup] = useState("");
  const [condition, setCondition] = useState<InductionCondition>("induction");
  const [model, setModel] = useState<(typeof MODELS)[number]>("gemma");
  const gateExamples = bundle.examples.filter((example) => gate in example.graphs);
  const groups = bundle.selection[gate].groups;
  const groupId = groups.includes(group) ? group : groups[0];
  const example = gateExamples.find((item) => item.groupId === groupId && item.condition === condition);
  const graph = example?.graphs[gate];
  const prompt = Array.from(example?.prompt ?? "");
  return <section className="ig-examples iq-section" aria-labelledby="induction-query-heading">
    <header>
      <div><span className="uw-eyebrow">MEASURED · FINAL QUERY</span><h3 id="induction-query-heading">Dependency graph</h3></div>
      <p>Same-anchor dependencies into {gate} representatives. Up to {bundle.provenance.kPar} parents per representative.</p>
    </header>
    <section className="ig-toolbar iq-toolbar">
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
        <QueryGraphPanel key={`${gate}/${example.exampleId}/${model}`} graph={graph} model={model} Graph={Graph}
          gateId={gate} annotationCheckpoint={bundle.provenance.checkpointManifestSha256}
          evidenceReference={`Induction · ${example.exampleId} · anchor ${graph.anchorId}`} />}
      <footer className="uw-footer"><span>Anchor {graph.anchorId} · {MODEL_LABELS[model]} token index {example.modelPositions[model]}</span><span>Job {bundle.provenance.runtime.slurm_job_id} · {bundle.provenance.runtime.git_sha.slice(0, 12)}</span></footer>
    </> : <p className="uw-empty">No stored graph record for this selection.</p>}
  </section>;
}

export function QueryGraphPanel({ graph, model, Graph, onExploreGate, traceTargets, gateId, annotationCheckpoint, evidenceReference,
  completeComponents = false, anchorLabel = "FINAL QUERY" }: {
  graph: QueryGraph;
  model: (typeof MODELS)[number];
  Graph: GraphRenderer;
  onExploreGate?: (gate: string) => void;
  traceTargets?: ReadonlyMap<string, { target: QueryTarget; representative: boolean }>;
  gateId: string;
  annotationCheckpoint?: string;
  evidenceReference?: string;
  completeComponents?: boolean;
  anchorLabel?: string;
}) {
  const [selectedComponent, setComponent] = useState<string | null>(null);
  const [selectedEdge, setEdge] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0);
  const [expanded, setExpanded] = useState<string[]>([]);
  const [allGateComponents, setAllGateComponents] = useState(false);
  const parents = graph.models[model];
  if (!parents) return <p className="uw-empty">Missing graph artifact for {MODEL_LABELS[model]}.</p>;
  const hasRepresentatives = parents.status !== "no_reference_representatives";
  if (!hasRepresentatives && !completeComponents) {
    return <p className="uw-empty">This gate fired but {MODEL_LABELS[model]} has no frozen β representative for it.</p>;
  }
  const components = new Map<string, Component>();
  const coefficients = new Map<string, number>();
  const measuredNorms = new Map<string, { contribution: number; scale?: number }>();
  const edges: DependencyEdge[] = [];
  const officialIds = new Set(parents.targets.map((target) => target.component.id));
  const displayedIds = new Set([...officialIds, ...expanded]);
  if (completeComponents && (allGateComponents || !hasRepresentatives)) {
    for (const [id, value] of traceTargets ?? []) if (value.target.component.gateId === gateId) displayedIds.add(id);
  }
  const targets = [
    ...parents.targets.map((target) => ({ target, representative: true })),
    ...[...displayedIds].flatMap((id) => {
      const entry = traceTargets?.get(id);
      return entry && !officialIds.has(id) ? [entry] : [];
    }),
  ];
  for (const { target, representative } of targets) {
    const id = target.component.id;
    components.set(id, { ...target.component, representative });
    coefficients.set(id, target.coefficient);
    if (target.contributionNorm !== undefined) {
      measuredNorms.set(id, { contribution: target.contributionNorm, scale: target.outputScale });
    }
    for (const parent of target.parents) {
      const source = parent.component.id;
      if (!components.has(source)) {
        components.set(source, { ...parent.component, representative: traceTargets?.get(source)?.representative ?? false });
      }
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
  const thresholdScale = edgeThresholdScale(edges);
  const visibleEdges = edgesForAnchor(edges, graph.anchorId, new Set(components.keys()), threshold);
  const trace = selectedComponent ? traceGraph(visibleEdges, selectedComponent, "both", 1) : null;
  const node = selectedComponent ? components.get(selectedComponent) : null;
  const norm = selectedComponent ? measuredNorms.get(selectedComponent) : undefined;
  const edge = visibleEdges.find((item) => item.id === selectedEdge);
  const isUpstream = node && edges.some((item) => item.source === node.id);
  const targetRecorded = node && traceTargets?.has(node.id);
  const edgeTrace = edge && !officialIds.has(edge.target) ? traceTargets?.get(edge.target) : undefined;
  const children = node && completeComponents ? [...(traceTargets ?? [])].flatMap(([id, record]) => {
    if (displayedIds.has(id)) return [];
    const incoming = record.target.parents.find((parent) => parent.component.id === node.id);
    return incoming ? [{ id, weight: incoming.w }] : [];
  }).sort((a, b) => b.weight - a.weight || a.id.localeCompare(b.id)) : [];
  return <div className="uw-main iq-main">
    <section className="uw-graph-area">
      <div className="uw-graph-toolbar">
        <button onClick={() => { setComponent(null); setEdge(null); setThreshold(0); setExpanded([]); }}>Restore graph</button>
        {completeComponents && hasRepresentatives && <label>
          <input type="checkbox" checked={allGateComponents} onChange={(event) => setAllGateComponents(event.target.checked)} /> All components for {gateId}
        </label>}
        <span>{components.size} components · {visibleEdges.length}/{edges.length} parent edges shown</span>
      </div>
      <div className="iq-edge-filter">
        <label className="uw-threshold-label">
          <span>Minimum normalized edge weight <i>w</i></span>
          <output>{formatEdgeMagnitude(threshold)}</output>
          <input
            aria-label="Minimum normalized edge weight"
            aria-valuetext={formatEdgeMagnitude(threshold)}
            type="range" min="0" max={thresholdScale.maximum || thresholdScale.step}
            step={Math.min(thresholdScale.maximum || thresholdScale.step, thresholdScale.step)}
            value={threshold} disabled={thresholdScale.maximum === 0}
            onChange={(event) => setThreshold(Number(event.target.value))}
          />
        </label>
        <small>Range 0–{formatEdgeMagnitude(thresholdScale.maximum)} for this example and model.</small>
      </div>
      <Graph modelId={model} components={[...components.values()]} edges={visibleEdges}
        spotlight={null} selectedComponent={selectedComponent} selectedEdge={edge?.id ?? null}
        annotationCheckpoint={annotationCheckpoint}
        trace={trace} colors={colors} direction="both" widthScale="sqrt"
        onComponent={(id) => { setComponent(id); setEdge(null); }}
        onEdge={(id) => { setEdge(id); setComponent(null); }} />
      {!visibleEdges.length && <p className="uw-empty">{edges.length
        ? "The display threshold hides all recorded edges."
        : "No parent passed the strict |E| threshold at this anchor."}</p>}
    </section>
    <aside className="uw-inspector uw-detail">
      <span className="uw-eyebrow">{anchorLabel} · SAME-ANCHOR PARENTS</span>
      <h2>{edge ? "Selected dependency" : node?.gateId ?? "Parent summary"}</h2>
      {edge && <dl><dt>From</dt><dd>{edge.source}</dd><dt>To</dt><dd>{edge.target}</dd><dt>Signed E</dt><dd>{number(edge.signedEffect)}</dd><dt>Normalized w</dt><dd>{number(edge.normalizedMagnitude)}</dd></dl>}
      {edgeTrace && <p>{edgeTrace.representative ? "Expanded representative dependency" : "Supplemental component dependency"}</p>}
      {node && <dl><dt>Component</dt><dd>{node.id}</dd><dt>Reference β</dt><dd>{number(node.beta)}</dd><dt>Coefficient e</dt><dd>{number(coefficients.get(node.id) ?? null)}</dd></dl>}
      {norm && <dl><dt>Contribution ‖Δy‖</dt><dd>{number(norm.contribution)}</dd><dt>Reference σ</dt><dd>{number(norm.scale ?? null)}</dd></dl>}
      {annotationCheckpoint && <ResearchNotesEditor
        key={node?.id ?? gateId}
        targets={notesForSelection(annotationCheckpoint, node?.gateId ?? gateId, model, node ?? undefined)}
        evidenceReference={[evidenceReference ?? `Anchor ${graph.anchorId}`, MODEL_LABELS[model], node?.id, edge?.id].filter(Boolean).join(" · ")}
        openByDefault={Boolean(node)}
      />}
      {node && onExploreGate && <button type="button" onClick={() => onExploreGate(node.gateId)}>Explore {node.gateId} on this example</button>}
      {node && isUpstream && traceTargets && !displayedIds.has(node.id)
        && (targetRecorded ? <button type="button" onClick={() => setExpanded([...expanded, node.id])}>
          Expand this component’s measured parents
        </button> : <p>No parent measurement is cached for this exact component and anchor.</p>)}
      {children.length > 0 && <button type="button" onClick={() => setExpanded([...expanded, ...children.slice(0, 12).map((item) => item.id)])}>
        Expand measured children ({Math.min(12, children.length)} of {children.length})
      </button>}
      {(expanded.length > 0 || allGateComponents) && <p>The summary below uses the selected gate’s representatives. Expanded component edges are shown in the graph.</p>}
      {hasRepresentatives ? <p>Parent strength M = <b>{number(parents.strength)}</b> · stored parent set</p>
        : <p>Representative parent distribution unavailable for {MODEL_LABELS[model]}. Exact component measurements appear in the graph.</p>}
      {parents.distribution && <dl>{Object.entries(parents.distribution).sort((left, right) => right[1] - left[1]).map(([id, value]) =>
        <DistributionRow key={id} gate={id} value={value} score={parents.scores?.[id]} onExploreGate={onExploreGate} />
      )}</dl>}
      {hasRepresentatives && !parents.distribution && <p>Parent distribution undefined at zero total strength.</p>}
      <p>Cross-model comparisons at this anchor</p>
      <dl>{graph.comparisons.map((pair) => <ComparisonRow key={pair.models.join("/")} pair={pair} />)}</dl>
    </aside>
  </div>;
}

function DistributionRow({ gate, value, score, onExploreGate }: {
  gate: string;
  value: number;
  score?: number;
  onExploreGate?: (gate: string) => void;
}) {
  return <><dt>{onExploreGate ? <button type="button" onClick={() => onExploreGate(gate)}>{gate}</button> : gate}</dt>
    <dd>π {number(value)}{score !== undefined && <> · S {number(score)}</>}</dd></>;
}

function ComparisonRow({ pair }: { pair: QueryGraph["comparisons"][number] }) {
  return <><dt>{pair.models.join(" / ")}</dt><dd>{pair.available === false
    ? "Unavailable · missing β representatives"
    : `J ${number(pair.jaccard)} · TV ${number(pair.total_variation)}`}</dd></>;
}
