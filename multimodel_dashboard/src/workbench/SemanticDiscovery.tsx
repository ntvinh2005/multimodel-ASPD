import { useMemo, useState, type CSSProperties, type ReactNode } from "react";
import type { GraphRenderer } from "./InductionQueryGraph";
import { componentIdentity, type Measurement } from "./circuitComposition";
import { edgeThresholdScale, formatEdgeMagnitude } from "./displayUtils";
import { traceGraph } from "./graphQueries";
import { MODEL_IDS, MODEL_NAMES, type EvalModel } from "./negationWorkbenchData";
import { ResearchNotesEditor } from "./ResearchNotes";
import { notesForSelection } from "./researchNotes";
import { useWorkbenchAsset } from "./TokenCircuitExplorer";
import type { DependencyEdge } from "./types";
import { SemanticDiscoveryGuide } from "./SemanticDiscoveryGuide";

export type Anchor = {
  anchor: string; prefix: string; pool: number; source: string; document: string;
  positions: Record<EvalModel, number>;
  native?: Record<EvalModel, [number, number, number, number]>;
};
type Values = { ot: number; centroid: number; anchorOverlap: number };
type Pair = {
  left: string; right: string;
  discovery: Comparison; heldout: Comparison;
};
type Comparison = {
  suffixRemovedOt?: number | null; suffixRemovedControlOt?: number | null;
  measures: Values | null; controlMeasures: Values | null; controlComponent: string | null;
  transport: Array<{ left: string; right: string; mass: number; cost: number; kind: string }>;
};
export type Witness = {
  anchor: string; component: string; model: EvalModel; artifact: string;
  edges: Array<{ source: string; target: string; E: number; w: number }>;
};
type PathRecord = {
  left: string; right: string; leftCluster: string; rightCluster: string; discovery: number; heldout: number;
  witnesses: Array<Omit<Witness, "component"> & { split: Split }>;
};
type PathCatalog = { checkpoint: string; dataset: string; catalogSha256: string; pairs: PathRecord[] };
type GraphExample = { left: string; right: string; kind: "shared" | "transport" | "contrast" | "separate" };
type GraphComparison = { sharedSupport: number; examples: GraphExample[] };
type PairGraphCatalog = {
  schemaVersion: number; checkpoint: string; dataset: string; catalogSha256: string; graphIndexSha256: string;
  pairs: Record<string, Record<Split, GraphComparison>>;
  graphs: Record<string, Witness & { pool: number }>;
};
type Cluster = { id: string; components: string[]; models: EvalModel[]; links: Pair[]; graphs: Witness[] };
type Totals = { measuredNonzero: number; normalizedMass: number; positiveMass: number; negativeMass: number };
type SemanticCatalog = {
  schemaVersion: number; dataset: string; provenance: { checkpoint: string; split: string; graphIndexSha256: string };
  summary: { selected: Record<EvalModel, number>; scoredDiscoveryPairs: number; crossModelClusters: number };
  clusters: Cluster[];
  anchors: Record<string, Anchor>;
  components: Record<string, {
    gate: string; model: EvalModel; totals: Partial<Record<Split, Totals>>;
    examples: Partial<Record<Split, Array<{ anchor: string; coefficient: number; normalizedMagnitude: number }>>>;
  }>;
  structuralOverlaps: Array<{ cluster: string; structural: string; components: string[] }>;
};
type Split = "discovery" | "heldout";
type ComparisonScope = "all" | "cross-model" | "three-models" | `${EvalModel}/${EvalModel}`;

export function matchesSemanticPair(left: EvalModel, right: EvalModel, scope: ComparisonScope) {
  if (scope === "all") return true;
  if (scope === "cross-model" || scope === "three-models") return left !== right;
  return scope === `${left}/${right}` || scope === `${right}/${left}`;
}

function show(value?: number | null) {
  return value == null ? "Unavailable" : value.toPrecision(3);
}

function Prefix({ row, model }: { row: Anchor; model: EvalModel }) {
  const end = row.prefix.match(/[^\p{L}\p{N}_\s]+\s*$|[\p{L}\p{N}_]+\s*$/u);
  const boundary = end?.index ?? Math.max(0, row.prefix.length - 1);
  const native = row.native?.[model];
  const codepoints = native && Array.from(row.prefix);
  const before = codepoints && native ? codepoints.slice(0, native[2]).join("") : row.prefix.slice(0, boundary);
  const token = codepoints && native ? codepoints.slice(native[2], native[3]).join("") : row.prefix.slice(boundary);
  return <div>
    <p className="ig-sequence td-sequence">{before}
      <mark title="The aligned anchor ends here">{token}</mark>
    </p>
    <small>{row.source} · {MODEL_NAMES[model]} token {row.positions[model]} · pool {row.pool}</small>
  </div>;
}

function dependencyEdges(witness: Witness): DependencyEdge[] {
  return witness.edges.map(record => ({
    id: `${witness.anchor}/${record.source}/${record.target}`, anchorId: witness.anchor,
    modelId: witness.model, source: record.source, target: record.target,
    signedEffect: record.E, normalizedMagnitude: record.w, upstreamCoefficient: null,
    gradient: null, normalization: "Stored parent weight", evidence: "measured-taylor",
  }));
}

function ContextGraph({ witness, row, checkpoint, Graph, hidePrefix, threshold = 0, colors, onComparisonNote, visibleSources, markedSources, onPatternEdge }: {
  witness: Witness; row: Anchor; checkpoint: string; Graph: GraphRenderer;
  hidePrefix?: boolean; threshold?: number; colors?: Record<string, string>;
  visibleSources?: ReadonlySet<string>;
  markedSources?: ReadonlySet<string>;
  onPatternEdge?: (source: string, target: string) => void;
  onComparisonNote?: (componentId: string | null) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const edges = dependencyEdges(witness);
  const focused = edges.filter(edge => !visibleSources || visibleSources.has(edge.source));
  const ids = [...new Set([witness.component, ...focused.flatMap(edge => [edge.source, edge.target])])];
  const components = ids.map(id => ({ ...componentIdentity(id), beta: null, rho: null, representative: false }));
  const visible = focused.filter(edge => (edge.normalizedMagnitude ?? 0) >= threshold);
  const measurement: Measurement = {
    scope: { checkpoint, dataset: "natural-validation", anchor: witness.anchor, example: row.document,
      pool: String(row.pool), artifact: witness.artifact },
    complete: false, targets: [...new Set(edges.map(edge => edge.target))],
    edges: witness.edges.map(edge => ({ source: edge.source, target: edge.target, signed: edge.E, weight: edge.w })),
  };
  const node = components.find(component => component.id === selected);
  const edge = visible.find(record => record.id === selectedEdge);
  const markedEdges = markedSources && new Set(
    visible.filter(record => markedSources.has(record.source)).map(record => record.id),
  );
  return <div>
    {!hidePrefix && <Prefix row={row} model={witness.model} />}
    <Graph modelId={witness.model} components={components} edges={visible}
      spotlight={componentIdentity(witness.component).gateId} selectedComponent={selected} selectedEdge={edge?.id ?? null}
      trace={selected ? traceGraph(visible, selected, "both", 1) : null} direction="both" widthScale="sqrt"
      onComponent={id => { setSelected(id); setSelectedEdge(null); onComparisonNote?.(id); }}
      onEdge={id => {
        setSelected(null);
        setSelectedEdge(id);
        onComparisonNote?.(null);
        const picked = visible.find(record => record.id === id);
        if (picked) onPatternEdge?.(picked.source, picked.target);
      }}
      markedEdges={markedEdges} pickPatternEdges={Boolean(onPatternEdge)}
      colors={colors} annotationCheckpoint={checkpoint} compositionMeasurement={measurement} />
    {edge && <p><code>{edge.source} → {edge.target}</code> · E {show(edge.signedEffect)} · w {show(edge.normalizedMagnitude)}</p>}
    {node && !onComparisonNote && <ResearchNotesEditor key={node.id} targets={notesForSelection(checkpoint, node.gateId, node.modelId, node)}
      evidenceReference={`Semantic Discovery · ${row.anchor}`} />}
  </div>;
}

function graphExampleLabel({ left, right, kind }: GraphExample) {
  if (left === right) return "Same aligned token";
  if (kind === "transport") return "OT-matched contexts";
  if (kind === "contrast") return "Less similar OT contexts";
  return "Separate measured contexts";
}

type GraphPanel = {
  model: EvalModel; component?: string; anchor?: string; witness?: Witness; controls?: ReactNode;
  visibleSources?: ReadonlySet<string>;
  markedSources?: ReadonlySet<string>;
};
type GraphView = { anchors: SemanticCatalog["anchors"]; checkpoint: string; Graph: GraphRenderer };

export function ComparisonPanels({ panels, anchors, checkpoint, Graph, scopeLabel, onAnnotation, onPatternEdge }: GraphView & {
  panels: GraphPanel[];
  scopeLabel?: string;
  onAnnotation?: (component: string | null, anchor: string) => void;
  onPatternEdge?: (source: string, target: string) => void;
}) {
  const [threshold, setThreshold] = useState(0);
  const [annotation, setAnnotation] = useState<{ componentId: string; anchor: string } | null>(null);
  const allEdges = panels.flatMap(panel => panel.witness ? dependencyEdges(panel.witness) : []);
  const scale = edgeThresholdScale(allEdges);
  const keys = [...new Set(allEdges.flatMap(edge => [edge.source, edge.target].map(id => componentIdentity(id).gateId)))].sort();
  const colors = Object.fromEntries(keys.map((gate, index) => [gate, `hsl(${index * 137.5} 54% 67%)`]));
  const sharedAnchor = panels[0]?.anchor && panels.every(panel => panel.anchor === panels[0].anchor) ? panels[0].anchor : undefined;
  const node = annotation ? componentIdentity(annotation.componentId) : undefined;

  return <>
    <p>{sharedAnchor ? "Same text and aligned token in all panels." : "Each panel shows its own text and token."}
      {" "}{scopeLabel ?? "Scope: saved edges touching the selected component."}</p>
    {sharedAnchor && <Prefix row={anchors[sharedAnchor]} model={panels[0].model} />}
    <div className="uw-edge-filter">
      <label className="uw-threshold-label">
        <span>Minimum edge weight across graphs</span><output>{formatEdgeMagnitude(threshold)}</output>
        <input type="range" min="0" max={scale.maximum || scale.step} step={scale.step} value={threshold}
          disabled={!scale.maximum} aria-label="Minimum edge weight across graphs"
          onChange={event => setThreshold(Number(event.target.value))} />
      </label>
    </div>
    <div className="semantic-graph-panels" style={{ "--graph-count": panels.length } as CSSProperties}>
      {panels.map(({ component, anchor, model, witness, controls, visibleSources, markedSources }, i) => {
        const row = anchor ? anchors[anchor] : undefined;
        return <section key={model + "/" + i} aria-label={`Dependency graph for ${MODEL_NAMES[model]}`}>
          <strong>{MODEL_NAMES[model]}</strong>{component && <>{" "}<code>{component}</code></>}
          {controls}
          {row && anchor && <small className="semantic-graph-anchor" title={anchor}>Token {row.positions[model]} · {anchor.slice(0, 12)}
            {" · "}{!witness ? "Graph unavailable" : witness.edges.length === 0 ? "Measured · no parent passed the rule"
              : `${witness.edges.filter(edge => edge.w >= threshold && (!visibleSources || visibleSources.has(edge.source))).length}/${witness.edges.length} edges shown`}</small>}
          {witness && row && anchor ? <ContextGraph witness={witness} row={row} checkpoint={checkpoint} Graph={Graph}
            hidePrefix={!!sharedAnchor} threshold={threshold} colors={colors} visibleSources={visibleSources}
            markedSources={markedSources} onPatternEdge={onPatternEdge}
            onComparisonNote={componentId => {
              if (onAnnotation) onAnnotation(componentId, anchor);
              else setAnnotation(componentId ? { componentId, anchor } : null);
            }} />
            : <>{row && !sharedAnchor && <Prefix row={row} model={model} />}<p>No saved graph for this selection.</p></>}
        </section>;
      })}
    </div>
    {node && annotation && <ResearchNotesEditor key={node.id}
      targets={notesForSelection(checkpoint, node.gateId, node.modelId, node)}
      evidenceReference={`Semantic Discovery · ${annotation.anchor}`} />}
  </>;
}

function hasBothGraphs(pair: Pair, example: GraphExample, graphs: PairGraphCatalog["graphs"]) {
  return Boolean(graphs[`${pair.left}|${example.left}`] && graphs[`${pair.right}|${example.right}`]);
}

function PairGraphComparison({ pair, comparison, graphs, ...view }: GraphView & {
  pair: Pair; comparison?: GraphComparison; graphs: PairGraphCatalog["graphs"];
}) {
  const [position, setPosition] = useState(() => Math.max(0,
    comparison?.examples.findIndex(row => hasBothGraphs(pair, row, graphs)) ?? 0,
  ));
  const example = comparison?.examples[position];
  if (!example || !comparison) return <p>No paired graph records in this split. Check all models in the group for other saved graphs.</p>;
  const panels = [
    { component: pair.left, anchor: example.left },
    { component: pair.right, anchor: example.right },
  ].map(({ component, anchor }) => ({
    component, anchor, model: componentIdentity(component).modelId as EvalModel, witness: graphs[`${component}|${anchor}`],
  }));
  return <>
    <label className="discovered-picker">Graph example<select className="uw-form-control" value={position}
      onChange={event => setPosition(Number(event.target.value))}>
      {comparison.examples.map((item, index) => <option key={item.left + "|" + item.right} value={index}>
        {graphExampleLabel(item)} · {item.left.slice(0, 8)}{item.left !== item.right && ` / ${item.right.slice(0, 8)}`}
      </option>)}
    </select></label>
    <ComparisonPanels key={example.left + "|" + example.right} panels={panels} {...view} />
  </>;
}

export function sharedGraphAnchors(models: ReadonlyArray<ReadonlyArray<{ anchor: string }>>) {
  if (models.length < 2) return [];
  const [first, ...rest] = models.map(options => new Set(options.map(row => row.anchor)));
  return [...first].filter(anchor => rest.every(anchors => anchors.has(anchor)));
}

function GroupGraphComparison({ group, data, graphs, split, ...view }: GraphView & {
  group: Cluster; data: SemanticCatalog; graphs: PairGraphCatalog["graphs"]; split: Split;
}) {
  const options = useMemo(() => MODEL_IDS.filter(model => group.models.includes(model)).map(model => ({
    model,
    graphs: group.components.flatMap(id => {
      const component = data.components[id];
      if (component.model !== model) return [];
      return (component.examples[split] ?? []).flatMap(example => {
        const witness = graphs[`${id}|${example.anchor}`];
        return witness ? [{ witness, key: `${id}|${example.anchor}`, weight: example.normalizedMagnitude }] : [];
      });
    }).sort((a, b) => b.weight - a.weight || a.key.localeCompare(b.key)),
  })), [group, data, graphs, split]);
  const shared = sharedGraphAnchors(options.map(option => option.graphs.map(row => row.witness)));
  const [position, setPosition] = useState(() => shared[0] ?? "");
  const [choices, setChoices] = useState<Partial<Record<EvalModel, string>>>({});
  const panels = options.map(({ model, graphs }) => {
    const available = position ? graphs.filter(row => row.witness.anchor === position) : graphs;
    const selected = available.find(row => row.key === choices[model]) ?? available[0];
    return {
      model, component: selected?.witness.component, anchor: selected?.witness.anchor, witness: selected?.witness,
      controls: available.length > 1 && <label className="semantic-graph-select">Graph for {MODEL_NAMES[model]}
        <select className="uw-form-control" value={selected?.key} onChange={event =>
          setChoices(previous => ({ ...previous, [model]: event.target.value }))}>
          {available.map(({ key, witness }) => <option key={key} value={key}>
            {witness.component} · {witness.anchor.slice(0, 8)}
          </option>)}
        </select>
      </label>,
    };
  });
  return <>
    <label className="discovered-picker">Graph alignment<select className="uw-form-control" value={position}
      onChange={event => { setPosition(event.target.value); setChoices({}); }}>
      {shared.map(anchor => <option key={anchor} value={anchor}>Same aligned token · {anchor.slice(0, 12)}</option>)}
      <option value="">Choose a saved graph per model</option>
    </select></label>
    <ComparisonPanels key={panels.map(panel => panel.component + "/" + panel.anchor).join("|")} panels={panels} {...view} />
  </>;
}

function SemanticGraphComparison({ pair, group, data, catalog, split, checkpoint, Graph }: {
  pair: Pair; group: Cluster; data: SemanticCatalog; catalog: PairGraphCatalog; split: Split; checkpoint: string; Graph: GraphRenderer;
}) {
  const comparison = catalog.pairs[`${pair.left}|${pair.right}`]?.[split];
  const hasCompletePair = comparison?.examples.some(row => hasBothGraphs(pair, row, catalog.graphs));
  const [mode, setMode] = useState(group.models.length > 2 || !hasCompletePair ? "group" : "pair");
  const view = { anchors: data.anchors, checkpoint, Graph, graphs: catalog.graphs };
  return <section className="semantic-graph-compare" aria-label="Compare dependency graphs">
    <h3>Compare dependency graphs</h3>
    <label className="discovered-picker">Models to show<select className="uw-form-control" value={mode}
      onChange={event => setMode(event.target.value)}>
      <option value="group">All models in group ({group.models.length})</option>
      <option value="pair">Selected component pair</option>
    </select></label>
    {mode === "group" ? <GroupGraphComparison key={group.id + "|" + split} group={group} data={data} split={split} {...view} />
      : <PairGraphComparison key={pair.left + "|" + pair.right + "|" + split} pair={pair} comparison={comparison} {...view} />}
  </section>;
}

export default function SemanticDiscovery({ checkpoint, Graph }: { checkpoint: string; Graph: GraphRenderer }) {
  const seal = useWorkbenchAsset<{ catalogSha256: string }>("semantic-discovery/COMPLETE.json");
  const asset = useWorkbenchAsset<SemanticCatalog>(seal?.data?.catalogSha256 ? "semantic-discovery/catalog.json" : undefined,
    seal?.data?.catalogSha256);
  const comparedGraphs = useWorkbenchAsset<PairGraphCatalog>(asset?.data ? "semantic-discovery/pair-graphs-v1.json" : undefined);
  const paths = useWorkbenchAsset<PathCatalog>(asset?.data ? "semantic-discovery/circuit-links.json" : undefined);
  const data = asset?.data?.schemaVersion === 1 && asset.data.dataset === "natural-validation"
    && asset.data.provenance.checkpoint === checkpoint && asset.data.provenance.split === "validation" ? asset.data : undefined;
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<ComparisonScope>("cross-model");
  const [heldoutOnly, setHeldoutOnly] = useState(false);
  const [clusterId, setClusterId] = useState("");
  const [pairIndex, setPairIndex] = useState(0);
  const [graphIndex, setGraphIndex] = useState(0);
  const [split, setSplit] = useState<Split>("discovery");
  const [pathIndex, setPathIndex] = useState(0);
  const byPair = useMemo(() => {
    if (!data) return [];
    const term = query.trim().toLowerCase();
    return data.clusters.filter(cluster => scope !== "three-models" || cluster.models.length === 3).map(cluster => ({
      cluster,
      pairs: cluster.links.filter(link =>
        matchesSemanticPair(data.components[link.left].model, data.components[link.right].model, scope)
        && [cluster.id, link.left, link.right,
          data.components[link.left].gate, data.components[link.right].gate].some(id => id.toLowerCase().includes(term)),
      ),
    })).filter(row => row.pairs.length);
  }, [data, query, scope]);
  const measuredHeldout = byPair.reduce((sum, row) => sum + row.pairs.filter(link => link.heldout.measures).length, 0);
  const groups = heldoutOnly ? byPair.map(row => ({
    cluster: row.cluster, pairs: row.pairs.filter(link => link.heldout.measures),
  })).filter(row => row.pairs.length) : byPair;
  const current = groups.find(row => row.cluster.id === clusterId) ?? groups[0];
  const selected = current?.cluster;
  const pairs = current?.pairs ?? [];
  const pair = pairs[pairIndex] ?? pairs[0];
  const graphs = selected?.graphs.filter(item => item.component === pair?.left || item.component === pair?.right) ?? [];
  const graph = graphs[graphIndex] ?? graphs[0];
  const graphComparison = comparedGraphs?.data?.schemaVersion === 1 && comparedGraphs.data.checkpoint === checkpoint
    && comparedGraphs.data.dataset === "natural-validation" && comparedGraphs.data.catalogSha256 === seal?.data?.catalogSha256
    && comparedGraphs.data.graphIndexSha256 === data?.provenance.graphIndexSha256 ? comparedGraphs.data : undefined;
  const result = pair?.[split];
  const matchingPaths = paths?.data?.checkpoint === checkpoint && paths.data.dataset === "natural-validation"
    && paths.data.catalogSha256 === seal?.data?.catalogSha256
    ? paths.data.pairs.filter(row => row.leftCluster === selected?.id || row.rightCluster === selected?.id) : [];
  const chosenPath = matchingPaths[pathIndex] ?? matchingPaths[0];
  const pathWitness = chosenPath?.witnesses.find(row => row.split === split);
  function reset() { setPairIndex(0); setGraphIndex(0); setPathIndex(0); }

  return <main className="td-section discovered-circuits" aria-label="Semantic Discovery">
    <div className="discovered-content">
      <SemanticDiscoveryGuide archived />
      <p>Original Semantic OT study · examples use archived dependency anchors. The outline shows the anchor location.</p>
      {!asset && !seal?.error && <p role="status">Loading Natural Validation Semantic OT results…</p>}
      {(asset?.error || seal?.error) && <p role="status">Semantic OT results are unavailable in this build.</p>}
      {asset?.data && !data && <p role="alert">This catalog belongs to a different dataset or checkpoint.</p>}
      {data && <>
        <div className="uw-expanded-search discovered-filters">
          <label>Find a component or group<input className="uw-form-control" type="search" value={query}
            onChange={event => { setQuery(event.target.value); reset(); }} /></label>
          <label>Compare<select className="uw-form-control" value={scope} onChange={event => {
            setScope(event.target.value as ComparisonScope); reset();
          }}>
            <option value="cross-model">Across models</option>
            <option value="three-models">Groups with all three models</option>
            <option value="gemma/smol">Gemma 2 ↔ SmolLM2</option>
            <option value="gemma/granite">Gemma 2 ↔ Granite</option>
            <option value="smol/granite">SmolLM2 ↔ Granite</option>
            <optgroup label="Within one model">
              {MODEL_IDS.map(id => <option key={id} value={`${id}/${id}`}>{MODEL_NAMES[id]} only</option>)}
            </optgroup>
            <option value="all">All pairs</option>
          </select></label>
          <label>Examples<select className="uw-form-control" value={split}
            onChange={event => {
              const next = event.target.value as Split;
              setSplit(next);
              if (next === "discovery") setHeldoutOnly(false);
            }}>
            <option value="discovery">Discovery</option><option value="heldout">Held-out documents</option>
          </select></label>
        </div>
        <p>{byPair.reduce((sum, row) => sum + row.pairs.length, 0)} pairs · {measuredHeldout} with held-out scores
          {split === "heldout" && <label className="ig-graph-filter">
            <input type="checkbox" checked={heldoutOnly} onChange={event => { setHeldoutOnly(event.target.checked); reset(); }} />
            Show only pairs with held-out scores
          </label>}
        </p>
        <label className="discovered-picker">Group<select className="uw-form-control" value={selected?.id ?? ""}
          onChange={event => { setClusterId(event.target.value); reset(); }}>
          {!groups.length && <option value="">No matching groups</option>}
          {groups.map(({ cluster, pairs }) => <option key={cluster.id} value={cluster.id}>{cluster.id} · {pairs.length} {
            pairs.length === 1 ? "pair" : "pairs"
          } · {cluster.models.length} {cluster.models.length === 1 ? "model" : "models"}</option>)}
        </select></label>
        {selected && pair && result && <div className="discovered-selection">
          <label className="discovered-picker">Component pair<select className="uw-form-control"
            value={pairs.indexOf(pair)} onChange={event => { setPairIndex(Number(event.target.value)); setGraphIndex(0); }}>
            {pairs.map((link, index) => <option key={link.left + link.right} value={index}>
              {link.left} ↔ {link.right}{link.heldout.measures ? " · Held-out scores" : ""}
            </option>)}
          </select></label>
          <div className="negation-context-grid">
            {[pair.left, pair.right].map(id => {
              const component = data.components[id];
              return <div key={id}>
                <strong>{MODEL_NAMES[component.model]}</strong>{" "}<code>{id}</code>
                <p>{component.gate} · {component.totals[split]
                  ? `${component.totals[split].measuredNonzero} measured nonzero contributions`
                  : "No measured nonzero contributions in this split"}</p>
              </div>;
            })}
          </div>
          {graphComparison ? <SemanticGraphComparison key={selected.id + "|" + split} pair={pair} group={selected}
            data={data} catalog={graphComparison} split={split} checkpoint={checkpoint} Graph={Graph} />
            : <p>{comparedGraphs ? "Graph comparisons are unavailable in this build." : "Loading graph comparisons…"}</p>}
          {result.measures ? <div className="cc-scroll"><table className="cc-table">
            <caption>Smaller distances indicate more similar contexts.</caption>
            <thead><tr><th>Pair</th><th>OT distance</th><th>OT without final word or punctuation</th><th>Average embedding distance</th><th>Shared anchor mass</th></tr></thead>
            <tbody><tr><th>Selected</th><td>{show(result.measures.ot)}</td><td>{show(result.suffixRemovedOt)}</td>
              <td>{show(result.measures.centroid)}</td><td>{show(result.measures.anchorOverlap)}</td></tr>
            <tr><th>Matched control{result.controlComponent && <> <code>{result.controlComponent}</code></>}</th>
              <td>{show(result.controlMeasures?.ot)}</td><td>{show(result.suffixRemovedControlOt)}</td><td>{show(result.controlMeasures?.centroid)}</td>
              <td>{show(result.controlMeasures?.anchorOverlap)}</td></tr></tbody>
          </table></div> : <p role="status">Too few measured contexts for this pair in {split === "heldout" ? "held-out documents" : "discovery"}.</p>}
          {result.transport.map((cell, index) => <details key={split + index} open={index === 0}>
            <summary>{cell.kind === "counterexample" ? "Less similar linked contexts" : "Contexts paired by OT"}
              {" · "}{(100 * cell.mass).toFixed(1)}% of mass · distance {show(cell.cost)}
              {cell.left === cell.right && " · Same aligned position"}</summary>
            <div className="negation-context-grid">
              {([["left", pair.left], ["right", pair.right]] as const).map(([side, id]) =>
                <div key={side}>
                  <strong>{MODEL_NAMES[data.components[id].model]}</strong>
                  <Prefix row={data.anchors[cell[side]]} model={data.components[id].model} />
                </div>)}
            </div>
          </details>)}
          <details><summary>Component contributions and examples</summary>
            {[pair.left, pair.right].map(id => {
              const component = data.components[id];
              const totals = component.totals[split];
              return <section key={id}>
                <h3><code>{id}</code></h3>
                {totals ? <p>{totals.measuredNonzero} measured nonzero contributions · total size {show(totals.normalizedMass)}
                  {" · "}positive coefficient mass {show(totals.positiveMass)} · negative coefficient mass {show(totals.negativeMass)}</p>
                  : <p>No measured positive mass in this split.</p>}
                {component.examples[split]?.map(example => <div key={example.anchor}>
                  <Prefix row={data.anchors[example.anchor]} model={component.model} />
                  <small>Coefficient e {show(example.coefficient)} · contribution relative to output scale {show(example.normalizedMagnitude)}</small>
                </div>)}
              </section>;
            })}
          </details>
          {chosenPath && <details><summary>Directed paths between grouped components</summary>
            <label className="discovered-picker">Path<select className="uw-form-control" value={matchingPaths.indexOf(chosenPath)}
              onChange={event => setPathIndex(Number(event.target.value))}>
              {matchingPaths.map((row, index) => <option key={row.left + row.right} value={index}>
                {row.leftCluster} {row.left} → {row.rightCluster} {row.right}
              </option>)}
            </select></label>
            <p>{chosenPath.discovery} discovery positions · {chosenPath.heldout} held-out positions with this path.</p>
            {pathWitness ? <ContextGraph key={chosenPath.left + chosenPath.right + pathWitness.anchor}
              witness={{ ...pathWitness, component: chosenPath.left }} row={data.anchors[pathWitness.anchor]}
              checkpoint={checkpoint} Graph={Graph} /> : <p>No saved path for this split.</p>}
          </details>}
          {!graphComparison && <details><summary>Individual graph examples</summary>
            {graph ? <>
              <label className="discovered-picker">Measured position<select className="uw-form-control"
                value={graphs.indexOf(graph)} onChange={event => setGraphIndex(Number(event.target.value))}>
                {graphs.map((item, index) => <option key={index} value={index}>{MODEL_NAMES[item.model]} · {item.component} · {item.anchor.slice(0, 10)}</option>)}
              </select></label>
              <ContextGraph key={graph.anchor + graph.component} witness={graph} row={data.anchors[graph.anchor]}
                checkpoint={checkpoint} Graph={Graph} />
            </> : <p>No matching Natural Validation edge records for this pair.</p>}
          </details>}
          {data.structuralOverlaps.filter(row => row.cluster === selected.id).map(row =>
            <p key={row.structural}>Shared component IDs with {row.structural} · {row.components.length}</p>)}
        </div>}
      </>}
    </div>
  </main>;
}
