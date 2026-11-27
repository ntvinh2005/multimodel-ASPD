import { useMemo, useRef, useState, type ReactNode } from "react";
import { MODEL_NAMES, type EvalModel } from "./negationWorkbenchData";
import { ResearchNotesEditor } from "./ResearchNotes";
import { notesForSelection } from "./researchNotes";
import { componentIdentity } from "./circuitComposition";
import { useWorkbenchAsset } from "./TokenCircuitExplorer";
import { SemanticDiscoveryGuide } from "./SemanticDiscoveryGuide";
import { ComparisonPanels, type Anchor, type Witness } from "./SemanticDiscovery";
import type { GraphRenderer } from "./InductionQueryGraph";

type Split = "discovery" | "heldout";
type Metrics = { ot: number; centroid: number; anchorOverlap: number; distinctContextMass: number };
type Comparison = {
  values: Metrics;
  controls: { left: Metrics; right: Metrics };
  advantage: number;
  contextChecks: { sameDocumentMass: number; sameTerminalMass: number; weightedPrefixCharacterGap: number };
  transport: { left: string; right: string; mass: number; cost: number; kind: string }[];
};
type Pair = {
  left: string;
  right: string;
  results: Record<Split, Record<string, Comparison | null>>;
  fullSupportJaccard: Record<Split, number>;
  heldoutAdvantageInterval: [number, number];
  controls: Record<"left" | "right", Record<Split, string[]>>;
};
type Component = {
  model: EvalModel;
  gate: string;
  coverage: Record<Split, number>;
  beta: number;
};
type Example = {
  anchor: string;
  document: string;
  source: string;
  pool: number;
  boundary: number;
  contrast: boolean;
  coefficient: number;
  magnitude: number;
  graph: "unavailable";
  sequence: {
    anchor: string | null;
    text: string;
    coefficient: number | null;
    magnitude: number | null;
    tokenId: number | null;
    tokenPosition: number | null;
  }[];
};
type ExampleAsset = {
  runId: string;
  component: string;
  maxMagnitude: number;
  examples: Record<Split, Example[]>;
};
type Release = { schemaVersion: number; runId: string; completeSha256: string };
type Seal = { schemaVersion: number; status: string; runId: string; catalogSha256: string };
export type RevisedCatalog = {
  schemaVersion: number;
  runId: string;
  dataset: string;
  checkpoint: string;
  components: Record<string, Component>;
  pairs: Pair[];
  groups: { components: string[] }[];
  examples: Record<string, { file: string; sha256: string }>;
  graphCoverage: "unavailable";
};

type GraphRelease = { catalogSha256: string; runId: string; sha256: string };
type MeasuredGraphs = {
  schemaVersion: number; runId: string; catalogSha256: string; checkpoint: string; dataset: string;
  anchors: Record<string, Anchor & { split: Split }>;
  graphs: Record<string, Witness & { pool: number }>;
  anchorOrder: string[];
};

const digest = /^[a-f\d]{64}$/;

export function hasSemanticCoverage(catalog: RevisedCatalog): boolean {
  const ids = new Set(catalog.pairs.flatMap(pair => [pair.left, pair.right]));
  if (!ids.size || Object.keys(catalog.components).length !== ids.size) return false;
  let total = 0;
  for (const id of ids) {
    const count = catalog.components[id]?.coverage?.heldout;
    const ref = catalog.examples[id];
    if (!Number.isSafeInteger(count) || count < 10 || !ref || !digest.test(ref.sha256)
      || !/^component-\d+\.json\.gz$/.test(ref.file)) return false;
    total += count;
  }
  return total / ids.size >= 20 && catalog.groups.every(group => group.components.every(id => ids.has(id)));
}

function number(value: number | undefined) {
  return value == null ? "Unavailable" : value.toPrecision(3);
}

function percent(value: number | undefined) {
  return value == null ? "Unavailable" : (100 * value).toFixed(1) + "%";
}


function Heatmap({ example, scale }: { example: Example; scale: number }) {
  return <p className="ig-sequence td-sequence semantic-contribution-sequence">
    {example.sequence.map((token, index) => {
      if (token.anchor === null) return <span key={index} title="No aligned measurement for this span">{token.text}</span>;
      const alpha = token.magnitude === null ? 0 : scale > 0 ? Math.min(1, token.magnitude / scale) : 0;
      return <span key={token.anchor} className={token.anchor === example.anchor ? "semantic-anchor-outline" : undefined}
        style={alpha ? { backgroundColor: `rgba(37, 99, 235, ${0.08 + 0.55 * Math.sqrt(alpha)})` } : undefined}
        title={`Native token ${token.tokenPosition} · ID ${token.tokenId} · coefficient ${number(token.coefficient ?? undefined)} · normalized contribution ${number(token.magnitude ?? undefined)} · anchor ${token.anchor}`}>
        {token.text}
      </span>;
    })}
  </p>;
}

function ComponentExamples({ component, identity, catalog, split, preferred, onNotes, graphAvailable, onGraph }: {
  component: Component; identity: string; catalog: RevisedCatalog; split: Split; preferred?: string;
  onNotes: (component: string, anchor: string) => void;
  graphAvailable: (anchor: string) => boolean;
  onGraph: (anchor: string) => void;
}) {
  const ref = catalog.examples[identity];
  const source = useWorkbenchAsset<ExampleAsset>(`semantic-discovery-v2/${catalog.runId}/${ref.file}`, ref.sha256);
  const [contrast, setContrast] = useState(false);
  const [ordering, setOrdering] = useState<"magnitude" | "weakest" | "document">("magnitude");
  const [anchor, setAnchor] = useState("");
  const asset = source?.data?.runId === catalog.runId && source.data.component === identity ? source.data : undefined;
  const rows = useMemo(() => {
    const examples = asset?.examples[split].filter(example => example.contrast === contrast) ?? [];
    return examples.sort((a, b) => ordering === "document" ? a.document.localeCompare(b.document)
      : (ordering === "weakest" ? a.magnitude - b.magnitude : b.magnitude - a.magnitude)
        || a.document.localeCompare(b.document));
  }, [asset, split, contrast, ordering]);
  const row = rows.find(example => example.anchor === (anchor || preferred)) ?? rows[0];
  const totals = asset?.examples[split].filter(example => !example.contrast).reduce((sum, example) => {
    sum[example.coefficient > 0 ? 0 : 1] += example.magnitude;
    return sum;
  }, [0, 0]);
  return <section>
    <h3>{MODEL_NAMES[component.model]} <code>{identity}</code></h3>
    <p>{component.coverage.discovery} discovery documents · {component.coverage.heldout} held-out documents
      {" · "}β {number(component.beta)} from reference validation</p>
    <div className="discovered-filters">
      <label>Examples<select className="uw-form-control" value={contrast ? "contrast" : "nonzero"}
        onChange={event => { setContrast(event.target.value === "contrast"); setAnchor(""); }}>
        <option value="nonzero">Measured nonzero</option><option value="contrast">Matched measured zero</option>
      </select></label>
      <label>Sort documents<select className="uw-form-control" value={ordering}
        onChange={event => { setOrdering(event.target.value as typeof ordering); setAnchor(""); }}>
        <option value="magnitude">Largest contribution</option><option value="weakest">Smallest contribution</option>
        <option value="document">Document ID</option>
      </select></label>
    </div>
    {!asset ? <p role="status">{source ? "Verified examples are unavailable." : "Loading examples…"}</p>
      : <>{totals && <p>Summed normalized size by signed coefficient · positive {number(totals[0])}
        {" · "}negative {number(totals[1])}</p>}
      <label className="discovered-picker">Document ({rows.length})<select className="uw-form-control"
        value={row?.anchor ?? ""} onChange={event => setAnchor(event.target.value)}>
        {!row && <option value="">No measured examples in this view</option>}
        {rows.map(example => <option value={example.anchor} key={example.anchor}>
          {example.source} · {example.document.slice(0, 14)} · {number(example.magnitude)}
        </option>)}
      </select></label>
      {row && <>
        <Heatmap example={row} scale={asset.maxMagnitude} />
        <small>Shading follows normalized contribution magnitude on a fixed scale for this component
          {" · "}maximum {number(asset.maxMagnitude)}. Hover for the measured values and native token ID.</small>
        <p>Selected anchor <code title={row.anchor}>{row.anchor.slice(0, 12)}</code>
          {" · "}pool {row.pool} · character boundary {row.boundary} · coefficient {number(row.coefficient)}
          {" · "}normalized contribution {number(row.magnitude)}</p>
        {graphAvailable(row.anchor) ? <button className="uw-form-control" type="button"
          onClick={() => onGraph(row.anchor)}>Compare dependency graphs at this anchor</button>
          : <small>Graph unavailable for this exact model, component, and anchor.</small>}
        <button className="uw-form-control" type="button"
          onClick={() => onNotes(identity, row.anchor)}>Open Research Notes</button>
      </>}
    </>}
  </section>;
}

const names: Record<string, string> = {
  combined: "Local + anchor", anchor: "Anchor only", local: "Local prefix",
  fullPrefix: "Full prefix (up to 512 encoder tokens)", prefix: "Original 128-token prefix pooling",
  withoutTerminal: "Local prefix without terminal unit",
};

function PairResults({ pair, catalog, split, measured, Graph }: {
  pair: Pair; catalog: RevisedCatalog; split: Split; measured?: MeasuredGraphs; Graph: GraphRenderer;
}) {
  const heading = useRef<HTMLElement>(null);
  const [anchor, setAnchor] = useState("");
  const [match, setMatch] = useState<number | null>(null);
  const [note, setNote] = useState<{ component: string; anchor: string } | null>(null);
  const target = note ? componentIdentity(note.component) : null;
  const metrics = pair.results[split];
  const transport = metrics.combined?.transport ?? [];
  const ids = [pair.left, pair.right];
  const witness = (id: string, at: string) => {
    const row = measured?.graphs[id + "|" + at];
    return row?.component === id && row.anchor === at && row.model === catalog.components[id].model
      && row.pool === measured?.anchors[at]?.pool ? row : undefined;
  };
  const available = measured?.anchorOrder.filter(at => measured.anchors[at]?.split === split
    && ids.some(id => witness(id, at))) ?? [];
  const shared = available.filter(at => ids.every(id => witness(id, at)));
  const transportPair = match == null ? undefined : transport[match];
  const current = available.includes(anchor) ? anchor : shared[0] ?? available[0];
  const preferred = transportPair ?? (current ? {
    left: witness(pair.left, current) ? current : undefined,
    right: witness(pair.right, current) ? current : undefined,
  } : undefined);
  return <>
    <section ref={heading} className="semantic-graph-compare" aria-label="Measured dependency comparison">
      <h3>Measured dependency graphs</h3>
      {measured && current ? <>
        <label className="discovered-picker">Aligned anchor
          <select className="uw-form-control" value={current} onChange={event => {
            setAnchor(event.target.value);
            setMatch(null);
            setNote(null);
          }}>
            {[...shared, ...available.filter(at => !shared.includes(at))].map(at => {
              const row = measured.anchors[at];
              return <option key={at} value={at}>
                {row.source} · pool {row.pool} · {at.slice(0, 12)} · {shared.includes(at) ? "both components" : "one component"}
              </option>;
            })}
          </select>
        </label>
        <small>{shared.length} shared anchors · {available.length} with either endpoint measured · top 10 incoming Taylor parents per endpoint.</small>
        <ComparisonPanels key={current} scopeLabel="Incoming Taylor edges for the selected component."
          panels={ids.map(id => ({ model: catalog.components[id].model, anchor: current, component: id, witness: witness(id, current) }))}
          anchors={measured.anchors} checkpoint={catalog.checkpoint} Graph={Graph}
          onAnnotation={(componentId, fromAnchor) => setNote(componentId ? { component: componentId, anchor: fromAnchor } : null)} />
      </> : <p>Graph unavailable for this pair and document split.</p>}
    </section>
    <p>Lower distances mean more similar contexts. The controls measure how each endpoint compares with matched contexts where the other component contributes zero.</p>
    <div className="cc-scroll"><table className="cc-table">
      <thead><tr><th>Representation</th><th>OT distance</th><th>Control OT range</th><th>Mean embedding distance</th><th>Exact-anchor overlap</th></tr></thead>
      <tbody>{Object.entries(names).map(([variant, label]) => {
        const result = metrics[variant];
        return <tr key={variant}><th>{label}</th><td>{number(result?.values.ot)}</td>
          <td>{result ? [result.controls.left.ot, result.controls.right.ot].sort((a, b) => a - b).map(number).join(" to ") : "Unavailable"}</td>
          <td>{number(result?.values.centroid)}</td><td>{percent(result?.values.anchorOverlap)}</td></tr>;
      })}</tbody>
    </table></div>
    <p>Exact-anchor overlap across all eligible documents {percent(pair.fullSupportJaccard[split])}.</p>
    {split === "heldout" && <p>Advantage over the closer control {number(metrics.combined?.advantage)}
      {" · "}exploratory 95% document-bootstrap interval {pair.heldoutAdvantageInterval.map(number).join(" to ")}.</p>}
    <details><summary>Context checks and OT matches</summary>
      <p>{percent(metrics.combined?.contextChecks.sameDocumentMass)} same-document mass · {percent(metrics.combined?.contextChecks.sameTerminalMass)} same terminal unit
        {" · "}weighted prefix length gap {number(metrics.combined?.contextChecks.weightedPrefixCharacterGap)} characters.</p>
      <p>Pick a linked pair to see which contexts carry transport mass. Matching context distributions is evidence about usage in text.</p>
      <label className="discovered-picker">OT match<select className="uw-form-control" value={match ?? ""}
        onChange={event => setMatch(event.target.value === "" ? null : Number(event.target.value))}>
        <option value="">Browse examples independently</option>
        {transport.map((item, i) => <option key={item.left + item.right} value={i}>
          {item.kind === "high-cost" && "High-distance match · "}
          {item.left.slice(0, 9)} ↔ {item.right.slice(0, 9)} · {percent(item.mass)} of mass · distance {number(item.cost)}
        </option>)}
      </select></label>
    </details>
    <div className="negation-context-grid">{(["left", "right"] as const).map(side => <ComponentExamples
      key={pair[side] + split + (preferred?.[side] ?? "")} identity={pair[side]} component={catalog.components[pair[side]]}
      split={split} catalog={catalog} preferred={preferred?.[side]}
      onNotes={(component, anchor) => setNote({ component, anchor })}
      graphAvailable={at => Boolean(witness(pair[side], at))}
      onGraph={at => {
        setAnchor(at);
        setMatch(null);
        setNote(null);
        heading.current?.scrollIntoView({ block: "nearest" });
      }} />)}</div>
    {note && target && <section aria-label="Semantic Discovery Research Notes">
      <ResearchNotesEditor key={target.id} targets={notesForSelection(catalog.checkpoint, target.gateId, target.modelId, target)}
        evidenceReference={`Semantic Discovery · ${note.anchor}`} />
    </section>}
  </>;
}

function RevisedWorkspace({ catalog, catalogSha256, Graph }: {
  catalog: RevisedCatalog; catalogSha256: string; Graph: GraphRenderer;
}) {
  const pointer = useWorkbenchAsset<GraphRelease>("semantic-discovery-graphs-release.json", undefined, "no-store")?.data;
  const sha = pointer?.catalogSha256 === catalogSha256 && pointer.runId === catalog.runId
    && digest.test(pointer.sha256) ? pointer.sha256 : undefined;
  const candidate = useWorkbenchAsset<MeasuredGraphs>(sha && `semantic-discovery-v2/${catalog.runId}/graphs-${sha}.json.gz`, sha)?.data;
  const measured = candidate?.schemaVersion === 1 && candidate.runId === catalog.runId
    && candidate.catalogSha256 === catalogSha256 && candidate.checkpoint === catalog.checkpoint
    && candidate.dataset === catalog.dataset ? candidate : undefined;
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState("across");
  const [split, setSplit] = useState<Split>("heldout");
  const [key, setKey] = useState("");
  const id = (pair: Pair) => pair.left + "|" + pair.right;
  const sameGatePairs = catalog.pairs.filter(({ left, right }) => catalog.components[left].gate === catalog.components[right].gate).length;
  const first = catalog.pairs[0];
  const modelNames = [first.left, first.right].map(c => MODEL_NAMES[catalog.components[c].model]).join(" and ");
  const pairs = catalog.pairs.filter(pair => {
    const a = catalog.components[pair.left], b = catalog.components[pair.right];
    if (scope === "across" && a.model === b.model) return false;
    if (scope === "within" && a.model !== b.model) return false;
    if (scope === "three" && !catalog.groups.some(group => group.components.includes(pair.left)
      && group.components.includes(pair.right)
      && new Set(group.components.map(component => catalog.components[component].model)).size === 3)) return false;
    return [pair.left, pair.right, a.gate, b.gate, a.model, b.model]
      .some(value => value.toLowerCase().replace("/", ":").includes(query.trim().toLowerCase().replace("/", ":")));
  });
  const current = pairs.find(pair => id(pair) === key) ?? pairs[0];
  return <main className="td-section discovered-circuits" aria-label="Semantic Discovery">
    <div className="discovered-content">
      <SemanticDiscoveryGuide example={{ left: first.left, right: first.right,
        distance: first.results.heldout.combined?.values.ot }} modelNames={modelNames} />
      <p>{catalog.pairs.length} pairs passed the held-out checks. {sameGatePairs} share the same gate ID on both sides.
        {" "}Shared gate support can lead both components to contribute in the same contexts.</p>
      <div className="discovered-filters">
        <label>Find a gate or component<input className="uw-form-control" type="search" value={query}
          onChange={event => setQuery(event.target.value)} /></label>
        <label>Compare<select className="uw-form-control" value={scope} onChange={event => setScope(event.target.value)}>
          <option value="across">Across models</option><option value="three">Groups with all three models</option>
          <option value="within">Within each model</option><option value="all">All pairs</option>
        </select></label>
        <label>Documents<select className="uw-form-control" value={split} onChange={event => setSplit(event.target.value as Split)}>
          <option value="heldout">Held-out</option><option value="discovery">Discovery</option>
        </select></label>
      </div>
      <label className="discovered-picker">Component pair ({pairs.length})
        <select className="uw-form-control" value={current ? id(current) : ""} onChange={event => setKey(event.target.value)}>
          {!current && <option value="">No matching pairs</option>}
          {pairs.map(pair => <option key={id(pair)} value={id(pair)}>{pair.left} ↔ {pair.right}</option>)}
        </select>
      </label>
      {current && <PairResults key={id(current) + split} catalog={catalog} pair={current} split={split} measured={measured} Graph={Graph} />}
    </div>
  </main>;
}

export default function SemanticDiscoveryRevision({ checkpoint, legacy, Graph }: {
  checkpoint: string; legacy: ReactNode; Graph: GraphRenderer;
}) {
  const [showArchive, setShowArchive] = useState(false);
  const release = useWorkbenchAsset<Release>("semantic-discovery-release.json", undefined, "no-store");
  const pointer = release?.data?.schemaVersion === 2 && digest.test(release.data.runId)
    && digest.test(release.data.completeSha256) ? release.data : undefined;
  const root = pointer && `semantic-discovery-v2/${pointer.runId}`;
  const seal = useWorkbenchAsset<Seal>(root && `${root}/COMPLETE.json`, pointer?.completeSha256);
  const sealed = seal?.data?.schemaVersion === 2 && seal.data.runId === pointer?.runId
    && seal.data.status === "COVERAGE_VALIDATED" && digest.test(seal.data.catalogSha256) ? seal.data : undefined;
  const asset = useWorkbenchAsset<RevisedCatalog>(sealed && `${root}/catalog.json`, sealed?.catalogSha256);
  if (release?.error) return <div className="discovered-content">
    <SemanticDiscoveryGuide />
    <p role="status">A verified revised catalog is unavailable.</p>
    <details onToggle={event => setShowArchive(event.currentTarget.open)}>
      <summary>Open original archived study</summary>
      {showArchive && legacy}
    </details>
  </div>;
  if (asset?.data) {
    const data = asset.data;
    if (data.checkpoint === checkpoint && data.dataset === "natural-validation"
      && data.schemaVersion === 2 && data.runId === sealed?.runId && hasSemanticCoverage(data)) {
      return <RevisedWorkspace catalog={data} catalogSha256={sealed.catalogSha256} Graph={Graph} />;
    }
    return <p role="alert">Semantic Discovery catalog failed its identity or coverage checks.</p>;
  }
  if (seal?.error || asset?.error || (release?.data && !pointer) || (seal?.data && !sealed)) {
    return <p role="alert">The revised Semantic Discovery release could not be verified.</p>;
  }
  return <p role="status">Checking the Semantic Discovery release…</p>;
}
