import { useMemo, useState, type CSSProperties } from "react";
import ComponentComputationInspector from "./ComponentComputationInspector";
import { componentIdentity, type Measurement } from "./circuitComposition";
import type { FiringExample } from "./expandedNaturalValidation";
import { MODEL_IDS, MODEL_NAMES, type EvalModel } from "./negationWorkbenchData";
import { naturalLocal, naturalMetric, naturalScale, naturalScalar, naturalTokenMetrics,
  type NaturalTokenAsset, type NaturalScalarAsset, type NaturalTokenMetric, type TestTokenScope } from "./naturalTokenEvidence";
import { useWorkbenchAsset } from "./TokenCircuitExplorer";
import { ResearchNotesEditor } from "./ResearchNotes";
import { notesForSelection } from "./researchNotes";
import "./naturalTokenExplorer.css";

export default function NaturalTokenExplorer({ checkpoint, candidate, gate, rows, components,
  component, onComponent, onAnchor, measurement }: {
  checkpoint: string;
  candidate: string;
  gate: string;
  rows: FiringExample[];
  components: string[];
  component?: string;
  onComponent: (id: string) => void;
  onAnchor: (scope: TestTokenScope) => void;
  measurement?: Measurement;
}) {
  const [selected, setSelected] = useState<{ center: string; index: number }>();
  const [activeGate, setActiveGate] = useState<string>();
  const [metric, setMetric] = useState<NaturalTokenMetric>("gate");
  const { data } = useWorkbenchAsset<NaturalTokenAsset>(`natural-token-pilots/${candidate}/pilot.json`) ?? {};
  const asset = data?.schemaVersion === 1 && data.checkpoint === checkpoint && data.candidate === candidate
    && data.split === "test" && data.provenance.poolSize === 2048 ? data : undefined;
  const identity = component ? componentIdentity(component) : undefined;
  const replayPath = asset && component ? `natural-token-pilots/${candidate}` : undefined;
  const replays = {
    gemma: useWorkbenchAsset<NaturalScalarAsset>(replayPath && `${replayPath}/gemma.json`)?.data,
    smol: useWorkbenchAsset<NaturalScalarAsset>(replayPath && `${replayPath}/smol.json`)?.data,
    granite: useWorkbenchAsset<NaturalScalarAsset>(replayPath && `${replayPath}/granite.json`)?.data,
  };
  const replay = identity ? replays[identity.modelId as EvalModel] : undefined;
  const point = selected && rows.some(row => row.anchor === selected.center)
    ? asset?.examples[selected.center]?.points[selected.index] : undefined;
  const displayGate = metric === "gate" ? activeGate ?? identity?.gateId ?? gate : identity?.gateId ?? gate;
  const componentScalar = point && asset && component ? naturalScalar(asset, replay, point, component) : undefined;
  const local = point && asset && component ? naturalLocal(asset, replay, point, component) : [];
  const format = (value: number | undefined) => value === undefined ? "Unavailable" : value.toPrecision(4);
  const scale = useMemo(() => asset ? naturalScale(Object.values(asset.examples).flatMap(window =>
    Object.values(window.points).map(token => naturalMetric(metric, token, displayGate,
      component ? naturalScalar(asset, replay, token, component) : undefined)))) : 0,
    [asset, metric, displayGate, replay, component]);
  const active = Object.entries(point?.gates ?? {}).sort((a, b) => b[1] - a[1]);

  return <section className="uw-natural-tokens" aria-label="Token exploration">
    <div className="uw-graph-toolbar">
      <label>Token color <select value={metric} onChange={event => setMetric(event.target.value as NaturalTokenMetric)}>
        {Object.entries(naturalTokenMetrics).map(([id, label]) => <option key={id} value={id}
          disabled={id !== "gate" && !component}>{label}</option>)}
      </select></label>
      <label>Component <select value={component ?? ""} onChange={event => {
        onComponent(event.target.value);
        setActiveGate(undefined);
      }}>
        {!component && <option value="" disabled>Select a component</option>}
        {[...new Set(component ? [...components, component] : components)].map(id =>
          <option key={id} value={id}>{id}</option>)}
      </select></label>
    </div>
    {asset && <p className="uw-token-legend"><b>{displayGate}</b> · shared scale across pilot examples
      {scale > 0 && ` · max |value| ${scale.toPrecision(4)}`}
      {(metric === "reading" || metric === "coefficient") && " · teal positive · purple negative"}
      {" · hatched = unavailable"}</p>}
    {rows.map(example => {
      const window = asset?.examples[example.anchor];
      return <article className="uw-example-card" key={example.anchor}>
        {!window ? <blockquote>{example.before}<mark>{example.token.replaceAll("▁", " ")}</mark>{example.after}</blockquote>
          : <p className="ig-sequence td-sequence">{window.tokens.map((token, index) => {
            const row = window.points[index];
            const value = naturalMetric(metric, row, displayGate,
              row && component ? naturalScalar(asset, replay, row, component) : undefined);
            const text = token.replaceAll("▁", " ").replaceAll("<0x0A>", "\n");
            const pressed = selected?.center === example.anchor && selected.index === index;
            const title = row ? [
              ...Object.entries(row.positions).map(([model, pos]) => `${MODEL_NAMES[model as EvalModel]} token ${pos}`),
              row.anchor, value == null ? "Measurement unavailable" : String(value),
            ].join(" · ") : "Exact archive anchor unavailable";
            return row ? <button type="button" key={index} aria-pressed={pressed} title={title}
              className={`${value == null ? "missing" : ""} ${value != null && value < 0 ? "negative" : ""}`}
              style={{ "--activation": value != null && scale > 0 ? .85 * Math.abs(value) / scale : 0 } as CSSProperties}
              onClick={() => { setSelected({ center: example.anchor, index }); onAnchor({ anchor: row.anchor, pool: row.pool, dataset: asset.dataset,
                archiveManifestSha256: asset.provenance.archiveManifestSha256 }); }}>{text}</button>
              : <span className="uw-unavailable-token" title={title} key={index}>{text}</span>;
          })}</p>}
        <dl><dt>Source</dt><dd>{example.source}</dd><dt>Center gˢ</dt><dd>{example.activation.toPrecision(5)}</dd>
          <dt>Test anchor</dt><dd>{example.anchor}</dd></dl>
      </article>;
    })}
    {!rows.length && <p>No saved held-out example for {gate}.</p>}
    {!asset && rows.length > 0 && <p>Exact token measurements have not been exported for this selection.</p>}
    {point && asset && <>
      <div className="td-anchor-bar">
        <p><b>Selected token</b> · {active.length} active shared gates · pool {point.pool + 1}</p>
        <code>{point.anchor}</code>
      </div>
      <details open><summary>Active shared gates</summary>
        <div className="uw-active-tokens">{active.map(([id, value]) => <button type="button" key={id}
          aria-pressed={id === displayGate} onClick={() => { setActiveGate(id); setMetric("gate"); }}
          title={`gˢ ${value} · g 1`}>{id} · {value.toPrecision(3)}</button>)}</div>
      </details>
      {component && identity && <>
        <ComponentComputationInspector id={component} checkpoint={checkpoint} anchor={point.anchor} candidateId={candidate} measurement={measurement}
          token={{ dataset: asset.dataset, pool: String(point.pool), activation: point.gates[identity.gateId] ?? 0,
            support: identity.gateId in point.gates, ...componentScalar, local }} />
        <details><summary>Compare model components at this token</summary>
          <p>Saved component endpoints sharing {identity.gateId} and matrix {identity.matrix}.</p>
          {MODEL_IDS.map(model => {
            const peers = components.filter(id => {
              const peer = componentIdentity(id);
              return peer.modelId === model && peer.gateId === identity.gateId && peer.matrix === identity.matrix;
            });
            return <section key={model}>
              <strong>{MODEL_NAMES[model]}</strong> · native token {point.positions[model] ?? "Unavailable"}
              {peers.length ? <ul>{peers.map(id => {
                const scalar = naturalScalar(asset, replays[model], point, id);
                return <li key={id}><button type="button" onClick={() => onComponent(id)}>{id}</button>
                  {" · reading "}{format(scalar?.reading)}
                  {" · e "}{format(naturalMetric("coefficient", point, identity.gateId, scalar))}
                  {" · ‖eu‖₂ "}{format(naturalMetric("magnitude", point, identity.gateId, scalar))}</li>;
              })}</ul> : <p>No saved component endpoint.</p>}
            </section>;
          })}
        </details>
        <ResearchNotesEditor key={component} targets={notesForSelection(checkpoint, identity.gateId, identity.modelId,
          identity)}
          evidenceReference={`${candidate} · test anchor ${point.anchor}`} />
      </>}
    </>}
  </section>;
}
