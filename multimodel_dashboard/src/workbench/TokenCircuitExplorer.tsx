import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { QueryGraphPanel, type GraphRenderer, type QueryGraph, type QueryTarget } from "./InductionQueryGraph";
import { activationAt, MODEL_IDS, MODEL_NAMES, readWorkbenchAsset, type EvalModel, type MatrixLocator } from "./negationWorkbenchData";
import { tokenGraph, type GraphWitnessIndex, type TokenExample, type TokenGraphAsset, type TokenGroup, type TokenIndex, type TokenPosition } from "./tokenDiscovery";
import CircuitCompositionExplorer from "./CircuitCompositionExplorer";
import { measurementFromToken } from "./compositionMeasurements";
import type { Measurement, Scope } from "./circuitComposition";
import { ResearchNotesEditor } from "./ResearchNotes";
import ComponentComputationInspector from "./ComponentComputationInspector";
import ComponentContextProbe from "./ComponentContextProbe";
import { componentIdentity } from "./circuitComposition";
import { noteKey, notesForGate, type NoteTarget } from "./researchNotes";
import "./tokenDiscovery.css";

export function useWorkbenchAsset<T>(path?: string, hash?: string) {
  const [result, setResult] = useState<{ path: string; hash?: string; data?: T; error?: string }>();
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    void readWorkbenchAsset<T>(path, controller.signal, hash).then(
      (data) => { if (!controller.signal.aborted) setResult({ path, hash, data }); },
      (error: Error) => { if (!controller.signal.aborted) setResult({ path, hash, error: error.message }); },
    );
    return () => controller.abort();
  }, [path, hash]);
  return result?.path === path && result?.hash === hash ? result : undefined;
}

export function useTokenIndex(checkpoint?: string) {
  const asset = useWorkbenchAsset<TokenIndex>(checkpoint ? "tokens/index.json" : undefined);
  const index = asset?.data?.checkpointSha256 === checkpoint ? asset?.data : undefined;
  return { index, error: asset?.error ?? (asset?.data && !index ? "The token inventory belongs to a different checkpoint." : undefined) };
}

export function useTokenGroup(index: TokenIndex | undefined, dataset: string, group: string | undefined) {
  const ref = group ? index?.datasets[dataset]?.[group] : undefined;
  const asset = useWorkbenchAsset<TokenGroup>(ref && `tokens/${ref.path}`, ref?.sha256);
  const data = asset?.data;
  const compatible = data && data.checkpointSha256 === index?.checkpointSha256 && data.dataset === dataset && data.groupId === group;
  return { group: compatible ? data : undefined, error: asset?.error ?? (data && !compatible ? "The token group has different provenance." : undefined) };
}

export function useGraphWitnesses(index: TokenIndex | undefined, dataset: string) {
  const asset = useWorkbenchAsset<GraphWitnessIndex>(index ? "graph-witness-index.json" : undefined);
  return asset?.data?.schemaVersion === 1 && asset.data.checkpointSha256 === index?.checkpointSha256
    ? asset.data.datasets[dataset] : undefined;
}

export function ModelSelect({ model, onChange }: { model: EvalModel; onChange: (id: EvalModel) => void }) {
  return <label>Model<select value={model} onChange={event => onChange(event.target.value as EvalModel)}>
    {MODEL_IDS.map(id => <option key={id} value={id}>{MODEL_NAMES[id]}</option>)}
  </select></label>;
}

export function TokenSequence({ example, gate, scale, anchor, model, onSelect, label }: {
  example: TokenExample;
  gate?: string;
  scale: number;
  anchor?: string;
  model: EvalModel;
  onSelect: (anchor: string) => void;
  label?: string;
}) {
  const prompt = Array.from(example.prompt);
  return <div>{label && <small>{label}</small>}
    <p className="ig-sequence td-sequence">{example.positions.map((position, i) => {
      const token = prompt.slice(example.positions[i - 1]?.character_boundary ?? 0, position.character_boundary).join("");
      const value = gate ? activationAt(position, gate) : undefined;
      return <button key={position.anchor_id} type="button"
        className={position.final_query ? "final-query" : undefined} aria-pressed={position.anchor_id === anchor}
        style={{ "--activation": scale > 0 && value ? .9 * value / scale : 0 } as CSSProperties}
        title={[`${MODEL_NAMES[model]} token ${position.model_positions[model]}`,
          value === undefined ? null : `gˢ ${value} · g ${Number(value > 0)}`, ...position.roles_ending].filter(Boolean).join(" · ")}
        onClick={() => onSelect(position.anchor_id)}>{token}</button>;
    })}</p>
  </div>;
}

export function TokenAnchorBar({ anchor, model, activeCount, anchorOnly, onToggle, onReset }: {
  anchor: TokenPosition; model: EvalModel; activeCount: number; anchorOnly: boolean;
  onToggle?: () => void; onReset: () => void;
}) {
  return <div className="td-anchor-bar">
    <p><b>{anchor.final_query ? "Final query" : "Selected token"}</b> · {MODEL_NAMES[model]} token {anchor.model_positions[model]}
      {" · "}{activeCount} active shared gates</p>
    {onToggle && <button type="button" onClick={onToggle}>{anchorOnly ? "Browse all gates" : "Show active gates at this token"}</button>}
    {!anchor.final_query && <button type="button" onClick={onReset}>Return to final token</button>}
  </div>;
}

type LegacyEvidence = {
  graph?: QueryGraph;
  scope?: Scope;
  targets?: ReadonlyMap<string, { target: QueryTarget; representative: boolean }>;
  error?: string;
};

export function useSelectedNote(context: string) {
  const editor = useRef<HTMLDivElement>(null);
  const [selection, setSelection] = useState<{ context: string; target: NoteTarget; reference?: string }>();
  function annotate(target: NoteTarget, reference?: string) {
    setSelection({ context, target: { ...target }, reference });
    const details = editor.current?.querySelector("details");
    if (details) details.open = true;
    editor.current?.scrollIntoView({ block: "nearest" });
  }
  return { editor, annotate, selection: selection?.context === context ? selection : undefined };
}

export function SelectedNoteEditor({ checkpoint, gate, evidenceReference, note }: {
  checkpoint: string; gate: string; evidenceReference?: string; note: ReturnType<typeof useSelectedNote>;
}) {
  const targets = notesForGate(checkpoint, gate);
  const selected = note.selection?.target;
  if (selected && !targets.some(target => noteKey(target) === noteKey(selected))) targets.push(selected);
  return <div ref={note.editor}>
    <ResearchNotesEditor targets={targets} requestedTarget={selected}
      evidenceReference={note.selection?.reference ?? evidenceReference} />
  </div>;
}

export function graphCoverageLabel(loading: boolean, failed: boolean, complete: boolean, hasRows: boolean) {
  if (loading) return "Loading measured graph data";
  if (failed) return "Graph data could not be loaded or verified";
  if (complete) return "Full component graphs measured at this token";
  if (hasRows) return "Some component graphs measured at this token";
  return "Dependency graphs unmeasured at this token";
}

export function TokenDependencyInspector({ checkpoint, dataset, index, example, examples, anchor, gate, model, matrices, Graph,
  onSelectToken, onSelectExampleToken, onExploreGate, legacy, onAnnotate, onOpenMeasuredExample }: {
  checkpoint: string; dataset: string; index?: TokenIndex; example: TokenExample; anchor: TokenPosition;
  gate?: string; model: EvalModel; matrices?: MatrixLocator[]; Graph: GraphRenderer;
  onSelectToken: (id: string) => void; onExploreGate: (id: string) => void; legacy?: LegacyEvidence;
  onAnnotate: (target: NoteTarget, reference?: string) => void;
  onOpenMeasuredExample?: () => void;
  examples?: TokenExample[];
  onSelectExampleToken?: (row: TokenExample, anchor: string) => void;
}) {
  const [inspectedComponent, setInspectedComponent] = useState<string>();
  const ref = index?.graphs[anchor.anchor_id];
  const asset = useWorkbenchAsset<TokenGraphAsset>(ref && `tokens/${ref.path}`, ref?.sha256);
  const loading = Boolean(ref && !asset);
  const composition = useMemo<{ error?: string; measurement?: Measurement }>(() => {
    if (!asset?.data || !ref) return {};
    try {
      const scope = { checkpoint, dataset, example: example.example_id, anchor: anchor.anchor_id, pool: anchor.pool_id,
        artifact: `tokens/${ref.path} · SHA-256 ${ref.sha256}` };
      return { measurement: measurementFromToken(asset.data, scope, example, anchor) };
    } catch (error) { return { error: error instanceof Error ? error.message : String(error) }; }
  }, [asset, ref, checkpoint, dataset, example, anchor]);
  const evidence = useMemo<Partial<ReturnType<typeof tokenGraph>> & { error?: string }>(() => {
    if (!composition.measurement || !asset?.data || !gate || !activationAt(anchor, gate)) return {};
    try { return tokenGraph(asset.data, example, anchor, checkpoint, gate, model, matrices); }
    catch (error) { return { error: error instanceof Error ? error.message : String(error) }; }
  }, [composition.measurement, asset, checkpoint, example, anchor, gate, model, matrices]);
  const issue = composition.error ?? evidence.error ?? asset?.error;
  const canFallBack = !issue && !loading && !evidence.graph && anchor.final_query && legacy?.graph?.anchorId === anchor.anchor_id
    && legacy.scope?.checkpoint === checkpoint && legacy.scope.dataset === dataset && legacy.scope.example === example.example_id
    && legacy.scope.anchor === anchor.anchor_id && legacy.scope.pool === anchor.pool_id
    && gate !== undefined && legacy.graph.activation === activationAt(anchor, gate);
  const graph = evidence.graph ?? (canFallBack ? legacy?.graph : undefined);
  const targets = evidence.targets ?? (canFallBack ? legacy?.targets : undefined);
  const active = gate ? activationAt(anchor, gate) : undefined;
  const status = issue ?? (loading ? "Loading this token’s dependencies…" : undefined);
  const evidenceReference = `${dataset} · ${example.example_id} · anchor ${anchor.anchor_id}`;
  const coverage = graphCoverageLabel(loading, !!issue, composition.measurement?.complete === true, Boolean(composition.measurement?.targets.length || graph));
  const hasGateRows = Boolean(graph?.models[model]?.targets.length
    || (targets && [...targets.values()].some(row => row.target.component.gateId === gate)));
  const inspected = inspectedComponent && componentIdentity(inspectedComponent).modelId === model ? inspectedComponent : undefined;
  const componentRow = inspected ? targets?.get(inspected) : undefined;
  const rawComponent = inspected ? asset?.data?.componentTargets[inspected] : undefined;
  const rawScale = inspected ? asset?.data?.outputScale[inspected.split("/")[0]] : undefined;
  const componentActivation = inspected ? activationAt(anchor, componentIdentity(inspected).gateId) : undefined;
  return <div className="td-inspector">
    <p className="ig-graph-status" role="status"><strong>{coverage}</strong>
      {gate && <> · {gate} · {active ? `gˢ ${active.toPrecision(4)}` : "Gate inactive at this token"}</>}</p>
    {gate && graph && active ? <QueryGraphPanel key={`${anchor.anchor_id}/${gate}/${model}`} graph={graph} Graph={Graph} model={model}
      gateId={gate} annotationCheckpoint={checkpoint} evidenceReference={evidenceReference} traceTargets={targets}
      compositionMeasurement={composition.measurement} compositionScope={canFallBack ? legacy?.scope : undefined}
      completeComponents={asset?.data?.completeComponents} anchorLabel={anchor.final_query ? "FINAL QUERY" : "SELECTED TOKEN"}
      onExploreGate={onExploreGate} onAnnotate={onAnnotate} onInspectComponent={setInspectedComponent} />
      : (issue || (!loading && gate && active && Boolean(composition.measurement?.targets.length))) && <p className="ig-graph-status" role="status">
        {issue ?? (anchor.final_query ? legacy?.error : undefined) ?? "This gate fired here, but its downstream components are outside the measured graph subset."}</p>}
    {!loading && !hasGateRows && onOpenMeasuredExample
      && <button type="button" onClick={onOpenMeasuredExample}>Open a measured example for {gate}</button>}
    {inspected && <ComponentComputationInspector id={inspected} checkpoint={checkpoint} anchor={anchor.anchor_id}
      measurement={composition.measurement} token={{
        dataset, pool: anchor.pool_id,
        activation: componentActivation, support: componentActivation !== undefined && componentActivation > 0,
        magnitude: componentRow?.target.contributionNorm ?? (rawComponent ? Math.abs(rawComponent[0]) * rawComponent[1] : undefined),
        scale: componentRow?.target.outputScale ?? rawScale,
        beta: componentRow?.target.component.beta, representative: componentRow?.representative,
      }} />}
    {inspected && examples && onSelectExampleToken && <ComponentContextProbe id={inspected} checkpoint={checkpoint}
      index={index} examples={examples} onSelect={onSelectExampleToken} />}
    <CircuitCompositionExplorer checkpoint={checkpoint} measurement={composition.measurement} example={example}
      anchor={anchor.anchor_id} onSelectToken={onSelectToken} model={model}
      unavailable={status} />
  </div>;
}
