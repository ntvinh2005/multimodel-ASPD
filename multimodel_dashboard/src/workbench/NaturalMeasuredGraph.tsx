import { useMemo, useState } from "react";
import { QueryGraphPanel, type GraphRenderer } from "./InductionQueryGraph";
import { measurementFromToken } from "./compositionMeasurements";
import { MODEL_IDS, type EvalModel } from "./negationWorkbenchData";
import { naturalGraphArchive, type NaturalGraphArchive, type TestTokenScope } from "./naturalTokenEvidence";
import { ModelSelect, useWorkbenchAsset } from "./TokenCircuitExplorer";
import { tokenGraph, type TokenExample, type TokenGraphAsset, type TokenPosition } from "./tokenDiscovery";

export function useNaturalGraph(candidate: string, checkpoint: string, scope?: TestTokenScope) {
  const directory = `natural-token-pilots/${candidate}/graphs`;
  const index = useWorkbenchAsset<NaturalGraphArchive>(scope ? `${directory}/index.json` : undefined);
  const archive = naturalGraphArchive(checkpoint, scope, index?.data);
  const ref = scope && archive?.graphs[scope.anchor];
  const path = ref && `${directory}/${ref.path}`;
  const asset = useWorkbenchAsset<TokenGraphAsset>(path, ref?.sha256);
  return useMemo(() => {
    const coverage = archive && `${archive.anchorsMeasured}/${archive.pilotAnchors} pilot anchors measured`;
    const loading = Boolean(scope && (!index || ref && !asset));
    if (!asset?.data || !scope || !path) return { coverage, loading, error: asset?.error };
    try {
      const { point, documentId } = scope;
      const support = (site: "PRE" | "MID") => {
        const entries = Object.entries(point.gates).filter(([key]) => key.startsWith(site + ":"));
        return { active_gate_ids: entries.map(([key]) => Number(key.split(":")[1])),
          values: entries.map(([, value]) => value) };
      };
      const positions = point.positions;
      if (point.characterBoundary === null || !MODEL_IDS.every(model => Number.isInteger(positions[model]))) {
        throw new Error("Native positions unavailable");
      }
      const anchor: TokenPosition = { anchor_id: scope.anchor, pool_id: String(scope.pool),
        character_boundary: point.characterBoundary, model_positions: positions as Record<EvalModel, number>,
        final_query: false, roles_ending: [], pre: support("PRE"), mid: support("MID") };
      const example: TokenExample = { example_id: documentId, group_id: documentId,
        condition: "test", prompt: "", positions: [anchor] };
      const measurement = measurementFromToken(asset.data,
        { checkpoint, dataset: scope.dataset, anchor: scope.anchor, pool: anchor.pool_id,
          example: documentId, artifact: path }, example, anchor);
      return { coverage, loading, asset: asset.data, measurement, anchor, example };
    } catch (error) {
      return { coverage, loading, error: String(error) };
    }
  }, [archive, scope, index, ref, asset, checkpoint, path]);
}

export default function NaturalMeasuredGraph({ data, scope, checkpoint, gate, Graph, onGate, onInspectComponent, onReturn }: {
  data: ReturnType<typeof useNaturalGraph>; scope: TestTokenScope; checkpoint: string; gate: string;
  Graph: GraphRenderer; onGate: (gate: string) => void; onInspectComponent: (id: string) => void; onReturn: () => void;
}) {
  const [model, setModel] = useState<EvalModel>("gemma");
  const choices = Object.keys(scope.point.gates).sort((a, b) => scope.point.gates[b] - scope.point.gates[a]);
  const selected = choices.includes(gate) ? gate : choices[0];
  const graph = useMemo(() => {
    if (!data.asset || !data.example || !data.anchor || !selected) return undefined;
    try { return tokenGraph(data.asset, data.example, data.anchor, checkpoint, selected, model); }
    catch (error) { return { error: String(error) }; }
  }, [data.asset, data.example, data.anchor, checkpoint, selected, model]);
  const issue = data.error ?? (graph && "error" in graph ? graph.error : undefined);
  return <>
    <div className="uw-graph-toolbar">
      <button type="button" onClick={onReturn}>Return to saved validation graph</button>
      <span>{data.coverage ?? "Held-out test token"}</span>
      {data.measurement && <><ModelSelect model={model} onChange={setModel} />
        {selected && <label>Active gate <select value={selected} onChange={event => onGate(event.target.value)}>
          {choices.map(id => <option key={id}>{id}</option>)}
        </select></label>}</>}
    </div>
    {issue ? <p className="uw-edge-state" role="alert">{issue}</p>
      : data.loading ? <p className="uw-edge-state" role="status">Loading this token’s dependencies…</p>
      : graph && "graph" in graph && graph.graph ? <QueryGraphPanel
        key={`${scope.anchor}/${selected}/${model}`} graph={graph.graph} model={model} Graph={Graph}
        gateId={selected} traceTargets={graph.targets} completeComponents={data.asset?.completeComponents}
        compositionMeasurement={data.measurement} annotationCheckpoint={checkpoint}
        evidenceReference={`Held-out test anchor ${scope.anchor}`} anchorLabel="SELECTED TOKEN"
        onExploreGate={onGate} onInspectComponent={onInspectComponent} />
      : data.measurement ? <p className="uw-edge-state">No active target components in this token’s measured graph.</p>
      : <p className="uw-edge-state">No dependency graph is saved for this test token.</p>}
    <p className="uw-edge-state"><code>{scope.anchor}</code></p>
  </>;
}
