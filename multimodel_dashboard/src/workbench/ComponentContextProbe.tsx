import { useEffect, useMemo, useState } from "react";
import { componentIdentity } from "./circuitComposition";
import { activationAt, readWorkbenchAsset } from "./negationWorkbenchData";
import type { TokenExample, TokenGraphAsset, TokenIndex, TokenPosition } from "./tokenDiscovery";

type Sample = { example: TokenExample; position: TokenPosition; gate: number; text: string };
type Row = Sample & { coefficient?: number; magnitude?: number; error?: string };
type Mode = "gate" | "coefficient" | "magnitude";
const modes: Record<Mode, string> = { gate: "Shared gate gˢ", coefficient: "Signed component e", magnitude: "Contribution ‖eu‖₂" };

export function sampleComponentContexts(examples: readonly TokenExample[], gate: string) {
  return examples.slice(0, 6).flatMap(example => {
    const positions = example.positions;
    const rank = [...positions].sort((a, b) => activationAt(b, gate) - activationAt(a, gate));
    const inactive = positions.find(position => activationAt(position, gate) === 0);
    const candidates = new Set([positions.find(position => position.final_query), ...rank.filter(p => activationAt(p, gate) > 0).slice(0, 2), inactive]);
    const prompt = Array.from(example.prompt);
    return [...candidates].filter((row): row is TokenPosition => row !== undefined).map(position => {
      const i = positions.indexOf(position);
      const text = prompt.slice(positions[i - 1]?.character_boundary ?? 0, position.character_boundary).join("");
      return { example, position, gate: activationAt(position, gate), text };
    });
  });
}

export default function ComponentContextProbe({ id, checkpoint, index, examples, onSelect }: {
  id: string; checkpoint: string; index?: TokenIndex; examples: TokenExample[];
  onSelect: (row: TokenExample, anchor: string) => void;
}) {
  const [mode, setMode] = useState<Mode>("gate");
  const [load, setLoad] = useState(false);
  const [resolved, setResolved] = useState<{ key: string; rows: Row[] }>();
  const gate = componentIdentity(id).gateId;
  const samples = useMemo(() => sampleComponentContexts(examples, gate), [examples, gate]);
  const key = JSON.stringify([id, checkpoint, ...samples.map(sample => sample.position.anchor_id)]);
  useEffect(() => {
    if (!load || !index || index.checkpointSha256 !== checkpoint) return;
    const controller = new AbortController();
    async function read() {
      const result: Row[] = [];
      for (const row of samples) {
        if (controller.signal.aborted) return;
        if (row.gate === 0) {
          result.push({ ...row, coefficient: 0, magnitude: 0 });
          continue;
        }
        const ref = index?.graphs[row.position.anchor_id];
        if (!ref) { result.push(row); continue; }
        try {
          const graph = await readWorkbenchAsset<TokenGraphAsset>(`tokens/${ref.path}`, controller.signal, ref.sha256);
          if (graph.checkpointSha256 !== checkpoint || graph.anchorId !== row.position.anchor_id
            || graph.exampleId !== row.example.example_id || graph.poolId !== row.position.pool_id) {
            throw new Error("Graph provenance differs from the selected token.");
          }
          const raw = graph.componentTargets[id];
          const selected = graph.supplementalTargets?.[id] ?? graph.official[gate]?.models[componentIdentity(id).modelId]
            ?.targets.find(target => target.component.id === id);
          result.push({ ...row, coefficient: raw?.[0] ?? selected?.coefficient,
            magnitude: raw ? Math.abs(raw[0]) * raw[1] : selected?.contributionNorm });
        } catch (error) {
          result.push({ ...row, error: error instanceof Error ? error.message : String(error) });
        }
      }
      if (!controller.signal.aborted) setResolved({ key, rows: result });
    }
    void read();
    return () => controller.abort();
  }, [samples, key, load, index, checkpoint, id, gate]);

  const rows: Row[] = resolved?.key === key ? resolved.rows : samples.map(row => row.gate ? row : { ...row, coefficient: 0, magnitude: 0 });
  const value = (row: Row) => row[mode];
  const ranked = [...rows].sort((a, b) => Number(value(a) === undefined) - Number(value(b) === undefined)
    || Math.abs(value(b) ?? 0) - Math.abs(value(a) ?? 0));
  const maximum = Math.max(0, ...rows.flatMap(row => value(row) === undefined ? [] : [Math.abs(value(row)!)]));
  const available = rows.filter(row => value(row) !== undefined).length;
  return <details className="ci-inspector"><summary>Component contexts · {samples.length} sampled tokens</summary>
    <p>Each condition contributes the final query, up to two strong gate firings and an inactive control. Rankings use the sampled tokens.</p>
    <label>Rank and color by <select value={mode} onChange={event => {
      const next = event.target.value as Mode;
      setMode(next);
      if (next !== "gate") setLoad(true);
    }}>{Object.entries(modes).map(([key, title]) => <option key={key} value={key}>{title}</option>)}</select></label>
    {mode !== "gate" && load && resolved?.key !== key && index && <p role="status">Loading verified cached coefficients…</p>}
    <p>{available}/{rows.length} available · {mode === "coefficient" ? "blue positive, rust negative" : "teal magnitude"} · shared scale {maximum.toPrecision(4)}</p>
    <ul className="ci-contexts">{ranked.map(row => {
      const n = value(row);
      const opacity = maximum && n !== undefined ? Math.abs(n) / maximum : 0;
      const color = mode === "coefficient" ? (n !== undefined && n < 0 ? "178 85 47" : "65 108 163") : "35 152 139";
      return <li key={row.position.anchor_id}><button type="button" onClick={() => onSelect(row.example, row.position.anchor_id)}
        title={row.error ?? `${row.example.example_id} · ${row.position.anchor_id}`}
        style={{ backgroundColor: `rgb(${color} / ${.25 * opacity})` }}>
        {row.example.condition.replaceAll("_", " ")} · {JSON.stringify(row.text)}
        {" · "}{n === undefined ? "Unavailable" : n.toPrecision(4)}
      </button></li>;
    })}</ul>
  </details>;
}
