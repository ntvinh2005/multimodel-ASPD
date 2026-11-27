import { useState } from "react";
import { MODEL_NAMES, type EvalModel } from "./negationWorkbenchData";
import type { Anchor } from "./SemanticDiscovery";
import {
  anchorsForPattern, recurringParents, sharedParentGates,
  type PatternEdge, type WitnessAt,
} from "./semanticPatterns";

type Props = {
  anchors: Record<string, Anchor>;
  ordered: string[];
  targets: [string, string];
  models: [EvalModel, EvalModel];
  selected: PatternEdge[];
  witnessAt: WitnessAt;
  onChange: (edges: PatternEdge[]) => void;
  onAnchor: (id: string) => void;
};

function ContextPicker({ label, ids, anchors, selected, onChange, onAnchor }: {
  label: string; ids: string[]; anchors: Props["anchors"]; selected: string;
  onChange: (id: string) => void; onAnchor: Props["onAnchor"];
}) {
  const id = ids.includes(selected) ? selected : ids[0];
  const example = anchors[id];
  return <div>
    <label className="discovered-picker">{label} ({ids.length})
      <select className="uw-form-control" value={id ?? ""} disabled={!example}
        onChange={event => onChange(event.target.value)}>
        {!example && <option value="">No measured examples</option>}
        {ids.map(anchor => <option value={anchor} key={anchor}>
          {anchors[anchor].source} · pool {anchors[anchor].pool} · {anchor.slice(0, 12)}
        </option>)}
      </select>
    </label>
    {example && <>
      <p className="ig-sequence td-sequence semantic-contribution-sequence">{example.prefix.length > 220 && "…"}
        {Array.from(example.prefix).slice(-220).join("")}</p>
      <button className="uw-form-control" type="button" onClick={() => onAnchor(id)}>Open this anchor in the graphs</button>
    </>}
  </div>;
}

function terminalUnit(prefix: string) {
  return prefix.trimEnd().match(/[\p{L}\p{N}_]+$|[^\s]$/u)?.[0].toLowerCase() ?? "";
}

function rankContrasts(ids: string[], reference: Anchor | undefined, anchors: Props["anchors"]) {
  if (!reference) return ids;
  const terminal = terminalUnit(reference.prefix);
  const rank = (row: Anchor) => +(row.source === reference.source) + 2 * +(terminalUnit(row.prefix) === terminal);
  return [...ids].sort((a, b) => rank(anchors[b]) - rank(anchors[a])
    || Math.abs(reference.prefix.length - anchors[a].prefix.length)
      - Math.abs(reference.prefix.length - anchors[b].prefix.length));
}

function TerminalCounts({ present, contrast, anchors }: {
  present: string[]; contrast: string[]; anchors: Props["anchors"];
}) {
  const count = (ids: string[]) => {
    const forms = new Map<string, number>();
    for (const id of ids) {
      const form = terminalUnit(anchors[id].prefix);
      if (form) forms.set(form, (forms.get(form) ?? 0) + 1);
    }
    return forms;
  };
  const withPattern = count(present), withoutPattern = count(contrast);
  const repeated = [...withPattern].filter(([, n]) => n > 1)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  if (!repeated.length) return null;
  return <details>
    <summary>Repeated terminal text</summary>
    <p>Check whether repeated wording could explain a pattern before comparing the surrounding contexts.</p>
    <div className="cc-scroll"><table className="cc-table">
      <thead><tr><th>Terminal text</th><th>All edges present</th><th>Contrast</th></tr></thead>
      <tbody>{repeated.map(([form, count]) => <tr key={form}>
        <th><code>{form}</code></th><td>{count}/{present.length}</td><td>{withoutPattern.get(form) ?? 0}/{contrast.length}</td>
      </tr>)}</tbody>
    </table></div>
  </details>;
}

export default function SemanticPatternInspector({
  anchors, ordered, targets, models, selected, witnessAt, onChange, onAnchor,
}: Props) {
  const [minimum, setMinimum] = useState("repeat");
  const [present, setPresent] = useState("");
  const [contrast, setContrast] = useState("");
  const common = sharedParentGates(ordered, targets, witnessAt);
  const contexts = anchorsForPattern(ordered, selected, witnessAt);
  const reference = anchors[contexts.present.includes(present) ? present : contexts.present[0]];
  return <section className="semantic-pattern-inspector" aria-label="Recurring parent patterns">
    <p>Counts use all measured anchors in this split. Choose several exact parent edges to compare contexts where they occur together with contexts where at least one is outside the saved top 10.</p>
    <label className="discovered-picker">Minimum parent recurrence
      <select className="uw-form-control" value={minimum} onChange={event => setMinimum(event.target.value)}>
        <option value="repeat">At least twice</option>
        <option value="0.25">At least 25% of measured anchors</option>
        <option value="0.5">At least 50% of measured anchors</option>
        <option value="1">Every measured anchor</option>
      </select>
    </label>
    <div className="negation-context-grid">
      {targets.map((target, index) => {
        const { measured, rows } = recurringParents(ordered, target, witnessAt);
        const existing = selected.filter(edge => edge.target === target);
        const eligible = rows.filter(row => row.count >= (minimum === "repeat" ? 2 : measured * Number(minimum))
          && !existing.some(edge => edge.source === row.id));
        const add = (source: string) => { if (source) onChange([...selected, { source, target }]); };
        return <div key={target}>
          <label className="discovered-picker">{MODEL_NAMES[models[index]]} parents · {measured} measured anchors
            <select className="uw-form-control" value="" onChange={event => add(event.target.value)}>
              <option value="">{eligible.length ? "Add a measured parent edge" : "No more parents at this recurrence"}</option>
              {eligible.map(({ id, count }) => <option key={id} value={id}>
                {id} · {count}/{measured} ({Math.round(100 * count / measured)}%)
              </option>)}
            </select>
          </label>
          {existing.map(edge => {
            const count = rows.find(row => row.id === edge.source)?.count ?? 0;
            return <button key={edge.source} type="button" className="uw-form-control"
              aria-label={`Remove ${edge.source} from the pattern`} onClick={() => onChange(selected.filter(item => item !== edge))}>
              {edge.source} · {count}/{measured} ×
            </button>;
          })}
        </div>;
      })}
    </div>
    {common.rows.length > 0 && <details>
      <summary>Shared parent gates at the same anchors</summary>
      <p>Each gate counts once per anchor, even when several of its components appear in one graph.</p>
      <ul>{common.rows.map(({ id, count }) => <li key={id}><code>{id}</code> · {count}/{common.measured}</li>)}</ul>
    </details>}
    {selected.length > 0 && <>
      <p>{contexts.present.length}/{contexts.present.length + contexts.contrast.length} measured anchors contain every selected edge.
        {contexts.unavailable > 0 && ` ${contexts.unavailable} anchors lack a required target measurement.`}
        {" "}The graphs focus on selected edges at the current anchor. Compare their signed values at each position.</p>
      <div className="negation-context-grid">
        <ContextPicker label="All selected edges present" ids={contexts.present} anchors={anchors}
          selected={present} onChange={setPresent} onAnchor={onAnchor} />
        <ContextPicker label="Contrast · at least one edge outside the top 10"
          ids={rankContrasts(contexts.contrast, reference, anchors)} anchors={anchors}
          selected={contrast} onChange={setContrast} onAnchor={onAnchor} />
      </div>
      <small>Contrasts are ordered by terminal text, source, and prefix length relative to the selected example.</small>
      <TerminalCounts present={contexts.present} contrast={contexts.contrast} anchors={anchors} />
    </>}
  </section>;
}
