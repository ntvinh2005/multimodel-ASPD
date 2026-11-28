import { useState } from "react";
import { MODEL_NAMES, type EvalModel } from "./negationWorkbenchData";
import { componentIdentity } from "./circuitComposition";
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
  onToggle: (source: string, target: string) => void;
  onAnchor: (id: string) => void;
};

function ContextPicker({ label, ids, anchors, selected, onChange, onAnchor }: {
  label: string; ids: string[]; anchors: Props["anchors"]; selected: string;
  onChange: (id: string) => void; onAnchor?: Props["onAnchor"];
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
      {onAnchor && <button className="uw-form-control" type="button" onClick={() => onAnchor(id)}>Open this anchor in the graphs</button>}
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

function TerminalCounts({ present, contrast, anchors, compareCircuits = false }: {
  present: string[]; contrast: string[]; anchors: Props["anchors"]; compareCircuits?: boolean;
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
  const forms = compareCircuits ? new Set([...withPattern.keys(), ...withoutPattern.keys()]) : new Set(withPattern.keys());
  const repeated = [...forms].map(form => [form, withPattern.get(form) ?? 0, withoutPattern.get(form) ?? 0] as const)
    .filter(([, left, right]) => left > 1 || (compareCircuits && right > 1))
    .sort((a, b) => b[1] + b[2] - a[1] - a[2] || a[0].localeCompare(b[0]));
  if (!repeated.length) return null;
  return <details>
    <summary>Repeated terminal text</summary>
    <p>Check whether repeated wording could explain a pattern before comparing the surrounding contexts.</p>
    <div className="cc-scroll"><table className="cc-table">
      <thead><tr><th>Terminal text</th><th>{compareCircuits ? "Left circuit" : "All edges present"}</th>
        <th>{compareCircuits ? "Right circuit" : "Contrast"}</th></tr></thead>
      <tbody>{repeated.map(([form, left, right]) => <tr key={form}>
        <th><code>{form}</code></th><td>{left}/{present.length}</td><td>{right}/{contrast.length}</td>
      </tr>)}</tbody>
    </table></div>
  </details>;
}

type CircuitSupport = {
  target: string;
  model: EvalModel;
  measured: number;
  parents: { id: string; count: number }[];
  examples: string[];
};

export function CircuitExampleComparison({ circuits, anchors, topK, maximum, onTopK, onCompare }: {
  circuits: readonly [CircuitSupport, CircuitSupport];
  anchors: Props["anchors"];
  topK: number;
  maximum: number;
  onTopK: (value: number) => void;
  onCompare: (left: string, right: string) => void;
}) {
  const [leftPick, setLeftPick] = useState("");
  const [rightPick, setRightPick] = useState("");
  const [distinct, setDistinct] = useState(false);
  const [left, right] = circuits;
  const leftIds = new Set(left.examples), rightIds = new Set(right.examples);
  const overlap = right.examples.filter(id => leftIds.has(id)).length;
  const leftExamples = distinct ? left.examples.filter(id => !rightIds.has(id)) : left.examples;
  const rightExamples = distinct ? right.examples.filter(id => !leftIds.has(id)) : right.examples;
  const leftId = leftExamples.includes(leftPick) ? leftPick : leftExamples[0];
  const otherDocument = rightExamples.find(id => anchors[id].document !== anchors[leftId]?.document);
  const rightId = rightExamples.includes(rightPick) ? rightPick : otherDocument ?? rightExamples[0];
  return <section className="semantic-pattern-inspector" aria-label="Circuit example comparison">
    <label className="uw-threshold-label semantic-top-k">
      <span>Top K recurring parents per circuit</span><output>{topK}</output>
      <input aria-label="Top K recurring parents per circuit" type="range" min="1" max={maximum} step="1"
        value={topK} onChange={event => onTopK(Number(event.target.value))} />
    </label>
    <p>Each side chooses its own most frequent parent components and shows only examples where its chosen edges all appear together.</p>
    <div className="semantic-pattern-toolbar">
      <small>{overlap} shared anchors in these two example sets.</small>
      <label className="ig-graph-filter">
        <input type="checkbox" checked={distinct} onChange={event => setDistinct(event.target.checked)} />
        Hide shared anchors
      </label>
    </div>
    <div className="negation-context-grid">
      {circuits.map((side, index) => <div key={side.target} className="semantic-circuit-side">
        <strong>{MODEL_NAMES[side.model]} <code>{side.target}</code></strong>
        <small>{side.examples.length}/{side.measured} measured anchors contain all {side.parents.length} edges</small>
        <ol className="semantic-circuit-parents">{side.parents.map(({ id, count }) => <li key={id}>
          <code>{id}</code> <small>{count}/{side.measured}</small>
        </li>)}</ol>
        <ContextPicker label={index === 0 ? "Left circuit examples" : "Right circuit examples"}
          ids={index === 0 ? leftExamples : rightExamples} anchors={anchors} selected={index === 0 ? leftId : rightId}
          onChange={index === 0 ? setLeftPick : setRightPick} />
      </div>)}
    </div>
    <div className="semantic-pattern-toolbar">
      <button className="uw-form-control" type="button" disabled={!leftId || !rightId}
        onClick={() => onCompare(leftId, rightId)}>Show both examples in the graphs</button>
    </div>
    <TerminalCounts present={leftExamples} contrast={rightExamples} anchors={anchors} compareCircuits />
  </section>;
}

const minimumOptions = [
  ["repeat", "≥2 anchors"], ["0.25", "≥25%"], ["0.5", "≥50%"], ["1", "Every anchor"],
];

export default function SemanticPatternInspector({
  anchors, ordered, targets, models, selected, witnessAt, onToggle, onAnchor,
}: Props) {
  const [minimum, setMinimum] = useState("repeat");
  const [search, setSearch] = useState("");
  const [present, setPresent] = useState("");
  const [contrast, setContrast] = useState("");
  const common = sharedParentGates(ordered, targets, witnessAt);
  const contexts = anchorsForPattern(ordered, selected, witnessAt);
  const reference = anchors[contexts.present.includes(present) ? present : contexts.present[0]];
  const query = search.trim().toLowerCase();
  return <section className="semantic-pattern-inspector" aria-label="Recurring parent patterns">
    <label className="discovered-picker">Find parent by component or gate ID
      <input className="uw-form-control" type="search" value={search} onChange={event => setSearch(event.target.value)}
        placeholder="MID:515 or layer/matrix ID" />
    </label>
    <div className="semantic-pattern-toolbar" role="group" aria-label="Minimum parent recurrence">
      {minimumOptions.map(([value, label]) => <button key={value} type="button" aria-pressed={minimum === value}
        onClick={() => setMinimum(value)}>{label}</button>)}
    </div>
    <div className="negation-context-grid">
      {targets.map((target, index) => {
        const { measured, rows } = recurringParents(ordered, target, witnessAt);
        const existing = selected.filter(edge => edge.target === target);
        const eligible = rows.filter(row => row.count >= (minimum === "repeat" ? 2 : measured * Number(minimum))
          && (row.id.toLowerCase().includes(query) || componentIdentity(row.id).gateId.toLowerCase().includes(query)));
        return <div key={target}>
          <strong>{MODEL_NAMES[models[index]]} parents · {measured} measured anchors</strong>
          {existing.length > 0 && <div className="semantic-pattern-chips">{existing.map(edge =>
            <button key={edge.source} type="button" aria-label={`Remove ${edge.source} from the pattern`}
              onClick={() => onToggle(edge.source, target)}>{edge.source} ×</button>,
          )}</div>}
          {eligible.length ? <ul className="semantic-parent-list" aria-label={`Measured parents for ${target}`}>
            {eligible.map(({ id, count }) => {
              const active = existing.some(edge => edge.source === id);
              return <li key={id}><button type="button" aria-pressed={active} onClick={() => onToggle(id, target)}>
                <code>{id}</code><small>{count}/{measured} · {Math.round(100 * count / measured)}%</small>
              </button></li>;
            })}
          </ul> : <p>No parents match this search and recurrence.</p>}
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
        {contexts.unavailable > 0 && ` ${contexts.unavailable} anchors lack a required target measurement.`}</p>
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
