import { useEffect, useMemo, useState } from "react";
import { CircuitSelect, type CircuitView } from "./InductionGateExplorer";
import { QueryGraphPanel, type GraphRenderer } from "./InductionQueryGraph";
import { searchSharedGates } from "./graphQueries";
import SharedGateSearch from "./SharedGateSearch";
import {
  activationAt, CONDITION_OUTCOMES, FAMILY_LABELS, graphAtFinalQuery, MODEL_IDS, MODEL_NAMES,
  NEGATION_LABELS, readNegationAsset, validateNegationIndex,
  type EvalModel, type MatrixLocator, type NegationCondition, type NegationExample,
  type NegationGate, type NegationGroup, type NegationIndex, type Outcome,
} from "./negationWorkbenchData";

type Split = keyof NegationIndex["splits"];
const numeric = (value: number | null | undefined) => value == null ? "—" : value.toPrecision(4);

function useAsset<T>(path: string | null, hash?: string) {
  const [result, setResult] = useState<{ path: string; data?: T; error?: string }>();
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    void readNegationAsset<T>(path, controller.signal, hash).then(
      (data) => { if (!controller.signal.aborted) setResult({ path, data }); },
      (error: Error) => { if (!controller.signal.aborted) setResult({ path, error: error.message }); },
    );
    return () => controller.abort();
  }, [path, hash]);
  return result?.path === path ? result : undefined;
}

export default function NegationGateExplorer({ onCircuitView, Graph }: {
  onCircuitView: (view: CircuitView) => void;
  Graph: GraphRenderer;
}) {
  const asset = useAsset<NegationIndex>("index.json");
  let index: NegationIndex | undefined;
  let error = asset?.error;
  if (asset?.data) {
    try { index = validateNegationIndex(asset.data); }
    catch (failure) { error = failure instanceof Error ? failure.message : String(failure); }
  }
  return <div className="uw-shell induction-shell negation-shell">
    <header className="uw-header">
      <div><span className="uw-kicker">MULTI-MODEL ASPD</span><h1>Negation Gate Explorer</h1>
        <p>Explore matched polarity and scope controls, gate activations, and final-query dependencies.</p></div>
      <CircuitSelect value="negation" onChange={onCircuitView} />
    </header>
    {index ? <NegationPanel index={index} Graph={Graph} />
      : <p className="uw-empty" role="status">{error ?? "Loading measured Negation data…"}</p>}
  </div>;
}

function NegationPanel({ index, Graph }: { index: NegationIndex; Graph: GraphRenderer }) {
  const [splitName, setSplitName] = useState<Split>("discovery");
  const [site, setSite] = useState("ALL");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("contrast");
  const [graphOnly, setGraphOnly] = useState(true);
  const [gateId, setGateId] = useState("");
  const [groupId, setGroupId] = useState("");
  const [condition, setCondition] = useState<NegationCondition>("negative");
  const [model, setModel] = useState<EvalModel>("gemma");
  const [outcome, setOutcome] = useState<Outcome | "">("");
  const [peer, setPeer] = useState("");
  const split = index.splits[splitName];
  const locator = useAsset<{ mechanism: string; matrices: MatrixLocator[] }>("locator.json", index.assetHashes["locator.json"]);
  const gates = useMemo(() => {
    const rows = split.gates.filter((gate) => (site === "ALL" || gate.id.startsWith(site + ":"))
      && (!graphOnly || gate.id in split.selection));
    rows.sort((left, right) => {
      if (sort === "fires") return queryFires(right) - queryFires(left);
      if (sort === "beta") return (maxBeta(right) ?? -1) - (maxBeta(left) ?? -1);
      return Math.abs(right.contrast) - Math.abs(left.contrast);
    });
    return searchSharedGates(rows, query);
  }, [split, site, graphOnly, query, sort]);
  const gate = gates.find((item) => item.id === gateId) ?? gates[0];
  const contextGroups = gate ? [...new Set([
    ...(split.selection[gate.id]?.groups ?? []), ...(split.gateGroups[gate.id] ?? []),
  ])] : [];
  const groupOptions = outcome
    ? Object.keys(split.groups).filter((id) => split.groups[id].outcomes.negative?.[model] === outcome)
    : [...new Set([...contextGroups, ...(groupId in split.groups ? [groupId] : [])])];
  const selectedGroup = groupOptions.includes(groupId) ? groupId : groupOptions[0];
  const groupMeta = selectedGroup ? split.groups[selectedGroup] : undefined;
  const source = groupMeta ? `${splitName}/${groupMeta.path}` : null;
  const groupAsset = useAsset<NegationGroup>(source, source ? index.assetHashes[source] : undefined);
  const group = groupAsset?.data?.mechanism === "negation" && groupAsset.data.groupId === selectedGroup
    ? groupAsset.data : undefined;
  const example = group?.examples.find((record) => record.condition === condition) ?? group?.examples[0];
  const graph = example && gate ? graphAtFinalQuery(example, gate.id) : undefined;
  const final = example?.positions.find((position) => position.final_query);

  function selectGate(id: string) {
    setGateId(id);
    setGroupId("");
    setPeer("");
  }

  function exploreSameExample(id: string) {
    if (id === gate?.id || !split.gates.some((item) => item.id === id)) return;
    setPeer(gate?.id ?? "");
    setGateId(id);
    setGroupId(selectedGroup ?? "");
    setQuery(id);
    setSite("ALL");
    setGraphOnly(false);
    setOutcome("");
  }

  return <>
    <div className="uw-warning"><strong>MEASURED · {splitName.toUpperCase()}</strong>
      Dependencies are measured at the final aligned query for the discovery shortlist. PRE and MID identify distinct shared gates.
    </div>
    <section className="ig-toolbar">
      <label>Split<select value={splitName} onChange={(event) => {
        const next = event.target.value as Split;
        setSplitName(next);
        setGraphOnly(next === "discovery");
        setOutcome("");
        selectGate("");
      }}>
        <option value="smoke">Smoke · {Object.keys(index.splits.smoke.groups).length} groups</option>
        <option value="discovery">Discovery · {Object.keys(index.splits.discovery.groups).length} groups</option>
      </select></label>
      <label>Site<select value={site} onChange={(event) => setSite(event.target.value)}>
        <option>ALL</option><option>PRE</option><option>MID</option>
      </select></label>
      <label>Sort<select value={sort} onChange={(event) => setSort(event.target.value)}>
        <option value="contrast">|Negative − affirmative|</option>
        <option value="fires">Final-query fire count</option>
        <option value="beta">Maximum representative β</option>
      </select></label>
      <div className="ig-run-meta"><strong>{split.gates.length.toLocaleString()}</strong><span>active gates</span>
        <strong>{split.poolCount}</strong><span>BatchTopK pools</span></div>
    </section>
    <main className="ig-layout">
      <aside className="ig-gate-list" aria-label="Shared gates">
        <header>
          <div><strong>Shared gates</strong><span>showing {Math.min(250, gates.length)} / {gates.length}</span></div>
          <SharedGateSearch value={query} onChange={setQuery} onSelectFirst={() => {
            if (gates[0]) selectGate(gates[0].id);
          }} />
          {splitName === "discovery" && <label className="ig-graph-filter">
            <input type="checkbox" checked={graphOnly} onChange={(event) => setGraphOnly(event.target.checked)} />
            With measured graphs ({Object.keys(split.selection).length})
          </label>}
        </header>
        {gates.slice(0, 250).map((item) => <button type="button" key={item.id}
          className={item.id === gate?.id ? "selected" : ""} aria-pressed={item.id === gate?.id}
          onClick={() => selectGate(item.id)}>
          <strong>{item.id}</strong><span>Δ {numeric(item.contrast)}</span>
          <small>{queryFires(item)} query fires · β {numeric(maxBeta(item))}</small>
          {item.id in split.selection && <small className="ig-graph-indicator">Measured graph</small>}
        </button>)}
      </aside>
      {gate ? <section className="ig-detail">
        <header><div><span className="uw-eyebrow">SHARED GATE</span><h2>{gate.id}</h2></div>
          <span className="ig-badges"><span>Final-query contrast {numeric(gate.contrast)}</span></span></header>
        <section className="ig-condition-grid negation-conditions">
          {Object.entries(gate.conditions).map(([key, row]) => <div key={key}>
            <strong>{NEGATION_LABELS[key as NegationCondition]}</strong>
            <span>{row.queryFires}/{split.denominators[key].queries} query fires · mean gˢ {numeric(row.queryMean)}</span>
            <small>{row.sequenceFires}/{split.denominators[key].examples} sequences with firing</small>
            <small>All-anchor mean {numeric(row.mean)} · when firing {numeric(row.meanWhenFiring)}</small>
          </div>)}
        </section>
        <details className="ig-graph-status"><summary>Activation metric definitions</summary>
          <p>Contrast is the mean across the two negative query means minus the mean across the two affirmative query means. Unconditional means include natural zero support. “When firing” averages selected positive gate values. Scope controls are separate.</p>
          <dl>{Object.entries(gate.conditions).map(([conditionId, row]) => <div key={conditionId}>
            <dt>{NEGATION_LABELS[conditionId as NegationCondition]}</dt>
            <dd>{split.denominators[conditionId].anchors.toLocaleString()} aligned anchors · SD {numeric(row.sd)}
              {Object.entries(row.roles).map(([role, strength]) => <span key={role}> · {role.replaceAll("_", " ")} mean {numeric(strength)}</span>)}
            </dd>
          </div>)}</dl>
        </details>
        <section className="ig-examples">
          <header><div><span className="uw-eyebrow">MATCHED EXAMPLES</span><h3>Activation and dependency</h3></div></header>
          <div className="ig-toolbar negation-example-toolbar">
            <label>Example group<select value={selectedGroup ?? ""} onChange={(event) => setGroupId(event.target.value)}>
              {!groupOptions.length && <option value="">No matching examples</option>}
              {groupOptions.map((id) => <option key={id} value={id}>
                {id.slice("negation-".length)} · {FAMILY_LABELS[split.groups[id].family]}
              </option>)}
            </select></label>
            <label>Condition<select value={example?.condition ?? condition}
              onChange={(event) => setCondition(event.target.value as NegationCondition)}>
              {groupMeta?.conditions.map((key) => <option key={key} value={key}>{NEGATION_LABELS[key]}</option>)}
            </select></label>
            <label>Model<select value={model} onChange={(event) => setModel(event.target.value as EvalModel)}>
              {MODEL_IDS.map((key) => <option key={key} value={key}>{MODEL_NAMES[key]}</option>)}
            </select></label>
            <label>Negative outcome<select value={outcome} onChange={(event) => {
              setOutcome(event.target.value as Outcome | "");
              if (event.target.value) setCondition("negative");
            }}>
              <option value="">All outcomes</option>
              {Object.entries(CONDITION_OUTCOMES).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
            </select></label>
          </div>
          {example && group ? <>
            <p className="ig-graph-status"><b>{NEGATION_LABELS[example.condition]}</b> · {example.example_id}
              {" · final query gˢ "}{numeric(final ? activationAt(final, gate.id) : null)}</p>
            {graph && graph.activation > 0 ? <QueryGraphPanel
              key={`${gate.id}/${example.example_id}/${model}`} graph={graph} Graph={Graph} model={model}
              onExploreGate={exploreSameExample} />
              : <p className="ig-graph-status" role="status">{graph
                ? "The gate did not fire at this measured final query."
                : "No dependency graph was measured for this gate and exact example."}</p>}
            <p className="ig-graph-status">Anchor {final?.anchor_id} · {MODEL_NAMES[model]} token {final?.model_positions[model]}</p>
            <label className="uw-gate-search negation-peer-search">
              Compare a second gate on these sequences
              <input type="search" placeholder="e.g. PRE:4312" value={peer} onChange={(event) => setPeer(event.target.value)} />
            </label>
            <MatchedContexts examples={group.examples} selected={example.condition} onSelect={setCondition}
              gate={gate.id} peer={split.gates.find((item) => item.id === peer.trim().toUpperCase())?.id} />
          </> : <p className="uw-empty" role="status">{groupAsset?.error ?? (groupOptions.length
            ? "Loading matched examples…" : "No activation contexts match this selection.")}</p>}
        </section>
        <NegationBetaLocator gate={gate.id} matrices={locator?.data?.mechanism === "negation"
          ? locator.data.matrices : undefined} error={locator?.error} />
      </section> : <p className="uw-empty">No gates match these filters.</p>}
    </main>
    <footer className="uw-footer"><span>Checkpoint 61,036 · {split.poolCount} × 2,048 aligned anchors</span>
      <span>{split.graphProvenance && <>Graph job {split.graphProvenance.runtime.slurm_job_id} · </>}{split.manifestSha256.slice(0, 12)}</span>
    </footer>
  </>;
}

function queryFires(gate: NegationGate) {
  return Object.values(gate.conditions).reduce((total, row) => total + row.queryFires, 0);
}

function maxBeta(gate: NegationGate) {
  const values = Object.values(gate.referenceBeta).filter((value) => value !== null);
  return values.length ? Math.max(...values) : null;
}

function MatchedContexts({ examples, selected, onSelect, gate, peer }: {
  examples: NegationExample[];
  selected: NegationCondition;
  onSelect: (value: NegationCondition) => void;
  gate: string;
  peer?: string;
}) {
  const keys = peer && peer !== gate ? [gate, peer] : [gate];
  const scale = Math.max(0, ...examples.flatMap((row) => row.positions.flatMap((position) =>
    keys.map((key) => activationAt(position, key)),
  )));
  return <section className="ig-examples">
    <header><div><span className="uw-eyebrow">EXACT CANONICAL ANCHORS</span><h3>Matched full-sequence activations</h3></div>
      <p>Shared color scale gˢ 0–{numeric(scale)} across these examples and gates. The outline marks the prediction position.</p>
    </header>
    <div className="negation-context-grid">{examples.map((row) => <article key={row.example_id}
      className={`ig-context ${row.condition === selected ? "negation-context-selected" : ""}`}>
      <header>
        <div><strong>{NEGATION_LABELS[row.condition]}</strong><span>{row.example_id}</span></div>
        <button type="button" onClick={() => onSelect(row.condition)}>{row.condition === selected ? "Selected" : "Inspect this condition"}</button>
      </header>
      {keys.map((id) => <SequenceHeatmap key={id} example={row} gate={id} scale={scale} />)}
      <p>Valid {row.valid_alternatives.join(", ")}{row.excluded_candidates.length > 0 && <> · excluded {row.excluded_candidates.join(", ")}</>}</p>
      <p>Fixed margin L = logit({row.candidate_orientation[0]}) − logit({row.candidate_orientation[1]})</p>
      <div className="ig-predictions">{MODEL_IDS.map((model) => {
        const prediction = row.predictions[model];
        return <span key={model}><b>{MODEL_NAMES[model]}</b> {JSON.stringify(prediction.predicted_token)}
          {" · L "}{numeric(prediction.fixed_margin)}{" · correctness margin "}{numeric(prediction.correctness_margin)}
          {prediction.negation_outcome ? <> · {CONDITION_OUTCOMES[prediction.negation_outcome]}</>
            : prediction.exact_correct ? " · Correct" : " · Incorrect"}
        </span>;
      })}</div>
      <details><summary>Candidate logits and probabilities</summary>
        {MODEL_IDS.map((model) => <div key={model}>
          <b>{MODEL_NAMES[model]}</b>
          <p>Valid mass {numeric(row.predictions[model].valid_mass)} · excluded {numeric(row.predictions[model].excluded_mass)}
            {" · unrelated "}{numeric(row.predictions[model].unrelated_mass)}</p>
          <ul>{Object.entries(row.predictions[model].candidates).map(([word, values]) => <li key={word}>
            {word} · logit {numeric(values.logit)} · P {numeric(values.probability)}
          </li>)}</ul>
        </div>)}
      </details>
    </article>)}</div>
  </section>;
}

function SequenceHeatmap({ example, gate, scale }: { example: NegationExample; gate: string; scale: number }) {
  const prompt = Array.from(example.prompt);
  return <div>
    <small>{gate}</small>
    <p className="ig-sequence">{example.positions.map((position, index) => {
      const start = example.positions[index - 1]?.character_boundary ?? 0;
      const value = activationAt(position, gate);
      const alpha = scale > 0 ? value / scale * .9 : 0;
      return <span key={position.anchor_id} className={position.final_query ? "final-query" : ""}
        style={{ "--activation": alpha } as React.CSSProperties}
        title={`gˢ ${value} · g ${Number(value > 0)}${position.roles_ending.length ? ` · ${position.roles_ending.join(", ")}` : ""}`}>
        {prompt.slice(start, position.character_boundary).join("")}
      </span>;
    })}</p>
  </div>;
}

function NegationBetaLocator({ gate, matrices, error }: { gate: string; matrices?: MatrixLocator[]; error?: string }) {
  const [site, index] = gate.split(":");
  const feature = Number(index);
  return <section className="ig-beta">
    <header><div><span className="uw-eyebrow">IMPORTED NATURAL-VALIDATION β</span><h3>Model-specific component locator</h3></div>
      <p>All component cells for this site. White insets mark the frozen β representatives.</p></header>
    {matrices ? <div className="ig-beta-models">{MODEL_IDS.map((model) => <div key={model}>
      <strong>{MODEL_NAMES[model]}</strong><div className="ig-beta-grid">{matrices.filter((matrix) => matrix.model === model && matrix.site === site).map((row) => {
        const beta = row.beta[feature];
        return <i key={`${row.layer}/${row.matrix}`} className={row.representatives.includes(feature) ? "representative" : ""}
          style={{ "--beta": Math.min(1, beta ?? 0) } as React.CSSProperties}
          title={`L${row.layer}/${row.matrix} · β ${numeric(beta)}`}>
          <small>L{row.layer}</small><b>{row.matrix}</b>
        </i>;
      })}</div>
    </div>)}</div> : <p className="ig-graph-status" role="status">{error ?? "Loading reference β…"}</p>}
  </section>;
}
