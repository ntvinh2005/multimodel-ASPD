import { useEffect, useMemo, useState } from "react";
import type { GraphRenderer } from "./InductionQueryGraph";
import { searchSharedGates } from "./graphQueries";
import SharedGateSearch from "./SharedGateSearch";
import { ResearchGateLabel } from "./ResearchNotes";
import ComponentBetaLocator from "./ComponentBetaLocator";
import { ModelSelect, SelectedNoteEditor, TokenAnchorBar, TokenDependencyInspector, TokenSequence, useGraphWitnesses, useSelectedNote, useTokenGroup, useTokenIndex } from "./TokenCircuitExplorer";
import { filterTokenGates, graphWitnessForGate, selectExample, selectGroup, selectToken, tokenGates, type GraphWitness, type TokenExample } from "./tokenDiscovery";
import WorkbenchHeader, { type CircuitView } from "./WorkbenchHeader";
import {
  activationAt, CONDITION_OUTCOMES, expandedAtFinalQuery, FAMILY_LABELS, graphAtFinalQuery,
  matchesExpandedDiscovery, MODEL_IDS, MODEL_NAMES,
  NEGATION_LABELS, readNegationAsset, validateNegationIndex,
  type EvalModel, type ExpandedNegationGroup, type ExpandedNegationIndex, type MatrixLocator,
  type NegationCondition, type NegationExample,
  type NegationGate, type NegationGroup, type NegationIndex, type Outcome,
} from "./negationWorkbenchData";

type Split = keyof NegationIndex["splits"];
const numeric = (value: number | null | undefined) => value == null ? "—" : value.toPrecision(4);
const FACTORIAL = ["affirmative", "negative", "counterfactual_affirmative", "counterfactual_negative"] as const;

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

export default function NegationGateExplorer({ onCircuitView, onOpenCandidate, Graph }: {
  onCircuitView: (view: CircuitView) => void;
  onOpenCandidate: (id: string) => void;
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
    <WorkbenchHeader title="Negation Gate Explorer" view="negation" onChangeView={onCircuitView}
      checkpoint={index?.checkpointSha256} onOpenCandidate={onOpenCandidate} />
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
  const [condition, setCondition] = useState<NegationCondition | null>(null);
  const [model, setModel] = useState<EvalModel>("gemma");
  const [outcome, setOutcome] = useState<Outcome | "">("");
  const [peer, setPeer] = useState("");
  const [comprehensive, setComprehensive] = useState(false);
  const [selectionFrame, setSelectionFrame] = useState(false);
  const [anchorId, setAnchorId] = useState("");
  const [anchorOnly, setAnchorOnly] = useState(false);
  const split = index.splits[splitName];
  const dataset = selectionFrame ? "selection-frame" : `negation-${splitName}`;
  const { index: tokenIndex, error: tokenIndexError } = useTokenIndex(index.checkpointSha256);
  const witnesses = useGraphWitnesses(tokenIndex, dataset);
  const expansionIndex = useAsset<ExpandedNegationIndex>("expanded/index.json")?.data;
  const expansions = !selectionFrame && splitName === "discovery" && expansionIndex && matchesExpandedDiscovery(expansionIndex, index)
    ? expansionIndex.groups : undefined;
  const expandedGroupIds = expansions ? Object.keys(expansions).filter((id) => id in split.groups) : [];
  const expandedGroupId = comprehensive && expandedGroupIds.length
    ? (expandedGroupIds.includes(groupId) ? groupId : expandedGroupIds[0]) : undefined;
  const expansion = expandedGroupId ? expansions?.[expandedGroupId] : undefined;
  const expansionAsset = useAsset<ExpandedNegationGroup>(expansion?.path ?? null, expansion?.sha256);
  const locator = useAsset<{ mechanism: string; matrices: MatrixLocator[] }>("locator.json", index.assetHashes["locator.json"]);
  const gates = useMemo(() => {
    const groupGates = expansion ? new Set(expansion.gates) : undefined;
    const rows = split.gates.filter((gate) => (site === "ALL" || gate.id.startsWith(site + ":"))
      && (groupGates ? groupGates.has(gate.id) : !graphOnly || gate.id in (witnesses?.gates ?? split.selection)));
    rows.sort((left, right) => {
      if (sort === "fires") return queryFires(right) - queryFires(left);
      if (sort === "beta") return (maxBeta(right) ?? -1) - (maxBeta(left) ?? -1);
      return Math.abs(right.contrast) - Math.abs(left.contrast);
    });
    return searchSharedGates(rows, query);
  }, [split, site, graphOnly, query, sort, expansion, witnesses]);
  const preferredGate = gates.find((item) => item.id === gateId) ?? gates[0];
  const defaultCondition = preferredGate && split.selection[preferredGate.id]?.kind === "affirmative" ? "affirmative" : "negative";
  const conditionChoice = expandedGroupId && condition && !FACTORIAL.some((key) => key === condition)
    ? defaultCondition : condition ?? defaultCondition;
  const contextGroups = preferredGate ? [...new Set([
    ...(split.selection[preferredGate.id]?.groups ?? []), ...(split.gateGroups[preferredGate.id] ?? []),
  ])] : [];
  const groupOptions = selectionFrame ? Object.keys(tokenIndex?.datasets[dataset] ?? {}) : expandedGroupId ? expandedGroupIds : outcome
    ? Object.keys(split.groups).filter((id) => split.groups[id].outcomes.negative?.[model] === outcome)
    : [...new Set([...contextGroups, ...Object.keys(split.groups)])];
  const defaultWitness = !groupId && !condition && !selectionFrame && preferredGate
    ? graphWitnessForGate(witnesses, preferredGate.id, model, undefined, defaultCondition) : undefined;
  const selectedGroup = expandedGroupId ?? selectGroup(groupOptions, groupId, [
    ...(defaultWitness ? [defaultWitness.group] : []), ...contextGroups,
  ]);
  const groupMeta = selectedGroup ? split.groups[selectedGroup] : undefined;
  const source = !selectionFrame && groupMeta ? `${splitName}/${groupMeta.path}` : null;
  const groupAsset = useAsset<NegationGroup>(source, source ? index.assetHashes[source] : undefined);
  const frame = useTokenGroup(selectionFrame ? tokenIndex : undefined, dataset, selectedGroup);
  const group = groupAsset?.data?.mechanism === "negation" && groupAsset.data.groupId === selectedGroup
    ? groupAsset.data : undefined;
  const nativeExamples = useMemo(() => group?.examples.filter((row) => !expandedGroupId || FACTORIAL.some((key) => key === row.condition))
    ?? [], [group, expandedGroupId]);
  const examples = useMemo<TokenExample[]>(() => selectionFrame ? frame.group?.examples ?? [] : nativeExamples.map(row => ({
    ...row, positions: row.positions.map(position => ({ ...position, pool_id: row.pool_id })),
  })), [selectionFrame, frame.group, nativeExamples]);
  const nativeById = new Map(nativeExamples.map(row => [row.example_id, row]));
  const initialWitness = selectedGroup && defaultWitness?.group === selectedGroup ? defaultWitness : undefined;
  const example = selectExample(examples, initialWitness?.condition ?? conditionChoice, selectionFrame ? "negative" : defaultCondition);
  const nativeExample = example && nativeById.get(example.example_id);
  const anchor = selectToken(example, anchorId || initialWitness?.anchor || "");
  const active = tokenGates(anchor);
  const focusToken = selectionFrame || anchorOnly;
  const tokenSelection = filterTokenGates(active, site, query, gateId);
  const gateKey = focusToken ? tokenSelection.selected : preferredGate?.id;
  const gate = gateKey && !selectionFrame ? split.gates.find(item => item.id === gateKey) : undefined;
  const noteReference = example && anchor
    ? `${dataset} · ${example.example_id} · anchor ${anchor.anchor_id}`
    : undefined;
  const note = useSelectedNote(`${noteReference}/${gateKey}/${model}`);
  const availableWitness = gateKey ? graphWitnessForGate(witnesses, gateKey, model, selectedGroup, example?.condition, anchor?.anchor_id) : undefined;
  let expandedQuery: ReturnType<typeof expandedAtFinalQuery> | undefined;
  let expansionError = expansionAsset?.error;
  if (expansionAsset?.data && nativeExample && gateKey && anchor?.final_query) {
    try {
      const data = expansionAsset.data;
      if (!matchesExpandedDiscovery(data, index) || data.groupId !== expandedGroupId) {
        throw new Error("Expanded artifact provenance differs from the selected Discovery group.");
      }
      expandedQuery = expandedAtFinalQuery(data, nativeExample, gateKey, model);
    } catch (failure) {
      expansionError = failure instanceof Error ? failure.message : String(failure);
    }
  }
  const graph = expandedGroupId ? expandedQuery?.graph
    : nativeExample && gateKey && anchor?.final_query ? graphAtFinalQuery(nativeExample, gateKey) : undefined;

  function inspectToken(row: TokenExample, id: string) {
    setGroupId(row.group_id);
    setCondition(row.condition as NegationCondition);
    setAnchorId(id);
    setAnchorOnly(true);
    setQuery("");
    setSite("ALL");
  }

  function selectGate(id: string) {
    setGateId(id);
    setPeer("");
    if (focusToken) return;
    const found = graphWitnessForGate(witnesses, id, model, selectedGroup, example?.condition, anchor?.anchor_id);
    if (found) openWitness(found);
    else if (!expandedGroupId) {
      setGroupId("");
      setCondition(outcome ? "negative" : null);
      setAnchorId("");
    }
  }

  function openWitness(found: GraphWitness) {
    if (!expandedGroupIds.includes(found.group)) setComprehensive(false);
    setGroupId(found.group);
    setCondition(found.condition as NegationCondition);
    setAnchorId(found.anchor);
    setOutcome("");
  }

  function exploreSameExample(id: string) {
    if (id === gateKey || !active.some(item => item.id === id)) return;
    setPeer(gateKey ?? "");
    setGateId(id);
    setGroupId(selectedGroup ?? "");
    setCondition((example?.condition as NegationCondition | undefined) ?? conditionChoice);
    setQuery("");
    setSite("ALL");
    setAnchorOnly(true);
  }

  const shown = focusToken ? tokenSelection.rows.map(row => ({ id: row.id, value: row.value, aggregate: undefined }))
    : gates.map(row => ({ id: row.id, value: undefined, aggregate: row }));
  const matrices = locator?.data?.mechanism === "negation" ? locator.data.matrices : undefined;
  const peerKey = /^(PRE|MID):[0-9]+$/.test(peer.trim().toUpperCase()) ? peer.trim().toUpperCase() : undefined;
  function changeCondition(value: NegationCondition) {
    setCondition(value);
    setAnchorId("");
  }

  return <>
    <section className="ig-toolbar ig-dataset-toolbar">
      <label>Dataset<select value={selectionFrame ? "selection-frame" : splitName} onChange={(event) => {
        const choice = event.target.value;
        const next = choice === "smoke" ? "smoke" : "discovery";
        setSelectionFrame(choice === "selection-frame");
        setSplitName(next);
        setComprehensive(false);
        setGraphOnly(next === "discovery");
        setOutcome("");
        setQuery("");
        setSite("ALL");
        setGroupId("");
        setGateId("");
        setCondition(null);
        setAnchorId("");
        setAnchorOnly(false);
        setPeer("");
      }}>
        <option value="smoke">Smoke · {Object.keys(index.splits.smoke.groups).length} groups</option>
        <option value="discovery">Discovery · {Object.keys(index.splits.discovery.groups).length} groups</option>
        {tokenIndex?.datasets["selection-frame"] && <option value="selection-frame">Selection Frame controls</option>}
      </select></label>
      {expandedGroupIds.length > 0 && <label>Examples<select value={expandedGroupId ? "expanded" : "all"}
        onChange={(event) => {
          setComprehensive(event.target.value === "expanded");
          setQuery("");
          setSite("ALL");
          setGroupId("");
          setOutcome("");
          setCondition(null);
          setAnchorId("");
          setAnchorOnly(false);
        }}>
        <option value="all">All Discovery groups</option>
        <option value="expanded">Groups with expanded final-query graphs · {expandedGroupIds.length}</option>
      </select></label>}
    </section>
    <main className="ig-layout">
      <aside className="ig-gate-list" aria-label="Shared gates">
        <header>
          <div><strong>{focusToken ? "Active gates at token" : "Shared gates"}</strong>
            <span>{Math.min(250, shown.length)} / {shown.length}</span></div>
          <SharedGateSearch value={query} onChange={setQuery} onSelectFirst={() => {
            if (shown[0]) selectGate(shown[0].id);
          }} />
          <div className="ig-gate-filters">
            <label>Site<select value={site} onChange={(event) => setSite(event.target.value)}>
              <option>ALL</option><option>PRE</option><option>MID</option>
            </select></label>
            {!focusToken && <label>Sort<select value={sort} onChange={(event) => setSort(event.target.value)}>
              <option value="contrast">|Negative − affirmative|</option>
              <option value="fires">Final-query fire count</option>
              <option value="beta">Maximum representative β</option>
            </select></label>}
          </div>
          {focusToken && !selectionFrame && <button type="button" onClick={() => {
            setAnchorOnly(false); setGraphOnly(false); setQuery("");
          }}>Browse all gates</button>}
          {!focusToken && splitName === "discovery" && !expandedGroupId && <label className="ig-graph-filter">
            <input type="checkbox" checked={graphOnly} onChange={(event) => setGraphOnly(event.target.checked)} />
            With measured graphs ({Object.keys(witnesses?.gates ?? split.selection).length})
          </label>}
        </header>
        {shown.slice(0, 250).map(({ id, value, aggregate }) => <button type="button" key={id}
          className={id === gateKey ? "selected" : ""} aria-pressed={id === gateKey} onClick={() => selectGate(id)}>
          <strong><ResearchGateLabel checkpoint={index.checkpointSha256} gateId={id} /></strong>
          <span>{value !== undefined ? `gˢ ${numeric(value)}` : `Δ ${numeric(aggregate?.contrast)}`}</span>
          {aggregate && <>
            <small>{queryFires(aggregate)} query fires · β {numeric(maxBeta(aggregate))}</small>
            {expandedGroupId ? <small>Active at a matched final query</small>
              : id in (witnesses?.gates ?? split.selection) && <small className="ig-graph-indicator">Measured graph available</small>}
          </>}
        </button>)}
        {!shown.length && <p className="uw-empty">{anchor && !active.length && focusToken ? "No gates fired at this token." : "No gates match these filters."}</p>}
      </aside>
      <section className="ig-detail">
        {gateKey && <header><div><span className="uw-eyebrow">SHARED GATE</span>
          <h2><ResearchGateLabel checkpoint={index.checkpointSha256} gateId={gateKey} /></h2></div>
          {gate && <span className="ig-badges"><span>Final-query contrast {numeric(gate.contrast)}</span></span>}
        </header>}
        <section className="ig-examples">
          <header><h3>Matched examples</h3><p>Click a token to inspect its active gates and dependencies.</p></header>
          <div className="ig-toolbar negation-example-toolbar">
            <label>Example group<select value={selectedGroup ?? ""} onChange={(event) => {
              setGroupId(event.target.value); setAnchorId("");
            }}>
              {!groupOptions.length && <option value="">No matching examples</option>}
              {groupOptions.map((id) => <option key={id} value={id}>{selectionFrame
                ? id.replace("selection-frame-discovery-", "").replaceAll("_", " ")
                : id.slice("negation-".length) + " · " + FAMILY_LABELS[split.groups[id].family]}</option>)}
            </select></label>
            <label>Condition<select value={example?.condition ?? conditionChoice}
              onChange={(event) => changeCondition(event.target.value as NegationCondition)}>
              {(examples.length ? examples.map(row => row.condition as NegationCondition) : groupMeta?.conditions ?? [])
                .map(key => <option key={key} value={key}>{NEGATION_LABELS[key]}</option>)}
            </select></label>
            <ModelSelect model={model} onChange={setModel} />
            {!selectionFrame && !expandedGroupId && <label>Negative outcome<select value={outcome} onChange={(event) => {
              setOutcome(event.target.value as Outcome | "");
              if (event.target.value) changeCondition("negative");
            }}>
              <option value="">All outcomes</option>
              {Object.entries(CONDITION_OUTCOMES).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
            </select></label>}
          </div>
          {example && anchor ? <>
            {gateKey && <label className="uw-gate-search negation-peer-search">
              Compare a second gate
              <input type="search" placeholder="e.g. PRE:4312" value={peer} onChange={(event) => setPeer(event.target.value)} />
            </label>}
            <MatchedContexts examples={examples} nativeById={nativeById} selected={example.example_id} anchor={anchor.anchor_id}
              model={model} onSelect={changeCondition} onToken={inspectToken} gate={gateKey} peer={peerKey} />
            <TokenAnchorBar anchor={anchor} model={model} activeCount={active.length} anchorOnly={focusToken}
              onToggle={!selectionFrame && !focusToken ? () => inspectToken(example, anchor.anchor_id) : undefined}
              onReset={() => setAnchorId(example.positions.find(position => position.final_query)?.anchor_id ?? "")} />
            <TokenDependencyInspector checkpoint={index.checkpointSha256} dataset={dataset} index={tokenIndex}
              example={example} anchor={anchor} gate={gateKey} model={model} matrices={matrices} Graph={Graph}
              examples={examples} onSelectExampleToken={inspectToken}
              onSelectToken={id => inspectToken(example, id)} onExploreGate={exploreSameExample} onAnnotate={note.annotate}
              onOpenMeasuredExample={availableWitness && availableWitness.anchor !== anchor.anchor_id
                ? () => { setGateId(gateKey ?? ""); openWitness(availableWitness); } : undefined}
              legacy={{ graph, targets: expandedQuery?.targets, error: expansionError,
                scope: graph ? { checkpoint: index.checkpointSha256, dataset, example: example.example_id,
                  anchor: graph.anchorId, pool: nativeExample?.pool_id ?? null,
                  artifact: (expansion?.sha256 ?? (source ? index.assetHashes[source] : undefined)) ?? split.manifestSha256 } : undefined }} />
          </> : <p className="uw-empty" role="status">{(selectionFrame ? frame.error ?? tokenIndexError : groupAsset?.error)
            ?? (groupOptions.length ? "Loading matched examples…" : "No activation contexts match this selection.")}</p>}
        </section>
        {gateKey && <>
          <SelectedNoteEditor checkpoint={index.checkpointSha256} gate={gateKey} evidenceReference={noteReference} note={note} />
          <ComponentBetaLocator key={gateKey} gate={gateKey} checkpoint={index.checkpointSha256}
            matrices={matrices} onAnnotate={note.annotate} error={locator?.error} />
        </>}
        {gate && <details className="ig-examples"><summary>Split-wide gate statistics</summary>
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
        </details>}
      </section>
    </main>
    <footer className="uw-footer"><span>Checkpoint 61,036 · {selectionFrame ? "Selection Frame controls" : `${split.poolCount} × 2,048 aligned anchors`}</span>
      {!selectionFrame && <span>{expansion ? <>Graph job {expansion.jobId} · </>
        : split.graphProvenance && <>Graph job {split.graphProvenance.runtime.slurm_job_id} · </>}{split.manifestSha256.slice(0, 12)}</span>}
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

function MatchedContexts({ examples, nativeById, selected, anchor, model, onSelect, onToken, gate, peer }: {
  examples: TokenExample[];
  nativeById: ReadonlyMap<string, NegationExample>;
  selected: string;
  anchor: string;
  model: EvalModel;
  onSelect: (value: NegationCondition) => void;
  onToken: (row: TokenExample, id: string) => void;
  gate?: string;
  peer?: string;
}) {
  const keys = [...new Set([gate, peer].filter((id): id is string => Boolean(id)))];
  const scale = Math.max(0, ...examples.flatMap((row) => row.positions.flatMap((position) =>
    keys.map((key) => activationAt(position, key)),
  )));
  return <div className="td-matched">
    {keys.length > 0 && <p className="ig-graph-status">Shared activation scale · gˢ 0–{numeric(scale)}</p>}
    <div className="negation-context-grid">{examples.map((row) => {
      const native = nativeById.get(row.example_id);
      const current = row.example_id === selected;
      const condition = row.condition as NegationCondition;
      return <article key={row.example_id} className={`ig-context ${current ? "negation-context-selected" : ""}`}>
      <header>
        <div><strong>{NEGATION_LABELS[condition]}</strong><span>{row.example_id}</span></div>
        {!current && <button type="button" onClick={() => onSelect(condition)}>Inspect this condition</button>}
      </header>
      {(keys.length ? keys : [undefined]).map((id) => <TokenSequence key={id ?? "prompt"} example={row} gate={id}
        scale={scale} anchor={current ? anchor : undefined} model={model} label={keys.length > 1 ? id : undefined}
        onSelect={id => onToken(row, id)} />)}
      {native && <>
      <p>Valid {native.valid_alternatives.join(", ")}{native.excluded_candidates.length > 0 && <> · excluded {native.excluded_candidates.join(", ")}</>}</p>
      <p>Fixed margin L = logit({native.candidate_orientation[0]}) − logit({native.candidate_orientation[1]})</p>
      <div className="ig-predictions">{MODEL_IDS.map((model) => {
        const prediction = native.predictions[model];
        return <span key={model}><b>{MODEL_NAMES[model]}</b> {JSON.stringify(prediction.predicted_token)}
          {" · L "}{numeric(prediction.fixed_margin)}{" · correctness margin "}{numeric(prediction.correctness_margin)}
          {prediction.negation_outcome ? <> · {CONDITION_OUTCOMES[prediction.negation_outcome]}</>
            : prediction.exact_correct ? " · Correct" : " · Incorrect"}
        </span>;
      })}</div>
      <details><summary>Candidate logits and probabilities</summary>
        {MODEL_IDS.map((model) => <div key={model}>
          <b>{MODEL_NAMES[model]}</b>
          <p>Valid mass {numeric(native.predictions[model].valid_mass)} · excluded {numeric(native.predictions[model].excluded_mass)}
            {" · unrelated "}{numeric(native.predictions[model].unrelated_mass)}</p>
          <ul>{Object.entries(native.predictions[model].candidates).map(([word, values]) => <li key={word}>
            {word} · logit {numeric(values.logit)} · P {numeric(values.probability)}
          </li>)}</ul>
        </div>)}
      </details>
      </>}
    </article>; })}</div>
  </div>;
}
