import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { DeltaCandidatesPayload, DeltaFeatureDetail, DeltaOverviewPayload } from "../types";

type Json = Record<string, unknown>;

function object(value: unknown): Json {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Json : {};
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function fmt(value: unknown, digits = 3): string {
  const numeric = number(value);
  return numeric === null ? "—" : numeric.toFixed(digits);
}

function count(value: unknown): string {
  const numeric = number(value);
  return numeric === null ? "—" : numeric.toFixed(0);
}

function PairContext({ value }: { value: unknown }) {
  const row = object(value);
  const context = object(row.response_context);
  return <div className="example-card">
    <small>{String(row.prompt_id ?? "unknown")} · max z={fmt(row.response_max, 4)} · boundary={String(row.response_boundary_maximum ?? false)}</small>
    <p><strong>Prompt:</strong> {String(row.prompt ?? "")}</p>
    <p><strong>Response:</strong> {String(row.generated_answer ?? "")}</p>
    {context.text ? <pre className="json-preview">{String(context.text)}{"\n"}tokens={JSON.stringify(context.token_strings)}{"\n"}z={JSON.stringify(context.token_activations)}</pre> : <p className="muted">No positive response activation context.</p>}
  </div>;
}

function FeatureDetailPanel({ detail }: { detail: DeltaFeatureDetail }) {
  const feature = detail.feature as Json;
  const v2 = object(detail.behavioral_v2);
  const target = object(v2.target_generated);
  const matched = object(target.matched);
  const bootstrap = object(matched.bootstrap_95);
  const support = object(target.support);
  const promptEffect = object(v2.prompt_region_target_matched);
  const contexts = object(detail.behavioral_v2_examples);
  const pairs = Array.isArray(contexts.matched_pairs) ? contexts.matched_pairs as Json[] : [];
  const [pairId, setPairId] = useState("");
  useEffect(() => { setPairId(pairs.length ? String(pairs[0].pair_id) : ""); }, [detail.feature.feature_id]);
  const selected = pairs.find((pair) => String(pair.pair_id) === pairId) ?? pairs[0];
  const targetPair = object(selected?.target_generated);
  const basePair = object(selected?.base_generated);

  return <div className="context-panel">
    <h3>Feature #{String(feature.feature_id)} · confirmatory V2</h3>
    {!Object.keys(v2).length ? <div className="empty-state">V2 artifact is not available for this feature.</div> : <>
      <div className="classification-split">
        <div className="measured-panel">
          <span className="scientific-kind">Measured</span>
          <strong>R_base {fmt(v2.r_base)} · R_target {fmt(v2.r_target)}</strong>
          <small>response risk prevalence {fmt(object(target.risk).prevalence)} · matched-control {fmt(object(target.sports_control).prevalence)}</small>
          <small>active risk {count(support.active_risk_responses)}/60 · positive-effect domains {count(support.positive_effect_domains)}</small>
          <small>prompt D {fmt(promptEffect.mean, 4)} · response D {fmt(matched.mean, 4)}</small>
        </div>
        <div className="derived-panel">
          <span className="scientific-kind">Derived</span>
          <strong>mean D {fmt(matched.mean, 4)}</strong>
          <small>median {fmt(matched.median, 4)} · fraction D&gt;0 {fmt(matched.fraction_positive)}</small>
          <small>95% CI [{fmt(bootstrap.lower_95, 4)}, {fmt(bootstrap.upper_95, 4)}]</small>
          <small>raw p {fmt(matched.raw_p, 5)} · BH q {fmt(v2.bh_q, 5)}</small>
          <small>boundary/template fraction {fmt(v2.special_or_boundary_top_context_fraction)}</small>
        </div>
      </div>
      <div className="interpretation-panel compact">
        <div className="interpretation-header"><span className="scientific-kind">Interpretation</span><span className={v2.automatic_confirmatory_criteria_pass ? "badge badge-pass" : "badge badge-warn"}>{String(v2.diagnostic_status)}</span></div>
        <p>Automated criteria: {String(v2.automatic_confirmatory_criteria_pass)}. Manual semantic review and blinded model-behavior scoring remain separate requirements.</p>
      </div>
      {pairs.length > 0 && <>
        <h3>Exact matched context view</h3>
        <label>pair_id <select value={pairId} onChange={(event) => setPairId(event.target.value)}>{pairs.map((pair) => <option key={String(pair.pair_id)} value={String(pair.pair_id)}>{String(pair.pair_id)} · D={fmt(pair.target_generated_d, 4)}</option>)}</select></label>
        <h3>Target-generated</h3><div className="two-column"><PairContext value={targetPair.risk}/><PairContext value={targetPair.sports_control}/></div>
        <h3>Base-generated control</h3><div className="two-column"><PairContext value={basePair.risk}/><PairContext value={basePair.sports_control}/></div>
      </>}
    </>}
  </div>;
}

export default function DeltaResearch({ mode }: { mode: "overview" | "features" | "comparison" }) {
  const [overview, setOverview] = useState<DeltaOverviewPayload | null>(null);
  const [rows, setRows] = useState<DeltaCandidatesPayload | null>(null);
  const [partition, setPartition] = useState("");
  const [feature, setFeature] = useState<number | null>(null);
  const [detail, setDetail] = useState<DeltaFeatureDetail | null>(null);
  useEffect(() => { api.deltaOverview().then(setOverview).catch((error) => setOverview({ available: false, diagnostics: [String(error)] })); }, []);
  useEffect(() => { if (mode === "features") api.deltaCandidates(partition || undefined).then(setRows).catch((error) => setRows({ available: false, diagnostics: [String(error)], items: [] })); }, [mode, partition]);
  useEffect(() => { if (feature !== null) api.deltaFeature(feature).then(setDetail).catch(() => setDetail(null)); }, [feature]);
  const exact = useMemo(() => object(object(overview?.metrics).validation_modes).exact_batchtopk, [overview]);
  if (!overview) return <div className="loading">Loading Delta-Crosscoder artifacts…</div>;
  if (!overview.available) return <section className="empty-state"><h2>Delta positive control unavailable</h2><p>Real Delta artifacts are required; no demo values are shown.</p>{overview.diagnostics.map((diagnostic) => <p key={diagnostic}>{diagnostic}</p>)}</section>;
  if (mode === "comparison") return <section className="panel"><h2>Delta ↔ ASPD comparison</h2><p>Not run in this confirmatory phase. Causal Delta and same-pair ASPD require a later instruction.</p></section>;
  const v1 = overview.behavioral_v1;
  const v2 = overview.behavioral_v2;
  const v2Metadata = object(v2?.metadata);
  const benchmark = object(v2Metadata.benchmark);
  const modelEffect = object(v2?.model_effect);
  const riskEffect = object(object(modelEffect.category_effects).risk);
  const pairedRiskEffect = object(riskEffect.paired_target_minus_base);
  const v1Candidate = object(v2?.v1_candidate);
  const v1Target = object(v1Candidate.target_generated);
  const v1Support = object(v1Target.support);

  return <section className="panel"><h2>{mode === "overview" ? "Delta-Crosscoder Overview" : "Delta Feature Explorer"}</h2>
    <p>Reduced-scale positive control · {overview.n_features ?? "?"} components · partitions: {JSON.stringify(overview.partition_counts ?? {})}</p>
    {mode === "overview" && <>
      <div className="metric-grid"><div><span>FVE total</span><strong>{fmt(object(exact).fve_total, 5)}</strong></div><div><span>Delta FVE</span><strong>{fmt(object(exact).delta_fve, 5)}</strong></div><div><span>Exact L0</span><strong>{count(object(exact).l0_total)}</strong></div><div><span>Dead fraction</span><strong>{number(object(exact).dead_fraction_total) === null ? "—" : `${(number(object(exact).dead_fraction_total)! * 100).toFixed(2)}%`}</strong></div></div>
      <div className="method-summary-grid">
        <div className="method-summary-card"><span className="scientific-kind">Behavioral V1 · exploratory</span><h2>{v1?.available ? v1.status : "Unavailable"}</h2><p>80 prompts; sparse zero-control denominators. Epsilon ratios are secondary diagnostics only.</p><strong>{v1?.features_measured ?? 0} features measured</strong></div>
        <div className="method-summary-card"><span className="scientific-kind">Behavioral V2 · confirmatory</span><h2>{v2?.available ? v2.status : "Not measured"}</h2><p>{v2?.available ? `${String(benchmark.total ?? 180)} prompts · ${String(benchmark.matched_pairs ?? 60)} matched pairs · exact frozen-token replay` : "The dashboard is ready; confirmatory artifacts have not been produced yet."}</p><strong>{v2?.features_measured ?? 0} frozen features measured</strong>{v2?.available && <small>actual pair: {JSON.stringify(v2Metadata.models ?? {})}</small>}</div>
      </div>
      {v2?.available && <div className="classification-split">
        <div className="measured-panel">
          <span className="scientific-kind">Measured · blind judge</span>
          <strong>Risk scores: Base {fmt(riskEffect.base_mean)} · Target {fmt(riskEffect.target_mean)}</strong>
          <small>valid {count(riskEffect.valid)}/{count(riskEffect.total)} · condition 1 {String(modelEffect.condition_1_status ?? "unresolved")}</small>
        </div>
        <div className="derived-panel">
          <span className="scientific-kind">Derived · paired model effect</span>
          <strong>Target − Base {fmt(pairedRiskEffect.estimate)}</strong>
          <small>bootstrap 95% CI [{fmt(pairedRiskEffect.lower_95)}, {fmt(pairedRiskEffect.upper_95)}]</small>
        </div>
        <div className="interpretation-panel compact">
          <span className="scientific-kind">Interpretation · frozen feature test</span>
          <strong>Feature #5408: {v1Candidate.automatic_confirmatory_criteria_pass ? "survived" : "did not survive"}</strong>
          <small>active Target-risk {count(v1Support.active_risk_responses)}/60 · mean D {fmt(object(v1Target.matched).mean, 4)} · BH q {fmt(v1Candidate.bh_q, 4)}</small>
          <small>surviving target-tail candidates: {v2.automatic_target_tail_candidates ?? 0}</small>
        </div>
      </div>}
    </>}
    {mode === "features" && <><p className="muted">V2 rows sort by confirmatory status and BH q. V1 risk/control epsilon ratios are not used for ordering.</p><label>Partition <select value={partition} onChange={(event) => setPartition(event.target.value)}><option value="">all</option><option value="delta">delta</option><option value="shared">shared</option></select></label>{rows?.items.map((row) => { const behavioral = object(row.behavioral_v2); const target = object(behavioral.target_generated); const matched = object(target.matched); const support = object(target.support); return <button className="table-row" key={row.feature_id} onClick={() => setFeature(row.feature_id)}><strong>#{row.feature_id}</strong> {String(behavioral.candidate_group ?? row.partition)} · R_base {fmt(behavioral.r_base ?? row.relative_decoder_norm_base)} · prevalence {fmt(object(target.risk).prevalence)}/{fmt(object(target.sports_control).prevalence)} · D {fmt(matched.mean, 4)} · CI [{fmt(object(matched.bootstrap_95).lower_95, 3)}, {fmt(object(matched.bootstrap_95).upper_95, 3)}] · q {fmt(behavioral.bh_q, 4)} · support {count(support.active_risk_responses)}/60 · domains {count(support.positive_effect_domains)} · {String(behavioral.diagnostic_status ?? "V2 not measured")}</button>; })}{detail && <FeatureDetailPanel detail={detail}/>}</>}
  </section>;
}
