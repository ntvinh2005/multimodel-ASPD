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
  const examples = Array.isArray(detail.examples) ? detail.examples : [];
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
    <h3>Feature #{String(feature.feature_id)} · measured diagnostics</h3>
    <div className="classification-split">
      <div className="measured-panel">
        <span className="scientific-kind">Measured · decoder geometry</span>
        <strong>{String(feature.partition ?? "unknown")} partition</strong>
        <small>R_base {fmt(feature.relative_decoder_norm_base, 5)} · R_target {fmt(feature.relative_decoder_norm_target_derived, 5)}</small>
        <small>decoder cosine {fmt(feature.decoder_cosine, 5)}</small>
      </div>
      <div className="measured-panel">
        <span className="scientific-kind">Measured · validation support</span>
        <strong>{count(feature.validation_activation_count)} activations</strong>
        <small>base-tail rank {count(feature.rank_base_tail)} · target-tail rank {count(feature.rank_target_tail)}</small>
        <small>within-partition rank {count(feature.rank_within_partition)}</small>
      </div>
    </div>
    {examples.length ? <><h3>Saved activation contexts</h3><pre className="json-preview">{JSON.stringify(examples, null, 2)}</pre></> : <div className="empty-state">No generic activation contexts were extracted for this retrain artifact. Decoder and firing diagnostics above remain measured.</div>}
    {!Object.keys(v2).length ? <div className="warn-box">Fresh behavioral validation has not been run for this checkpoint. No behavioral label is inferred from decoder tails or firing counts.</div> : <>
      <h3>Historical confirmatory V2 evidence</h3>
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
  if (!overview.available) return <section className="empty-state"><h2>Delta-Crosscoder artifact unavailable</h2><p>Real Delta artifacts are required; no demo values are shown.</p>{overview.diagnostics.map((diagnostic) => <p key={diagnostic}>{diagnostic}</p>)}</section>;
  if (mode === "comparison") return <section className="panel"><h2>Delta ↔ ASPD comparison</h2><p>Not run for this controlled retrain. Causal Delta and same-pair ASPD require a later, separately scoped phase.</p></section>;
  const metadata = object(overview.metadata);
  const label = String(metadata.scientific_label ?? metadata.run_name ?? "Delta-Crosscoder run");
  const engineering = object(overview.engineering_audit);
  const comparison = object(engineering.comparison);
  const comparisonMetrics = object(comparison.metrics);
  const totalFveComparison = object(comparisonMetrics.fve_total);
  const deltaFveComparison = object(comparisonMetrics.delta_fve);
  const deadComparison = object(comparisonMetrics.dead_count_total);
  const resource = object(object(overview.metrics).resource_benchmark);
  const v1 = overview.behavioral_v1;
  const v2 = overview.behavioral_v2;
  const hasBehavioralEvidence = Boolean(v1?.available || v2?.available);
  const v2Metadata = object(v2?.metadata);
  const benchmark = object(v2Metadata.benchmark);
  const modelEffect = object(v2?.model_effect);
  const riskEffect = object(object(modelEffect.category_effects).risk);
  const pairedRiskEffect = object(riskEffect.paired_target_minus_base);
  const v1Candidate = object(v2?.v1_candidate);
  const v1Target = object(v1Candidate.target_generated);
  const v1Support = object(v1Target.support);

  return <section className="panel"><h2>{mode === "overview" ? "Delta-Crosscoder Overview" : "Delta Feature Explorer"}</h2>
    <p>{label} · {overview.n_features ?? "?"} components · partitions: {JSON.stringify(overview.partition_counts ?? {})}</p>
    {mode === "overview" && <>
      <div className="metric-grid"><div><span>FVE total</span><strong>{fmt(object(exact).fve_total, 5)}</strong></div><div><span>Base FVE</span><strong>{fmt(object(exact).fve_base, 5)}</strong></div><div><span>Target FVE</span><strong>{fmt(object(exact).fve_target, 5)}</strong></div><div><span>Delta FVE</span><strong>{fmt(object(exact).delta_fve, 5)}</strong></div><div><span>Exact L0 S / Δ / total</span><strong>{count(object(exact).l0_shared)} / {count(object(exact).l0_delta)} / {count(object(exact).l0_total)}</strong></div><div><span>Dead S / Δ / total</span><strong>{count(object(exact).dead_count_shared)} / {count(object(exact).dead_count_delta)} / {count(object(exact).dead_count_total)}</strong></div><div><span>Training throughput</span><strong>{fmt(resource.tokens_per_second, 0)} tok/s</strong></div><div><span>Peak allocated GPU</span><strong>{number(resource.peak_allocated_gpu_bytes) === null ? "—" : `${(number(resource.peak_allocated_gpu_bytes)! / 2 ** 30).toFixed(2)} GiB`}</strong></div></div>
      {engineering.decision && <div className="result-callout"><span>Engineering decision · not a behavioral result</span><strong>{String(engineering.decision)}</strong><small>Old → new: total FVE {fmt(totalFveComparison.baseline, 6)} → {fmt(totalFveComparison.current, 6)}; Delta FVE {fmt(deltaFveComparison.baseline, 6)} → {fmt(deltaFveComparison.current, 6)}; dead features {count(deadComparison.baseline)} → {count(deadComparison.current)}.</small><small>Next phase: {String(engineering.next_phase ?? "fresh held-out validation")}. Fresh behavioral claim: none.</small></div>}
      {hasBehavioralEvidence ? <div className="method-summary-grid">
        <div className="method-summary-card"><span className="scientific-kind">Behavioral V1 · exploratory</span><h2>{v1?.available ? v1.status : "Unavailable"}</h2><p>80 prompts; sparse zero-control denominators. Epsilon ratios are secondary diagnostics only.</p><strong>{v1?.features_measured ?? 0} features measured</strong></div>
        <div className="method-summary-card"><span className="scientific-kind">Behavioral V2 · confirmatory</span><h2>{v2?.available ? v2.status : "Not measured"}</h2><p>{v2?.available ? `${String(benchmark.total ?? 180)} prompts · ${String(benchmark.matched_pairs ?? 60)} matched pairs · exact frozen-token replay` : "The dashboard is ready; confirmatory artifacts have not been produced yet."}</p><strong>{v2?.features_measured ?? 0} frozen features measured</strong>{v2?.available && <small>actual pair: {JSON.stringify(v2Metadata.models ?? {})}</small>}</div>
      </div> : <div className="method-summary-grid"><div className="method-summary-card"><span className="scientific-kind">Engineering validation</span><h2>{String(engineering.decision ?? "Not recorded")}</h2><p>The checkpoint, reconstruction metrics, sparsity, dead features, throughput, and provenance are available for inspection.</p></div><div className="method-summary-card"><span className="scientific-kind">Behavioral validation</span><h2>Not run</h2><p>No V1/V2 result is transferred to this new checkpoint. A fresh frozen V3 is the next phase.</p><strong>No behavioral claim</strong></div></div>}
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
    {mode === "features" && <><p className="muted">Rows are measured decoder/firing diagnostics. Decoder-tail rank is not a behavioral label; behavioral fields appear only when an explicit evaluation artifact exists.</p><label>Partition <select value={partition} onChange={(event) => setPartition(event.target.value)}><option value="">all</option><option value="delta">delta</option><option value="shared">shared</option></select></label>{rows?.items.map((row) => { const behavioral = object(row.behavioral_v2); const target = object(behavioral.target_generated); const matched = object(target.matched); const support = object(target.support); return <button className="table-row" key={row.feature_id} onClick={() => setFeature(row.feature_id)}><strong>#{row.feature_id}</strong> {String(behavioral.candidate_group ?? row.partition)} · R_base {fmt(behavioral.r_base ?? row.relative_decoder_norm_base)} · fires {count(row.validation_activation_count ?? row.fire_count)} · cosine {fmt(row.decoder_cosine, 4)}{Object.keys(behavioral).length ? ` · prevalence ${fmt(object(target.risk).prevalence)}/${fmt(object(target.sports_control).prevalence)} · D ${fmt(matched.mean, 4)} · q ${fmt(behavioral.bh_q, 4)} · support ${count(support.active_risk_responses)}/60 · ${String(behavioral.diagnostic_status)}` : " · behavior not measured"}</button>; })}{detail && <FeatureDetailPanel detail={detail}/>}</>}
  </section>;
}
